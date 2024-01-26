import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np

import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset, DataLoader
import common.vision.datasets as datasets
from common.vision.transforms import ResizeImage
import time
from fix_utils import ImageClassifier
from sourcefree import *
import losses

parser = argparse.ArgumentParser('argument for training')

parser.add_argument('--print_freq', type=int, default=500,
                    help='print frequency')
parser.add_argument('--save_freq', type=int, default=1000,
                    help='save frequency')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch_size')
parser.add_argument('--num_workers', type=int, default=8,
                    help='num of workers to use')
parser.add_argument('--epochs', type=int, default=180,
                    help='number of training epochs')

# optimization
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')

# model dataset
parser.add_argument('root', metavar='DIR',
                    help='root path of dataset')
parser.add_argument('-d', '--data', metavar='DATA', default='OfficeHome')
parser.add_argument('-s', '--source', default='Pr', help='source domain(s)')
parser.add_argument('-t', '--target', default='Ar', help='target domain(s)')

# other setting
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--beta', default=0.7, type=float, help='gjs weight')
parser.add_argument('--num', default=10, type=int, help='elr loss hyper parameter')
parser.add_argument('--thred', default=0.5, type=float, help='threshold')
parser.add_argument('--thred2', default=0.7, type=float, help='threshold')

args = parser.parse_args()
print(args)
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


class TwoCropsTransform:
    def __init__(self, transform, transform1):
        self.transform = transform
        self.transform1 = transform1
        # self.transform_s = transform_s

    def __call__(self, x):
        if self.transform is None:
            return x, x
        else:
            q = self.transform(x)
            k = self.transform1(x)
            # p = self.transform_s(x)
            return [q, k]


def set_loader(args):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #
    # train_transform = T.Compose([
    #     ResizeImage(256),
    #     T.RandomResizedCrop(224),
    #     T.RandomHorizontalFlip(),
    #     T.ToTensor(),
    #     normalize
    # ])

    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.2, 1.)),
        T.ColorJitter(brightness=0.4,
                      contrast=0.4,
                      saturation=0.4,
                      hue=0.2),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])

    val_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    dataset = datasets.__dict__[args.data]
    train_source_dataset = dataset(root=args.root, task=args.target, r=0, download=False, transform=TwoCropsTransform(train_transform, train_transform))

    num_classes = train_source_dataset.num_classes
    args.nb_classes = num_classes

    val_dataset = dataset(root=args.root, task=args.target, r=0, download=False, transform=val_transform)

    args.nb_samples = len(train_source_dataset)
    print("training samples size: ", args.nb_samples)


    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # train_target_dataset = dataset(root=args.root, task=args.target, download=True, transform=train_transform)
    # train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
    #                                  shuffle=True, num_workers=args.workers, drop_last=True)
    # val_dataset = dataset(root=args.root, task=args.target, download=True, transform=val_transform)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)



    return train_source_loader, val_loader


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_step(net1,net2, train_source_loader, optimizer, criterion, epoch, args):
    net1.train()
    net2.train()
    # adjust_learning_rate(optimizer_b, epoch, args)
    time1 = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    acc = AverageMeter()

    end = time.time()
    # target_train_iter = iter(train_target_loader)
    for idx, (inputs, targets, index) in enumerate(train_source_loader):

        lr_scheduler(optimizer, iter_num=idx+epoch*len(train_source_loader), max_iter=max_iter)

        inputs1, inputs2 = inputs[0], inputs[1]

        inputs1 = inputs1.cuda()
        inputs2 = inputs2.cuda()

        targets, index = targets.cuda(), index.cuda()
        # inputs_target = inputs_target.cuda()

        data_time.update(time.time() - end)

        bsz = targets.shape[0]

        # inputs = inputs.cuda()

        outputs = []
        outputs.append(net2(net1(inputs1)))
        outputs.append(net2(net1(inputs2)))
        loss = criterion(outputs, targets)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        losses.update(loss.mean().item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()
        acc.update(1.0, bsz)

        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss1 {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                epoch, idx + 1, len(train_source_loader), batch_time=batch_time, loss=losses,  acc=acc))
            sys.stdout.flush()
    time2 = time.time()
    print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))


def test_step(net1, net2, test_loader, lowconfreg, epoch, args):
    net1.eval()
    net2.eval()

    time1 = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc = AverageMeter()

    end = time.time()
    preds = torch.zeros(args.nb_samples, dtype=torch.bool)
    probs = torch.zeros(args.nb_samples, args.nb_classes)
    for idx, (inputs, targets, index) in enumerate(test_loader):
        data_time.update(time.time() - end)

        bsz = targets.shape[0]

        inputs = inputs.cuda()
        targets = targets.cuda()

        with torch.no_grad():
            output = net2(net1(inputs))
            pred = output.max(1)[1]
            preds[index] = pred.eq(targets).cpu()
            acc1 = accuracy(output, targets, topk=(1,))
            probs[index] = output.softmax(1).cpu()

        batch_time.update(time.time() - end)
        end = time.time()
        acc.update(acc1[0].item(), bsz)

        if (idx + 1) % args.print_freq == 0:
            print('Test: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                epoch, idx + 1, len(test_loader), batch_time=batch_time,
                acc=acc))
            sys.stdout.flush()
    time2 = time.time()
    acc2 = preds[lowconfreg].float().mean()
    print('Test time: total time {:.2f}, acc {:.2f}, lowconf {:.2f}'.format(time2 - time1, acc.avg, acc2))
    return acc.avg, acc2, probs


def pseudo_labels(net1, net2, loader, updated_loader, args):
    net1.eval()
    net2.eval()

    time1 = time.time()


    pseudos_values = torch.zeros(args.nb_samples)
    pseudos_index = torch.zeros(args.nb_samples).long()
    for idx, (inputs, targets, index) in enumerate(loader):

        inputs = inputs.cuda()
        with torch.no_grad():
            output = net2(net1(inputs)).softmax(1)

            max_values, max_index = output.cpu().max(1)

            pseudos_values[index] = max_values

            pseudos_index[index] = max_index

            # acc1 = accuracy(output, targets, topk=(1,))

    sum_list = torch.zeros(args.nb_classes)
    for i in range(args.nb_classes):
        # mask_i = pseudos_index == i
        sum_i = (pseudos_index == i).sum().item()
        sum_list[i] = sum_i
        # values_i = pseudos_values[mask_i]

    sum_list = sum_list/max(sum_list)
    pseudos = -torch.ones(args.nb_samples).long()
    m = 0
    c = 0
    selected_index = torch.zeros(args.nb_samples, dtype=torch.bool)
    lowconf_index = torch.zeros(args.nb_samples, dtype=torch.bool)
    for k, (i, j) in enumerate(zip(pseudos_values, pseudos_index)):
        if i > (args.thred * sum_list[j]):
            m = m + 1
            pseudos[k] = j
            # if pseudos[k] == updated_loader.dataset.targets[k]:
            #     c = c + 1
            #     selected_index[k] = True

        else:
            pseudos[k] = random.randint(0, args.nb_classes-1)
            # pseudos[k] = torch.ones((1, args.nb_classes))/args.nb_classes
            selected_index[k] = True
            lowconf_index[k] = True

    print("confident region: ", m)
    print("confident correct: ", c)
    print("sum of index: ", selected_index.sum())
    acc = 1
    # updated_loader.dataset.targets = pseudos[selected_index]
    updated_loader.dataset.targets = pseudos
    # samples = []
    # for i in range(len(updated_loader.dataset.samples)):
    #     if selected_index[i].item() is True:
    #         samples.append(updated_loader.dataset.samples[i])
    # updated_loader.dataset.samples = samples
    time2 = time.time()
    print('Pseudo Label time: total time {:.2f}, acc {:.2f}'.format(time2 - time1, acc))
    return updated_loader, selected_index, lowconf_index, pseudos


train_source_loader, test_loader = set_loader(args)
criterion_gjs = losses.get_criterion(args.nb_classes, args)

##############################################################################################################
# backbone = models.resnet50(pretrained=True)
# classifier = ImageClassifier(backbone, args.nb_classes, bottleneck_dim=256).cuda()
# optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr)
##############################################################################################################
netF = ResNet_FE().cuda()
netC = feat_classifier(class_num=args.nb_classes).cuda()
weights_dir = args.source.lower()[0] + '2' + args.target.lower()[0]
netF.load_state_dict(torch.load('/home/lyi7/da_weights/office-home/{}/source_F.pt'.format(weights_dir)))
netC.load_state_dict(torch.load('/home/lyi7/da_weights/office-home/{}/source_C.pt'.format(weights_dir)))
optimizer = optim.SGD(
    [
        {
            'params': netF.feature_layers.parameters(),
            'lr': args.lr * 0.1  # 1
        },
        {
            'params': netF.bottle.parameters(),
            'lr': args.lr * 1  # 10
        },
        {
            'params': netF.bn.parameters(),
            'lr': args.lr * 1  # 10
        },
        {
            'params': netC.parameters(),
            'lr': args.lr * 1  # 10
        }
    ],
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True)
##############################################################################################################

def refurbish(probs, lowconf, loader):
    max_val, max_ind = probs[lowconf].max(1)
    mask = max_val > args.thred2
    loader.dataset.targets[lowconf][mask] = max_ind[mask]
    print("refurbish size: ", max_ind[mask].size())
    return loader

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * decay
    return optimizer


best = 0
max_iter = args.epochs * len(train_source_loader)


#pseudo labels
train_source_loader, selected_index, lowconfreg, pseudos = pseudo_labels(netF, netC, test_loader, train_source_loader, args)
best_acc2 = 0
kkk=1
initial = args.num
initial0 = args.num + 1
for epoch in range(1, args.epochs + 1):
    train_step(netF, netC, train_source_loader, optimizer, criterion_gjs, epoch-kkk+1, args)
    acc, acc2, pprobs = test_step(netF, netC, test_loader, lowconfreg, epoch, args)
    best = max(best, acc)
    if epoch == initial:
        kkk = initial
        train_source_loader = refurbish(pprobs, lowconfreg, train_source_loader)
        netF = ResNet_FE().cuda()
        netC = feat_classifier(class_num=args.nb_classes).cuda()
        weights_dir = args.source.lower()[0] + '2' + args.target.lower()[0]
        netF.load_state_dict(torch.load('/home/lyi7/da_weights/office-home/{}/source_F.pt'.format(weights_dir)))
        netC.load_state_dict(torch.load('/home/lyi7/da_weights/office-home/{}/source_C.pt'.format(weights_dir)))
        torch.save(train_source_loader.dataset.targets, './saves/targets_{}.pt'.format(epoch))
        optimizer = optim.SGD(
            [
                {
                    'params': netF.feature_layers.parameters(),
                    'lr': args.lr * 0.1  # 1
                },
                {
                    'params': netF.bottle.parameters(),
                    'lr': args.lr * 1  # 10
                },
                {
                    'params': netF.bn.parameters(),
                    'lr': args.lr * 1  # 10
                },
                {
                    'params': netC.parameters(),
                    'lr': args.lr * 1  # 10
                }
            ],
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True)
        initial = initial + initial0
        initial0 = initial0+1

    best_acc2 = max(best_acc2, acc2.item())



print("best acc is", best)
# with open('./sfnet/r' + str(args.r) + 'seed' + str(args.seed) +'da.txt', 'a') as f:
with open('gjs_da_acc.txt', 'a') as f:
    f.write('{}\tlr{}\tthreshold1/2:{}{}\tbeta{}\tnum{}\tbsz{}\tsrc/tar:{}/{}\t{}\n'.format(args.data, args.lr, args.thred,args.thred2, args.beta, args.num, args.batch_size, args.source, args.target, best))

