import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
import random
from loss import CrossEntropyLabelSmooth
from sklearn.metrics import confusion_matrix
import common.vision.datasets as datasets

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return  transforms.Compose([
          transforms.Resize((resize_size, resize_size)),
          transforms.RandomCrop(crop_size),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize
      ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return  transforms.Compose([
          transforms.Resize((resize_size, resize_size)),
          transforms.CenterCrop(crop_size),
          transforms.ToTensor(),
          normalize
      ])

def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size

    image_list = {
        "c": "clipart",
        "i": "infograph",
        "p": "painting",
        "q": "quickdraw",
        "r": "real",
        "s": "sketch",
    }

    dataset = datasets.__dict__['DomainNet']
    train_source_dataset = dataset(root=args.root, task=args.source, r=0, split='train', download=False,
                                   transform=image_train())

    num_classes = train_source_dataset.num_classes
    args.nb_classes = num_classes

    val_dataset = dataset(root=args.root, task=args.source, r=0, split='test', download=False, transform=image_test())

    args.nb_samples = len(train_source_dataset)
    print("training samples size: ", args.nb_samples)

    test_dataset = dataset(root=args.root, task=args.target, r=0, split='test', download=False, transform=image_test())

    train_source_loader = DataLoader(train_source_dataset, batch_size=train_bs,
                                     shuffle=True, num_workers=args.worker, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=train_bs, shuffle=False, num_workers=args.worker)

    test_loader = DataLoader(test_dataset, batch_size=train_bs * 2, shuffle=False, num_workers=args.worker)

    dset_loaders["source_tr"] = train_source_loader
    dset_loaders["source_te"] = val_loader
    dset_loaders["test"] = test_loader

    return dset_loaders

def cal_acc(loader, netF, netB, netC,t=0, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs),t=t)[0])
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent


def train_source(args):
    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck_sdaE(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    smax=100
    #while iter_num < max_iter:
    for epoch in range(args.max_epoch):
        iter_source = iter(dset_loaders["source_tr"])
        for batch_idx, (inputs_source,
                        labels_source,_) in enumerate(iter_source):

            if inputs_source.size(0) == 1:
                continue

            iter_num += 1
            progress_ratio = batch_idx / (len(dset_loaders) - 1)
            s = (smax - 1 / smax) * progress_ratio + 1 / smax
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            feature_src,masks = netB(netF(inputs_source), t=0,s=100,all_mask=True)

            # sparsity regularization for domain attention
            reg = 0
            count = 0
            for m in masks[0]:
                reg += m.sum()  # numerator
                count += np.prod(m.size()).item()  # denominator
            for m in masks[1]:
                reg += m.sum()  # numerator
                count += np.prod(m.size()).item()  # denominator
            reg /= count

            outputs_source1 = netC(feature_src[0])
            outputs_source2 = netC(feature_src[1])
            classifier_loss = CrossEntropyLabelSmooth(
                num_classes=args.class_num, epsilon=args.smooth)(
                    outputs_source1, labels_source)+CrossEntropyLabelSmooth(
                num_classes=args.class_num, epsilon=args.smooth)(
                    outputs_source2, labels_source) + 0.75 * reg

            optimizer.zero_grad()
            classifier_loss.backward()

            # gradient compensation for embedding layer
            for n, p in netB.em.named_parameters():
                num = torch.cosh(
                    torch.clamp(s * p.data, -10, 10)) + 1
                den = torch.cosh(p.data) + 1
                p.grad.data *= smax / s * num / den
            torch.nn.utils.clip_grad_norm_(netF.parameters(), 10000)

            optimizer.step()

        #if iter_num % interval_iter == 0 or iter_num == max_iter:
        netF.eval()
        netB.eval()
        netC.eval()
        acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC,t=0, flag=False)
        log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)+ '\n'

        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str+'\n')

        if acc_s_te >= acc_init:
            acc_init = acc_s_te
            best_netF = netF.state_dict()
            best_netB = netB.state_dict()
            best_netC = netC.state_dict()

        netF.train()
        netB.train()
        netC.train()

    netF.eval()
    netB.eval()
    netC.eval()
    acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC,t=0,flag= False)

    log_str = 'Task: {}; Accuracy on target = {:.2f}%'.format(args.name_src, acc_s_te)+ '\n'
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return netF, netB, netC

def test_target(args):
    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck_sdaE(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    acc, _ = cal_acc(dset_loaders['test'],
                                netF,
                                netB,
                                netC,
                                t=0,
                                flag=False)
    log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc) + '\n'
      
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ours')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='9', help="device id to run")
    parser.add_argument('-s', '--source', default='c', help='source domain(s)')
    parser.add_argument('-t', '--target', default='i', help='target domain(s)')
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=8, help="number of workers")
    parser.add_argument(
        '--dset',
        type=str,
        default='domainnet')
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')

    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='./domainnet/')
    parser.add_argument('--da', type=str, default='uda')
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'visda-2017':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'domainnet':
        names = ['c', 'p', 'r', 's']
        args.class_num = 40


    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    args.output_dir_src = osp.join(args.output, args.dset, args.source)
    args.name_src = args.source
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    train_source(args)

    args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
    for i in names:
        if i == args.source:
            continue
        args.target = i
        args.name = args.source + '2' + args.target
        print(args)
        test_target(args)