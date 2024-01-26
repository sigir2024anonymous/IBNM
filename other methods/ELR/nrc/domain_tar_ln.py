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
from data_list_domain import ImageList, ImageList_idx
import random, pdb, math, copy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import time

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter)**(-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


class ELR_loss(nn.Module):
    def __init__(self, beta, lamb, gamma, num, cls):
        super(ELR_loss, self).__init__()
        self.ema = torch.zeros(num, cls).cuda()
        self.beta = beta
        self.lamb = lamb
        self.gamma = gamma

    def entropy(self, preds):
        msoftmax = preds.mean(dim=0)
        gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        return gentropy_loss

    def forward(self, index,  outputs):
        y_pred = torch.nn.functional.softmax(outputs, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()

        self.ema[index] = self.beta * self.ema[index] + (1 - self.beta) * ((y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        elr_reg = ((1 - (self.ema[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = self.lamb * elr_reg + self.entropy(outputs.softmax(1))
        return final_loss



class GCE_loss(nn.Module):
    def __init__(self, q=0.2):
        super(GCE_loss, self).__init__()
        self.q = q

    def forward(self, ind, outputs, targets):
        targets = torch.zeros(targets.size(0), args.class_num).cuda().scatter_(1, targets.view(-1, 1), 1)
        pred = torch.nn.functional.softmax(outputs, dim=1)
        pred_y = torch.sum(targets * pred, dim=1)
        final_loss = torch.mean((1.0 - pred_y ** self.q) / self.q, dim=0)
        return final_loss

class CE_loss(nn.Module):
    def __init__(self):
        super(CE_loss, self).__init__()

    def forward(self, ind, outputs, targets):
        return torch.nn.functional.cross_entropy(outputs, targets)

class SL_loss(nn.Module):
    def __init__(self, q=1.0, b=1.0):
        super(SL_loss, self).__init__()
        self.q = q
        self.b = b

    def forward(self, ind, outputs, targets):
        targets_ = torch.zeros(targets.size(0), args.class_num).cuda().scatter_(1, targets.view(-1, 1), 1)
        pred = torch.nn.functional.softmax(outputs, dim=1)
        targets_ = torch.clamp(targets_, 1e-4, 1.0)
        final_loss = -torch.mean(torch.sum(torch.log(targets_)*pred, dim=1))*self.q
        return final_loss


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(), normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size

    txt_test = open(args.test_dset_path).readlines()


    txt_tar = open(args.t_dset_path).readlines()

    dsets["target"] = ImageList_idx(txt_test, transform=image_train(), tmp=args.root)
    dset_loaders["target"] = DataLoader(dsets["target"],
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=False)

    dsets["test"] = ImageList_idx(txt_tar, transform=image_test(), tmp=args.root)
    dset_loaders["test"] = DataLoader(dsets["test"],
                                      batch_size=train_bs * 3,
                                      shuffle=False,
                                      num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])
    mean_ent = torch.mean(Entropy(
        nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent



def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier,
                                   feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()

    netF.load_state_dict(torch.load('/home/lyi7/da_weights/domainnet/{}/source_F.pt'.format(args.source)))
    netB.load_state_dict(torch.load('/home/lyi7/da_weights/domainnet/{}/source_B.pt'.format(args.source)))
    netC.load_state_dict(torch.load('/home/lyi7/da_weights/domainnet/{}/source_C.pt'.format(args.source)))


    param_group = []
    param_group_c=[]
    for k, v in netF.named_parameters():
        if True:
            param_group += [{'params': v, 'lr': args.lr * 0.1}]

    for k, v in netB.named_parameters():
        if True:
            param_group += [{'params': v, 'lr': args.lr * 1}]
    for k, v in netC.named_parameters():
        param_group_c += [{'params': v, 'lr': args.lr * 1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    #building feature bank and score bank
    loader = dset_loaders["target"]
    num_sample=len(loader.dataset)
    fea_bank=torch.randn(num_sample,256)
    score_bank = torch.randn(num_sample, 40).cuda()

    if args.type == 0:
        criterion = ELR_loss(args.beta, args.lamb, args.gamma, args.nb_samples, args.nb_classes)
    elif args.type == 1:
        criterion = GCE_loss(args.beta)
    elif args.type == 2:
        criterion = SL_loss(args.beta, args.lamb)
    elif args.type == 3:
        criterion = CE_loss()
    elif args.type == 4:
        criterion = CE_loss()

    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            indx=data[-1]
            #labels = data[1]
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            output_norm=F.normalize(output)
            outputs = netC(output)
            outputs=nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    netF.train()
    netB.train()
    netC.train()
    acc_log=0
    best = 0
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        #output_re = softmax_out.unsqueeze(1)
        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()

            distance = output_f_@fea_bank.T
            _, idx_near = torch.topk(distance,
                                    dim=-1,
                                    largest=True,
                                    k=args.K+1)
            idx_near = idx_near[:, 1:]  #batch x K
            score_near = score_bank[idx_near]    #batch x K x C

            fea_near = fea_bank[idx_near]  #batch x K x num_dim
            fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0],-1,-1) # batch x n x dim
            distance_ = torch.bmm(fea_near, fea_bank_re.permute(0,2,1))  # batch x K x n
            _,idx_near_near=torch.topk(distance_,dim=-1,largest=True,k=args.KK+1)  # M near neighbors for each of above K ones
            idx_near_near = idx_near_near[:,:,1:] # batch x K x M
            tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
            match = (
                idx_near_near == tar_idx_).sum(-1).float()  # batch x K
            weight = torch.where(
                match > 0., match,
                torch.ones_like(match).fill_(0.1))  # batch x K

            weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                    args.KK)  # batch x K x M
            weight_kk = weight_kk.fill_(0.1)

            # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
            #weight_kk[idx_near_near == tar_idx_]=0

            score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
            #print(weight_kk.shape)
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                    -1)  # batch x KM

            score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                               40)  # batch x KM x C

            score_self = score_bank[tar_idx]

        # nn of nn
        output_re = softmax_out.unsqueeze(1).expand(-1, args.K * args.KK,
                                                    -1)  # batch x C x 1
        const = torch.mean(
            (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
             weight_kk.cuda()).sum(1)) # kl_div here equals to dot product since we do not use log for score_near_kk
        loss = torch.mean(const)


        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.K,
                                                         -1)  # batch x K x C

        loss += torch.mean((
            F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) *
            weight.cuda()).sum(1))

        # self, if not explicitly removing the self feature in expanded neighbor then no need for this
        #loss += -torch.mean((softmax_out * score_self).sum(-1))

        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(msoftmax *
                                    torch.log(msoftmax + args.epsilon))
        loss += gentropy_loss

        with torch.no_grad():
            targets = softmax_out.max(1)[1]

        loss += criterion(tar_idx, outputs_test, targets) * args.lamb

        optimizer.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()

            acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            # if args.dset == 'visda-2017':
            #     acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB,
            #                                  netC, flag= True)
            #     log_str = 'Task: {}, Iter:{}/{}; Accuracy on target = {:.2f}%'.format(
            #         args.name, iter_num, max_iter, acc_s_te
            #     ) + '\n' + 'T: ' + acc_list

            best = max(best, acc_s_te)
            print(log_str + '\n')
            netF.train()
            netB.train()
            netC.train()



    return netF, netB, netC, best


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neighbors')
    parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='9',
                        help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch',
                        type=int,
                        default=40,
                        help="max iterations")
    parser.add_argument('--interval', type=int, default=40)
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="batch_size")
    parser.add_argument('--worker',
                        type=int,
                        default=8,
                        help="number of workers")

    parser.add_argument('--beta', default=0.7, type=float, help='ema for t')
    parser.add_argument('--lamb', default=7.0, type=float, help='elr loss hyper parameter')
    parser.add_argument('--gamma', default=1.0, type=float, help='ema for t')

    # model dataset
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('--all', type=bool, default=False)


    parser.add_argument(
        '--dset',
        type=str,
        default='domainnet')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net',
                        type=str,
                        default='resnet50')
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--type', type=int, default=0,
                        help='0:elr; 1:gce; 2:sl')
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--KK', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer',
                        type=str,
                        default="wn",
                        choices=["linear", "wn"])
    parser.add_argument('--classifier',
                        type=str,
                        default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='weight/target/')
    parser.add_argument('--output_src', type=str, default='weight/source/')
    parser.add_argument('--tag', type=str, default='selfplus')
    parser.add_argument('--da',
                        type=str,
                        default='uda')
    parser.add_argument('--issave', type=bool, default=True)
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
    torch.backends.cudnn.deterministic = True

    image_list = {
        "c": "clipart",
        "i": "infograph",
        "p": "painting",
        "q": "quickdraw",
        "r": "real",
        "s": "sketch",
    }

    if args.all:
        task = ['c', 'p', 'r', 's']
        acc_acg = []
        for my_source in task:
            task1 = {'c', 'p', 'r', 's'}
            task1_rm = {my_source}
            task2 = list(task1.difference(task1_rm))
            for my_target in task2:
                args.source = my_source
                args.target = my_target

                args.t_dset_path = '/home/lyi7/data/domainnet/{}_train.txt'.format(image_list[args.target])
                args.test_dset_path = '/home/lyi7/data/domainnet/{}_test.txt'.format(image_list[args.target])

                args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
                args.name = args.source + str(2) + args.target
                print(args)
                _, _, _ , accs = train_target(args)
                acc_acg.append(accs)

        meanacc = sum(acc_acg) / len(acc_acg)
        # with open('./domain_accs/domain_elr.txt', 'a') as f:
        #     f.write('lr{}\tbeta{}\tlamb{}\t{}\n'.format(args.lr, args.beta, args.lamb,  meanacc))
        loss_type = {
            0: 'elr',
            1: 'gce',
            2: 'sl',
            3: 'ce',
            4: 'plc'
        }
        with open('./threeln/domain_{}.txt'.format(loss_type[args.type]), 'a') as f:
            f.write('beta{}\tlamb{}\t{}\n'.format(args.beta, args.lamb, meanacc))

    else:
        args.source = names[args.s]
        args.target = names[args.t]

        args.t_dset_path = '/home/lyi7/data/domainnet/{}_train.txt'.format(image_list[args.target])
        args.test_dset_path = '/home/lyi7/data/domainnet/{}_test.txt'.format(image_list[args.target])
        args.output_dir_src = osp.join(args.output_src, args.da, args.dset,
                                       names[args.s][0].upper())
        args.name = args.source + str(2) + args.target
        train_target(args)

        _, _, _, accs = train_target(args)
