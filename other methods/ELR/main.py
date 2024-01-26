#!/usr/bin/env python
import argparse
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import faiss

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.nn import functional as F

import loader
import builder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

import datetime

from models import build_model
from configs.config import get_config
import timm

now = datetime.datetime.now()
time_str = now.strftime("%Y-%m-%d %H:%M:%S")
print("Time: ", time_str)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--cfg', type=str,
                    default="configs/vit_base.yaml",
                    metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--head_lr_ratio', type=float, default=3, help='hyper-parameters head lr')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--alpha', type=float, default=0.8, help='hyper-parameters alpha')
parser.add_argument('--beta', type=float, default=3, help='hyper-parameters beta')


parser.add_argument('--dataset', type=str, default='office_home',
                        choices=['office31', 'office_home', 'VisDA', 'domainnet','pacs'], help='dataset used')
parser.add_argument('--data-A', metavar='DIR Domain A', help='path to domain A dataset')
parser.add_argument('--data-B', metavar='DIR Domain B', help='path to domain B dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names + ['vit'],
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 2x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--clean-model', default='', type=str, metavar='PATH',
                    help='path to clean model (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--low-dim', default=512, type=int,
                    help='feature dimension (default: 512)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--temperature', default=0.2, type=float,
                    help='softmax temperature')
parser.add_argument('--warmup-epoch', default=20, type=int,
                    help='number of warm-up epochs to only train with InfoNCE loss')
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco-v2/SimCLR data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--exp-dir', default='experiment_pcl', type=str,
                    help='the directory of the experiment')
parser.add_argument('--ckpt-save', default=20, type=int,
                    help='the frequency of saving ckpt')
parser.add_argument('--num_cluster', default='250,500,1000', type=str,
                    help='number of clusters for self entropy loss')

parser.add_argument('--instcon-weight', default=1.0, type=float,
                    help='the weight for instance contrastive loss after warm up')
parser.add_argument('--cwcon-weightstart', default=0.0, type=float,
                    help='the starting weight for cluster-wise contrastive loss')
parser.add_argument('--cwcon-weightsature', default=1.0, type=float,
                    help='the satuate weight for cluster-wise contrastive loss')
parser.add_argument('--cwcon-startepoch', default=20, type=int,
                    help='the start epoch for scluster-wise contrastive loss')
parser.add_argument('--aug-startepoch', default=20, type=int,
                    help='the start epoch for scluster-wise contrastive loss')
parser.add_argument('--cwcon-satureepoch', default=100, type=int,
                    help='the saturated epoch for cluster-wise contrastive loss')
parser.add_argument('--cwcon_filterthresh', default=0.2, type=float,
                    help='the threshold of filter for cluster-wise contrastive loss')
parser.add_argument('--selfentro-temp', default=0.2, type=float,
                    help='the temperature for self-entropy loss')
parser.add_argument('--selfentro-startepoch', default=20, type=int,
                    help='the start epoch for self entropy loss')
parser.add_argument('--selfentro-weight', default=1.0, type=float,
                    help='the start weight for self entropy loss')
parser.add_argument('--distofdist-startepoch', default=200, type=int,
                    help='the start epoch for dist of dist loss')
parser.add_argument('--distofdist-weight', default=1.0, type=float,
                    help='the start weight for dist of dist loss')
parser.add_argument('--divide-num', default=0.8, type=float,
                    help='the start weight for dist of dist loss')
parser.add_argument('--prec_nums', default='1,5,15', type=str,
                    help='the evaluation metric')

parser.add_argument('--mixup', action='store_false',help="whether to use mixup in-domain")

args, unparsed = parser.parse_known_args()
config = get_config(args)

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.num_cluster = args.num_cluster.split(',')
    args.domainA = os.path.split(args.data_A)[-1]
    args.domainB = os.path.split(args.data_B)[-1]
    args.exp_dir = os.path.join(args.exp_dir,args.domainA+'_'+args.domainB)
    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("=> creating model '{}'".format(args.arch))

    cudnn.benchmark = False

    """ traindirA = os.path.join(args.data_A, 'train')
    traindirB = os.path.join(args.data_B, 'train') """
    
    traindirA = os.path.join(args.data_A)
    traindirB = os.path.join(args.data_B)

    train_dataset = loader.TrainDataset(traindirA, traindirB, args.aug_plus)
    eval_dataset = loader.EvalDataset(traindirA, traindirB)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None)

    if args.arch == 'resnet50':
        model = builder.UCDIR(
        models.__dict__[args.arch],
        dim=args.low_dim, K_A=eval_dataset.domainA_size, K_B=eval_dataset.domainB_size,
        m=args.moco_m, T=args.temperature, mlp=args.mlp, selfentro_temp=args.selfentro_temp,
        num_cluster=args.num_cluster,  cwcon_filterthresh=args.cwcon_filterthresh,num_workers=args.workers,gpu=args.gpu)
    
    if args.arch == 'vit':
        model = build_model(config)
        model = builder.UCDIR(
            model.backbone,
            dim=model.backbone.embed_dim, K_A=eval_dataset.domainA_size, K_B=eval_dataset.domainB_size,
            m=args.moco_m, T=args.temperature, mlp=args.mlp, selfentro_temp=args.selfentro_temp,
            num_cluster=args.num_cluster,  cwcon_filterthresh=args.cwcon_filterthresh,num_workers=args.workers,gpu=args.gpu)
        args.low_dim = model.base_encoder.embed_dim
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # resnet weights
    if args.clean_model and args.arch == 'resnet50':
        if os.path.isfile(args.clean_model):
            print("=> loading pretrained clean model '{}'".format(args.clean_model))

            loc = 'cuda:{}'.format(args.gpu)
            clean_checkpoint = torch.load(args.clean_model, map_location=loc)

            current_state = model.state_dict()
            used_pretrained_state = {}

            for k in current_state:
                if 'encoder' in k and 'fc' not in k:
                    k_parts = '.'.join(k.split('.')[1:])
                    used_pretrained_state[k] = clean_checkpoint['state_dict']['module.encoder_q.'+k_parts]
            current_state.update(used_pretrained_state)
            model.load_state_dict(current_state)
            
        else:
            print("=> no clean model found at '{}'".format(args.clean_model))

    if args.clean_model and args.arch == 'vit':
        if os.path.isfile(args.clean_model):
            print("=> loading pretrained clean model '{}'".format(args.clean_model))

            loc = 'cuda:{}'.format(args.gpu)
            clean_checkpoint = torch.load(args.clean_model, map_location=loc)
            
            current_state = model.state_dict()
            used_pretrained_state = {}

            for k in current_state:
                if 'base_encoder' in k and 'head' not in k:
                    k_parts = '.'.join(k.split('.')[1:])
                    used_pretrained_state[k] = clean_checkpoint[k_parts]
                    used_pretrained_state[k].requires_grad_(True)
            current_state.update(used_pretrained_state)
            model.load_state_dict(current_state)
        else:
            print("=> no clean model found at '{}'".format(args.clean_model))
        config.defrost()    
        config.MODEL.NUM_FEATURES = config.MODEL.VIT.EMBED_DIM    
        config.freeze()
        mem_fea = torch.rand(len(train_loader.dataset), config.MODEL.NUM_FEATURES).cuda()
        mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
        mem_cls = torch.ones(len(train_loader.dataset), config.MODEL.NUM_CLASSES).cuda() / config.MODEL.NUM_CLASSES
    for param in model.parameters():
        param.requires_grad_(True)
        
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    if args.arch == 'vit':
   
        parameters = set_weight_decay(model, lr_mult=1, cfg=config)
        optimizer = torch.optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                             lr=args.lr, weight_decay=config.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, 
                                weight_decay=args.weight_decay)
    

    info_save = open(os.path.join(args.exp_dir, time_str+' info.txt'), 'w')

    best_res_A = [0., 0., 0.]
    best_res_B = [0., 0., 0.]

    info_save.write(time_str+"\n")
    info_save.write(print_args(args))
    for epoch in range(args.epochs):
        
        features_A, features_B, _, _ = compute_features(eval_loader, model, args)

        features_A = features_A.numpy()
        features_B = features_B.numpy()
        
        if epoch == 0:
            model.queue_A.data = torch.tensor(features_A).T.cuda(args.gpu)
            model.queue_B.data = torch.tensor(features_B).T.cuda(args.gpu)

        cluster_result = None
        probs = None
        if epoch >= args.warmup_epoch:
            cluster_result = run_kmeans(features_A, features_B, args)
            features_A_m = torch.from_numpy(features_A)
            features_B_m = torch.from_numpy(features_B)
            probs = Divide_by_CE_loss(features_A_m,features_B_m,cluster_result,args)
            indices = {}
            mask_A = probs['domain_A'] >= args.divide_num
            mask_B = probs['domain_B'] >= args.divide_num
            indices['A'] = mask_A.nonzero()[0]
            indices['B'] = mask_B.nonzero()[0]
            strrA = "In domain, select {}/{} instances from A\n".format(len(indices['A']),len(probs['domain_A']))
            strrB = "In domain, Select {}/{} instances from B\n".format(len(indices['B']),len(probs['domain_B']))
            print(strrA,strrB)
            info_save.write(strrA)
            info_save.write(strrB)

        if args.arch == 'resnet50':
            adjust_learning_rate(optimizer, epoch, args)
            train(train_loader, model, criterion, optimizer, epoch, args, info_save, cluster_result, probs)
        else:
            train(train_loader, model, criterion, optimizer, epoch, args, info_save, cluster_result, probs, mem_fea, mem_cls)
            pass
            
        if epoch >= args.aug_startepoch:
            cluster_result = run_kmeans(features_A, features_B, args,cat=True)
            features_A = torch.from_numpy(features_A)
            features_B = torch.from_numpy(features_B)
            probs,pseudo = Divide_by_CE_loss(features_A,features_B,cluster_result,args,divide=True)
            prob_A,prob_B,psl_A,psl_B = probs[:features_A.shape[0]],probs[features_A.shape[0]:],pseudo[:features_A.shape[0]],pseudo[features_A.shape[0]:]
            mask_A = prob_A >= args.divide_num
            mask_B = prob_B >= args.divide_num
            indices = {}
            far_indices = {}
            psls ={}
            indices['A'] = mask_A.nonzero()[0]
            indices['B'] = mask_B.nonzero()[0]
            far_indices['A'] = np.where(mask_A == 0)[0]
            far_indices['B'] = np.where(mask_B == 0)[0]
            psls['A'] = psl_A[mask_A]
            psls['B'] = psl_B[mask_B]
            strrA = "Select {}/{} instances from A\n".format(len(indices['A']),len(prob_A))
            strrB = "Select {}/{} instances from B\n".format(len(indices['B']),len(prob_B))
            if args.arch == 'vit':
                k=1
                close_A = features_A[mask_A]
                close_B = features_B[mask_B]
                mask_A_f = ~mask_A
                mask_B_f = ~mask_B
                far_A = features_A[mask_A_f]
                far_B = features_B[mask_B_f]
                knn = KNeighborsClassifier(n_neighbors=k,metric='cosine')
                fea_close = torch.cat((close_A,close_B)).cpu()
                psl_close = torch.cat((psls['A'],psls['B'])).cpu()
                knn.fit(fea_close,psl_close)
                far_psl_A = knn.predict(far_A)
                far_psl_B = knn.predict(far_B)
                psls['A'] = torch.cat((psls['A'].cpu(),torch.from_numpy(far_psl_A)))
                psls['B'] = torch.cat((psls['B'].cpu(),torch.from_numpy(far_psl_B)))
                indices['A'] = np.concatenate((indices['A'],far_indices['A']))
                indices['B'] = np.concatenate((indices['B'],far_indices['B']))
                print("KNN K={} \n".format(k)) 
                info_save.write("KNN K={} \n".format(k))
            
            print(strrA,strrB)
            info_save.write(strrA)
            info_save.write(strrB)
            divide_dataset = loader.TrainDataset(traindirA, traindirB, args.aug_plus,indices,psls,return_label =True)
            divide_loader = torch.utils.data.DataLoader(
                divide_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)
            if args.arch == 'vit':
                divide_train(divide_loader, model, criterion, optimizer, epoch, args, info_save, mem_fea, mem_cls)
                pass
            else:
                divide_train(divide_loader, model, criterion, optimizer, epoch, args, info_save)
                pass
            
        features_A, features_B, targets_A, targets_B = compute_features(eval_loader, model, args)
        features_A = features_A.numpy()
        targets_A = targets_A.numpy()

        features_B = features_B.numpy()
        targets_B = targets_B.numpy()

        prec_nums = args.prec_nums.split(',')
        res_A, res_B = retrieval_precision_cal(features_A, targets_A, features_B, targets_B,
                                               preck=(int(prec_nums[0]), int(prec_nums[1]), int(prec_nums[2])))
        
        if best_res_A[0]< res_A[0]:
            best_res_A = res_A
        if best_res_B[0]< res_B[0]:
            best_res_B = res_B


        Printstr1 = "Domain "+args.domainA+"->"+args.domainB+": P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), best_res_A[0],
                                                                                int(prec_nums[1]), best_res_A[1],
                                                                                int(prec_nums[2]), best_res_A[2])
        Printstr2 = "Domain "+args.domainB+"->"+args.domainA+": P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), best_res_B[0],
                                                                        int(prec_nums[1]), best_res_B[1],
                                                                        int(prec_nums[2]), best_res_B[2])
        
        Printstr3 = "Precision now: "+"Domain "+args.domainA+"->"+args.domainB+": P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), res_A[0],
                                                                                int(prec_nums[1]), res_A[1],
                                                                                int(prec_nums[2]), res_A[2])
        Printstr4 = "Precision now: "+"Domain "+args.domainB+"->"+args.domainA+": P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), res_B[0],
                                                                        int(prec_nums[1]), res_B[1],
                                                                        int(prec_nums[2]), res_B[2])
        print(Printstr1,Printstr2,Printstr3,Printstr4)
        info_save.write(Printstr1)
        info_save.write(Printstr2)
        info_save.write(Printstr3)
        info_save.write(Printstr4)


    info_save.write("Domain "+args.domainA+"->"+args.domainB+": P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), best_res_A[0],
                                                                          int(prec_nums[1]), best_res_A[1],
                                                                          int(prec_nums[2]), best_res_A[2]))
    info_save.write("Domain "+args.domainB+"->"+args.domainA+": P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), best_res_B[0],
                                                                  int(prec_nums[1]), best_res_B[1],
                                                                  int(prec_nums[2]), best_res_B[2]))

def train(train_loader, model, criterion, optimizer, epoch, args, info_save, cluster_result, probs, mem_fea=None, mem_cls=None):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    losses = {'Inst_A': AverageMeter('Inst_Loss_A', ':.4e'),
              'Inst_B': AverageMeter('Inst_Loss_B', ':.4e'),
              'Cwcon_A': AverageMeter('Cwcon_Loss_A', ':.4e'),
              'Cwcon_B': AverageMeter('Cwcon_Loss_B', ':.4e'),
              'SelfEntropy': AverageMeter('Loss_SelfEntropy', ':.4e'),
              'DistLogits': AverageMeter('Loss_DistLogits', ':.4e'),
              'Mixup_A': AverageMeter('Loss_Mixup_A', ':.4e'),
              'Mixup_B': AverageMeter('Loss_Mixup_B', ':.4e'),
              'Total_loss': AverageMeter('Loss_Total', ':.4e')}

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time,
         losses['SelfEntropy'],
         losses['DistLogits'],
         losses['Total_loss'],
         losses['Inst_A'], losses['Inst_B'],
         losses['Cwcon_A'], losses['Cwcon_B'],
         losses['Mixup_A'], losses['Mixup_B']],
        prefix="Epoch: [{}]".format(epoch))

    class_weight_src = torch.ones(config.MODEL.NUM_CLASSES, ).cuda()

    model.train()
    
    if epoch >= args.warmup_epoch:
        indices = {}
        mask_A = probs['domain_A'] >= args.divide_num
        mask_B = probs['domain_B'] >= args.divide_num
        indices['A'] = mask_A.nonzero()[0]
        indices['B'] = mask_B.nonzero()[0]
        threshold_A = len(probs['domain_A'])
        threshold_B = len(probs['domain_B'])
        indices['A'] = [x for x in indices['A'] if x < threshold_A]
        indices['B'] = [x for x in indices['B'] if x < threshold_B]
    end = time.time()
    with torch.enable_grad():
        for i, (images_A, image_ids_A, images_B, image_ids_B, cates_A, cates_B) in enumerate(train_loader):

            data_time.update(time.time() - end)
            intersection_A = intersection_B = None
            psl_A = psl_B = None
            if epoch >= args.warmup_epoch:
                intersection_A = set(indices['A']) & set(image_ids_A.numpy())
                intersection_B = set(indices['B']) & set(image_ids_B.numpy())
                if len(intersection_A) >= 2:
                    psl_A = [cluster_result['im2cluster_A'][0][i] for i in list(intersection_A)]
                else:
                    psl_A = None
                if len(intersection_B) >= 2:
                    psl_B = [cluster_result['im2cluster_B'][0][i] for i in list(intersection_B)]    
                else:
                    psl_B = None

            if args.arch == 'vit':
                idx_step = epoch * len(train_loader) + i
                optimizer = inv_lr_scheduler(optimizer, idx_step, lr=args.lr)

            if args.gpu is not None:
                images_A[0] = images_A[0].cuda(args.gpu, non_blocking=True)
                images_A[1] = images_A[1].cuda(args.gpu, non_blocking=True)
                image_ids_A = image_ids_A.cuda(args.gpu, non_blocking=True)

                images_B[0] = images_B[0].cuda(args.gpu, non_blocking=True)
                images_B[1] = images_B[1].cuda(args.gpu, non_blocking=True)
                image_ids_B = image_ids_B.cuda(args.gpu, non_blocking=True)

            losses_instcon, \
            q_A,  q_B, \
            losses_selfentro, \
            losses_distlogits, \
            losses_cwcon, \
            losses_mix, \
            losses_patchmix   = model(im_q_A=images_A[0], im_k_A=images_A[1],
                                 im_id_A=image_ids_A, im_q_B=images_B[0],
                                 im_k_B=images_B[1], im_id_B=image_ids_B,
                                 criterion=criterion,cluster_result = cluster_result,
                                 mem_fea=mem_fea, mem_cls=mem_cls, 
                                 class_weight_src=class_weight_src,
                                 psl_A=psl_A, psl_B=psl_B,
                                 mix_indice_A=intersection_A, mix_indice_B=intersection_B,)
            inst_loss_A = losses_instcon['domain_A']
            inst_loss_B = losses_instcon['domain_B']

            losses['Inst_A'].update(inst_loss_A.sum().item(), images_A[0].size(0))
            losses['Inst_B'].update(inst_loss_B.sum().item(), images_B[0].size(0))

            loss_A = inst_loss_A * args.instcon_weight
            loss_B = inst_loss_B * args.instcon_weight

            if epoch >= args.warmup_epoch:

                cwcon_loss_A = losses_cwcon['domain_A']
                cwcon_loss_B = losses_cwcon['domain_B']

                losses['Cwcon_A'].update(cwcon_loss_A.sum().item(), images_A[0].size(0))
                losses['Cwcon_B'].update(cwcon_loss_B.sum().item(), images_B[0].size(0))

                if epoch <= args.cwcon_startepoch:
                    cur_cwcon_weight = args.cwcon_weightstart
                elif epoch < args.cwcon_satureepoch:
                    cur_cwcon_weight = args.cwcon_weightstart + (args.cwcon_weightsature - args.cwcon_weightstart) * \
                                       ((epoch - args.cwcon_startepoch) / (args.cwcon_satureepoch - args.cwcon_startepoch))
                else:
                    cur_cwcon_weight = args.cwcon_weightsature

                loss_A += cwcon_loss_A * cur_cwcon_weight
                loss_B += cwcon_loss_B * cur_cwcon_weight
                
                loss_mix_A = losses_mix['domain_A']
                loss_mix_B = losses_mix['domain_B']
                
                if args.mixup:
                    if loss_mix_A is not None:
                        losses['Mixup_A'].update(loss_mix_A.sum().item(), images_A[0].size(0))
                        loss_A += loss_mix_A
                    if loss_mix_B is not None:
                        losses['Mixup_B'].update(loss_mix_B.sum().item(), images_B[0].size(0))
                        loss_B += loss_mix_B   

            all_loss = (loss_A + loss_B) / 2

            if epoch >= args.selfentro_startepoch:

                losses_selfentro_list = []
                if losses_selfentro is not None:
                    for key in losses_selfentro.keys():
                        losses_selfentro_list.extend(losses_selfentro[key])

                    losses_selfentro_mean = torch.mean(torch.stack(losses_selfentro_list))
                    losses['SelfEntropy'].update(losses_selfentro_mean.item(), images_A[0].size(0))

                    all_loss += losses_selfentro_mean * args.selfentro_weight

            if epoch >= args.distofdist_startepoch:

                losses_distlogits_list = []
                if losses_distlogits is not None:
                    for key in losses_distlogits.keys():
                        losses_distlogits_list.extend(losses_distlogits[key])
    
                    losses_distlogits_mean = torch.mean(torch.stack(losses_distlogits_list))
                    losses['DistLogits'].update(losses_distlogits_mean.item(), images_A[0].size(0))
    
                    all_loss += losses_distlogits_mean * args.distofdist_weight

            if (args.arch == 'vit') & (losses_patchmix is not None):
                all_loss += losses_patchmix.squeeze_()

            all_loss = all_loss.sum()
            all_loss.requires_grad_(True)
            losses['Total_loss'].update(all_loss.sum().item(), images_A[0].size(0))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                info = progress.display(i)
                info_save.write(info + '\n')
            
def divide_train(divide_loader, model, criterion, optimizer, epoch, args, info_save, mem_fea=None, mem_cls=None):
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    losses = {'divide_loss_A': AverageMeter('divide_loss_A', ':.4e'),
              'divide_loss_B': AverageMeter('divide_loss_B', ':.4e'),
              'mix_loss': AverageMeter('mix_loss', ':.4e'),}

    progress = ProgressMeter(
        len(divide_loader),
        [batch_time, data_time,
         losses['divide_loss_A'],
         losses['divide_loss_B'],
         losses['mix_loss']],
        prefix="Epoch: [{}]".format(epoch))
    
    print_str  = "start divide train at epoch {}".format(epoch)
    print(print_str)
    info_save.write(print_str+"\n")
    model.train()

    with torch.enable_grad():
        for i, (images_A, psl_A, index_A, images_B, psl_B, index_B) in enumerate(divide_loader):

            if args.arch == 'vit':
                idx_step = epoch * len(divide_loader) + i
                optimizer = inv_lr_scheduler(optimizer, idx_step, lr=args.lr)

            if args.gpu is not None:
                images_A[0] = images_A[0].cuda(args.gpu, non_blocking=True)
                images_A[1] = images_A[1].cuda(args.gpu, non_blocking=True)
                psl_A = psl_A.cuda(args.gpu, non_blocking=True)

                images_B[0] = images_B[0].cuda(args.gpu, non_blocking=True)
                images_B[1] = images_B[1].cuda(args.gpu, non_blocking=True)
                psl_B = psl_B.cuda(args.gpu, non_blocking=True)
            loss_A,loss_B,mix_loss = model(im_q_A=images_A[0],im_q_B=images_B[0],im_id_A=index_A,im_id_B=index_B,
                                           criterion=criterion, divide=True,psl_A=psl_A,psl_B=psl_B,mem_fea=mem_fea, mem_cls=mem_cls)
            losses['divide_loss_A'].update(loss_A.sum().item(), images_A[0].size(0))
            losses['divide_loss_B'].update(loss_B.sum().item(), images_B[0].size(0))
            losses['mix_loss'].update(mix_loss.sum().item(), images_A[0].size(0))
            all_loss = (loss_A + loss_B) / 2 
            all_loss = all_loss + mix_loss
            all_loss.requires_grad_(True)

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()            

            if i % args.print_freq == 0:
                    info = progress.display(i)
                    info_save.write(info + '\n')
            
def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()

    features_A = torch.zeros(eval_loader.dataset.domainA_size, args.low_dim).cuda(args.gpu)
    features_B = torch.zeros(eval_loader.dataset.domainB_size, args.low_dim).cuda(args.gpu)

    indice = torch.zeros(eval_loader.dataset.domainA_size,dtype=torch.int64).cuda(args.gpu)
    targets_all_A = torch.zeros(eval_loader.dataset.domainA_size, dtype=torch.int64).cuda(args.gpu)
    targets_all_B = torch.zeros(eval_loader.dataset.domainB_size, dtype=torch.int64).cuda(args.gpu)

    for i, (images_A, indices_A, targets_A, images_B, indices_B, targets_B) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images_A = images_A.cuda(args.gpu,non_blocking=True)
            images_B = images_B.cuda(args.gpu,non_blocking=True)

            targets_A = targets_A.cuda(args.gpu,non_blocking=True)
            targets_B = targets_B.cuda(args.gpu,non_blocking=True)

            feats_A, feats_B = model(im_q_A=images_A, im_q_B=images_B, is_eval=True)
            
            indices_A = indices_A.cuda(args.gpu)
            indices_B = indices_B.cuda(args.gpu)
            features_A[indices_A] = feats_A
            features_B[indices_B] = feats_B

            targets_all_A[indices_A] = targets_A
            targets_all_B[indices_B] = targets_B

    return features_A.cpu(), features_B.cpu(), targets_all_A.cpu(), targets_all_B.cpu()

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def run_kmeans(x_A, x_B, args,cat=False):
    print('performing kmeans clustering')
    if cat:
        results = {'im2cluster_A': [], 'centroids': [], 'im2cluster_B': []}
        x = np.concatenate([x_A, x_B], axis=0)
        split_num = x_A.shape[0]
        for seed, num_cluster in enumerate(args.num_cluster):
            d = x.shape[1]
            k = int(num_cluster)
            clus = faiss.Clustering(d, k)
            clus.verbose = True
            clus.niter = 20
            clus.nredo = 5
            clus.seed = seed
            clus.max_points_per_centroid = 2000
            clus.min_points_per_centroid = 2
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = args.gpu
            index = faiss.IndexFlatL2(d)
            clus.train(x, index)
            D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
            im2cluster = [int(n[0]) for n in I]

            centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

            centroids = torch.Tensor(centroids).cuda(args.gpu)
            centroids_normed = nn.functional.normalize(centroids, p=2, dim=1)
            im2cluster = torch.LongTensor(im2cluster).cuda(args.gpu)
            a=(torch.split(im2cluster,split_num))
            split_list = [split_num,len(im2cluster)-split_num]
            im2cluster_A , im2cluster_B = torch.split(im2cluster,split_list)
            results['centroids']=centroids
            results['im2cluster_A']=im2cluster_A
            results['im2cluster_B']=im2cluster_B
            results['im2cluster']=im2cluster
    else: 
        results = {'im2cluster_A': [], 'centroids_A': [],
                'im2cluster_B': [], 'centroids_B': []}
        for domain_id in ['A', 'B']:
            if domain_id == 'A':
                x = x_A
            elif domain_id == 'B':
                x = x_B
            else:
                x = np.concatenate([x_A, x_B], axis=0)

            for seed, num_cluster in enumerate(args.num_cluster):
                d = x.shape[1]
                k = int(num_cluster)
                clus = faiss.Clustering(d, k)
                clus.verbose = True
                clus.niter = 20
                clus.nredo = 5
                clus.seed = seed
                clus.max_points_per_centroid = 2000
                clus.min_points_per_centroid = 2
                cfg = faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = False
                cfg.device = args.gpu
                index = faiss.IndexFlatL2(d)
                clus.train(x, index)
                D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
                im2cluster = [int(n[0]) for n in I]

                centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

                centroids = torch.Tensor(centroids).cuda(args.gpu)
                centroids_normed = nn.functional.normalize(centroids, p=2, dim=1)
                im2cluster = torch.LongTensor(im2cluster).cuda(args.gpu)
                results['centroids_'+domain_id].append(centroids_normed)
                results['im2cluster_'+domain_id].append(im2cluster)

    return results


def retrieval_precision_cal(features_A, targets_A, features_B, targets_B, preck=(1, 5, 15)):

    dists = cosine_similarity(features_A, features_B)

    res_A = []
    res_B = []
    for domain_id in ['A', 'B']:
        if domain_id == 'A':
            query_targets = targets_A
            gallery_targets = targets_B

            all_dists = dists

            res = res_A
        else:
            query_targets = targets_B
            gallery_targets = targets_A

            all_dists = dists.transpose()
            res = res_B

        sorted_indices = np.argsort(-all_dists, axis=1)
        sorted_cates = gallery_targets[sorted_indices.flatten()].reshape(sorted_indices.shape)
        correct = (sorted_cates == np.tile(query_targets[:, np.newaxis], sorted_cates.shape[1]))

        for k in preck:
            total_num = 0
            positive_num = 0
            for index in range(all_dists.shape[0]):

                temp_total = min(k, (gallery_targets == query_targets[index]).sum())
                pred = correct[index, :temp_total]

                total_num += temp_total
                positive_num += pred.sum()
            res.append(positive_num / total_num * 100.0)

    return res_A, res_B


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        return ' '.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def divide_by_cw_loss(x_A,x_B,cluster_result,args):
    all_losses = {'domain_A': [], 'domain_B': []}
    all_probs = {'domain_A': [], 'domain_B': []}
    all_ids = {'domain_A': [], 'domain_B': []}
    
    for domain_id in ['A','B']:
        if domain_id == 'A':
            feat_all = x_A.cuda(args.gpu)
        else:
            feat_all = x_B.cuda(args.gpu) 
        
        mask = 1.0
        for n, (im2cluster, prototypes) in enumerate(zip(cluster_result['im2cluster_' + domain_id],
                                                         cluster_result['centroids_' + domain_id])):
            
            mask *= torch.eq(im2cluster.contiguous().view(-1,1),
                             im2cluster.contiguous().view(1,-1)).float().cuda(args.gpu)    # num_feature * num_feature
            
            all_score = torch.div(torch.matmul(feat_all,feat_all.T),args.temperature).cuda(args.gpu)
            
            exp_all_score = torch.exp(all_score).cuda(args.gpu)
            
            log_prob = all_score - torch.log(exp_all_score.sum(1,keepdim=True)).cuda(args.gpu)
            
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
            
            cor_proto = prototypes[im2cluster]
            
            inst_pos_value = torch.exp(
                torch.div(torch.einsum('nc,nc->n',[feat_all,cor_proto]),args.temperature)   # N
            )
            
            inst_all_value = torch.exp(
                torch.div(torch.einsum('nc,ck->nk',[feat_all,prototypes.T]),args.temperature)# N * num_cluster
            )
            
            filters = ((inst_pos_value / torch.sum(inst_all_value,dim=1)) > args.cwcon_filterthresh).float()
             
            im_id_now = 0
            loss = - mean_log_prob_pos
            loss_id = []
            losses = []
            for i in filters:
                if i > 0:
                    loss_id.append(im_id_now)
                    a = loss[im_id_now].unsqueeze_(0)
                    losses.append(a)
                im_id_now += 1
            
            if losses != []:    
                all_ids['domain_' + domain_id].append(loss_id)
                losses = torch.cat(losses)
                if (losses.max() - losses.min()) > 1e-3:
                    losses = (losses - losses.min()) / (losses.max() - losses.min())
                else:
                    losses = (losses - losses.min()) / 0.2
                losses = losses.cpu()
                all_losses['domain_' + domain_id].append(losses)

                n = 0
                for i in losses:
                    if torch.isnan(i):
                        losses[n] = torch.tensor(1000.,dtype = torch.float32)
                        n += 1
                
                input_loss = losses.reshape(-1,1)
                if input_loss.shape[0] > 1:
                    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2,reg_covar=5e-4)
                    gmm.fit(input_loss)
                    prob = gmm.predict_proba(input_loss)

                    prob = prob[:,gmm.means_.argmin()] 
                    all_probs['domain_' + domain_id].append(prob)
            
            
    return all_probs, all_ids, all_losses
        
        

def Divide_by_CE_loss(x_A,x_B,cluster_result,args,divide=False):       
    divide_results={}
    
    if divide == True: 
        feat = torch.cat((x_A,x_B),dim=0)

        proto = cluster_result['centroids']

        feat = feat.cuda(args.gpu)
        proto = proto.cuda(args.gpu)
        feat_expaned = feat.unsqueeze(1)
        proto_expaned = proto.unsqueeze(0)
        if args.dataset == 'domainnet':
            logits = torch.div(torch.nn.CosineSimilarity(dim=2)(feat_expaned, proto_expaned) , args.selfentro_temp/5)
        else:
            logits = torch.div(torch.nn.CosineSimilarity(dim=2)(feat_expaned, proto_expaned) , args.selfentro_temp)

        all_losses = self_entropy = -torch.sum(F.log_softmax(logits, dim=1) * F.softmax(logits, dim=1), dim=1).detach().cpu()
        self_entropy = self_entropy.reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2,reg_covar=5e-4)
        gmm.fit(self_entropy)
        prob = gmm.predict_proba(self_entropy)

        prob = prob[:,gmm.means_.argmin()] 

        divide_results["losses"]= all_losses
        divide_results["probs"]= prob
        divide_results["pseudo"]= cluster_result['im2cluster']
        divide_results["split_num"] = x_A.shape[0]
        return prob,cluster_result['im2cluster']
    else:
        probs = {'domain_A': [], 'domain_B': []}
        for domain_id in ['A','B']:
            if domain_id == 'A':
                feat = x_A.cuda(args.gpu)
                proto = cluster_result['centroids_A'][0].cuda(args.gpu)
            else:
                feat = x_B.cuda(args.gpu)
                proto = cluster_result['centroids_B'][0].cuda(args.gpu)

            feat_expaned = feat.unsqueeze(1)
            proto_expaned = proto.unsqueeze(0)
            if args.dataset == 'domainnet':
                logits = torch.div(torch.nn.CosineSimilarity(dim=2)(feat_expaned, proto_expaned) , args.selfentro_temp/5)
            else:
                logits = torch.div(torch.nn.CosineSimilarity(dim=2)(feat_expaned, proto_expaned) , args.selfentro_temp)
            all_losses = self_entropy = -torch.sum(F.log_softmax(logits, dim=1) * F.softmax(logits, dim=1), dim=1).detach().cpu()
            self_entropy = self_entropy.reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2,reg_covar=5e-4)
            gmm.fit(self_entropy)
            prob = gmm.predict_proba(self_entropy)
            prob = prob[:,gmm.means_.argmin()]
            probs['domain_' + domain_id] = prob
                        
        return probs


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.5 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def inv_lr_scheduler(optimizer, iter_num, power=0.75, gamma=0.001, lr=0.001):
    lr = lr * (1 + gamma * iter_num) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
    return optimizer


def set_weight_decay(model, cfg, lr_mult=1):
    features_has_decay = []
    features_no_decay = []
    classifier_has_decay = []
    classifier_no_decay = []
    params_has_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if name.startswith("my_fc"):
            #print(name)
            if len(param.shape) == 1 or name.endswith(".bias"):
                classifier_no_decay.append(param)
            else:
                classifier_has_decay.append(param)
        elif name.startswith("s_dist") or name.endswith('_ratio'):
            params_has_decay.append(param)
        else:
            if len(param.shape) == 1 or name.endswith(".bias"):
                features_no_decay.append(param)
                # print(f"{name} has no weight decay")
            else:
                features_has_decay.append(param)
    if len(classifier_has_decay) > 0:
        res = [{'params': classifier_has_decay, 'lr_mult': cfg.head_lr_ratio * lr_mult},
               {'params': params_has_decay, 'lr_mult': cfg.head_lr_ratio * lr_mult},
               {'params': classifier_no_decay, 'lr_mult': cfg.head_lr_ratio * lr_mult, 'weight_decay': 0.},
               {'params': features_has_decay, 'lr_mult': lr_mult},
               {'params': features_no_decay, 'lr_mult': lr_mult, 'weight_decay': 0.}]
    else:
        res = [{'params': features_has_decay, 'lr_mult': lr_mult},
               {'params': features_no_decay, 'lr_mult': lr_mult, 'weight_decay': 0.}]
    return res
    
if __name__ == '__main__':
    main()
