import argparse
import os, sys

import numpy as np

sys.path.append('./')

import os.path as osp
import torch

import random
from collate import SimCLRCollateFunction
from Trainer import trainer_coda
from data_clus import office_load_idx, domainnet_load_idx
import clustering
from network import Model
from utils.utils import *

from sklearn.metrics.pairwise import cosine_similarity



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


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser(
        description='Domain Adaptation on office-home dataset')
    parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='5',
                        help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch',
                        type=int,
                        default=32,
                        help="maximum epoch")
    parser.add_argument("--lr_update",
                        default=20,
                        type=int,
                        help="Number of epochs to update the learning rate.")
    parser.add_argument("--pretrain_epoch",
                        default=20,#20,
                        type=int,
                        help="warm up epochs")
    parser.add_argument("--warmup_epoch2",
                        default=10,#5,
                        type=int,
                        help="warm up epochs")
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help="batch_size")
    parser.add_argument('--worker',
                        type=int,
                        default=4,
                        help="number of workers")
    parser.add_argument('--dset', type=str, default='r2p')
    parser.add_argument('--interval_epoch', type=int, default=1)
    parser.add_argument('--lr',
                        type=float,
                        default=3e-3,
                        help="learning rate")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--class_num', type=int, default=65)
    # parser.add_argument('--cluster_num_list', type=list, default=[65, 130, 195, 260])
    parser.add_argument('--backbone_output', type=int, default=2048)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--pretretrained', type=str2bool,
                        default=True,
                        help='use pretrained model')
    parser.add_argument('--layer',
                        type=str,
                        default="wn",
                        choices=["linear", "wn"])
    parser.add_argument('--arch',
                        type=str,
                        default="resnet50",
                        choices=["resnet50", "resnet18"])
    parser.add_argument('--classifier',
                        type=str,
                        default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='Office-Home')
    parser.add_argument('--file', type=str, default='target')
    parser.add_argument('--office31', action='store_true')
    parser.add_argument('--dataset', default='office_home', choices=['office_home', 'office31',
                                                                     'image_CLEF', 'DomainNet',
                                                                     'adaptiope'])
    parser.add_argument('--nce-t', default=0.1, type=float,
                        metavar='T', help='temperature parameter for softmax')
    parser.add_argument('--nce-m', default=0.95, type=float,
                        metavar='M', help='momentum for non-parametric updates')
    parser.add_argument('--low_dim', type=int, default=512)
    parser.add_argument('--in_domain',
                        type=str2bool,
                        default=True,
                        help='in domain learning')
    parser.add_argument('--cross_domain',
                        type=str2bool,
                        default=True,
                        help='cross domain learning')
    parser.add_argument('--kmeans_all_features',
                        type=str2bool,
                        default=True,
                        help='kmeans all features to initialize clusters centroids')

    parser.add_argument('--lambda_cross_domain', default=0.01, type=float,
                        metavar='T', help='parameter for cross domain loss')
    parser.add_argument('--cross_domain_loss', default='l1', type=str,
                        choices=['l1', 'l2'], help='type of cross domain loss')
    parser.add_argument('--cross_domain_softmax',
                        type=str2bool,
                        default=False,
                        help='use softmax after cross-domain logits')

    parser.add_argument('--verbose', action='store_true', default=True,
                        help='verbose')

    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', type=int, default=65,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--save_prefix', type=str, default='', help='prefix for saving results')
    parser.add_argument('--cluster_num', type=int, default=100)
    args = parser.parse_args()

    args.cluster_num_list = [args.cluster_num, args.cluster_num*2, args.cluster_num*3, args.cluster_num*4]
    return args

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

if __name__ == "__main__":

    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    seed_torch(SEED)

    current_folder = "./"
    args.output_dir = osp.join(current_folder, args.output,
                               'seed' + str(args.seed), args.dset)
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    args.out_file = open(osp.join(args.output_dir, args.file + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()

    BEST_PATH = './model_best.pth.tar'
    

    print(args)
    
    if args.dataset == 'office_home':
        dset_loaders = office_load_idx(args)
    else:
        dset_loaders = domainnet_load_idx(args)
        


    memorySize_s = dset_loaders['source_tr'].dataset.__len__()
    memorySize_t = dset_loaders['target'].dataset.__len__()
    args.memorySize_s = memorySize_s
    args.memorySize_t = memorySize_t
    model = Model(args, memorySize_s, memorySize_t)
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
            
    # vit weights
    if args.clean_model and args.arch == 'vit':
        if os.path.isfile(args.clean_model):
            print("=> loading pretrained clean model '{}'".format(args.clean_model))

            loc = 'cuda:{}'.format(args.gpu)
            clean_checkpoint = torch.load(args.clean_model, map_location=loc)
            
            current_state = model.state_dict()
            used_pretrained_state = {}

            for k in current_state:
                if 'base_encoder' in k and 'head' not in k and 'experts' not in k and 'gates' not in k:
                    k_parts = '.'.join(k.split('.')[1:])
                    used_pretrained_state[k] = clean_checkpoint[k_parts]
                    used_pretrained_state[k].requires_grad_(True)
            current_state.update(used_pretrained_state)
            model.load_state_dict(current_state)
        else:
            print("=> no clean model found at '{}'".format(args.clean_model))
    
    model = model.cuda() 
    model.train()
  

    best_map_s2t = 0.
   
    best_res_A = [0., 0., 0.]
    best_res_B = [0., 0., 0.]
    if args.dataset == 'office_home':
        prec_nums = '1,5,15'
    else:
        prec_nums = '50,100,200'
    prec_nums = prec_nums.split(',')
    res_A, res_B = trainer_coda.test_target(args, dset_loaders['source_te'], dset_loaders['test'], model.net,prec_nums)
    Printstr1 = "Domain "+args.dset[2]+"->"+args.dset[0]+": P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), best_res_A[0],
                                                                                int(prec_nums[1]), best_res_A[1],
                                                                                    int(prec_nums[2]), best_res_A[2])
    Printstr2 = "Domain "+args.dset[0]+"->"+args.dset[2]+": P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), best_res_B[0],
                                                                    int(prec_nums[1]), best_res_B[1],
                                                                    int(prec_nums[2]), best_res_B[2])
    Printstr3 = "Precision now: "+"Domain "+args.dset[2]+"->"+args.dset[0]+": P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), res_A[0],
                                                                            int(prec_nums[1]), res_A[1],
                                                                            int(prec_nums[2]), res_A[2])
    Printstr4 = "Precision now: "+"Domain "+args.dset[0]+"->"+args.dset[2]+": P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), res_B[0],
                                                                        int(prec_nums[1]), res_B[1],
                                                                        int(prec_nums[2]), res_B[2])
            
    print("First:",Printstr1,Printstr2,Printstr3,Printstr4)
    #print("First:{:.1f}".format(map_s2t))
    
   
    if args.pretrain_epoch > 0:
        for epoch in range(0, args.pretrain_epoch):
            print("[{}/{}] Pretrain model".format(epoch + 1, args.pretrain_epoch))
            trainer_coda.train_model(args, model, dset_loaders, epoch)
            
            print('Compute MAP of Model')
            res_A, res_B = trainer_coda.test_target(args, dset_loaders['source_te'], dset_loaders['test'],
                                          model.net,prec_nums)
            
            # if map_s2t > best_map_s2t:
            #     best_map_s2t = map_s2t
                
            if best_res_A[0]< res_A[0]:
                best_res_A = res_A
            if best_res_B[0]< res_B[0]:
                best_res_B = res_B
            
            #str_s2t = "Task        : {}, Best:{:.1f}, Last:{:.1f}".format(args.dset, best_map_s2t, map_s2t)
            

            Printstr1 = "Domain "+args.dset[2]+"->"+args.dset[0]+": P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), best_res_A[0],
                                                                                int(prec_nums[1]), best_res_A[1],
                                                                                    int(prec_nums[2]), best_res_A[2])
            Printstr2 = "Domain "+args.dset[0]+"->"+args.dset[2]+": P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), best_res_B[0],
                                                                            int(prec_nums[1]), best_res_B[1],
                                                                            int(prec_nums[2]), best_res_B[2])

            Printstr3 = "Precision now: "+"Domain "+args.dset[2]+"->"+args.dset[0]+": P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), res_A[0],
                                                                                    int(prec_nums[1]), res_A[1],
                                                                                    int(prec_nums[2]), res_A[2])
            Printstr4 = "Precision now: "+"Domain "+args.dset[0]+"->"+args.dset[2]+": P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), res_B[0],
                                                                        int(prec_nums[1]), res_B[1],
                                                                        int(prec_nums[2]), res_B[2])
            
            print(Printstr1,Printstr2,Printstr3,Printstr4)
