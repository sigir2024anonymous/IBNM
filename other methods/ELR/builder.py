import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.nn.parallel as parallel
import torchvision.models as models

import loader
import os 

from einops import rearrange
import numpy as np

from timm.models.registry import register_model
from timm import create_model
from timm.models.swin_transformer import SwinTransformer
from timm.models.layers import Mlp, PatchEmbed, DropPath
from timm.models.vision_transformer import VisionTransformer
from collections import OrderedDict
import torch.distributions as dists

class UCDIR(nn.Module):

    def __init__(self, base_encoder, dim=128, K_A=65536, K_B=65536,
                 m=0.999, T=0.1, mlp=False, selfentro_temp=0.2,
                 num_cluster=None, cwcon_filterthresh=0.2,num_workers=4,gpu=0):

        super(UCDIR, self).__init__()

        self.K_A = K_A
        self.K_B = K_B
        self.m = m
        self.T = T
        self.num_workers = num_workers
        self.gpu = gpu

        self.selfentro_temp = selfentro_temp
        self.num_cluster = num_cluster
        self.cwcon_filterthresh = cwcon_filterthresh
        self.base_encoder = base_encoder
        self.mix = 1.0

        norm_layer = partial(SplitBatchNorm, num_splits=2)
        num = int(self.num_cluster[0])
        if isinstance(base_encoder, VisionTransformer):
        # vit feature 
            self.my_fc_head = Mlp(self.base_encoder.num_features, hidden_features=self.base_encoder.num_features*2, out_features=int(num_cluster[0]), act_layer=nn.GELU, drop=.2)
            self.head = nn.Linear(self.base_encoder.num_features, int(num_cluster[0])) if int(num_cluster[0]) > 0 else nn.Identity()
            self.num_patch = self.base_encoder.patch_embed.num_patches
        
            self.s_dist_alpha = nn.Parameter(torch.Tensor([1]))
            self.s_dist_beta = nn.Parameter(torch.Tensor([1]))
            self.super_ratio = nn.Parameter(torch.Tensor([-2]))
            self.unsuper_ratio = nn.Parameter(torch.Tensor([-2]))
            
            self.encoder_q = base_encoder
            self.encoder_k = base_encoder
            dim = base_encoder.embed_dim
        else:
            self.encoder_q = base_encoder(num_classes=dim)
            self.encoder_k = base_encoder(num_classes=dim, norm_layer=norm_layer)
        self.cluster_result = None

        if mlp and not isinstance(base_encoder, VisionTransformer):  # hack: brute-force replacement

            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp//2), nn.ReLU(), nn.Linear(dim_mlp//2, dim_mlp//4))
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp//2), nn.ReLU(), nn.Linear(dim_mlp//2, dim_mlp//4))
            self.head = nn.Linear(dim_mlp//4, int(num_cluster[0])) if int(num_cluster[0]) > 0 else nn.Identity()
            
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        self.encoder_q = self.encoder_q.cuda(self.gpu)
        self.encoder_k = self.encoder_k.cuda(self.gpu)


        self.register_buffer("queue_A", torch.randn(dim, K_A))

        self.queue_A = F.normalize(self.queue_A, dim=0)

        self.register_buffer("queue_A_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_B", torch.randn(dim, K_B))
        self.queue_B = F.normalize(self.queue_B, dim=0)
        self.register_buffer("queue_B_ptr", torch.zeros(1, dtype=torch.long))



    @torch.no_grad()
    def _momentum_update_key_encoder(self):

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_singlegpu(self, keys, key_ids, domain_id):
        if domain_id == 'A':
            self.queue_A.index_copy_(1, key_ids, keys.T)
        elif domain_id == 'B':
            self.queue_B.index_copy_(1, key_ids, keys.T)

    @torch.no_grad()
    def _batch_shuffle_singlegpu(self, x):

        idx_shuffle = torch.randperm(x.shape[0]).cuda(self.gpu)
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_singlegpu(self, x, idx_unshuffle):

        return x[idx_unshuffle]
    

    def forward(self, im_q_A, im_q_B, im_k_A=None, im_id_A=None,
                im_k_B=None, im_id_B=None, is_eval=False,
                cluster_result=None,criterion=None,divide= False,psl_A= None,psl_B= None,
                mem_fea=None,mem_cls=None,class_weight_src=None,mix_indice_A=None,mix_indice_B=None,):
        if divide:
            im_q = torch.cat([im_q_A, im_q_B], dim=0)
            if is_eval:
                k = self.encoder_k(im_q)
                k = F.normalize(k, dim=1)

                k_A, k_B = torch.split(k, im_q_A.shape[0])
                return k_A, k_B            
            if isinstance(self.base_encoder, VisionTransformer):

                token = self.encoder_q.patch_embed(im_q)
                logits, p, attn = self.encoder_q.forward_features(token, patch=True)
                pred = self.my_fc_head(logits)
                
                A_logits, B_logits = torch.split(logits, im_q_A.shape[0])
                A_attn, B_attn = torch.split(attn, im_q_A.shape[0])
                A_pred, B_pred = torch.split(pred, im_q_A.shape[0])
            else:
                feature = self.encoder_q(im_q)
                pred = self.head(feature)
            A_pred, B_pred = torch.split(pred, im_q_A.shape[0])

            loss_A = CrossEntropyLabelSmooth(reduction='none', num_classes=int(self.num_cluster[0]),
                                                    epsilon=0.1)(A_pred, psl_A).requires_grad_(True)
            loss_B = CrossEntropyLabelSmooth(reduction='none', num_classes=int(self.num_cluster[0]),
                                                    epsilon=0.1)(B_pred, psl_B).requires_grad_(True)
            
            
            if isinstance(self.base_encoder, VisionTransformer):    
                
                B_token, A_token = torch.split(token, im_q_A.shape[0])
                A_scores = self.attn_map(attn=A_attn)
                B_scores = self.attn_map(attn=B_attn)
                
                dis = -torch.mm(A_logits.detach(), mem_fea.t())
                for di in range(dis.size(0)):
                    dis[di, im_id_A[di]] = torch.max(dis)
                _, p1 = torch.sort(dis, dim=1)
                w = torch.zeros(A_logits.size(0), mem_fea.size(0)).cuda()
                for wi in range(w.size(0)):
                    for wj in range(5):
                        w[wi][p1[wi, wj]] = 1 / 5
                weight_A, pred_A = torch.max(w.mm(mem_cls), 1)
                
                dis = -torch.mm(B_logits.detach(), mem_fea.t())
                for di in range(dis.size(0)):
                    dis[di, im_id_B[di]] = torch.max(dis)
                _, p1 = torch.sort(dis, dim=1)
                w = torch.zeros(B_logits.size(0), mem_fea.size(0)).cuda()
                for wi in range(w.size(0)):
                    for wj in range(5):
                        w[wi][p1[wi, wj]] = 1 / 5
                weight_B, pred_B = torch.max(w.mm(mem_cls), 1)
                
                gpu = torch.cuda.current_device()

                A_lambda = dists.Beta(softplus(self.s_dist_alpha), softplus(self.s_dist_beta)).rsample((im_q_A.shape[0], self.num_patch,)).to(f'cuda:{gpu}').squeeze(-1)
                B_lambda = 1 - A_lambda

                feature_space_loss, label_space_loss = self.mix_domainA_domainB(B_token,A_token,B_lambda,A_lambda,pred_A,pred_B,psl_A,psl_B,B_logits,A_logits,B_scores,A_scores,mem_fea,mem_cls,weight_A,weight_B)
                mix_loss = softplus(self.super_ratio)*feature_space_loss + softplus(self.unsuper_ratio) * label_space_loss
            else:
                # mixup loss
                alpha = 0.3
                lam = np.random.beta(alpha, alpha)
                mixed_images = lam * im_q_A + (1 - lam) * im_q_B
                target_A = torch.from_numpy(convert_to_onehot(psl_A, int(self.num_cluster[0]))).to(f'cuda:{self.gpu}',dtype=torch.float32)
                target_B = torch.from_numpy(convert_to_onehot(psl_B, int(self.num_cluster[0]))).to(f'cuda:{self.gpu}',dtype=torch.float32)
                mixed_targets = (lam * target_A + (1 - lam) * target_B).detach()
                update_batch_stats(self.encoder_q, False)
                update_batch_stats(self.head, False)
                mixed_feature = self.encoder_q(mixed_images)
                mixed_logits = self.head(mixed_feature)
                update_batch_stats(self.encoder_q, True)
                update_batch_stats(self.head, True)
                mixed_pred = F.softmax(mixed_logits, dim=-1)
                mix_loss = self.mix*nn.KLDivLoss(reduction='batchmean')(mixed_pred.log(), mixed_targets)
                
            return loss_A,loss_B,mix_loss
            
        else:
            im_q = torch.cat([im_q_A, im_q_B], dim=0)

            if is_eval:
                k = self.encoder_k(im_q)
                k = F.normalize(k, dim=1)

                k_A, k_B = torch.split(k, im_q_A.shape[0])
                return k_A, k_B

            q = self.encoder_q(im_q)
            q = F.normalize(q, dim=1)

            q_A, q_B = torch.split(q, im_q_A.shape[0])

            im_k = torch.cat([im_k_A, im_k_B], dim=0)

            with torch.no_grad():
                self._momentum_update_key_encoder()

                im_k, idx_unshuffle = self._batch_shuffle_singlegpu(im_k)

                k = self.encoder_k(im_k)
                k = F.normalize(k, dim=1)

                k = self._batch_unshuffle_singlegpu(k, idx_unshuffle)

                k_A, k_B = torch.split(k, im_k_A.shape[0])

            self._dequeue_and_enqueue_singlegpu(k_A, im_id_A, 'A')
            self._dequeue_and_enqueue_singlegpu(k_B, im_id_B, 'B')

            loss_instcon_A, \
            loss_instcon_B = self.instance_contrastive_loss(q_A, k_A, im_id_A,
                                                            q_B, k_B, im_id_B,
                                                            criterion)

            losses_instcon = {'domain_A': loss_instcon_A,
                              'domain_B': loss_instcon_B}
            
            if (cluster_result is not None) & isinstance(self.base_encoder, VisionTransformer):
                loss_cwcon_A, \
                loss_cwcon_B = self.cluster_contrastive_loss(q_A, k_A, im_id_A,
                                                             q_B, k_B, im_id_B,
                                                             cluster_result)
                
                losses_cwcon = {'domain_A': loss_cwcon_A,
                                'domain_B': loss_cwcon_B}
                loss_mix_A, \
                loss_mix_B = self.mixup_in_domain(im_q_A, im_q_B, im_id_A, im_id_B, mix_indice_A, mix_indice_B, psl_A, psl_B)

                losses_mix = {'domain_A': loss_mix_A,
                                'domain_B': loss_mix_B}

                losses_selfentro = self.self_entropy_loss(q_A, q_B, cluster_result)

                losses_distlogit = self.dist_of_logit_loss(q_A, q_B, cluster_result, self.num_cluster)

                return losses_instcon, q_A, q_B, losses_selfentro, losses_distlogit, losses_cwcon, losses_mix
            elif cluster_result is not None:
                loss_cwcon_A, \
                loss_cwcon_B = self.cluster_contrastive_loss(q_A, k_A, im_id_A,
                                                             q_B, k_B, im_id_B,
                                                             cluster_result)
                                
                losses_cwcon = {'domain_A': loss_cwcon_A,
                                'domain_B': loss_cwcon_B}
                loss_mix_A, \
                loss_mix_B = self.mixup_in_domain(im_q_A, im_q_B, im_id_A, im_id_B, mix_indice_A, mix_indice_B, psl_A, psl_B)

                losses_mix = {'domain_A': loss_mix_A,
                                'domain_B': loss_mix_B}
                
                losses_selfentro = self.self_entropy_loss(q_A, q_B, cluster_result)

                losses_distlogit = self.dist_of_logit_loss(q_A, q_B, cluster_result, self.num_cluster)

                return losses_instcon, q_A, q_B, losses_selfentro, losses_distlogit, losses_cwcon, losses_mix, None
            else:
                return losses_instcon, None, None, None, None, None, None, None
            
    def attn_map(self, patch=None, label=None, attn=None, mlp=False):
        scores = attn
            
        n_p_e = int(np.sqrt(self.num_patch))
        n_p_f = int(np.sqrt(scores.size(1)))

        scores = F.interpolate(rearrange(scores, 'B (H W) -> B 1 H W', H = n_p_f), size=(n_p_e, n_p_e)).squeeze(1)
        scores = rearrange(scores, 'B H W -> B (H W)')
        return scores.softmax(dim=-1)

    def mix_domainA_domainB_no_divide(self,s_token,t_token,s_lambda,t_lambda,pred_A,pred_B,s_logits,t_logits,s_scores,t_scores,mem_fea,mem_cls,weight_A, weight_B):
        m_s_t_tokens = mix_token(s_token, t_token, s_lambda)        
        m_s_t_logits, m_s_t_p, _ = self.encoder_q.forward_features(m_s_t_tokens, patch=True)
        m_s_t_pred = self.my_fc_head(m_s_t_logits)
        t_scores = (torch.ones(32,self.num_patch)/self.num_patch).cuda()
        s_lambda = mix_lambda_atten(s_scores,t_scores,s_lambda,self.num_patch)#with attention map
        t_lambda = 1 - s_lambda

        m_s_t_s = cosine_distance(m_s_t_logits, s_logits)
        m_s_t_s_similarity = mixup_unsupervised_dis(m_s_t_s, s_lambda)
        m_s_t_t = cosine_distance(m_s_t_logits, t_logits)
        m_s_t_t_similarity = mixup_unsupervised_dis(m_s_t_t, t_lambda)
        feature_space_loss= (m_s_t_s_similarity + m_s_t_t_similarity) / torch.sum(s_lambda + t_lambda)

        unsuper_m_s_s_loss = mixup_soft_ce(m_s_t_pred,      pred_B  ,weight_B, s_lambda)
        unsuper_m_s_t_loss = mixup_soft_ce(m_s_t_pred,      pred_A  ,weight_A, t_lambda)
        label_space_loss  = (unsuper_m_s_s_loss + unsuper_m_s_t_loss)/torch.sum(s_lambda + t_lambda)

        return feature_space_loss,label_space_loss
    
    # for divide
    def mix_domainA_domainB(self,B_token,A_token,B_lambda,A_lambda,pred_A,pred_B,psl_A,psl_B,B_logits,A_logits,B_scores,A_scores,mem_fea,mem_cls,weight_A,weight_B,):
        # t: A
        m_b_a_tokens = mix_token(B_token, A_token, B_lambda)        
        m_b_a_logits, m_b_a_p, _ = self.encoder_q.forward_features(m_b_a_tokens, patch=True)
        m_b_a_pred = self.my_fc_head(m_b_a_logits)

        B_lambda = mix_lambda_atten(B_scores,A_scores,B_lambda,self.num_patch)#with attention map

        A_lambda = 1 - B_lambda

        b_onehot = torch.tensor(convert_to_onehot(psl_B, m_b_a_pred.shape[1]), dtype=torch.float32).cuda()
        a_onehot = torch.tensor(convert_to_onehot(psl_A, m_b_a_pred.shape[1]), dtype=torch.float32).cuda()
        m_b_a_b = cosine_distance(m_b_a_logits, B_logits)
        m_b_a_b_similarity = mixup_supervised_dis(m_b_a_b, b_onehot, B_lambda)
        m_b_a_a = cosine_distance(m_b_a_logits, A_logits)
        m_b_a_a_similarity = mixup_supervised_dis(m_b_a_a, a_onehot, A_lambda)
        feature_space_loss= (m_b_a_b_similarity + m_b_a_a_similarity) / torch.sum(B_lambda + A_lambda)
        super_m_b_a_b_loss = mixup_soft_ce(m_b_a_pred, psl_B,  weight_B, B_lambda)
        super_m_b_a_a_loss = mixup_soft_ce(m_b_a_pred, psl_A,  weight_A, A_lambda)
        label_space_loss  = (super_m_b_a_b_loss + super_m_b_a_a_loss) / torch.sum(B_lambda + A_lambda)

        return feature_space_loss,label_space_loss
            

    def instance_contrastive_loss(self,
                                  q_A, k_A, im_id_A,
                                  q_B, k_B, im_id_B,
                                  criterion):

        l_pos_A = torch.einsum('nc,nc->n', [q_A, k_A]).unsqueeze(-1)
        l_pos_B = torch.einsum('nc,nc->n', [q_B, k_B]).unsqueeze(-1)

        l_all_A = torch.matmul(q_A, self.queue_A.clone().detach())
        l_all_B = torch.matmul(q_B, self.queue_B.clone().detach())

        mask_A = torch.arange(self.queue_A.shape[1]).cuda(self.gpu) != im_id_A[:, None]

        l_neg_A = torch.masked_select(l_all_A, mask_A).reshape(q_A.shape[0], -1)

        mask_B = torch.arange(self.queue_B.shape[1]).cuda(self.gpu) != im_id_B[:, None]
        l_neg_B = torch.masked_select(l_all_B, mask_B).reshape(q_B.shape[0], -1)

        logits_A = torch.cat([l_pos_A, l_neg_A], dim=1)
        logits_B = torch.cat([l_pos_B, l_neg_B], dim=1)

        logits_A /= self.T
        logits_B /= self.T

        labels_A = torch.zeros(logits_A.shape[0], dtype=torch.long).cuda(self.gpu)
        labels_B = torch.zeros(logits_B.shape[0], dtype=torch.long).cuda(self.gpu)

        loss_A = criterion(logits_A, labels_A).requires_grad_(True)
        loss_B = criterion(logits_B, labels_B).requires_grad_(True)

        return loss_A, loss_B

    def cluster_contrastive_loss(self, q_A, k_A, im_id_A, q_B, k_B, im_id_B, cluster_result):

        all_losses = {'domain_A': [], 'domain_B': []}

        for domain_id in ['A', 'B']:
            if domain_id == 'A':
                im_id = im_id_A.cuda(self.gpu)
                q_feat = q_A
                k_feat = k_A
                queue = self.queue_A.clone().detach()
            else:
                im_id = im_id_B.cuda(self.gpu)
                q_feat = q_B
                k_feat = k_B
                queue = self.queue_B.clone().detach()

            mask = 1.0
            for n, (im2cluster, prototypes) in enumerate(zip(cluster_result['im2cluster_' + domain_id],
                                                             cluster_result['centroids_' + domain_id])):
                
                cor_cluster_id = im2cluster[im_id]

                mask *= torch.eq(cor_cluster_id.contiguous().view(-1, 1),
                                 im2cluster.contiguous().view(1, -1)).float().cuda(self.gpu)  # batch size x queue length

                all_score = torch.div(torch.matmul(q_feat, queue), self.T)

                exp_all_score = torch.exp(all_score)

                log_prob = all_score - torch.log(exp_all_score.sum(1, keepdim=True))

                log_prob = log_prob.cuda(self.gpu)
                mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

                cor_proto = prototypes[cor_cluster_id]

                k_feat = k_feat.cuda(self.gpu)
                cor_proto = cor_proto.cuda(self.gpu)
                prototypes = prototypes.cuda(self.gpu)
                inst_pos_value = torch.exp(
                    torch.div(torch.einsum('nc,nc->n', [k_feat, cor_proto]), self.T))  # N

                inst_all_value = torch.exp(
                    torch.div(torch.einsum('nc,ck->nk', [k_feat, prototypes.T]), self.T))  # N x r

                filters = ((inst_pos_value / torch.sum(inst_all_value, dim=1)) > self.cwcon_filterthresh).float()

                filters_sum = filters.sum()

                # f = filters * mean_log_prob_pos
                loss = - (filters * mean_log_prob_pos).sum() / (filters_sum + 1e-8)
                loss.requires_grad_(True)

                all_losses['domain_' + domain_id].append(loss)
                # a = torch.stack(all_losses['domain_A'])

        return torch.mean(torch.stack(all_losses['domain_A'])), torch.mean(torch.stack(all_losses['domain_B']))

    def mixup_in_domain(self, im_q_A, im_q_B, im_id_A, im_id_B, mix_indice_A, mix_indice_B, psl_A, psl_B):
        mix_loss_A,mix_loss_B = None,None
        
        if isinstance(self.base_encoder, VisionTransformer):
            if psl_A is not None:
                # find corresponding image
                index_A=[]
                for i in mix_indice_A:
                    #index_A.append(im_id_A.index(i))
                    index_A.append(torch.nonzero(im_id_A == i).squeeze_())
                for i in range(len(index_A)):
                    if index_A[i].shape != torch.Size([]):
                        index_A[i] = index_A[i][0]
                im_A = im_q_A[index_A,:,:,:]
                # mixup loss
                alpha = 0.3
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(len(psl_A)).cuda(self.gpu)
                mixed_images = lam * im_A + (1 - lam) * im_A[index]
                target_A = torch.from_numpy(convert_to_onehot(torch.tensor(psl_A), int(self.num_cluster[0]))).to(f'cuda:{self.gpu}',dtype=torch.float32)
                mixed_targets = lam * target_A + (1 - lam) * target_A[index]
 
                update_batch_stats(self.encoder_q, False)
                mixed_token = self.encoder_q.patch_embed(mixed_images)
                logits, p, attn = self.encoder_q.forward_features(mixed_token, patch=True)
                mixed_pred = self.my_fc_head(logits)

                update_batch_stats(self.encoder_q, True)
                mixed_pred = F.softmax(mixed_pred, dim=-1)
                mix_loss_A = self.mix*nn.KLDivLoss(reduction='batchmean')(mixed_pred.log(), mixed_targets)

            if psl_B is not None:
                index_B=[]
                for i in mix_indice_B:
                    index_B.append(torch.nonzero(im_id_B == i).squeeze_())
                for i in range(len(index_B)):
                    if index_B[i].shape != torch.Size([]):
                        index_B[i] = index_B[i][0]
                im_B = im_q_B[index_B,:,:,:]
                # mixup loss
                alpha = 0.3
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(len(psl_B)).cuda(self.gpu)
                mixed_images = lam * im_B + (1 - lam) * im_B[index]
                target_B = torch.from_numpy(convert_to_onehot(torch.tensor(psl_B), int(self.num_cluster[0]))).to(f'cuda:{self.gpu}',dtype=torch.float32)
                mixed_targets = lam * target_B + (1 - lam) * target_B[index]
                update_batch_stats(self.encoder_q, False)
                mixed_token = self.encoder_q.patch_embed(mixed_images)
                logits, p, attn = self.encoder_q.forward_features(mixed_token, patch=True)
                mixed_pred = self.my_fc_head(logits)
                update_batch_stats(self.encoder_q, True)
                mixed_pred = F.softmax(mixed_pred, dim=-1)
                mix_loss_B = self.mix*nn.KLDivLoss(reduction='batchmean')(mixed_pred.log(), mixed_targets)
            
        else:    
            if psl_A is not None:
                # find corresponding image
                index_A=[]
                for i in mix_indice_A:
                    index_A.append(torch.nonzero(im_id_A == i).squeeze_())
                for i in range(len(index_A)):
                    if index_A[i].shape != torch.Size([]):
                        index_A[i] = index_A[i][0]
                im_A = im_q_A[index_A,:,:,:]
                # mixup loss
                alpha = 0.3
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(len(psl_A)).cuda(self.gpu)
                mixed_images = lam * im_A + (1 - lam) * im_A[index]
                target_A = torch.from_numpy(convert_to_onehot(torch.tensor(psl_A), int(self.num_cluster[0]))).to(f'cuda:{self.gpu}',dtype=torch.float32)
                mixed_targets = lam * target_A + (1 - lam) * target_A[index]
                update_batch_stats(self.encoder_q, False)
                update_batch_stats(self.head, False)
                mixed_feature = self.encoder_q(mixed_images)
                mixed_logits = self.head(mixed_feature)
                update_batch_stats(self.encoder_q, True)
                update_batch_stats(self.head, True)
                mixed_pred = F.softmax(mixed_logits, dim=-1)
                mix_loss_A = self.mix*nn.KLDivLoss(reduction='batchmean')(mixed_pred.log(), mixed_targets)

            if psl_B is not None:
                index_B=[]
                for i in mix_indice_B:
                    index_B.append(torch.nonzero(im_id_B == i).squeeze_())
                for i in range(len(index_B)):
                    if index_B[i].shape != torch.Size([]):
                        index_B[i] = index_B[i][0]
                im_B = im_q_B[index_B,:,:,:]
                # mixup loss
                alpha = 0.3
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(len(psl_B)).cuda(self.gpu)
                mixed_images = lam * im_B + (1 - lam) * im_B[index]
                target_B = torch.from_numpy(convert_to_onehot(torch.tensor(psl_B), int(self.num_cluster[0]))).to(f'cuda:{self.gpu}',dtype=torch.float32)
                mixed_targets = lam * target_B + (1 - lam) * target_B[index]
                update_batch_stats(self.encoder_q, False)
                update_batch_stats(self.head, False)
                mixed_feature = self.encoder_q(mixed_images)
                mixed_logits = self.head(mixed_feature)
                update_batch_stats(self.encoder_q, True)
                update_batch_stats(self.head, True)
                mixed_pred = F.softmax(mixed_logits, dim=-1)
                mix_loss_B = self.mix*nn.KLDivLoss(reduction='batchmean')(mixed_pred.log(), mixed_targets)
        return mix_loss_A,mix_loss_B
            
            
            
        
    def self_entropy_loss(self, q_A, q_B, cluster_result):

        losses_selfentro = {}
        for feat_domain in ['A', 'B']:
            if feat_domain == 'A':
                feat = q_A
            else:
                feat = q_B

            cross_proto_domains = ['A', 'B']
            for cross_proto_domain in cross_proto_domains:
                for n, (im2cluster, self_proto, cross_proto) in enumerate(
                        zip(cluster_result['im2cluster_' + feat_domain],
                            cluster_result['centroids_' + feat_domain],
                            cluster_result['centroids_' + cross_proto_domain])):

                    if str(self_proto.shape[0]) in self.num_cluster:

                        key_selfentro = 'feat_domain_' + feat_domain + '-proto_domain_' \
                                        + cross_proto_domain + '-cluster_' + str(cross_proto.shape[0])
                        if key_selfentro in losses_selfentro.keys():
                            losses_selfentro[key_selfentro].append(self.self_entropy_loss_onepair(feat, cross_proto))
                        else:
                            losses_selfentro[key_selfentro] = [self.self_entropy_loss_onepair(feat, cross_proto)]
        return losses_selfentro

    def self_entropy_loss_onepair(self, feat, prototype):

        feat = feat.cuda(self.gpu)
        prototype = prototype.cuda(self.gpu)
        logits = torch.div(torch.matmul(feat, prototype.T), self.selfentro_temp)

        self_entropy = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * F.softmax(logits, dim=1), dim=1))
        self_entropy.requires_grad_(True)

        return self_entropy

    def dist_of_logit_loss(self, q_A, q_B, cluster_result, num_cluster):

        all_losses = {}

        for n, (proto_A, proto_B) in enumerate(zip(cluster_result['centroids_A'],
                                                   cluster_result['centroids_B'])):

            if str(proto_A.shape[0]) in num_cluster:
                domain_ids = ['A', 'B']

                for domain_id in domain_ids:
                    if domain_id == 'A':
                        feat = q_A
                    elif domain_id == 'B':
                        feat = q_B
                    else:
                        feat = torch.cat([q_A, q_B], dim=0)

                    loss_A_B = self.dist_of_dist_loss_onepair(feat, proto_A, proto_B).requires_grad_(True)

                    key_A_B = 'feat_domain_' + domain_id + '_A_B' + '-cluster_' + str(proto_A.shape[0])
                    if key_A_B in all_losses.keys():
                        all_losses[key_A_B].append(loss_A_B.mean())
                    else:
                        all_losses[key_A_B] = [loss_A_B.mean()]

        return all_losses
    
    def cluster_divide_loss(self, features1, features2,index,domain_id):

        cluster_result = self.cluster_result
            
        # cur_cwcon_weight =1
            
        if domain_id == 'A':
            queue = self.queue_A.clone().detach()
        else:
            queue = self.queue_B.clone().detach()
            
        loss = 0
        mask = 1.0
        index = index.cuda(self.gpu)
        if len(index) == 0:
            return loss
        else:
            for n, (im2cluster, prototypes) in enumerate(zip(cluster_result['im2cluster_' + domain_id],
                                                             cluster_result['centroids_' + domain_id])):

                cor_cluster_id = im2cluster[index]

                mask *= torch.eq(cor_cluster_id.contiguous().view(-1, 1),
                                 im2cluster.contiguous().view(1, -1)).float().cuda(self.gpu)  # batch size x queue lengthh
                all_score1 = torch.div(torch.matmul(features1, queue), self.T)
                all_score2 = torch.div(torch.matmul(features2, queue), self.T)
                exp_all_score1 = torch.exp(all_score1)
                exp_all_score2 = torch.exp(all_score2)
                log_prob1 = all_score1 - torch.log(exp_all_score1.sum(1, keepdim=True))
                log_prob2 = all_score2 - torch.log(exp_all_score2.sum(1, keepdim=True))

                log_prob1 = log_prob1.cuda(self.gpu)
                log_prob2 = log_prob2.cuda(self.gpu)
                mean_log_prob_pos1 = (mask * log_prob1).sum(1) / (mask.sum(1) + 1e-8)
                mean_log_prob_pos2 = (mask * log_prob2).sum(1) / (mask.sum(1) + 1e-8)

                if len(index) != 0:
                    loss1 = - mean_log_prob_pos1.sum() / (len(index) + 1e-8)
                    loss2 = - mean_log_prob_pos2.sum() / (len(index) + 1e-8)
                else:
                    loss1 = - mean_log_prob_pos1.sum()
                    loss2 = - mean_log_prob_pos2.sum()
                loss = (loss1 + loss2) / 2
            loss.requires_grad_(True)
            return loss
            
            
    
    def dist_of_dist_loss_onepair(self, feat, proto_1, proto_2):

        proto1_distlogits = self.dist_cal(feat, proto_1)
        proto2_distlogits = self.dist_cal(feat, proto_2)
        # cross-domain distance-of-distance L2 distance
        loss_A_B = F.pairwise_distance(proto1_distlogits, proto2_distlogits, p=2) ** 2
        loss_A_B.requires_grad_(True)

        return loss_A_B

    def dist_cal(self, feat, proto, temp=0.01):
        
        feat = feat.cuda(self.gpu)
        proto = proto.cuda(self.gpu)
        proto_logits = F.softmax(torch.matmul(feat, proto.T) / temp, dim=1)

        proto_distlogits = 1.0 - torch.matmul(F.normalize(proto_logits, dim=1), F.normalize(proto_logits.T, dim=0))

        return proto_distlogits


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss
    
def cosine_distance(source_hidden_features, target_hidden_features):
    "similarity between different features"
    n_s = source_hidden_features.shape[0]
    n_t = target_hidden_features.shape[0]
    temp_matrix = torch.mm(source_hidden_features, target_hidden_features.t())
    for i in range(n_s):
        vec = source_hidden_features[i]
        temp_matrix[i] /= torch.norm(vec, p=2)
    for j in range(n_t):
        vec = target_hidden_features[j]
        temp_matrix[:, j] /= torch.norm(vec, p=2)
    return temp_matrix
def convert_to_onehot(s_label, class_num):
    s_sca_label = s_label.cpu().data.numpy()
    return np.eye(class_num)[s_sca_label]

def mixup_soft_ce(pred, targets, weight, lam):
    """ mixed categorical cross-entropy loss
    """
    loss = torch.nn.CrossEntropyLoss(reduction='none')(pred, targets)
    loss = torch.sum(lam* weight* loss) / (torch.sum(weight*lam).item())
    loss = loss*torch.sum(lam)
    return loss

def mixup_supervised_dis(preds,s_label, lam):
    """ mixup_distance_in_feature_space_for_intermediate_source
    """
    label = torch.mm(s_label,s_label.t())
    mixup_loss = -torch.sum(label * F.log_softmax(preds,dim=1), dim=1)
    mixup_loss = torch.sum (torch.mul(mixup_loss, lam))
    return mixup_loss

def mixup_unsupervised_dis(preds,lam):
    """ mixup_distance_in_feature_space_for_intermediate_target
    """
    label = torch.eye(preds.shape[0]).cuda()
    mixup_loss = -torch.sum(label* F.log_softmax(preds,dim=1), dim=1)
    mixup_loss = torch.sum(torch.mul(mixup_loss,lam))
    return mixup_loss

def mix_token(s_token,t_token,s_lambda):
    s_token = torch.einsum('bnc,bn -> bnc', s_token, s_lambda)
    t_token = torch.einsum('bnc,bn -> bnc', t_token, 1-s_lambda)
    m_tokens =s_token+t_token
    return m_tokens

def mix_lambda_atten(s_scores,t_scores,s_lambda,num_patch):
    t_lambda = 1-s_lambda
    if s_scores is None or t_scores is None:
        s_lambda = torch.sum(s_lambda, dim=1)/num_patch # important for /self.num_patch
        t_lambda = torch.sum(t_lambda, dim=1)/num_patch
        s_lambda = s_lambda/(s_lambda+t_lambda)        
    else:
        s_lambda = torch.sum(torch.mul(s_scores, s_lambda), dim=1)/num_patch # important for /self.num_patch
        t_lambda = torch.sum(torch.mul(t_scores, t_lambda), dim=1)/num_patch
        s_lambda = s_lambda/(s_lambda+t_lambda)
    return s_lambda

def mix_lambda (s_lambda,t_lambda):
    return torch.sum(s_lambda,dim=1) / (torch.sum(s_lambda,dim=1) + torch.sum(t_lambda,dim=1))

def softplus(x):
    return  torch.log(1+torch.exp(x))

def update_batch_stats(model, flag):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.update_batch_stats = flag

# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape

        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)

            outcome = F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)
            


            
