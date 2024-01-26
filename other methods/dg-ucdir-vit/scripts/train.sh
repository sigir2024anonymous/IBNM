#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=1  --master_port 48724 main_dgucdir.py \
--batch-size 64 \
--mlp \
--multi_q \
--aug-plus \
--lr 0.0002 \
--epochs 200 \
--print-freq 50 \
--clean-model  \
--exp_folder_name domainnet_clipart-sketch \
--save_n_epochs 10 \
--data domainnet-ucdir-list/clipart.txt,domainnet-ucdir-list/sketch.txt \
--eval-data domainnet-ucdir-list/clipart.txt,domainnet-ucdir-list/sketch.txt \
--workers 16 \
--imagenet_pretrained '' \
--prec-nums '50,100,200' \
--moco-t 0.1 \
--hpf_range 20 \
--hpf_alpha 0.4 \
--aug_alpha 1.0 \
--warmup_epoch 0 \
--cluster_loss_w 1.0 \
--contra_intra_phase 1.0 \
--contra_intra_rgb 1.0 \
--contra_cross_phase 1.0 \
--contra_cross_rgb 1.0 \

