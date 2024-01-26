#!/bin/bash
# python main.py \
#   -a resnet50 \
#   --batch-size 32 \
#   --mlp --aug-plus --cos \
#   --data-A '' \
#   --data-B '' \
#   --dataset 'pacs' \
#   --gpu 0 \
#   --num_cluster '7' \
#   --warmup-epoch 20 \
#   --temperature 0.2 \
#   --exp-dir './pacs' \
#   --lr 0.0002 \
#   --clean-model '' \
#   --instcon-weight 1.0 \
#   --cwcon-startepoch 20 \
#   --cwcon-satureepoch 100 \
#   --cwcon-weightstart 0.0 \
#   --cwcon-weightsature 1.0 \
#   --cwcon_filterthresh 0.2 \
#   --epochs 200 \
#   --selfentro-temp 0.1 \
#   --selfentro-weight 0.5 \
#   --selfentro-startepoch 100 \
#   --aug-startepoch 100 \
#   --divide-num 0.8 \
#   --prec_nums '50,100,200' \

python main.py \
  -a vit \
  --batch-size 32 \
  --mlp --aug-plus --cos \
  --data-A '' \
  --data-B '' \
  --dataset 'pacs' \
  --gpu 3 \
  --num_cluster '7' \
  --warmup-epoch 0 \
  --temperature 0.2 \
  --exp-dir './pacs' \
  --lr 5e-6 \
  --clean-model '' \
  --instcon-weight 1.0 \
  --cwcon-startepoch 0 \
  --cwcon-satureepoch 10 \
  --cwcon-weightstart 0.0 \
  --cwcon-weightsature 1.0 \
  --cwcon_filterthresh 0.2 \
  --epochs 30 \
  --selfentro-temp 0.1 \
  --selfentro-weight 0.5 \
  --selfentro-startepoch 0 \
  --aug-startepoch 0 \
  --divide-num 0.8 \
  --prec_nums '50,100,200' \