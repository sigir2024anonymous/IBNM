#!/bin/bash

python main.py \
  -a resnet50 \
  --batch-size 64 \
  --mlp --aug-plus --cos \
  --data-A '' \
  --data-B '' \
  --dataset 'domainnet' \
  --gpu 5 \
  --num_cluster '7' \
  --warmup-epoch 20 \
  --temperature 0.2 \
  --exp-dir './domainnet' \
  --lr 0.0002 \
  --clean-model '' \
  --instcon-weight 1.0 \
  --cwcon-startepoch 20 \
  --cwcon-satureepoch 100 \
  --cwcon-weightstart 0.0 \
  --cwcon-weightsature 1.0 \
  --cwcon_filterthresh 0.2 \
  --epochs 200 \
  --selfentro-temp 0.1 \
  --selfentro-weight 0.5 \
  --selfentro-startepoch 100 \
  --aug-startepoch 170 \
  --distofdist-weight 0.1 \
  --distofdist-startepoch 200 \
  --divide-num 0.8 \
  --prec_nums '50,100,200' \


