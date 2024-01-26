#!/bin/bash
python main.py \
  -a resnet50 \
  --batch-size 64 \
  --mlp --aug-plus --cos \
  --data-A '' \
  --data-B '' \
  --gpu 2 \
  --num_cluster '65' \
  --warmup-epoch 20 \
  --temperature 0.2 \
  --exp-dir './result' \
  --lr 0.0002 \
  --clean-model '' \
  --instcon-weight 1.0 \
  --cwcon-startepoch 200 \
  --cwcon-satureepoch 100 \
  --cwcon-weightstart 0.0 \
  --cwcon-weightsature 0.5 \
  --cwcon_filterthresh 0.2 \
  --epochs 200 \
  --selfentro-temp 0.01 \
  --selfentro-weight 1.0 \
  --selfentro-startepoch 100 \
  --aug-startepoch 50 \
  --distofdist-weight 0.5 \
  --distofdist-startepoch 200 \
  --divide-num 0.8 \
  --prec_nums '1,5,15' \
