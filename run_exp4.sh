#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --batch_size 32 \
  --num_iterations 100000 \
  --use_additional_condition \
  --weighting adaptive \
  --adaptive_p 1.0 \
  --save_dir ./results/exp4_adaptive \
  --save_interval 10000 \
  --log_interval 100