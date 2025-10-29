#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --batch_size 32 \
  --num_iterations 100000 \
  --use_additional_condition \
  --weighting adaptive \
  --adaptive_p 1.0 \
  --save_dir ./results/exp4_adaptive_2 \
  --save_interval 10000 \
  --log_interval 100 \
  --time_mu -2.0 \
  --time_sigma 2.0