#!/bin/bash
# Use GPU 0 and 1 (others are occupied)
CUDA_VISIBLE_DEVICES=0 python train.py \
  --batch_size 32 \
  --num_iterations 250000 \
  --use_additional_condition \
  --weighting adaptive \
  --adaptive_p 0.75 \
  --time_mu -2.0 \
  --time_sigma 2.0 \
  --ratio_r_not_equal_t 0.75 \
  --save_dir ./results/meanflow_final_1 \
  --save_interval 10000 \
  --fid_interval 10000 \
  --log_interval 100 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_weight_decay 0.0 \
  --max_grad_norm 1.0