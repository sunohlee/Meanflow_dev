#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate image_gen

# Set GPUs to use (export for all child processes)
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Enable detailed error traceback for debugging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

# DDP (4 GPUs)
torchrun --standalone --nnodes=1 --nproc_per_node=4 train_ddp.py \
  --batch_size 128 \
  --num_iterations 250000 \
  --use_additional_condition \
  --weighting adaptive \
  --time_mu -2.0 \
  --time_sigma 2.0 \
  --adaptive_p 0.75 \
  --save_dir ./results/meanflow_final_7_ddp_4gpu \
  --save_interval 10000 \
  --fid_interval 10000 \
  --log_interval 100 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --lr 8e-5 \
  --resume_from ./results/meanflow_final_7_ddp_4gpu/checkpoint_iter_10000.pt

# # Single GPU
# python train_ddp.py --batch_size 32 ...