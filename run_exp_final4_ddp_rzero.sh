#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate image_gen

# Set GPUs to use (export for all child processes)
export CUDA_VISIBLE_DEVICES=2,3

# Enable detailed error traceback for debugging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

# DDP (2 GPUs)
torchrun --standalone --nnodes=1 --nproc_per_node=2 train_ddp.py \
  --batch_size 64 \
  --num_iterations 250000 \
  --use_additional_condition \
  --weighting adaptive \
  --time_mu -2.0 \
  --time_sigma 2.0 \
  --adaptive_p 0.75 \
  --save_dir ./results/meanflow_final_4_ddp_rzero \
  --save_interval 10000 \
  --fid_interval 10000 \
  --log_interval 100 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --rzero \
  # --resume_from ./results/meanflow_final_1_ddp/checkpoint_iter_10000.pt

# # Single GPU
# python train_ddp.py --batch_size 32 ...