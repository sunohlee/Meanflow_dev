#!/usr/bin/env python3
"""
Skeleton training script for generative models
Students need to implement their own model classes and modify this script accordingly.
"""

import json
import argparse
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import matplotlib.pyplot as plt
import os
import time
from datetime import timedelta
from tqdm import tqdm

from src.network import UNet
from dataset import SimpsonsDataModule, get_data_iterator
from src.utils import tensor_to_pil_image, get_current_time, save_model, seed_everything
from src.base_model import BaseScheduler, BaseGenerativeModel
from custom_model import create_custom_model


def setup_ddp(rank, world_size):
    """Initialize DDP - use torchrun's automatic setup"""
    # torchrun sets MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE automatically
    # We just need to call init_process_group
    
    # Suppress NCCL debug info
    os.environ['NCCL_DEBUG'] = 'WARN'  # Only show warnings and errors
    os.environ['NCCL_DEBUG_SUBSYS'] = 'OFF'  # Disable subsystem debug info
    # os.environ['NCCL_DEBUG'] = 'VERSION'  # Even less verbose option
    
    print(f"[Rank {rank}] Initializing process group...")
    print(f"[Rank {rank}] Available CUDA devices: {torch.cuda.device_count()}")
    print(f"[Rank {rank}] Env RANK: {os.environ.get('RANK')}")
    print(f"[Rank {rank}] Env WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print(f"[Rank {rank}] Env LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    print(f"[Rank {rank}] Env MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"[Rank {rank}] Env MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    
    try:
        # Use default initialization (torchrun sets everything)
        # Set long timeout for FID evaluation (rank 0 may take time while others wait)
        timeout = timedelta(minutes=30)  # 30 minutes timeout for FID evaluation
        
        dist.init_process_group(
            backend="nccl",
            timeout=timeout
        )
        
        # Get local rank from environment (set by torchrun)
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        torch.cuda.set_device(local_rank)
        
        print(f"[Rank {rank}] Set to CUDA device: {local_rank}")
        print(f"[Rank {rank}] Device name: {torch.cuda.get_device_name(local_rank)}")
        print(f"[Rank {rank}] Process group initialized successfully!")
        
        return local_rank
    except Exception as e:
        print(f"[Rank {rank}] ERROR during DDP initialization: {e}")
        import traceback
        traceback.print_exc()
        raise


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def train_model(
    model,
    train_iterator,
    num_iterations=100000,
    lr=1e-4,
    save_dir="./results",
    device="cpu",
    log_interval=500,
    save_interval=10000,
    fid_interval=50000,
    model_config=None,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_weight_decay=0.0,
    max_grad_norm=1.0,
    rank=0,
    world_size=1,
    start_iteration=0,
    resume_checkpoint_path=None
):
    """
    Train a generative model.
    
    Args:
        model: Generative model to train
        train_iterator: Training data iterator
        num_iterations: Number of training iterations
        lr: Learning rate
        save_dir: Directory to save checkpoints
        device: Device to run training on
        log_interval: Interval for logging
        save_interval: Interval for saving checkpoints and samples
        model_config: Model configuration dictionary to save with checkpoints
    """
    
    # Create save directory (only on rank 0)
    save_dir = Path(save_dir)
    if rank == 0:
        save_dir.mkdir(exist_ok=True, parents=True)
    
    # Synchronize all processes
    if world_size > 1:
        dist.barrier()
    
    # Create optimizer with MeanFlow parameters
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=1e-08
    )
    
    # Load checkpoint if resuming (only on rank 0, then broadcast)
    if resume_checkpoint_path is not None:
        if rank == 0:
            print(f"\nLoading checkpoint from: {resume_checkpoint_path}")
        
        checkpoint = torch.load(resume_checkpoint_path, map_location=device) if rank == 0 else None
        
        # Get model without DDP wrapper
        model_to_load = model.module if hasattr(model, 'module') else model
        
        if rank == 0:
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            model_to_load.load_state_dict(state_dict)
            print(f"Model state loaded from checkpoint")
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Optimizer state loaded from checkpoint")
    
    # Synchronize checkpoint loading across all ranks
    # DDP automatically broadcasts rank 0's parameters during initialization,
    # but since we loaded checkpoint AFTER DDP wrap, we need to manually broadcast
    if world_size > 1 and resume_checkpoint_path is not None:
        # Broadcast parameters from rank 0 to all other ranks
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        if rank == 0:
            print("Checkpoint weights broadcasted to all ranks")
        
        # Verify all ranks have same weights (checksum)
        dist.barrier()
        param_sum = sum(p.sum().item() for p in model.parameters())
        param_sums = [torch.tensor(0.0, device=device) for _ in range(world_size)]
        dist.all_gather(param_sums, torch.tensor(param_sum, device=device))
        
        if rank == 0:
            param_sums_cpu = [p.item() for p in param_sums]
            if len(set([f"{p:.6f}" for p in param_sums_cpu])) == 1:
                print(f"✓ All ranks have identical weights (checksum: {param_sums_cpu[0]:.6f})")
            else:
                print(f"✗ WARNING: Ranks have different weights! {param_sums_cpu}")
    
    if world_size > 1:
        dist.barrier()
    
    # Save training configuration (only on rank 0)
    if rank == 0:
        config = {
            'num_iterations': num_iterations,
            'lr': lr,
            'log_interval': log_interval,
            'save_interval': save_interval,
            'device': str(device),
            'world_size': world_size,
        }
        with open(save_dir / 'training_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Training config saved to: {save_dir / 'training_config.json'}")
    
    # Training loop
    train_losses = []
    
    if rank == 0:
        print(f"Starting training for {num_iterations} iterations...")
        if start_iteration > 0:
            print(f"Resuming from iteration {start_iteration}")
        print(f"Model: {type(model).__name__}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Save directory: {save_dir}")
    
    print(f"[Rank {rank}] Setting model to train mode...")
    model.train()
    
    # Only show progress bar on rank 0
    print(f"[Rank {rank}] Creating progress bar...")
    pbar = tqdm(range(start_iteration, num_iterations), desc="Training", disable=(rank != 0), initial=start_iteration, total=num_iterations)
    
    print(f"[Rank {rank}] Starting training loop...")
    
    for iteration in pbar:
        # Get batch from infinite iterator
        data = next(train_iterator)
        data = data.to(device)
        
        # Generate noise (Gaussian noise, NOT a copy of data!)
        noise = torch.randn_like(data)
        
        # Compute loss - access underlying model if wrapped with DDP
        try:
            model_unwrapped = model.module if hasattr(model, 'module') else model
            loss = model_unwrapped.compute_loss(data, noise)
        except NotImplementedError:
            print("Error: compute_loss method not implemented!")
            print("Please implement the compute_loss method in your model class.")
            return
        except Exception as e:
            print(f"Error computing loss: {e}")
            return
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Update progress bar (only on rank 0)
        if rank == 0:
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # Logging and save loss curve (only on rank 0)
        if rank == 0 and (iteration + 1) % log_interval == 0:
            avg_loss = sum(train_losses[-log_interval:]) / min(log_interval, len(train_losses))
            print(f"Iteration {iteration+1}/{num_iterations}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
            
            # Save training loss curve
            try:
                # Original curve with all data points
                plt.figure(figsize=(10, 6))
                plt.plot(train_losses, alpha=0.3, label='Raw Loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Training Loss')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                # Smoothed curve with moving average
                if len(train_losses) > 50:
                    plt.figure(figsize=(10, 6))
                    # Plot raw data with low opacity
                    plt.plot(train_losses, alpha=0.2, color='blue', linewidth=0.5, label='Raw Loss')
                    
                    # Calculate moving average with window size
                    window_size = min(100, max(10, len(train_losses) // 20))
                    import numpy as np
                    losses_array = np.array(train_losses)
                    moving_avg = np.convolve(losses_array, np.ones(window_size)/window_size, mode='valid')
                    
                    # Plot moving average
                    x_smooth = np.arange(window_size-1, len(train_losses))
                    plt.plot(x_smooth, moving_avg, color='red', linewidth=2, label=f'Moving Avg (window={window_size})')
                    
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.title('Training Loss (Smoothed)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(save_dir / "training_curves_smoothed.png", dpi=150, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                print(f"Warning: Could not save training curve: {e}")
        
        # Save checkpoint (only on rank 0)
        if rank == 0 and (iteration + 1) % save_interval == 0:
            # Unwrap DDP model for saving
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint_path = save_dir / f"checkpoint_iter_{iteration+1}.pt"
            save_model(model_to_save, str(checkpoint_path), model_config)
            print(f"\n  Checkpoint saved: {checkpoint_path}")
            print(f"  Config saved: {checkpoint_path.parent / 'model_config.json'}")
        
            # Generate samples
            print("\n  Generating samples...")
            model.eval()
            model_unwrapped = model.module if hasattr(model, 'module') else model
            shape = (4, 3, 64, 64)
            samples = model_unwrapped.sample(shape, 
                                    num_inference_timesteps=20)
            model.train()
            
            # Save samples
            pil_images = tensor_to_pil_image(samples)
            for i, img in enumerate(pil_images):
                img.save(save_dir / f"iter={iteration+1}_sample_{i}.png")
        
        # FID evaluation at intervals (only on rank 0)
        if rank == 0 and fid_interval > 0 and (iteration + 1) % fid_interval == 0:
            print(f"\n{'='*60}")
            print(f"FID Evaluation at iteration {iteration+1}")
            print(f"{'='*60}")
            
            model.eval()
            
            # Unwrap DDP model
            model_to_eval = model.module if hasattr(model, 'module') else model
            
            # Generate samples for FID (NFE=1 and NFE=20)
            for nfe in [1, 2, 4]:
                sample_dir = save_dir / f"fid_samples_iter{iteration+1}_nfe{nfe}"
                sample_dir.mkdir(exist_ok=True, parents=True)
                
                print(f"\nGenerating samples for NFE={nfe}...")
                cmd = f"CUDA_VISIBLE_DEVICES=0 python sampling.py --ckpt_path {checkpoint_path} --save_dir {sample_dir} --nfe_list {nfe}"
                os.system(cmd)
                
                print(f"\nComputing FID for NFE={nfe}...")
                fid_result = os.popen(
                    f"python measure_fid.py --generated_dir {sample_dir} "
                ).read()
                
                # Save FID result
                fid_log_path = save_dir / "fid_results.txt"
                with open(fid_log_path, "a") as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Iteration: {iteration+1}, NFE: {nfe}\n")
                    f.write(fid_result)
                    f.write(f"{'='*60}\n")
                
                print(fid_result)
            
            model.train()
            print(f"{'='*60}\n")
        
        # Synchronize all processes after checkpoint/FID
        if world_size > 1:
            dist.barrier()
                
    # Close progress bar
    if rank == 0:
        pbar.close()
    
    # Save final model (only on rank 0)
    if rank == 0:
        # Unwrap DDP model for saving
        model_to_save = model.module if hasattr(model, 'module') else model
        final_path = save_dir / "final_model.pt"
        try:
            save_model(model_to_save, str(final_path), model_config)
            print(f"Final model saved: {final_path}")
            print(f"Config saved: {final_path.parent / 'model_config.json'}")
        except Exception as e:
            print(f"Error saving final model: {e}")
        
        print(f"\nTraining completed! Results saved to: {save_dir}")
        print("Check the training_curves.png for loss visualization.")


def main_worker(rank, world_size, args):
    """Worker function for DDP training"""
    try:
        print(f"[Rank {rank}] Worker started")
        
        # Initialize DDP and get local_rank
        local_rank = setup_ddp(rank, world_size)
        
        # Set device for this process (use local rank for device)
        device = torch.device(f"cuda:{local_rank}")
        
        if rank == 0:
            print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"Number of GPUs available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Create dataset (using SimpsonsDataModule)
        if rank == 0:
            print("\nLoading dataset...")
        
        print(f"[Rank {rank}] Creating data module...")
        data_module = SimpsonsDataModule(
            batch_size=args.batch_size // world_size,  # Split batch size across GPUs
            num_workers=args.num_workers
        )
        
        # Get the train dataset from data_module
        print(f"[Rank {rank}] Getting dataset...")
        train_dataset = data_module.train_ds
        
        # Create distributed sampler
        print(f"[Rank {rank}] Creating distributed sampler...")
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        
        # Create dataloader with distributed sampler
        print(f"[Rank {rank}] Creating train dataloader...")
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size // world_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        # Create infinite iterator
        print(f"[Rank {rank}] Creating data iterator...")
        train_iterator = get_data_iterator(train_loader)
        
        # Create model
        if rank == 0:
            print("\nCreating model...")
        
        print(f"[Rank {rank}] Building model config...")
        
        # Prepare kwargs for create_custom_model from args
        model_kwargs = {}
        # Pass all custom arguments except training-specific ones
        # Network hyperparameters (ch, ch_mult, attn, num_res_blocks, dropout) are FIXED
        # Students can add their own custom arguments for scheduler/model configuration
        excluded_keys = ['device', 'batch_size', 'num_iterations', 
                        'lr', 'save_dir', 'log_interval', 'save_interval', 'seed']
        for key, value in args.__dict__.items():
            if key not in excluded_keys and value is not None:
                model_kwargs[key] = value

        try:
            # Create model WITHOUT moving to device first
            model = create_custom_model(
                device='cpu',  # Create on CPU first
                **model_kwargs
            )

            
        except NotImplementedError as e:
            print(f"Error: {e}")
            print("Please implement the CustomScheduler and CustomGenerativeModel classes in custom_model.py")
            return
        except Exception as e:
            print(f"Error creating model: {e}")
            return
        
        # Prepare model configuration for reproducibility
        # This will be saved together with each checkpoint
        model_config = {
            'model_type': type(model).__name__,
            'scheduler_type': type(model.scheduler).__name__,
            **model_kwargs  # Include all custom model arguments
        }

        
        print(f"[Rank {rank}] Creating model...")
        model = create_custom_model(**model_config)
        
        print(f"[Rank {rank}] Moving model to device {device}...")
        model = model.to(device)
        
        # Wrap model with DDP
        print(f"[Rank {rank}] Wrapping model with DDP...")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        print(f"[Rank {rank}] Model wrapped successfully!")
        
        if rank == 0:
            # Count parameters (only on rank 0)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
        
        # Extract start iteration from checkpoint filename if resuming
        start_iteration = 0
        if args.resume_from is not None:
            import re
            match = re.search(r'iter_(\d+)', str(args.resume_from))
            if match:
                start_iteration = int(match.group(1))
            if rank == 0:
                print(f"Resuming from iteration {start_iteration}")
        
        # Train model
        train_model(
            model=model,
            train_iterator=train_iterator,
            num_iterations=args.num_iterations,
            lr=args.lr,
            save_dir=args.save_dir,
            device=device,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            fid_interval=args.fid_interval,
            model_config=model_config,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_weight_decay=args.adam_weight_decay,
            max_grad_norm=args.max_grad_norm,
            rank=rank,
            world_size=world_size,
            start_iteration=start_iteration,
            resume_checkpoint_path=args.resume_from
        )
        
        # Cleanup DDP
        cleanup_ddp()
    
    except Exception as e:
        print(f"[Rank {rank}] FATAL ERROR in main_worker: {e}")
        import traceback
        traceback.print_exc()
        raise


def main(args):
    # Set seed for reproducibility
    seed_everything(args.seed)
    
    # Add timestamp if save directory is not specified
    if args.save_dir == "./results":
        args.save_dir = f"./results/{get_current_time()}"
    
    # Check if using DDP (launched with torchrun)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # DDP mode - launched with torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        if rank == 0:
            print(f"Running in DDP mode with {world_size} GPUs")
        
        main_worker(rank, world_size, args)
    else:
        # Single GPU mode
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"Running in single GPU mode")
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"Available GPUs: {torch.cuda.device_count()}")
        
        # Create dataset (using SimpsonsDataModule)
        print("\nLoading dataset...")
        
        data_module = SimpsonsDataModule(
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        train_loader = data_module.train_dataloader()
        train_iterator = get_data_iterator(train_loader)

        # Prepare kwargs for create_custom_model from args
        model_kwargs = {}
        # Pass all custom arguments except training-specific ones
        # Network hyperparameters (ch, ch_mult, attn, num_res_blocks, dropout) are FIXED
        # Students can add their own custom arguments for scheduler/model configuration
        excluded_keys = ['device', 'batch_size', 'num_iterations', 
                        'lr', 'save_dir', 'log_interval', 'save_interval', 'seed']
        for key, value in args.__dict__.items():
            if key not in excluded_keys and value is not None:
                model_kwargs[key] = value

        try:
            # Create model WITHOUT moving to device first
            model = create_custom_model(
                device='cpu',  # Create on CPU first
                **model_kwargs
            )
        except NotImplementedError as e:
            print(f"Error: {e}")
            print("Please implement the CustomScheduler and CustomGenerativeModel classes in custom_model.py")
            return
        except Exception as e:
            print(f"Error creating model: {e}")
            return

        # Create model
        print("\nCreating model...")
        
        # Prepare model configuration for reproducibility
        # This will be saved together with each checkpoint
        model_config = {
            'model_type': type(model).__name__,
            'scheduler_type': type(model.scheduler).__name__,
            **model_kwargs  # Include all custom model arguments
        }
        
        model = create_custom_model(**model_config)
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Train model (single GPU mode: rank=0, world_size=1)
        train_model(
            model=model,
            train_iterator=train_iterator,
            num_iterations=args.num_iterations,
            lr=args.lr,
            save_dir=args.save_dir,
            device=device,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            fid_interval=args.fid_interval,
            model_config=model_config,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_weight_decay=args.adam_weight_decay,
            max_grad_norm=args.max_grad_norm,
            rank=0,
            world_size=1
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train generative model with DDP support")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Total batch size for training (will be split across GPUs)")
    parser.add_argument("--num_iterations", type=int, default=250000,
                       help="Number of training iterations (default: 250k)")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./results",
                       help="Directory to save")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run training on")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Interval for logging")
    parser.add_argument("--save_interval", type=int, default=10000,
                       help="Interval for saving checkpoints and samples")
    parser.add_argument("--fid_interval", type=int, default=50000,
                       help="Interval for FID evaluation (0 to disable)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from (e.g., ./results/exp/checkpoint_iter_10000.pt)")
    
    # Optimizer parameters
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                       help="Adam beta1 parameter")
    parser.add_argument("--adam_beta2", type=float, default=0.95,
                       help="Adam beta2 parameter")
    parser.add_argument("--adam_weight_decay", type=float, default=0.0,
                       help="Adam weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm for clipping (0 to disable)")

    # MeanFlow specific arguments
    parser.add_argument("--use_additional_condition", action="store_true",
                       help="Use additional condition embedding in U-Net (e.g., step size for Shortcut Models or end timestep for Consistency Trajectory Models)")
    parser.add_argument("--weighting", type=str, default="uniform", choices=["uniform", "adaptive"],
                       help="Loss weighting strategy: uniform or adaptive")
    parser.add_argument("--adaptive_p", type=float, default=0.75,
                       help="Power parameter for adaptive weighting")
    parser.add_argument("--time_mu", type=float, default=-2.0,
                       help="Mean parameter for logit_normal time sampler")
    parser.add_argument("--time_sigma", type=float, default=2.0,
                       help="Std parameter for logit_normal time sampler")
    
    args = parser.parse_args()

    main(args)