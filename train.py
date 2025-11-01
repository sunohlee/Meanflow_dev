#!/usr/bin/env python3
"""
Skeleton training script for generative models
Students need to implement their own model classes and modify this script accordingly.
"""

# --------------------------------------------------------------------------------
# Parts of this file were modified based on the code from the MeanFlow project.
# Original Repository: https://github.com/zhuyu-cs/MeanFlow
#
# MeanFlow: Recovering General 3D Clothed Human Shape and Motion from Monocular Videos
# Yu Zhu, Siyou Peng, Yuxiang Wang, Yipeng Yang, C. L. Philip Lei, Yujun Yang
# CVPR 2024
# --------------------------------------------------------------------------------

import json
import argparse
import torch
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

from src.network import UNet
from dataset import SimpsonsDataModule, get_data_iterator
from src.utils import tensor_to_pil_image, get_current_time, save_model, seed_everything
from src.base_model import BaseScheduler, BaseGenerativeModel
from custom_model import create_custom_model


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
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Create optimizer with MeanFlow parameters
    optimizer = optim.Adam(
        model.network.parameters(),
        lr=lr,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=1e-08
    )
    
    # Save training configuration
    config = {
        'num_iterations': num_iterations,
        'lr': lr,
        'log_interval': log_interval,
        'save_interval': save_interval,
        'device': str(device),
    }
    with open(save_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to: {save_dir / 'training_config.json'}")
    
    # Training loop
    train_losses = []
    
    print(f"Starting training for {num_iterations} iterations...")
    if start_iteration > 0:
        print(f"Resuming from iteration {start_iteration}")
    print(f"Model: {type(model).__name__}")
    print(f"Device: {device}")
    print(f"Save directory: {save_dir}")
    
    # Run FID evaluation if resuming
    if start_iteration > 0 and resume_checkpoint_path is not None:
        print(f"\n{'='*60}")
        print(f"Running FID evaluation before resuming training")
        print(f"Checkpoint: {resume_checkpoint_path}")
        print(f"{'='*60}")
        
        model.eval()
        
        # Generate samples for FID (NFE=1, 2, 4)
        for nfe in [1, 2, 4]:
            sample_dir = save_dir / f"fid_samples_iter{start_iteration}_nfe{nfe}"
            sample_dir.mkdir(exist_ok=True, parents=True)
            
            print(f"\nGenerating samples for NFE={nfe}...")
            cmd = f"python sampling.py --ckpt_path {resume_checkpoint_path} --save_dir {sample_dir} --num_samples 1000 --batch_size 32 --nfe_list {nfe}"
            print(f"Running: {cmd}")
            os.system(cmd)
            
            print(f"\nComputing FID for NFE={nfe}...")
            fid_result = os.popen(
                f"python measure_fid.py --generated_dir {sample_dir}"
            ).read()
            
            # Save FID result
            fid_log_path = save_dir / "fid_results.txt"
            with open(fid_log_path, "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Iteration: {start_iteration}, NFE: {nfe}\n")
                f.write(fid_result)
                f.write(f"{'='*60}\n")
            
            print(fid_result)
        
        print(f"{'='*60}\n")
        print("FID evaluation completed! Resuming training...\n")
    
    model.train()
    
    pbar = tqdm(range(start_iteration, num_iterations), desc="Training")
    
    for iteration in pbar:
        # Get batch from infinite iterator
        data = next(train_iterator)
        data = data.to(device)
        
        # Generate noise (Gaussian noise, NOT a copy of data!)
        noise = torch.randn_like(data)
        
        # Compute loss
        try:
            loss = model.compute_loss(data, noise)
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
            torch.nn.utils.clip_grad_norm_(model.network.parameters(), max_grad_norm)
        
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Update progress bar
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # Logging and save loss curve
        if (iteration + 1) % log_interval == 0:
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
        
        if (iteration + 1) % save_interval == 0:
            # Save checkpoint
            checkpoint_path = save_dir / f"checkpoint_iter_{iteration+1}.pt"
            save_model(model, str(checkpoint_path), model_config)
            print(f"\n  Checkpoint saved: {checkpoint_path}")
            print(f"  Config saved: {checkpoint_path.parent / 'model_config.json'}")
        
            # Generate samples
            print("\n  Generating samples...")
            model.eval()
            shape = (4, 3, 64, 64)
            samples = model.sample(shape, 
                                    num_inference_timesteps=20)
            model.train()
            
            # Save samples
            pil_images = tensor_to_pil_image(samples)
            for i, img in enumerate(pil_images):
                img.save(save_dir / f"iter={iteration+1}_sample_{i}.png")
        
        # FID evaluation at intervals
        if fid_interval > 0 and (iteration + 1) % fid_interval == 0:
            print(f"\n{'='*60}")
            print(f"FID Evaluation at iteration {iteration+1}")
            print(f"{'='*60}")
            
            model.eval()
            
            # Generate samples for FID (NFE=1 and NFE=20)
            for nfe in [1, 2, 4]:
                sample_dir = save_dir / f"fid_samples_iter{iteration+1}_nfe{nfe}"
                sample_dir.mkdir(exist_ok=True, parents=True)
                
                print(f"\nGenerating samples for NFE={nfe}...")
                cmd = f"python sampling.py --ckpt_path {checkpoint_path} --save_dir {sample_dir} --nfe_list {nfe}"
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
                
    # Save final model
    final_path = save_dir / "final_model.pt"
    try:
        save_model(model, str(final_path), model_config)
        print(f"Final model saved: {final_path}")
        print(f"Config saved: {final_path.parent / 'model_config.json'}")
    except Exception as e:
        print(f"Error saving final model: {e}")
    
    print(f"\nTraining completed! Results saved to: {save_dir}")
    print("Check the training_curves.png for loss visualization.")


def main(args):
    # Set seed for reproducibility
    seed_everything(args.seed)
    print(f"Seed set to: {args.seed}")
    
    # Set CUDA devices if specified
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"CUDA_VISIBLE_DEVICES set to: {args.cuda_visible_devices}")
    
    # Add timestamp if save directory is not specified
    if args.save_dir == "./results":
        args.save_dir = f"./results/{get_current_time()}"
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Create model
    print("Creating model...")

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
        print(f"Model created: {type(model).__name__}")
        print(f"Scheduler: {type(model.scheduler).__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.network.parameters()):,}")
        
        # Use DataParallel for multi-GPU if available
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            num_gpus = torch.cuda.device_count()
            print(f"\n=== Multi-GPU Setup with DataParallel ===")
            print(f"Number of GPUs: {num_gpus}")
            print(f"Batch size per GPU: {args.batch_size // num_gpus}")
            print(f"Total effective batch size: {args.batch_size}")
            
            # Move model to GPU first
            model = model.to(device)
            # Then wrap network in DataParallel
            model.network = torch.nn.DataParallel(model.network)
            print(f"DataParallel enabled on GPUs: {list(range(num_gpus))}")
        else:
            print(f"\nUsing single GPU/CPU")
            model = model.to(device)
        
        # Load checkpoint if resuming
        start_iteration = 0
        if args.resume_from is not None:
            print(f"\nLoading checkpoint from: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # # Check if state_dict has 'network.' prefix (full model) or not (network only)
            # first_key = list(state_dict.keys())[0] if state_dict else ""
            
            # if first_key.startswith('network.'):
            #     # Full model state dict - load to model
            #     if hasattr(model.network, 'module'):
            #         # DataParallel case: need to handle prefix
            #         # Remove 'network.' prefix and load to module
            #         network_state = {k.replace('network.', ''): v for k, v in state_dict.items() if k.startswith('network.')}
            #         model.network.module.load_state_dict(network_state)
            #     else:
            #         # Load full model state dict
            #         model.load_state_dict(state_dict)
            # else:
            #     # Network only state dict
            #     if hasattr(model.network, 'module'):
            #         model.network.module.load_state_dict(state_dict)
            #     else:
            #         model.network.load_state_dict(state_dict)
            # Simply load the full model state dict
            model.load_state_dict(state_dict)
            
            # Try to get iteration number from checkpoint
            if 'iteration' in checkpoint:
                start_iteration = checkpoint['iteration']
            else:
                # Extract from filename: checkpoint_iter_20000.pt
                import re
                match = re.search(r'iter_(\d+)', str(args.resume_from))
                if match:
                    start_iteration = int(match.group(1))
            
            print(f"Checkpoint loaded! Resuming from iteration {start_iteration}")
            
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
    
    # Load dataset
    print("Loading dataset...")
    try:
        data_module = SimpsonsDataModule(
            batch_size=args.batch_size,
            num_workers=4
        )
        
        train_loader = data_module.train_dataloader()
        train_iterator = get_data_iterator(train_loader)
        
        print(f"Total iterations: {args.num_iterations}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
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
        start_iteration=start_iteration,
        resume_checkpoint_path=args.resume_from
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train generative model")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training (will be split across GPUs)")
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
    parser.add_argument("--cuda_visible_devices", type=str, default=None,
                       help="Comma-separated GPU IDs to use (e.g., '0,1' for GPU 0 and 1)")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume training from (e.g., results/xxx/checkpoint_iter_20000.pt)")
    
    # Optimizer parameters
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                       help="Adam beta1 parameter")
    parser.add_argument("--adam_beta2", type=float, default=0.95,
                       help="Adam beta2 parameter")
    parser.add_argument("--adam_weight_decay", type=float, default=0.0,
                       help="Adam weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm for clipping (0 to disable)")

    # Model-specific arguments (students can add more)
    # DO NOT MODIFY THE PROVIDED NETWORK HYPERPARAMETERS 
    # (ch=128, ch_mult=[1,2,2,2], attn=[1], num_res_blocks=4, dropout=0.1)
    parser.add_argument("--use_additional_condition", action="store_true",
                       help="Use additional condition embedding in U-Net (e.g., step size for Shortcut Models or end timestep for Consistency Trajectory Models)")
    parser.add_argument("--num_train_timesteps", type=int, default=1000,
                       help="Number of training timesteps for scheduler")
    
    # MeanFlow specific arguments
    parser.add_argument("--weighting", type=str, default="uniform", choices=["uniform", "adaptive"],
                       help="Loss weighting strategy: uniform or adaptive")
    parser.add_argument("--adaptive_p", type=float, default=1.0,
                       help="Power parameter for adaptive weighting (only used when weighting=adaptive)")
    parser.add_argument("--time_sampler", type=str, default="logit_normal", choices=["uniform", "logit_normal"],
                       help="Time sampling strategy")
    parser.add_argument("--time_mu", type=float, default=-0.4,
                       help="Mean parameter for logit_normal time sampler")
    parser.add_argument("--time_sigma", type=float, default=1.0,
                       help="Std parameter for logit_normal time sampler")
    parser.add_argument("--ratio_r_not_equal_t", type=float, default=0.75,
                       help="Ratio of samples where r!=t (bootstrap ratio)")
    
    args = parser.parse_args()

    main(args)