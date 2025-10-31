#!/usr/bin/env python3
"""
MeanFlow implementation - Minimal version based on MeanFlow paper
Single-step flow matching with bootstrap mechanism
"""

import torch
import torch.nn as nn
import numpy as np
from torch.func import jvp
from src.base_model import BaseScheduler, BaseGenerativeModel
from src.network import UNet


# ============================================================================
# MEANFLOW SCHEDULER
# ============================================================================

class MeanFlowScheduler(BaseScheduler):
    """
    MeanFlow Scheduler for flow-based generation with linear interpolation
    """
    
    def __init__(self, num_train_timesteps: int = 1000, **kwargs):
        super().__init__(num_train_timesteps, **kwargs)
        self.path_type = kwargs.get('path_type', 'linear')
        self.time_sampler = kwargs.get('time_sampler', 'logit_normal')
        self.time_mu = kwargs.get('time_mu', -0.4)
        self.time_sigma = kwargs.get('time_sigma', 1.0)
        self.ratio_r_not_equal_t = kwargs.get('ratio_r_not_equal_t', 0.75)
    
    def interpolant(self, t):
        """Define interpolation function for flow matching"""
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t = 1
        else:
            raise NotImplementedError(f"Path type {self.path_type} not implemented")
        
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def sample_timesteps(self, batch_size: int, device: torch.device):
        """
        Sample time steps (r, t) for MeanFlow training
        Returns two timesteps: r (start) and t (end) where t > r
        """
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * self.time_sigma + self.time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        
        # Ensure t > r by sorting
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        
        # Control proportion of r=t samples
        fraction_equal = 1.0 - self.ratio_r_not_equal_t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        r = torch.where(equal_mask, t, r)
        
        return r, t
    
    def forward_process(self, data, noise, t):
        """Apply forward interpolation: z_t = (1-t) * data + t * noise"""
        alpha_t, sigma_t, _, _ = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * data + sigma_t * noise
        return z_t
    
    def reverse_process_step(self, xt, pred, t, t_next):
        """
        Perform one reverse step: z_r = z_t - (t-r) * u(z_t, r, t)
        
        Args:
            xt: Current state at time t
            pred: Predicted velocity u(z_t, r, t)
            t: Current time (scalar or batch)
            t_next: Next time (r, target time)
        """
        if isinstance(t, float):
            time_diff = t - t_next
        else:
            time_diff = (t - t_next).view(-1, 1, 1, 1)
        
        return xt - time_diff * pred
    
    def get_target(self, data, noise, t):
        """
        Get instantaneous velocity v_t for flow matching
        v_t = d_alpha_t * data + d_sigma_t * noise
        """
        _, _, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        v_t = d_alpha_t * data + d_sigma_t * noise
        return v_t


# ============================================================================
# MEANFLOW MODEL
# ============================================================================

class MeanFlowModel(BaseGenerativeModel):
    """
    MeanFlow Generative Model with bootstrap mechanism
    """
    
    def __init__(self, network, scheduler, **kwargs):
        super().__init__(network, scheduler, **kwargs)
        self.weighting = kwargs.get('weighting', 'uniform')
        self.adaptive_p = kwargs.get('adaptive_p', 1.0)
    
    def compute_loss(self, data, noise, **kwargs):
        """
        Compute MeanFlow loss with bootstrap mechanism (Eq. 10 in paper)
        
        Loss = ||u(z_t, r, t) - [v_t - (t-r) * ∂u/∂t]||^2
        """
        batch_size = data.shape[0]
        device = data.device
        
        # Sample time steps (r, t)
        r, t = self.scheduler.sample_timesteps(batch_size, device)
        
        # Get interpolated state z_t
        z_t = self.scheduler.forward_process(data, noise, t)
        
        # Get instantaneous velocity v_t
        v_t = self.scheduler.get_target(data, noise, t)
        time_diff = (t - r).view(-1, 1, 1, 1)
        
        # Predict u(z_t, r, t) using the network
        u = self.predict(z_t, t, r=r)
        
        # Compute JVP: ∂u/∂t using automatic differentiation
        # CRITICAL: Pre-compute time_diff outside JVP to avoid unwanted derivatives
        network = self.network.module if isinstance(self.network, torch.nn.DataParallel) else self.network
        time_diff_for_condition = t - r  # Pre-compute OUTSIDE JVP
        
        def fn_for_jvp(z, cur_t):
            # Use pre-computed time_diff (constant in JVP)
            # This ensures we only compute ∂u/∂t, not ∂u/∂condition
            return network(z, cur_t, condition=time_diff_for_condition)
        
        # JVP w.r.t. z and t only
        primals = (z_t, t)
        tangents = (v_t, torch.ones_like(t))
        _, dudt = jvp(fn_for_jvp, primals, tangents)
        
        # Bootstrap target: v_t - (t-r) * ∂u/∂t
        u_target = v_t - time_diff * dudt
        
        # Compute loss
        error = u - u_target.detach()
        loss_per_sample = torch.sum((error**2).reshape(error.shape[0], -1), dim=-1)
        
        # Apply adaptive weighting if specified
        if self.weighting == "adaptive":
            # print(loss_per_sample)
            # print(self.adaptive_p)
            weights = 1.0 / (loss_per_sample.detach() + 1e-3).pow(self.adaptive_p)
            loss = weights * loss_per_sample
        else:
            loss = loss_per_sample
        
        return loss.mean()
    
    def predict(self, xt, t, **kwargs):
        """
        Make prediction u(z_t, r, t)
        
        Args:
            xt: State at time t
            t: Current time
            r: Start time (from kwargs)
        """
        r = kwargs.get('r', None)
        
        if r is not None:
            # Encode both t and (t-r) as time conditioning
            # Use t-r as the additional condition
            time_diff = t - r
            return self.network(xt, t, condition=time_diff)
        else:
            # Inference mode: r is implicitly 0 or needs to be specified
            return self.network(xt, t)
    
    @torch.no_grad()
    def sample(self, shape, num_inference_timesteps=1, return_traj=False, verbose=False, use_2nd_order=True, **kwargs):
        """
        MeanFlow sampling with optional 2nd order solver
        
        For single-step (NFE=1): x_0 = x_1 - u(x_1, 0, 1)
        For multi-step: 
            - 1st order: z_r = z_t - (t-r) * u(z_t, r, t)
            - 2nd order (Heun): improved accuracy with midpoint correction
        
        Args:
            shape: Shape of samples to generate
            num_inference_timesteps: Number of denoising steps (NFE)
            return_traj: Whether to return the full trajectory
            verbose: Whether to show progress
            use_2nd_order: Use 2nd order Heun method for NFE>=2 (default: True)
        """
        device = self.device
        batch_size = shape[0]
        
        # Start from noise (z_1)
        z = torch.randn(shape, device=device)
        
        if return_traj:
            trajectory = [z.cpu()]
        
        if num_inference_timesteps == 1:
            # Single-step generation (always 1st order)
            r = torch.zeros(batch_size, device=device)
            t = torch.ones(batch_size, device=device)
            
            u = self.predict(z, t, r=r)
            x0 = z - u  # z_0 = z_1 - u(z_1, 0, 1)
            
            if return_traj:
                trajectory.append(x0.cpu())
                return trajectory
            return x0
        else:
            # Multi-step generation
            time_steps = torch.linspace(1, 0, num_inference_timesteps + 1, device=device)
            
            if use_2nd_order:
                # 2nd order Heun method (improved accuracy)
                if verbose:
                    print("Using 2nd order Heun sampler")
                
                for i in range(num_inference_timesteps):
                    t_cur = time_steps[i].item()
                    t_next = time_steps[i + 1].item()
                    dt = t_next - t_cur  # negative
                    
                    t = torch.full((batch_size,), t_cur, device=device)
                    r = torch.full((batch_size,), t_next, device=device)
                    
                    # First stage: Euler step
                    u_cur = self.predict(z, t, r=r)
                    z_temp = z + dt * u_cur  # z at t_next (predictor)
                    
                    # Second stage: Corrector with midpoint
                    t_next_tensor = torch.full((batch_size,), t_next, device=device)
                    # For the second evaluation, we need u at (z_temp, t_next, t_next)
                    # But this is still for going from t_next to somewhere
                    # Actually, we evaluate u at the predicted point
                    u_next = self.predict(z_temp, t_next_tensor, r=r)
                    
                    # Heun's method: average of two slopes
                    z = z + dt * (u_cur + u_next) / 2
                    
                    if return_traj:
                        trajectory.append(z.cpu())
            else:
                # 1st order Euler method
                if verbose:
                    print("Using 1st order Euler sampler")
                
                for i in range(num_inference_timesteps):
                    t_cur = time_steps[i].item()
                    t_next = time_steps[i + 1].item()
                    
                    t = torch.full((batch_size,), t_cur, device=device)
                    r = torch.full((batch_size,), t_next, device=device)
                    
                    u = self.predict(z, t, r=r)
                    z = self.scheduler.reverse_process_step(z, u, t_cur, t_next)
                    
                    if return_traj:
                        trajectory.append(z.cpu())
            
            if return_traj:
                return trajectory
            return z


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_custom_model(device="cpu", **kwargs):
    """
    Create a MeanFlow model
    
    Args:
        device: Device to place model on
        **kwargs: Additional arguments for scheduler and model configuration
    """
    
    # Create U-Net backbone with additional condition for (t-r)
    network = UNet(
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_additional_condition=kwargs.get('use_additional_condition', False)
    )
    
    # Create MeanFlow scheduler - pass all kwargs directly
    scheduler = MeanFlowScheduler(**kwargs)
    
    # Create MeanFlow model
    model = MeanFlowModel(network, scheduler, **kwargs)
    
    return model.to(device)