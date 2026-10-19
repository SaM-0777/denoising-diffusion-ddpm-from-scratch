"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    # TODO: return a linear beta schedule of length T
    t = torch.linspace(beta_start, beta_end, steps=T)
    return t

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    # TODO: return 1 - betas
    aplha_t = 1 - betas
    return aplha_t

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    # TODO: cumulative product of alphas
    return torch.cumprod(alphas, axis=0)

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    # TODO: gather a[t] and reshape to (B, 1, 1, 1) for broadcasting with x
    B, _, _, _ = x.shape
    return a[t].view(B, 1, 1, 1)

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    # TODO: x_t = sqrt(bar_alpha_t) * x0 + sqrt(1 - bar_alpha_t) * noise
    bar_alpha_t = alphas_cumprod[t][:, None, None, None]
    x_t = torch.sqrt(bar_alpha_t) * x0 + torch.sqrt(1 - bar_alpha_t) * noise
    return x_t

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    # TODO: build betas, alphas, alphas_cumprod and useful sqrts
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1 - betas

    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    
    return {
        "T": T,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "betas": betas,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    # TODO: MSE between predicted and true noise
    return F.mse_loss(noise, noise_pred)

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    # TODO: q_sample -> model -> MSE(noise_pred, noise)
    bar_alpha_t = alphas_cumprod[t][:, None, None, None]
    x_t = torch.sqrt(bar_alpha_t) * x0 + torch.sqrt(1 - bar_alpha_t) * noise

    noise_pred = model(x_t, t)

    return F.mse_loss(noise_pred, noise)

# Step 9 - timestep_embedding
import math
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    # TODO: sinusoidal timestep embedding of shape (B, dim)
    half_dim = dim // 2
    max_period = 10000
    freq = torch.exp(
        -torch.log(torch.tensor(max_period, dtype=torch.float32))
        * torch.arange(half_dim, dtype=torch.float32)
        / half_dim
    )
    args = t.reshape(-1, 1).float() * freq
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

# Step 10 - init_tiny_unet
import torch
import torch.nn.functional as F

def init_tiny_unet(in_ch: int = 1, hidden: int = 16, time_dim: int = 16, seed: int = 0) -> dict:
    # TODO: initialize tiny residual denoiser parameters
    torch.manual_seed(seed)
    return {
        "conv_in_w": torch.nn.Parameter(
            torch.randn(hidden, in_ch, 3, 3) * 0.02
        ),
        "conv_in_b": torch.nn.Parameter(torch.zeros(hidden)),
        "time_mlp_w": torch.nn.Parameter(
            torch.randn(hidden, time_dim) * 0.02
        ),
        "time_mlp_b": torch.nn.Parameter(torch.zeros(hidden)),
        "conv_mid_w": torch.nn.Parameter(torch.randn(hidden, hidden, 3, 3) * 0.02),
        "conv_mid_b": torch.nn.Parameter(torch.zeros(hidden)),
        "conv_out_w": torch.nn.Parameter(torch.randn(in_ch, hidden, 3, 3) * 0.02),
        "conv_out_b": torch.nn.Parameter(torch.zeros(in_ch)),
    }

# Step 11 - tiny_unet_forward
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    half_dim = dim // 2
    max_period = 10000
    freq = torch.exp(
        -torch.log(torch.tensor(max_period, dtype=torch.float32, device=t.device))
        * torch.arange(half_dim, dtype=torch.float32, device=t.device)
        / half_dim
    )
    args = t.reshape(-1, 1).float() * freq
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


def tiny_unet_forward(x, t, params: dict):
    conv_in_w = params["conv_in_w"]
    conv_in_b = params["conv_in_b"]
    time_mlp_w = params["time_mlp_w"]
    time_mlp_b = params["time_mlp_b"]
    conv_mid_w = params["conv_mid_w"]
    conv_mid_b = params["conv_mid_b"]
    conv_out_w = params["conv_out_w"]
    conv_out_b = params["conv_out_b"]

    h = F.conv2d(x, conv_in_w, conv_in_b, padding=1)

    temb = timestep_embedding(t, time_mlp_w.shape[1])
    temb = F.relu(F.linear(temb, time_mlp_w, time_mlp_b))

    h = h + temb[:, :, None, None]
    h = F.relu(h)
    h = F.relu(F.conv2d(h, conv_mid_w, conv_mid_b, padding=1))

    return F.conv2d(h, conv_out_w, conv_out_b, padding=1)

# Step 12 - make_blob_dataset (not yet solved)
# TODO: implement

# Step 13 - ddpm_train_step (not yet solved)
# TODO: implement

# Step 14 - train_ddpm (not yet solved)
# TODO: implement

# Step 15 - predict_x0_from_eps (not yet solved)
# TODO: implement

# Step 16 - ddpm_p_mean_variance (not yet solved)
# TODO: implement

# Step 17 - ddpm_p_sample (not yet solved)
# TODO: implement

# Step 18 - ddpm_sample_loop (not yet solved)
# TODO: implement

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

