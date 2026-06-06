from __future__ import annotations

import math
import os
from typing import Dict, Tuple

import torch

_DIFF_ASSERT_INDICES = os.environ.get("DIFF_ASSERT_INDICES", "0") == "1"


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Generate beta schedule using cosine schedule from Ho et al. (2020).

    Args:
        num_timesteps: Number of diffusion timesteps (T)
        s: Small offset to prevent beta from being too small near t=0

    Returns:
        Beta values for each timestep, shape (T,)
    """
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize to start at 1
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)


def compute_alphas(betas: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute alpha values and related quantities from beta schedule.

    Args:
        betas: Beta values for each timestep, shape (T,)

    Returns:
        Dictionary containing:
        - alphas: Alpha values (1 - beta), shape (T,)
        - alphas_cumprod: Cumulative product of alphas, shape (T,)
        - alphas_cumprod_prev: Previous cumulative alphas (shifted), shape (T,)
        - sqrt_alphas_cumprod: Square root of cumulative alphas, shape (T,)
        - sqrt_one_minus_alphas_cumprod: Square root of (1 - cumulative alphas), shape (T,)
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # Previous cumulative alphas (shifted by 1, with 1.0 at t=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=betas.device), alphas_cumprod[:-1]], dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    return {
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }


def q_sample(
    y0: torch.Tensor, t: torch.Tensor, sqrt_alphas_cumprod: torch.Tensor, sqrt_one_minus_alphas_cumprod: torch.Tensor, noise: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Sample from q(x_t | x_0) - the forward diffusion process.

    This implements: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise

    Args:
        y0: Clean data x_0, shape (batch_size, ...)
        t: Timestep indices, shape (batch_size,)
        sqrt_alphas_cumprod: Square root of cumulative alphas, shape (T,)
        sqrt_one_minus_alphas_cumprod: Square root of (1 - cumulative alphas), shape (T,)
        noise: Optional noise tensor. If None, sampled from N(0, I)

    Returns:
        Noisy data x_t, same shape as y0
    """
    if noise is None:
        noise = torch.randn_like(y0)

    # Extract values for current timesteps and broadcast to match y0 shape
    sqrt_alpha_t = extract(sqrt_alphas_cumprod, t, y0.shape)
    sqrt_one_minus_alpha_t = extract(sqrt_one_minus_alphas_cumprod, t, y0.shape)

    return sqrt_alpha_t * y0 + sqrt_one_minus_alpha_t * noise


def predict_start_from_noise(
    y_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, sqrt_alphas_cumprod: torch.Tensor, sqrt_one_minus_alphas_cumprod: torch.Tensor
) -> torch.Tensor:
    """
    Predict x_0 from x_t and predicted noise using the reparameterization trick.

    This implements: x_0 = (x_t - sqrt(1 - alpha_cumprod_t) * noise) / sqrt(alpha_cumprod_t)

    Args:
        y_t: Noisy data x_t, shape (batch_size, ...)
        t: Timestep indices, shape (batch_size,)
        noise: Predicted noise, same shape as y_t
        sqrt_alphas_cumprod: Square root of cumulative alphas, shape (T,)
        sqrt_one_minus_alphas_cumprod: Square root of (1 - cumulative alphas), shape (T,)

    Returns:
        Predicted clean data x_0, same shape as y_t
    """
    sqrt_one_minus_alpha_t = extract(sqrt_one_minus_alphas_cumprod, t, y_t.shape)
    sqrt_alpha_t = extract(sqrt_alphas_cumprod, t, y_t.shape)

    return (y_t - sqrt_one_minus_alpha_t * noise) / sqrt_alpha_t


def p_mean_variance(
    eps_pred: torch.Tensor, y_t: torch.Tensor, t: torch.Tensor, alphas: torch.Tensor, alphas_cumprod: torch.Tensor, alphas_cumprod_prev: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the mean and variance of the reverse process p(x_{t-1} | x_t).

    This implements the DDPM reverse process using the predicted noise.

    Args:
        eps_pred: Predicted noise, same shape as y_t
        y_t: Noisy data x_t, shape (batch_size, ...)
        t: Timestep indices, shape (batch_size,)
        alphas: Alpha values (1 - beta), shape (T,)
        alphas_cumprod: Cumulative product of alphas, shape (T,)
        alphas_cumprod_prev: Previous cumulative alphas (shifted), shape (T,)

    Returns:
        Tuple of (mean, variance, log_variance) for p(x_{t-1} | x_t)
    """
    # Extract values for current timesteps
    beta_t = extract(1 - alphas, t, y_t.shape)
    sqrt_alphas_cumprod_t = extract(torch.sqrt(alphas_cumprod), t, y_t.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(torch.sqrt(1.0 - alphas_cumprod), t, y_t.shape)

    # Predict x_0 from x_t and predicted noise
    y0_pred = (y_t - sqrt_one_minus_alphas_cumprod_t * eps_pred) / sqrt_alphas_cumprod_t

    # Compute coefficients for mean calculation
    alphas_cumprod_t = extract(alphas_cumprod, t, y_t.shape)
    alphas_cumprod_prev_t = extract(alphas_cumprod_prev, t, y_t.shape)
    alphas_t = extract(alphas, t, y_t.shape)

    coef1 = extract(torch.sqrt(alphas_cumprod_prev), t, y_t.shape) * beta_t / (1 - alphas_cumprod_t)
    coef2 = torch.sqrt(alphas_t) * (1 - alphas_cumprod_prev_t) / (1 - alphas_cumprod_t)

    # Compute mean and variance
    mean = coef1 * y0_pred + coef2 * y_t
    var = beta_t * (1.0 - alphas_cumprod_prev_t) / (1.0 - alphas_cumprod_t)
    log_var = torch.log(var.clamp(min=1e-20))

    return mean, var, log_var


def ddim_sample(
    model,
    y_shape: Tuple[int, ...],
    num_steps: int,
    num_timesteps: int,
    cond: torch.Tensor,
    betas: torch.Tensor,
    device: torch.device,
    predict_type: str = "v",
    ctx_tokens: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    DDIM sampling algorithm for faster inference.

    This implements deterministic sampling with fewer steps than training.

    Args:
        model: The diffusion model that predicts noise
        y_shape: Shape of the output tensor (batch_size, ...)
        num_steps: Number of sampling steps (typically much less than num_timesteps)
        num_timesteps: Total number of training timesteps (T)
        cond: Conditioning information
        betas: Beta schedule from training, shape (T,)
        device: Device to run on

    Returns:
        Generated samples, shape y_shape
    """
    with torch.no_grad():
        # Start from pure noise (allow explicit generator for deterministic parity)
        y_t = torch.randn(y_shape, device=device, generator=generator)

        # Compute alpha values
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, dim=0)
        sqrt_ab = torch.sqrt(ab)
        sqrt_1m = torch.sqrt(1.0 - ab)

        # Create timestep sequence for sampling (reverse order)
        timestep_seq = torch.linspace(num_timesteps - 1, 0, num_steps, device=device).long()

        for i, t in enumerate(timestep_seq):
            # Create batch of timesteps
            t_batch = torch.full((y_shape[0],), int(t.item()), device=device, dtype=torch.long)

            # Model prediction (v or eps)
            if ctx_tokens is not None:
                # Broadcast ctx_tokens to batch if needed
                if ctx_tokens.dim() == 2:
                    ctx_tokens_b = ctx_tokens.unsqueeze(0).expand(y_shape[0], -1, -1).contiguous()
                elif ctx_tokens.shape[0] != y_shape[0]:
                    # Tile or slice to match batch size
                    if ctx_tokens.shape[0] == 1:
                        ctx_tokens_b = ctx_tokens.expand(y_shape[0], -1, -1).contiguous()
                    else:
                        ctx_tokens_b = ctx_tokens[: y_shape[0]]
                else:
                    ctx_tokens_b = ctx_tokens
                pred = model(y_t, t_batch, cond, ctx_tokens_b)
            else:
                pred = model(y_t, t_batch, cond)

            # Predict x_0 from current x_t and pred
            if predict_type == "v":
                y0_pred = extract(sqrt_ab, t_batch, y_t.shape) * y_t - extract(sqrt_1m, t_batch, y_t.shape) * pred
                eps = (y_t - extract(sqrt_ab, t_batch, y_t.shape) * y0_pred) / extract(sqrt_1m, t_batch, y_t.shape)
            else:
                eps = pred
                y0_pred = predict_start_from_noise(y_t, t_batch, eps, sqrt_ab, sqrt_1m)

            if i == num_steps - 1:
                # Final step: return predicted x_0
                y_t = y0_pred
            else:
                # DDIM step: deterministic update
                t_next = timestep_seq[i + 1]
                alpha_next = ab[t_next]

                # DDIM update formula (deterministic, sigma=0)
                y_t = torch.sqrt(alpha_next) * y0_pred + torch.sqrt(1 - alpha_next) * eps

        return y_t


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Extract values from tensor a at timesteps t and broadcast to match x_shape.

    This is a utility function for diffusion processes to get the right values
    for each timestep and broadcast them to match the data shape.

    Args:
        a: Tensor to extract from, shape (T, ...)
        t: Timestep indices, shape (batch_size,)
        x_shape: Target shape to broadcast to

    Returns:
        Extracted and broadcasted tensor, shape (batch_size, 1, 1, ...) matching x_shape
    """
    if _DIFF_ASSERT_INDICES:
        if t.dtype not in (torch.int64, torch.int32):
            raise RuntimeError(f"diffusion.extract: expected integer timesteps, got t.dtype={t.dtype}")
        if t.numel() > 0:
            t_min = int(t.min().item())
            t_max = int(t.max().item())
            size = int(a.shape[0])
            if t_min < 0 or t_max >= size:
                raise RuntimeError(f"diffusion.extract: timestep index out of bounds (min={t_min} max={t_max} size={size})")

    out = a.gather(0, t)  # Extract values at timesteps t
    # Add dimensions to match x_shape for broadcasting
    while out.dim() < len(x_shape):
        out = out.unsqueeze(-1)
    return out
