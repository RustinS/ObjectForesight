from __future__ import annotations

from typing import Tuple

import torch

from .diffusion import extract


@torch.no_grad()
def ar_rollout_sample(
    poser_model,
    cond_embed: torch.Tensor,
    H: int,
    ctx_tokens_9d: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Autoregressive rollout sampling for AR transformer models.
    Returns y0 tokens with shape (B, H, 9).
    """
    device = cond_embed.device

    # Use the temporal model's rollout method
    if hasattr(poser_model, "temporal") and hasattr(poser_model.temporal, "rollout_infer"):
        if ctx_tokens_9d is None:
            raise RuntimeError("AR rollout requires context tokens.")
        ctx = ctx_tokens_9d
        if not isinstance(ctx, torch.Tensor):
            ctx = torch.as_tensor(ctx, device=device, dtype=torch.float32)
        else:
            ctx = ctx.to(device=device, dtype=torch.float32)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0)

        ctx_n = ctx
        if hasattr(poser_model, "_perdim_normalize"):
            try:
                ctx_n = poser_model._perdim_normalize(ctx)
            except Exception:
                ctx_n = ctx

        y_pred_n = poser_model.temporal.rollout_infer(H, cond_embed, ctx_tokens=ctx_n)
        if hasattr(poser_model, "_perdim_unnormalize"):
            try:
                return poser_model._perdim_unnormalize(y_pred_n)
            except Exception:
                return y_pred_n
        return y_pred_n

    # Fallback: step-by-step rollout
    if ctx_tokens_9d is None:
        raise RuntimeError("AR rollout requires context tokens when rollout_infer is not available")

    ctx = ctx_tokens_9d
    if not isinstance(ctx, torch.Tensor):
        ctx = torch.as_tensor(ctx, device=device, dtype=torch.float32)
    if ctx.dim() == 2:
        ctx = ctx.unsqueeze(0)

    ctx_n = ctx
    if hasattr(poser_model, "_perdim_normalize"):
        try:
            ctx_n = poser_model._perdim_normalize(ctx)
        except Exception:
            ctx_n = ctx

    y_prev = ctx_n[:, -1, :]
    preds = []
    for _ in range(H):
        y_next = poser_model.temporal.forward_step(y_prev=y_prev, cond_embed=cond_embed, ctx_tokens=ctx_n)
        preds.append(y_next)
        y_prev = y_next
    y_pred_n = torch.stack(preds, dim=1)
    if hasattr(poser_model, "_perdim_unnormalize"):
        try:
            return poser_model._perdim_unnormalize(y_pred_n)
        except Exception:
            return y_pred_n
    return y_pred_n


@torch.no_grad()
def _ddim_step(
    model,
    y_t: torch.Tensor,
    t_batch: torch.Tensor,
    cond_embed: torch.Tensor,
    ab: torch.Tensor,
    sqrt_ab: torch.Tensor,
    sqrt_1m: torch.Tensor,
    predict_type: str,
    eta: float,
    generator: torch.Generator | None,
    ctx_tokens: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform one DDIM step and return (y_t_next, y0_pred).
    """
    if ctx_tokens is not None:
        # Broadcast/align ctx_tokens to batch if needed
        if ctx_tokens.dim() == 2:
            ctx_tokens_b = ctx_tokens.unsqueeze(0).expand(y_t.shape[0], -1, -1).contiguous()
        elif ctx_tokens.shape[0] != y_t.shape[0]:
            if ctx_tokens.shape[0] == 1:
                ctx_tokens_b = ctx_tokens.expand(y_t.shape[0], -1, -1).contiguous()
            else:
                ctx_tokens_b = ctx_tokens[: y_t.shape[0]]
        else:
            ctx_tokens_b = ctx_tokens
        pred = model(y_t, t_batch, cond_embed, ctx_tokens_b)
    else:
        pred = model(y_t, t_batch, cond_embed)

    if predict_type == "v":
        y0_pred = extract(sqrt_ab, t_batch, y_t.shape) * y_t - extract(sqrt_1m, t_batch, y_t.shape) * pred
        eps = (y_t - extract(sqrt_ab, t_batch, y_t.shape) * y0_pred) / extract(sqrt_1m, t_batch, y_t.shape)
    else:
        eps = pred
        y0_pred = (y_t - extract(sqrt_1m, t_batch, y_t.shape) * eps) / extract(sqrt_ab, t_batch, y_t.shape)

    return eps, y0_pred


@torch.no_grad()
def ddim_sample_tokens(
    poser_model,
    cond_embed: torch.Tensor,
    H: int,
    steps: int,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
    ctx_tokens_9d: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Deterministic DDIM sampling that mirrors infer_main PoserV1.sample for DiT.
    Returns y0 tokens with shape (B, H, 9) in the model's tokenization (se3(t+rot6d)).
    """
    device = cond_embed.device
    batch_size = int(cond_embed.shape[0])

    # Route by temporal backend kind
    kind = getattr(poser_model, "_temporal_kind", "dit")
    if kind == "ar_transformer":
        return ar_rollout_sample(poser_model=poser_model, cond_embed=cond_embed, H=H, ctx_tokens_9d=ctx_tokens_9d)

    # Prepare optional context tokens in model's normalized depth-parameterized space
    ctx_dn_n = None
    if ctx_tokens_9d is not None:
        try:
            ctx9 = ctx_tokens_9d
            if not isinstance(ctx9, torch.Tensor):
                ctx9 = torch.as_tensor(ctx9, dtype=torch.float32, device=device)
            else:
                ctx9 = ctx9.to(device=device, dtype=torch.float32)
            if ctx9.dim() == 2:
                ctx9 = ctx9.unsqueeze(0)
            ctx_dn = poser_model._tok_to_dnorm(ctx9)
            ctx_dn_n = (ctx_dn - poser_model._dnorm_means) / poser_model._dnorm_scales
        except Exception:
            ctx_dn_n = None

    # Schedules/buffers from trained model (parity with training)
    T_train = int(poser_model._T_train.item())
    betas = poser_model.betas.to(device=device)
    alphas = 1.0 - betas
    ab = torch.cumprod(alphas, dim=0)  # (T,)
    sqrt_ab = torch.sqrt(ab)
    sqrt_1m = torch.sqrt(1.0 - ab)

    # DDIM timestep schedule: linearly spaced indices (T-1 → 0)
    timestep_seq = torch.linspace(T_train - 1, 0, int(steps), device=device).long()

    # Start from noise
    y_shape = (batch_size, int(H) * 9)
    if eta != 0.0 and generator is None:
        # still deterministic with global RNG if user seeds outside; prefer explicit generator when possible
        pass
    y_t = torch.randn(y_shape, device=device, generator=generator)

    for i, t in enumerate(timestep_seq):
        t_batch = torch.full((batch_size,), int(t.item()), device=device, dtype=torch.long)
        eps, y0_pred = _ddim_step(
            poser_model.dit,
            y_t,
            t_batch,
            cond_embed,
            ab,
            sqrt_ab,
            sqrt_1m,
            predict_type="v",
            eta=eta,
            generator=generator,
            ctx_tokens=ctx_dn_n,
        )

        if i == len(timestep_seq) - 1:
            # final: output y0 prediction
            y_t = y0_pred
            break

        t_next = timestep_seq[i + 1]
        alpha_t = ab[t]
        alpha_next = ab[t_next]

        if eta == 0.0:
            # deterministic DDIM
            y_t = torch.sqrt(alpha_next) * y0_pred + torch.sqrt(1.0 - alpha_next) * eps
        else:
            # stochastic DDIM with eta
            sigma = eta * torch.sqrt((1.0 - alpha_next) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_next))
            noise = torch.randn_like(y_t, generator=generator)
            c = torch.sqrt(torch.clamp(1.0 - alpha_next - sigma * sigma, min=0.0))
            y_t = torch.sqrt(alpha_next) * y0_pred + c * eps + sigma * noise

    # Map from normalized depth-parameterization back to tokens
    y0_dn_n = y_t.view(batch_size, int(H), 9)
    y0_dn = poser_model._dnorm_unnormalize(y0_dn_n, poser_model._dnorm_scales)
    # round-trip safety check (same as model.sample)
    _ = poser_model._tok_to_dnorm(poser_model._dnorm_to_tok(y0_dn))
    y0 = poser_model._dnorm_to_tok(y0_dn)
    return y0
