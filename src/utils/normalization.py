"""Centralized token normalization logic.

This module provides a single source of truth for 9D token normalization.
9D tokens: [t_x, t_y, t_z, rot6d(6)]

Supports two normalization modes:
- 'perdim': Per-dimension mean/std normalization across all 9 dimensions
- 'dnorm': Depth-based normalization (divide trans by z, then standardize)

Usage:
    # Per-dimension normalization
    normalizer = TokenNormalizer(mode='perdim')
    normalizer.fit_from_batch(tokens)  # tokens: [B, H, 9]
    tokens_norm = normalizer.normalize(tokens)
    tokens_denorm = normalizer.denormalize(tokens_norm)

    # Depth normalization
    normalizer = TokenNormalizer(mode='dnorm', z_min=0.05)
    tokens_dn = normalizer.tok_to_dnorm(tokens)  # [tx,ty,tz,r6] -> [u,v,log(z),r6]
    tokens_back = normalizer.dnorm_to_tok(tokens_dn)

Note:
    This class consolidates normalization logic previously scattered across:
    - src/models/poser_v1/model.py (_tok_to_dnorm, _dnorm_to_tok, _perdim_normalize, etc.)
    - src/utils/validate.py (normalize_pred_to_cam_i0)
    - src/temporal/sampling.py (context token normalization)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist


class TokenNormalizer:
    """Single source of truth for 9D token normalization.

    9D tokens: [t_x, t_y, t_z, rot6d(6)]

    Supports two modes:
    - 'perdim': Per-dimension mean/std normalization
    - 'dnorm': Depth-based normalization (divide trans by z)

    Args:
        mode: Normalization mode ('perdim' or 'dnorm')
        t_mean: Mean for per-dimension normalization [9] or translation mean [3]
        t_std: Std for per-dimension normalization [9] or translation std [3]
        t_scale: Global translation scale (legacy, typically 1.0)
        z_min: Minimum depth threshold for dnorm mode (default: 0.05)
        device: Device for tensors
        dtype: Data type for tensors
    """

    def __init__(
        self,
        mode: str = "perdim",
        t_mean: torch.Tensor | None = None,
        t_std: torch.Tensor | None = None,
        t_scale: float = 1.0,
        z_min: float = 0.05,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if mode not in ("perdim", "dnorm"):
            raise ValueError(f"mode must be 'perdim' or 'dnorm', got {mode}")

        self.mode = mode
        self.z_min = float(z_min)
        self.t_scale = float(t_scale)
        self.device = device
        self.dtype = dtype

        # Initialize mean/std based on mode
        if mode == "perdim":
            # Per-dimension normalization uses 9D mean/std
            if t_mean is None:
                self._means = torch.zeros(9, device=device, dtype=dtype)
            else:
                self._means = self._to_tensor(t_mean, expected_shape=(9,))
            if t_std is None:
                self._stds = torch.ones(9, device=device, dtype=dtype)
            else:
                self._stds = self._to_tensor(t_std, expected_shape=(9,))
        else:
            # Depth normalization uses 9D mean/std for depth-normalized space
            if t_mean is None:
                self._means = torch.zeros(9, device=device, dtype=dtype)
            else:
                self._means = self._to_tensor(t_mean, expected_shape=(9,))
            if t_std is None:
                self._stds = torch.ones(9, device=device, dtype=dtype)
            else:
                self._stds = self._to_tensor(t_std, expected_shape=(9,))

    def _to_tensor(self, data: torch.Tensor | list | tuple, expected_shape: tuple | None = None) -> torch.Tensor:
        """Convert input to tensor with correct device/dtype."""
        if isinstance(data, torch.Tensor):
            t = data.to(device=self.device, dtype=self.dtype)
        else:
            t = torch.tensor(data, device=self.device, dtype=self.dtype)

        if expected_shape is not None and t.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {t.shape}")
        return t

    @property
    def means(self) -> torch.Tensor:
        """Get normalization means."""
        return self._means

    @property
    def stds(self) -> torch.Tensor:
        """Get normalization stds/scales."""
        return self._stds

    def update_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Update normalization statistics.

        Args:
            mean: New mean tensor [9]
            std: New std tensor [9]
        """
        self._means = self._to_tensor(mean, expected_shape=(9,))
        self._stds = self._to_tensor(std, expected_shape=(9,))

    def tok_to_dnorm(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert standard tokens [tx,ty,tz,r6] to depth-normalized [u,v,log(z),r6].

        Args:
            tokens: Input tokens [..., 9] with [tx, ty, tz, rot6d(6)]

        Returns:
            Depth-normalized tokens [..., 9] with [u, v, log(z), rot6d(6)]
        """
        if tokens.shape[-1] != 9:
            raise ValueError(f"Expected tokens with shape [..., 9], got {tokens.shape}")

        t, r6 = tokens[..., :3], tokens[..., 3:]
        z = t[..., 2].clamp_min(self.z_min)
        u = t[..., 0] / z.clamp_min(1e-6)
        v = t[..., 1] / z.clamp_min(1e-6)
        s = torch.log(z)

        return torch.cat([u[..., None], v[..., None], s[..., None], r6], dim=-1)

    def dnorm_to_tok(self, tokens_dn: torch.Tensor) -> torch.Tensor:
        """Convert depth-normalized [u,v,log(z),r6] back to standard [tx,ty,tz,r6].

        Args:
            tokens_dn: Depth-normalized tokens [..., 9] with [u, v, log(z), rot6d(6)]

        Returns:
            Standard tokens [..., 9] with [tx, ty, tz, rot6d(6)]
        """
        if tokens_dn.shape[-1] != 9:
            raise ValueError(f"Expected tokens with shape [..., 9], got {tokens_dn.shape}")

        u, v, s = tokens_dn[..., 0], tokens_dn[..., 1], tokens_dn[..., 2]
        r6 = tokens_dn[..., 3:]
        z = torch.exp(s).clamp_min(self.z_min)
        tx, ty = u * z, v * z

        return torch.cat([torch.stack([tx, ty, z], dim=-1), r6], dim=-1)

    def normalize(self, tokens: torch.Tensor) -> torch.Tensor:
        """Normalize 9D tokens according to the configured mode.

        Args:
            tokens: Input tokens [..., 9] with [tx, ty, tz, rot6d(6)]

        Returns:
            Normalized tokens [..., 9]
        """
        if tokens.shape[-1] != 9:
            raise ValueError(f"Expected tokens with shape [..., 9], got {tokens.shape}")

        if self.mode == "perdim":
            # Per-dimension normalization: (x - mean) / std
            return (tokens - self._means) / self._stds
        else:
            # Depth normalization: convert to dnorm space, then standardize
            tokens_dn = self.tok_to_dnorm(tokens)
            return (tokens_dn - self._means) / self._stds

    def denormalize(self, tokens_norm: torch.Tensor, scales: torch.Tensor | None = None) -> torch.Tensor:
        """Denormalize 9D tokens according to the configured mode.

        Args:
            tokens_norm: Normalized tokens [..., 9]
            scales: Optional scale override (for dnorm mode)

        Returns:
            Denormalized tokens [..., 9]
        """
        if tokens_norm.shape[-1] != 9:
            raise ValueError(f"Expected tokens with shape [..., 9], got {tokens_norm.shape}")

        # Use provided scales or default to instance stds
        scale = scales if scales is not None else self._stds

        if self.mode == "perdim":
            # Per-dimension denormalization: x * std + mean
            return tokens_norm * scale + self._means
        else:
            # Depth denormalization: unstandardize, then convert back from dnorm
            tokens_dn = tokens_norm * scale + self._means
            return self.dnorm_to_tok(tokens_dn)

    def fit_from_batch(self, tokens: torch.Tensor) -> None:
        """Fit normalization statistics from a batch of tokens.

        Args:
            tokens: Token batch with shape [B, H, 9] or [N, 9]
        """
        if tokens.shape[-1] != 9:
            raise ValueError(f"Expected tokens with shape [..., 9], got {tokens.shape}")

        with torch.no_grad():
            if self.mode == "perdim":
                # Compute per-dimension stats across all samples
                # Flatten batch dimensions but keep token dimension
                if tokens.dim() == 3:
                    # [B, H, 9] -> compute over B and H
                    mean = tokens.mean(dim=(0, 1))
                    std = tokens.std(dim=(0, 1)).clamp_min(1e-6)
                elif tokens.dim() == 2:
                    # [N, 9] -> compute over N
                    mean = tokens.mean(dim=0)
                    std = tokens.std(dim=0).clamp_min(1e-6)
                else:
                    raise ValueError(f"Expected 2D or 3D tokens, got {tokens.dim()}D")

                self._means = mean.to(device=self.device, dtype=self.dtype)
                self._stds = std.to(device=self.device, dtype=self.dtype)
            else:
                # Fit stats in depth-normalized space
                tokens_dn = self.tok_to_dnorm(tokens)
                if tokens_dn.dim() == 3:
                    mean = tokens_dn.mean(dim=(0, 1))
                    std = tokens_dn.std(dim=(0, 1)).clamp_min(1e-3)
                elif tokens_dn.dim() == 2:
                    mean = tokens_dn.mean(dim=0)
                    std = tokens_dn.std(dim=0).clamp_min(1e-3)
                else:
                    raise ValueError(f"Expected 2D or 3D tokens, got {tokens_dn.dim()}D")

                self._means = mean.to(device=self.device, dtype=self.dtype)
                self._stds = std.to(device=self.device, dtype=self.dtype)

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "TokenNormalizer":
        """Move normalizer tensors to device/dtype.

        Args:
            device: Target device
            dtype: Target dtype

        Returns:
            Self for chaining
        """
        if device is not None:
            self.device = device
            self._means = self._means.to(device=device)
            self._stds = self._stds.to(device=device)
        if dtype is not None:
            self.dtype = dtype
            self._means = self._means.to(dtype=dtype)
            self._stds = self._stds.to(dtype=dtype)
        return self

    def state_dict(self) -> dict:
        """Get state dict for serialization."""
        return {
            "mode": self.mode,
            "means": self._means,
            "stds": self._stds,
            "z_min": self.z_min,
            "t_scale": self.t_scale,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from dict."""
        self.mode = state.get("mode", self.mode)
        self._means = state["means"].to(device=self.device, dtype=self.dtype)
        self._stds = state["stds"].to(device=self.device, dtype=self.dtype)
        self.z_min = state.get("z_min", self.z_min)
        self.t_scale = state.get("t_scale", self.t_scale)

    def __repr__(self) -> str:
        return f"TokenNormalizer(mode={self.mode}, z_min={self.z_min}, t_scale={self.t_scale})"


class OnlineStatsAccumulator:
    """Welford's online algorithm for numerically stable mean/variance.

    Used for accumulating depth-norm statistics from multiple batches
    before training starts. This provides more robust normalization
    statistics than fitting from a single batch.

    Usage:
        accumulator = OnlineStatsAccumulator(dim=9, device=device)
        for batch in dataloader:
            accumulator.update_batch(batch['target_future'])
        mean, std = accumulator.get_stats()
    """

    def __init__(self, dim: int = 9, device: torch.device | None = None, dtype: torch.dtype = torch.float32):
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.n = 0
        self.mean = torch.zeros(dim, device=device, dtype=dtype)
        self.M2 = torch.zeros(dim, device=device, dtype=dtype)

    def update_batch(self, batch: torch.Tensor) -> None:
        """Batch-optimized update using parallel algorithm.

        Args:
            batch: Tensor with shape [B, H, dim] or [N, dim]
        """
        if batch.dim() == 3:
            batch = batch.view(-1, self.dim)

        batch = batch.to(device=self.device, dtype=self.dtype)
        batch_size = batch.shape[0]

        if batch_size == 0:
            return

        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)

        if self.n == 0:
            self.mean = batch_mean
            self.M2 = batch_var * batch_size
            self.n = batch_size
        else:
            delta = batch_mean - self.mean
            total_n = self.n + batch_size
            self.mean = self.mean + delta * batch_size / total_n
            self.M2 = self.M2 + batch_var * batch_size + delta**2 * self.n * batch_size / total_n
            self.n = total_n

    @property
    def std(self) -> torch.Tensor:
        """Get standard deviation (clamped to minimum value)."""
        if self.n < 2:
            return torch.ones(self.dim, device=self.device, dtype=self.dtype)
        return torch.sqrt(self.M2 / (self.n - 1)).clamp_min(1e-3)  # Match model's clamp

    def get_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get accumulated mean and std."""
        return self.mean.clone(), self.std.clone()


def warmup_dnorm_stats(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    n_batches: int = 100,
    device: torch.device | None = None,
    rank: int = 0,
    world_size: int = 1,
    verbose: bool = True,
) -> dict[str, torch.Tensor]:
    """Fit depth-norm statistics from multiple batches.

    Pre-computes dnorm statistics from multiple batches before training starts.
    This provides more robust normalization than fitting from a single batch.

    Only rank0 accumulates stats, then broadcasts to all ranks.
    Reuses model's _tok_to_dnorm to avoid formula drift.

    Args:
        model: PoserV1 model (or DDP-wrapped model)
        dataloader: Training dataloader
        n_batches: Number of batches to accumulate (default: 100)
        device: Device for computation
        rank: Current process rank
        world_size: Total number of processes
        verbose: Print progress messages

    Returns:
        Dict with 'mean' and 'std' tensors [9]
    """
    if device is None:
        device = next(model.parameters()).device

    core_model = model.module if hasattr(model, "module") else model

    # Skip if already fitted (e.g., from checkpoint)
    if core_model._dnorm_fitted.item():
        if verbose and rank == 0:
            print("[dnorm] Already fitted from checkpoint, skipping warmup")
        return {
            "mean": core_model._dnorm_means.clone(),
            "std": core_model._dnorm_scales.clone(),
        }

    # Only rank0 accumulates
    if rank == 0:
        accumulator = OnlineStatsAccumulator(dim=9, device=device)

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= n_batches:
                    break

                target = batch.get("target_future")
                if target is None:
                    continue

                # Handle different input types
                if isinstance(target, np.ndarray):
                    target = torch.from_numpy(target)
                elif isinstance(target, list):
                    target = torch.from_numpy(np.stack(target))
                # else: already a tensor

                target = target.to(device=device, dtype=torch.float32)
                if target.dim() == 2:
                    target = target.unsqueeze(0)

                # Reuse model's conversion to avoid formula drift
                y_dn = core_model._tok_to_dnorm(target)

                accumulator.update_batch(y_dn)

                if verbose and (i + 1) % 20 == 0:
                    print(f"[dnorm warmup] {i + 1}/{n_batches} batches, n={accumulator.n:,}")

        mean, std = accumulator.get_stats()

        if verbose:
            print(f"[dnorm] Fitted from {accumulator.n:,} samples")
            print(f"  mean[:3] = {mean[:3].tolist()}")
            print(f"  std[:3]  = {std[:3].tolist()}")
    else:
        # Other ranks: create placeholder tensors for broadcast
        mean = torch.zeros(9, device=device, dtype=torch.float32)
        std = torch.ones(9, device=device, dtype=torch.float32)

    # Broadcast from rank0 to all ranks
    if world_size > 1 and dist.is_initialized():
        dist.broadcast(mean, src=0)
        dist.broadcast(std, src=0)

    # Update model buffers on all ranks
    core_model._dnorm_means.copy_(mean)
    core_model._dnorm_scales.copy_(std)
    core_model._dnorm_fitted.fill_(1)

    if core_model._dnorm_normalizer is not None:
        core_model._dnorm_normalizer.update_stats(mean, std)

    return {"mean": mean, "std": std}
