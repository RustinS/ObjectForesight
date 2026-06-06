# Architecture Improvements for Pose Prediction

This document describes four key architectural improvements to reduce the train/val gap and improve model generalization.

---

## Table of Contents

1. [Remove FiLM, Add AdaLN-Zero to DiT](#1-remove-film-add-adaln-zero-to-dit)
2. [Fit Normalization from Multiple Batches](#2-fit-normalization-from-multiple-batches)
3. [Horizon-Aware Loss Weighting](#3-horizon-aware-loss-weighting)

---

## 1. Remove FiLM, Add AdaLN-Zero to DiT

### Problem

Current conditioning is scattered and suboptimal:

1. **PTv3 Adapter**: `ContextFiLM` post-modulation (redundant - PTv3 backbone already sees context via PDNorm)
2. **DiT**: FiLM-style modulation before transformer (`a_t`, `a_c`, `b_t`, `b_c` parameters) - happens once, not layer-by-layer

### Solution

1. **Delete `ContextFiLM` class** from `ptv3_adapter.py` entirely
2. **Delete FiLM parameters** from `DiTPose` (`a_t`, `a_c`, `b_t`, `b_c`, `pre_ln`, `transformer`)
3. **Add AdaLN-Zero blocks** to DiT for layer-wise conditioning

**Note**: PTv3 backbone still receives `context_vec` for PDNorm (`src/encoders/ptv3_model.py:296-305`).

**Breaking change**: Old checkpoints will not load. Start fresh training.

**New architecture:**
```
context_vec ─────────────────────┐
                                 ↓
scene_pcd ──→ PTv3 (PDNorm uses context) ──→ pooled_feat (768-dim)
                                                   ↓
                                             cond_adapter
                                                   ↓
ctx_tokens ────────────────────────────→ DiT + AdaLN-Zero ──→ predictions
timestep ───────────────────────────────────────────↑
```

### Implementation

#### Step 1: Update Configuration

```yaml
# conf/debug.yaml

model:
  temporal_dit:
    kind: dit
    latent_dim: 512
    n_layers: 6
    n_heads: 8
    dropout: 0.1
    mlp_ratio: 4.0    # NEW: MLP expansion ratio for AdaLN blocks
    out_dim: ${eval:'${data.H}*9'}
    T: 1000
    ddim_steps: 50
```

#### Step 2: Update config_adapter.py

Add `mlp_ratio` to the config propagation:

```python
# src/utils/config_adapter.py

def apply_config_adapter(cfg: DictConfig) -> DictConfig:
    # ... existing code ...

    # DiT-only extras (add mlp_ratio)
    mlp_ratio = float(_pick("mlp_ratio", 4.0))

    # Write back compact set to top-level
    with open_dict(cfg.temporal):
        # ... existing keys ...
        if selected_kind == "dit":
            cfg.temporal.out_dim = out_dim
            cfg.temporal.T = T
            cfg.temporal.ddim_steps = ddim_steps
            cfg.temporal.mlp_ratio = mlp_ratio      # NEW

    # Mirror into model.temporal
    with open_dict(cfg.model.temporal):
        for k in ["kind", "latent_dim", "n_layers", "n_heads", "dropout",
                  "out_dim", "T", "ddim_steps", "use_context_token", "max_seq_len",
                  "mlp_ratio"]:  # Add mlp_ratio
            if hasattr(cfg.temporal, k):
                cfg.model.temporal[k] = getattr(cfg.temporal, k)

    return cfg
```

#### Step 3: Update build_temporal_model

Wire `mlp_ratio` through to DiTPose in `src/temporal/__init__.py`:

```python
def build_temporal_model(cfg: Any, w_rot: float = 2.0, w_trans: float = 2.0) -> torch.nn.Module:
    kind = str(getattr(cfg, "kind", "dit")).lower()
    if kind == "ar_transformer":
        return AutoregTransformerPose(
            # ... existing AR params ...
        )

    # DiT with AdaLN-Zero
    out_dim = getattr(cfg, "out_dim", None)
    if out_dim is None:
        raise ValueError(
            "build_temporal_model(kind='dit') requires cfg.out_dim (expected out_dim = H * 9)."
        )
    return DiTPose(
        latent_dim=int(getattr(cfg, "latent_dim", 768)),
        n_layers=int(getattr(cfg, "n_layers", 12)),
        n_heads=int(getattr(cfg, "n_heads", 12)),
        dropout=float(getattr(cfg, "dropout", 0.1)),
        mlp_ratio=float(getattr(cfg, "mlp_ratio", 4.0)),  # NEW
        out_dim=int(out_dim),
        T=int(getattr(cfg, "T", 1000)),
        ddim_steps=int(getattr(cfg, "ddim_steps", 50)),
        cond_dim=int(getattr(cfg, "cond_dim", getattr(cfg, "latent_dim", 768))),
    )
```

#### Step 4: Delete ContextFiLM from PTV3Encoder

Remove the class and all references in `src/encoders/ptv3_adapter.py`:

```python
# DELETE this class entirely:
# class ContextFiLM(torch.nn.Module):
#     ...

class PTV3Encoder(EncoderBase):
    """PointTransformerV3 encoder.

    Note: PTv3 backbone receives context_vec for PDNorm conditioning internally.
    No post-backbone FiLM modulation.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        context_dim: int = 64,   # For PTv3 PDNorm (pdnorm_context_channels)
        grid_size: float = 0.02,
        in_channels: int = 3,
        return_dict: bool = True,
        backbone_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.grid_size = float(grid_size)
        self.in_channels = int(in_channels)
        self.return_dict = return_dict

        bb_kwargs = dict(backbone_kwargs or {})
        bb_kwargs["in_channels"] = self.in_channels

        self.backbone = PointTransformerV3(**bb_kwargs)

        # No FiLM - deleted

        # Projection layer
        cls_mode = bb_kwargs.get("cls_mode", False)
        if cls_mode:
            ptv3_out_dim = bb_kwargs.get("enc_channels", (32, 64, 128, 256, 512))[-1]
        else:
            ptv3_out_dim = bb_kwargs.get("dec_channels", (64, 64, 128, 256))[0]
        self._proj = torch.nn.Linear(ptv3_out_dim, embed_dim, bias=False) if ptv3_out_dim != embed_dim else None

    def forward(
        self,
        scene_pcd: torch.Tensor,
        context_vec: torch.Tensor,
        *,
        T_cam_anchor_obj: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        B, N, _ = scene_pcd.shape
        device = scene_pcd.device

        # ... existing point dict construction code ...

        # Pass context to backbone for PDNorm (unchanged)
        point_dict = {
            "coord": coords,
            "feat": feat,
            "batch": batch,
            "grid_size": self.grid_size,
            "offset": offset,
            "context": context_vec,  # PDNorm uses this internally
        }

        # ... existing backbone forward code ...

        # Per-batch mean pooling
        pooled = []
        for b in range(B):
            m = batch_all == b
            pooled.append(feat_all[m].mean(dim=0, keepdim=True) if m.any() else torch.zeros(1, feat_all.shape[-1], device=device, dtype=feat_all.dtype))

        global_feat = self._maybe_project(torch.cat(pooled, dim=0))

        # No FiLM - return directly

        if not self.return_dict:
            return global_feat

        # ... rest of method ...
```

#### Step 5: Replace DiTPose with AdaLN-Zero Version

Completely rewrite `src/temporal/dit.py` - delete FiLM, add AdaLN:

```python
class AdaLNZeroBlock(nn.Module):
    """Transformer block with Adaptive Layer Normalization (AdaLN-Zero).

    From "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim

        # Self-attention with non-affine LayerNorm
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

        # MLP with non-affine LayerNorm
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

        # AdaLN modulation: projects conditioning to 6 * dim
        # [shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

        # Zero-initialize modulation output
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tokens [B, L, D]
            c: Conditioning vector [B, D]

        Returns:
            Output tokens [B, L, D]
        """
        modulation = self.adaLN_modulation(c)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = \
            modulation.chunk(6, dim=-1)

        # Expand: [B, D] -> [B, 1, D]
        shift_attn = shift_attn.unsqueeze(1)
        scale_attn = scale_attn.unsqueeze(1)
        gate_attn = gate_attn.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        gate_mlp = gate_mlp.unsqueeze(1)

        # Modulated self-attention
        x_norm1 = self.norm1(x)
        x_mod1 = x_norm1 * (1 + scale_attn) + shift_attn
        attn_out, _ = self.attn(x_mod1, x_mod1, x_mod1, need_weights=False)
        x = x + gate_attn * attn_out

        # Modulated MLP
        x_norm2 = self.norm2(x)
        x_mod2 = x_norm2 * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(x_mod2)
        x = x + gate_mlp * mlp_out

        return x


class DiTPose(torch.nn.Module):
    """Diffusion Transformer with AdaLN-Zero conditioning."""

    def __init__(
        self,
        latent_dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        out_dim: int | None = None,
        T: int = 1000,
        ddim_steps: int = 50,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        if out_dim is None:
            raise ValueError("DiTPose requires 'out_dim' (expected H * 9).")
        self.out_dim = int(out_dim)
        self.T = T
        self.ddim_steps = ddim_steps
        self._cond_dim = int(cond_dim) if cond_dim is not None else int(latent_dim)

        # Timestep embedding
        self.t_embed = SinusoidalTimestepEmbedding(latent_dim, T=self.T)
        self.t_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Conditioning projection
        self.cond_proj = nn.Linear(self._cond_dim, latent_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Input/output projections
        self.in_proj = nn.Linear(9, latent_dim)
        self.out_proj = nn.Linear(latent_dim, 9)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # AdaLN-Zero transformer blocks
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(latent_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

        # Final adaptive layer norm
        self.final_norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 2 * latent_dim),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        # Positional embeddings
        self.pos = nn.Parameter(torch.zeros(1, 512, latent_dim))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

        # Token type embedding (context=0, future=1)
        self.token_type_embed = nn.Embedding(2, latent_dim)
        nn.init.normal_(self.token_type_embed.weight, std=0.02)
        self.token_type_scale = nn.Parameter(torch.ones(1))

        # Relative temporal embedding (preserved from original)
        self.rel_temp_embed = SignedSinusoidalEmbedding(latent_dim)
        self.rel_temp_scale = nn.Parameter(torch.full((1,), 0.02))

    def forward(self, noisy_y: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor,
                ctx_tokens: torch.Tensor | None = None) -> torch.Tensor:
        B, D = noisy_y.shape

        # Horizon mismatch guard (preserved)
        if D % 9 != 0:
            raise ValueError(f"Expected noisy_y dim multiple of 9, got D={D}.")
        H_from_input = D // 9
        H_cfg = int(self.out_dim) // 9
        if H_from_input != H_cfg:
            raise ValueError(
                f"Horizon mismatch: cfg H={H_cfg} but input H={H_from_input}."
            )

        y = noisy_y.reshape(B, H_from_input, 9)  # reshape for non-contiguous safety
        y_tok = self.in_proj(y)

        # Context tokens
        if ctx_tokens is not None:
            if ctx_tokens.dim() == 2:
                ctx_tokens = ctx_tokens.unsqueeze(0)
            P = int(ctx_tokens.shape[1])
            ctx_tok = self.in_proj(ctx_tokens)
            tokens = torch.cat([ctx_tok, y_tok], dim=1)
        else:
            P = 0
            tokens = y_tok

        L = P + H_from_input
        device = tokens.device

        # Positional embeddings
        tokens = tokens + self.pos[:, :L, :]

        # Token type embeddings
        type_ids = torch.cat([
            torch.zeros(P, dtype=torch.long, device=device),
            torch.ones(H_from_input, dtype=torch.long, device=device)
        ])
        tokens = tokens + self.token_type_scale * self.token_type_embed(type_ids)

        # Relative temporal embeddings (preserved)
        ctx_offsets = torch.arange(-P + 1, 1, device=device)
        fut_offsets = torch.arange(0, H_from_input, device=device)
        rel_offsets = torch.cat([ctx_offsets, fut_offsets])
        tokens = tokens + self.rel_temp_scale * self.rel_temp_embed(rel_offsets, dtype=tokens.dtype)

        # Conditioning: timestep + context
        t_emb = self.t_mlp(self.t_embed(timestep))
        c_emb = self.cond_mlp(self.cond_proj(cond))
        c = t_emb + c_emb  # [B, D]

        # AdaLN-Zero blocks
        for block in self.blocks:
            tokens = block(tokens, c)

        # Final adaptive layer norm
        final_mod = self.final_adaLN(c)
        shift, scale = final_mod.chunk(2, dim=-1)
        tokens = self.final_norm(tokens)
        tokens = tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # Output projection (future tokens only)
        x9 = self.out_proj(tokens).reshape(B, L, 9)
        eps = x9[:, -H_from_input:, :].reshape(B, H_from_input * 9)

        return eps
```

---

## 2. Fit Normalization from Multiple Batches ✅ IMPLEMENTED

### Problem

Current depth-norm statistics are fit from the **first batch only** (`src/models/poser_v1/model.py:219`).

### Solution

Pre-compute dnorm statistics from multiple batches before training. Only rank0 accumulates, then broadcasts to all ranks.

### Implementation

#### Step 1: Add Configuration

```yaml
# conf/debug.yaml

model:
  norm_warmup_batches: 100  # Number of batches to accumulate
  auto_fit_dnorm: true
```

#### Step 2: Add Online Stats Accumulator

Add to `src/utils/normalization.py`:

```python
import numpy as np
import torch
import torch.distributed as dist


class OnlineStatsAccumulator:
    """Welford's online algorithm for numerically stable mean/variance."""

    def __init__(self, dim: int = 9, device: torch.device = None, dtype: torch.dtype = torch.float32):
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.n = 0
        self.mean = torch.zeros(dim, device=device, dtype=dtype)
        self.M2 = torch.zeros(dim, device=device, dtype=dtype)

    def update_batch(self, batch: torch.Tensor) -> None:
        """Batch-optimized update using parallel algorithm."""
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
        if self.n < 2:
            return torch.ones(self.dim, device=self.device, dtype=self.dtype)
        return torch.sqrt(self.M2 / (self.n - 1)).clamp_min(1e-3)  # Match model's clamp

    def get_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean.clone(), self.std.clone()
```

#### Step 3: Add Warmup Function

Add to `src/utils/normalization.py`:

```python
def warmup_dnorm_stats(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    n_batches: int = 100,
    device: torch.device = None,
    rank: int = 0,
    world_size: int = 1,
    verbose: bool = True,
) -> dict[str, torch.Tensor]:
    """Fit depth-norm statistics from multiple batches.

    Only rank0 accumulates stats, then broadcasts to all ranks.
    Reuses model's _tok_to_dnorm to avoid formula drift.
    """
    if device is None:
        device = next(model.parameters()).device

    core_model = model.module if hasattr(model, 'module') else model

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
```

#### Step 4: Integrate into Training Loop

In `src/train_main.py`:

```python
from .utils.normalization import warmup_dnorm_stats

# IMPORTANT: Checkpoint loading must happen BEFORE warmup so that _dnorm_fitted
# is correctly set from the checkpoint. Otherwise warmup will run unnecessarily.

# After checkpoint load (if resuming), before DDP wrapping:
is_dit = str(cfg.model.temporal_kind).lower() == "dit"
n_warmup = int(getattr(cfg.model, "norm_warmup_batches", 100))
resume_path = getattr(cfg.train, "resume_ckpt", None) or getattr(cfg.train, "resume", None)

# Skip warmup if resuming (checkpoint already has fitted stats)
if is_dit and n_warmup > 0 and not model._dnorm_fitted.item() and not resume_path:
    if rank == 0:
        print(f"[dnorm warmup] Fitting from {n_warmup} batches...")

    warmup_g = torch.Generator().manual_seed(base_seed)
    warmup_loader = DataLoader(
        training_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=debug_collate,
        generator=warmup_g,
    )
    warmup_dnorm_stats(
        model, warmup_loader,
        n_batches=n_warmup,
        device=device,
        rank=rank,
        world_size=world_size,
        verbose=(rank == 0),
    )
    del warmup_loader
    if world_size > 1:
        dist.barrier()
elif is_dit and resume_path and rank == 0:
    print("[dnorm warmup] Skipped - resuming from checkpoint")
```

---

## 3. Horizon-Aware Loss Weighting

### Problem

Current loss treats all horizon steps equally. Predicting H=8 is harder than H=1.

### Solution

Apply increasing weights to later horizon steps, normalized to mean=1.

### Implementation

#### Step 1: Add Configuration

```yaml
# conf/debug.yaml

model:
  horizon_weighting: true
  horizon_weight_start: 1.0
  horizon_weight_end: 2.0
  horizon_weight_schedule: "linear"  # linear, quadratic, exponential, cosine
```

#### Step 2: Add Horizon Weight Utility

Add to `src/models/poser_v1/model.py`:

```python
import math

def compute_horizon_weights(
    H: int,
    start: float = 1.0,
    end: float = 2.0,
    schedule: str = "linear",
    device: torch.device = None,
) -> torch.Tensor:
    """Compute per-horizon-step loss weights, normalized to mean=1."""
    if H == 1:
        return torch.ones(1, device=device)

    t = torch.linspace(0, 1, H, device=device)

    if schedule == "linear":
        weights = start + (end - start) * t
    elif schedule == "quadratic":
        weights = start + (end - start) * (t ** 2)
    elif schedule == "exponential":
        ratio = end / max(start, 1e-6)
        weights = start * (ratio ** t)
    elif schedule == "cosine":
        weights = start + (end - start) * (1 - torch.cos(t * math.pi)) / 2
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return weights / weights.mean()
```

#### Step 3: Integrate into DiT Loss

Modify `_compute_dit_loss` in `src/models/poser_v1/model.py`:

```python
def _compute_dit_loss(self, batch, gt_poses, B, H, device, cond_embed):
    # ... existing forward diffusion code ...

    # Get horizon weights
    use_hw = bool(getattr(self.cfg, "horizon_weighting", False))
    if use_hw:
        h_weights = compute_horizon_weights(
            H,
            start=float(getattr(self.cfg, "horizon_weight_start", 1.0)),
            end=float(getattr(self.cfg, "horizon_weight_end", 2.0)),
            schedule=str(getattr(self.cfg, "horizon_weight_schedule", "linear")),
            device=device,
        )
    else:
        h_weights = None

    # Per-step MSE: [B, H]
    v_pred_9d = v_pred.view(B, H, 9)
    v_target_9d = v_target.view(B, H, 9)
    mse_per_step = (v_pred_9d - v_target_9d).pow(2).mean(dim=-1)

    # Apply horizon weighting
    if h_weights is not None:
        mse_weighted = mse_per_step * h_weights.view(1, H)
    else:
        mse_weighted = mse_per_step

    # Apply SNR weighting (P2)
    snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
    gamma = float(getattr(self.cfg, "p2_gamma", 0.5))
    snr_weights = (1.0 + snr.squeeze(-1)) ** (-gamma)

    main_loss = (snr_weights.unsqueeze(1) * mse_weighted).mean()

    # ... SE(3) aux loss with horizon weighting ...
```

#### Step 4: Modify AR Interface for Per-Step Losses

Modify `AutoregTransformerPose.compute_loss_ar` in `src/temporal/ar_transformer.py`:

```python
def compute_loss_ar(
    self,
    tokens_gt: torch.Tensor,
    cond_embed: torch.Tensor,
    ctx_tokens: Optional[torch.Tensor] = None,
    scheduled_sampling_p: float = 0.0,
) -> dict:
    # ... existing forward code to get y_pred ...

    # Per-step MSE (not reduced) for horizon weighting
    loss_per_step = ((y_pred - tokens_gt) ** 2).mean(dim=-1)  # [B, H]

    # Scalar loss (non-detached for backprop)
    loss_main = loss_per_step.mean()

    return {
        "loss": loss_main,                    # Use this for backprop
        "loss_main": loss_main.detach(),      # For logging only
        "loss_per_step": loss_per_step,       # [B, H] for horizon weighting
        "loss_aux": torch.tensor(0.0, device=tokens_gt.device, dtype=loss_main.dtype),
        "pred_9d": y_pred,
    }
```

Then in `PoserV1._compute_ar_loss`:

```python
def _compute_ar_loss(self, batch, gt_poses, cond_embed, device):
    # ... normalize ctx_tokens before passing to AR ...
    ctx_tokens_norm = ...  # Ensure these are normalized

    ar_out = self.temporal.compute_loss_ar(
        tokens_gt=gt_poses_norm,
        cond_embed=cond_embed,
        ctx_tokens=ctx_tokens_norm,  # Must be normalized
        scheduled_sampling_p=ss_p,
    )

    H = gt_poses.shape[1]
    use_hw = bool(getattr(self.cfg, "horizon_weighting", False))

    if use_hw and "loss_per_step" in ar_out:
        h_weights = compute_horizon_weights(
            H,
            start=float(getattr(self.cfg, "horizon_weight_start", 1.0)),
            end=float(getattr(self.cfg, "horizon_weight_end", 2.0)),
            schedule=str(getattr(self.cfg, "horizon_weight_schedule", "linear")),
            device=device,
        )
        loss_per_step = ar_out["loss_per_step"]  # [B, H]
        loss_main = (loss_per_step * h_weights.view(1, H)).mean()
    else:
        # Use non-detached loss for backprop
        loss_main = ar_out["loss"]

    # ... rest of loss computation ...
```

---

## Summary

| Change | Files | Breaking? | Status |
|--------|-------|-----------|--------|
| Delete ContextFiLM, add AdaLN-Zero | `ptv3_adapter.py`, `dit.py`, `__init__.py`, `config_adapter.py` | Yes (new checkpoints only) | Pending |
| Multi-batch dnorm warmup | `normalization.py`, `train_main.py`, `debug.yaml` | No | ✅ Implemented |
| Horizon-aware loss weighting | `model.py`, `ar_transformer.py` | No | Pending |
