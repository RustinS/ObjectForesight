from __future__ import annotations

import math

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from ...common.types import ModelBatch
from ...geom.canonicalize import canonicalize_preds_to_anchor
from ...geom.se3_ops import _det3x3_dtype_safe, _svd_dtype_safe, geodesic_distance_deg, rot6d_to_matrix
from ...temporal import build_temporal_model
from ...temporal.diffusion import compute_alphas, cosine_beta_schedule, ddim_sample, q_sample
from ...utils.logger import rprint as print
from ...utils.normalization import TokenNormalizer

DEG2RAD = 57.29577951308232


def compute_horizon_weights(
    H: int,
    start: float = 1.0,
    end: float = 2.0,
    schedule: str = "linear",
    device: torch.device = None,
) -> torch.Tensor:
    """Compute per-horizon-step loss weights, normalized to mean=1.

    Args:
        H: Number of horizon steps
        start: Weight for first horizon step
        end: Weight for last horizon step
        schedule: Weight schedule type - "linear", "quadratic", "exponential", or "cosine"
        device: Device for output tensor

    Returns:
        Tensor of shape [H] with weights normalized to mean=1
    """
    if H == 1:
        return torch.ones(1, device=device)

    t = torch.linspace(0, 1, H, device=device)

    if schedule == "linear":
        weights = start + (end - start) * t
    elif schedule == "quadratic":
        weights = start + (end - start) * (t**2)
    elif schedule == "exponential":
        ratio = end / max(start, 1e-6)
        weights = start * (ratio**t)
    elif schedule == "cosine":
        weights = start + (end - start) * (1 - torch.cos(t * math.pi)) / 2
    else:
        raise ValueError(f"Unknown horizon weight schedule: {schedule}")

    return weights / weights.mean()


class SignedSinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding for signed integer offsets (anchor-relative time)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        half = dim // 2
        log_freq = math.log(10000.0) / (half - 1)
        freq = torch.exp(torch.arange(half, dtype=torch.float32) * -log_freq)
        self.register_buffer("freq", freq)

    def forward(self, offsets: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        scaled = offsets.float().unsqueeze(-1) * self.freq
        emb = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        if dtype is not None:
            emb = emb.to(dtype)
        return emb


class AnchorQueryAttentionPool(nn.Module):
    """Cross-attention pooling: anchor frame queries all context frames."""

    def __init__(self, embed_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})")
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.time_embed = SignedSinusoidalEmbedding(embed_dim)
        self.time_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: (B, P, D) context frame embeddings, anchor is proj[:, -1, :]
        Returns:
            pooled: (B, D) anchor-attended context vector
        """
        B, P, D = proj.shape

        # Query = anchor frame (last context frame)
        query = proj[:, -1:, :]  # (B, 1, D)

        # Keys/Values = all context frames with time embeddings
        # Anchor-relative offsets: [-P+1, -P+2, ..., -1, 0]
        offsets = torch.arange(-P + 1, 1, device=proj.device)
        time_emb = self.time_embed(offsets, dtype=proj.dtype)  # (P, D)
        scale = self.time_scale.to(dtype=proj.dtype)
        kv = proj + scale * time_emb.unsqueeze(0)  # (B, P, D)

        # Cross-attention: query attends to all context frames
        out, _ = self.attn(query, kv, kv, need_weights=False)  # (B, 1, D)
        return out.squeeze(1)  # (B, D)


class PoserV1(torch.nn.Module):
    """Future pose prediction with DiT (diffusion) or AR transformer backends."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Loss weights (required)
        w_rot = getattr(cfg, "w_rot", None) or getattr(getattr(cfg, "model", None), "w_rot", None)
        w_trans = getattr(cfg, "w_trans", None) or getattr(getattr(cfg, "model", None), "w_trans", None)
        if w_rot is None or w_trans is None:
            raise ValueError("PoserV1 requires 'w_rot' and 'w_trans' in cfg or cfg.model")
        self._w_rot = float(w_rot)
        self._w_trans = float(w_trans)

        self.encoder = hydra.utils.instantiate(cfg.encoder)
        self.mesh_encoder = self._instantiate_mesh(cfg.mesh)
        self.temporal = build_temporal_model(cfg.temporal, w_rot=self._w_rot, w_trans=self._w_trans)
        self.dit = self.temporal

        self._temporal_kind = str(getattr(cfg.temporal, "kind", "dit")).lower()
        ctx_dim = int(cfg.encoder.get("context_dim", cfg.context_dim))
        # Hand context: adds 42D (21D per hand × 2) to context input
        self.use_hand_context = cfg.get("use_hand_context", False)
        hand_dim = 42 if self.use_hand_context else 0
        context_input_dim = 13 + hand_dim  # init9d(9) + bbox(4) + hands(42)
        self.context_proj = nn.Linear(context_input_dim, ctx_dim)
        self._context_pool_mode = str(getattr(cfg, "context_pool", "attn")).lower()
        if self._context_pool_mode == "attn":
            self.context_pool = AnchorQueryAttentionPool(ctx_dim, n_heads=4)
        else:
            self.context_pool = None  # Use mean pooling

        # Per-dimension normalization buffers (for both AR and DiT)
        self.register_buffer("_perdim_scales", torch.ones(9), persistent=True)
        self.register_buffer("_perdim_means", torch.zeros(9), persistent=True)
        self.register_buffer("_perdim_fitted", torch.tensor(0, dtype=torch.uint8), persistent=True)

        # Centralized normalizer for per-dimension mode
        self._perdim_normalizer = None  # Lazy initialization

        if self._temporal_kind == "dit":
            self.register_buffer("_scale_t", torch.tensor(float(getattr(cfg, "trans_scale", 0.1))), persistent=True)
            self.register_buffer("_scale_r", torch.tensor(float(getattr(cfg, "rot6d_scale", 1.0))), persistent=True)
            self.register_buffer("_T_train", torch.tensor(int(cfg.temporal.T)), persistent=True)
            self.register_buffer("_dnorm_scales", torch.ones(9), persistent=True)
            self.register_buffer("_dnorm_means", torch.zeros(9), persistent=True)
            self.register_buffer("_dnorm_fitted", torch.tensor(0, dtype=torch.uint8), persistent=True)
            self._setup_diffusion_schedule(cfg)

            # Centralized normalizer for depth-norm mode
            self._dnorm_normalizer = None  # Lazy initialization

        if getattr(cfg, "grad_ckpt", False):
            self._enable_grad_checkpointing()

        # Eagerly initialize cond_adapter (fixes optimizer/DDP sync issues)
        encoder_out_dim = int(cfg.encoder.get("embed_dim", ctx_dim))
        temporal_latent_dim = int(getattr(self.temporal, "latent_dim", encoder_out_dim))
        if encoder_out_dim != temporal_latent_dim:
            self._cond_adapter = nn.Linear(encoder_out_dim, temporal_latent_dim, bias=False)
        else:
            self._cond_adapter = nn.Identity()

        self._grad_audit = False
        self._ga_tensors = {}

        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        enc_name = cfg.encoder.get("_target_", "encoder").split(".")[-1]
        print(f"[bold]PoserV1[/bold]: encoder=[green]{enc_name}[/green], params≈[yellow]{total_params:.2f}M[/yellow]")

    def _setup_diffusion_schedule(self, cfg: DictConfig) -> None:
        betas = cosine_beta_schedule(int(cfg.temporal.T))
        self.register_buffer("betas", betas, persistent=False)
        for key, value in compute_alphas(betas).items():
            self.register_buffer(key, value, persistent=False)

    def _instantiate_mesh(self, mesh_cfg: DictConfig | None) -> nn.Module | None:
        if mesh_cfg is None or mesh_cfg.get("_target_") is None:
            return None
        mod = hydra.utils.instantiate(mesh_cfg)
        return mod if isinstance(mod, nn.Module) else None

    def to_device_dtype(self, device: str | torch.device, dtype: torch.dtype) -> None:
        self.to(device=device, dtype=dtype, non_blocking=True)

    def num_params(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)

    def encode(self, scene_pcd: torch.Tensor, context_vec: torch.Tensor, T_cam_anchor_obj: torch.Tensor | None = None, mesh: dict | None = None) -> torch.Tensor:
        out = self.encoder(scene_pcd, context_vec, T_cam_anchor_obj=T_cam_anchor_obj)
        enc = out["global"] if isinstance(out, dict) and "global" in out else out
        enc = self._cond_adapter(enc)
        self._ga_retain("encoder_out", enc)
        return enc

    def forward(
        self, batch_or_y_noisy: torch.Tensor | dict, t: torch.Tensor | None = None, cond_embed: torch.Tensor | None = None, ctx_tokens: torch.Tensor | None = None
    ) -> torch.Tensor | dict:
        """Unified forward for training (dict batch) and DiT inference (tensor).

        Training mode: Pass dict batch, routes to compute_loss() for DDP compatibility.
        Inference mode: Pass tensor args for DiT denoising step.
        """
        # Training mode: batch_or_y_noisy is a dict batch
        if isinstance(batch_or_y_noisy, dict):
            return self.compute_loss(batch_or_y_noisy)

        # DiT inference mode: batch_or_y_noisy is y_noisy tensor
        out = self.dit(batch_or_y_noisy, t, cond_embed, ctx_tokens)
        self._ga_retain("dit_out", out)
        if self._grad_audit and hasattr(self.dit, "_ga"):
            for k, v in (self.dit._ga or {}).items():
                self._ga_retain(f"dit_{k}", v)
        return out

    def _ga_retain(self, name: str, t: torch.Tensor | None) -> None:
        """Conditionally retain grad for audit."""
        if self._grad_audit and isinstance(t, torch.Tensor) and t.requires_grad:
            t.retain_grad()
            self._ga_tensors[name] = t

    # Token normalization (DiT-only)
    def _get_dnorm_normalizer(self) -> TokenNormalizer:
        """Get or create the depth-norm normalizer (lazy init)."""
        if self._dnorm_normalizer is None:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            z_min = float(getattr(self.cfg, "z_min", 0.05))
            self._dnorm_normalizer = TokenNormalizer(mode="dnorm", t_mean=self._dnorm_means, t_std=self._dnorm_scales, z_min=z_min, device=device, dtype=dtype)
        return self._dnorm_normalizer

    def _tok_to_dnorm(self, y9: torch.Tensor) -> torch.Tensor:
        """Convert [tx,ty,tz,r6] to depth-normalized [u,v,log(z),r6]."""
        return self._get_dnorm_normalizer().tok_to_dnorm(y9)

    def _dnorm_to_tok(self, y9_dn: torch.Tensor) -> torch.Tensor:
        """Convert depth-normalized [u,v,log(z),r6] back to [tx,ty,tz,r6]."""
        return self._get_dnorm_normalizer().dnorm_to_tok(y9_dn)

    def _dnorm_normalize(self, y9_dn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize depth-normalized tokens."""
        # Note: y9_dn is already in depth-normalized space, so we just standardize
        return (y9_dn - self._dnorm_means) / self._dnorm_scales, self._dnorm_scales

    def _dnorm_unnormalize(self, y9_dn_n: torch.Tensor, scales: torch.Tensor | None = None) -> torch.Tensor:
        """Unnormalize depth-normalized tokens."""
        return y9_dn_n * (scales if scales is not None else self._dnorm_scales) + self._dnorm_means

    def _maybe_fit_dnorm_from_batch(self, y0_9d: torch.Tensor) -> None:
        """One-time fit of d-norm stats from first training batch."""
        if not self.training or self._dnorm_fitted.item() or not getattr(self.cfg, "auto_fit_dnorm", True):
            return
        with torch.no_grad():
            y_dn = self._tok_to_dnorm(y0_9d)
            self._dnorm_means.copy_(y_dn.mean(dim=(0, 1)))
            self._dnorm_scales.copy_(y_dn.std(dim=(0, 1)).clamp_min(1e-3))
            self._dnorm_fitted.fill_(1)
            # Update the normalizer if it exists
            if self._dnorm_normalizer is not None:
                self._dnorm_normalizer.update_stats(self._dnorm_means, self._dnorm_scales)

    def _get_perdim_normalizer(self) -> TokenNormalizer:
        """Get or create the per-dimension normalizer (lazy init)."""
        if self._perdim_normalizer is None:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            self._perdim_normalizer = TokenNormalizer(mode="perdim", t_mean=self._perdim_means, t_std=self._perdim_scales, device=device, dtype=dtype)
        return self._perdim_normalizer

    def _maybe_fit_perdim_from_batch(self, y0_9d: torch.Tensor) -> None:
        """One-time fit of per-dimension normalization stats from first training batch."""
        if not self.training or self._perdim_fitted.item() or not getattr(self.cfg, "auto_fit_perdim", True):
            return
        with torch.no_grad():
            # y0_9d: (B, H, 9) - compute mean and std across batch and horizon
            self._perdim_means.copy_(y0_9d.mean(dim=(0, 1)))
            self._perdim_scales.copy_(y0_9d.std(dim=(0, 1)).clamp_min(1e-6))
            self._perdim_fitted.fill_(1)
            # Update the normalizer if it exists
            if self._perdim_normalizer is not None:
                self._perdim_normalizer.update_stats(self._perdim_means, self._perdim_scales)

    def _perdim_normalize(self, y9: torch.Tensor) -> torch.Tensor:
        """Normalize 9D tokens to approx unit variance per dimension."""
        return self._get_perdim_normalizer().normalize(y9)

    def _perdim_unnormalize(self, y9_n: torch.Tensor) -> torch.Tensor:
        """Unnormalize 9D tokens back to original scale."""
        return self._get_perdim_normalizer().denormalize(y9_n)

    def condition_from_batch(self, batch: ModelBatch) -> dict:
        device = next(self.parameters()).device
        pcd_data = batch.get("scene_pcd")
        if pcd_data is None:
            pcd_data = batch.get("pointcloud0")
        scene_pcd = self._to_pcd(pcd_data, device)

        init9 = self._to_tensor(batch["context_init_9d"], device)
        bboxs = self._to_tensor(batch["context_bbox_norm"], device)
        if init9.dim() == 2:
            init9 = init9.unsqueeze(0)
        if bboxs.dim() == 2:
            bboxs = bboxs.unsqueeze(0)
        assert init9.size(-1) == 9 and bboxs.size(-1) == 4, f"Expected init9(...,9) bbox(...,4), got {init9.shape}, {bboxs.shape}"
        assert init9.shape[:2] == bboxs.shape[:2], f"Shape mismatch: {init9.shape} vs {bboxs.shape}"

        ctx_seq = torch.cat([init9, bboxs], dim=-1)  # (B, P, 13)

        # Concatenate hand poses if enabled
        if self.use_hand_context:
            hand_poses = batch.get("context_hand_poses")
            if hand_poses is not None:
                hand_poses = self._to_tensor(hand_poses, device)
                if hand_poses.dim() == 2:
                    hand_poses = hand_poses.unsqueeze(0)
                ctx_seq = torch.cat([ctx_seq, hand_poses], dim=-1)  # (B, P, 55)
            else:
                # Zero-fill if dataset doesn't provide hands (e.g., EPIC)
                B_tmp, P_tmp, _ = ctx_seq.shape
                hand_zeros = torch.zeros(B_tmp, P_tmp, 42, device=device, dtype=ctx_seq.dtype)
                ctx_seq = torch.cat([ctx_seq, hand_zeros], dim=-1)

        B, P, D = ctx_seq.shape
        proj = self.context_proj(ctx_seq.view(B * P, D)).view(B, P, -1)

        # Extract T_cam_anchor_obj from context (last frame = anchor)
        ctx_T = batch.get("context_T_cam_anchor_obj")
        if ctx_T is not None:
            ctx_T = self._to_tensor(ctx_T, device)
            T_anchor = ctx_T[:, -1, :, :]  # (B, 4, 4)
        else:
            T_anchor = None

        context_vec = self.context_pool(proj) if self.context_pool is not None else proj.mean(dim=1)
        return {"scene_pcd": scene_pcd, "context_vec": context_vec, "ctx_raw": ctx_seq, "T_cam_anchor_obj": T_anchor}

    def _to_pcd(self, scene, device: torch.device) -> torch.Tensor:
        if scene is None:
            raise ValueError("scene_pcd/pointcloud0 missing from batch")
        if isinstance(scene, list):
            return torch.from_numpy(np.stack(scene)).float().to(device, non_blocking=True)
        if isinstance(scene, np.ndarray):
            return torch.from_numpy(scene).unsqueeze(0).float().to(device, non_blocking=True)
        return scene.to(device, non_blocking=True).float()

    def _to_tensor(self, data, device: torch.device) -> torch.Tensor:
        if data is None:
            return torch.zeros(4, device=device)
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().to(device, non_blocking=True)
        if isinstance(data, (list, tuple)):
            return torch.tensor(data, dtype=torch.float32, device=device)
        return data.to(device, non_blocking=True).float()

    def _item_b(self, v, b: int, default=None):
        """Extract per-batch item for anchor meta construction."""
        if v is None:
            return default
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
            return arr[b] if arr.ndim >= 1 else arr
        if isinstance(v, (list, tuple)):
            return v[b]
        if hasattr(v, "__array__"):
            arr = np.asarray(v)
            return arr[b] if arr.ndim >= 1 else arr
        return v

    def compute_loss(self, batch: ModelBatch) -> dict:
        device = next(self.parameters()).device
        target = batch["target_future"]
        if isinstance(target, list):
            target = torch.from_numpy(np.stack(target)).float()
        elif isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()

        B, H = target.shape[0], int(self.cfg.H)
        gt_poses = target.view(B, H, 9).to(device)

        if self._temporal_kind == "dit" and self.training:
            self._maybe_fit_dnorm_from_batch(gt_poses)

        info = self.condition_from_batch(batch)
        T_cam_anchor_obj = info.get("T_cam_anchor_obj")
        cond_embed = self.encode(info["scene_pcd"], info["context_vec"], T_cam_anchor_obj=T_cam_anchor_obj)

        if self._temporal_kind == "ar_transformer":
            return self._compute_ar_loss(batch, gt_poses, cond_embed, device)
        return self._compute_dit_loss(batch, gt_poses, B, H, device, cond_embed)

    def _compute_auxiliary_losses(self, y_pred: torch.Tensor, y_gt: torch.Tensor, device: torch.device, w_t: torch.Tensor | None = None) -> dict:
        """Compute velocity, acceleration, displacement losses with SE(3)-proper deltas.

        Args:
            y_pred: Predicted poses [B, H, 9]
            y_gt: Ground truth poses [B, H, 9]
            device: Device for tensor creation
            w_t: Optional per-sample time weight [B] for downweighting high-noise samples (DiT only)

        Returns:
            Dict of loss name -> loss value
        """
        B, H = y_pred.shape[:2]

        # Extract translations and rotations
        t_pred, t_gt = y_pred[..., :3], y_gt[..., :3]
        R_pred = rot6d_to_matrix(y_pred[..., 3:9].reshape(-1, 6)).view(B, H, 3, 3)
        R_gt = rot6d_to_matrix(y_gt[..., 3:9].reshape(-1, 6)).view(B, H, 3, 3)

        # Velocity loss (SE(3) deltas) - per-sample then weighted
        loss_vel_per_sample = torch.zeros(B, device=device)
        if getattr(self.cfg, "loss_velocity", False) and H > 1:
            # Translation velocity
            delta_t_pred = t_pred[:, 1:] - t_pred[:, :-1]  # [B, H-1, 3]
            delta_t_gt = t_gt[:, 1:] - t_gt[:, :-1]
            vel_trans_err = (delta_t_pred - delta_t_gt).pow(2).mean(dim=(1, 2))  # [B]

            # Rotation velocity (relative rotations R_k^T @ R_{k+1})
            R_rel_pred = R_pred[:, :-1].transpose(-1, -2) @ R_pred[:, 1:]
            R_rel_gt = R_gt[:, :-1].transpose(-1, -2) @ R_gt[:, 1:]
            vel_rot_err_rad = (geodesic_distance_deg(R_rel_pred.reshape(-1, 3, 3), R_rel_gt.reshape(-1, 3, 3)) / DEG2RAD).view(B, H - 1)
            vel_rot_err = vel_rot_err_rad.pow(2).mean(dim=1)  # [B]

            loss_vel_per_sample = vel_trans_err + vel_rot_err

        # Acceleration loss (second differences) - per-sample then weighted
        loss_acc_per_sample = torch.zeros(B, device=device)
        if getattr(self.cfg, "loss_acceleration", False) and H > 2:
            # Translation acceleration
            delta_t_pred = t_pred[:, 1:] - t_pred[:, :-1]
            delta_t_gt = t_gt[:, 1:] - t_gt[:, :-1]
            acc_t_pred = delta_t_pred[:, 1:] - delta_t_pred[:, :-1]  # [B, H-2, 3]
            acc_t_gt = delta_t_gt[:, 1:] - delta_t_gt[:, :-1]
            acc_trans_err = (acc_t_pred - acc_t_gt).pow(2).mean(dim=(1, 2))  # [B]

            # Rotation acceleration (relative of relative)
            R_rel_pred = R_pred[:, :-1].transpose(-1, -2) @ R_pred[:, 1:]
            R_rel_gt = R_gt[:, :-1].transpose(-1, -2) @ R_gt[:, 1:]
            R_acc_pred = R_rel_pred[:, :-1].transpose(-1, -2) @ R_rel_pred[:, 1:]
            R_acc_gt = R_rel_gt[:, :-1].transpose(-1, -2) @ R_rel_gt[:, 1:]
            acc_rot_err_rad = (geodesic_distance_deg(R_acc_pred.reshape(-1, 3, 3), R_acc_gt.reshape(-1, 3, 3)) / DEG2RAD).view(B, H - 2)
            acc_rot_err = acc_rot_err_rad.pow(2).mean(dim=1)  # [B]

            loss_acc_per_sample = acc_trans_err + acc_rot_err

        # Displacement loss (translation only) - per-sample then weighted
        loss_disp_per_sample = torch.zeros(B, device=device)
        if getattr(self.cfg, "loss_displacement", False) and H > 1:
            disp_pred = t_pred[:, -1] - t_pred[:, 0]
            disp_gt = t_gt[:, -1] - t_gt[:, 0]
            loss_disp_per_sample = (disp_pred - disp_gt).norm(dim=-1)  # [B]

        # Combine with config weights
        w_vel = float(getattr(self.cfg, "w_velocity", 0.5))
        w_acc = float(getattr(self.cfg, "w_acceleration", 0.1))
        w_disp = float(getattr(self.cfg, "w_displacement", 0.5))

        loss_extra_per_sample = w_vel * loss_vel_per_sample + w_acc * loss_acc_per_sample + w_disp * loss_disp_per_sample

        # Apply per-sample time weighting if provided (DiT: alpha_bar)
        if w_t is not None:
            loss_extra = (w_t * loss_extra_per_sample).mean()
        else:
            loss_extra = loss_extra_per_sample.mean()

        return {
            "loss_extra": loss_extra,
            "loss_vel": loss_vel_per_sample.mean().detach(),  # For logging, unweighted
            "loss_acc": loss_acc_per_sample.mean().detach(),
            "loss_disp": loss_disp_per_sample.mean().detach(),
        }

    def _compute_ar_loss(self, batch: ModelBatch, gt_poses: torch.Tensor, cond_embed: torch.Tensor, device: torch.device) -> dict:
        ctx_tokens = batch.get("context_init_9d")
        if ctx_tokens is not None and not isinstance(ctx_tokens, torch.Tensor):
            ctx_tokens = torch.from_numpy(np.asarray(ctx_tokens)).float().to(device)
        elif ctx_tokens is not None:
            ctx_tokens = ctx_tokens.to(device, dtype=torch.float32, non_blocking=True)

        # Fit per-dimension normalization stats if not already fitted
        if self.training:
            self._maybe_fit_perdim_from_batch(gt_poses)

        # Normalize tokens for training (better convergence)
        gt_poses_norm = self._perdim_normalize(gt_poses)
        ctx_tokens_norm = self._perdim_normalize(ctx_tokens) if ctx_tokens is not None else None

        out = self.temporal.compute_loss_ar(
            tokens_gt=gt_poses_norm,
            cond_embed=cond_embed,
            ctx_tokens=ctx_tokens_norm,
            scheduled_sampling_p=float(batch.get("scheduled_sampling_p", 0.0)),
        )

        # Apply horizon weighting if enabled
        H = gt_poses.shape[1]
        use_hw = bool(getattr(self.cfg, "horizon_weighting", False))
        if use_hw and "loss_per_step" in out:
            h_weights = compute_horizon_weights(
                H,
                start=float(getattr(self.cfg, "horizon_weight_start", 1.0)),
                end=float(getattr(self.cfg, "horizon_weight_end", 2.0)),
                schedule=str(getattr(self.cfg, "horizon_weight_schedule", "linear")),
                device=device,
            )
            loss_per_step = out["loss_per_step"]  # [B, H]
            out["loss"] = (loss_per_step * h_weights.view(1, H)).mean()

        y_pred_norm = out.get("pred_9d")
        if y_pred_norm is not None:
            # Unnormalize predictions for metrics
            y_pred = self._perdim_unnormalize(y_pred_norm)
            out["pred_9d"] = y_pred

            B, Hn = y_pred.shape[0], y_pred.shape[1]
            t_pred, r6_pred = y_pred[..., :3], y_pred[..., 3:9]
            t_gt = gt_poses[..., :3]
            R_pred = rot6d_to_matrix(r6_pred.reshape(-1, 6)).view(B, Hn, 3, 3)
            R_gt = rot6d_to_matrix(gt_poses[..., 3:9].reshape(-1, 6)).view(B, Hn, 3, 3)

            rot_deg = geodesic_distance_deg(R_pred.view(-1, 3, 3), R_gt.view(-1, 3, 3)).view(B, Hn).mean()
            trans_l2 = (t_pred - t_gt).norm(dim=-1).mean()

            aux_loss = self._w_rot * (rot_deg / DEG2RAD) + self._w_trans * trans_l2

            # Compute auxiliary losses (no time weighting for AR)
            aux_losses = self._compute_auxiliary_losses(y_pred, gt_poses, device, w_t=None)

            out["loss"] = out["loss"] + aux_loss + aux_losses["loss_extra"]
            out["loss_aux"] = aux_loss.detach()
            out["rot_deg"], out["trans_l2"] = rot_deg, trans_l2
            out["loss_vel"] = aux_losses["loss_vel"]
            out["loss_acc"] = aux_losses["loss_acc"]
            out["loss_disp"] = aux_losses["loss_disp"]

            self._ga_retain("ar_out", y_pred)
            self._ga_retain("y0_pred_9d", y_pred)
            self._ga_retain("t_pred", t_pred)
            self._ga_retain("R_pred", R_pred)
        return out

    def _compute_dit_loss(self, batch: ModelBatch, gt_poses: torch.Tensor, B: int, H: int, device: torch.device, cond_embed: torch.Tensor) -> dict:
        # Fit per-dimension normalization stats if not already fitted
        if self.training:
            self._maybe_fit_perdim_from_batch(gt_poses)

        # Depth-normalize and standardize
        y0_dn = self._tok_to_dnorm(gt_poses)
        y0_n, scales = self._dnorm_normalize(y0_dn)
        y0_flat = y0_n.view(B, H * 9)

        # Forward diffusion
        timesteps = torch.randint(0, int(self.cfg.temporal.T), (B,), device=device)
        noise = torch.randn_like(y0_flat)
        y_t = q_sample(y0_flat, timesteps, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, noise)

        # v-parameterization
        alpha_bar = self.alphas_cumprod[timesteps].unsqueeze(-1)
        sqrt_ab, sqrt_1m = torch.sqrt(alpha_bar), torch.sqrt(1.0 - alpha_bar)
        v_target = sqrt_ab * noise - sqrt_1m * y0_flat

        # Context tokens in depth-norm space
        ctx_dn_n = None
        ctx9 = batch.get("context_init_9d")
        if ctx9 is not None:
            ctx9_t = self._to_tensor(ctx9, device)
            if ctx9_t.dim() == 2:
                ctx9_t = ctx9_t.unsqueeze(0)
            ctx_dn = self._tok_to_dnorm(ctx9_t)
            ctx_dn_n = (ctx_dn - self._dnorm_means) / scales

        # Selective bf16 for DiT only (encoder stays fp32)
        dit_bf16 = bool(getattr(self.cfg, "dit_bf16", False))
        if dit_bf16 and torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v_pred = self.forward(y_t.bfloat16(), timesteps, cond_embed.bfloat16(), ctx_tokens=ctx_dn_n.bfloat16() if ctx_dn_n is not None else None)
            v_pred = v_pred.float()  # Cast back to fp32 for loss computation
        else:
            v_pred = self.forward(y_t, timesteps, cond_embed, ctx_tokens=ctx_dn_n)
        self._ga_retain("v_pred", v_pred)

        # SNR-weighted loss with non-vanishing P2-style weights
        snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
        gamma = float(getattr(self.cfg, "p2_gamma", 0.5))
        snr_weights = (1.0 + snr.squeeze(-1)) ** (-gamma)  # Non-vanishing at high noise [B]

        # Compute per-step MSE for horizon weighting
        v_pred_9d = v_pred.view(B, H, 9)
        v_target_9d = v_target.view(B, H, 9)
        mse_per_step = (v_pred_9d - v_target_9d).pow(2).mean(dim=-1)  # [B, H]

        # Apply horizon weighting if enabled
        use_hw = bool(getattr(self.cfg, "horizon_weighting", False))
        if use_hw:
            h_weights = compute_horizon_weights(
                H,
                start=float(getattr(self.cfg, "horizon_weight_start", 1.0)),
                end=float(getattr(self.cfg, "horizon_weight_end", 2.0)),
                schedule=str(getattr(self.cfg, "horizon_weight_schedule", "linear")),
                device=device,
            )
            mse_weighted = mse_per_step * h_weights.view(1, H)
        else:
            mse_weighted = mse_per_step

        # Combine SNR and horizon weighting
        main_loss = (snr_weights.unsqueeze(1) * mse_weighted).mean()

        # Reconstruct x0
        x0_flat = sqrt_ab * y_t - sqrt_1m * v_pred
        x0_dn = self._dnorm_unnormalize(x0_flat.view(B, H, 9), scales)
        pred_poses = self._dnorm_to_tok(x0_dn)
        self._ga_retain("y0_pred_9d", pred_poses)

        # SE3 metrics (per-sample for time weighting)
        t_pred, r6_pred = pred_poses[..., :3], pred_poses[..., 3:9]
        t_gt, r6_gt = gt_poses[..., :3], gt_poses[..., 3:9]
        R_pred = rot6d_to_matrix(r6_pred.reshape(-1, 6)).view(B, H, 3, 3)
        R_gt = rot6d_to_matrix(r6_gt.reshape(-1, 6)).view(B, H, 3, 3)
        self._ga_retain("R_pred", R_pred)
        self._ga_retain("t_pred", t_pred)

        # Per-sample SE3 metrics for time-weighted aux_loss
        rot_deg_per_sample = geodesic_distance_deg(R_pred.view(-1, 3, 3), R_gt.view(-1, 3, 3)).view(B, H).mean(dim=1)  # [B]
        trans_l2_per_sample = (t_pred - t_gt).norm(dim=-1).mean(dim=1)  # [B]

        # Time-weighted aux loss (downweight at high noise where x0 reconstruction is unreliable)
        w_aux_t = alpha_bar.squeeze(-1)  # [B]
        aux_loss_per_sample = self._w_rot * (rot_deg_per_sample / DEG2RAD) + self._w_trans * trans_l2_per_sample
        aux_loss = (w_aux_t * aux_loss_per_sample).mean()

        # Keep unweighted metrics for logging
        rot_deg = rot_deg_per_sample.mean()
        trans_l2 = trans_l2_per_sample.mean()

        # Anchor-frame loss (skip if abs_in_anchor_cam to avoid double-counting with aux_loss)
        out_fmt = getattr(self.cfg, "output_format", None)
        pred_mode = {"delta_from_prev": "deltas_from_prev_cam", "delta_from_anchor": "deltas_from_anchor_cam"}.get(out_fmt, "abs_in_anchor_cam")

        if pred_mode == "abs_in_anchor_cam":
            anchor_loss = torch.tensor(0.0, device=device)
            rot_errs, trans_errs = [], []
        else:
            # For delta modes, also time-weight anchor_loss by alpha_bar
            anchor_loss_raw, rot_errs, trans_errs = self._compute_anchor_loss(batch, pred_poses, gt_poses, B, device)
            anchor_loss = w_aux_t.mean() * anchor_loss_raw

        # Depth floor penalty
        z_min = float(getattr(self.cfg, "z_min", 0.05))
        L_zmin = torch.relu(z_min - torch.exp(x0_dn[..., 2])).mean() * 0.01

        # Compute auxiliary losses with time weighting (pass w_aux_t to function)
        aux_losses = self._compute_auxiliary_losses(pred_poses, gt_poses, device, w_t=w_aux_t)

        total_loss = main_loss + anchor_loss + aux_loss + L_zmin + aux_losses["loss_extra"]
        rot_metric = torch.stack(rot_errs).mean() if rot_errs else rot_deg
        trans_metric = torch.stack(trans_errs).mean() if trans_errs else trans_l2

        diag = self._compute_rotation_diagnostics(R_pred, t_gt, R_gt, B, H, device)
        return {
            "loss": total_loss,
            "loss_main": main_loss.detach(),
            "loss_aux": aux_loss.detach(),
            "loss_anchor": anchor_loss.detach(),
            "rot_deg": rot_metric,
            "trans_l2": trans_metric,
            "pred_9d": pred_poses,
            "loss_vel": aux_losses["loss_vel"],
            "loss_acc": aux_losses["loss_acc"],
            "loss_disp": aux_losses["loss_disp"],
            **diag,
        }

    def _compute_anchor_loss(self, batch: ModelBatch, pred_poses: torch.Tensor, gt_poses: torch.Tensor, B: int, device: torch.device):
        """Compute anchor-frame canonicalized loss."""
        out_fmt = getattr(self.cfg, "output_format", None)
        pred_mode = {"delta_from_prev": "deltas_from_prev_cam", "delta_from_anchor": "deltas_from_anchor_cam"}.get(out_fmt, "abs_in_anchor_cam")

        anchor_losses, rot_errs, trans_errs = [], [], []
        for b in range(B):
            meta = {
                "K": self._item_b(batch.get("K"), b),
                "frame_ids": self._item_b(batch.get("frame_ids"), b),
                "anchor_frame_idx": int(self._item_b(batch.get("anchor_frame_idx"), b, 0) or 0),
                "anchor_local_idx": int(self._item_b(batch.get("anchor_local_idx"), b, 0) or 0),
                "T_c_w": self._item_b(batch.get("T_c_w"), b),
                "T_c_o": self._item_b(batch.get("T_c_o"), b),
                "T_cam_anchor_obj": self._item_b(batch.get("T_cam_anchor_obj"), b),
                "t_mean": [0.0, 0.0, 0.0],
                "t_std": [1.0, 1.0, 1.0],
            }
            T_pred, _ = canonicalize_preds_to_anchor(pred_poses[b : b + 1], meta, pred_mode, "w<-c", True, False)
            T_gt, _ = canonicalize_preds_to_anchor(gt_poses[b : b + 1], meta, pred_mode, "w<-c", True, False)
            Rp, Rg = T_pred[0][..., :3, :3], T_gt[0][..., :3, :3]
            tp, tg = T_pred[0][..., :3, 3], T_gt[0][..., :3, 3]
            r_err = geodesic_distance_deg(Rp, Rg).mean()
            t_err = (tp - tg).norm(dim=-1).mean()
            rot_errs.append(r_err)
            trans_errs.append(t_err)
            anchor_losses.append(self._w_rot * (r_err / DEG2RAD) + self._w_trans * t_err)

        anchor_loss = torch.stack(anchor_losses).mean() if anchor_losses else torch.tensor(0.0, device=device)
        return anchor_loss, rot_errs, trans_errs

    def _compute_rotation_diagnostics(self, R_pred: torch.Tensor, t_gt: torch.Tensor, R_gt: torch.Tensor, B: int, H: int, device: torch.device) -> dict:
        with torch.no_grad():
            R_flat = R_pred.view(-1, 3, 3)
            RtR = R_flat.transpose(-1, -2) @ R_flat
            eye3 = torch.eye(3, device=device, dtype=R_flat.dtype)
            ortho_err = (RtR - eye3).abs().amax()

            U, _, V = _svd_dtype_safe(R_flat)
            det_err = (_det3x3_dtype_safe(U @ V.transpose(-1, -2)) - 1.0).abs().amax()

            t_norm_med = t_gt.norm(dim=-1).median()
            I_exp = torch.eye(3, device=device).expand(B, H, 3, 3)
            r_deg_med = geodesic_distance_deg(R_gt.view(-1, 3, 3), I_exp.reshape(-1, 3, 3)).view(B, H).median()

        return {"R_ortho_max_pred": ortho_err, "R_det_err_max_pred": det_err, "gt_t_norm_med": t_norm_med, "gt_r_deg_med": r_deg_med}

    @torch.no_grad()
    def sample(
        self,
        scene_pcd: torch.Tensor,
        context_vec: torch.Tensor,
        T_cam_anchor_obj: torch.Tensor | None = None,
        steps: int | None = None,
        sampler: str = "ddim",
        ctx_tokens_9d: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        B, H = scene_pcd.shape[0], int(self.cfg.H)
        cond = self.encode(scene_pcd, context_vec, T_cam_anchor_obj=T_cam_anchor_obj)

        if self._temporal_kind == "ar_transformer":
            ctx = ctx_tokens_9d
            if ctx is not None and not isinstance(ctx, torch.Tensor):
                ctx = torch.from_numpy(np.asarray(ctx)).float()
            if ctx is not None:
                ctx = ctx.to(cond.device, dtype=torch.float32, non_blocking=True)
                if ctx.dim() == 2:
                    ctx = ctx.unsqueeze(0)
                ctx = self._perdim_normalize(ctx)
            y_pred_n = self.temporal.rollout_infer(H, cond, ctx_tokens=ctx)
            return self._perdim_unnormalize(y_pred_n)

        # DiT DDIM sampling
        num_steps = steps or int(self.cfg.temporal.ddim_steps)
        ctx_dn_n = None
        if ctx_tokens_9d is not None:
            ctx9 = ctx_tokens_9d if isinstance(ctx_tokens_9d, torch.Tensor) else torch.as_tensor(ctx_tokens_9d, dtype=torch.float32)
            ctx9 = ctx9.to(cond.device, dtype=torch.float32)
            if ctx9.dim() == 2:
                ctx9 = ctx9.unsqueeze(0)
            ctx_dn_n = (self._tok_to_dnorm(ctx9) - self._dnorm_means) / self._dnorm_scales

        out_flat = ddim_sample(
            self.dit,
            (B, H * 9),
            num_steps,
            int(self._T_train.item()),
            cond,
            self.betas.to(cond.device),
            cond.device,
            predict_type="v",
            ctx_tokens=ctx_dn_n,
            generator=generator,
        )
        y0_dn = self._dnorm_unnormalize(out_flat.view(B, H, 9), self._dnorm_scales)
        return self._dnorm_to_tok(y0_dn)

    def load_weights(self, path: str, strict: bool = False) -> None:
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            print(f"[yellow]Missing keys[/yellow]: {len(missing)}")
        if unexpected:
            print(f"[yellow]Unexpected keys[/yellow]: {len(unexpected)}")

    def _enable_grad_checkpointing(self) -> None:
        for m in self.modules():
            if hasattr(m, "gradient_checkpointing_enable"):
                m.gradient_checkpointing_enable()

    def set_grad_audit(self, enabled: bool) -> None:
        self._grad_audit = bool(enabled)
        if hasattr(self.dit, "__dict__"):
            self.dit._grad_audit = enabled
        if not self._grad_audit:
            self._ga_tensors = {}

    def get_grad_audit_tensors(self) -> dict[str, torch.Tensor]:
        return dict(self._ga_tensors)
