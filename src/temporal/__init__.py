from __future__ import annotations

from typing import Any

import torch

from .ar_transformer import AutoregTransformerPose
from .dit import DiTPose


def build_temporal_model(cfg: Any, w_rot: float = 2.0, w_trans: float = 2.0) -> torch.nn.Module:
    """
    ar-transformer: tiny factory to select temporal backend.

    cfg is expected to have `kind` specifying one of {"dit", "ar_transformer"}.
    Falls back to DiT when missing for backward-compat.
    """
    kind = str(getattr(cfg, "kind", "dit")).lower()
    if kind == "ar_transformer":
        return AutoregTransformerPose(
            latent_dim=int(getattr(cfg, "latent_dim", 768)),
            n_layers=int(getattr(cfg, "n_layers", 8)),
            n_heads=int(getattr(cfg, "n_heads", 12)),
            dropout=float(getattr(cfg, "dropout", 0.0)),
            max_seq_len=int(getattr(cfg, "max_seq_len", 512)),
            use_context_token=bool(getattr(cfg, "use_context_token", True)),
            w_rot=float(w_rot),
            w_trans=float(w_trans),
        )
    # Default: DiT diffusion (require out_dim to come from config; no hard-coded H=8).
    out_dim = getattr(cfg, "out_dim", None)
    if out_dim is None:
        raise ValueError(
            "build_temporal_model(kind='dit') requires cfg.out_dim (expected out_dim = H * 9). "
            "Ensure your Hydra config sets model.temporal_dit.out_dim or temporal.out_dim via data.H."
        )
    return DiTPose(
        latent_dim=int(getattr(cfg, "latent_dim", 768)),
        n_layers=int(getattr(cfg, "n_layers", 12)),
        n_heads=int(getattr(cfg, "n_heads", 12)),
        dropout=float(getattr(cfg, "dropout", 0.1)),
        out_dim=int(out_dim),
        T=int(getattr(cfg, "T", 1000)),
        ddim_steps=int(getattr(cfg, "ddim_steps", 50)),
        cond_dim=int(getattr(cfg, "cond_dim", getattr(cfg, "latent_dim", 768))),
        conditioning=str(getattr(cfg, "conditioning", "film")),
        mlp_ratio=float(getattr(cfg, "mlp_ratio", 4.0)),
    )
