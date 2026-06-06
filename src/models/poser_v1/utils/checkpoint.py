from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def _strip_module_prefix(sd: Dict[str, Any]) -> Dict[str, Any]:
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def _pick_state_dict(blob: Dict[str, Any], prefer_ema: bool) -> Dict[str, Any]:
    # Prefer EMA for eval if requested and present
    if prefer_ema and isinstance(blob.get("ema_state_dict"), dict):
        ema_sd = blob["ema_state_dict"]
        # EMA only tracks parameters, not buffers (BatchNorm running stats, etc.)
        # Merge buffers from regular state_dict to ensure correct inference
        regular_sd = blob.get("state_dict") or blob.get("model") or {}
        if regular_sd:
            ema_keys = set(ema_sd.keys())
            for key, value in regular_sd.items():
                if key not in ema_keys:
                    ema_sd[key] = value
        return ema_sd
    for key in ("state_dict", "model", "ema_state_dict", "model_ema"):
        if key in blob and isinstance(blob[key], dict):
            return blob[key]
    # blob itself may already be a state_dict
    return blob


def resolve_and_load_state_dict(ckpt_path: str, map_location: torch.device, prefer_ema: bool = False) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    blob = torch.load(ckpt_path, map_location=map_location)
    sd = _pick_state_dict(blob, prefer_ema=prefer_ema)
    sd = _strip_module_prefix(sd)
    # Keep ancillary metadata; do not drop ema_state_dict so callers can optionally restore EMA during training resume
    meta = {k: v for k, v in blob.items() if k not in ("state_dict", "model")}
    return sd, meta if isinstance(meta, dict) else {}


# DEPRECATED: These functions are no longer needed since encoder._proj is now eagerly initialized.
# Kept for backward compatibility with old calling code - they are now no-ops.


def pop_lazy_encoder_proj_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """DEPRECATED: No-op. Encoder projection is now eagerly initialized."""
    return {}


def restore_lazy_encoder_proj(core_model: nn.Module, lazy_state: Dict[str, torch.Tensor]) -> Tuple[bool, List[torch.nn.Parameter]]:
    """DEPRECATED: No-op. Encoder projection is now eagerly initialized."""
    return (False, [])
