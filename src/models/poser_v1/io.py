from __future__ import annotations

import os  # [dist-audit] atomic save
from typing import Any, Dict, Tuple

import torch
from omegaconf import OmegaConf

from ...utils.logger import rprint as print


def save(model: torch.nn.Module, path: str, extra: Dict[str, Any] | None = None, ema_state_dict: Dict[str, Any] | None = None) -> None:
    cfg = getattr(model, "cfg", None)
    cfg_container = OmegaConf.to_container(cfg, resolve=True) if cfg is not None else None
    payload = {
        "state_dict": model.state_dict(),
        "cfg": cfg_container,
        "extra": (extra or {}),
    }
    if isinstance(ema_state_dict, dict) and len(ema_state_dict) > 0:
        payload["ema_state_dict"] = ema_state_dict
    # [dist-audit] atomic save: write to tmp then rename
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    try:
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort fallback
        torch.save(payload, path)
    print(f"[dim]ckpt[/dim] → [bold]{path}[/bold] (saved)")


def load(path: str, map_location: str | torch.device = "cpu") -> Tuple[Dict[str, Any], Dict[str, Any] | None, Dict[str, Any] | None]:
    data = torch.load(path, map_location=map_location)
    state_dict = data.get("state_dict", data)
    cfg = data.get("cfg")
    extra = data.get("extra")
    return state_dict, cfg, extra


