from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from .model import PoserV1


def build_poser_v1(**kwargs: Any) -> PoserV1:
    """Hydra factory for PoserV1.

    Expects cfg to contain nested configs for encoder, temporal, mesh.
    Accepts extra kwargs (ignored) to be robust to Hydra passing extras.
    """
    cfg = OmegaConf.create(kwargs)
    return PoserV1(cfg)


