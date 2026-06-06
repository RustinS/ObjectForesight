from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, TypedDict

import numpy as np
import torch


@dataclass
class InitPose:
    t0: np.ndarray  # shape (3,)
    rot6d0: np.ndarray  # shape (6,)


@dataclass
class ObjectContext:
    mask: Optional[np.ndarray]  # HxW bool, optional
    bbox: Optional[Tuple[float, float, float, float]]  # x1,y1,x2,y2
    class_id: Optional[int] = None
    mesh_path: Optional[str] = None


@dataclass
class SceneSample:
    image0: np.ndarray  # HxWx3 uint8 or path-resolved image
    K: np.ndarray  # 3x3 float32
    T_wc0: np.ndarray  # 4x4 float32 (world->camera at frame 0)
    pointcloud0: Optional[np.ndarray]  # Nx3 float32, or None if depth0 provided
    depth0: Optional[np.ndarray]  # HxW float32, or None
    object: ObjectContext
    init_pose: InitPose
    target_future: np.ndarray  # Hx9 float32 (relative to frame 0)


# ============================================================================
# Model interfaces (formerly models/poser_v1/interfaces.py)
# ============================================================================


class ModelBatch(TypedDict, total=False):
    scene_pcd: torch.Tensor  # [B, N, 3]
    context_vec: torch.Tensor  # [B, C]
    mesh: Dict[str, Any] | None
    target_future: torch.Tensor  # [B, H, 9]
    color0: torch.Tensor | None
    normal0: torch.Tensor | None


class SampleOutput(TypedDict):
    poses: torch.Tensor  # [B, H, 9]
    poses_T: torch.Tensor  # [B, H, 4, 4]


@dataclass
class PoserConfig:
    H: int
    out_dim: int
    context_dim: int
    compile: bool = False
    # rm-freeze-backbone: removed user-accessible freeze flag (kept out of config)
    grad_ckpt: bool = False
    encoder: Any = None
    temporal: Any = None
    mesh: Any = None


Tensor = torch.Tensor
