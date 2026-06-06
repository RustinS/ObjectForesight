from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def project_points(K: torch.Tensor, T_wc: torch.Tensor, points_w: torch.Tensor) -> torch.Tensor:
    """Project 3D world points to image. K: (3,3), T_wc: (4,4), points_w: (N,3) -> (N,2)"""
    device = points_w.device
    ones = torch.ones((points_w.shape[0], 1), device=device, dtype=points_w.dtype)
    Pw = torch.cat([points_w, ones], dim=-1).T  # 4xN
    Pc = (T_wc @ Pw)[:3]
    uvw = K @ Pc
    return (uvw[:2] / uvw[2:].clamp(min=1e-6)).T


def build_object_axes(length: float = 0.1) -> torch.Tensor:
    """Return 3 axes endpoints in object frame from origin: shape (3, 2, 3)."""
    origin = torch.zeros(3)
    axes = torch.eye(3) * length
    return torch.stack([torch.stack([origin, origin + axes[i]], dim=0) for i in range(3)], dim=0)


def scale_intrinsics(K: np.ndarray, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> np.ndarray:
    """Scale intrinsics from source image size to destination image size."""
    if K is None:
        return K
    if not isinstance(K, np.ndarray) or K.shape != (3, 3):
        raise ValueError(f"scale_intrinsics expects K as (3,3), got {None if K is None else K.shape}")
    hs, ws = int(src_hw[0]), int(src_hw[1])
    hd, wd = int(dst_hw[0]), int(dst_hw[1])
    if hs <= 0 or ws <= 0 or hd <= 0 or wd <= 0 or (hs == hd and ws == wd):
        return K.astype(np.float32)
    sx, sy = wd / float(ws), hd / float(hs)
    K_scaled = K.astype(np.float32).copy()
    K_scaled[0, :] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[1, 2] *= sy
    return K_scaled


def infer_image_size_from_K(K: np.ndarray) -> Tuple[int, int]:
    """Infer (H, W) from K's principal point (cx, cy)."""
    if not isinstance(K, np.ndarray) or K.shape != (3, 3):
        raise ValueError(f"infer_image_size_from_K expects K as (3,3), got {None if K is None else K.shape}")
    cx, cy = float(K[0, 2]), float(K[1, 2])

    def center_to_size(c: float) -> int:
        if not np.isfinite(c):
            return 1
        frac = c - np.floor(c)
        if abs(frac - 0.5) < 0.1:
            return max(1, int(np.round(2.0 * c + 1.0)))
        return max(1, int(np.round(2.0 * c)))

    return center_to_size(cy), center_to_size(cx)
