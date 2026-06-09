from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

from ..utils.logger import rprint as print

_warned_no_depth: set[str] = set()


def depth_to_pointcloud(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project depth (HxW) to Nx3 points in camera frame."""
    H, W = depth.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth.ravel()
    x = (xs.ravel() - K[0, 2]) * z / K[0, 0]
    y = (ys.ravel() - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1)


def subsample_points(points: np.ndarray, n_points: int, rng: np.random.RandomState | None = None) -> np.ndarray:
    if len(points) <= n_points:
        return points
    rng = rng or np.random.RandomState(0)
    return points[rng.choice(len(points), size=n_points, replace=False)]


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Simple voxel grid downsample using spatial hashing."""
    if voxel_size <= 0.0:
        return points
    v = np.floor(points / voxel_size).astype(np.int64)
    h = v[:, 0] * 73856093 ^ v[:, 1] * 19349663 ^ v[:, 2] * 83492791
    _, unique_idx = np.unique(h, return_index=True)
    return points[np.sort(unique_idx)]


def pad_points_by_interpolation(points: np.ndarray, target_n: int, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Add interpolated points between existing pairs to reach target_n."""
    pts = points.astype(np.float32, copy=False)
    n = pts.shape[0]
    if target_n <= 0:
        return pts
    if n == 0:
        return np.zeros((target_n, 3), dtype=np.float32)
    if n >= target_n:
        return subsample_points(pts, target_n)

    k = target_n - n
    rng = rng or np.random.RandomState(0)
    i_idx = rng.randint(0, n, size=k)
    j_idx = rng.randint(0, n, size=k)
    same = i_idx == j_idx
    j_idx[same] = (j_idx[same] + 1) % n
    alpha = rng.rand(k, 1).astype(np.float32)
    new_pts = (1.0 - alpha) * pts[i_idx] + alpha * pts[j_idx]
    return np.concatenate([pts, new_pts], axis=0)


def depth_to_world_pointcloud(depth: np.ndarray, K: np.ndarray, T_c_w0: np.ndarray, n_points: int) -> np.ndarray:
    """Backproject depth and transform to world coordinates."""
    pts_c = depth_to_pointcloud(depth.astype(np.float32), K.astype(np.float32))
    R, t = T_c_w0[:3, :3].astype(np.float32), T_c_w0[:3, 3].astype(np.float32)
    pts_w = (pts_c @ R.T) + t
    return subsample_points(pts_w.astype(np.float32), n_points)


def maybe_build_scene_pointcloud(
    video_id: str,
    has_depth: bool,
    depth_src: Optional[Union[np.ndarray, Callable[[int], np.ndarray]]],
    k0: int,
    K: np.ndarray,
    T_c_w_seq: np.ndarray,
    n_points: int,
    verbose: bool = False,
    *,
    downsample_cfg: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """Build scene point cloud in world at frame k0 if depth available."""
    if not (has_depth and depth_src is not None):
        if video_id not in _warned_no_depth:
            if verbose:
                print(f"[yellow]depth[/yellow] missing for {video_id}; scene_pcd0=None")
            _warned_no_depth.add(video_id)
        return None

    depth0 = depth_src(k0) if callable(depth_src) else depth_src[k0]
    pcd = depth_to_world_pointcloud(depth0, K, T_c_w_seq[k0], n_points)

    if isinstance(downsample_cfg, dict) and downsample_cfg.get("enabled"):
        method = downsample_cfg.get("method", "voxel")
        if method == "voxel":
            pcd = voxel_downsample(pcd, float(downsample_cfg.get("voxel_size", 0.01)))
        elif method == "random":
            pcd = subsample_points(pcd, int(downsample_cfg.get("target_n", n_points)))

        tgt = int(downsample_cfg.get("target_n", 0))
        if tgt > 0:
            if pcd.shape[0] > tgt:
                pcd = subsample_points(pcd, tgt)
            elif pcd.shape[0] < tgt:
                pcd = pad_points_by_interpolation(pcd, tgt)

    if verbose:
        print(f"[blue]PCD[/blue] {video_id} k0={k0} • depth→XYZ world • N={pcd.shape[0]}")
    return pcd
