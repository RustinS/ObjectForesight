from __future__ import annotations

from typing import Dict

import numpy as np
from torch.utils.data import Dataset


class SceneSequenceDatasetSynth(Dataset):
    """Synthetic dataset for testing: generates random images, intrinsics, point clouds, and pose targets."""

    def __init__(self, H: int, n_points: int, num_samples: int = 8, image_size=(240, 320), context_len: int = 1) -> None:
        self.H = H
        self.n_points = n_points
        self.num_samples = num_samples
        self.image_size = tuple(image_size)
        self.context_len = context_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        h, w = self.image_size
        image0 = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        fx = fy = 300.0
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        T_wc0 = np.eye(4, dtype=np.float32)
        pointcloud0 = (np.random.randn(self.n_points, 3) * 0.5).astype(np.float32)
        # Synthetic color from normalized coords
        mins = pointcloud0.min(axis=0, keepdims=True)
        ptp = np.maximum(pointcloud0.max(axis=0, keepdims=True) - mins, 1e-6)
        color0 = ((pointcloud0 - mins) / ptp).astype(np.float32)
        # Synthetic normals via PCA on k-NN
        diffs = pointcloud0[:, None, :] - pointcloud0[None, :, :]
        d2 = (diffs * diffs).sum(-1)
        k = min(16, self.n_points)
        knn_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        normals = np.zeros_like(pointcloud0)
        for i in range(self.n_points):
            nbrs = pointcloud0[knn_idx[i]] - pointcloud0[i]
            cov = (nbrs.T @ nbrs) / float(k)
            eigvals, eigvecs = np.linalg.eigh(cov)
            n = eigvecs[:, 0]
            normals[i] = (n / (np.linalg.norm(n) + 1e-6)).astype(np.float32)
        normal0 = normals.astype(np.float32)
        # Simple rectangular object mask and bbox
        x1, y1 = int(w * 0.3), int(h * 0.3)
        x2, y2 = int(w * 0.7), int(h * 0.7)
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True
        t0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        rot6d0 = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        target_future = (np.random.randn(self.H, 9) * 0.01).astype(np.float32)
        P = self.context_len
        context_init_9d = (np.random.randn(P, 9) * 0.01).astype(np.float32)
        context_bbox_norm = np.tile(np.array([x1 / w, y1 / h, x2 / w, y2 / h], dtype=np.float32), (P, 1))
        return {
            "image0": image0,
            "K": K,
            "T_wc0": T_wc0,
            "pointcloud0": pointcloud0,
            "color0": color0,
            "normal0": normal0,
            "depth0": np.zeros((h, w), dtype=np.float32),
            "object": {
                "mask": mask,
                "bbox": np.array([float(x1), float(y1), float(x2), float(y2)], dtype=np.float32),
                "class_id": -1,
                "mesh_path": "",
            },
            "init_pose": {"t0": t0, "rot6d0": rot6d0},
            "target_future": target_future,
            "context_len": P,
            "context_frame_ids": np.arange(P, dtype=np.int32),
            "context_T_cam_anchor_obj": np.tile(np.eye(4, dtype=np.float32), (P, 1, 1)),
            "context_init_9d": context_init_9d,
            "context_bbox_norm": context_bbox_norm,
        }
