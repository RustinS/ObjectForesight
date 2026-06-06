from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np
import torch

from .geometry import project_points


def overlay_axes_on_image(
    img: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    poses_c: torch.Tensor,
    axis_length: float = 0.1,
    annotate_indices: bool = False,
    traj_labels: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Overlay predicted axes on image for each pose in camera frame. poses_c: (H, 4, 4)"""
    out = img.copy()
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)

    dev = torch.device("cpu")
    poses_cpu = poses_c.to(dev)
    K_t = torch.from_numpy(K).float().to(dev)
    T_wc_t = torch.from_numpy(T_wc).float().to(dev)
    Himg, Wimg = out.shape[:2]

    origin = torch.tensor([[0.0, 0.0, 0.0]], device=dev)
    axes = torch.eye(3, device=dev) * axis_length
    pts = torch.cat([origin, axes], dim=0)  # 4x3
    ones = torch.ones((4, 1), dtype=torch.float32, device=dev)
    Pw = torch.cat([pts, ones], dim=-1)  # (4,4)

    prev_o = None
    for k in range(poses_cpu.shape[0]):
        T = poses_cpu[k]
        Pc_w = (T @ Pw.T)[:3].T[..., :3]
        uv = project_points(K_t, T_wc_t, Pc_w).detach().cpu().numpy()

        def to_point(a: np.ndarray) -> tuple[int, int]:
            return (int(np.clip(np.round(a[0]), 0, Wimg - 1)), int(np.clip(np.round(a[1]), 0, Himg - 1)))

        o, x, y, z = to_point(uv[0]), to_point(uv[1]), to_point(uv[2]), to_point(uv[3])
        cv2.circle(out, o, 3, (255, 255, 255), -1)
        cv2.line(out, o, x, (0, 0, 255), 2)
        cv2.line(out, o, y, (0, 255, 0), 2)
        cv2.line(out, o, z, (255, 0, 0), 2)

        if isinstance(traj_labels, (list, tuple)) and k < len(traj_labels) and traj_labels[k]:
            px = (int(o[0] + 6), int(o[1] - 6))
            text = str(traj_labels[k])
            cv2.putText(out, text, px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, text, px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if prev_o is not None:
            cv2.line(out, prev_o, o, (255, 255, 255), 2)
        prev_o = o
    return out
