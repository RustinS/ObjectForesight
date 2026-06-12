from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from ..geom.se3_ops import matrix_info, read_pose_4x4
from ..utils.logger import rprint as print

# FP adapter config for ingest-time transforms
_FP_ADAPTER: Dict[str, bool] = {
    "transpose_on_load": False,
    "inverse_pose": False,
    "flip_y": False,
    "flip_z": False,
}


def set_fp_adapter(**kwargs) -> None:
    for k, v in kwargs.items():
        if k in _FP_ADAPTER:
            _FP_ADAPTER[k] = bool(v)


def get_fp_adapter() -> Dict[str, bool]:
    return dict(_FP_ADAPTER)


def _apply_fp_adapter(M: np.ndarray) -> np.ndarray:
    A = _FP_ADAPTER
    T = M.copy().astype(np.float32)

    if A["transpose_on_load"]:
        T = T.T
    if A["inverse_pose"]:
        R, t = T[:3, :3], T[:3, 3]
        Rinv = R.T
        T[:3, :3], T[:3, 3] = Rinv, -Rinv @ t
    if A["flip_y"]:
        T[:3, 1] *= -1.0
        T[1, 3] *= -1.0
    if A["flip_z"]:
        T[:3, 2] *= -1.0
        T[2, 3] *= -1.0

    T[3] = [0.0, 0.0, 0.0, 1.0]

    # Re-orthonormalize if needed
    R = T[:3, :3]
    RtR = R.T @ R
    if not np.allclose(RtR, np.eye(3), atol=1e-6) or np.linalg.det(R) < 0.999:
        U, _, Vt = np.linalg.svd(R.astype(np.float64))
        R_ortho = (U @ Vt).astype(np.float32)
        if np.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = (U @ Vt).astype(np.float32)
        T[:3, :3] = R_ortho
    return T


def _read_pose_txt(path: str, raw: bool = False) -> np.ndarray:
    M = read_pose_4x4(path)
    if not raw:
        M = _apply_fp_adapter(M)
    info = matrix_info(M)
    if not info["last_row_ok"] or abs(info["detR"] - 1.0) > 5e-3 or info["ortho_err"] > 5e-3:
        raise ValueError(f"Malformed pose in {path}: detR={info['detR']:.6f} ortho_err={info['ortho_err']:.2e}")
    return M


def _frame_id_from_path(p: str) -> int:
    stem = os.path.splitext(os.path.basename(p))[0]
    try:
        return int(stem)
    except ValueError:
        return 10**9


def load_object_poses(ob_in_cam_dir: str, verbose: bool = False) -> Dict[str, np.ndarray]:
    """Read FoundationPose ob_in_cam/*.txt and return frame_ids (T,) and T_c_o (T,4,4)."""
    empty = {"frame_ids": np.zeros((0,), dtype=np.int32), "T_c_o": np.zeros((0, 4, 4), dtype=np.float32)}
    _npz = os.path.join(os.path.dirname(ob_in_cam_dir), "poses.npz")
    if os.path.isfile(_npz):
        with np.load(_npz) as z:
            return {"frame_ids": z["frame_ids"].astype(np.int32), "T_c_o": z["T_c_o"].astype(np.float32)}
    if not os.path.isdir(ob_in_cam_dir):
        return empty

    txts = sorted(
        [os.path.join(ob_in_cam_dir, f) for f in os.listdir(ob_in_cam_dir) if f.lower().endswith(".txt")],
        key=_frame_id_from_path,
    )
    if not txts:
        return empty

    frame_ids: List[int] = []
    mats: List[np.ndarray] = []
    for p in txts:
        frame_ids.append(_frame_id_from_path(p))
        mats.append(_read_pose_txt(p))

    out = {"frame_ids": np.array(frame_ids, dtype=np.int32), "T_c_o": np.stack(mats, axis=0).astype(np.float32)}

    if verbose:
        obj_name = ob_in_cam_dir.split("/")[-3]
        print(f"[green]Obj poses[/green] {obj_name} • frames={out['frame_ids'].shape[0]} • dir={ob_in_cam_dir}")
    return out


def read_raw_pose_for_frame(ob_in_cam_dir: str, frame_id: int) -> Optional[np.ndarray]:
    """Read a single raw FP pose 4x4 (no adapter) for a frame id if exists."""
    p = os.path.join(ob_in_cam_dir, f"{int(frame_id)}.txt")
    if not os.path.exists(p):
        _npz = os.path.join(os.path.dirname(ob_in_cam_dir), "poses.npz")
        if os.path.isfile(_npz):
            with np.load(_npz) as z:
                _i = np.where(z["frame_ids"] == int(frame_id))[0]
                return z["T_c_o"][int(_i[0])].astype(np.float32) if len(_i) else None
        return None
    return _read_pose_txt(p, raw=True)
