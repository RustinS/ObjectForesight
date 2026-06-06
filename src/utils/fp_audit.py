from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from ..data.fpose_io import get_fp_adapter, read_raw_pose_for_frame, set_fp_adapter
from .geometry import project_points
from .validate import K_signature
from .viz import overlay_axes_on_image


def _geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Return geodesic angle in degrees between two 3x3 rotations."""
    M = Ra.T @ Rb
    tr = np.clip((np.trace(M) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def _project_basis_px(K: np.ndarray, T_cam_obj: np.ndarray, axis_length: float = 0.1) -> np.ndarray:
    """Project origin and axis endpoints. Returns (4,2) pixel coords."""
    pts = torch.tensor([[0.0, 0.0, 0.0], [axis_length, 0.0, 0.0], [0.0, axis_length, 0.0], [0.0, 0.0, axis_length]], dtype=torch.float32)
    ones = torch.ones((4, 1), dtype=torch.float32)
    Pw = torch.cat([pts, ones], dim=-1).T
    Tt = torch.from_numpy(T_cam_obj.astype(np.float32))
    Pc = (Tt @ Pw)[:3].T[..., :3]
    uv = project_points(torch.from_numpy(K.astype(np.float32)), torch.eye(4, dtype=torch.float32), Pc)
    return uv.detach().cpu().numpy()


def _handedness_signature(K: np.ndarray, T_cam_obj: np.ndarray) -> Dict[str, bool | float]:
    uv = _project_basis_px(K, T_cam_obj)
    o, x, y = uv[0], uv[1], uv[2]
    detR = float(np.linalg.det(T_cam_obj[:3, :3]))
    return {
        "x_right": bool((x[0] - o[0]) > 0.0),
        "y_down": bool((y[1] - o[1]) > 0.0),
        "z_forward": bool(T_cam_obj[2, 3] > 0.0),
        "detR>0": bool(detR > 0.0),
        "detR": detR,
    }


def _save_axis_overlay(img: np.ndarray, K: np.ndarray, T_cam_obj: np.ndarray, out_path: str, axis_length: float = 0.1) -> None:
    T = torch.from_numpy(T_cam_obj.astype(np.float32)).unsqueeze(0)
    canvas = overlay_axes_on_image(img, K, np.eye(4, dtype=np.float32), T, axis_length=axis_length)
    cv2.imwrite(out_path, canvas)


def fp_signature_block(ob_in_cam_dir: str, frame_id: int, K: np.ndarray) -> Tuple[str, Optional[np.ndarray]]:
    """Build a one-page FP signature block; returns block text and raw pose used."""
    M_raw = read_raw_pose_for_frame(ob_in_cam_dir, frame_id)
    if M_raw is None:
        return ("FP signature: no raw pose found", None)
    R, t = M_raw[:3, :3], M_raw[:3, 3]
    pose_type = "T_cam_obj"
    if float(M_raw[2, 3]) <= 0.0 and float((R.T @ (-R.T @ t))[2]) > 0.0:
        pose_type = "T_obj_cam?"
    handed = _handedness_signature(K, M_raw)
    block = (
        "FP signature:\n"
        f"- pose_type: {pose_type}\n"
        f"- storage: row-major; no transpose on load\n"
        f"- units: meters\n"
        f"- handedness: x_right={handed['x_right']} y_down={handed['y_down']} z_forward={handed['z_forward']} det(R)>0={handed['detR>0']}\n"
        f"- axis colors: X=R, Y=G, Z=B\n"
        f"- K used for FP render test: {K_signature(K)}\n"
    )
    return block, M_raw


def compare_ours_vs_fp(K: np.ndarray, T_ours_cam: np.ndarray, T_fp_raw: np.ndarray, img_size: Tuple[int, int]) -> Dict[str, object]:
    """Compute angular and pixel differences for a single frame."""
    R_ours, R_fp = T_ours_cam[:3, :3], T_fp_raw[:3, :3]
    ang = _geodesic_deg(R_ours, R_fp)
    uv_ours, uv_fp = _project_basis_px(K, T_ours_cam), _project_basis_px(K, T_fp_raw)
    diffs = np.linalg.norm(uv_ours - uv_fp, axis=-1)
    dots = [float(np.dot(R_ours[:, i], R_fp[:, i])) for i in range(3)]
    return {
        "rot_diff_deg": float(ang),
        "px_diff": {"origin": float(diffs[0]), "X": float(diffs[1]), "Y": float(diffs[2]), "Z": float(diffs[3])},
        "axis_flip": {"X": dots[0] < 0.0, "Y": dots[1] < 0.0, "Z": dots[2] < 0.0},
        "dots": {"X": dots[0], "Y": dots[1], "Z": dots[2]},
    }


def propose_adapter_from_tests(K: np.ndarray, T_fp_raw: np.ndarray) -> Dict[str, bool]:
    """Heuristic minimal adapter proposal from raw pose tests."""
    h = _handedness_signature(K, T_fp_raw)
    adapter = get_fp_adapter()
    if not h["y_down"]:
        adapter["flip_y"] = True
    if not h["z_forward"]:
        adapter["flip_z"] = True
        R, t = T_fp_raw[:3, :3], T_fp_raw[:3, 3]
        Tinv_z = float((-R.T @ t)[2])
        if Tinv_z > 0.0:
            adapter["inverse_pose"] = True
    adapter["transpose_on_load"] = False
    return adapter


def apply_adapter(adapter: Dict[str, bool]) -> None:
    set_fp_adapter(**adapter)


def save_fp_and_ours_overlays(image0: np.ndarray, K: np.ndarray, T_ours_cam: np.ndarray, T_fp_raw: np.ndarray, out_dir: str, axis_length: float = 0.1) -> Tuple[str, str]:
    fp_path = os.path.join(out_dir, "fp_axis_raw.png")
    ours_path = os.path.join(out_dir, "ours_axis.png")
    _save_axis_overlay(image0, K, T_fp_raw, fp_path, axis_length=axis_length)
    _save_axis_overlay(image0, K, T_ours_cam, ours_path, axis_length=axis_length)
    return fp_path, ours_path


def build_audit_report(
    pose_type_before: str,
    pose_type_after: str,
    storage_transposed: bool,
    axis_fix: str,
    applied_inverse: bool,
    K: np.ndarray,
    two_path_rot_err: float,
    two_path_trans_err: float,
    px_errs: Dict[str, float],
    verdict: str,
) -> str:
    o = px_errs.get("origin", float("nan"))
    x = px_errs.get("X", float("nan"))
    y = px_errs.get("Y", float("nan"))
    z = px_errs.get("Z", float("nan"))
    return (
        "FP Audit Report:\n"
        f"  pose_type({pose_type_before})->({pose_type_after})\n"
        f"  storage(transpose?): {'yes' if storage_transposed else 'no'}\n"
        f"  axis_fix: {axis_fix}\n"
        f"  applied_inverse: {'yes' if applied_inverse else 'no'}\n"
        f"  K_sig: {K_signature(K)}\n"
        f"  two_path_error: rot={two_path_rot_err:.4g}°, trans={two_path_trans_err:.4g}\n"
        f"  axis_px_error(origin, X, Y, Z): ({o:.2f}, {x:.2f}, {y:.2f}, {z:.2f})\n"
        f"  final verdict: {verdict}\n"
    )
