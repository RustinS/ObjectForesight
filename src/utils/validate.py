from __future__ import annotations

import math
from typing import Dict, Tuple

import cv2
import numpy as np
import torch

from ..geom.se3_ops import (
    _det3x3_dtype_safe,
    _ensure_homogeneous,
    _svd_dtype_safe,
    compose_T,
    geodesic_distance_deg,
    invert_T,
    rot6d_to_matrix,
    se3_to_matrix,
)
from .geometry import infer_image_size_from_K, project_points


def check_se3_hygiene(Ts: torch.Tensor, name: str = "T") -> Dict[str, float]:
    """Return simple hygiene metrics; raises on NaN/Inf or wrong shape."""
    Ts = _ensure_homogeneous(Ts)
    if not torch.isfinite(Ts).all():
        raise ValueError(f"{name} contains NaN/Inf")
    R = Ts[:, :3, :3]
    RtR = torch.matmul(R.transpose(-1, -2), R)
    eye3 = torch.eye(3, dtype=R.dtype, device=R.device).unsqueeze(0)
    ortho_err = (RtR - eye3).abs().amax(dim=(1, 2))
    U, S, V = _svd_dtype_safe(R)
    det = _det3x3_dtype_safe(torch.matmul(U, V.transpose(-1, -2)))
    det_err = (det - 1.0).abs()
    return {"ortho_err_max": float(ortho_err.max().item()), "det_err_max": float(det_err.max().item())}


def reexpress_obj_to_anchor_via_world(T_c_w_win: torch.Tensor, T_c_o_win: torch.Tensor, anchor_idx: int) -> torch.Tensor:
    """Path A: world composition. Returns T_cam_anchor_obj_k for all k."""
    T_c_w, T_c_o = _ensure_homogeneous(T_c_w_win), _ensure_homogeneous(T_c_o_win)
    T_w_c = invert_T(T_c_w)
    T_w_o = compose_T(T_w_c, T_c_o)
    T_c_w_anchor = T_c_w[anchor_idx : anchor_idx + 1].repeat(T_c_w.shape[0], 1, 1)
    return compose_T(T_c_w_anchor, T_w_o)


def pose_errors_deg_m(T_pred: torch.Tensor, T_gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (rot_err_deg, trans_err_m) per frame."""
    T_pred, T_gt = _ensure_homogeneous(T_pred), _ensure_homogeneous(T_gt)
    Rp, tp = T_pred[:, :3, :3], T_pred[:, :3, 3]
    Rg, tg = T_gt[:, :3, :3], T_gt[:, :3, 3]
    return geodesic_distance_deg(Rp, Rg), torch.linalg.norm(tp - tg, dim=-1)


def rel_pose(T_next: torch.Tensor, T_prev: torch.Tensor) -> torch.Tensor:
    """Relative transform: T_prev^{-1} * T_next."""
    return compose_T(invert_T(T_prev), T_next)


def ate_rpe(T_pred: torch.Tensor, T_gt: torch.Tensor) -> Dict[str, float]:
    """Compute simple window-local ATE and RPE."""
    rot_err, trans_err = pose_errors_deg_m(T_pred, T_gt)
    H = T_pred.shape[0]
    if H >= 2:
        rpe_rot, rpe_trans = pose_errors_deg_m(rel_pose(T_pred[1:], T_pred[:-1]), rel_pose(T_gt[1:], T_gt[:-1]))
    else:
        rpe_rot = rpe_trans = torch.zeros(1, dtype=rot_err.dtype)
    return {
        "ate_rot_mean_deg": float(rot_err.mean().item()),
        "ate_trans_mean": float(trans_err.mean().item()),
        "rpe_rot_mean_deg": float(rpe_rot.mean().item()),
        "rpe_trans_mean": float(rpe_trans.mean().item()),
    }


def reprojection_error_px(K: np.ndarray, T_pred_cam: torch.Tensor, T_gt_cam: torch.Tensor, axis_length: float = 0.1) -> float:
    """Compute mean reprojection error (px) using canonical object points."""
    if K is None or not isinstance(K, np.ndarray) or K.shape != (3, 3):
        return float("nan")
    T_pred, T_gt = _ensure_homogeneous(T_pred_cam), _ensure_homogeneous(T_gt_cam)
    H = min(T_pred.shape[0], T_gt.shape[0])
    if H == 0:
        return float("nan")
    K_t = torch.from_numpy(K.astype(np.float32)).to(T_pred.device)
    pts = torch.tensor([[0.0, 0.0, 0.0], [axis_length, 0.0, 0.0], [0.0, axis_length, 0.0], [0.0, 0.0, axis_length]], dtype=T_pred.dtype, device=T_pred.device)
    errs = []
    for k in range(H):
        if T_pred[k, 2, 3] <= 1e-3 or T_gt[k, 2, 3] <= 1e-3:
            continue
        uv_p, uv_g = project_points(K_t, T_pred[k], pts), project_points(K_t, T_gt[k], pts)
        errs.append(float(torch.linalg.norm(uv_p - uv_g, dim=-1).mean().item()))
    return float(np.mean(errs)) if errs else float("nan")


def reprojection_error_norm(K: np.ndarray, T_pred_cam: torch.Tensor, T_gt_cam: torch.Tensor, axis_length: float = 0.1) -> float:
    """Compute mean reprojection error on normalized plane."""
    if K is None or not isinstance(K, np.ndarray) or K.shape != (3, 3):
        return float("nan")
    T_pred, T_gt = _ensure_homogeneous(T_pred_cam), _ensure_homogeneous(T_gt_cam)
    H = min(T_pred.shape[0], T_gt.shape[0])
    if H == 0:
        return float("nan")
    K_t = torch.from_numpy(K.astype(np.float32)).to(T_pred.device)
    fx, fy, cx, cy = float(K_t[0, 0]), float(K_t[1, 1]), float(K_t[0, 2]), float(K_t[1, 2])
    pts = torch.tensor([[0.0, 0.0, 0.0], [axis_length, 0.0, 0.0], [0.0, axis_length, 0.0], [0.0, 0.0, axis_length]], dtype=T_pred.dtype, device=T_pred.device)
    errs = []
    for k in range(H):
        if T_pred[k, 2, 3] <= 1e-3 or T_gt[k, 2, 3] <= 1e-3:
            continue
        uv_p, uv_g = project_points(K_t, T_pred[k], pts), project_points(K_t, T_gt[k], pts)
        n_p = torch.stack([(uv_p[..., 0] - cx) / max(fx, 1e-8), (uv_p[..., 1] - cy) / max(fy, 1e-8)], dim=-1)
        n_g = torch.stack([(uv_g[..., 0] - cx) / max(fx, 1e-8), (uv_g[..., 1] - cy) / max(fy, 1e-8)], dim=-1)
        errs.append(float(torch.linalg.norm(n_p - n_g, dim=-1).mean().item()))
    return float(np.mean(errs)) if errs else float("nan")


def K_signature(K: np.ndarray) -> str:
    """Return a concise signature string for intrinsics K."""
    if K is None or not isinstance(K, np.ndarray) or K.shape != (3, 3):
        return "K:none"
    return f"K(fx={K[0, 0]:.3f},fy={K[1, 1]:.3f},cx={K[0, 2]:.3f},cy={K[1, 2]:.3f})"


def depth_sanity(T_abs_cam: torch.Tensor, near: float = 1e-3) -> Dict[str, float]:
    """Return simple depth stats for absolute camera(anchor)<-object poses."""
    T_abs = _ensure_homogeneous(T_abs_cam)
    z = T_abs[:, 2, 3]
    return {"z_bad_count": float((z <= near).sum().item()), "z_min": float(z.min().item()), "z_median": float(z.median().item())}


def reprojection_sanity_origin_px(K: np.ndarray, T_pred_cam: torch.Tensor, T_gt_cam: torch.Tensor) -> float:
    """Single-point sanity: project object origin with GT vs Pred; return mean px error."""
    if K is None or not isinstance(K, np.ndarray) or K.shape != (3, 3):
        return float("nan")
    T_pred, T_gt = _ensure_homogeneous(T_pred_cam), _ensure_homogeneous(T_gt_cam)
    H = min(T_pred.shape[0], T_gt.shape[0])
    if H == 0:
        return float("nan")
    K_t = torch.from_numpy(K.astype(np.float32)).to(T_pred.device)
    pts = torch.zeros((1, 3), dtype=T_pred.dtype, device=T_pred.device)
    errs = []
    for k in range(H):
        if T_pred[k, 2, 3] <= 1e-3 or T_gt[k, 2, 3] <= 1e-3:
            continue
        uv_p, uv_g = project_points(K_t, T_pred[k], pts), project_points(K_t, T_gt[k], pts)
        errs.append(float(torch.linalg.norm(uv_p - uv_g, dim=-1).mean().item()))
    return float(np.mean(errs)) if errs else float("nan")


def compute_unit_metric_audit(
    T_pred_cam: torch.Tensor,
    T_gt_cam: torch.Tensor,
    K: np.ndarray | None,
    image_size: tuple[int, int] | None,
    pinfo: Dict | None = None,
    extr: Dict | None = None,
) -> Dict[str, float | str]:
    """Compute a compact set of unit/scale audit metrics for a window."""
    T_pred, T_gt = _ensure_homogeneous(T_pred_cam), _ensure_homogeneous(T_gt_cam)
    H = min(T_pred.shape[0], T_gt.shape[0])
    out: Dict[str, float | str] = {}
    tp = torch.linalg.norm(T_pred[:H, :3, 3], dim=-1)
    tg = torch.linalg.norm(T_gt[:H, :3, 3], dim=-1) + 1e-8
    out["t_gt_norm_mean"] = float(tg.mean().item())
    out["t_pred_norm_mean"] = float(tp.mean().item())
    out["ratio_med"] = float((tp / tg).median().item())
    if H >= 2:
        Tpr, Tgr = rel_pose(T_pred[1:H], T_pred[: H - 1]), rel_pose(T_gt[1:H], T_gt[: H - 1])
        dtp = torch.linalg.norm(Tpr[:, :3, 3], dim=-1)
        dtg = torch.linalg.norm(Tgr[:, :3, 3], dim=-1) + 1e-8
        out["step_ratio_med"] = float((dtp / dtg).median().item())
    out["K_sig"] = K_signature(K) if K is not None else "K:none"
    W = Hsz = None
    if K is not None:
        Hsz, W = infer_image_size_from_K(K)
    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        Hsz, W = int(image_size[0]), int(image_size[1])
    out["canvas_W"], out["canvas_H"] = float(W) if W else float("nan"), float(Hsz) if Hsz else float("nan")
    if K is not None:
        out["reproj_px_mean"] = reprojection_error_px(K, T_pred[:H], T_gt[:H])
        out["reproj_norm_mean"] = reprojection_error_norm(K, T_pred[:H], T_gt[:H])
    zs = depth_sanity(T_pred[:H])
    out.update({"z_pred_min": zs["z_min"], "z_pred_mean": float(T_pred[:H, 2, 3].mean().item()), "depth_bad_frames": zs["z_bad_count"]})
    out.update({"dir_corr_mean": float("nan"), "dir_corr_min": float("nan")})
    if isinstance(extr, dict) and extr.get("T_c_w") is not None and extr.get("T_c_o") is not None:
        T_c_w_win = torch.as_tensor(extr["T_c_w"]).float() if not isinstance(extr["T_c_w"], torch.Tensor) else extr["T_c_w"]
        T_c_o_win = torch.as_tensor(extr["T_c_o"]).float() if not isinstance(extr["T_c_o"], torch.Tensor) else extr["T_c_o"]
        ds = directionality_probe(T_c_w_win, T_c_o_win, T_gt, int(extr.get("anchor_local_idx", 0)), str(extr.get("extrinsics_convention", "c2w")))
        out["dir_corr_mean"], out["dir_corr_min"] = ds.get("dir_corr_mean", float("nan")), ds.get("dir_corr_min", float("nan"))
    out["pred_mode"] = str(pinfo.get("pred_mode", "?")) if pinfo else "?"
    out["parameterization"] = str(pinfo.get("pred_repr", "se3(t+rot6d)")) if pinfo else "se3(t+rot6d)"
    out["denorm_scale"] = float(pinfo.get("scale_correction_factor", 1.0)) if pinfo and pinfo.get("scale_correction_applied") else 1.0
    out["units"], out["applied_denorm_once"] = "meters", True
    return out


def build_diagnostics_block(
    T_pred_cam: torch.Tensor,
    T_gt_cam: torch.Tensor,
    K: np.ndarray | None,
    image_size: tuple[int, int] | None,
    pinfo: Dict | None = None,
    isolate_rt: str | None = None,
) -> Dict[str, float]:
    """Assemble a single diagnostics dict shared by infer & viz."""
    T_pred, T_gt = _ensure_homogeneous(T_pred_cam), _ensure_homogeneous(T_gt_cam)
    H = min(T_pred.shape[0], T_gt.shape[0])
    T_pred, T_gt = T_pred[:H], T_gt[:H]
    rot_err, trans_err = pose_errors_deg_m(T_pred, T_gt)
    stats = ate_rpe(T_pred, T_gt)
    reproj_px, reproj_norm = reprojection_error_px(K, T_pred, T_gt), reprojection_error_norm(K, T_pred, T_gt)
    reproj_skip = int(((T_pred[:, 2, 3] <= 1e-3) | (T_gt[:, 2, 3] <= 1e-3)).sum().item())
    zstats = depth_sanity(T_pred)
    audit = compute_unit_metric_audit(T_pred, T_gt, K, image_size, pinfo, None)
    eps = 1e-8
    vp, vg = T_pred[:, :3, 3], T_gt[:, :3, 3]
    den = torch.linalg.norm(vp, dim=-1) * torch.linalg.norm(vg, dim=-1) + eps
    cos = (vp * vg).sum(-1) / den
    mask = den > 1e-6
    cos_mean = float(cos[mask].mean().item()) if mask.any() else float("nan")
    cos_min = float(cos[mask].min().item()) if mask.any() else float("nan")
    T_predR_gtT = T_pred.clone()
    T_predR_gtT[:, :3, 3] = T_gt[:, :3, 3]
    rot_only_err, _ = pose_errors_deg_m(T_predR_gtT, T_gt)
    iso_predR = {
        "ATE(rot)_predR_gtT": float(rot_only_err.mean().item()),
        "reproj_px_mean_predR_gtT": reprojection_error_px(K, T_predR_gtT, T_gt),
        "z_bad_count_predR_gtT": float((T_predR_gtT[:, 2, 3] <= 1e-3).sum().item()),
    }
    T_gtR_predT = T_gt.clone()
    T_gtR_predT[:, :3, 3] = T_pred[:, :3, 3]
    _, trans_only_err = pose_errors_deg_m(T_gtR_predT, T_gt)
    iso_gtR = {
        "ATE(trans)_gtR_predT": float(trans_only_err.mean().item()),
        "reproj_px_mean_gtR_predT": reprojection_error_px(K, T_gtR_predT, T_gt),
        "z_bad_count_gtR_predT": float((T_gtR_predT[:, 2, 3] <= 1e-3).sum().item()),
    }
    if isinstance(isolate_rt, str):
        key = isolate_rt.strip().lower()
        if key == "predr_gtt":
            stats, reproj_px, reproj_norm = ate_rpe(T_predR_gtT, T_gt), reprojection_error_px(K, T_predR_gtT, T_gt), reprojection_error_norm(K, T_predR_gtT, T_gt)
            rot_err, trans_err = pose_errors_deg_m(T_predR_gtT, T_gt)
        elif key == "gtr_predt":
            stats, reproj_px, reproj_norm = ate_rpe(T_gtR_predT, T_gt), reprojection_error_px(K, T_gtR_predT, T_gt), reprojection_error_norm(K, T_gtR_predT, T_gt)
            rot_err, trans_err = pose_errors_deg_m(T_gtR_predT, T_gt)
    return {
        **stats,
        "rot_deg_err_mean": float(rot_err.mean().item()),
        "trans_l2_err_mean": float(trans_err.mean().item()),
        "reproj_px_mean": reproj_px,
        "reproj_norm_mean": reproj_norm,
        "reproj_skipped": float(reproj_skip),
        "cos_dir_t_mean": cos_mean,
        "cos_dir_t_min": cos_min,
        "z_bad_count_pred": zstats["z_bad_count"],
        "t_norm_ratio_med": audit.get("ratio_med", float("nan")),
        **iso_predR,
        **iso_gtR,
    }


def _rotation_from_cams(T_c_w_win: torch.Tensor, anchor_idx: int, convention: str) -> torch.Tensor:
    """Return per-k rotation R_cam_anchor_from_cam_k as (K,3,3)."""
    T_c_w = _ensure_homogeneous(T_c_w_win)
    if convention == "c2w":
        T_ci0_w = invert_T(T_c_w[anchor_idx : anchor_idx + 1])
        T_ci0_ck = compose_T(T_ci0_w.repeat(T_c_w.shape[0], 1, 1), T_c_w)
    else:
        T_ci0_w = T_c_w[anchor_idx : anchor_idx + 1]
        T_ci0_ck = compose_T(T_ci0_w.repeat(T_c_w.shape[0], 1, 1), invert_T(T_c_w))
    return T_ci0_ck[:, :3, :3]


def directionality_probe(T_c_w_win: torch.Tensor, T_c_o_win: torch.Tensor, T_cam_anchor_obj: torch.Tensor, anchor_idx: int, convention: str = "c2w") -> Dict[str, float]:
    """Compare motion direction before/after re-expression."""
    T_c_w, T_c_o = _ensure_homogeneous(T_c_w_win), _ensure_homogeneous(T_c_o_win)
    T_ci0_ok = _ensure_homogeneous(T_cam_anchor_obj)
    K = T_c_o.shape[0]
    if K < 2:
        return {"dir_corr_mean": float("nan"), "dir_corr_min": float("nan"), "used_pairs": 0.0}
    R_ai = _rotation_from_cams(T_c_w, anchor_idx, convention)
    v_cam = T_c_o[1:, :3, 3] - T_c_o[:-1, :3, 3]
    v_cam_in_anchor = torch.einsum("kij,kj->ki", R_ai[:-1], v_cam)
    v_anchor = T_ci0_ok[1:, :3, 3] - T_ci0_ok[:-1, :3, 3]
    eps = 1e-8
    num = (v_cam_in_anchor * v_anchor).sum(-1)
    den = torch.linalg.norm(v_cam_in_anchor, dim=-1) * torch.linalg.norm(v_anchor, dim=-1) + eps
    cos = num / den
    mask = den > 1e-6
    if not mask.any():
        return {"dir_corr_mean": float("nan"), "dir_corr_min": float("nan"), "used_pairs": 0.0}
    cos_v = cos[mask]
    return {"dir_corr_mean": float(cos_v.mean().item()), "dir_corr_min": float(cos_v.min().item()), "used_pairs": float(mask.sum().item())}


def run_gt_pose_checks(sample: Dict, logger, atol_rot_deg: float = 0.2, atol_trans: float = 1e-3) -> Dict[str, float]:
    """Run two-path equivalence and anchor self-consistency for GT cameras."""
    T_c_w_np, T_c_o_np, T_cam_anchor_obj_np = sample.get("T_c_w"), sample.get("T_c_o"), sample.get("T_cam_anchor_obj")
    anchor_local = int(sample.get("anchor_local_idx", 0))
    if T_c_w_np is None or T_c_o_np is None or T_cam_anchor_obj_np is None:
        return {"skipped": 1.0}
    T_c_w, T_c_o = torch.from_numpy(T_c_w_np).float(), torch.from_numpy(T_c_o_np).float()
    T_cam_anchor_obj = torch.from_numpy(T_cam_anchor_obj_np).float()
    metrics_ex = check_se3_hygiene(T_c_w, name="T_c_w")
    metrics_gt = check_se3_hygiene(T_cam_anchor_obj, name="T_cam_anchor_obj")
    conv = str(sample.get("extrinsics_convention", "c2w"))
    logger(f"[bold]extrinsics_convention[/bold]={conv} • ortho_max={metrics_ex['ortho_err_max']:.2e} det_err_max={metrics_ex['det_err_max']:.2e}")
    logger(f"[bold]gt_hygiene[/bold] • ortho_max={metrics_gt['ortho_err_max']:.2e} det_err_max={metrics_gt['det_err_max']:.2e}")
    Ta = reexpress_obj_to_anchor_via_world(T_c_w, T_c_o, anchor_local)
    T_c_w_anchor = T_c_w[anchor_local : anchor_local + 1]
    Tb = compose_T(compose_T(T_c_w_anchor.repeat(T_c_w.shape[0], 1, 1), invert_T(T_c_w)), T_c_o)
    rot_err_ab, trans_err_ab = pose_errors_deg_m(Ta, Tb)
    rot_err_ag, trans_err_ag = pose_errors_deg_m(Ta, T_cam_anchor_obj)
    Ta_anchor, Tk_anchor = Ta[anchor_local], T_c_o[anchor_local]
    rot_anchor = geodesic_distance_deg(Ta_anchor[:3, :3].unsqueeze(0), Tk_anchor[:3, :3].unsqueeze(0))[0]
    trans_anchor = torch.linalg.norm(Ta_anchor[:3, 3] - Tk_anchor[:3, 3])
    logger(f"[bold]GT check[/bold] • two-path max: rot={rot_err_ab.max():.3g}° trans={trans_err_ab.max():.3g}")
    logger(f"  vs-ds max: rot={rot_err_ag.max():.3g}° trans={trans_err_ag.max():.3g} • anchor: rot={float(rot_anchor):.3g}° trans={float(trans_anchor):.3g}")
    assert float(rot_err_ab.max()) <= max(1e-3, atol_rot_deg), "Two-path rot mismatch"
    assert float(trans_err_ab.max()) <= max(1e-6, atol_trans), "Two-path trans mismatch"
    assert float(rot_anchor) <= max(1e-3, atol_rot_deg), "Anchor rot mismatch"
    assert float(trans_anchor) <= max(1e-6, atol_trans), "Anchor trans mismatch"
    i1 = T_c_w.shape[0] - 1 if T_c_w.shape[0] > 1 else anchor_local
    T_cam_i1_o = reexpress_obj_to_anchor_via_world(T_c_w, T_c_o, i1)
    T_c_i1_c_i0 = compose_T(T_c_w[i1 : i1 + 1].repeat(T_c_w.shape[0], 1, 1), invert_T(T_c_w[anchor_local : anchor_local + 1].repeat(T_c_w.shape[0], 1, 1)))
    rot_err_re, trans_err_re = pose_errors_deg_m(T_cam_i1_o, compose_T(T_c_i1_c_i0, Ta))
    logger(f"[bold]re-anchor[/bold] • rot_max={rot_err_re.max():.3g}° rot_mean={rot_err_re.mean():.3g}°")
    logger(f"  trans_max={trans_err_re.max():.3g} trans_mean={trans_err_re.mean():.3g}")
    return {
        "two_path_rot_max_deg": float(rot_err_ab.max().item()),
        "two_path_trans_max": float(trans_err_ab.max().item()),
        "anchor_rot_deg": float(rot_anchor.item()),
        "anchor_trans": float(trans_anchor.item()),
        "reanchor_rot_max_deg": float(rot_err_re.max().item()),
        "reanchor_trans_max": float(trans_err_re.max().item()),
    }


def _cumprod_T(T_rel: torch.Tensor) -> torch.Tensor:
    """Left-multiply cumulative product over time."""
    T_rel = _ensure_homogeneous(T_rel)
    out = [T_rel[0]]
    for k in range(1, T_rel.shape[0]):
        out.append(out[-1] @ T_rel[k])
    return torch.stack(out, dim=0)


def _adjoint_compose_deltas_from_prev_cam(T_c_w_win: torch.Tensor, T_deltas_prev_cam: torch.Tensor, T_gt_cam: torch.Tensor, anchor_idx: int) -> torch.Tensor:
    """Compose deltas expressed in camera(k-1) into absolute poses in anchor camera using adjoint."""
    T_c_w, T_prev = _ensure_homogeneous(T_c_w_win), _ensure_homogeneous(T_deltas_prev_cam)
    T_seed = _ensure_homogeneous(T_gt_cam[:1])
    H = T_prev.shape[0]
    out, cur = [], T_seed
    for s in range(H):
        T_camA_camPrev = compose_T(T_c_w[anchor_idx : anchor_idx + 1], invert_T(T_c_w[s : s + 1]))
        Delta_anchor = compose_T(compose_T(T_camA_camPrev, T_prev[s : s + 1]), invert_T(T_camA_camPrev))
        cur = compose_T(Delta_anchor, cur)
        out.append(cur[0])
    return torch.stack(out, dim=0)


def _to_world_from_cam_series(T_c_w_win: torch.Tensor, T_c_o_series: torch.Tensor) -> torch.Tensor:
    """Compute world<-object series: T_w_o = inv(T_c_w) @ T_c_o."""
    return compose_T(invert_T(T_c_w_win), T_c_o_series)


def _to_T_cam_anchor_from_world_series(sample: Dict, T_world_obj: torch.Tensor) -> torch.Tensor:
    """Re-express world<-object to camera(anchor)<-object using provided extrinsics."""
    T_c_w_win, anchor_idx = sample.get("T_c_w"), sample.get("anchor_local_idx")
    if T_c_w_win is not None and anchor_idx is not None and 0 <= anchor_idx < np.asarray(T_c_w_win).shape[0]:
        T_c_w_t = torch.from_numpy(np.asarray(T_c_w_win)).float()
        T_w_c_anchor = invert_T(T_c_w_t)[anchor_idx : anchor_idx + 1].repeat(T_world_obj.shape[0], 1, 1)
        return compose_T(T_w_c_anchor, T_world_obj)
    T_wc0_np = sample.get("T_wc0")
    if T_wc0_np is None:
        raise ValueError("sample missing T_c_w or T_wc0 for re-expression")
    T_wc0 = torch.from_numpy(np.asarray(T_wc0_np)).float().unsqueeze(0).repeat(T_world_obj.shape[0], 1, 1)
    return compose_T(T_wc0, T_world_obj)


def normalize_pred_to_cam_i0(pred: np.ndarray, sample: Dict, mode_hint: str | None = None) -> Tuple[torch.Tensor, Dict]:
    """Normalize model outputs to camera(anchor)<-object series and infer interpretation."""
    info: Dict = {}
    H = pred.shape[0]
    if pred.ndim == 2 and pred.shape[1] == 9:
        t_raw = torch.from_numpy(pred[:, :3]).float()
        R_raw = rot6d_to_matrix(torch.from_numpy(pred[:, 3:9]).float())
        T_series = se3_to_matrix(t_raw, R_raw)
        hyg = check_se3_hygiene(T_series, name="T_pred_raw")
        info.update({"ortho_max_pred": hyg["ortho_err_max"], "det_err_max_pred": hyg["det_err_max"], "pred_repr": "se3(t+rot6d)"})
        t_denorm = t_raw.clone()
    elif pred.ndim == 2 and pred.shape[1] == 3:
        T_gt_cam = torch.from_numpy(np.asarray(sample.get("T_cam_anchor_obj"))).float()
        t_raw = torch.from_numpy(pred[:, :3]).float()
        t_denorm = t_raw.clone()
        T_series = se3_to_matrix(t_denorm, T_gt_cam[:, :3, :3])
        info["pred_repr"] = "translation_only(+GT_R)"
    else:
        raise ValueError(f"Unsupported pred shape {pred.shape}")

    T_c_w = torch.from_numpy(np.asarray(sample.get("T_c_w"))).float() if sample.get("T_c_w") is not None else None
    T_c_o = torch.from_numpy(np.asarray(sample.get("T_c_o"))).float() if sample.get("T_c_o") is not None else None
    T_gt_cam = torch.from_numpy(np.asarray(sample.get("T_cam_anchor_obj"))).float() if sample.get("T_cam_anchor_obj") is not None else None
    anchor_idx = int(sample.get("anchor_local_idx", 0))

    # Scale correction
    if T_gt_cam is not None and T_gt_cam.shape[0] >= H:
        gt_med = float(torch.linalg.norm(T_gt_cam[:H, :3, 3], dim=-1).median().item() + 1e-8)
        pred_med = float(torch.linalg.norm(T_series[:, :3, 3], dim=-1).median().item() + 1e-8)
        ratio_med = pred_med / gt_med
        if ratio_med > 100.0 or ratio_med < 0.01:
            s = gt_med / pred_med
            t_denorm = t_denorm * s
            T_series = se3_to_matrix(t_denorm, T_series[:, :3, :3])
            info.update({"scale_correction_applied": True, "scale_correction_factor": float(s)})
        else:
            info["scale_correction_applied"] = False
        info.update({"gt_t_norm_med": gt_med, "pred_t_norm_med_after": float(torch.linalg.norm(T_series[:, :3, 3], dim=-1).median().item())})
    else:
        info["scale_correction_applied"] = False

    T_base_rep = T_gt_cam[anchor_idx : anchor_idx + 1].repeat(H, 1, 1) if T_gt_cam is not None and 0 <= anchor_idx < T_gt_cam.shape[0] else None

    # Build candidate interpretations
    cands: Dict[str, torch.Tensor] = {"abs_in_anchor_cam": T_series, "deltas_from_prev_cam": _cumprod_T(T_series)}
    if T_base_rep is not None:
        cands["deltas_from_anchor_cam"] = compose_T(T_series, T_base_rep)
    if T_c_w is not None and T_gt_cam is not None:
        cands["deltas_from_prev_cam_adjoint"] = _adjoint_compose_deltas_from_prev_cam(T_c_w, T_series, T_gt_cam[anchor_idx : anchor_idx + 1], anchor_idx)
    if T_c_w is not None and T_c_o is not None:
        T_w_oa = _to_world_from_cam_series(T_c_w[anchor_idx : anchor_idx + 1], T_c_o[anchor_idx : anchor_idx + 1])[0]
        T_w_o_series = compose_T(T_w_oa.unsqueeze(0), T_series)
        cands["relative_world_from_o0"] = _to_T_cam_anchor_from_world_series(sample, T_w_o_series)

    # Select best candidate
    best_key, best_score, best_T = None, math.inf, None
    if T_gt_cam is not None and T_gt_cam.shape[0] >= H:
        T_gt_cmp = T_gt_cam[:H]
        for k, Tv in cands.items():
            rot_err, trans_err = pose_errors_deg_m(Tv, T_gt_cmp)
            score = float(rot_err.mean().item()) + float(trans_err.mean().item())
            if isinstance(mode_hint, str) and k == mode_hint:
                score *= 0.98
            if isinstance(k, str) and k.endswith("_adjoint"):
                score *= 0.98
            if score < best_score:
                best_score, best_key, best_T = score, k, Tv
    else:
        n_norm = {k: float(torch.linalg.norm(Tv[0, :3, 3]).item()) for k, Tv in cands.items()}
        best_key = mode_hint if isinstance(mode_hint, str) and mode_hint in n_norm else min(n_norm, key=lambda x: n_norm[x])
        best_T = cands[best_key]

    info["pred_mode"] = best_key
    if mode_hint is not None and mode_hint != best_key:
        info["pred_mode_forced"] = mode_hint
    if isinstance(best_key, str) and best_key == "abs_in_anchor_cam":
        info["branch_note"] = "abs_in_anchor_cam: using direct [R|t] in anchor camera"

    if T_gt_cam is not None and T_gt_cam.shape[0] >= H:
        T_gt_cmp = T_gt_cam[:H]
        t_raw_norm = torch.linalg.norm(t_raw, dim=-1)
        t_abs_norm = torch.linalg.norm(best_T[:, :3, 3], dim=-1)
        gt_norm = torch.linalg.norm(T_gt_cmp[:, :3, 3], dim=-1) + 1e-8
        ratio = t_abs_norm / gt_norm
        info.update(
            {
                "t_raw_norm_med": float(t_raw_norm.median().item()),
                "t_denorm_norm_med": float(torch.linalg.norm(t_denorm, dim=-1).median().item()),
                "t_abs_norm_med": float(t_abs_norm.median().item()),
                "scale_ratio_med": float(ratio.median().item()),
                "scale_ratio_mean": float(ratio.mean().item()),
                "t_pred_raw_norm_mean": float(t_raw_norm.mean().item()),
                "t_pred_denorm_norm_mean": float(torch.linalg.norm(t_denorm, dim=-1).mean().item()),
                "t_gt_norm_mean": float(gt_norm.mean().item()),
            }
        )
        if float(ratio.median().item()) > 100.0 or float(ratio.median().item()) < 0.01:
            info["unit_warning"] = f"translation scale ratio pred/gt ≈ {float(ratio.median().item()):.1f}"
    return best_T, info


def add_panel_label(img: np.ndarray, title: str, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw a small corner label for panel identification."""
    h, w = img.shape[:2]
    pad = 6
    bg = img.copy()
    cv2.rectangle(bg, (pad, pad), (pad + 80, pad + 28), color, thickness=-1)
    out = img.copy()
    out[pad : pad + 28, pad : pad + 80] = (0.25 * out[pad : pad + 28, pad : pad + 80] + 0.75 * bg[pad : pad + 28, pad : pad + 80]).astype(out.dtype)
    cv2.putText(out, title, (pad + 8, pad + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return out
