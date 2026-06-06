from __future__ import annotations

import os
import random
from datetime import datetime

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .data.cache import load_split
from .data.datasets import HOT3DClipsDataset, SceneSequenceDataset, SceneSequenceDatasetSynth
from .geom.canonicalize import canonicalize_preds_to_anchor
from .geom.se3_ops import geodesic_distance_deg
from .models.poser_v1 import PoserV1
from .models.poser_v1.utils.checkpoint import pop_lazy_encoder_proj_state, resolve_and_load_state_dict, restore_lazy_encoder_proj
from .models.poser_v1.utils.inferviz_common import (
    apply_sanity_checks,
    compute_train_style_metrics,
    display_diagnostics,
    save_error_metrics,
)
from .utils.batch_utils import as_int_list, pred_mode_from_outfmt
from .utils.config_adapter import apply_config_adapter
from .utils.data_utils import get_dataset
from .utils.geometry import scale_intrinsics
from .utils.logger import epoch_table
from .utils.logger import rprint as print
from .utils.torch_compile import maybe_enable_compile

_geod_deg = geodesic_distance_deg


# _as_int_list imported from utils.batch_utils
_as_int_list = as_int_list


def _has_len(x):
    try:
        return len(x) >= 0
    except Exception:
        return False


# Dataset creation now delegated to utils.data_utils.get_dataset


# _pred_mode_from_outfmt imported from utils.batch_utils
_pred_mode_from_outfmt = pred_mode_from_outfmt


def _initialize_model_components(model: PoserV1, dataset: SceneSequenceDataset | SceneSequenceDatasetSynth | HOT3DClipsDataset, device: torch.device) -> None:
    """
    Initialize model components before loading weights.

    This ensures that all model parameters exist and can be properly loaded from checkpoint,
    maintaining parity with the training setup.

    Args:
        model: PoserV1 model instance
        dataset: Dataset for initialization
        device: Device to run on
    """
    try:
        if hasattr(dataset, "__len__") and len(dataset) > 0:
            sample = dataset[0]
            if isinstance(sample, dict):
                # Warm encoder projections
                conditioning_info = model.condition_from_batch(sample)
                with torch.no_grad():
                    T_cam_anchor_obj = conditioning_info.get("T_cam_anchor_obj")
                    if T_cam_anchor_obj is not None:
                        T_cam_anchor_obj = T_cam_anchor_obj.to(device)
                    _ = model.encode(
                        conditioning_info["scene_pcd"].to(device),
                        conditioning_info["context_vec"].to(device),
                        T_cam_anchor_obj=T_cam_anchor_obj,
                    )
                # Optional: warm full temporal path exactly like training
                try:
                    train_batch = {**sample, **conditioning_info}
                    with torch.no_grad():
                        _ = model.compute_loss(train_batch)
                except Exception:
                    pass
    except Exception:
        pass


def _resolve_checkpoint_path(cfg: DictConfig, debug_mode: bool) -> str:
    """
    Resolve checkpoint path with priority order.

    Priority order:
    1. Explicit cfg.infer.ckpt if it exists
    2. train.out_dir/checkpoints/best.pt
    3. train.out_dir/checkpoints/last.pt

    In debug mode, prefer best.pt if available, else last.pt.

    Args:
        cfg: Configuration dictionary
        debug_mode: Whether in debug mode

    Returns:
        Path to checkpoint file
    """
    checkpoint_path = cfg.infer.ckpt
    prefer_training_checkpoint = False

    try:
        best_checkpoint = os.path.join(cfg.train.out_dir, "checkpoints", "best.pt")
        last_checkpoint = os.path.join(cfg.train.out_dir, "checkpoints", "last.pt")

        if isinstance(checkpoint_path, str) and os.path.exists(checkpoint_path):
            prefer_training_checkpoint = False
        elif os.path.exists(best_checkpoint):
            checkpoint_path = best_checkpoint
            prefer_training_checkpoint = True
        else:
            checkpoint_path = last_checkpoint
            prefer_training_checkpoint = True
    except Exception:
        pass

    # Add note about checkpoint source
    note = " (auto from train.out_dir)" if prefer_training_checkpoint else ""
    print(f"[bold green]Loading[/bold green] ckpt: [yellow]{checkpoint_path}[/yellow]{note}")

    return checkpoint_path


def _build_inference_batch(sample: dict, cfg: DictConfig) -> dict:
    """
    Build inference batch from sample data.

    This function constructs a batch that matches the training batch structure,
    ensuring compatibility with model.condition_from_batch.

    Args:
        sample: Dataset sample
        cfg: Configuration dictionary

    Returns:
        Inference batch dictionary
    """
    # Extract scene point cloud with fallback options
    scene_pointcloud = sample.get("scene_pcd", None)
    if scene_pointcloud is None:
        scene_pointcloud = sample.get("scene_pcd0", None)
    if scene_pointcloud is None:
        scene_pointcloud = sample.get("pointcloud0", None)
    if scene_pointcloud is None:
        scene_pointcloud = np.zeros((int(cfg.data.n_points), 3), dtype=np.float32)

    # Require sequence context (no fallback to single frame)
    context_len = int(sample.get("context_len", 0))
    if context_len <= 0:
        raise ValueError("Inference requires sequence context: sample.context_len must be > 0.")
    context_init_9d = sample.get("context_init_9d", None)
    context_bbox_norm = sample.get("context_bbox_norm", None)
    if context_init_9d is None or context_bbox_norm is None:
        raise KeyError("Inference requires context sequence: 'context_init_9d' and 'context_bbox_norm' must be present in the sample.")

    # Extract object information
    object_info = sample.get("object", {})
    bbox_normalized = object_info.get("bbox_norm", None)

    return {
        "scene_pcd": scene_pointcloud,
        "init_pose": sample["init_pose"],
        # Required temporal context (P frames)
        "context_len": context_len,
        "context_frame_ids": sample.get("context_frame_ids"),
        "context_T_cam_anchor_obj": sample.get("context_T_cam_anchor_obj"),
        "context_init_9d": context_init_9d,
        "context_bbox_norm": context_bbox_norm,
        # mirror train meta keys so condition_from_batch has identical view
        "frame_ids": sample.get("frame_ids"),
        "anchor_frame_idx": sample.get("anchor_frame_idx", 0),
        "anchor_local_idx": sample.get("anchor_local_idx", 0),
        "T_c_w": sample.get("T_c_w"),
        "T_c_o": sample.get("T_c_o"),
        "T_cam_anchor_obj": sample.get("T_cam_anchor_obj"),
        # Provide targets for optional train-like reconstruction (DiT)
        "target_future": sample.get("target_future"),
        # Expose dataset output format when present
        "output_format": sample.get("output_format"),
        "object": {
            "bbox_norm": bbox_normalized,
            "bbox": object_info.get("bbox", None),
            "mesh_path": sample.get("mesh_path", None),
        },
    }


def _get_temporal_backend_type(cfg: DictConfig) -> str:
    """
    Get temporal backend type from configuration.

    Args:
        cfg: Configuration dictionary

    Returns:
        Temporal backend type ("dit" or "ar_transformer")
    """
    return str(cfg.model.temporal_kind).lower()


# === Helper functions for inference config access ===
def _get_infer_flag(cfg, name, default=False):
    return bool(getattr(cfg.infer, name, default))


def _get_infer_int(cfg, name, default=0):
    return int(getattr(cfg.infer, name, default))


def _get_infer_float(cfg, name, default=0.0):
    return float(getattr(cfg.infer, name, default))


# === NEW: helpers for auto extrinsics selection and gauge fix (inference-only) ===
@torch.no_grad()
def _poses_to_9d(T: torch.Tensor) -> torch.Tensor:
    """
    T: (H,4,4) -> (H,9) as [t(3), rot6d(6)] with columns R[:,0], R[:,1].
    """
    assert T.ndim == 3 and T.shape[-2:] == (4, 4), f"Expected (H,4,4), got {tuple(T.shape)}"
    t = T[:, :3, 3]
    R = T[:, :3, :3]
    r6 = torch.stack([R[:, :, 0], R[:, :, 1]], dim=1).reshape(T.shape[0], 6)
    return torch.cat([t, r6], dim=-1)


@torch.no_grad()
def pick_conv_arrow_auto(meta: dict, pred_mode: str, gt_tokens_9d: torch.Tensor, P: int, H: int) -> str:
    """
    Try both arrows and choose the one that minimizes rot geodesic error when
    feeding GT tokens through the prediction canonicalization path.
    - meta must include: "T_cam_anchor_obj" covering [0 : P+H]
    - do_denorm=False for GT tokens
    """
    arrows = ["w<-c", "c<-w"]
    best_arrow, best_err = arrows[0], float("inf")
    T_gt_slice = torch.as_tensor(meta["T_cam_anchor_obj"][P : P + H], dtype=torch.float32)  # (H,4,4)
    for arrow in arrows:
        T_via_predpath, _ = canonicalize_preds_to_anchor(
            gt_tokens_9d[None, ...].to(dtype=torch.float32),  # (1,H,9)
            meta,
            pred_mode,
            arrow,
            do_denorm=False,
            return_intermediates=False,
        )
        T0 = T_via_predpath[0]  # (H,4,4)
        err = _geod_deg(T0[:, :3, :3], T_gt_slice[:, :3, :3]).mean().item()
        if err < best_err:
            best_err, best_arrow = err, arrow
    return best_arrow


def _reorthonormalize_R(R: torch.Tensor) -> torch.Tensor:
    # SVD re-orth with det +1
    U, _, Vt = torch.linalg.svd(R)
    R_ = U @ Vt
    det = torch.det(R_)
    # Ensure positive determinant
    if det.dim() == 0:
        if det.item() < 0:
            U[..., 2] *= -1.0
            return U @ Vt
        return R_
    fix = torch.where(det < 0, torch.tensor(-1.0, device=R.device, dtype=R.dtype), torch.tensor(1.0, device=R.device, dtype=R.dtype))
    U[..., 2] = U[..., 2] * fix.unsqueeze(-1)
    return U @ Vt


def _compute_indices(sample: dict, cfg: DictConfig, Hn: int) -> tuple[int, int, int, int, int]:
    """
    Compute indices using explicit ID mapping (no guessing):
      - P: provided context length
      - aLoc: dataset-provided canonicalization anchor (local)
      - anchor_idx: index of the last context frame inside frame_ids
      - start_gt: anchor_idx + 1  (the frame right after the last context)
      - stop_gt:  start_gt + Hn
    Falls back to the original P-based logic if IDs are missing/inconsistent.
    """
    # Use configuration P (excludes the anchor) for horizon alignment.
    # Dataset augments sample['context_len'] to include the anchor (P+1); we keep P from cfg.
    P = int(cfg.data.context_len)
    aLoc = int(sample.get("anchor_local_idx", 0))
    assert P >= 1
    # Pull frame id lists
    frames = _as_int_list(sample.get("frame_ids", None))
    ctx_frames = _as_int_list(sample.get("context_frame_ids", None))
    # Default/fallback path: old behavior
    anchor_idx_default = max(0, min(P - 1, len(frames) - 1 if frames else P - 1))
    start_default = anchor_idx_default + 1
    stop_default = start_default + int(Hn)
    # If we don't have both lists, keep old behavior
    if not _has_len(frames) or len(frames) < (P + Hn):
        return P, aLoc, anchor_idx_default, start_default, stop_default
    # Determine the last pre-anchor context global id
    if _has_len(ctx_frames) and len(ctx_frames) >= (P + 1):
        # ctx_frames includes the anchor at the end; take the penultimate as last pre-anchor context
        last_ctx_gid = ctx_frames[P - 1] if (P - 1) < len(ctx_frames) else ctx_frames[-2]
    else:
        # Fallback: assume the last pre-anchor context aligns with frames[P-1]
        last_ctx_gid = frames[P - 1] if (P - 1) < len(frames) else frames[min(len(frames) - 1, 0)]
    # Map last context gid into frames[…]
    try:
        idx_last_ctx = frames.index(last_ctx_gid)
    except ValueError:
        # If the exact gid is not present (e.g., dropped frames), fall back
        return P, aLoc, anchor_idx_default, start_default, stop_default
    # Deterministic indices (metrics/diagnostics only)
    anchor_idx = idx_last_ctx
    start_gt = idx_last_ctx + 1
    stop_gt = start_gt + int(Hn)
    # If OOB (e.g., short tails), fall back to the default to avoid crashes
    if stop_gt > len(frames) or start_gt < 0:
        return P, aLoc, anchor_idx_default, start_default, stop_default
    # Warn when this differs from the legacy P-based expectation
    if anchor_idx != anchor_idx_default or start_gt != start_default:
        try:
            from .utils.logger import rprint as _rprint
        except Exception:

            def _rprint(x):
                print(x)

        _rprint(f"[yellow]ID-mapped slicing[/yellow]: anchor_idx {anchor_idx_default}→{anchor_idx}, start_gt {start_default}→{start_gt} (P={P}, aLoc={aLoc})")
    return P, aLoc, anchor_idx, start_gt, stop_gt


def _nineD_to_T(y9: torch.Tensor) -> torch.Tensor:
    """
    y9: (..., 9) with [tx,ty,tz, rot6d(6)] -> (...,4,4)
    """
    from .geom.se3_ops import rot6d_to_matrix

    t = y9[..., :3]
    r6 = y9[..., 3:]
    R = rot6d_to_matrix(r6)
    eye = torch.eye(4, device=y9.device, dtype=y9.dtype)
    T = eye.expand(y9.shape[:-1] + (4, 4)).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T


def _mat_to_euler_zyx(R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # returns yaw(z), pitch(y), roll(x) in radians
    sy = torch.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    pitch = torch.atan2(-R[..., 2, 0], sy)
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    return yaw, pitch, roll


def _resolve_extrinsics_indexing(T_c_w_all: torch.Tensor, frame_ids: list[int], a_glob: int, anchor_idx: int):
    """
    Return:
      - use_local: whether T_c_w_all is window-sized and we should index by local indices
      - idx_map: callable that maps a global id g -> index into T_c_w_all
      - a_sel: index for the anchor in T_c_w_all
    """
    F = int(T_c_w_all.shape[0])
    use_local = F == len(frame_ids)
    frame_to_local = {g: i for i, g in enumerate(frame_ids)} if use_local else {}

    def idx_map(g: int) -> int:
        if use_local:
            return frame_to_local[g]
        else:
            if g < 0 or g >= F:
                raise IndexError(f"global frame {g} out of bounds for extrinsics of length {F}")
            return g

    a_sel = anchor_idx if use_local else a_glob
    if a_sel < 0 or a_sel >= F:
        # try remapping anchor via frame_to_local if available
        if use_local and a_glob in frame_to_local:
            a_sel = frame_to_local[a_glob]
        else:
            raise IndexError(f"anchor index {a_sel} out of bounds for extrinsics len {F}")
    return use_local, idx_map, a_sel, F


def _rebase_to_first_horizon_anchor(
    T_pred_abs: torch.Tensor,  # (1,H,4,4)
    T_gt_b: torch.Tensor,  # (1,H,4,4)
    sample: dict,
    frame_ids: list[int],
    ds_anchor_local: int,
    rep_anchor_local: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Left-multiply predictions and GT so that both are expressed in the camera coordinates
    of the first horizon frame (rep_anchor_local). Returns (T_pred_abs_rebased, T_gt_b_rebased, T_rep_c_ds).
    """
    T_c_w_all = torch.as_tensor(sample["T_c_w"], device=T_pred_abs.device, dtype=T_pred_abs.dtype)
    # Resolve local/global indexing
    a_glob_ds = int(sample.get("anchor_frame_idx", frame_ids[ds_anchor_local] if frame_ids else 0))
    use_local, idx_map, _, _ = _resolve_extrinsics_indexing(
        T_c_w_all,
        frame_ids,
        a_glob_ds,
        ds_anchor_local,
    )
    # indices into T_c_w_all for dataset anchor and first-H anchor
    if use_local:
        j0 = int(ds_anchor_local)
        jH = int(rep_anchor_local)
    else:
        j0 = idx_map(int(frame_ids[ds_anchor_local]))
        jH = idx_map(int(frame_ids[rep_anchor_local]))
    # Transform from dataset anchor camera to first-H anchor camera: T_{c_H <- c_ds}
    # Note: T_c_w_all is camera->world (c2w). To map poses expressed in camera(ds) to camera(H),
    # we need: T_{c_H <- c_ds} = inv(T_{w <- c_H}) @ (T_{w <- c_ds}) = inv(T_c_w[jH]) @ T_c_w[j0]
    T_rep_c_ds = torch.linalg.inv(T_c_w_all[jH]) @ T_c_w_all[j0]
    # Rebase prediction and GT
    T_pred_abs_rebased = T_pred_abs.clone()
    T_pred_abs_rebased[0] = T_rep_c_ds @ T_pred_abs[0]
    T_gt_b_rebased = T_gt_b.clone()
    T_gt_b_rebased[0] = T_rep_c_ds @ T_gt_b[0]
    return T_pred_abs_rebased, T_gt_b_rebased, T_rep_c_ds


def _compute_evaluation_metrics(
    predicted_poses: np.ndarray,
    sample: dict,
    window_index: int,
    output_directory: str,
    cfg: DictConfig,
    dataset_ref=None,
    t_scale: float = 1.0,
    debug_sample_count: int | None = None,
    model_outfmt: str | None = None,
    dnorm_means: list[float] | torch.Tensor | None = None,
    dnorm_stds: list[float] | torch.Tensor | None = None,
    dnorm_fitted: bool | None = None,
) -> dict:
    """
    Compute evaluation metrics and diagnostics for predicted poses, aligned with training.
    """
    from .utils.validate import ate_rpe

    # Indices (A)
    Hn = int(predicted_poses.shape[0])
    # debug alignment sweep removed
    P, aLoc, anchor_idx, start_gt, stop_gt = _compute_indices(sample, cfg, Hn)
    print(f"ctx_len={P} anchor_idx={anchor_idx} pred_range=[{start_gt}, {stop_gt - 1}] H={Hn}")
    # Early gap detection: if window frame_ids are contiguous and context aligns with frames[P-1], then start_gt must equal P
    try:
        frames_all = _as_int_list(sample.get("frame_ids", None))
        ctx_frames = _as_int_list(sample.get("context_frame_ids", None))
        if len(frames_all) >= (P + 1):
            is_contiguous = all((frames_all[i] - frames_all[i - 1]) == 1 for i in range(1, len(frames_all)))
            ctx_ok = len(ctx_frames) >= 1 and int(ctx_frames[-1]) == frames_all[P - 1]
            if is_contiguous and ctx_ok:
                assert start_gt == P, (
                    f"Expected start_gt == P ({P}) for contiguous frames with aligned context; got start_gt={start_gt}. This indicates a dataset gap or ID mismatch."
                )
    except Exception:
        # Non-fatal: only enforce when conditions are fully satisfied
        pass

    # Pred mode mapping (D)
    # Prefer model's training output_format when available — avoids format mismatch
    try:
        ds_ref = dataset_ref.dataset if hasattr(dataset_ref, "dataset") else dataset_ref
        ds_outfmt = sample.get("output_format", None)
        if ds_outfmt is None:
            ds_outfmt = getattr(ds_ref, "output_format", None)
    except Exception:
        ds_outfmt = sample.get("output_format", None)
    outfmt = model_outfmt if model_outfmt is not None else (ds_outfmt if ds_outfmt is not None else "abs_in_anchor")
    pred_mode = _pred_mode_from_outfmt(str(outfmt))
    # Default convention (overridden below by auto/force)
    conv_arrow = "w<-c"

    # Build GT slice using window-local indices (A, D) early (needed for auto arrow)
    T_gt_all = torch.from_numpy(np.asarray(sample["T_cam_anchor_obj"])).float()
    if stop_gt > T_gt_all.shape[0]:
        print(f"[yellow]Index error[/yellow] GT slice OOB: aLoc={aLoc} P={P} start_gt={start_gt} stop_gt={stop_gt} Hn={Hn} len(T_gt_all)={int(T_gt_all.shape[0])}")
    T_gt_slice = T_gt_all[start_gt:stop_gt]
    assert int(T_gt_slice.shape[0]) == int(Hn), (
        f"GT slice len {int(T_gt_slice.shape[0])} != H {int(Hn)} (P={P}, aLoc={aLoc}, start_gt={start_gt}, stop_gt={stop_gt}, total={int(T_gt_all.shape[0])})"
    )

    # Canonicalize predictions to anchor camera (D)
    pred_9d = torch.from_numpy(np.asarray(predicted_poses)).float().unsqueeze(0)
    # Match training: do NOT apply dataset/model normalization in canonicalization.
    # Training passed identity denorm (means=[0,0,0], std=[1,1,1]) for metrics.
    t_mean_vec: list[float] = [0.0, 0.0, 0.0]
    t_std_vec: list[float] = [1.0, 1.0, 1.0]
    # Meta for canonicalization (train-parity fields + denorm stats)
    sample_conv = str(sample.get("extrinsics_convention", "") or "").lower()
    meta = {
        "K": sample.get("K"),
        "frame_ids": sample.get("frame_ids"),
        "anchor_frame_idx": int(sample.get("anchor_frame_idx", 0)),
        # Use dataset-defined anchor for canonicalization (training perspective)
        "anchor_local_idx": int(aLoc),
        "T_c_w": sample.get("T_c_w"),
        "T_c_o": sample.get("T_c_o"),
        "T_cam_anchor_obj": sample.get("T_cam_anchor_obj"),
        "t_mean": t_mean_vec,
        "t_std": t_std_vec,
        "extrinsics_convention": sample_conv,
    }

    conv_arrow = "w<-c"

    # Canonicalize predictions
    T_pred_abs, canon_info = canonicalize_preds_to_anchor(pred_9d, meta, pred_mode, conv_arrow, do_denorm=True, return_intermediates=True)
    predicted_camera_poses = T_pred_abs[0]

    # Resolve frames and reporting anchor (use dataset-provided anchor)
    frames_all = _as_int_list(sample.get("frame_ids", None))
    rep_anchor_local = int(sample.get("anchor_local_idx", 0))
    try:
        rep_anchor_global = int(sample.get("anchor_frame_idx", int(frames_all[rep_anchor_local]) if len(frames_all) > rep_anchor_local else rep_anchor_local))
    except Exception:
        rep_anchor_global = int(frames_all[rep_anchor_local]) if (len(frames_all) > rep_anchor_local) else int(rep_anchor_local)
    # Frame signature one-liner (report dataset anchor)
    frames_str = str(frames_all) if len(frames_all) else "[]"
    print(
        f"signature • pred_mode={pred_mode}  (model_outfmt={model_outfmt}, dataset_outfmt={ds_outfmt})  "
        f"extrinsics={conv_arrow} frames={frames_str} anchor=(g={rep_anchor_global},l={rep_anchor_local}) repr=se3(t+rot6d)"
    )
    print(f"[indices] P={P}  rep_anchor_local={rep_anchor_local}  pred_range=[{start_gt}, {stop_gt - 1}]  H={Hn}")
    print(f"[indices] P={P}  start_gt={start_gt}  stop_gt={stop_gt}  (eval_shift_offset={_get_infer_int(cfg, 'eval_shift_offset', 0)})")

    # Apply optional sanities (parity guard: must be a no-op)
    _raw = predicted_camera_poses.clone()
    predicted_camera_poses = apply_sanity_checks(predicted_camera_poses, sample, cfg, namespace="infer")
    if not torch.allclose(predicted_camera_poses, _raw):
        raise RuntimeError("Inference parity: apply_sanity_checks changed predictions. Set cfg.infer.sanity_pred_equals_gt=false and cfg.infer.sanity_pred_unit_scale=1.0")

    # debug alignment sweep removed
    # Ensure GT has batch dim for all diagnostics
    T_gt_b = T_gt_slice.unsqueeze(0).to(T_pred_abs.device, T_pred_abs.dtype)  # (1,H,4,4)

    # Identity sanity (E)
    try:
        # Probe GT through the prediction path WITHOUT denorm to pick/validate arrow identity
        gt_tokens_9d = _poses_to_9d(T_gt_slice.to(dtype=pred_9d.dtype))
        T_gt_via_predpath, _ = canonicalize_preds_to_anchor(gt_tokens_9d.unsqueeze(0), meta, pred_mode, conv_arrow, do_denorm=False, return_intermediates=False)
        Rp_i = T_gt_via_predpath[0][..., :3, :3]
        Rg_i = T_gt_slice[..., :3, :3]
        canon_id_rot_deg = geodesic_distance_deg(Rp_i, Rg_i).mean()
        print(f"[auto-conv] identity rot_err_deg={float(canon_id_rot_deg):.4f}")
    except Exception:
        canon_id_rot_deg = torch.tensor(float("nan"))
    # Assert canonicalization identity holds tightly (parity with training)
    tol_deg = _get_infer_float(cfg, "identity_tol_deg", 0.2)  # default 0.2°
    assert float(canon_id_rot_deg) <= tol_deg, f"Parity fail: canon_id_rot_deg={float(canon_id_rot_deg):.4g} deg > tol={tol_deg}° (adjust infer.identity_tol_deg if needed)"

    # Use dataset-provided anchor for all metrics/logs (no rebase)
    print("[anchor] Using dataset anchor for all metrics/logs")
    print(f"frames={frames_all} • anchor_global={rep_anchor_global} • anchor_local={rep_anchor_local}")

    # Optional debug: reinterpret predictions under different pred_modes
    if _get_infer_flag(cfg, "debug_mode_sweep", False):
        modes = ["abs_in_anchor_cam", "deltas_from_prev_cam", "deltas_from_anchor_cam"]
        for m in modes:
            try:
                T_abs_m, _ = canonicalize_preds_to_anchor(pred_9d, meta, m, conv_arrow, True, False)
                r_m = geodesic_distance_deg(T_abs_m[..., :3, :3], T_gt_slice.unsqueeze(0)[..., :3, :3]).mean()
                t_m = torch.linalg.norm(T_abs_m[..., :3, 3] - T_gt_slice.unsqueeze(0)[..., :3, 3], dim=-1).mean()
                print(f"[mode-sweep] {m}: rot={float(r_m):.3f}° trans={float(t_m):.4f}m")
            except Exception:
                pass

    # Decide which absolute poses in ANCHOR camera will be used for metrics (canonicalized only)
    used_path = "canonicalized"
    T_pred_used = T_pred_abs

    # SE(3) metrics (D) computed with selected pose sequence
    Rp = T_pred_used[..., :3, :3]
    tp = T_pred_used[..., :3, 3]
    Rg = T_gt_b[..., :3, :3]
    tg = T_gt_b[..., :3, 3]
    # Per-frame metrics
    rot_deg_k = geodesic_distance_deg(Rp, Rg)[0]  # (H,)
    trans_l2_k = torch.linalg.norm(tp - tg, dim=-1)[0]  # (H,)
    rot_deg = rot_deg_k.mean()
    trans_l2 = trans_l2_k.mean()
    print(f"[metrics] rot_deg_mean={float(rot_deg):.4f}  trans_l2_mean={float(trans_l2):.4f}")

    # Train-style param metrics and extras
    compute_train_style_metrics(predicted_poses, sample, window_index, cfg)

    # Reprojection, directionality, and ATE/RPE for diagnostics
    statistics = ate_rpe(T_pred_used[0], T_gt_b[0])
    reprojection_errors = _compute_reprojection_errors(T_pred_used[0], T_gt_b[0], sample, cfg)
    directionality_stats = _compute_directionality_stats(sample)

    # Display diagnostics (re-use existing pretty block)
    display_diagnostics(
        T_pred_used[0],
        T_gt_b[0],
        sample,
        canon_info,
        statistics,
        geodesic_distance_deg(Rp, Rg),  # per-step not used; pass full tensors if needed
        torch.linalg.norm(tp - tg, dim=-1),
        reprojection_errors,
        directionality_stats,
        window_index,
        cfg,
        report_frames=frames_all,
        report_anchor_global=rep_anchor_global,
        report_anchor_local=rep_anchor_local,
        report_anchor_tag="first-H",
    )

    # Save error metrics bundle (existing JSON)
    from .utils.validate import pose_errors_deg_m as _pose_errs

    rot_err_ps, trans_err_ps = _pose_errs(T_pred_used[0], T_gt_b[0])
    save_error_metrics(
        rot_err_ps,
        trans_err_ps,
        statistics,
        reprojection_errors,
        directionality_stats,
        window_index,
        output_directory,
        debug_sample_count=debug_sample_count if (isinstance(debug_sample_count, int) and debug_sample_count > 1) else 1,
    )
    # New: drift per step and error slope (per sequence)
    try:
        e_r = [float(x) for x in rot_err_ps.detach().cpu().tolist()]
        e_t = [float(x) for x in trans_err_ps.detach().cpu().tolist()]
        Tn = int(len(e_t))
        denom = float(max(Tn - 1, 1))
        trans_drift_per_step = (e_t[-1] - e_t[0]) / denom if Tn >= 1 else float("nan")
        rot_drift_per_step = (e_r[-1] - e_r[0]) / denom if Tn >= 1 else float("nan")
        t_idx = np.arange(Tn, dtype=np.float64) if Tn >= 1 else np.array([0.0], dtype=np.float64)
        if Tn >= 2:
            trans_error_slope, _ = np.polyfit(t_idx, np.asarray(e_t, dtype=np.float64), 1)
            rot_error_slope, _ = np.polyfit(t_idx, np.asarray(e_r, dtype=np.float64), 1)
        else:
            trans_error_slope, rot_error_slope = float("nan"), float("nan")
    except Exception:
        trans_drift_per_step = float("nan")
        rot_drift_per_step = float("nan")
        trans_error_slope = float("nan")
        rot_error_slope = float("nan")

    return {
        "rot_deg_mean": float(rot_deg.detach().cpu().item()),
        "trans_l2_mean": float(trans_l2.detach().cpu().item()),
        "canon_id_rot_deg_mean": float(canon_id_rot_deg.detach().cpu().item()),
        # New drift/slope scalars for this sequence
        "trans_drift_per_step": float(trans_drift_per_step),
        "rot_drift_per_step": float(rot_drift_per_step),
        "trans_error_slope": float(trans_error_slope),
        "rot_error_slope": float(rot_error_slope),
        "ctx_len": int(P),
        # Report the first-H anchor as the anchor for metrics/logs
        "anchor_idx": int(rep_anchor_local),
        "pred_start": int(start_gt),
        "pred_stop_exclusive": int(stop_gt),
        "H": int(Hn),
        "pred_mode": pred_mode,
        "canon_t_std_used": float(t_std_vec[0] if isinstance(t_std_vec, (list, tuple)) and len(t_std_vec) >= 1 else 1.0),
        "used_path": used_path,
        # Export reporting anchor for consumers (e.g., metadata save)
        "anchor_local_report": int(rep_anchor_local),
        "anchor_global_report": int(rep_anchor_global),
    }


def _compute_reprojection_errors(predicted_poses: torch.Tensor, ground_truth_poses: torch.Tensor, sample: dict, cfg: DictConfig) -> dict:
    """Compute reprojection errors."""
    try:
        from .data.video_io import get_video_frame_size
        from .utils.validate import K_signature, reprojection_error_norm, reprojection_error_px, reprojection_sanity_origin_px

        # Get camera intrinsics
        camera_intrinsics = sample.get("K")
        scaled_intrinsics = camera_intrinsics

        # Scale intrinsics to match actual image size
        try:
            if camera_intrinsics is not None:
                from .utils.geometry import infer_image_size_from_K

                intrinsic_height, intrinsic_width = infer_image_size_from_K(camera_intrinsics)
                image_height, image_width = 0, 0

                rgb_path = sample.get("rgb_path", None)
                if isinstance(rgb_path, str):
                    # HOT3D tar file support
                    if rgb_path.endswith(".tar"):
                        from .data.video_io import get_hot3d_frame_size

                        frame_size = get_hot3d_frame_size(rgb_path)
                    else:
                        frame_size = get_video_frame_size(rgb_path)
                    if isinstance(frame_size, tuple):
                        image_height, image_width = int(frame_size[0]), int(frame_size[1])

                if image_height > 0 and image_width > 0 and (intrinsic_height != image_height or intrinsic_width != image_width):
                    scaled_intrinsics = scale_intrinsics(camera_intrinsics, (intrinsic_height, intrinsic_width), (image_height, image_width))
        except Exception:
            pass

        # Pixel-space sanity (anchor camera triad reprojection)
        if camera_intrinsics is not None:
            try:
                # Prepare a small 0.1 m triad anchored at camera t
                axes = torch.tensor([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], device=predicted_poses.device, dtype=predicted_poses.dtype)  # (4,3)
                # Ensure batch dimension exists for unified broadcasting
                Ts_pred = predicted_poses
                Ts_gt = ground_truth_poses
                if Ts_pred.dim() == 3:
                    Ts_pred = Ts_pred.unsqueeze(0)
                if Ts_gt.dim() == 3:
                    Ts_gt = Ts_gt.unsqueeze(0)
                # Extract intrinsics
                fx = float(scaled_intrinsics[0, 0])
                fy = float(scaled_intrinsics[1, 1])
                cx = float(scaled_intrinsics[0, 2])
                cy = float(scaled_intrinsics[1, 2])
                # Project predicted
                R_pred = Ts_pred[..., :3, :3]  # (B,H,3,3)
                t_pred = Ts_pred[..., :3, 3]  # (B,H,3)
                pts_pred = (R_pred @ axes.T).transpose(-1, -2) + t_pred.unsqueeze(-2)  # (B,H,4,3)
                x_p, y_p, z_p = pts_pred[..., 0], pts_pred[..., 1], pts_pred[..., 2]
                u_p = fx * (x_p / z_p) + cx
                v_p = fy * (y_p / z_p) + cy
                uv_pred = torch.stack([u_p, v_p], dim=-1)  # (B,H,4,2)
                # Project ground truth
                R_gt = Ts_gt[..., :3, :3]  # (B,H,3,3)
                t_gt = Ts_gt[..., :3, 3]  # (B,H,3)
                pts_gt = (R_gt @ axes.T).transpose(-1, -2) + t_gt.unsqueeze(-2)  # (B,H,4,3)
                x_g, y_g, z_g = pts_gt[..., 0], pts_gt[..., 1], pts_gt[..., 2]
                u_g = fx * (x_g / z_g) + cx
                v_g = fy * (y_g / z_g) + cy
                uv_gt = torch.stack([u_g, v_g], dim=-1)  # (B,H,4,2)
                px_err = torch.linalg.norm(uv_pred - uv_gt, dim=-1).mean(dim=-1)  # (B,H) -> (B,)
                print(f"[pixels] mean axis-endpoint error = {float(px_err.mean()):.2f} px")
            except Exception as e:
                print(f"[yellow][pixels] reprojection sanity failed: {e}[/yellow]")

        # Compute reprojection errors
        reprojection_px = reprojection_error_px(scaled_intrinsics, predicted_poses, ground_truth_poses)
        reprojection_norm = reprojection_error_norm(scaled_intrinsics, predicted_poses, ground_truth_poses)
        origin_px = reprojection_sanity_origin_px(scaled_intrinsics, predicted_poses, ground_truth_poses)
        intrinsics_signature = K_signature(scaled_intrinsics)

        return {
            "reprojection_px": reprojection_px,
            "reprojection_norm": reprojection_norm,
            "origin_px": origin_px,
            "K_signature": intrinsics_signature,
            "scaled_intrinsics": scaled_intrinsics,
        }
    except Exception:
        return {}


def _compute_directionality_stats(sample: dict) -> dict:
    """Compute directionality statistics."""
    try:
        from .utils.validate import directionality_probe

        camera_to_world = torch.from_numpy(np.asarray(sample.get("T_c_w"))).float()
        camera_to_object = torch.from_numpy(np.asarray(sample.get("T_c_o"))).float()
        camera_anchor_object = torch.from_numpy(np.asarray(sample.get("T_cam_anchor_obj"))).float()
        anchor_local_index = int(sample.get("anchor_local_idx", 0))
        extrinsics_convention = str(sample.get("extrinsics_convention", "c2w"))

        return directionality_probe(
            camera_to_world,
            camera_to_object,
            camera_anchor_object,
            anchor_local_index,
            extrinsics_convention,
        )
    except Exception:
        return {"dir_corr_mean": float("nan"), "dir_corr_min": float("nan")}


def _display_inference_summary(cfg: DictConfig, temporal_backend_type: str) -> None:
    """
    Display inference summary with backend-specific information.

    Args:
        cfg: Configuration dictionary
        temporal_backend_type: Type of temporal backend ("dit" or "ar_transformer")
    """
    if temporal_backend_type == "dit" and hasattr(cfg.model.temporal, "ddim_steps"):
        # Show effective steps (honor infer.ddim_full_T override)
        use_full_T = bool(cfg.infer.ddim_full_T)
        if use_full_T:
            effective_steps = int(cfg.model.temporal.T)
        else:
            # Prefer infer.ddim_steps if provided; else fall back to model.temporal.ddim_steps
            effective_steps = int(cfg.infer.ddim_steps)
        epoch_table(
            "Inference Summary",
            {
                "H": float(cfg.data.H),
                "ddim_steps": float(effective_steps),
            },
            prec=0,
        )
    else:
        # AR transformer: print backend name separately; keep epoch_table numeric-only
        print("temporal=AutoregTransformerPose")
        epoch_table(
            "Inference Summary",
            {
                "H": float(cfg.data.H),
            },
            prec=0,
        )


@hydra.main(config_path="../conf", config_name="epic", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main inference function for PoserV1 model.

    This function supports inference with both DiT (Diffusion Transformer) and AR (Autoregressive)
    backends. The inference loop automatically adapts based on the temporal backend type
    specified in the configuration, matching the training setup.

    Args:
        cfg: Hydra configuration dictionary
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[bold green]{current_time}[/bold green] Starting inference…")

    # Register eval resolver BEFORE apply_config_adapter (it accesses temporal_dit.out_dim which uses ${eval:...})
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: int(eval(expr, {"__builtins__": {}}, {})))

    # Apply configuration adapters and setup
    cfg = apply_config_adapter(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu")

    # Deterministic seeding for reproducible DDIM sampling
    seed = int(cfg.infer.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Basic guard
    assert int(cfg.data.H) > 0, "data.H must be > 0"

    # Dataset setup - matches training configuration
    dataset = get_dataset(cfg)

    # Debug mode setup - matches training tiny_overfit behavior
    debug_mode = bool(cfg.train.tiny_overfit)
    # Determine start index and number of windows to process
    start_idx = int(cfg.infer.start_idx or 0)
    start_idx = max(0, min(start_idx, len(dataset)))
    first_n = int(cfg.infer.first_n or 0)
    total_available = max(0, len(dataset) - start_idx)
    num_windows = total_available if first_n <= 0 else min(int(first_n), total_available)

    # Align inference items with training selection under tiny_overfit by using the saved train split
    selected_indices: list[int] = list(range(start_idx, start_idx + num_windows))
    try:
        if isinstance(dataset, SceneSequenceDataset) and bool(cfg.train.tiny_overfit):
            split_tuple = load_split(dataset.dataset_root)
            if split_tuple is not None:
                train_rel, _ = split_tuple
                train_set = set(train_rel)

                def _rel_obj_dir(p: str) -> str:
                    try:
                        return os.path.relpath(str(p), start=str(dataset.dataset_root))
                    except Exception:
                        return str(p)

                train_indices = [i for i, w in enumerate(dataset.windows) if _rel_obj_dir(w.get("object_dir", "")) in train_set]
                tiny_n = int(cfg.train.tiny_n)
                # Use the exact first tiny_n windows from the training split (matches train_main)
                if len(train_indices) > 0:
                    selected_indices = train_indices[: max(1, tiny_n)]
                    num_windows = len(selected_indices)
                    print(f"[bold cyan]tiny_overfit[/bold cyan]=true • using {num_windows} window(s) from training split")
    except Exception:
        # Keep default selection if anything goes wrong
        pass

    # Model setup - matches training configuration
    model: PoserV1 = instantiate(cfg.model, _recursive_=False)
    model.to(device).eval()

    # Initialize model components before loading weights (matches training setup)
    _initialize_model_components(model, dataset, device)

    # Load checkpoint with priority order
    checkpoint_path = _resolve_checkpoint_path(cfg, debug_mode)

    state_dict, ckpt_meta = resolve_and_load_state_dict(checkpoint_path, device)
    lazy_state = pop_lazy_encoder_proj_state(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Restore lazily-instantiated encoder projection weights (and bias if present)
    restored, _ = restore_lazy_encoder_proj(model, lazy_state)
    if restored:
        print(f"[dim]restored lazy encoder params[/dim]: {', '.join(sorted(lazy_state.keys()))}")
    # Recompute effective missing after restoring lazy params (suppress benign warnings)
    lazy_keys = set(lazy_state.keys())
    eff_missing = [k for k in (missing or []) if k not in lazy_keys]
    print(f"ckpt loaded • params={len(state_dict)} missing={len(eff_missing)} unexpected={len(unexpected)}")
    # --- D-norm stats visibility ---
    try:
        dn_fitted = int(model._dnorm_fitted.item())
        dn_means = model._dnorm_means
        dn_scales = model._dnorm_scales
        print(f"[dnorm] fitted={dn_fitted}  means[:3]={dn_means[:3].tolist()}  stds[:3]={dn_scales[:3].tolist()}")
    except Exception as _e:
        print(f"[yellow][dnorm] unable to read dnorm buffers: {_e}[/yellow]")
    if len(eff_missing) or len(unexpected):
        print("[yellow]ckpt key mismatch[/yellow] (see counts above)")

    # Check conditioning mode mismatch
    ckpt_conditioning = (ckpt_meta or {}).get("cfg", {}).get("temporal", {}).get("conditioning", "film")
    curr_conditioning = getattr(cfg.temporal, "conditioning", "film")
    if ckpt_conditioning != curr_conditioning:
        print(f"[yellow]WARNING: Config conditioning='{curr_conditioning}' differs from checkpoint conditioning='{ckpt_conditioning}'. Results may be incorrect.[/yellow]")

    # Optional torch.compile (must happen after state_dict restore)
    maybe_enable_compile(model, cfg, print_fn=print)

    # DiT-only sanity
    if getattr(model, "_temporal_kind", "dit") == "dit":
        t_train = int(model._T_train.item())
        t_infer = int(cfg.model.temporal.T)
        assert t_train == t_infer, f"T mismatch: trained T={t_train}, infer T={t_infer}"
        print(f"DiT scales • trans={float(model._scale_t):.6g}  rot6d={float(model._scale_r):.6g}")
        # Shape consistency
        assert int(cfg.data.H) * 9 == int(model.dit.out_dim), "H*9 must equal DiT out_dim"

    # Setup output directory
    output_directory = os.path.join(cfg.train.out_dir, "preds")
    os.makedirs(output_directory, exist_ok=True)
    print(f"writing {int(num_windows)} preds → {output_directory}")

    # Determine backend once (also used for summary after loop)
    temporal_backend_type = _get_temporal_backend_type(cfg)

    # Model training-time output format (prefer checkpoint cfg if present)
    model_outfmt = None
    try:
        # Prefer ckpt-embedded cfg (exact training config)
        ckpt_cfg = ckpt_meta.get("cfg", None) if isinstance(ckpt_meta, dict) else None
        if isinstance(ckpt_cfg, dict):
            # Try common locations
            model_outfmt = ckpt_cfg.get("model", {}).get("output_format", None) or ckpt_cfg.get("data", {}).get("output_format", None) or ckpt_cfg.get("output_format", None)
        # Fallback to live model cfg
        if model_outfmt is None:
            model_outfmt = getattr(getattr(model, "cfg", None), "output_format", None)
    except Exception:
        model_outfmt = None

    # Run inference on dataset samples
    for dataset_index in selected_indices:
        sample = dataset[dataset_index]

        # Build inference batch - matches training batch structure
        inference_batch = _build_inference_batch(sample, cfg)

        pred_numpy_override = None
        # Run model inference
        with torch.no_grad():
            core = model.module if hasattr(model, "module") else model
            # Record training-time translation scale for denorm (single place via canonicalizer)
            t_scale = float(getattr(core, "_scale_t", 1.0))
            conditioning_info = core.condition_from_batch(inference_batch)

            # Determine temporal backend and sampling parameters
            sampling_steps = cfg.infer.ddim_steps if temporal_backend_type == "dit" else None
            if temporal_backend_type == "dit" and bool(cfg.infer.ddim_full_T):
                sampling_steps = int(cfg.model.temporal.T)
                print(f"[bold yellow]ddim_full_T[/bold yellow]=true • using full training T steps: {int(sampling_steps)}")

            # Generate predictions using appropriate backend
            teacher_forced = bool(cfg.infer.teacher_forced)
            # Default to reconstruct path under tiny_overfit for DiT unless explicitly disabled
            recon_cfg = cfg.infer.reconstruct_dit
            if recon_cfg is None:
                reconstruct_dit = False
            else:
                reconstruct_dit = bool(recon_cfg) and (temporal_backend_type == "dit")
            # Per-sample deterministic generator for DiT parity with eval_main
            gen: torch.Generator | None = None
            if temporal_backend_type == "dit":
                try:
                    gen = torch.Generator(device=device)
                    gen.manual_seed(int(seed) + int(dataset_index))
                except Exception:
                    gen = None

            if teacher_forced and temporal_backend_type == "ar_transformer":
                # Teacher-forced inference: feed GT tokens as in training
                try:
                    target_future = sample.get("target_future")
                    if target_future is None:
                        raise KeyError("target_future missing; fallback to AR rollout")
                    gt = torch.from_numpy(np.asarray(target_future)).float().to(device)  # [H, 9] or [1,H,9]
                    if gt.dim() == 2:
                        gt = gt.unsqueeze(0)
                    # Encode conditioning once
                    T_cam_anchor_obj = conditioning_info.get("T_cam_anchor_obj")
                    cond_embed = core.encode(
                        conditioning_info["scene_pcd"],
                        conditioning_info["context_vec"],
                        T_cam_anchor_obj=T_cam_anchor_obj,
                    )
                    # Forward with teacher forcing to predict next tokens aligned to GT, with context tokens
                    ctx9 = sample.get("context_init_9d")
                    ctx9_t = None
                    if ctx9 is not None:
                        ctx9_t = torch.from_numpy(np.asarray(ctx9)).float().to(device)
                        if ctx9_t.dim() == 2:
                            ctx9_t = ctx9_t.unsqueeze(0)
                    predicted_poses_tensor = core.temporal.forward_train(gt, cond_embed, ctx_tokens=ctx9_t)  # [B,H,9]
                except Exception:
                    # Fallback to normal AR rollout
                    T_cam_anchor_obj = conditioning_info.get("T_cam_anchor_obj")
                    predicted_poses_tensor = core.sample(
                        conditioning_info["scene_pcd"],
                        conditioning_info["context_vec"],
                        T_cam_anchor_obj=T_cam_anchor_obj,
                        steps=sampling_steps,
                        generator=gen,
                    )
            elif reconstruct_dit and temporal_backend_type == "dit":
                # Train-style reconstruction for DiT.
                try:
                    tf = inference_batch.get("target_future")
                    if tf is None:
                        raise KeyError("target_future required for reconstruct_dit")
                    tf_np = np.asarray(tf)
                    if tf_np.ndim == 2:
                        tf_np = tf_np[None, ...]
                    elif tf_np.ndim != 3:
                        raise ValueError(f"target_future must be (H,9) or (B,H,9); got shape={tuple(tf_np.shape)}")
                    recon_use_gt = bool(cfg.infer.reconstruct_from_gt)
                    if recon_use_gt:
                        pred_numpy_override = tf_np.astype(np.float32, copy=True)
                        predicted_poses_tensor = torch.from_numpy(pred_numpy_override).float().to(device)
                        print("[bold yellow]reconstruct_dit[/bold yellow]=true • using ground-truth tokens for train-parity metrics")
                    else:
                        train_like_batch = {**inference_batch, "target_future": tf_np}
                        loss_out = core.compute_loss(train_like_batch)
                        y_pred = loss_out.get("pred_9d", None)
                        if not isinstance(y_pred, torch.Tensor):
                            raise RuntimeError("compute_loss did not return pred_9d")
                        predicted_poses_tensor = y_pred.detach()
                        print("[bold yellow]reconstruct_dit[/bold yellow]=true • using denoised x0 from training path")
                except Exception as e:
                    print(f"[yellow]reconstruct_dit[/yellow] failed ({e}); falling back to DDIM sampling")
                    T_cam_anchor_obj = conditioning_info.get("T_cam_anchor_obj")
                    predicted_poses_tensor = core.sample(
                        conditioning_info["scene_pcd"],
                        conditioning_info["context_vec"],
                        T_cam_anchor_obj=T_cam_anchor_obj,
                        steps=sampling_steps,
                        ctx_tokens_9d=inference_batch.get("context_init_9d"),
                        generator=gen,
                    )
            else:
                # Backend switch (C)
                kind = str(cfg.model.temporal_kind).lower()
                ctx_tokens_9d = inference_batch.get("context_init_9d")
                if kind == "ar_transformer":
                    if not isinstance(ctx_tokens_9d, torch.Tensor):
                        ctx_tokens_9d = torch.as_tensor(ctx_tokens_9d, dtype=torch.float32, device=conditioning_info["scene_pcd"].device)
                    else:
                        ctx_tokens_9d = ctx_tokens_9d.to(device=conditioning_info["scene_pcd"].device, dtype=torch.float32, non_blocking=True)
                    if ctx_tokens_9d.dim() == 2:
                        ctx_tokens_9d = ctx_tokens_9d.unsqueeze(0)

                    ctx_tokens_n = core._perdim_normalize(ctx_tokens_9d)
                    T_cam_anchor_obj = conditioning_info.get("T_cam_anchor_obj")
                    cond_embed = core.encode(conditioning_info["scene_pcd"], conditioning_info["context_vec"], T_cam_anchor_obj=T_cam_anchor_obj)
                    Hn = int(cfg.data.H)
                    tf = sample.get("target_future", None)
                    if tf is not None and hasattr(tf, "shape") and len(tf.shape) >= 2:
                        Hn = int(tf.shape[0])  # match training’s horizon resolution
                    y_pred_n = core.temporal.rollout_infer(Hn, cond_embed, ctx_tokens=ctx_tokens_n)
                    predicted_poses_tensor = core._perdim_unnormalize(y_pred_n)
                else:
                    T_cam_anchor_obj = conditioning_info.get("T_cam_anchor_obj")
                    predicted_poses_tensor = core.sample(
                        conditioning_info["scene_pcd"],
                        conditioning_info["context_vec"],
                        T_cam_anchor_obj=T_cam_anchor_obj,
                        steps=sampling_steps,
                        ctx_tokens_9d=ctx_tokens_9d,
                        generator=gen,
                    )

        # Fail-fast shape check: predicted horizon vs configured/GT horizon
        try:
            if isinstance(predicted_poses_tensor, torch.Tensor):
                Hn_pred = int(predicted_poses_tensor.shape[1]) if predicted_poses_tensor.dim() >= 3 else int(predicted_poses_tensor.shape[0])
            else:
                Hn_pred = int(np.asarray(predicted_poses_tensor).shape[0])
            expected_H = int(cfg.data.H)
            tf = sample.get("target_future", None)
            if tf is not None and hasattr(tf, "shape") and len(tf.shape) >= 2:
                expected_H = int(tf.shape[0])
            assert Hn_pred == expected_H, f"predicted horizon H={Hn_pred} != expected {expected_H}"
        except Exception:
            pass

        # Convert to numpy and fix rotation representation
        predicted_poses_numpy = predicted_poses_tensor.squeeze(0).cpu().numpy()
        if isinstance(pred_numpy_override, np.ndarray):
            predicted_poses_numpy = pred_numpy_override.squeeze(0) if pred_numpy_override.ndim == 3 else pred_numpy_override.copy()

        # Save predictions
        multi = not (num_windows == 1)
        prediction_filename = f"pred_{dataset_index:03d}.npy" if multi else "pred.npy"
        prediction_path = os.path.join(output_directory, prediction_filename)
        np.save(prediction_path, predicted_poses_numpy)
        print(f"→ saved {prediction_path}")

        # Save lightweight meta alongside predictions
        try:
            import json as _json

            # Resolve model-fitted normalizer stats (means/stds) for denorm during canonicalization
            dn_means3 = None
            dn_stds3 = None
            dn_fitted = False
            try:
                core = model.module if hasattr(model, "module") else model
                norm = getattr(getattr(core, "temporal", None), "normalizer", None)
                if norm is not None and bool(getattr(norm, "fitted", False)):
                    m = getattr(norm, "means", None)
                    s = getattr(norm, "stds", None)
                    if m is not None and s is not None:
                        m_list = m.tolist() if hasattr(m, "tolist") else list(m)
                        s_list = s.tolist() if hasattr(s, "tolist") else list(s)
                        if len(m_list) >= 3 and len(s_list) >= 3:
                            dn_means3 = [float(m_list[0]), float(m_list[1]), float(m_list[2])]
                            dn_stds3 = [float(s_list[0]), float(s_list[1]), float(s_list[2])]
                            dn_fitted = True
            except Exception:
                pass
            # Fallback to model buffers if available
            if not dn_fitted:
                try:
                    core = model.module if hasattr(model, "module") else model
                    dn_means_buf = getattr(core, "_dnorm_means", None)
                    dn_stds_buf = getattr(core, "_dnorm_scales", None)
                    dn_fit_buf = getattr(core, "_dnorm_fitted", None)
                    if dn_means_buf is not None and dn_stds_buf is not None:
                        dn_means3 = dn_means_buf[:3].tolist() if hasattr(dn_means_buf, "tolist") else None
                        dn_stds3 = dn_stds_buf[:3].tolist() if hasattr(dn_stds_buf, "tolist") else None
                        if dn_fit_buf is not None:
                            try:
                                dn_fitted = bool(int(dn_fit_buf.item()))
                            except Exception:
                                dn_fitted = True
                except Exception:
                    pass

            # Compute evaluation + diagnostics and gather metrics bundle first
            eval_bundle = _compute_evaluation_metrics(
                predicted_poses_numpy,
                sample,
                dataset_index,
                output_directory,
                cfg,
                dataset_ref=dataset,
                t_scale=t_scale,
                debug_sample_count=num_windows,
                model_outfmt=model_outfmt,
                dnorm_means=dn_means3,
                dnorm_stds=dn_stds3,
                dnorm_fitted=dn_fitted,
            )

            meta = {
                "frames": _as_int_list(sample.get("frame_ids", None)) or None,
                "anchor_global": int(eval_bundle.get("anchor_global_report")),
                "anchor_local": int(eval_bundle.get("anchor_local_report")),
                "K_sig": str(sample.get("K").shape) if sample.get("K") is not None else None,
                "temporal": temporal_backend_type,
                "ckpt_path": checkpoint_path,
                "t_scale": float(t_scale),
                "canon_t_std_used": float(eval_bundle.get("canon_t_std_used", 1.0)),
                # Aggregated metrics (parity with training logs)
                "rot_deg_mean": eval_bundle.get("rot_deg_mean"),
                "trans_l2_mean": eval_bundle.get("trans_l2_mean"),
                "canon_id_rot_deg_mean": eval_bundle.get("canon_id_rot_deg_mean"),
                # Index alignment summary
                "ctx_len": eval_bundle.get("ctx_len"),
                "anchor_idx": eval_bundle.get("anchor_idx"),
                "pred_range": [eval_bundle.get("pred_start"), eval_bundle.get("pred_stop_exclusive") - 1],
                "H": eval_bundle.get("H"),
                "pred_mode": eval_bundle.get("pred_mode"),
                "used_path": eval_bundle.get("used_path"),
            }
            meta_path = os.path.join(
                output_directory,
                "pred_meta.json" if not multi else f"pred_meta_{dataset_index:03d}.json",
            )
            with open(meta_path, "w") as f:
                _json.dump(meta, f, indent=2)
        except Exception:
            if bool(getattr(getattr(cfg, "debug", None) or {}, "raise_on_error", False)):
                raise
            pass
        # (Metrics + diagnostics already computed and saved above)

    # Display inference summary
    _display_inference_summary(cfg, temporal_backend_type)
    if bool(cfg.infer.teacher_forced) and temporal_backend_type == "ar_transformer":
        print("[bold yellow]teacher_forced[/bold yellow]=true (AR, GT tokens used)")


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
