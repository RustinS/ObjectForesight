from __future__ import annotations

import json
import os

import cv2
import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .data.video_io import read_frame
from .geom.canonicalize import build_projection_signature, canonicalize_preds_to_anchor
from .models.poser_v1.utils.inferviz_common import (
    apply_sanity_checks,
    create_ground_truth_overlay,
    create_prediction_overlay,
    display_diagnostics,
    display_per_step_evaluation,
    save_error_metrics,
)
from .utils.config_adapter import apply_config_adapter
from .utils.fp_audit import (
    apply_adapter,
    compare_ours_vs_fp,
    fp_signature_block,
    propose_adapter_from_tests,
    save_fp_and_ours_overlays,
)
from .utils.geometry import infer_image_size_from_K, scale_intrinsics
from .utils.logger import rprint as print
from .utils.validate import (
    K_signature,
    add_panel_label,
    ate_rpe,
    directionality_probe,
    invert_T,
    pose_errors_deg_m,
    reprojection_error_norm,
    reprojection_error_px,
    reprojection_sanity_origin_px,
    run_gt_pose_checks,
)
from .utils.viz import overlay_axes_on_image


def _pred_mode_from_outfmt(out_fmt: str) -> str:
    if out_fmt == "abs_in_anchor":
        return "abs_in_anchor_cam"
    if out_fmt == "delta_from_prev":
        return "deltas_from_prev_cam"
    if out_fmt == "delta_from_anchor":
        return "deltas_from_anchor_cam"
    return "abs_in_anchor_cam"


def _compute_indices(sample: dict, cfg: DictConfig, Hn: int) -> tuple[int, int, int, int, int]:
    """
    Compute consistent indices for anchor and prediction slicing.
    - aLoc: canonicalization anchor (dataset-provided; defines camera(anchor) frame).
    - anchor_idx: last pre-anchor context local index (P-1), used for logging/overlay only.
    - start_gt = P, stop_gt = P + H: GT horizon (window-local, strictly after the last pre-anchor context).

    Note: Dataset sample['context_len'] includes the anchor (P+1). For parity with
    train_main/infer_main, use cfg.data.context_len as P (pre-anchor count).
    """
    if not hasattr(cfg.data, "context_len"):
        raise ValueError("cfg.data.context_len must be set in the config for viz_main (no default fallback).")
    P = int(cfg.data.context_len)
    aLoc = int(sample.get("anchor_local_idx", 0))
    assert P >= 1
    # Last pre-anchor context index for logging/overlay
    anchor_idx = max(P - 1, 0)
    # Ground-truth slice for prediction horizon is strictly window-local [P : P+H]
    start_gt = P
    stop_gt = P + int(Hn)
    return P, aLoc, anchor_idx, start_gt, stop_gt


def _create_visualization_dataset(cfg: DictConfig):
    """
    Create dataset for visualization based on configuration.

    This function delegates to the centralized get_dataset() factory to ensure
    consistency with training and inference pipelines.

    Args:
        cfg: Configuration dictionary containing data parameters

    Returns:
        Dataset instance (SceneSequenceDataset, SceneSequenceDatasetSynth, or HOT3DClipsDataset)
    """
    from .utils.data_utils import get_dataset

    # Validate required config fields
    ds_name = str(getattr(cfg.data, "dataset_name", "")).lower()
    if ds_name in {"epic", "post_train", "synth", "hot3d"}:
        if not hasattr(cfg.data, "H"):
            raise ValueError("cfg.data.H must be set in the config for viz_main (no default fallback).")
        if not hasattr(cfg.data, "context_len"):
            raise ValueError("cfg.data.context_len must be set in the config for viz_main (no default fallback).")

    return get_dataset(cfg)


def _resolve_prediction_path(cfg: DictConfig, window_index: int, debug_sample_count: int) -> str | None:
    """
    Resolve prediction file path with fallback options.

    Args:
        cfg: Configuration dictionary
        window_index: Current window index
        debug_sample_count: Number of debug samples

    Returns:
        Path to prediction file or None if not found
    """
    # 1) Explicit override via cfg.viz.pred_path (file or directory)
    try:
        override = getattr(cfg.viz, "pred_path", None) if hasattr(cfg, "viz") else None
    except Exception:
        override = None
    if override:
        override = os.path.expanduser(os.path.expandvars(str(override)))
        if os.path.isdir(override):
            candidate = os.path.join(override, f"pred_{window_index:03d}.npy")
            if os.path.exists(candidate):
                return candidate
            candidate = os.path.join(override, "pred.npy")
            if os.path.exists(candidate):
                return candidate
        elif os.path.isfile(override):
            # If a specific file is provided (e.g., pred.npy), prefer enumerated sibling if it exists
            base_dir = os.path.dirname(override)
            candidate = os.path.join(base_dir, f"pred_{window_index:03d}.npy")
            if os.path.exists(candidate):
                return candidate
            return override

    # 2) Build candidate base directories
    base_dirs = []
    try:
        if hasattr(cfg, "train") and hasattr(cfg.train, "out_dir"):
            out_dir = str(cfg.train.out_dir)
            base_dirs.append(os.path.join(out_dir, "preds"))
            # Add debug and non-debug variants to bridge default vs debug configs
            if "debug" not in out_dir.split(os.sep):
                base_dirs.append(os.path.join(out_dir, "debug", "preds"))
            else:
                non_debug = out_dir.replace(f"{os.sep}debug", "")
                base_dirs.append(os.path.join(non_debug, "preds"))
    except Exception:
        pass
    try:
        ckpt_path = getattr(cfg.infer, "ckpt", None) if hasattr(cfg, "infer") else None
        if isinstance(ckpt_path, str):
            ckpt_path = os.path.expanduser(os.path.expandvars(ckpt_path))
            # If ckpt is under .../checkpoints/, infer its train out_dir
            if os.path.splitext(ckpt_path)[1] == ".pt":
                ckpt_dir = os.path.dirname(ckpt_path)
                train_out = os.path.dirname(ckpt_dir)
                base_dirs.append(os.path.join(train_out, "preds"))
    except Exception:
        pass

    # Deduplicate while preserving order
    seen = set()
    unique_base_dirs = []
    for d in base_dirs:
        d_norm = os.path.normpath(d)
        if d_norm not in seen:
            unique_base_dirs.append(d_norm)
            seen.add(d_norm)

    # 3) Try candidates in order
    filenames = [f"pred_{window_index:03d}.npy", "pred.npy"]
    for base in unique_base_dirs:
        for name in filenames:
            candidate = os.path.join(base, name)
            if os.path.exists(candidate):
                return candidate

    return None


def _load_anchor_image(sample: dict, preferred_local_idx: int | None = None) -> np.ndarray | None:
    """
    Load the RGB frame used as the overlay background.
    By default, prefer the first prediction frame (H[0]) when its local index is provided;
    otherwise fall back to the dataset anchor (last context frame) and finally to image0.

    Supports both video files (EPIC) and HOT3D tar archives.

    Args:
        sample: Dataset sample
        preferred_local_idx: Window-local index to prefer when selecting the RGB frame

    Returns:
        Anchor image array or None if not found
    """
    from .data.video_io import read_any_frame_from_hot3d_tar, read_frame_from_hot3d_tar

    rgb_path = sample.get("rgb_path", None)
    frame_ids = sample.get("frame_ids", None)

    # Determine target frame
    target_local = None
    if preferred_local_idx is not None and frame_ids is not None and 0 <= preferred_local_idx < len(frame_ids):
        target_local = int(preferred_local_idx)
    elif "anchor_local_idx" in sample and frame_ids is not None:
        anchor_local = int(sample.get("anchor_local_idx", 0))
        if 0 <= anchor_local < len(frame_ids):
            target_local = anchor_local

    # HOT3D tar file support: check if rgb_path is a .tar file
    if isinstance(rgb_path, str) and rgb_path.endswith(".tar"):
        try:
            # For HOT3D, frame_ids are string timestamps
            if target_local is not None and frame_ids is not None and target_local < len(frame_ids):
                frame_id = str(frame_ids[target_local])
                img = read_frame_from_hot3d_tar(rgb_path, frame_id)
                if img is not None:
                    return img
            # Fallback: read any frame from tar
            img = read_any_frame_from_hot3d_tar(rgb_path)
            if img is not None:
                return img
        except Exception:
            pass

    # Standard video file (EPIC/3DManip)
    try:
        if isinstance(rgb_path, str):
            if target_local is not None and frame_ids is not None:
                target_global_idx = int(frame_ids[target_local])
            elif "anchor_frame_idx" in sample:
                target_global_idx = int(sample.get("anchor_frame_idx", 0))
            else:
                target_global_idx = int(sample.get("image0_idx", 0)) if "image0_idx" in sample else 0
            img = read_frame(rgb_path, target_global_idx)
            if img is not None:
                return img
    except Exception:
        pass

    # Fallbacks: preloaded image or image0_path/index
    anchor_image = sample.get("image0", None)
    if anchor_image is not None:
        return anchor_image
    image_path = sample.get("image0_path")
    image_index = int(sample.get("image0_idx", 0)) if "image0_idx" in sample else 0
    if image_path is not None:
        try:
            img = read_frame(image_path, image_index)
            if img is not None:
                return img
        except Exception:
            pass

    return None


def _resolve_anchor_and_start(sample: dict, P: int) -> tuple[int, int]:
    """
    Returns (anchor_idx, start_idx) in window-local indexing.
    anchor_idx: local index of the anchor (last context frame)
    start_idx : first predicted frame index (anchor_idx + 1)
    Priority:
      (a) sample['anchor_local_idx'] (already the anchor used by train/infer)
      (b) map sample['anchor_frame_idx'] into local via sample['frame_ids']
      (c) fallback: treat anchor as last of P context frames starting at window_start_local_idx
    """
    if sample.get("anchor_local_idx") is not None:
        anchor_idx = int(sample["anchor_local_idx"])
        return anchor_idx, anchor_idx + 1
    fids = sample.get("frame_ids")
    g_anchor = sample.get("anchor_frame_idx")
    if fids is not None and g_anchor is not None:
        import numpy as _np

        loc = _np.where(_np.asarray(fids) == int(g_anchor))[0]
        if len(loc) > 0:
            anchor_idx = int(loc[0])
            return anchor_idx, anchor_idx + 1
    aLoc0 = int(sample.get("window_start_local_idx", 0))
    anchor_idx = aLoc0 + max(P - 1, 0)
    return anchor_idx, anchor_idx + 1


def _run_identity_delta_sanity_check(predicted_camera_poses: torch.Tensor, ground_truth_poses: torch.Tensor, anchor_location: int, cfg: DictConfig) -> None:
    """
    Run identity-delta sanity check: ΔT_pred = I ⇒ Pred ≈ Anchor GT

    Args:
        predicted_camera_poses: Predicted camera poses [H, 4, 4]
        ground_truth_poses: Ground truth poses [H, 4, 4]
        anchor_location: Anchor frame location
        cfg: Configuration object
    """
    try:
        if not bool(getattr(cfg.viz, "debug_identity_delta", False)):
            return

        prediction_horizon = int(predicted_camera_poses.shape[0])
        anchor_base_poses = ground_truth_poses[anchor_location : anchor_location + 1].repeat(prediction_horizon, 1, 1)

        rotation_error, translation_error = pose_errors_deg_m(anchor_base_poses, predicted_camera_poses)

        print(f"identity-delta • rot_deg(mean)={rotation_error.mean():.3f} trans(mean)={translation_error.mean():.4f} (expect near-zero at anchor only)")
    except Exception:
        pass


def _run_translation_scale_probe(predicted_camera_poses: torch.Tensor, ground_truth_poses: torch.Tensor, cfg: DictConfig) -> None:
    """
    Run translation scale probe: set t_pred_after_denorm = t_gt, keep predicted R

    Args:
        predicted_camera_poses: Predicted camera poses [H, 4, 4]
        ground_truth_poses: Ground truth poses [H, 4, 4]
        cfg: Configuration object
    """
    try:
        if not bool(getattr(cfg.viz, "debug_translation_scale_probe", False)):
            return

        # Create modified poses with GT translation but predicted rotation
        modified_poses = predicted_camera_poses.clone()
        modified_poses[:, :3, 3] = ground_truth_poses[:, :3, 3]

        # Compute errors
        rotation_error, translation_error = pose_errors_deg_m(modified_poses, ground_truth_poses)

        print(f"t-scale-probe • trans(mean)={translation_error.mean():.4f} (should collapse) • rot_deg(mean) unaffected={rotation_error.mean():.3f}")
    except Exception:
        pass


def _display_per_step_evaluation(evaluation_results: dict) -> None:
    """
    Display per-step evaluation summary.

    Args:
        evaluation_results: Dictionary containing evaluation results
    """
    try:
        from .utils.geometry import project_points as _proj

        # Get predicted poses from the evaluation results
        predicted_poses = evaluation_results.get("predicted_poses")
        ground_truth_poses = evaluation_results["aligned_ground_truth"]
        final_intrinsics = evaluation_results["final_intrinsics"]

        if predicted_poses is None:
            return

        intrinsics_tensor = torch.from_numpy(np.asarray(final_intrinsics)).float()
        origin_points = torch.zeros((1, 3), dtype=predicted_poses.dtype)
        prediction_horizon = int(predicted_poses.shape[0])

        for step_index in range(prediction_horizon):
            predicted_step = predicted_poses[step_index : step_index + 1]
            ground_truth_step = ground_truth_poses[step_index : step_index + 1]

            # Per-step errors in anchor camera
            rotation_error, translation_error = pose_errors_deg_m(predicted_step, ground_truth_step)

            # Per-step origin reprojection pixel error
            predicted_projection = _proj(intrinsics_tensor, predicted_step[0], origin_points)[0]
            ground_truth_projection = _proj(intrinsics_tensor, ground_truth_step[0], origin_points)[0]
            pixel_error = float(torch.linalg.norm(predicted_projection - ground_truth_projection).item())

            # Depth values
            predicted_depth = float(predicted_step[0, 2, 3].item())
            ground_truth_depth = float(ground_truth_step[0, 2, 3].item())

            print(
                f"step {step_index + 1}: z_pred={predicted_depth:.3f} z_gt={ground_truth_depth:.3f} rot_deg={float(rotation_error.mean().item()):.3f} trans={float(translation_error.mean().item()):.4f} origin_px={pixel_error:.2f}"
            )

            # Print full matrices for step 1 only (concise dump)
            if step_index == 0:
                print("Pred T_camA_obj[1]:\n" + np.array2string(predicted_step[0].detach().cpu().numpy(), formatter={"float_kind": lambda x: f"{x: .4f}"}))
                print("GT   T_camA_obj[1]:\n" + np.array2string(ground_truth_step[0].detach().cpu().numpy(), formatter={"float_kind": lambda x: f"{x: .4f}"}))
    except Exception:
        pass


def _run_fp_audit(cfg: DictConfig, sample: dict, predicted_camera_poses: torch.Tensor, anchor_image: np.ndarray, camera_intrinsics: np.ndarray) -> None:
    """
    Run FoundationPose audit if enabled.

    Args:
        cfg: Configuration dictionary
        sample: Dataset sample
        predicted_camera_poses: Predicted camera poses
        anchor_image: Anchor image
        camera_intrinsics: Camera intrinsics
    """
    try:
        if not bool(getattr(cfg, "fp_audit", False)):
            return

        # Derive FP ob_in_cam dir and anchor frame id
        ob_in_cam_dir = os.path.join(sample.get("object_dir"), "foundationpose", "ob_in_cam")
        a_frame = int(sample.get("anchor_frame_idx", int(sample.get("frame_ids")[0])))

        # Build FP signature from raw pose and print
        sig_block, M_raw = fp_signature_block(ob_in_cam_dir, a_frame, camera_intrinsics)
        print(sig_block)

        if M_raw is not None:
            # Save FP-only and ours axis overlays for the same frame
            ground_truth_poses = torch.from_numpy(sample["T_cam_anchor_obj"]).float()
            anchor_local_index = int(sample.get("anchor_local_idx", 0))
            T_anchor_only_np = ground_truth_poses[anchor_local_index : anchor_local_index + 1].detach().cpu().numpy()[0]
            fp_png, ours_png = save_fp_and_ours_overlays(anchor_image, camera_intrinsics, T_anchor_only_np, M_raw, cfg.viz.save_dir, axis_length=cfg.viz.axis_length)
            print(f"fp_axis_raw → {fp_png}")
            print(f"ours_axis → {ours_png}")

            # Compare rotations and pixel endpoints
            cmp = compare_ours_vs_fp(camera_intrinsics, T_anchor_only_np, M_raw, anchor_image.shape[:2])
            print(
                f"FP vs Ours • rot_diff_deg={cmp['rot_diff_deg']:.2f} • px(origin,X,Y,Z)={cmp['px_diff']['origin']:.1f},{cmp['px_diff']['X']:.1f},{cmp['px_diff']['Y']:.1f},{cmp['px_diff']['Z']:.1f} • axis_flip={cmp['axis_flip']}"
            )

            # Two-path consistency (use existing GT check metrics on window)
            try:
                gt2 = run_gt_pose_checks(sample, print)
                two_rot = float(gt2.get("two_path_rot_max_deg", float("nan")))
                two_tr = float(gt2.get("two_path_trans_max", float("nan")))
            except Exception:
                two_rot, two_tr = float("nan"), float("nan")

            # Propose minimal adapter and apply
            adapter = propose_adapter_from_tests(camera_intrinsics, M_raw)
            axis_fix = (
                "flipY+flipZ" if (adapter.get("flip_y") and adapter.get("flip_z")) else ("flipY" if adapter.get("flip_y") else ("flipZ" if adapter.get("flip_z") else "none"))
            )
            pose_before = "T_cam_obj"
            pose_after = "T_cam_obj"
            applied_inverse = bool(adapter.get("inverse_pose", False))
            storage_transposed = bool(adapter.get("transpose_on_load", False))

            # Apply adapter for downstream usage (ingest only)
            apply_adapter(adapter)
            verdict = (
                "MATCH"
                if cmp["rot_diff_deg"] < 1.0 and max(cmp["px_diff"].values()) < 2.0
                else ("FIXED" if (axis_fix != "none" or applied_inverse or storage_transposed) else "NEEDS ATTENTION")
            )

            # Summarize FP Audit Report block
            from .utils.fp_audit import build_audit_report

            report = build_audit_report(
                pose_before,
                pose_after,
                storage_transposed,
                axis_fix,
                applied_inverse,
                camera_intrinsics,
                two_rot,
                two_tr,
                {
                    "origin": cmp["px_diff"]["origin"],
                    "X": cmp["px_diff"]["X"],
                    "Y": cmp["px_diff"]["Y"],
                    "Z": cmp["px_diff"]["Z"],
                },
                verdict,
            )
            print(report)
    except Exception as _e:
        print(f"[yellow]fp_audit[/yellow] skipped: {_e}")


def _create_gif_export(
    sample: dict,
    predicted_camera_poses: torch.Tensor,
    anchor_image: np.ndarray,
    camera_intrinsics: np.ndarray,
    cfg: DictConfig,
    debug_sample_count: int,
    window_index: int,
    indices: dict,
) -> None:
    """
    Create GIF export with per-step GT and Pred side-by-side frames.

    Args:
        sample: Dataset sample
        predicted_camera_poses: Predicted camera poses
        anchor_image: Anchor image
        camera_intrinsics: Camera intrinsics
        cfg: Configuration dictionary
        debug_sample_count: Number of debug samples
        window_index: Window index
    """
    try:
        frames = []
        Hn = int(predicted_camera_poses.shape[0])
        # indices unpack
        anchor_idx = int(indices["anchor_idx"])
        start = int(indices["start"])

        # Attempt to read images and extrinsics for each step; fall back to anchor image if unavailable
        img_base_path = sample.get("rgb_path", None)
        fids_np = np.asarray(sample.get("frame_ids")) if sample.get("frame_ids") is not None else None
        T_c_w_win = torch.from_numpy(np.asarray(sample.get("T_c_w"))).float() if sample.get("T_c_w") is not None else None

        # Handle extrinsics convention
        conv = str(sample.get("extrinsics_convention", "c2w"))
        # a_loc_int no longer used; using anchor_idx explicitly

        # Precompute camera(anchor)->world under convention
        if T_c_w_win is not None and T_c_w_win.shape[0] > anchor_idx:
            if conv == "c2w":
                T_cw_anchor = T_c_w_win[anchor_idx]
            else:  # w2c
                T_cw_anchor = invert_T(T_c_w_win[anchor_idx : anchor_idx + 1])[0]
        else:
            T_cw_anchor = torch.eye(4)

        # Build single-step visuals (GT_k vs Pred_k) over actual frame k+1 relative to anchor
        for k in range(Hn):
            idx_frame = start + k

            # Select per-frame image if available
            img_k = anchor_image
            if img_base_path is not None and fids_np is not None and idx_frame < fids_np.shape[0]:
                try:
                    # HOT3D tar file support
                    if img_base_path.endswith(".tar"):
                        from .data.video_io import read_frame_from_hot3d_tar

                        frame_id = str(fids_np[idx_frame])
                        loaded = read_frame_from_hot3d_tar(img_base_path, frame_id)
                        if loaded is not None:
                            img_k = loaded
                    else:
                        img_k = read_frame(img_base_path, int(fids_np[idx_frame]))
                except Exception:
                    img_k = anchor_image

            # Scale K to this image if needed
            K_k = camera_intrinsics
            try:
                Hk, Wk = infer_image_size_from_K(camera_intrinsics)
                Himg, Wimg = img_k.shape[:2]
                if (Hk != Himg) or (Wk != Wimg):
                    from .utils.geometry import scale_intrinsics

                    K_k = scale_intrinsics(camera_intrinsics, (Hk, Wk), (Himg, Wimg))
            except Exception:
                K_k = camera_intrinsics

            # Determine camera(k) <- camera(anchor) mapping under convention
            if T_c_w_win is not None and T_c_w_win.shape[0] > idx_frame:
                if conv == "c2w":
                    T_wc_k = invert_T(T_c_w_win[idx_frame : idx_frame + 1])[0]
                else:  # w2c
                    T_wc_k = T_c_w_win[idx_frame]
                T_ck_ca = (T_wc_k @ T_cw_anchor).detach().cpu().numpy()
            else:
                T_ck_ca = np.eye(4)

            # Compose single-pose overlays in frame k+1
            ground_truth_poses = torch.from_numpy(sample["T_cam_anchor_obj"]).float()
            gt_slice = ground_truth_poses[idx_frame : idx_frame + 1]
            pred_slice = predicted_camera_poses[k : k + 1]
            gt_k = overlay_axes_on_image(img_k, K_k, T_ck_ca, gt_slice, axis_length=cfg.viz.axis_length, traj_labels=[str(k + 1)])
            pred_k = overlay_axes_on_image(img_k, K_k, T_ck_ca, pred_slice, axis_length=cfg.viz.axis_length, traj_labels=[str(k + 1)])

            # Per-frame side-by-side quick numeric check for step 1 in camera(k)
            if k == 0:
                try:
                    from .utils.geometry import project_points as _proj

                    Kt = torch.from_numpy(np.asarray(K_k)).float()
                    origin_pts = torch.zeros((1, 3), dtype=pred_slice.dtype)
                    # Map to camera(k) frame: apply T_ck_ca to anchor-camera poses
                    T_ck_ca_t = torch.from_numpy(np.asarray(T_ck_ca)).float()
                    Tp_camk = (T_ck_ca_t @ pred_slice[0]).unsqueeze(0)
                    Tg_camk = (T_ck_ca_t @ gt_slice[0]).unsqueeze(0)
                    uvp = _proj(Kt, Tp_camk[0], origin_pts)[0]
                    uvg = _proj(Kt, Tg_camk[0], origin_pts)[0]
                    px_camk = float(torch.linalg.norm(uvp - uvg).item())
                    zp = float(Tp_camk[0, 2, 3].item())
                    zg = float(Tg_camk[0, 2, 3].item())
                    rot_kc, trans_kc = pose_errors_deg_m(Tp_camk, Tg_camk)
                    print(
                        f"frame[{idx_frame}] step 1 in cam(k): z_pred={zp:.3f} z_gt={zg:.3f} rot_deg={float(rot_kc.mean().item()):.3f} trans={float(trans_kc.mean().item()):.4f} origin_px={px_camk:.2f}"
                    )
                except Exception:
                    pass

            gt_k_l = add_panel_label(gt_k, f"GT[{k + 1}]", color=(0, 200, 0))
            pred_k_l = add_panel_label(pred_k, f"Pred[{k + 1}]", color=(200, 0, 200))
            side_k = np.concatenate([gt_k_l, pred_k_l], axis=1)
            frames.append(cv2.cvtColor(side_k, cv2.COLOR_BGR2RGB))

        gif_path = os.path.join(cfg.viz.save_dir, "overlay.gif" if debug_sample_count == 1 else f"overlay_{window_index:03d}.gif")
        imageio.mimsave(gif_path, frames, duration=0.6)
        print(f"[bold]gif[/bold] → {gif_path}")
    except Exception as e:
        print(f"[yellow]gif export skipped[/yellow]: {e}")


@hydra.main(config_path="../conf", config_name="debug", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main visualization function for PoserV1 model predictions.

    This function supports visualization for both DiT (Diffusion Transformer) and AR (Autoregressive)
    backends. It loads predictions from inference and creates visual overlays showing predicted
    vs ground truth pose trajectories.

    Args:
        cfg: Hydra configuration dictionary
    """
    # Register eval resolver BEFORE apply_config_adapter (it accesses temporal_dit.out_dim which uses ${eval:...})
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: int(eval(expr, {"__builtins__": {}}, {})))

    # Apply configuration adapters and setup
    cfg = apply_config_adapter(cfg)
    assert int(cfg.data.H) > 0, "data.H must be > 0"

    # Dataset setup - matches training and inference configuration
    dataset = _create_visualization_dataset(cfg)

    # Debug mode setup - matches training tiny_overfit behavior
    debug_mode = bool(getattr(cfg.train, "tiny_overfit", False)) if hasattr(cfg, "train") else False
    # Determine start index and number of windows to visualize (use cfg.viz.log_first_n when >0, else all)
    try:
        start_idx = int(getattr(cfg.viz, "start_idx", 0) or 0)
    except Exception:
        start_idx = 0
    start_idx = max(0, min(start_idx, len(dataset)))
    try:
        log_first_n = int(getattr(cfg.viz, "log_first_n", 0) or 0)
    except Exception:
        log_first_n = 0
    total_available = max(0, len(dataset) - start_idx)
    num_windows = total_available if log_first_n <= 0 else min(int(log_first_n), total_available)

    # Setup output directory
    if debug_mode:
        cfg.viz.save_dir = os.path.join(cfg.train.out_dir, "overlays")
    os.makedirs(cfg.viz.save_dir, exist_ok=True)

    # Visualization summaries for multi-window debugging
    visualization_summaries = []

    # Process each window for visualization
    for dataset_index in range(start_idx, start_idx + num_windows):
        sample = dataset[dataset_index]

        # Load prediction file
        prediction_path = _resolve_prediction_path(cfg, dataset_index, num_windows)
        if prediction_path is None:
            print("Prediction not found. Run inference first.")
            return

        print(f"Loading prediction from {prediction_path}")
        predicted_poses = np.load(prediction_path)  # Shape: (H, 9)
        # Load adjacent pred_meta.json to get t_scale (optional; opt-in)
        t_scale = 1.0
        try:
            pred_dir = os.path.dirname(prediction_path)
            base = os.path.basename(prediction_path)
            cand_jsons = []
            if base.startswith("pred_") and base.endswith(".npy"):
                name = os.path.splitext(base)[0]  # pred_### or pred
                suffix = name[len("pred_") :] if name != "pred" else None
                if suffix is not None and suffix.isdigit():
                    cand_jsons.append(os.path.join(pred_dir, f"pred_meta_{suffix}.json"))
            cand_jsons.append(os.path.join(pred_dir, "pred_meta.json"))
            meta_json = None
            for p in cand_jsons:
                if os.path.exists(p):
                    with open(p, "r") as f:
                        meta_json = json.load(f)
                    break
            use_meta_t_scale = bool(getattr(cfg.viz, "use_pred_meta_t_scale", False))
            if isinstance(meta_json, dict) and use_meta_t_scale:
                t_scale = float(meta_json.get("t_scale", 1.0))
            if not use_meta_t_scale and isinstance(meta_json, dict) and "t_scale" in meta_json:
                print("[viz] Ignoring pred_meta.t_scale to preserve parity (set viz.use_pred_meta_t_scale=true to use it).")
        except Exception:
            t_scale = 1.0

        # Indices (A)
        Hn = int(predicted_poses.shape[0])
        P, aLoc, anchor_idx, start_gt, stop_gt = _compute_indices(sample, cfg, Hn)
        print(f"ctx_len={P} anchor_idx={anchor_idx} pred_range=[{start_gt}, {stop_gt - 1}] H={Hn}")
        # Make anchor consistent and build a corrected sample view
        import numpy as _np

        frames = sample.get("frame_ids")
        if frames is not None:
            anchor_global = int(_np.asarray(frames)[anchor_idx])
        else:
            anchor_global = int(sample.get("anchor_frame_idx", 0))
        ds_anchor_local = int(sample.get("anchor_local_idx", 0))
        if frames is not None and 0 <= ds_anchor_local < len(frames):
            ds_anchor_global = int(_np.asarray(frames)[ds_anchor_local])
        else:
            ds_anchor_global = int(sample.get("anchor_frame_idx", anchor_global))

        # Canonicalize predictions to camera(anchor)<-object coordinate system (D)
        try:
            ds_ref = dataset.dataset if hasattr(dataset, "dataset") else dataset
            out_fmt = getattr(ds_ref, "output_format", "abs_in_anchor")
        except Exception:
            out_fmt = "abs_in_anchor"
        pred_mode = _pred_mode_from_outfmt(str(out_fmt))
        conv_arrow = "w<-c"

        canonicalization_metadata = {
            "K": sample.get("K"),
            "frame_ids": sample.get("frame_ids"),
            "anchor_frame_idx": int(sample.get("anchor_frame_idx", 0)),
            # IMPORTANT: use dataset-defined anchor for canonicalization (parity with infer_main)
            "anchor_local_idx": int(aLoc),
            "T_c_w": sample.get("T_c_w"),
            "T_c_o": sample.get("T_c_o"),
            "T_cam_anchor_obj": sample.get("T_cam_anchor_obj"),
            "t_mean": [0.0, 0.0, 0.0],
            "t_std": [float(t_scale), float(t_scale), float(t_scale)],
        }

        print(build_projection_signature(canonicalization_metadata, pred_mode, conv_arrow, canonicalization_metadata["t_mean"], canonicalization_metadata["t_std"]))

        # Convert to tensor and canonicalize (B, D)
        predicted_poses_tensor = torch.from_numpy(predicted_poses).float().unsqueeze(0)
        absolute_poses, canonicalization_info = canonicalize_preds_to_anchor(
            predicted_poses_tensor, canonicalization_metadata, pred_mode, conv_arrow, do_denorm=True, return_intermediates=True
        )
        # Frame signature one-liner (parity with train/infer)
        try:
            frames_sig = canonicalization_metadata.get("frame_ids")
            frames_str = str(list(map(int, frames_sig))) if frames_sig is not None else "[]"
        except Exception:
            frames_str = "[]"
        try:
            anchor_sig = f"(g={int(canonicalization_metadata.get('anchor_frame_idx', 0))},l={int(canonicalization_metadata.get('anchor_local_idx', 0))})"
        except Exception:
            anchor_sig = "(g=0,l=0)"
        print(f"signature • pred_mode={pred_mode} extrinsics={conv_arrow} frames={frames_str} anchor={anchor_sig} repr=se3(t+rot6d)")
        predicted_camera_poses = absolute_poses[0]
        T_cam_anchor_full = torch.from_numpy(np.asarray(sample["T_cam_anchor_obj"])).float()

        # Load and prepare image for visualization
        anchor_image = _load_anchor_image(sample, preferred_local_idx=start_gt)
        if anchor_image is None:
            print("[yellow]image0[/yellow] missing; cannot render overlay.")
            continue

        camera_intrinsics = sample["K"]
        scaled_intrinsics = camera_intrinsics
        # Scale K to match the actual anchor image size (most robust for overlays)
        try:
            Hk, Wk = infer_image_size_from_K(camera_intrinsics)
            Himg, Wimg = int(anchor_image.shape[0]), int(anchor_image.shape[1])
            if (Himg > 0 and Wimg > 0) and ((Hk, Wk) != (Himg, Wimg)):
                scaled_intrinsics = scale_intrinsics(camera_intrinsics, (Hk, Wk), (Himg, Wimg))
        except Exception:
            pass

        # Log frame and anchor information
        frame_ids = sample.get("frame_ids")
        print(f"frames={list(map(int, frame_ids)) if frame_ids is not None else 'n/a'} • anchor_global={anchor_global} • anchor_local={anchor_idx}")
        print(f"dataset anchor_global={ds_anchor_global} • dataset anchor_local={ds_anchor_local}")
        print(f"K_sig={K_signature(camera_intrinsics)}")

        # Apply sanity checks to predicted poses
        predicted_camera_poses = apply_sanity_checks(predicted_camera_poses, sample, cfg, namespace="viz")

        # slice GT exactly like infer_main (A, D)
        Hn = int(predicted_camera_poses.shape[0])
        T_gt_all = T_cam_anchor_full
        if stop_gt > T_gt_all.shape[0]:
            print(f"[yellow]Index error[/yellow] GT slice OOB: anchor_idx={anchor_idx} start_gt={start_gt} stop_gt={stop_gt} Hn={Hn} len(T_gt_all)={int(T_gt_all.shape[0])}")
        # IMPORTANT: GT horizon is always window-local [P : P+H]; anchor_local_idx affects only frame re-expression,
        # not which GT time steps are compared to predictions.
        T_gt_slice = T_gt_all[start_gt:stop_gt]
        assert int(T_gt_slice.shape[0]) == int(Hn), (
            f"GT slice len {int(T_gt_slice.shape[0])} != H {int(Hn)} (P={P}, aLoc={aLoc}, start_gt={start_gt}, stop_gt={stop_gt}, total={int(T_gt_all.shape[0])})"
        )

        # Rebase predictions/GT to the first horizon frame (parity with infer_main)
        T_pred_metrics = predicted_camera_poses.clone()
        T_gt_metrics = T_gt_slice.clone()
        overlay_rebase_matrix = None
        overlay_frame_local = int(ds_anchor_local)
        overlay_frame_global = int(ds_anchor_global)
        T_c_w_np = sample.get("T_c_w")
        if T_c_w_np is not None:
            T_c_w_tensor = torch.from_numpy(np.asarray(T_c_w_np)).float()
            if T_c_w_tensor.dim() == 3 and 0 <= start_gt < T_c_w_tensor.shape[0] and 0 <= ds_anchor_local < T_c_w_tensor.shape[0]:
                T_anchor = T_c_w_tensor[ds_anchor_local]
                T_view = T_c_w_tensor[start_gt]
                overlay_rebase_matrix = torch.linalg.inv(T_view) @ T_anchor
                overlay_frame_local = int(start_gt)
                if frame_ids is not None and 0 <= start_gt < len(frame_ids):
                    overlay_frame_global = int(frame_ids[start_gt])
                T_pred_metrics = torch.matmul(overlay_rebase_matrix.unsqueeze(0), T_pred_metrics)
                T_gt_metrics = torch.matmul(overlay_rebase_matrix.unsqueeze(0), T_gt_metrics)
        if overlay_rebase_matrix is not None:
            print(f"[viz] rebasing metrics/overlays to frame g={overlay_frame_global} (local={overlay_frame_local})")
        else:
            print(f"[viz] metrics/overlays remain in dataset anchor frame g={overlay_frame_global} (local={overlay_frame_local})")

        # metrics
        rot_err, trans_err = pose_errors_deg_m(T_pred_metrics, T_gt_metrics)
        stats = ate_rpe(T_pred_metrics, T_gt_metrics)
        # reprojection using the same scaled_intrinsics as above
        reproj_px = reprojection_error_px(scaled_intrinsics, T_pred_metrics, T_gt_metrics)
        reproj_norm = reprojection_error_norm(scaled_intrinsics, T_pred_metrics, T_gt_metrics)
        origin_px = reprojection_sanity_origin_px(scaled_intrinsics, T_pred_metrics, T_gt_metrics)
        # directionality (parity with infer)
        dir_stats = directionality_probe(
            torch.from_numpy(np.asarray(sample.get("T_c_w"))).float(),
            torch.from_numpy(np.asarray(sample.get("T_c_o"))).float(),
            torch.from_numpy(np.asarray(sample.get("T_cam_anchor_obj"))).float(),
            int(anchor_idx),
            str(sample.get("extrinsics_convention", "c2w")),
        )
        # display/save using the same helpers as infer_main
        display_diagnostics(
            T_pred_metrics,
            T_gt_metrics,
            sample,
            None,
            stats,
            rot_err,
            trans_err,
            {
                "reprojection_px": float(reproj_px),
                "reprojection_norm": float(reproj_norm),
                "origin_px": float(origin_px),
                "K_signature": K_signature(camera_intrinsics),
                "scaled_intrinsics": scaled_intrinsics,
            },
            dir_stats,
            dataset_index,
            cfg,
        )
        save_error_metrics(
            rot_err,
            trans_err,
            stats,
            {"reprojection_px": float(reproj_px), "reprojection_norm": float(reproj_norm), "origin_px": float(origin_px)},
            dir_stats,
            dataset_index,
            cfg.viz.save_dir,
            num_windows,
        )

        # Display prediction diagnostics
        if canonicalization_info and "ortho_max_pred" in canonicalization_info and "det_err_max_pred" in canonicalization_info:
            print(f"R_hygiene • ortho_max_pred={canonicalization_info['ortho_max_pred']:.2e} det_err_max_pred={canonicalization_info['det_err_max_pred']:.2e}")

        if canonicalization_info and all(k in canonicalization_info for k in ("t_pred_raw_norm_mean", "t_pred_denorm_norm_mean", "t_gt_norm_mean", "scale_ratio_med")):
            print(
                f"t_norms • raw={canonicalization_info['t_pred_raw_norm_mean']:.3f} denorm={canonicalization_info['t_pred_denorm_norm_mean']:.3f} gt={canonicalization_info['t_gt_norm_mean']:.3f} • ratio_med={canonicalization_info['scale_ratio_med']:.3f}"
            )

        if overlay_rebase_matrix is not None:
            gt_all_for_overlay = torch.matmul(overlay_rebase_matrix.unsqueeze(0), T_cam_anchor_full)
            pred_for_overlay = T_pred_metrics
        else:
            gt_all_for_overlay = T_cam_anchor_full
            pred_for_overlay = predicted_camera_poses

        if overlay_rebase_matrix is not None and 0 <= overlay_frame_local < gt_all_for_overlay.shape[0]:
            traj_labels = []
            for idx in range(gt_all_for_overlay.shape[0]):
                offset = idx - overlay_frame_local
                traj_labels.append("0" if offset == 0 else f"{offset}")
            ground_truth_overlay = overlay_axes_on_image(
                anchor_image,
                scaled_intrinsics,
                np.eye(4, dtype=np.float32),
                gt_all_for_overlay,
                axis_length=cfg.viz.axis_length,
                annotate_indices=False,
                traj_labels=traj_labels,
            )
            # Pred overlay: include anchor (label 0) followed by H predicted steps (labels 1..H)
            pred_labels = ["0"] + [str(i + 1) for i in range(pred_for_overlay.shape[0])]
            anchor_pose_for_pred = gt_all_for_overlay[overlay_frame_local : overlay_frame_local + 1]
            pred_traj = torch.cat([anchor_pose_for_pred, pred_for_overlay], dim=0)
            prediction_overlay = overlay_axes_on_image(
                anchor_image,
                scaled_intrinsics,
                np.eye(4, dtype=np.float32),
                pred_traj,
                axis_length=cfg.viz.axis_length,
                annotate_indices=False,
                traj_labels=pred_labels,
            )
        else:
            ground_truth_overlay = create_ground_truth_overlay(anchor_image, scaled_intrinsics, sample, cfg.viz.axis_length)
            prediction_overlay = create_prediction_overlay(anchor_image, scaled_intrinsics, sample, predicted_camera_poses, cfg.viz.axis_length)

        # Also overlay previous P context poses (projected in anchor camera) on both panels
        if overlay_rebase_matrix is None:
            try:
                # Only overlay pre-anchor contexts (P), not the anchor token (sample.context_len may be P+1)
                P_pre = int(getattr(cfg.data, "context_len", int(sample.get("context_len", 0)) or 0))
                if P_pre > 0:
                    ctx_T = sample.get("context_T_cam_anchor_obj")
                    if ctx_T is not None:
                        from .utils.viz import overlay_axes_on_image as _overlay_axes

                        ctx_arr = np.asarray(ctx_T)
                        # Guard: context_T may include the anchor at the end; select only first P_pre pre-anchor frames
                        if ctx_arr.ndim == 3 and ctx_arr.shape[0] >= P_pre:
                            ctx_arr = ctx_arr[:P_pre]
                        ctx_poses = torch.from_numpy(ctx_arr).float()
                        I_wc = np.eye(4, dtype=np.float32)
                        ctx_labels = [str(i - P_pre) for i in range(int(ctx_poses.shape[0]))]
                        ground_truth_overlay = _overlay_axes(
                            ground_truth_overlay,
                            scaled_intrinsics,
                            I_wc,
                            ctx_poses,
                            axis_length=cfg.viz.axis_length,
                            annotate_indices=False,
                            traj_labels=ctx_labels,
                        )
                        prediction_overlay = _overlay_axes(
                            prediction_overlay,
                            scaled_intrinsics,
                            I_wc,
                            ctx_poses,
                            axis_length=cfg.viz.axis_length,
                            annotate_indices=False,
                            traj_labels=ctx_labels,
                        )
                    else:
                        print("[dim]context_T_cam_anchor_obj missing; skipping context overlay.[/dim]")
            except Exception:
                pass

        # Optional pose checks and lightweight metrics
        pose_checks_enabled = bool(getattr(cfg, "check_poses", False)) or bool(getattr(cfg.viz, "check_poses", False))

        if pose_checks_enabled:
            # Display per-step evaluation summary using aligned GT slice
            evaluation_results = {
                "predicted_poses": T_pred_metrics,
                "aligned_ground_truth": T_gt_metrics,
                "final_intrinsics": scaled_intrinsics,
            }
            display_per_step_evaluation(evaluation_results)
            # Identity-delta sanity: use full GT and anchor index
            _run_identity_delta_sanity_check(
                predicted_camera_poses,
                T_gt_all,
                int(anchor_idx),
                cfg,
            )
            # Translation scale probe with sliced GT
            _run_translation_scale_probe(
                predicted_camera_poses,
                T_gt_slice,
                cfg,
            )
            # Append concise summary
            visualization_summaries.append(
                {
                    "idx": dataset_index,
                    "gt_checks": run_gt_pose_checks(sample, print),
                    "pred_stats": {
                        "rot_deg": [float(x) for x in rot_err.detach().cpu().tolist()],
                        "trans_l2": [float(x) for x in trans_err.detach().cpu().tolist()],
                        **stats,
                        "reproj_px_mean": float(reproj_px),
                        "origin_px": float(origin_px),
                    },
                    "pred_info": canonicalization_info,
                }
            )

        # fp_audit: FoundationPose I/O audit and FP-only renderer (anchor frame)
        _run_fp_audit(cfg, sample, predicted_camera_poses, anchor_image, camera_intrinsics)

        # parity: overlay must use canonicalized T_camA_obj_pred
        # Quick sanity: origin_px variance vs RPE(trans)
        try:
            ground_truth_poses = torch.from_numpy(sample["T_cam_anchor_obj"]).float()
            T_gt_local = ground_truth_poses[anchor_idx + 1 : anchor_idx + 1 + predicted_camera_poses.shape[0]]
            rpe_stats = ate_rpe(predicted_camera_poses, T_gt_local)
            origin_px = float(reprojection_sanity_origin_px(camera_intrinsics, predicted_camera_poses, T_gt_local))
            # Estimate variance over projected origins via small window around mean
            # (reuse origin_px as a proxy; true var requires collecting points)
            if (origin_px < 1.0) and (rpe_stats.get("rpe_trans_mean", 0.0) > 0.005):
                print("[yellow]WARNING[/yellow] parity: overlay likely not using canonicalized T (origin_px < 1px while RPE(trans) > 0.005m)")
        except Exception:
            pass

        # Side-by-side with small labels for clarity
        img_gt_labeled = add_panel_label(ground_truth_overlay, "GT", color=(0, 200, 0))
        img_pred_labeled = add_panel_label(prediction_overlay, "Pred", color=(200, 0, 200))
        side = np.concatenate([img_gt_labeled, img_pred_labeled], axis=1)
        if num_windows == 1 and start_idx == 0:
            out_path = os.path.join(cfg.viz.save_dir, "overlay.png")
        else:
            out_path = os.path.join(cfg.viz.save_dir, f"overlay_{dataset_index:03d}.png")
        cv2.imwrite(out_path, side)
        print(
            f"[bold]overlay[/bold] → {out_path} • img0 with H={cfg.data.H} poses • anchor_mode={sample.get('anchor_mode')} a={overlay_frame_global} (local={overlay_frame_local})"
        )

        # ---- GIF export: per-step GT and Pred side-by-side frames (rendered on each frame's image) ----
        _create_gif_export(
            sample,
            predicted_camera_poses,
            anchor_image,
            camera_intrinsics,
            cfg,
            (1 if (num_windows == 1 and start_idx == 0) else 2),
            dataset_index,
            indices={"P": P, "anchor_idx": anchor_idx, "start": start_gt, "stop": stop_gt},
        )

    # Save a brief summary only when debugging multiple windows
    if num_windows > 1:
        try:
            with open(os.path.join(cfg.viz.save_dir, "summary.json"), "w") as f:
                json.dump({"windows": visualization_summaries}, f, indent=2)
        except Exception:
            pass


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
