from __future__ import annotations

import json
import os
from typing import Literal

import numpy as np
import torch
from omegaconf import DictConfig

from ....utils.logger import epoch_table
from ....utils.logger import rprint as print


def apply_sanity_checks(
    predicted_poses: torch.Tensor,
    sample: dict,
    cfg: DictConfig,
    namespace: Literal["infer", "viz"] = "infer",
) -> torch.Tensor:
    """Apply sanity checks to predicted poses.

    Reads flags from cfg.<namespace>.* with top-level fallback for backwards compatibility.
    """
    try:
        # sanity_pred_equals_gt
        equals_gt = False
        try:
            equals_gt = bool(getattr(getattr(cfg, namespace), "sanity_pred_equals_gt", False))
        except Exception:
            equals_gt = False
        if not equals_gt:
            equals_gt = bool(getattr(cfg, "sanity_pred_equals_gt", False))

        # sanity_pred_unit_scale
        try:
            unit_scale = float(getattr(getattr(cfg, namespace), "sanity_pred_unit_scale", 1.0))
        except Exception:
            unit_scale = float(getattr(cfg, "sanity_pred_unit_scale", 1.0))

        if equals_gt:
            all_gt = torch.from_numpy(np.asarray(sample.get("T_cam_anchor_obj"))).float()
            if namespace == "viz":
                try:
                    a_loc = int(sample.get("anchor_local_idx", 0))
                except Exception:
                    a_loc = 0
                prediction_horizon = int(predicted_poses.shape[0])
                predicted_poses = all_gt[a_loc + 1 : a_loc + 1 + prediction_horizon]
            else:  # infer path behavior
                predicted_poses = all_gt[: predicted_poses.shape[0]]
        elif abs(unit_scale - 1.0) > 1e-9:
            predicted_poses = predicted_poses.clone()
            predicted_poses[:, :3, 3] = predicted_poses[:, :3, 3] * float(unit_scale)
    except Exception:
        # keep original predictions on any issue
        pass

    return predicted_poses


def display_diagnostics(
    predicted_poses: torch.Tensor,
    ground_truth_poses: torch.Tensor,
    sample: dict,
    canonicalization_info: dict,
    statistics: dict,
    rotation_errors: torch.Tensor,
    translation_errors: torch.Tensor,
    reprojection_errors: dict,
    directionality_stats: dict,
    window_index: int,
    cfg: DictConfig,
    report_frames=None,
    report_anchor_global=None,
    report_anchor_local=None,
    report_anchor_tag="first-H",
) -> None:
    """Display comprehensive diagnostics."""
    try:
        from ....utils.validate import build_diagnostics_block, infer_image_size_from_K

        # Display directionality correlation
        if isinstance(directionality_stats, dict) and "dir_corr_mean" in directionality_stats:
            print(f"dir_corr • mean={directionality_stats['dir_corr_mean']:.3f} min={directionality_stats.get('dir_corr_min', float('nan')):.3f}")

        # Build diagnostics block
        try:
            isolation_mode = getattr(cfg, "isolate_rt", None) or getattr(getattr(cfg, "viz", None) or getattr(cfg, "infer", None) or cfg, "isolate_rt", None)
            camera_intrinsics = reprojection_errors.get("scaled_intrinsics")
            if camera_intrinsics is not None:
                image_height, image_width = infer_image_size_from_K(camera_intrinsics)
            else:
                image_height, image_width = float("nan"), float("nan")

            diagnostics = build_diagnostics_block(predicted_poses, ground_truth_poses, camera_intrinsics, None, canonicalization_info, isolation_mode)

            # Display diagnostics
            print(
                f"[bold cyan]Diagnostics[/bold cyan] • pred_mode={canonicalization_info.get('pred_mode', '?')} • repr={canonicalization_info.get('pred_repr', 'se3(t+rot6d)')}"
            )
            print(
                f"cos_dir_t_mean={diagnostics['cos_dir_t_mean']:.3f} cos_dir_t_min={diagnostics['cos_dir_t_min']:.3f} • z_bad_count={int(diagnostics['z_bad_count_pred'])} • skipped_for_reproj={int(diagnostics['reproj_skipped'])}"
            )
            print(
                f"ATE(rot)={diagnostics['ate_rot_mean_deg']:.3f} ATE(trans)={diagnostics['ate_trans_mean']:.4f} • RPE(rot)={diagnostics['rpe_rot_mean_deg']:.3f} RPE(trans)={diagnostics['rpe_trans_mean']:.4f}"
            )
            print(
                f"reproj_px_mean@{int(image_width)}x{int(image_height)}={diagnostics['reproj_px_mean']:.2f} • reproj_norm_mean={diagnostics['reproj_norm_mean']:.5f} • t_norm ratio_med={diagnostics['t_norm_ratio_med']:.3f}"
            )
            print(
                f"isolate[predR_gtT]: ATE(rot)={diagnostics['ATE(rot)_predR_gtT']:.3f} • reproj_px_mean={diagnostics['reproj_px_mean_predR_gtT']:.2f} • z_bad={int(diagnostics['z_bad_count_predR_gtT'])}"
            )
            print(
                f"isolate[gtR_predT]: ATE(trans)={diagnostics['ATE(trans)_gtR_predT']:.4f} • reproj_px_mean={diagnostics['reproj_px_mean_gtR_predT']:.2f} • z_bad={int(diagnostics['z_bad_count_gtR_predT'])}"
            )
        except Exception:
            pass

        # Display frame and anchor information (use reporting overrides if provided)
        frames = report_frames if report_frames is not None else sample.get("frame_ids", None)
        a_glob = int(report_anchor_global if report_anchor_global is not None else sample.get("anchor_frame_idx", 0))
        a_loc = int(report_anchor_local if report_anchor_local is not None else sample.get("anchor_local_idx", 0))
        print(f"[report:{report_anchor_tag}] frames={list(map(int, frames)) if frames is not None else 'n/a'} • anchor_global={a_glob} • anchor_local={a_loc}")

        # Display canonicalization information
        intrinsics_signature = reprojection_errors.get("K_signature", "n/a")
        print(f"pred_mode={canonicalization_info.get('pred_mode', '?')} • repr={canonicalization_info.get('pred_repr', '?')} • K_sig={intrinsics_signature}")

        # Display rotation hygiene
        if "ortho_max_pred" in canonicalization_info and "det_err_max_pred" in canonicalization_info:
            print(f"R_hygiene • ortho_max_pred={canonicalization_info['ortho_max_pred']:.2e} det_err_max_pred={canonicalization_info['det_err_max_pred']:.2e}")

        # Display translation norms
        if all(k in canonicalization_info for k in ("t_pred_raw_norm_mean", "t_pred_denorm_norm_mean", "t_gt_norm_mean", "scale_ratio_med")):
            print(
                f"t_norms • raw={canonicalization_info['t_pred_raw_norm_mean']:.3f} denorm={canonicalization_info['t_pred_denorm_norm_mean']:.3f} gt={canonicalization_info['t_gt_norm_mean']:.3f} • ratio_med={canonicalization_info['scale_ratio_med']:.3f}"
            )

        # Display final metrics table
        epoch_table(
            title=f"Window {window_index} Canonicalized Diagnostics",
            metrics={
                "ate_rot_mean_deg": float(statistics["ate_rot_mean_deg"]),
                "ate_trans_mean": float(statistics["ate_trans_mean"]),
                "rpe_rot_mean_deg": float(statistics["rpe_rot_mean_deg"]),
                "rpe_trans_mean": float(statistics["rpe_trans_mean"]),
                "rot_deg_err_mean": float(rotation_errors.mean().item()),
                "trans_l2_err_mean": float(translation_errors.mean().item()),
                "reproj_px_mean": float(reprojection_errors.get("reprojection_px", 0.0)),
                "reproj_norm_mean": float(reprojection_errors.get("reprojection_norm", 0.0)),
                "origin_px": float(reprojection_errors.get("origin_px", 0.0)),
                "dir_corr_mean": float(directionality_stats.get("dir_corr_mean", float("nan"))),
                "dir_corr_min": float(directionality_stats.get("dir_corr_min", float("nan"))),
            },
            prec=int(getattr(cfg.log, "precision", 4)) if hasattr(cfg, "log") else 4,
        )
    except Exception:
        pass


def save_error_metrics(
    rotation_errors: torch.Tensor,
    translation_errors: torch.Tensor,
    statistics: dict,
    reprojection_errors: dict,
    directionality_stats: dict,
    window_index: int,
    output_directory: str,
    debug_sample_count: int,
) -> None:
    """Save error metrics to JSON file."""
    try:
        # Convert to CPU lists
        rot_list = [float(x) for x in rotation_errors.detach().cpu().tolist()]
        trans_list = [float(x) for x in translation_errors.detach().cpu().tolist()]
        # New: drift per step and linear slope of error vs time
        try:
            import numpy as _np

            T = int(len(trans_list))
            denom = float(max(T - 1, 1))
            trans_drift = (trans_list[-1] - trans_list[0]) / denom if T >= 1 else float("nan")
            rot_drift = (rot_list[-1] - rot_list[0]) / denom if T >= 1 else float("nan")
            t_idx = _np.arange(T, dtype=_np.float64) if T >= 1 else _np.array([0.0], dtype=_np.float64)
            # Use polyfit degree=1 to get slope and intercept
            if T >= 2:
                a_trans, _ = _np.polyfit(t_idx, _np.asarray(trans_list, dtype=_np.float64), 1)
                a_rot, _ = _np.polyfit(t_idx, _np.asarray(rot_list, dtype=_np.float64), 1)
            else:
                a_trans, a_rot = float("nan"), float("nan")
        except Exception:
            trans_drift = float("nan")
            rot_drift = float("nan")
            a_trans = float("nan")
            a_rot = float("nan")

        error_metrics = {
            "rot_deg": rot_list,
            "trans_l2": trans_list,
            # New scalar metrics
            "trans_drift_per_step": float(trans_drift),
            "rot_drift_per_step": float(rot_drift),
            "trans_error_slope": float(a_trans),
            "rot_error_slope": float(a_rot),
            **statistics,
            "reproj_px_mean": float(reprojection_errors.get("reprojection_px", 0.0)),
            "origin_px": float(reprojection_errors.get("origin_px", 0.0)),
            "dir_corr_mean": float(directionality_stats.get("dir_corr_mean", float("nan"))),
            "dir_corr_min": float(directionality_stats.get("dir_corr_min", float("nan"))),
        }

        error_filename = f"errors_{window_index:03d}.json" if debug_sample_count > 1 else "errors.json"
        error_path = os.path.join(output_directory, error_filename)

        with open(error_path, "w") as f:
            json.dump(error_metrics, f, indent=2)
    except Exception:
        pass


def compute_train_style_metrics(predicted_poses: np.ndarray, sample: dict, window_index: int, cfg: DictConfig) -> None:
    """Compute train-style parameterization metrics."""
    try:
        from ....geom.se3_ops import geodesic_distance_deg as _geod
        from ....geom.se3_ops import rot6d_to_matrix as _r6_to_R

        ground_truth_poses = sample.get("target_future")
        if ground_truth_poses is not None:
            ground_truth_poses = np.asarray(ground_truth_poses)
            horizon_length = min(int(ground_truth_poses.shape[0]), int(predicted_poses.shape[0]))

            # Extract translation and rotation components
            predicted_translations = torch.from_numpy(predicted_poses[:horizon_length, :3]).float()
            predicted_rotations_6d = torch.from_numpy(predicted_poses[:horizon_length, 3:9]).float()
            ground_truth_translations = torch.from_numpy(ground_truth_poses[:horizon_length, :3]).float()
            ground_truth_rotations_6d = torch.from_numpy(ground_truth_poses[:horizon_length, 3:9]).float()

            # Convert to rotation matrices and compute errors
            predicted_rotation_matrices = _r6_to_R(predicted_rotations_6d)
            ground_truth_rotation_matrices = _r6_to_R(ground_truth_rotations_6d)
            rotation_degrees_train = _geod(predicted_rotation_matrices, ground_truth_rotation_matrices)
            translation_l2_train = torch.linalg.norm(predicted_translations - ground_truth_translations, dim=-1)

            # Display train-style metrics
            epoch_table(
                title="Train-style Param Metrics (Window 0)",
                metrics={
                    "rot_deg_mean": float(rotation_degrees_train.mean().item()),
                    "trans_l2_mean": float(translation_l2_train.mean().item()),
                },
                prec=int(getattr(cfg.log, "precision", 4)) if hasattr(cfg, "log") else 4,
            )
    except Exception:
        pass


def create_ground_truth_overlay(
    anchor_image: np.ndarray,
    scaled_intrinsics: np.ndarray,
    sample: dict,
    axis_length: float,
) -> np.ndarray:
    """Create ground truth trajectory overlay on anchor image."""
    from ....utils.viz import overlay_axes_on_image

    # Use dataset-provided re-expressed poses directly
    ground_truth_poses = torch.from_numpy(sample["T_cam_anchor_obj"]).float()

    # Identity transformation: poses already in camera frame
    world_to_camera_identity = np.eye(4, dtype=np.float32)

    # Build trajectory labels with anchor=0, context negative, future positive
    try:
        a_loc = int(sample.get("anchor_local_idx", 0))
    except Exception:
        a_loc = 0
    # Labels are offsets relative to the anchor index: ...,-2,-1,0,1,2,...
    ground_truth_labels = []
    L_total = int(ground_truth_poses.shape[0])
    for i in range(L_total):
        off = i - a_loc
        ground_truth_labels.append("0" if off == 0 else str(off))

    return overlay_axes_on_image(
        anchor_image,
        scaled_intrinsics,
        world_to_camera_identity,
        ground_truth_poses,
        axis_length=axis_length,
        annotate_indices=False,
        traj_labels=ground_truth_labels,
    )


def create_prediction_overlay(
    anchor_image: np.ndarray,
    scaled_intrinsics: np.ndarray,
    sample: dict,
    predicted_poses: torch.Tensor,
    axis_length: float,
) -> np.ndarray:
    """Create prediction trajectory overlay on anchor image."""
    from ....utils.viz import overlay_axes_on_image

    # Get anchor local index
    try:
        anchor_local_index = int(sample.get("anchor_local_idx", 0))
    except Exception:
        anchor_local_index = 0

    # Include the initial anchor in the predicted overlay for parity with GT
    ground_truth_poses = torch.from_numpy(sample["T_cam_anchor_obj"]).float()
    anchor_pose_only = ground_truth_poses[anchor_local_index : anchor_local_index + 1]
    prediction_overlay_poses = torch.cat([anchor_pose_only, predicted_poses], dim=0)

    # Labels for prediction overlay: anchor as 0, then 1..H
    prediction_labels = ["0"] + [str(i + 1) for i in range(predicted_poses.shape[0])]

    # Identity transformation: poses already in camera frame
    world_to_camera_identity = np.eye(4, dtype=np.float32)

    return overlay_axes_on_image(
        anchor_image,
        scaled_intrinsics,
        world_to_camera_identity,
        prediction_overlay_poses,
        axis_length=axis_length,
        annotate_indices=False,
        traj_labels=prediction_labels,
    )


def display_per_step_evaluation(evaluation_results: dict) -> None:
    """Display per-step evaluation summary."""
    try:
        from ....utils.geometry import project_points as _proj
        from ....utils.validate import pose_errors_deg_m

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
                print(
                    "Pred T_camA_obj[1]:\n"
                    + np.array2string(
                        predicted_step[0].detach().cpu().numpy(),
                        formatter={"float_kind": lambda x: f"{x: .4f}"},
                    )
                )
                print(
                    "GT   T_camA_obj[1]:\n"
                    + np.array2string(
                        ground_truth_step[0].detach().cpu().numpy(),
                        formatter={"float_kind": lambda x: f"{x: .4f}"},
                    )
                )
    except Exception:
        pass
