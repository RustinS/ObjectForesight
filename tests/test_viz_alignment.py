import numpy as np
import torch

from src.infer_main import _compute_indices as infer_compute_indices, _pred_mode_from_outfmt
from src.viz_main import _compute_indices as viz_compute_indices
from src.geom.canonicalize import canonicalize_preds_to_anchor


def _mat_to_r6(R: np.ndarray) -> np.ndarray:
	# rot6d from first two columns
	return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1).astype(np.float32)


def _poses_to_9d(T_seq: np.ndarray) -> np.ndarray:
	t = T_seq[:, :3, 3]
	R = T_seq[:, :3, :3]
	r6 = _mat_to_r6(R)
	return np.concatenate([t, r6], axis=1).astype(np.float32)


def test_viz_indices_match_infer_and_gt_slice_parity():
	# Synthetic window with P=2, Hn=5
	P = 2
	Hn = 5
	L = P + Hn + 1  # one extra for safety
	frames = np.arange(L, dtype=np.int32)
	# Dataset anchor from training perspective
	aLoc = 1  # also tests non-zero anchor

	# Identity extrinsics and GT poses (all identity in anchor camera)
	T_eye = np.eye(4, dtype=np.float32)
	T_cam_anchor_obj = np.stack([T_eye.copy() for _ in range(L)], axis=0).astype(np.float32)
	T_c_w = np.stack([T_eye.copy() for _ in range(L)], axis=0).astype(np.float32)

	# Sample consistent with dataset output
	sample = {
		"context_len": int(P),
		"frame_ids": frames,
		"anchor_frame_idx": int(frames[aLoc]) if 0 <= aLoc < L else 0,
		"anchor_local_idx": int(aLoc),
		"T_c_w": T_c_w,
		"T_cam_anchor_obj": T_cam_anchor_obj,
		"extrinsics_convention": "c2w",
	}
	cfg = type("cfg", (), {"data": type("obj", (), {"context_len": P})})()

	# Indices from infer and viz must match
	P_i, a_i, anchor_idx_i, start_i, stop_i = infer_compute_indices(sample, cfg, Hn)
	P_v, a_v, anchor_idx_v, start_v, stop_v = viz_compute_indices(sample, cfg, Hn)
	assert (P_i, a_i, anchor_idx_i, start_i, stop_i) == (P_v, a_v, anchor_idx_v, start_v, stop_v)

	# GT slices used by both should be identical
	T_slice_infer = T_cam_anchor_obj[start_i:stop_i]
	T_slice_viz = T_cam_anchor_obj[start_v:stop_v]
	assert T_slice_infer.shape[0] == Hn and T_slice_viz.shape[0] == Hn
	assert np.allclose(T_slice_infer, T_slice_viz)

	# Viz canonicalization path with GT tokens should return the same GT slice
	gt_tokens = _poses_to_9d(T_slice_viz)
	pred_mode = _pred_mode_from_outfmt("abs_in_anchor")
	conv_arrow = "w<-c"
	pred_9d = torch.from_numpy(gt_tokens).float().unsqueeze(0)
	meta_v = {
		"K": None,
		"frame_ids": frames,
		# Viz uses last-context anchor for display; keep identical to indices above
		"anchor_frame_idx": int(frames[anchor_idx_v]),
		"anchor_local_idx": int(anchor_idx_v),
		"T_c_w": T_c_w,
		"T_c_o": None,
		"T_cam_anchor_obj": T_cam_anchor_obj,
		"t_mean": [0.0, 0.0, 0.0],
		"t_std": [1.0, 1.0, 1.0],
	}
	T_via_pred, _ = canonicalize_preds_to_anchor(pred_9d, meta_v, pred_mode, conv_arrow, do_denorm=True, return_intermediates=False)
	T_via_pred = T_via_pred[0]
	R1 = T_via_pred[:, :3, :3]
	R2 = torch.from_numpy(T_slice_viz[:, :3, :3]).float()
	M = torch.matmul(R1.transpose(-1, -2), R2)
	tr = torch.clamp((M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2] - 1.0) * 0.5, -1.0, 1.0)
	ang = torch.arccos(tr) * (180.0 / np.pi)
	assert float(ang.mean().item()) < 1e-3, f"viz canon identity fail: {float(ang.mean().item()):.4g} deg"


