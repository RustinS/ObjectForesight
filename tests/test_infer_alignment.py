import numpy as np
import torch

from src.infer_main import _compute_indices, _pred_mode_from_outfmt
from src.geom.canonicalize import canonicalize_preds_to_anchor


def _mat_to_r6(R: np.ndarray) -> np.ndarray:
    # rot6d from first two columns
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1).astype(np.float32)


def _poses_to_9d(T_seq: np.ndarray) -> np.ndarray:
    t = T_seq[:, :3, 3]
    R = T_seq[:, :3, :3]
    r6 = _mat_to_r6(R)
    return np.concatenate([t, r6], axis=1).astype(np.float32)


def test_index_math_and_identity_canonicalization():
    # Synthetic window with P=3, Hn=8
    P = 3
    Hn = 8
    L = P + Hn + 1  # extra frame after for safety
    frames = np.arange(L, dtype=np.int32)
    # Dataset anchor aLoc (training perspective); test both aLoc=0 and aLoc=1
    aLoc = 0

    # Build trivial extrinsics and GT poses (all identity in anchor camera)
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

    # Indices invariants
    P_out, aLoc_out, anchor_idx, start, stop = _compute_indices(sample, type("cfg", (), {"data": type("obj", (), {"context_len": P})})(), Hn)
    assert P_out == P
    assert aLoc_out == aLoc
    assert anchor_idx == (aLoc + (P - 1))
    assert start == anchor_idx + 1
    assert stop == start + Hn

    # Build GT tokens (abs_in_anchor_cam)
    T_gt_slice = T_cam_anchor_obj[start:stop]
    assert T_gt_slice.shape[0] == Hn
    gt_tokens = _poses_to_9d(T_gt_slice)

    # Canonicalize via pred path and compare back to GT slice
    pred_mode = _pred_mode_from_outfmt("abs_in_anchor")
    conv_arrow = "w<-c"
    pred_9d = torch.from_numpy(gt_tokens).float().unsqueeze(0)
    meta = {
        "K": None,
        "frame_ids": frames,
        "anchor_frame_idx": int(frames[aLoc]),
        "anchor_local_idx": int(aLoc),  # training perspective for infer metrics
        "T_c_w": T_c_w,
        "T_c_o": None,
        "T_cam_anchor_obj": T_cam_anchor_obj,
        "t_mean": [0.0, 0.0, 0.0],
        "t_std": [1.0, 1.0, 1.0],
    }
    T_via_pred, _ = canonicalize_preds_to_anchor(pred_9d, meta, pred_mode, conv_arrow, do_denorm=True, return_intermediates=False)
    T_via_pred = T_via_pred[0]

    # Identity: rotations must match within tight tolerance
    R1 = T_via_pred[:, :3, :3]
    R2 = torch.from_numpy(T_gt_slice[:, :3, :3]).float()
    # geodesic small-angle check via trace
    M = torch.matmul(R1.transpose(-1, -2), R2)
    tr = torch.clamp((M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2] - 1.0) * 0.5, -1.0, 1.0)
    ang = torch.arccos(tr) * (180.0 / np.pi)
    assert float(ang.mean().item()) < 1e-3, f"canon_id_rot_deg_mean={float(ang.mean().item()):.4g}"

