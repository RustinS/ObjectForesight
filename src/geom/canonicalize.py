from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from ..utils.validate import K_signature
from .se3_ops import compose_T, invert_T, rot6d_to_matrix, se3_to_matrix


def _to_tensor(x, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    if x is None:
        return torch.empty(0, device=device, dtype=dtype)
    return torch.from_numpy(np.asarray(x)).to(device=device, dtype=dtype)


def _ensure_hom(T: torch.Tensor) -> torch.Tensor:
    if T.dim() != 3 or T.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (N,4,4), got {tuple(T.shape)}")
    return T


def _to_vec3(val, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(val, torch.Tensor):
        return val.to(device=device, dtype=dtype).view(1, 1, 3)
    if isinstance(val, (list, tuple)):
        return torch.tensor(val, device=device, dtype=dtype).view(1, 1, 3)
    return torch.tensor([float(val)] * 3, device=device, dtype=dtype).view(1, 1, 3)


def _fmt_vec3(val) -> str:
    vals = val if hasattr(val, "__iter__") else [val, val, val]
    return ",".join(f"{float(v):.6f}" for v in vals)


def _pack_sig(meta: Dict, pred_mode: str, extrinsics_convention: str, t_mean, t_std) -> str:
    frames = list(map(int, meta.get("frame_ids", []))) if meta.get("frame_ids") else []
    a_glob = int(meta.get("anchor_frame_idx", meta.get("anchor_global", 0)))
    a_loc = int(meta.get("anchor_local_idx", meta.get("anchor_local", 0)))
    K = meta.get("K")
    ksig = K_signature(K) if K is not None else "K:none"
    return "\n".join(
        [
            "Projection Signature:",
            f"  pred_mode={pred_mode}",
            "  repr=se3(t+rot6d)",
            f"  extrinsics_convention={extrinsics_convention}",
            f"  frames={frames}, anchor_global={a_glob}, anchor_local={a_loc}",
            f"  K_sig={ksig}",
            f"  denorm: mean=[{_fmt_vec3(t_mean)}], std=[{_fmt_vec3(t_std)}], applied_once=True",
        ]
    )


def _adjoint_compose_deltas(T_c_w: torch.Tensor, T_deltas: torch.Tensor, T_seed: torch.Tensor, anchor_idx: int) -> torch.Tensor:
    """Adjoint transport of deltas from cam(k-1) to anchor cam frame."""
    T_c_w = _ensure_hom(T_c_w)
    T_deltas = _ensure_hom(T_deltas)
    cur = _ensure_hom(T_seed[:1])
    out = []
    for s in range(T_deltas.shape[0]):
        T_camA_camPrev = compose_T(T_c_w[anchor_idx : anchor_idx + 1], invert_T(T_c_w[s : s + 1]))
        Delta_anchor = compose_T(compose_T(T_camA_camPrev, T_deltas[s : s + 1]), invert_T(T_camA_camPrev))
        cur = compose_T(Delta_anchor, cur)
        out.append(cur[0])
    return torch.stack(out, dim=0)


def _cumulative_compose(Tb: torch.Tensor) -> torch.Tensor:
    """Simple cumulative product in anchor frame."""
    cur = Tb[0:1]
    outs = [cur[0]]
    for k in range(1, Tb.shape[0]):
        cur = compose_T(cur, Tb[k : k + 1])
        outs.append(cur[0])
    return torch.stack(outs, dim=0)


def canonicalize_preds_to_anchor(
    pred_9d: torch.Tensor,
    meta: Dict,
    pred_mode: str,
    extrinsics_convention: str,
    do_denorm: bool = True,
    return_intermediates: bool = False,
) -> Tuple[torch.Tensor, Dict | None]:
    """Convert (B,H,9) predictions to (B,H,4,4) absolute poses in anchor camera frame."""
    if pred_9d.dim() != 3 or pred_9d.shape[-1] != 9:
        raise ValueError(f"pred_9d must be (B,H,9), got {tuple(pred_9d.shape)}")

    B, H, _ = pred_9d.shape
    device, dtype = pred_9d.device, pred_9d.dtype

    t_raw = pred_9d[..., :3]
    R = rot6d_to_matrix(pred_9d[..., 3:9].reshape(B * H, 6)).reshape(B, H, 3, 3)

    t_mean = meta.get("t_mean", 0.0)
    t_std = meta.get("t_std", 1.0)
    t = _to_vec3(t_std, device, dtype) * t_raw + _to_vec3(t_mean, device, dtype) if do_denorm else t_raw

    T_series = se3_to_matrix(t.reshape(B * H, 3), R.reshape(B * H, 3, 3)).reshape(B, H, 4, 4)

    anchor_local = int(meta.get("anchor_local_idx", 0))
    Tw = _to_tensor(meta.get("T_c_w"), device, dtype)
    To = _to_tensor(meta.get("T_c_o"), device, dtype)
    Tgt = _to_tensor(meta.get("T_cam_anchor_obj"), device, dtype)

    conv_arrow = extrinsics_convention.strip().lower()
    is_c2w = conv_arrow in ("w<-c", "c2w")

    T_abs_list = []
    for b in range(B):
        Tb = T_series[b]

        if pred_mode == "abs_in_anchor_cam":
            T_camA_obj = Tb

        elif pred_mode == "deltas_from_anchor_cam":
            if Tgt.numel() > 0 and 0 <= anchor_local < Tgt.shape[0]:
                T_base = Tgt[anchor_local : anchor_local + 1].repeat(H, 1, 1)
            else:
                T_base = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(H, 1, 1)
            T_camA_obj = compose_T(Tb, T_base)

        elif pred_mode == "deltas_from_prev_cam":
            if Tw.numel() > 0 and Tgt.numel() > 0:
                T_camA_obj = _adjoint_compose_deltas(Tw, Tb, Tgt[anchor_local : anchor_local + 1], anchor_local)
            else:
                T_camA_obj = _cumulative_compose(Tb)

        elif pred_mode == "relative_world_from_o0":
            if Tw.numel() == 0:
                raise ValueError("relative_world_from_o0 requires T_c_w in meta")
            T_w_c = invert_T(Tw[anchor_local : anchor_local + 1])
            if To.numel() > 0:
                T_w_oa = compose_T(T_w_c, To[anchor_local : anchor_local + 1])
            elif Tgt.numel() > 0:
                T_w_oa = compose_T(T_w_c, Tgt[anchor_local : anchor_local + 1])
            else:
                raise ValueError("relative_world_from_o0 requires T_c_o or T_cam_anchor_obj in meta")

            T_w_o_series = compose_T(T_w_oa.repeat(H, 1, 1), Tb)
            if is_c2w:
                T_w_c_anchor = invert_T(Tw[anchor_local : anchor_local + 1].repeat(H, 1, 1))
            else:
                T_w_c_anchor = Tw[anchor_local : anchor_local + 1].repeat(H, 1, 1)
            T_camA_obj = compose_T(T_w_c_anchor, T_w_o_series)

        else:
            raise ValueError(f"Unsupported pred_mode: {pred_mode}")

        T_abs_list.append(_ensure_hom(T_camA_obj))

    T_abs = torch.stack(T_abs_list, dim=0)

    if not return_intermediates:
        return T_abs, None

    info = {
        "pred_repr": "se3(t+rot6d)",
        "pred_mode": pred_mode,
        "t_raw": t_raw.detach().cpu().tolist(),
        "t_denorm": t.detach().cpu().tolist(),
        "R6d": pred_9d[..., 3:9].detach().cpu().tolist(),
        "T_camA_obj_pred": T_abs[0].detach().cpu().tolist(),
        "T_c_w": Tw.detach().cpu().tolist() if Tw.numel() > 0 else None,
        "T_cam_anchor_obj": Tgt.detach().cpu().tolist() if Tgt.numel() > 0 else None,
        "frames": list(map(int, meta["frame_ids"])) if meta.get("frame_ids") else None,
        "anchor_global": int(meta["anchor_frame_idx"]) if meta.get("anchor_frame_idx") is not None else None,
        "anchor_local": int(meta["anchor_local_idx"]) if meta.get("anchor_local_idx") is not None else None,
        "K_sig": K_signature(meta["K"]) if meta.get("K") is not None else "K:none",
        "extrinsics_convention": conv_arrow,
        "denorm": {
            "mean": list(t_mean) if isinstance(t_mean, (list, tuple)) else [float(t_mean)] * 3,
            "std": list(t_std) if isinstance(t_std, (list, tuple)) else [float(t_std)] * 3,
            "applied_once": do_denorm,
        },
        "projection_signature": _pack_sig(meta, pred_mode, conv_arrow, t_mean, t_std),
    }
    return T_abs, info


def build_projection_signature(meta: Dict, pred_mode: str, extrinsics_convention: str, t_mean, t_std) -> str:
    """Exported for train/infer/viz to print byte-identical signature."""
    return _pack_sig(meta, pred_mode, extrinsics_convention, t_mean, t_std)
