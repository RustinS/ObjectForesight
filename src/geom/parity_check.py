from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

from .se3_ops import geodesic_distance_deg


def _to_tensor_T(arr: Any) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        T = arr.float()
    else:
        T = torch.from_numpy(np.asarray(arr)).float()
    if T.dim() == 2:
        T = T.unsqueeze(0)
    return T


def _first_mismatch(Ta: torch.Tensor, Tb: torch.Tensor, atol_rot_deg: float, atol_t: float) -> tuple[int | None, float, float]:
    H = min(Ta.shape[0], Tb.shape[0])
    if H == 0:
        return None, 0.0, 0.0
    Ra, ta = Ta[:H, :3, :3], Ta[:H, :3, 3]
    Rb, tb = Tb[:H, :3, :3], Tb[:H, :3, 3]
    rerr = geodesic_distance_deg(Ra, Rb)
    terr = torch.linalg.norm(ta - tb, dim=-1)
    for k in range(H):
        if float(rerr[k]) > atol_rot_deg or float(terr[k]) > atol_t:
            return int(k), float(rerr[k]), float(terr[k])
    return None, float(rerr.max()), float(terr.max())


def parity_assert(train_json: str, other_json: str, atol_rot_deg: float = 1e-3, atol_t: float = 1e-6) -> None:
    with open(train_json, "r") as f:
        A = json.load(f)
    with open(other_json, "r") as f:
        B = json.load(f)

    # Compare metadata
    keys = ["pred_mode", "pred_repr", "frames", "anchor_global", "anchor_local", "K_sig"]
    for k in keys:
        va = A.get(k)
        vb = B.get(k)
        if va != vb:
            raise AssertionError(f"parity mismatch: meta.{k}: train={va} other={vb}")

    # Compare transforms for first sample (per spec)
    Ta = _to_tensor_T(A.get("T_camA_obj_pred"))
    Tb = _to_tensor_T(B.get("T_camA_obj_pred"))
    idx, r, t = _first_mismatch(Ta, Tb, atol_rot_deg, atol_t)
    if idx is not None:
        raise AssertionError(f"parity mismatch: first k={idx} exceeds tol (rot={r:.6f}deg, trans={t:.6e}m)")

    # If no mismatch, print concise OK line
    print("parity OK: infer matches train canonicalization")
