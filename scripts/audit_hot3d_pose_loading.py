#!/usr/bin/env python3
"""Audit HOT3D pose loading vs EPIC conventions.

Runs lightweight checks that HOT3DClipsDataset returns SE(3) fields with the same
meaning as SceneSequenceDataset (EPIC):
  - T_c_w: world<-camera (camera->world), i.e. "c2w" / "w<-c"
  - T_wc0: camera<-world (world->camera) at the first window frame
  - T_c_o: camera<-object (object pose in camera)
  - T_cam_anchor_obj: camera(anchor)<-object for all window frames

Usage (from repo root):
  uv run python scripts/audit_hot3d_pose_loading.py --n 5
"""

from __future__ import annotations

import argparse
import math
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from src.geom.se3_ops import geodesic_distance_deg, invert_T, compose_T
from src.utils.data_utils import get_dataset


def _to_torch(T: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(T)).float()


def _rt_err(Ta: torch.Tensor, Tb: torch.Tensor) -> tuple[float, float]:
    """Return (rot_err_deg_max, trans_err_max)."""
    Ra, ta = Ta[:, :3, :3], Ta[:, :3, 3]
    Rb, tb = Tb[:, :3, :3], Tb[:, :3, 3]
    rot = geodesic_distance_deg(Ra, Rb)
    trans = torch.linalg.norm(ta - tb, dim=-1)
    return float(rot.max().item()), float(trans.max().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="conf/debug.yaml", help="Path to a config yaml (default: conf/debug.yaml)")
    ap.add_argument("--n", type=int, default=5, help="Number of random windows to audit")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    # Avoid globally resolving the full config here: debug.yaml contains Hydra-only resolvers like `${eval:...}`.
    # get_dataset() only depends on cfg.data, so leaving the rest unresolved is fine.

    if str(cfg.data.get("dataset_name", "")).lower() != "hot3d":
        raise ValueError(f"Expected data.dataset_name=hot3d in {args.config}, got {cfg.data.get('dataset_name')}")

    ds = get_dataset(cfg)
    n_total = len(ds)
    if n_total == 0:
        raise ValueError("Dataset is empty")

    rng = random.Random(int(args.seed))
    indices = [rng.randrange(n_total) for _ in range(max(1, int(args.n)))]

    for i, idx in enumerate(indices):
        sample = ds[idx]

        conv = str(sample.get("extrinsics_convention", ""))
        if conv.lower() not in {"c2w", "w<-c"}:
            raise AssertionError(f"[{i}] unexpected extrinsics_convention={conv!r} (expected c2w / w<-c)")

        T_c_w = _to_torch(sample["T_c_w"])
        T_c_o = _to_torch(sample["T_c_o"])
        T_camA_obj = _to_torch(sample["T_cam_anchor_obj"])

        anchor_local = int(sample.get("anchor_local_idx", 0))
        T_c_w0 = _to_torch(sample["T_c_w0"]).unsqueeze(0)
        T_wc0 = _to_torch(sample["T_wc0"]).unsqueeze(0)

        I4 = torch.eye(4)
        inv_err = float(torch.linalg.norm((T_wc0 @ T_c_w0 - I4).reshape(-1)).item())

        # EPIC-style re-expression: T_camA_obj[k] = inv(T_c_w[anchor]) @ (T_c_w[k] @ T_c_o[k])
        T_w_o = compose_T(T_c_w, T_c_o)
        T_camA_w = invert_T(T_c_w[anchor_local : anchor_local + 1]).repeat(T_c_w.shape[0], 1, 1)
        T_camA_obj_epic = compose_T(T_camA_w, T_w_o)

        rot_epic, trans_epic = _rt_err(T_camA_obj_epic, T_camA_obj)

        # Anchor self-consistency: at k==anchor, T_camA_obj == T_c_o
        Ta = T_camA_obj[anchor_local : anchor_local + 1]
        Tk = T_c_o[anchor_local : anchor_local + 1]
        rot_anchor = float(geodesic_distance_deg(Ta[:, :3, :3], Tk[:, :3, :3])[0].item())
        trans_anchor = float(torch.linalg.norm(Ta[:, :3, 3] - Tk[:, :3, 3]).item())

        z_min = float(T_c_o[:, 2, 3].min().item())
        z_med = float(T_c_o[:, 2, 3].median().item())

        vid = sample.get("video_id")
        oid = sample.get("object_id")
        print(
            f"[{i}] idx={idx} vid={vid} obj={oid} anchor_local={anchor_local} "
            f"inv_err={inv_err:.2e} epic_vs_ds=(rot_max={rot_epic:.3g}°, trans_max={trans_epic:.2e}) "
            f"anchor=(rot={rot_anchor:.3g}°, trans={trans_anchor:.2e}) z=(min={z_min:.3g}, med={z_med:.3g})"
        )

        if not math.isfinite(inv_err) or inv_err > 5e-3:
            raise AssertionError(f"[{i}] T_wc0 @ T_c_w0 not identity (err={inv_err:.3e})")
        if rot_epic > 0.25 or trans_epic > 2e-3:
            raise AssertionError(f"[{i}] HOT3D poses do not match EPIC-style re-expression (rot={rot_epic:.3g}°, trans={trans_epic:.3e})")
        if rot_anchor > 0.25 or trans_anchor > 2e-3:
            raise AssertionError(f"[{i}] anchor pose mismatch vs T_c_o (rot={rot_anchor:.3g}°, trans={trans_anchor:.3e})")

        if z_med < 0.05:
            raise AssertionError(f"[{i}] suspicious depth: median z={z_med:.4f} m (expected object in front of camera)")


if __name__ == "__main__":
    main()
