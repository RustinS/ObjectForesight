"""Shared batch/tensor utilities for train/eval/infer entrypoints."""

from __future__ import annotations

import numpy as np
import torch


def to_numpy(x):
    """Convert tensor/array to numpy array."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def slice_batch_item(batch: dict, key: str, b: int):
    """Extract item b from a batched tensor/list/array."""
    x = batch.get(key)
    if x is None:
        return None
    if isinstance(x, (torch.Tensor, list, tuple)):
        return x[b]
    arr = np.asarray(x)
    return arr[b] if arr.ndim >= 1 else arr


def as_int_list(x):
    """Convert list/tuple/ndarray/torch.Tensor/scalars to a Python list[int]."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.reshape(-1).tolist()]
    if isinstance(x, torch.Tensor):
        return [int(v) for v in x.detach().cpu().flatten().tolist()]
    return [int(x)]


def pred_mode_from_outfmt(out_fmt: str) -> str:
    """Map dataset output_format to canonicalizer pred_mode."""
    mapping = {
        "abs_in_anchor": "abs_in_anchor_cam",
        "delta_from_prev": "deltas_from_prev_cam",
        "delta_from_anchor": "deltas_from_anchor_cam",
    }
    return mapping.get(out_fmt, "abs_in_anchor_cam")
