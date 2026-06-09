from __future__ import annotations

import os

import numpy as np


def _stem_to_fid(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem) if stem.isdigit() else 10**9


def _poses_to_9d(T_seq: np.ndarray) -> np.ndarray:
    t = T_seq[:, :3, 3]
    R = T_seq[:, :3, :3]
    r6 = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1).astype(np.float32)
    return np.concatenate([t, r6], axis=1).astype(np.float32)
