from __future__ import annotations

import os
import random

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp


def seed_worker(worker_id: int) -> None:
    """Robust worker seeding compatible with spawn. Derives per-worker seeds from torch.initial_seed()."""
    base = torch.initial_seed() % (2**32)
    seed = int((base + worker_id) % (2**32 - 1))
    random.seed(seed)
    np.random.seed(seed)

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)


CTX_SPAWN = mp.get_context("spawn")
