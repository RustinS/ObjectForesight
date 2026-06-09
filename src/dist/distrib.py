from __future__ import annotations

import datetime
import os
import random
import signal
import subprocess
import sys
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

# ============================================================================
# Launch detection utilities (formerly launch_detect.py)
# ============================================================================


def world_size() -> int:
    try:
        return int(os.getenv("WORLD_SIZE", "1"))
    except Exception:
        return 1


def is_dist() -> bool:
    return world_size() > 1


def local_rank() -> int:
    try:
        return int(os.getenv("LOCAL_RANK", "0"))
    except Exception:
        return 0


def global_rank() -> int:
    try:
        return int(os.getenv("RANK", "0"))
    except Exception:
        return 0


def is_torchrun() -> bool:
    # torchrun sets these envs
    return "TORCHELASTIC_RUN_ID" in os.environ or "LOCAL_RANK" in os.environ


def is_slurm() -> bool:
    # srun/sbatch envs
    return "SLURM_JOB_ID" in os.environ or "SLURM_PROCID" in os.environ


# ============================================================================
# Distributed process group management
# ============================================================================


_DIST_INITIALIZED: bool = False
_SIGTERM_REGISTERED: bool = False
_SIGTERM_CALLBACK: Optional[Callable[[], None]] = None


def _env(var: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(var, default)


def has_slurm_env() -> bool:
    return _env("SLURM_PROCID") is not None and _env("SLURM_NTASKS") is not None and _env("SLURM_LOCALID") is not None


def infer_master_addr() -> str:
    """
    Infer master node hostname from SLURM_JOB_NODELIST.
    Falls back to current hostname if scontrol/var is unavailable.
    """
    nodelist = _env("SLURM_JOB_NODELIST")
    if not nodelist:
        return os.uname().nodename
    try:
        out = subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True)
        first = out.strip().splitlines()[0].strip()
        return first or os.uname().nodename
    except Exception:
        return os.uname().nodename


def _set_default_env_if_missing() -> None:
    if _env("MASTER_ADDR") is None:
        os.environ["MASTER_ADDR"] = infer_master_addr()
    if _env("MASTER_PORT") is None:
        # Avoid port collisions when multiple jobs share a node.
        # Prefer a deterministic, job-scoped port derived from SLURM_JOB_ID.
        job_id = _env("SLURM_JOB_ID")
        if job_id and job_id.isdigit():
            os.environ["MASTER_PORT"] = str(20000 + (int(job_id) % 20000))
        else:
            os.environ["MASTER_PORT"] = "29500"
    if _env("RANK") is None and _env("SLURM_PROCID") is not None:
        os.environ["RANK"] = _env("SLURM_PROCID", "0")  # type: ignore[arg-type]
    if _env("WORLD_SIZE") is None and _env("SLURM_NTASKS") is not None:
        os.environ["WORLD_SIZE"] = _env("SLURM_NTASKS", "1")  # type: ignore[arg-type]
    if _env("LOCAL_RANK") is None and _env("SLURM_LOCALID") is not None:
        os.environ["LOCAL_RANK"] = _env("SLURM_LOCALID", "0")  # type: ignore[arg-type]


def init_dist_from_slurm(backend: str = "nccl") -> bool:
    """
    Initialize torch.distributed using Slurm-provided environment variables.

    Returns True if distributed was initialized, False if Slurm env was not found
    and the process should run single-process locally.
    """
    global _DIST_INITIALIZED

    if not has_slurm_env():
        _DIST_INITIALIZED = False
        return False

    _set_default_env_if_missing()

    local_rank_str = os.environ.get("LOCAL_RANK", "0")
    try:
        local_rank = int(local_rank_str)
    except ValueError:
        local_rank = 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://", device_id=local_rank)
    _DIST_INITIALIZED = True
    return True


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_rank0() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def allreduce_health_check() -> bool:
    """
    Simple all-reduce sanity check. Returns True on success.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world = get_world_size()
    tensor = torch.ones(1, device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    ok = int(tensor.item()) == world
    if is_rank0():
        print(f"[health] all-reduce sum={int(tensor.item())} world={world} ok={ok}")
    return ok


def destroy_pg() -> None:
    global _DIST_INITIALIZED
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    finally:
        _DIST_INITIALIZED = False


def init_distributed_env(
    base_seed: int = 42,
    timeout_minutes: int = 30,
) -> Tuple[torch.device, bool, int, int, int]:
    """
    Unified distributed + device initialization for train/eval entrypoints.

    Handles:
    - LOCAL_RANK detection (torchrun, Slurm, single-process)
    - CUDA device setup
    - Process group initialization (NCCL)
    - Deterministic seeding with rank offset

    Args:
        base_seed: Base random seed (rank is added to it)
        timeout_minutes: Timeout for NCCL init

    Returns:
        Tuple of (device, is_dist, rank, world_size, seed)
    """
    _LOCAL_RANK = int(os.getenv("LOCAL_RANK", local_rank()))
    if torch.cuda.is_available():
        torch.cuda.set_device(_LOCAL_RANK)

    if world_size() > 1 and dist.is_available() and not dist.is_initialized():
        device_id = torch.device(f"cuda:{_LOCAL_RANK}") if torch.cuda.is_available() else None
        dist.init_process_group("nccl", init_method="env://", timeout=datetime.timedelta(minutes=timeout_minutes), device_id=device_id)

    _world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _is_dist = (_world_size > 1) or (dist.is_available() and dist.is_initialized())
    device = torch.device(f"cuda:{_local_rank}" if torch.cuda.is_available() else "cpu")

    rank = int(os.environ.get("RANK", "0"))
    seed = base_seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return device, _is_dist, rank, _world_size, seed


def _sigterm_handler(signum: int, _frame: Optional[object]) -> None:  # noqa: ARG001
    if is_rank0():
        try:
            if _SIGTERM_CALLBACK is not None:
                _SIGTERM_CALLBACK()
        except Exception as exc:  # noqa: BLE001 - last-ditch attempt to save state
            print(f"[SIGTERM] on_terminate raised: {exc}", file=sys.stderr)
    sys.exit(0)


def register_sigterm_handler(on_terminate: Optional[Callable[[], None]] = None) -> None:
    """
    Install a SIGTERM handler on rank 0 to run finalization code (e.g., save ckpt).
    Safe to call multiple times.
    """
    global _SIGTERM_REGISTERED
    if _SIGTERM_REGISTERED:
        return
    global _SIGTERM_CALLBACK
    _SIGTERM_CALLBACK = on_terminate
    signal.signal(signal.SIGTERM, _sigterm_handler)
    _SIGTERM_REGISTERED = True
