from __future__ import annotations

from typing import Optional

import torch
from torch.distributed.fsdp import (  # type: ignore[attr-defined]
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (  # type: ignore[attr-defined]
    size_based_auto_wrap_policy,
)


def _mixed_precision_for(precision: str) -> Optional[MixedPrecision]:
    p = precision.lower()
    if p == "bf16":
        dtype = torch.bfloat16
    elif p == "fp16":
        dtype = torch.float16
    else:
        return None
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def wrap_fsdp(
    model: torch.nn.Module,
    *,
    precision: str = "bf16",
    wrap_threshold: int = 2_000_000,
    use_sharded_optim: bool = True,
) -> FSDP:
    """
    Optionally wrap a model with FSDP using a simple size-based auto-wrap policy.

    - precision: "bf16" | "fp16" | "float32" determines mixed precision.
    - wrap_threshold: Parameter count threshold for wrapping submodules.
    - use_sharded_optim: If True, use full-shard (parameter + optimizer state).

    Notes:
      - For checkpointing, prefer saving on rank 0 using state_dict() consolidation.
    """
    mp = _mixed_precision_for(precision)
    policy = size_based_auto_wrap_policy(min_num_params=wrap_threshold)
    offload = CPUOffload(offload_params=False)
    shard = (
        ShardingStrategy.FULL_SHARD if use_sharded_optim else ShardingStrategy.SHARD_GRAD_OP
    )

    wrapped = FSDP(
        model,
        auto_wrap_policy=policy,
        cpu_offload=offload,
        sharding_strategy=shard,
        mixed_precision=mp,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        use_orig_params=True,
    )
    return wrapped


