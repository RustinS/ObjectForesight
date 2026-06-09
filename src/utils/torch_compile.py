from __future__ import annotations

from typing import Callable

import torch


def _ensure_cuda_initialized() -> None:
    """Ensure CUDA context and backend settings are initialized before torch.compile.

    This prevents inductor from failing when accessing torch.backends.cuda.matmul.allow_tf32
    during graph compilation (PyTorch 2.9+ bug workaround).
    """
    if not torch.cuda.is_available():
        return

    # Force CUDA context initialization
    if not torch.cuda.is_initialized():
        torch.cuda.init()

    # Ensure backend settings are accessible (triggers lazy initialization)
    try:
        _ = torch.backends.cuda.matmul.allow_tf32
        _ = torch.backends.cudnn.allow_tf32
    except Exception:
        pass


def maybe_enable_compile(model: torch.nn.Module, cfg, *, print_fn: Callable[[str], None] | None = None) -> torch.nn.Module:
    """
    Enable torch.compile on safe submodules based on cfg.model.compile.

    Notes:
    - This project often calls custom methods like `compute_loss(...)` (not `forward(...)`), so compiling the *entire*
      PoserV1 module is usually not effective unless training is refactored.
    - Instead, we compile the heavy submodules that are invoked from those paths:
        - DiT: compile the temporal module itself (DiTPose.forward).
        - AR: compile the internal TransformerEncoder stack used by forward_train/forward_step/rollout.
    """
    enabled = bool(getattr(getattr(cfg, "model", None), "compile", False))
    if not enabled:
        return model

    log = print_fn or (lambda _msg: None)

    if not hasattr(torch, "compile"):
        log("[yellow]torch.compile not available[/yellow] • ignoring cfg.model.compile=true")
        return model

    # Ensure CUDA is fully initialized before compile (workaround for PyTorch 2.9+ inductor bug)
    _ensure_cuda_initialized()

    core = model.module if hasattr(model, "module") else model
    kind = str(getattr(getattr(cfg, "model", None), "temporal_kind", getattr(getattr(cfg, "temporal", None), "kind", "dit"))).lower()

    # Default to max-autotune for best performance; can override via cfg.model.compile_mode
    mode = str(getattr(getattr(cfg, "model", None), "compile_mode", "max-autotune"))
    compiled: list[str] = []

    if kind == "dit":
        try:
            core.temporal = torch.compile(core.temporal, mode=mode)
            core.dit = core.temporal
            compiled.append(f"temporal({kind})")
        except Exception as e:
            log(f"[yellow]torch.compile failed[/yellow] for temporal({kind}): {e}")

    elif kind == "ar_transformer":
        try:
            transformer = getattr(getattr(core, "temporal", None), "transformer", None)
            if isinstance(transformer, torch.nn.Module):
                core.temporal.transformer = torch.compile(transformer, mode=mode)
                compiled.append("temporal.transformer")
        except Exception as e:
            log(f"[yellow]torch.compile failed[/yellow] for temporal.transformer: {e}")

    else:
        log(f"[yellow]torch.compile[/yellow] skipped • unknown temporal_kind={kind!r}")

    if compiled:
        log(f"[dim]torch.compile[/dim] enabled ({mode}) • {', '.join(compiled)}")
    return model
