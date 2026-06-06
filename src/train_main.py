from __future__ import annotations

import gc
import json
import os
import time
import warnings
from contextlib import suppress
from pathlib import Path

import hydra
import torch
import torch.autograd as autograd
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from .dist.distrib import barrier as _barrier
from .dist.distrib import destroy_pg as _destroy_pg
from .dist.distrib import init_distributed_env
from .eval_main import evaluate_loss as evaluate
from .eval_main import evaluate_sampler
from .models.poser_v1 import PoserV1
from .models.poser_v1.io import save as save_ckpt
from .models.poser_v1.utils.checkpoint import pop_lazy_encoder_proj_state, resolve_and_load_state_dict, restore_lazy_encoder_proj
from .utils.config_adapter import apply_config_adapter
from .utils.data_utils import build_train_val_datasets, get_dataset
from .utils.logger import StepAverager, Throughput, checkpoint_line, epoch_table, print_config_summary, print_model_summary, step_line
from .utils.logger import rprint as print
from .utils.normalization import warmup_dnorm_stats
from .utils.parallel import CTX_SPAWN, seed_worker
from .utils.torch_compile import maybe_enable_compile
from .utils.train_utils import (
    _EMA,
    GradFlowAudit,
    IndexDataset,
    _batch_size_from,
    _ensure_optimizer_tracks_params,
    _resolve_resume_checkpoint_path,
    _scheduled_sampling_prob,
    compute_grad_weight_norms_if_enabled,
    debug_collate,
    grad_probe_if_enabled,
    log_training_anomaly_if_any,
    write_model_io_report,
)

# ============================================================================
# PyTorch backend configuration (formerly utils/backend_setup.py)
# ============================================================================


def _dump_batch_debug(
    batch: dict,
    *,
    out_dir: str,
    rank: int,
    world_size: int,
    epoch: int,
    step_idx: int,
    global_step: int,
    exc: Exception,
) -> Path:
    dump_dir = Path(out_dir) / "debug_crashes"
    dump_dir.mkdir(parents=True, exist_ok=True)
    host = os.uname().nodename

    def _tensor_stats(x: torch.Tensor) -> dict:
        # Keep this CPU-only to remain safe when CUDA is in an error state.
        x_cpu = x.detach().cpu()
        stats = {"shape": list(x_cpu.shape), "dtype": str(x_cpu.dtype)}
        if x_cpu.numel() == 0:
            return stats | {"nan": 0, "inf": 0, "abs_max": 0.0, "min": 0.0, "max": 0.0}
        try:
            stats["nan"] = int(torch.isnan(x_cpu).sum().item())
            stats["inf"] = int(torch.isinf(x_cpu).sum().item())
            stats["abs_max"] = float(x_cpu.abs().max().item())
            stats["min"] = float(x_cpu.min().item())
            stats["max"] = float(x_cpu.max().item())
        except Exception:
            # If a dtype doesn't support these ops, just omit.
            pass
        return stats

    ds_idx = batch.get("__dataset_idx")
    if isinstance(ds_idx, torch.Tensor):
        ds_idx_list = [int(x) for x in ds_idx.detach().cpu().flatten().tolist()]
    elif isinstance(ds_idx, (list, tuple)):
        ds_idx_list = [int(x) for x in ds_idx]
    else:
        ds_idx_list = None

    # Useful identifiers (EPIC/HOT3D)
    video_id = batch.get("video_id")
    object_id = batch.get("object_id")
    anchor_frame_idx = batch.get("anchor_frame_idx")

    scene = batch.get("scene_pcd")
    if scene is None:
        scene = batch.get("pointcloud0")
    target_future = batch.get("target_future")

    dump = {
        "host": host,
        "rank": int(rank),
        "world_size": int(world_size),
        "epoch": int(epoch),
        "step_idx": int(step_idx),
        "global_step": int(global_step),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_procid": os.environ.get("SLURM_PROCID"),
        "keys": sorted(list(batch.keys())),
        "dataset_idx": ds_idx_list,
        "video_id": list(video_id) if isinstance(video_id, (list, tuple)) else video_id,
        "object_id": list(object_id) if isinstance(object_id, (list, tuple)) else object_id,
        "anchor_frame_idx": _tensor_stats(anchor_frame_idx) if isinstance(anchor_frame_idx, torch.Tensor) else anchor_frame_idx,
        "scene_pcd": _tensor_stats(scene) if isinstance(scene, torch.Tensor) else str(type(scene)),
        "target_future": _tensor_stats(target_future) if isinstance(target_future, torch.Tensor) else str(type(target_future)),
        "exception": {"type": type(exc).__name__, "message": str(exc)},
    }

    stem = f"rank{rank}_ep{epoch}_step{step_idx}_gs{global_step}"
    dump_path = dump_dir / f"{stem}.json"
    dump_path.write_text(json.dumps(dump, indent=2))
    if ds_idx_list:
        (dump_dir / f"{stem}_indices.txt").write_text("\n".join(str(i) for i in ds_idx_list) + "\n")
    return dump_path


def _configure_torch_backends() -> None:
    """Configure TF32, cudnn, and threading settings for optimal performance."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    # Use only the new TF32 APIs to avoid mixing legacy/new APIs (breaks torch.compile in PyTorch 2.9+)
    # See: https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")  # Enables TF32 for matmul

    # Enable TF32 for cuDNN convolutions
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


_configure_torch_backends()


def _is_rank0() -> bool:
    """Return True if this is rank-0 or non-distributed."""
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _build_lr_scheduler(optimizer, cfg, total_steps: int, warmup_steps: int, min_lr: float):
    """Build cosine LR scheduler with optional warmup."""
    if warmup_steps > 0:
        warmup_start = min(max(float(cfg.train.warmup_start_factor), 1e-8), 1.0)
        warmup_sched = LinearLR(optimizer, start_factor=warmup_start, end_factor=1.0, total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr)
        return SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
    return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)


def _restore_training_state(extra: dict, optimizer, lr_scheduler, gradient_scaler, print_fn=None) -> dict:
    """Restore optimizer, scheduler, and scaler state from checkpoint extras.

    Returns dict with keys: optimizer_ok, scheduler_ok, scaler_ok, and error messages
    """
    status = {"optimizer_ok": False, "scheduler_ok": False, "scaler_ok": False, "optimizer_error": None, "scheduler_error": None, "scaler_error": None}
    log = print_fn or (lambda x: None)

    # Optimizer
    try:
        opt_sd = extra.get("optimizer_state") or extra.get("optimizer")
        if isinstance(opt_sd, dict):
            optimizer.load_state_dict(opt_sd)
            status["optimizer_ok"] = True
    except Exception as e:
        status["optimizer_error"] = str(e)
        log(f"[yellow]Optimizer restore failed[/yellow]: {e}")

    # LR Scheduler
    try:
        sched_sd = extra.get("lr_scheduler_state") or extra.get("lr_scheduler")
        if lr_scheduler is not None and isinstance(sched_sd, dict):
            lr_scheduler.load_state_dict(sched_sd)
            status["scheduler_ok"] = True
        elif lr_scheduler is None:
            status["scheduler_ok"] = True  # No scheduler to restore
    except Exception as e:
        status["scheduler_error"] = str(e)
        log(f"[yellow]LR scheduler restore failed[/yellow]: {e}")

    # Gradient scaler
    try:
        scaler_sd = extra.get("scaler_state") or extra.get("grad_scaler")
        if gradient_scaler.is_enabled() and isinstance(scaler_sd, dict):
            gradient_scaler.load_state_dict(scaler_sd)
            status["scaler_ok"] = True
        elif not gradient_scaler.is_enabled():
            status["scaler_ok"] = True  # Scaler disabled, nothing to restore
    except Exception as e:
        status["scaler_error"] = str(e)
        log(f"[yellow]Gradient scaler restore failed[/yellow]: {e}")

    return status


def _restore_counters(extra: dict, defaults: tuple[int, int, float], print_fn=None) -> tuple[int, int, float]:
    """Restore epoch, step, and best_val from checkpoint extras."""
    epoch, step, best_val = defaults
    log = print_fn or (lambda x: None)

    try:
        epoch = int(extra.get("epoch", epoch))
    except Exception as e:
        log(f"[yellow]Failed to restore epoch[/yellow]: {e}")

    try:
        step = int(extra.get("step", step))
    except Exception as e:
        log(f"[yellow]Failed to restore step[/yellow]: {e}")

    try:
        best_val = float(extra.get("best_val_loss", best_val))
    except Exception as e:
        log(f"[yellow]Failed to restore best_val_loss[/yellow]: {e}")

    return epoch, step, best_val


def _check_scheduler_compatibility(saved_meta: dict | None, lr_scheduler, warmup_steps: int) -> tuple[bool, str]:
    """Check if saved scheduler config is compatible with current config.

    Returns (is_compatible, reason_if_not)
    """
    if saved_meta is None:
        return False, "no scheduler metadata in checkpoint"

    current_type = type(lr_scheduler).__name__ if lr_scheduler else None
    saved_type = saved_meta.get("scheduler_type")

    if current_type != saved_type:
        return False, f"scheduler type mismatch: checkpoint={saved_type}, current={current_type}"

    # Check critical params that affect state dict structure
    saved_warmup = saved_meta.get("warmup_steps")
    if saved_warmup != warmup_steps:
        return False, f"warmup_steps mismatch: checkpoint={saved_warmup}, current={warmup_steps}"

    return True, ""


def _fast_forward_scheduler(lr_scheduler, target_step: int, print_fn) -> None:
    """Manually advance scheduler to target_step when state restoration fails."""
    if lr_scheduler is None or target_step <= 0:
        return
    print_fn(f"[yellow]Fast-forwarding LR scheduler from step 0 to {target_step}[/yellow]")
    for _ in range(target_step):
        lr_scheduler.step()
    try:
        final_lr = lr_scheduler.get_last_lr()[0]
        print_fn(f"[dim]LR after fast-forward: {final_lr:.6e}[/dim]")
    except Exception:
        pass  # Some schedulers may not support get_last_lr


def _log_dnorm_buffers(core_model):
    """Log diffusion normalization buffer values."""
    try:
        dn_fitted = int(core_model._dnorm_fitted.item())
        print(f"[dnorm] fitted={dn_fitted}  means[:3]={core_model._dnorm_means[:3].tolist()}  stds[:3]={core_model._dnorm_scales[:3].tolist()}")
    except Exception:
        print("[yellow][dnorm] unable to read dnorm buffers[/yellow]")


def _check_temporal_schedule(core_model, cfg):
    """Verify temporal schedule T matches config for DiT/FM (both share the sinusoidal embedding range)."""
    try:
        if str(cfg.model.temporal_kind).lower() == "dit":
            t_train = int(core_model._T_train.item())
            t_cfg = int(cfg.model.temporal.T)
            assert t_train == t_cfg, f"T mismatch: trained={t_train}, cfg={t_cfg}"
    except Exception as e:
        print(f"[yellow]T parity check skipped[/yellow]: {e}")


def _sync_z_min(core_model, cfg_blob: dict, cfg):
    """Sync z_min buffer from checkpoint or config."""
    try:
        ckpt_z = None
        if isinstance(cfg_blob, dict):
            ckpt_z = cfg_blob.get("model", {}).get("z_min") or cfg_blob.get("z_min")
        if ckpt_z is not None and hasattr(core_model, "_z_min"):
            with torch.no_grad():
                core_model._z_min.copy_(torch.tensor(float(ckpt_z)))
            print(f"[dim]resume[/dim] synced _z_min from ckpt: {float(ckpt_z)}")
        elif hasattr(core_model, "_z_min"):
            print(f"[dim]resume[/dim] using cfg z_min={float(cfg.model.z_min)}")
    except Exception as e:
        print(f"[yellow]z_min sync skipped[/yellow]: {e}")


@hydra.main(config_path="../conf", config_name="epic", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train PoserV1 model with DiT or AR backend."""
    print("Starting training…")

    with warnings.catch_warnings():
        if os.environ.get("PYTHONWARNINGS", "").lower() in ("", "ignore", "default"):
            warnings.simplefilter("ignore")

    # Register eval resolver BEFORE apply_config_adapter (it accesses temporal_dit.out_dim which uses ${eval:...})
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: int(eval(expr, {"__builtins__": {}}, {})))

    cfg = apply_config_adapter(cfg)

    base_seed = int(cfg.train.seed)
    device, is_dist, rank, world_size, seed = init_distributed_env(base_seed=base_seed, timeout_minutes=30)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Dataset setup
    dataset = get_dataset(cfg)
    tiny_overfit_mode = bool(cfg.train.tiny_overfit)
    tiny_overfit_size = int(cfg.train.tiny_n)
    train_dataset, val_dataset = build_train_val_datasets(dataset, cfg, seed, rank)

    if tiny_overfit_mode:
        subset_indices = list(range(min(tiny_overfit_size, len(train_dataset))))
        training_dataset = Subset(train_dataset, subset_indices)
        val_dataset = training_dataset
        print(f"[bold]tiny_overfit[/bold] • using {len(subset_indices)} windows for training/validation")
    else:
        training_dataset = train_dataset

    # Include dataset indices in each sample for crash triage (only when debug/checks enabled).
    if bool(cfg.train.debug) or bool(cfg.train.checks):
        training_dataset = IndexDataset(training_dataset)
        val_dataset = IndexDataset(val_dataset)

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # Samplers - use drop_last=False in tiny_overfit mode to avoid dropping the only samples
    train_sampler = DistributedSampler(training_dataset, shuffle=True, drop_last=not tiny_overfit_mode) if is_dist else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if is_dist else None

    effective_batch_size = int(cfg.train.batch_size)
    if tiny_overfit_mode:
        effective_batch_size = max(1, min(effective_batch_size, tiny_overfit_size))

    # Data loaders
    loader_kwargs = dict(
        num_workers=int(cfg.train.num_workers),
        prefetch_factor=int(cfg.train.prefetch_factor),
        pin_memory=True,
        persistent_workers=True,
        multiprocessing_context=CTX_SPAWN,
        collate_fn=debug_collate,
        worker_init_fn=seed_worker,
    )
    # Eval loaders: disable persistent_workers to reduce memory fragmentation after DDIM sampling
    eval_loader_kwargs = dict(
        num_workers=int(cfg.train.num_workers),
        prefetch_factor=int(cfg.train.prefetch_factor),
        pin_memory=False,
        persistent_workers=False,
        multiprocessing_context=CTX_SPAWN,
        collate_fn=debug_collate,
        worker_init_fn=seed_worker,
    )

    train_loader = DataLoader(
        training_dataset,
        batch_size=effective_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=not tiny_overfit_mode,
        generator=g,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, sampler=val_sampler, drop_last=False, **eval_loader_kwargs)

    # Small eval subset for DDIM metrics
    tiny_n_eval = int(cfg.train.tiny_n_eval)
    eval_indices = list(range(min(tiny_n_eval, len(val_dataset))))
    eval_dataset = Subset(val_dataset, eval_indices)
    val_eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        sampler=DistributedSampler(eval_dataset, shuffle=False, drop_last=False) if is_dist else None,
        drop_last=False,
        **eval_loader_kwargs,
    )

    # Model setup
    model: PoserV1 = instantiate(cfg.model, _recursive_=False)
    model.to(device)
    if is_dist and not isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
            static_graph=True,
        )
    core_model = model.module if hasattr(model, "module") else model

    # EMA
    use_ema = bool(cfg.train.ema)
    ema = _EMA(model, decay=float(cfg.train.ema_decay)) if use_ema else None

    # Optimizer
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=float(cfg.train.weight_decay), fused=True)
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=float(cfg.train.weight_decay))

    # LR scheduler
    use_cosine = (str(cfg.train.lr_schedule).lower() == "cosine") or bool(cfg.train.cosine_lr)
    min_lr = float(cfg.train.min_lr)
    warmup_steps = int(cfg.train.warmup_steps)
    total_steps = int(cfg.train.epochs) * max(1, len(train_loader)) if use_cosine else None
    lr_scheduler = _build_lr_scheduler(optimizer, cfg, total_steps, warmup_steps, min_lr) if use_cosine else None

    # Training flags
    anomaly_detection = bool(cfg.train.anomaly_detection)
    debug = bool(cfg.train.debug)
    reset_lr_on_resume = bool(cfg.train.reset_lr_on_resume)

    if anomaly_detection:
        autograd.set_detect_anomaly(True)

    # AMP setup - NOTE: For DiT/FM, use model.dit_bf16=true for selective bf16 (encoder stays fp32)
    # Global AMP is disabled for DiT/FM because spconv doesn't support bf16. FM reuses DiTPose
    # plus the same spconv encoder, so it gates the same way.
    gradient_audit = GradFlowAudit(model, optimizer, cfg)
    is_dit_like = str(cfg.model.temporal_kind).lower() == "dit"
    precision = os.environ.get("PRECISION", "bf16").lower()
    amp_dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else None)
    use_amp = bool(cfg.train.amp) and torch.cuda.is_available() and (not is_dit_like) and (amp_dtype is not None)
    gradient_scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype is torch.float16))
    os.makedirs(os.path.join(cfg.train.out_dir, "checkpoints"), exist_ok=True)

    # Resume from checkpoint
    resumed_epoch, resumed_global_step, resumed_best_val = 0, 0, float("inf")
    resume_path = _resolve_resume_checkpoint_path(cfg)

    if isinstance(resume_path, str) and os.path.exists(resume_path):
        try:
            if _is_rank0():
                print(f"[bold green]Resuming[/bold green] from: [yellow]{resume_path}[/yellow]")

            state_dict, meta = resolve_and_load_state_dict(resume_path, map_location="cpu", prefer_ema=False)
            lazy_encoder_state = pop_lazy_encoder_proj_state(state_dict)
            missing, unexpected = core_model.load_state_dict(state_dict, strict=False)
            lazy_restored, lazy_params = restore_lazy_encoder_proj(core_model, lazy_encoder_state)

            if lazy_restored:
                _ensure_optimizer_tracks_params(optimizer, lazy_params)
                if _is_rank0():
                    print(f"[dim]restored lazy encoder params[/dim]: {', '.join(sorted(lazy_encoder_state.keys()))}")

            if _is_rank0():
                print(f"ckpt loaded • params={len(state_dict)} missing={len(missing)} unexpected={len(unexpected)}")

            cfg_blob = (meta.get("cfg") or {}) if isinstance(meta, dict) else {}
            extra = (meta.get("extra") or {}) if isinstance(meta, dict) else {}

            # Detect conditioning mode change (DiT: FiLM <-> AdaLN-Zero)
            # Check multiple possible locations for conditioning in saved config
            _cond_t = (cfg_blob.get("temporal") or {}).get("conditioning")
            _cond_td = (cfg_blob.get("temporal_dit") or {}).get("conditioning")
            _cond_mt = ((cfg_blob.get("model") or {}).get("temporal") or {}).get("conditioning")
            _cond_mtd = ((cfg_blob.get("model") or {}).get("temporal_dit") or {}).get("conditioning")
            ckpt_conditioning = _cond_t or _cond_td or _cond_mt or _cond_mtd
            curr_conditioning = getattr(cfg.temporal, "conditioning", "film")
            # If checkpoint has no config (legacy), assume conditioning matches current config
            if not cfg_blob:
                if _is_rank0():
                    print("[dim]ckpt has no saved config - assuming conditioning mode matches current config[/dim]")
                conditioning_changed = False
            elif not ckpt_conditioning:
                if _is_rank0():
                    print(f"[dim]ckpt config has no conditioning field - assuming it matches current ({curr_conditioning})[/dim]")
                conditioning_changed = False
            else:
                conditioning_changed = ckpt_conditioning != curr_conditioning
                if conditioning_changed and _is_rank0():
                    print(
                        f"[yellow]WARNING: conditioning mode changed ({ckpt_conditioning} → {curr_conditioning}). Optimizer/scheduler/EMA state will NOT be restored.[/yellow]"
                    )

            if _is_rank0():
                _log_dnorm_buffers(core_model)
                _check_temporal_schedule(core_model, cfg)
                _sync_z_min(core_model, cfg_blob, cfg)

            # Restore counters first (needed for scheduler fast-forward fallback)
            log_fn = print if _is_rank0() else (lambda x: None)
            resumed_epoch, resumed_global_step, resumed_best_val = _restore_counters(extra, (0, 0, float("inf")), print_fn=log_fn)
            if _is_rank0():
                print(f"resumed_epoch: {resumed_epoch}, resumed_global_step: {resumed_global_step}")

            if reset_lr_on_resume or conditioning_changed:
                if _is_rank0():
                    reason = "reset_lr_on_resume" if reset_lr_on_resume else "conditioning mode changed"
                    print(f"[bold yellow]{reason}[/bold yellow] • skipping optimizer/scheduler/scaler restore")
            else:
                # Check scheduler compatibility before restoration
                sched_meta = extra.get("lr_scheduler_meta")
                sched_compatible, sched_reason = _check_scheduler_compatibility(sched_meta, lr_scheduler, warmup_steps)

                if not sched_compatible and _is_rank0():
                    print(f"[yellow]Scheduler config mismatch[/yellow]: {sched_reason}")

                restore_status = _restore_training_state(extra, optimizer, lr_scheduler, gradient_scaler, print_fn=log_fn)

                # If scheduler restoration failed, use fast-forward fallback
                if not restore_status["scheduler_ok"] and lr_scheduler is not None:
                    if _is_rank0():
                        err_msg = restore_status.get("scheduler_error") or "unknown"
                        print(f"[yellow]Scheduler state not restored[/yellow]: {err_msg}")
                    _fast_forward_scheduler(lr_scheduler, resumed_global_step, log_fn)

            # Restore EMA (skip if conditioning mode changed)
            if ema is not None and isinstance(meta, dict) and not conditioning_changed:
                ema_sd = meta.get("ema_state_dict")
                if isinstance(ema_sd, dict) and ema_sd:
                    ema.load_state_dict(ema_sd)
                    if _is_rank0():
                        print("[dim]resume[/dim] restored EMA state")
            elif conditioning_changed and ema is not None and _is_rank0():
                print("[dim]skipping EMA restore due to conditioning mode change[/dim]")

        except Exception as e:
            if _is_rank0():
                print(f"[yellow]Resume failed[/yellow]: {e}. Proceeding without resume.")
        _barrier()

    # DiT/FM depth-norm warmup: fit stats from multiple batches before training
    # IMPORTANT: Must happen AFTER checkpoint loading so _dnorm_fitted is respected.
    # FM uses the same depth-norm path as DiT, so gate the warmup the same way —
    # otherwise per-rank single-batch fitting produces divergent _dnorm_* across the 16 ranks.
    n_warmup = int(getattr(cfg.model, "norm_warmup_batches", 0))
    if is_dit_like and n_warmup > 0 and not core_model._dnorm_fitted.item():
        if _is_rank0():
            print(f"[bold cyan][dnorm warmup][/bold cyan] Fitting from {n_warmup} batches...")

        # Create a separate loader with deterministic shuffling for warmup
        # Use num_workers=0 to avoid fork+CUDA issues (CUDA already initialized at this point)
        warmup_g = torch.Generator().manual_seed(base_seed)
        warmup_loader = DataLoader(
            training_dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=0,  # Single-process to avoid fork+CUDA crash
            collate_fn=debug_collate,
            generator=warmup_g,
        )
        warmup_dnorm_stats(
            model,
            warmup_loader,
            n_batches=n_warmup,
            device=device,
            rank=rank,
            world_size=world_size,
            verbose=_is_rank0(),
        )
        del warmup_loader
        if is_dist:
            _barrier()
    elif is_dit_like and resume_path and _is_rank0():
        print("[dnorm warmup] Skipped - resuming from checkpoint with fitted stats")

    # Reset LR on resume (only when actually resuming from a checkpoint)
    if reset_lr_on_resume and resume_path:
        for group in optimizer.param_groups:
            group["lr"] = float(cfg.train.lr)
        if use_cosine:
            remaining_epochs = max(int(cfg.train.epochs) - int(resumed_epoch), 1)
            t_max = remaining_epochs * max(1, len(train_loader))
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)
        if _is_rank0():
            print("[bold yellow]reset_lr_on_resume[/bold yellow]=true • LR scheduler restarted")

    # Optional torch.compile (compile submodules used by compute_loss paths)
    maybe_enable_compile(model, cfg, print_fn=(print if _is_rank0() else None))

    # Logging setup
    if cfg.log.show_config_summary:
        print_config_summary(cfg)
    if cfg.log.show_model_summary:
        total_params_m = sum(p.numel() for p in model.parameters()) / 1e6
        encoder_name = cfg.model.encoder._target_.split(".")[-1] if "_target_" in cfg.model.encoder else "encoder"
        _kind = str(cfg.model.temporal_kind).lower()
        if _kind == "ar_transformer":
            temporal_name = "AutoregTransformerPose"
        else:
            temporal_name = "DiT"
        print_model_summary(encoder_name=encoder_name, temporal_name=temporal_name, params_m=total_params_m, horizon=cfg.data.H, context_len=cfg.data.context_len)

    write_model_io_report(dataset, train_loader, cfg, print)

    # Training state
    loss_ema = StepAverager(beta=1 - 1.0 / max(int(cfg.log.ema_window), 1))
    throughput_tracker = Throughput(window_steps=max(int(cfg.log.throughput_window), 1))
    global_step = int(resumed_global_step)
    max_steps = int(getattr(cfg.train, "max_steps", 0))
    grad_explode_thr = float(cfg.train.grad_explode_thr)
    track_diag = bool(cfg.train.train_diagnostics)
    best_val_loss = float(resumed_best_val)
    best_by_infer = float("inf")

    # Main training loop
    stop_training = False
    for epoch in range(int(resumed_epoch), int(cfg.train.epochs)):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        epoch_start = time.time()
        last_step_time = time.time()
        eval_every = int(cfg.train.eval_every_steps)
        is_ar = str(cfg.model.temporal_kind).lower() == "ar_transformer"

        for step_idx, batch in enumerate(train_loader, start=1):
            if max_steps > 0 and global_step >= max_steps:
                stop_training = True
                break
            batch_size = _batch_size_from(batch)
            gradient_audit.begin_step()
            step_grad_norm: float | None = None

            if is_ar and getattr(core_model.temporal, "supports_scheduled_sampling", False):
                batch["scheduled_sampling_p"] = _scheduled_sampling_prob(cfg, global_step)

            # Wrap forward-backward-step in try-except to handle CUDA scatter errors
            # that occur sporadically on large multi-node runs (8+ nodes)
            try:
                # Forward through DDP wrapper for proper gradient synchronization
                with torch.amp.autocast(device_type="cuda", dtype=(amp_dtype or torch.float32), enabled=use_amp):
                    loss_out = model(batch)  # Goes through DDP -> forward() -> compute_loss()
                    loss = loss_out["loss"]

                if grad_probe_if_enabled(model, batch, cfg, gradient_audit, optimizer, global_step, batch_size):
                    return

                # Backward
                if anomaly_detection:
                    with autograd.detect_anomaly():
                        gradient_scaler.scale(loss).backward()
                else:
                    gradient_scaler.scale(loss).backward()
                gradient_scaler.unscale_(optimizer)

                # Skip step if gradients have NaN/Inf (DDP-safe, single GPU sync)
                bad_grad = torch.zeros(1, device=device)
                for p in model.parameters():
                    if p.grad is not None:
                        bad_grad += (~torch.isfinite(p.grad)).any()  # accumulate on GPU, no sync
                if world_size > 1:
                    torch.distributed.all_reduce(bad_grad, op=torch.distributed.ReduceOp.MAX)
                if bad_grad.item() > 0:  # single sync point
                    if _is_rank0():
                        print(f"[WARN] Bad grad at ep {epoch + 1} step {step_idx} gs {global_step}, skipping")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                grad_norm = 0.0
                if anomaly_detection or track_diag:
                    grad_norm, _ = compute_grad_weight_norms_if_enabled(model, track_diag)

                # Gradient clipping
                clip_norm = float(cfg.train.grad_clip_norm)
                if clip_norm > 0.0:
                    pre_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                    if _is_rank0():
                        step_grad_norm = float(pre_clip)
                    if gradient_audit.is_active():
                        gradient_audit.record_preclip(float(pre_clip))
                        gradient_audit.record_postclip(gradient_audit._sum_param_grad_norms())
                elif gradient_audit.is_active():
                    norm = gradient_audit._sum_param_grad_norms()
                    gradient_audit.record_preclip(norm)
                    gradient_audit.record_postclip(norm)
                elif (anomaly_detection or track_diag) and _is_rank0():
                    step_grad_norm = float(grad_norm) if grad_norm > 0.0 else None

                if debug and gradient_audit.is_active():
                    gradient_audit.build_and_emit_report(global_step + 1, optimizer.param_groups[0].get("lr", float(cfg.train.lr)))

                # Optimizer step
                gradient_scaler.step(optimizer)
                gradient_scaler.update()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if ema is not None:
                    ema.update(model)

            except RuntimeError as e:
                err_str = str(e).lower()
                # Check if it's a CUDA scatter/index error, similar CUDA error, or gradient NaN detection
                if "cuda" in err_str or "scatter" in err_str or "index" in err_str or "assert" in err_str or "gradnandetector" in err_str:
                    print(f"[WARN] CUDA error at epoch {epoch + 1}, step {step_idx}, global_step {global_step}: {e}")
                    try:
                        dump_path = _dump_batch_debug(
                            batch,
                            out_dir=str(cfg.train.out_dir),
                            rank=int(rank),
                            world_size=int(world_size),
                            epoch=int(epoch + 1),
                            step_idx=int(step_idx),
                            global_step=int(global_step),
                            exc=e,
                        )
                        print(f"[WARN] Wrote crash batch dump: {dump_path}")
                    except Exception as dump_e:
                        print(f"[WARN] Failed to write crash batch dump: {dump_e}")

                    if "device-side assert" in err_str and "gradnandetector" not in err_str:
                        print("[WARN] CUDA device-side assert is not recoverable; aborting.")
                        raise
                    print("[WARN] Attempting recovery: syncing CUDA, clearing gradients, skipping batch...")
                    # Synchronize to clear CUDA error state
                    try:
                        torch.cuda.synchronize()
                    except Exception as sync_e:
                        print(f"[WARN] CUDA synchronize failed during recovery (likely not recoverable): {sync_e}")
                        raise
                    # Clear any pending gradients
                    optimizer.zero_grad(set_to_none=True)
                    # Clear CUDA cache to free any corrupted memory
                    torch.cuda.empty_cache()
                    # Clear spconv hash maps if available
                    try:
                        import spconv.pytorch as spconv

                        if hasattr(spconv, "core") and hasattr(spconv.core, "clear_global_allocator"):
                            spconv.core.clear_global_allocator()
                    except (ImportError, AttributeError):
                        pass
                    print("[WARN] Skipped batch, continuing training...")
                    # Skip to next batch - don't update global_step or loss metrics
                    continue
                else:
                    # Re-raise non-CUDA errors
                    raise

            # Update metrics
            step_dur = time.time() - last_step_time
            last_step_time = time.time()
            global_step += 1

            loss_val = float(loss.detach().cpu())
            ema_loss_val = loss_ema.update(loss_val)
            cur_lr = optimizer.param_groups[0]["lr"]
            tput = throughput_tracker.update(batch_size, step_dur)

            # Logging
            if _is_rank0() and ((step_idx % int(cfg.log.print_every) == 0) or step_idx == 1):

                def to_float(x):
                    return float(x.detach().cpu()) if isinstance(x, torch.Tensor) else float(x)

                rot_deg = to_float(loss_out.get("rot_deg", 0.0))
                trans_l2 = to_float(loss_out.get("trans_l2", 0.0))
                loss_main = to_float(loss_out.get("loss_main", 0.0))
                loss_aux = to_float(loss_out.get("loss_aux", 0.0))

                # SE3 metrics group (with units)
                se3_grp = {"rot": (rot_deg, "°"), "t": (trans_l2, "m")}

                # Loss components group
                loss_grp = {"main": loss_main, "aux": loss_aux}
                loss_anchor = to_float(loss_out.get("loss_anchor", 0.0))
                if loss_anchor > 1e-6:  # Only show if non-zero
                    loss_grp["anchor"] = loss_anchor
                if is_ar:
                    loss_grp["ss"] = float(batch.get("scheduled_sampling_p", 0.0))

                # Auxiliary losses (only show non-zero)
                aux_grp = {}
                for key in ("loss_vel", "loss_acc", "loss_disp"):
                    val = to_float(loss_out.get(key, 0.0))
                    if val > 1e-6:
                        aux_grp[key.replace("loss_", "")] = val

                groups = [se3_grp, loss_grp]
                if aux_grp:
                    groups.append(aux_grp)

                step_line(
                    epoch=epoch + 1,
                    step=step_idx,
                    total_steps=len(train_loader),
                    loss=loss_val,
                    loss_ema=ema_loss_val,
                    lr=cur_lr,
                    grad_norm=step_grad_norm,
                    tput=tput,
                    groups=groups,
                    prec=int(cfg.log.precision),
                )

            if anomaly_detection:
                log_training_anomaly_if_any(loss, grad_norm, grad_explode_thr, track_diag, print)

            # Periodic memory cleanup (every 100 steps) to prevent spconv hash map accumulation
            if global_step % 100 == 0:
                # Clear grad audit tensors if they've accumulated
                if hasattr(core_model, "_ga_tensors"):
                    core_model._ga_tensors.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Clear spconv hash maps if available (prevents memory accumulation)
                try:
                    import spconv.pytorch as spconv

                    if hasattr(spconv, "core") and hasattr(spconv.core, "clear_global_allocator"):
                        spconv.core.clear_global_allocator()
                except (ImportError, AttributeError):
                    pass

            # Periodic checkpoint
            if (global_step % int(cfg.train.save_every) == 0) and _is_rank0():
                path = os.path.join(cfg.train.out_dir, "checkpoints", f"step_{global_step}.pt")
                save_ckpt(
                    model,
                    path,
                    extra={
                        "epoch": epoch + 1,
                        "step": global_step,
                        "best_val_loss": best_val_loss,
                        "optimizer_state": optimizer.state_dict(),
                        "lr_scheduler_state": lr_scheduler.state_dict() if lr_scheduler else None,
                        "scaler_state": gradient_scaler.state_dict() if gradient_scaler.is_enabled() else None,
                        "lr_scheduler_meta": {
                            "scheduler_type": type(lr_scheduler).__name__ if lr_scheduler else None,
                            "warmup_steps": warmup_steps,
                            "total_steps": total_steps,
                            "min_lr": min_lr,
                            "lr": float(cfg.train.lr),
                        }
                        if lr_scheduler
                        else None,
                    },
                    ema_state_dict=ema.state_dict() if ema else None,
                )
                checkpoint_line(path, note="periodic")

            # Step-based validation
            if eval_every > 0 and (global_step % eval_every == 0):
                if ema is not None:
                    ema.store(model)
                    ema.copy_to(model)

                val_metrics = evaluate(model, val_loader, cfg, device)
                # Support both new and legacy config key names
                use_full_val = bool(getattr(cfg.train, "sampler_eval_full_val", cfg.train.ddim_eval_full_val))
                sampler_loader = val_loader if use_full_val else val_eval_loader
                val_sampler = evaluate_sampler(model, sampler_loader, cfg, device)

                agg_sampler = None
                per_repeat = None
                if isinstance(val_sampler, tuple) and len(val_sampler) == 2:
                    agg_sampler, per_repeat = val_sampler
                elif isinstance(val_sampler, dict):
                    agg_sampler = val_sampler

                if ema is not None:
                    ema.restore(model)

                # Use model-appropriate naming for metrics
                sampler_type = agg_sampler.get("sampler_type", "DDIM" if not is_ar else "AR rollout") if isinstance(agg_sampler, dict) else ("AR rollout" if is_ar else "DDIM")

                if _is_rank0() and cfg.log.epoch_table:
                    epoch_dur = time.time() - epoch_start
                    if isinstance(agg_sampler, dict) and isinstance(per_repeat, (list, tuple)):
                        for i, m in enumerate(per_repeat):
                            title = f"Validation ({sampler_type}) Repeat {i + 1}/{len(per_repeat)} @ step {global_step}"
                            epoch_table(title=title, metrics=m, prec=int(cfg.log.precision))
                        agg_with_loss = {"val_loss": float(val_metrics.get("val_loss", float("nan"))), **agg_sampler}
                        title = f"Validation ({sampler_type}) Aggregate @ step {global_step} (epoch {epoch + 1}, {epoch_dur:.1f}s)"
                        epoch_table(title=title, metrics=agg_with_loss, prec=int(cfg.log.precision))
                    else:
                        agg = agg_sampler or {}
                        merged = {
                            **val_metrics,
                            "val_rot_deg_sampler": float(agg.get("rot_deg_mean", float("nan"))),
                            "val_trans_l2_sampler": float(agg.get("trans_l2_mean", float("nan"))),
                            "val_canon_rot_deg_sampler": float(agg.get("canon_rot_deg_mean", float("nan"))),
                            "val_canon_trans_l2_sampler": float(agg.get("canon_trans_l2_mean", float("nan"))),
                            "val_ade_t_sampler": float(agg.get("ade_t", float("nan"))),
                            "val_fde_t_sampler": float(agg.get("fde_t", float("nan"))),
                            "val_are_r_sampler": float(agg.get("are_r", float("nan"))),
                            "val_fre_r_sampler": float(agg.get("fre_r", float("nan"))),
                            "val_trans_drift_per_step_sampler": float(agg.get("trans_drift_per_step", float("nan"))),
                            "val_rot_drift_per_step_sampler": float(agg.get("rot_drift_per_step", float("nan"))),
                            "val_trans_error_slope_sampler": float(agg.get("trans_error_slope", float("nan"))),
                            "val_rot_error_slope_sampler": float(agg.get("rot_error_slope", float("nan"))),
                        }
                        epoch_table(title=f"Validation @ step {global_step} (epoch {epoch + 1}, {epoch_dur:.1f}s)", metrics=merged, prec=int(cfg.log.precision))

                # Best checkpoint by inference metric
                cur_val_loss = float(val_metrics.get("val_loss", float("inf")))
                agg_for_best = agg_sampler or {}

                # Prefer canonicalized metrics if available, otherwise fall back to raw metrics
                if "canon_rot_deg_mean" in agg_for_best and "canon_trans_l2_mean" in agg_for_best:
                    cur_rot = float(agg_for_best["canon_rot_deg_mean"])
                    cur_trans_m = float(agg_for_best["canon_trans_l2_mean"]) * 100.0
                else:
                    # Fall back to non-canonicalized metrics (always available)
                    cur_rot = float(agg_for_best.get("rot_deg_mean", float("inf")))
                    cur_trans_m = float(agg_for_best.get("trans_l2_mean", float("inf"))) * 100.0

                if cur_val_loss < best_val_loss:
                    best_val_loss = cur_val_loss

                infer_score = cur_rot + cur_trans_m
                used_canon = "canon_rot_deg_mean" in agg_for_best and "canon_trans_l2_mean" in agg_for_best
                metric_type = "canon" if used_canon else "raw"
                if _is_rank0() and infer_score < best_by_infer:
                    best_by_infer = infer_score
                    best_path = os.path.join(cfg.train.out_dir, "checkpoints", "best.pt")
                    save_ckpt(
                        model,
                        best_path,
                        extra={
                            "epoch": epoch + 1,
                            "step": global_step,
                            "val_loss": cur_val_loss,
                            "best_val_loss": best_val_loss,
                            "best_infer_score": best_by_infer,
                            "best_infer_metric_type": metric_type,
                            "optimizer_state": optimizer.state_dict(),
                            "lr_scheduler_state": lr_scheduler.state_dict() if lr_scheduler else None,
                            "scaler_state": gradient_scaler.state_dict() if gradient_scaler.is_enabled() else None,
                            "lr_scheduler_meta": {
                                "scheduler_type": type(lr_scheduler).__name__ if lr_scheduler else None,
                                "warmup_steps": warmup_steps,
                                "total_steps": total_steps,
                                "min_lr": min_lr,
                                "lr": float(cfg.train.lr),
                            }
                            if lr_scheduler
                            else None,
                        },
                        ema_state_dict=ema.state_dict() if ema else None,
                    )
                    print(f"[bold green]Saving best model[/bold green] by {metric_type} sampler metric ({infer_score:.6f}) → {best_path}")
                    checkpoint_line(best_path, note="best")

                # Clear CUDA cache to reduce memory fragmentation after DDIM sampling
                del val_metrics, val_sampler, agg_sampler, per_repeat
                gc.collect()
                torch.cuda.empty_cache()

                # Clear spconv hash maps if available (prevents memory accumulation)
                try:
                    import spconv.pytorch as spconv

                    if hasattr(spconv, "core") and hasattr(spconv.core, "clear_global_allocator"):
                        spconv.core.clear_global_allocator()
                except (ImportError, AttributeError):
                    pass

                model.train()

        if stop_training:
            if _is_rank0():
                print(f"[bold yellow]Stopping[/bold yellow] • reached train.max_steps={max_steps}")
            break

    # Final checkpoint
    if _is_rank0():
        final_path = os.path.join(cfg.train.out_dir, "checkpoints", "last.pt")
        save_ckpt(
            model,
            final_path,
            extra={
                "epoch": int(cfg.train.epochs),
                "step": global_step,
                "best_val_loss": best_val_loss,
                "optimizer_state": optimizer.state_dict(),
                "lr_scheduler_state": lr_scheduler.state_dict() if lr_scheduler else None,
                "scaler_state": gradient_scaler.state_dict() if gradient_scaler.is_enabled() else None,
                "lr_scheduler_meta": {
                    "scheduler_type": type(lr_scheduler).__name__ if lr_scheduler else None,
                    "warmup_steps": warmup_steps,
                    "total_steps": total_steps,
                    "min_lr": min_lr,
                    "lr": float(cfg.train.lr),
                }
                if lr_scheduler
                else None,
            },
            ema_state_dict=ema.state_dict() if ema else None,
        )
        checkpoint_line(final_path, note="final")

    _barrier()

    # Clean up DataLoaders with persistent_workers to avoid "terminate called without an active exception"
    for loader in [train_loader, val_loader, val_eval_loader]:
        # Shutdown internal iterator workers if active
        if hasattr(loader, "_iterator") and loader._iterator is not None:
            try:
                loader._iterator._shutdown_workers()
            except Exception:
                pass
    del train_loader, val_loader, val_eval_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with suppress(Exception):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            _destroy_pg()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
