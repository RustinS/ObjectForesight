from __future__ import annotations

import json
import math
import os

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from .data.cache import (
    load_hot3d_shared_scenes_split,
    load_hot3d_split,
    load_split,
    save_hot3d_shared_scenes_split,
    save_hot3d_split,
)
from .data.datasets import HOT3DClipsDataset, SceneSequenceDataset, SceneSequenceDatasetSynth
from .dist.distrib import barrier as _barrier
from .dist.distrib import destroy_pg as _destroy_pg
from .dist.distrib import get_rank, init_distributed_env
from .geom.canonicalize import canonicalize_preds_to_anchor
from .geom.se3_ops import geodesic_distance_deg as _geod_deg
from .geom.se3_ops import rot6d_to_matrix as _r6_to_R
from .models.poser_v1 import PoserV1
from .models.poser_v1.utils.checkpoint import pop_lazy_encoder_proj_state, resolve_and_load_state_dict, restore_lazy_encoder_proj
from .models.poser_v1.utils.inferviz_common import apply_sanity_checks
from .temporal.sampling import ar_rollout_sample, ddim_sample_tokens
from .utils.batch_utils import as_int_list, pred_mode_from_outfmt, slice_batch_item, to_numpy
from .utils.config_adapter import apply_config_adapter
from .utils.data_utils import _build_hot3d_per_clip_split, _build_hot3d_shared_scenes_split, get_dataset
from .utils.logger import epoch_table, print_config_summary, print_model_summary
from .utils.logger import rprint as print
from .utils.parallel import CTX_SPAWN, seed_worker
from .utils.progress import tqdm_auto as tqdm
from .utils.torch_compile import maybe_enable_compile
from .utils.train_utils import _batch_size_from, debug_collate

# ============================================================================
# PyTorch backend configuration (formerly utils/backend_setup.py)
# ============================================================================


def _configure_torch_backends() -> None:
    """Configure TF32, cudnn, and threading settings for optimal performance."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    elif hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        torch.backends.cudnn.conv.fp32_precision = "tf32"

    torch.backends.cudnn.benchmark = True


_configure_torch_backends()


def _is_rank0() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _safe_div(a: float, b: float) -> float:
    return a / max(1.0, b)


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if len(vals) <= 1:
        return (vals[0] if vals else 0.0, 0.0)
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(max(0.0, v))


@torch.no_grad()
def ar_rollout_predict(core_model: PoserV1, batch: dict, device: torch.device) -> torch.Tensor:
    cond = core_model.condition_from_batch(batch)
    tb = {**batch, **cond}

    _tf = tb.get("target_future")
    if isinstance(_tf, torch.Tensor) and _tf.dim() >= 2:
        H = int(_tf.shape[1])
    elif hasattr(core_model, "cfg") and hasattr(core_model.cfg, "H"):
        H = int(core_model.cfg.H)
    else:
        raise RuntimeError("ar_rollout_predict requires core_model.cfg.H to be set.")

    ctx = tb.get("context_init_9d")
    if ctx is None or (isinstance(ctx, torch.Tensor) and ctx.shape[1] == 0):
        raise RuntimeError("Rollout requires non-empty context.")
    if not isinstance(ctx, torch.Tensor):
        ctx = torch.as_tensor(ctx, device=device, dtype=torch.float32)
    else:
        ctx = ctx.to(device=device, dtype=torch.float32)
    if ctx.dim() == 2:
        ctx = ctx.unsqueeze(0)

    T_cam_anchor_obj = cond.get("T_cam_anchor_obj")
    cond_embed = core_model.encode(tb["scene_pcd"], tb["context_vec"], T_cam_anchor_obj=T_cam_anchor_obj)
    try:
        ctx_n = core_model._perdim_normalize(ctx)
    except Exception:
        ctx_n = ctx
    y_pred_n = core_model.temporal.rollout_infer(H, cond_embed, ctx_tokens=ctx_n)
    try:
        return core_model._perdim_unnormalize(y_pred_n)
    except Exception:
        return y_pred_n


def _tokenwise_trans_rot_errors(pred_9d: torch.Tensor, gt_9d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if gt_9d.dim() == 2:
        gt_9d = gt_9d.unsqueeze(0)
    gt_9d = gt_9d.view_as(pred_9d)
    trans_err = torch.linalg.norm(pred_9d[..., :3] - gt_9d[..., :3], dim=-1)
    rot_err = _geod_deg(_r6_to_R(pred_9d[..., 3:9].reshape(-1, 6)), _r6_to_R(gt_9d[..., 3:9].reshape(-1, 6))).view_as(trans_err)
    return trans_err, rot_err


def _compute_drift_and_slope(trans_err: torch.Tensor, rot_err: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    T = trans_err.shape[-1]
    denom = max(T - 1, 1)
    drift_t = (trans_err[..., -1] - trans_err[..., 0]) / denom
    drift_r = (rot_err[..., -1] - rot_err[..., 0]) / denom

    t_idx = torch.arange(T, device=trans_err.device, dtype=trans_err.dtype).unsqueeze(0).expand(trans_err.shape[0], T)
    x = t_idx - t_idx.mean(dim=1, keepdim=True)
    denom_x = (x * x).sum(dim=1) + 1e-12
    slope_t = (x * (trans_err - trans_err.mean(dim=1, keepdim=True))).sum(dim=1) / denom_x
    slope_r = (x * (rot_err - rot_err.mean(dim=1, keepdim=True))).sum(dim=1) / denom_x
    return drift_t, drift_r, slope_t, slope_r


def _trim_tensor(x: torch.Tensor | np.ndarray, P_keep: int):
    if not isinstance(x, (torch.Tensor, np.ndarray)) or x.ndim < 2:
        return x
    P_total = x.shape[-2]
    if P_keep <= 0 or P_keep >= P_total:
        return x
    return x[..., (P_total - P_keep) :, :]


def _maybe_trim_context_for_eval(batch: dict, cfg: DictConfig) -> None:
    P_train = cfg.eval.model_context_len
    if P_train is None or int(P_train) <= 0:
        return
    ctx_init, ctx_bbox = batch.get("context_init_9d"), batch.get("context_bbox_norm")
    if ctx_init is None or ctx_bbox is None:
        return
    P_keep = int(P_train) + 1
    batch["context_init_9d"] = _trim_tensor(ctx_init, P_keep)
    batch["context_bbox_norm"] = _trim_tensor(ctx_bbox, P_keep)


def _build_sample_meta(batch: dict, b: int) -> dict:
    return {
        "K": to_numpy(slice_batch_item(batch, "K", b)),
        "frame_ids": as_int_list(slice_batch_item(batch, "frame_ids", b)),
        "anchor_frame_idx": int(slice_batch_item(batch, "anchor_frame_idx", b) or 0),
        "anchor_local_idx": int(slice_batch_item(batch, "anchor_local_idx", b) or 0),
        "T_c_w": to_numpy(slice_batch_item(batch, "T_c_w", b)),
        "T_c_o": to_numpy(slice_batch_item(batch, "T_c_o", b)),
        "T_cam_anchor_obj": to_numpy(slice_batch_item(batch, "T_cam_anchor_obj", b)),
        "t_mean": [0.0, 0.0, 0.0],
        "t_std": [1.0, 1.0, 1.0],
    }


def _get_output_format(loader: DataLoader) -> str:
    ds_ref = loader.dataset.dataset if hasattr(loader.dataset, "dataset") else loader.dataset
    return str(getattr(ds_ref, "output_format", "abs_in_anchor"))


@torch.no_grad()
def evaluate_loss(model: PoserV1, val_loader: DataLoader, cfg: DictConfig, device: torch.device) -> dict:
    model.eval()
    core_model = model.module if hasattr(model, "module") else model
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    if rank == 0:
        print(f"[bold cyan]Starting evaluation[/bold cyan] • batches={len(val_loader)}")

    is_ar = str(cfg.model.temporal_kind).lower() == "ar_transformer"
    do_roll = bool(cfg.train.rollout_eval) and is_ar
    compute_canon = bool(cfg.eval.compute_canon_metrics)
    pred_mode_eval = pred_mode_from_outfmt(_get_output_format(val_loader))

    total_loss, total_rot_err, total_trans_err = 0.0, 0.0, 0.0
    total_canon_rot, total_canon_trans, canon_count = 0.0, 0.0, 0.0
    ro_rot_sum, ro_trans_sum = 0.0, 0.0
    ade_trans_sum, fde_trans_sum, ade_rot_sum, fde_rot_sum = 0.0, 0.0, 0.0, 0.0
    drift_trans_sum, drift_rot_sum, slope_trans_sum, slope_rot_sum = 0.0, 0.0, 0.0, 0.0
    num_samples, seq_count = 0.0, 0.0

    if do_roll and rank == 0:
        print("[dim]AR[/dim] computing extra rollout metrics in evaluate_loss (may duplicate evaluate_sampler)")

    for batch in tqdm(val_loader, desc="Evaluating", disable=rank != 0):
        _maybe_trim_context_for_eval(batch, cfg)
        cond = core_model.condition_from_batch(batch)
        out = core_model.compute_loss({**batch, **cond})
        bs = _batch_size_from(batch)

        total_loss += float(out["loss"].detach()) * bs
        total_rot_err += float(out.get("rot_deg", torch.tensor(0.0)).detach()) * bs
        total_trans_err += float(out.get("trans_l2", torch.tensor(0.0)).detach()) * bs
        num_samples += bs

        pred_9d = out.get("pred_9d")
        if isinstance(pred_9d, torch.Tensor) and pred_9d.dim() == 3 and pred_9d.shape[-1] == 9:
            gt_9d = torch.as_tensor(batch["target_future"], device=pred_9d.device, dtype=pred_9d.dtype).view_as(pred_9d)
            seq_count += pred_9d.shape[0]
            trans_err, rot_err = _tokenwise_trans_rot_errors(pred_9d, gt_9d)
            ade_trans_sum += float(trans_err.mean(dim=-1).sum())
            fde_trans_sum += float(trans_err[..., -1].sum())
            ade_rot_sum += float(rot_err.mean(dim=-1).sum())
            fde_rot_sum += float(rot_err[..., -1].sum())

            drift_t, drift_r, slope_t, slope_r = _compute_drift_and_slope(trans_err, rot_err)
            drift_trans_sum += float(drift_t.sum())
            drift_rot_sum += float(drift_r.sum())
            slope_trans_sum += float(slope_t.sum())
            slope_rot_sum += float(slope_r.sum())

            if compute_canon:
                for b in range(pred_9d.shape[0]):
                    meta = _build_sample_meta(batch, b)
                    if meta["T_cam_anchor_obj"] is None:
                        continue
                    pred_T, _ = canonicalize_preds_to_anchor(pred_9d[b : b + 1], meta, pred_mode_eval, "w<-c", True, False)
                    gt_T, _ = canonicalize_preds_to_anchor(gt_9d[b : b + 1], meta, pred_mode_eval, "w<-c", True, False)
                    total_canon_rot += float(_geod_deg(pred_T[..., :3, :3].reshape(-1, 3, 3), gt_T[..., :3, :3].reshape(-1, 3, 3)).mean())
                    total_canon_trans += float(torch.linalg.norm(pred_T[..., :3, 3] - gt_T[..., :3, 3], dim=-1).mean())
                    canon_count += 1.0

        if do_roll:
            y_pred = ar_rollout_predict(core_model, batch, device)
            y_gt = torch.as_tensor(batch["target_future"], device=y_pred.device, dtype=y_pred.dtype).view_as(y_pred)
            ro_rot_sum += float(_geod_deg(_r6_to_R(y_pred[..., 3:9].reshape(-1, 6)), _r6_to_R(y_gt[..., 3:9].reshape(-1, 6))).mean()) * bs
            ro_trans_sum += float(torch.linalg.norm(y_pred[..., :3] - y_gt[..., :3], dim=-1).mean()) * bs

    vec = torch.tensor(
        [
            total_loss,
            total_rot_err,
            total_trans_err,
            ro_rot_sum,
            ro_trans_sum,
            total_canon_rot,
            total_canon_trans,
            canon_count,
            ade_trans_sum,
            fde_trans_sum,
            ade_rot_sum,
            fde_rot_sum,
            drift_trans_sum,
            drift_rot_sum,
            slope_trans_sum,
            slope_rot_sum,
            seq_count,
            num_samples,
        ],
        device=device,
        dtype=torch.float64,
    )
    if is_dist:
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)

    vals = vec.tolist()
    tot_n, tot_seq = max(1.0, vals[17]), max(1.0, vals[16])

    metrics = {
        "val_loss": vals[0] / tot_n,
        "rot_deg_mean": vals[1] / tot_n,
        "trans_l2_mean": vals[2] / tot_n,
        "ade_t": vals[8] / tot_seq,
        "fde_t": vals[9] / tot_seq,
        "are_r": vals[10] / tot_seq,
        "fre_r": vals[11] / tot_seq,
        "trans_drift_per_step": vals[12] / tot_seq,
        "rot_drift_per_step": vals[13] / tot_seq,
        "trans_error_slope": vals[14] / tot_seq,
        "rot_error_slope": vals[15] / tot_seq,
    }
    if vals[7] > 0.0:
        metrics["canon_rot_deg_mean"] = vals[5] / vals[7]
        metrics["canon_trans_l2_mean"] = vals[6] / vals[7]
    if do_roll:
        metrics["rollout_rot_deg_mean"] = vals[3] / tot_n
        metrics["rollout_trans_l2_mean"] = vals[4] / tot_n
    return metrics


def _infer_model_horizon(core_model, cfg) -> int:
    temporal_kind = str(getattr(core_model, "_temporal_kind", getattr(cfg.temporal, "kind", "dit"))).lower()

    if temporal_kind == "dit":
        dit_mod = getattr(core_model, "dit", getattr(core_model, "temporal", None))
        if hasattr(dit_mod, "out_dim"):
            return int(dit_mod.out_dim) // 9

    if hasattr(cfg.model, "H"):
        return int(cfg.model.H)
    if hasattr(cfg, "data") and hasattr(cfg.data, "H"):
        return int(cfg.data.H)
    raise RuntimeError("Could not infer model horizon; set model.temporal.out_dim or cfg.model.H / cfg.data.H")


def _parse_eval_horizons(horizons_cfg, H_model: int, max_len: int) -> list[int]:
    if isinstance(horizons_cfg, (int, float)):
        horizons = [int(horizons_cfg)]
    elif isinstance(horizons_cfg, str):
        horizons = [int(p.strip()) for p in horizons_cfg.split(",") if p.strip()]
    elif horizons_cfg is None:
        horizons = [H_model]
    else:
        horizons = [int(h) for h in horizons_cfg]
    horizons = sorted({h for h in horizons if 1 <= h <= max_len})
    return horizons if horizons else [max_len]


def _metric_vals_for_horizon(arr, sample_counts, h_idx: int, repeats: int) -> list[float]:
    return [_safe_div(arr[r][h_idx], sample_counts[r][h_idx]) for r in range(repeats)]


@torch.no_grad()
def evaluate_sampler(model: PoserV1, val_loader: DataLoader, cfg: DictConfig, device: torch.device) -> dict:
    """
    Sampler-based evaluation that dispatches to DDIM (DiT) or AR rollout based on model type.
    """
    model.eval()
    core_model = model.module if hasattr(model, "module") else model
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    # Determine model type
    _kind = str(cfg.model.temporal_kind).lower()
    is_ar = _kind == "ar_transformer"
    if is_ar:
        sampler_type = "AR rollout"
    else:
        sampler_type = "DDIM"

    # DDIM-specific parameters (only used for DiT)
    steps, eta = int(cfg.eval.steps), float(cfg.eval.eta)
    repeats, base_seed = int(cfg.eval.repeats), int(cfg.eval.seed)

    if rank == 0:
        if is_ar:
            print(f"eval_mode=sampler • type={sampler_type} seed_base={base_seed} repeats={repeats}")
        else:
            print(f"eval_mode=sampler • type={sampler_type} steps={steps} eta={eta:.3f} seed_base={base_seed} repeats={repeats}")

    H_data = int(cfg.data.H) if hasattr(cfg.data, "H") else None
    H_model = _infer_model_horizon(core_model, cfg)
    if H_data is not None and H_model > H_data:
        if rank == 0:
            print(f"[yellow]Warning[/yellow] • model horizon H={H_model} exceeds dataset H={H_data}; clamping.")
        H_model = H_data

    max_len = H_model if H_data is None else min(H_model, H_data)
    eval_horizons = _parse_eval_horizons(cfg.eval.horizons, H_model, max_len)
    num_horizons = len(eval_horizons)
    base_idx = eval_horizons.index(H_model) if H_model in eval_horizons else num_horizons - 1
    base_h = eval_horizons[base_idx]

    if rank == 0:
        print(f"evaluate_sampler • model_H={H_model} data_H={H_data} eval_horizons={eval_horizons}")

    rot_sums = [[0.0] * num_horizons for _ in range(repeats)]
    trans_sums = [[0.0] * num_horizons for _ in range(repeats)]
    sample_counts = [[0.0] * num_horizons for _ in range(repeats)]
    ade_trans_sums = [[0.0] * num_horizons for _ in range(repeats)]
    fde_trans_sums = [[0.0] * num_horizons for _ in range(repeats)]
    ade_rot_sums = [[0.0] * num_horizons for _ in range(repeats)]
    fde_rot_sums = [[0.0] * num_horizons for _ in range(repeats)]
    drift_trans_sums = [[0.0] * num_horizons for _ in range(repeats)]
    drift_rot_sums = [[0.0] * num_horizons for _ in range(repeats)]
    slope_trans_sums = [[0.0] * num_horizons for _ in range(repeats)]
    slope_rot_sums = [[0.0] * num_horizons for _ in range(repeats)]
    canon_rot_sums, canon_trans_sums, canon_counts = [0.0] * repeats, [0.0] * repeats, [0.0] * repeats

    pred_mode_eval = pred_mode_from_outfmt(_get_output_format(val_loader))
    compute_canon, apply_sanity = bool(cfg.eval.compute_canon_metrics), bool(cfg.eval.apply_sanity_checks)

    global_sample_idx = 0
    progress_desc = f"Evaluating ({sampler_type})"
    for batch in tqdm(val_loader, desc=progress_desc, disable=rank != 0):
        _maybe_trim_context_for_eval(batch, cfg)
        B, H = _batch_size_from(batch), H_model

        cond = core_model.condition_from_batch(batch)
        T_cam_anchor_obj = cond.get("T_cam_anchor_obj")
        cond_embed = core_model.encode(cond["scene_pcd"], cond["context_vec"], T_cam_anchor_obj=T_cam_anchor_obj)
        ctx_tokens_9d = batch.get("context_init_9d")

        y_gt = batch.get("target_future")
        if isinstance(y_gt, list):
            y_gt = torch.from_numpy(np.stack(y_gt)).float()
        elif hasattr(y_gt, "__array__") and not isinstance(y_gt, torch.Tensor):
            y_gt = torch.as_tensor(y_gt).float()
        if isinstance(y_gt, torch.Tensor) and y_gt.dim() == 2:
            y_gt = y_gt.unsqueeze(0)

        for r in range(repeats):
            gen = torch.Generator(device=device)
            gen.manual_seed(base_seed + global_sample_idx + r * 10_000_003)

            # Dispatch to appropriate sampler based on model type
            if is_ar:
                y_pred = ar_rollout_sample(poser_model=core_model, cond_embed=cond_embed, H=H, ctx_tokens_9d=ctx_tokens_9d).to(device)
            else:
                y_pred = ddim_sample_tokens(poser_model=core_model, cond_embed=cond_embed, H=H, steps=steps, eta=eta, generator=gen, ctx_tokens_9d=ctx_tokens_9d).to(device)

            y_gt_t = torch.as_tensor(y_gt, device=y_pred.device, dtype=y_pred.dtype)
            if y_gt_t.dim() == 2:
                y_gt_t = y_gt_t.unsqueeze(0)
            if y_gt_t.shape[-2] < H:
                continue
            if y_gt_t.shape[-2] > H:
                y_gt_t = y_gt_t[..., :H, :]

            trans_err_tok, rot_err_tok = _tokenwise_trans_rot_errors(y_pred, y_gt_t)
            T_total = trans_err_tok.shape[-1]

            for h_idx, h_eval in enumerate(eval_horizons):
                if h_eval > T_total:
                    continue
                err_t, err_r = trans_err_tok[..., :h_eval], rot_err_tok[..., :h_eval]
                rot_sums[r][h_idx] += float(err_r.mean()) * B
                trans_sums[r][h_idx] += float(err_t.mean()) * B
                sample_counts[r][h_idx] += B
                ade_trans_sums[r][h_idx] += float(err_t.mean(dim=-1).sum())
                fde_trans_sums[r][h_idx] += float(err_t[..., -1].sum())
                ade_rot_sums[r][h_idx] += float(err_r.mean(dim=-1).sum())
                fde_rot_sums[r][h_idx] += float(err_r[..., -1].sum())

                drift_t, drift_r, slope_t, slope_r = _compute_drift_and_slope(err_t, err_r)
                drift_trans_sums[r][h_idx] += float(drift_t.sum())
                drift_rot_sums[r][h_idx] += float(drift_r.sum())
                slope_trans_sums[r][h_idx] += float(slope_t.sum())
                slope_rot_sums[r][h_idx] += float(slope_r.sum())

            if compute_canon:
                for b in range(B):
                    meta = _build_sample_meta(batch, b)
                    if meta["T_cam_anchor_obj"] is None:
                        continue
                    pred_T, _ = canonicalize_preds_to_anchor(y_pred[b : b + 1], meta, pred_mode_eval, "w<-c", True, False)
                    if apply_sanity:
                        sample = {"T_cam_anchor_obj": meta["T_cam_anchor_obj"], "anchor_local_idx": meta["anchor_local_idx"]}
                        pred_T = apply_sanity_checks(pred_T[0], sample, cfg, namespace="infer").unsqueeze(0)
                    gt_T, _ = canonicalize_preds_to_anchor(y_gt_t[b : b + 1], meta, pred_mode_eval, "w<-c", True, False)
                    canon_rot_sums[r] += float(_geod_deg(pred_T[..., :3, :3].reshape(-1, 3, 3), gt_T[..., :3, :3].reshape(-1, 3, 3)).mean())
                    canon_trans_sums[r] += float(torch.linalg.norm(pred_T[..., :3, 3] - gt_T[..., :3, 3], dim=-1).mean())
                    canon_counts[r] += 1.0

        global_sample_idx += B

    if is_dist:
        for r in range(repeats):
            vec_canon = torch.tensor([canon_rot_sums[r], canon_trans_sums[r], canon_counts[r]], device=device, dtype=torch.float64)
            dist.all_reduce(vec_canon, op=dist.ReduceOp.SUM)
            canon_rot_sums[r], canon_trans_sums[r], canon_counts[r] = vec_canon.tolist()

            for h_idx in range(num_horizons):
                vec = torch.tensor(
                    [
                        rot_sums[r][h_idx],
                        trans_sums[r][h_idx],
                        sample_counts[r][h_idx],
                        ade_trans_sums[r][h_idx],
                        fde_trans_sums[r][h_idx],
                        ade_rot_sums[r][h_idx],
                        fde_rot_sums[r][h_idx],
                        drift_trans_sums[r][h_idx],
                        drift_rot_sums[r][h_idx],
                        slope_trans_sums[r][h_idx],
                        slope_rot_sums[r][h_idx],
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)
                v = vec.tolist()
                rot_sums[r][h_idx], trans_sums[r][h_idx], sample_counts[r][h_idx] = v[0], v[1], v[2]
                ade_trans_sums[r][h_idx], fde_trans_sums[r][h_idx] = v[3], v[4]
                ade_rot_sums[r][h_idx], fde_rot_sums[r][h_idx] = v[5], v[6]
                drift_trans_sums[r][h_idx], drift_rot_sums[r][h_idx] = v[7], v[8]
                slope_trans_sums[r][h_idx], slope_rot_sums[r][h_idx] = v[9], v[10]

    metrics_by_horizon: dict[str, dict[str, float]] = {}
    all_stats = {k: [] for k in ["rot", "trans", "ade_t", "fde_t", "are_r", "fre_r", "drift_t", "drift_r", "slope_t", "slope_r"]}

    for h_idx, h_eval in enumerate(eval_horizons):
        rm, rs = _mean_std(_metric_vals_for_horizon(rot_sums, sample_counts, h_idx, repeats))
        tm, ts = _mean_std(_metric_vals_for_horizon(trans_sums, sample_counts, h_idx, repeats))
        ade_tm, ade_ts = _mean_std(_metric_vals_for_horizon(ade_trans_sums, sample_counts, h_idx, repeats))
        fde_tm, fde_ts = _mean_std(_metric_vals_for_horizon(fde_trans_sums, sample_counts, h_idx, repeats))
        are_rm, are_rs = _mean_std(_metric_vals_for_horizon(ade_rot_sums, sample_counts, h_idx, repeats))
        fre_rm, fre_rs = _mean_std(_metric_vals_for_horizon(fde_rot_sums, sample_counts, h_idx, repeats))
        drift_tm, drift_ts = _mean_std(_metric_vals_for_horizon(drift_trans_sums, sample_counts, h_idx, repeats))
        drift_rm, drift_rs = _mean_std(_metric_vals_for_horizon(drift_rot_sums, sample_counts, h_idx, repeats))
        slope_tm, slope_ts = _mean_std(_metric_vals_for_horizon(slope_trans_sums, sample_counts, h_idx, repeats))
        slope_rm, slope_rs = _mean_std(_metric_vals_for_horizon(slope_rot_sums, sample_counts, h_idx, repeats))

        all_stats["rot"].append((rm, rs))
        all_stats["trans"].append((tm, ts))
        all_stats["ade_t"].append((ade_tm, ade_ts))
        all_stats["fde_t"].append((fde_tm, fde_ts))
        all_stats["are_r"].append((are_rm, are_rs))
        all_stats["fre_r"].append((fre_rm, fre_rs))
        all_stats["drift_t"].append((drift_tm, drift_ts))
        all_stats["drift_r"].append((drift_rm, drift_rs))
        all_stats["slope_t"].append((slope_tm, slope_ts))
        all_stats["slope_r"].append((slope_rm, slope_rs))

        h_metrics: dict[str, float] = {
            "rot_deg_mean": rm,
            "trans_l2_mean": tm,
            "ade_t": ade_tm,
            "fde_t": fde_tm,
            "are_r": are_rm,
            "fre_r": fre_rm,
            "trans_drift_per_step": drift_tm,
            "rot_drift_per_step": drift_rm,
            "trans_error_slope": slope_tm,
            "rot_error_slope": slope_rm,
        }
        if repeats > 1:
            h_metrics.update(
                {
                    "rot_deg_mean_std": rs,
                    "trans_l2_mean_std": ts,
                    "ade_t_std": ade_ts,
                    "fde_t_std": fde_ts,
                    "are_r_std": are_rs,
                    "fre_r_std": fre_rs,
                    "trans_drift_per_step_std": drift_ts,
                    "rot_drift_per_step_std": drift_rs,
                    "trans_error_slope_std": slope_ts,
                    "rot_error_slope_std": slope_rs,
                }
            )
        metrics_by_horizon[str(h_eval)] = h_metrics

    canon_rot_m, canon_rot_s = _mean_std([_safe_div(canon_rot_sums[r], canon_counts[r]) for r in range(repeats)])
    canon_trans_m, canon_trans_s = _mean_std([_safe_div(canon_trans_sums[r], canon_counts[r]) for r in range(repeats)])

    metrics = {
        "H_model": H_model,
        "H_data": H_data,
        "eval_horizons": eval_horizons,
        "base_horizon": base_h,
        "sampler_type": sampler_type,
        "rot_deg_mean": all_stats["rot"][base_idx][0],
        "trans_l2_mean": all_stats["trans"][base_idx][0],
        "ade_t": all_stats["ade_t"][base_idx][0],
        "fde_t": all_stats["fde_t"][base_idx][0],
        "are_r": all_stats["are_r"][base_idx][0],
        "fre_r": all_stats["fre_r"][base_idx][0],
        "trans_drift_per_step": all_stats["drift_t"][base_idx][0],
        "rot_drift_per_step": all_stats["drift_r"][base_idx][0],
        "trans_error_slope": all_stats["slope_t"][base_idx][0],
        "rot_error_slope": all_stats["slope_r"][base_idx][0],
        "seed_base": base_seed,
        "repeats": repeats,
        "metrics_by_horizon": metrics_by_horizon,
    }
    # Only include DDIM-specific parameters for DiT models
    if not is_ar:
        metrics["steps"] = steps
        metrics["eta"] = eta
    if sum(canon_counts) > 0.0:
        metrics["canon_rot_deg_mean"] = canon_rot_m
        metrics["canon_trans_l2_mean"] = canon_trans_m
        if repeats > 1:
            metrics["canon_rot_deg_mean_std"] = canon_rot_s
            metrics["canon_trans_l2_mean_std"] = canon_trans_s

    if rank == 0:
        for h_eval in eval_horizons:
            epoch_table(title=f"Evaluation Results • horizon={h_eval}", metrics=metrics_by_horizon[str(h_eval)], prec=int(cfg.log.precision))

    if repeats > 1:
        per_repeat = []
        for r in range(repeats):
            md = {
                "rot_deg_mean": _safe_div(rot_sums[r][base_idx], sample_counts[r][base_idx]),
                "trans_l2_mean": _safe_div(trans_sums[r][base_idx], sample_counts[r][base_idx]),
                "ade_t": _safe_div(ade_trans_sums[r][base_idx], sample_counts[r][base_idx]),
                "fde_t": _safe_div(fde_trans_sums[r][base_idx], sample_counts[r][base_idx]),
                "are_r": _safe_div(ade_rot_sums[r][base_idx], sample_counts[r][base_idx]),
                "fre_r": _safe_div(fde_rot_sums[r][base_idx], sample_counts[r][base_idx]),
                "trans_drift_per_step": _safe_div(drift_trans_sums[r][base_idx], sample_counts[r][base_idx]),
                "rot_drift_per_step": _safe_div(drift_rot_sums[r][base_idx], sample_counts[r][base_idx]),
                "trans_error_slope": _safe_div(slope_trans_sums[r][base_idx], sample_counts[r][base_idx]),
                "rot_error_slope": _safe_div(slope_rot_sums[r][base_idx], sample_counts[r][base_idx]),
            }
            if canon_counts[r] > 0.0:
                md["canon_rot_deg_mean"] = _safe_div(canon_rot_sums[r], canon_counts[r])
                md["canon_trans_l2_mean"] = _safe_div(canon_trans_sums[r], canon_counts[r])
            per_repeat.append(md)
        return metrics, per_repeat
    return metrics


# Backward compatibility alias
evaluate_inference = evaluate_sampler


def _rel_path(p: str, root: str) -> str:
    return os.path.relpath(str(p), start=root)


def build_val_subset(
    dataset: SceneSequenceDataset | SceneSequenceDatasetSynth | HOT3DClipsDataset,
    cfg: DictConfig,
    seed: int,
) -> Subset | SceneSequenceDataset | SceneSequenceDatasetSynth | HOT3DClipsDataset:
    """Build validation subset from dataset.

    For HOT3DClipsDataset:
      - In tiny_overfit mode: returns first tiny_n samples
      - Otherwise: returns full dataset (no train/val split for HOT3D currently)

    For SceneSequenceDataset:
      - Uses existing train/val split logic based on object directories
    """
    # Handle HOT3D dataset
    if isinstance(dataset, HOT3DClipsDataset):
        if cfg.train.tiny_overfit:
            tiny_n = max(1, int(cfg.train.tiny_n))
            indices = list(range(min(tiny_n, len(dataset))))
            return Subset(dataset, indices)

        # Skip internal splitting for official test set
        if dataset.split != "train":
            return dataset

        rank = get_rank()

        # Use base seed (rank-invariant) for deterministic split
        base_seed = int(cfg.train.seed)
        train_ratio = float(getattr(cfg.data, "train_ratio_per_video", 0.70))

        # Check if shared_scenes_split mode is enabled
        shared_scenes_split = bool(getattr(cfg.data, "shared_scenes_split", False))

        # IMPORTANT: Barrier before checking split to avoid GPFS metadata caching issues.
        _barrier()

        if shared_scenes_split:
            # ---- Shared-scenes split: all clips appear in both train and val ----
            # Get windows cache key to ensure split matches current windows (post_train params, etc.)
            windows_key = dataset._get_cache_key() if hasattr(dataset, "_get_cache_key") else ""

            split_tuple = load_hot3d_shared_scenes_split(dataset.clips_root, dataset.split, windows_key)

            if split_tuple is None:
                # Create split if not present - rank0 only with barrier
                if rank == 0:
                    train_pairs, val_pairs, train_win_keys, val_win_keys = _build_hot3d_shared_scenes_split(dataset.windows, base_seed, train_ratio)
                    save_hot3d_shared_scenes_split(dataset.clips_root, dataset.split, train_pairs, val_pairs, train_win_keys, val_win_keys, windows_key)
                _barrier()
                split_tuple = load_hot3d_shared_scenes_split(dataset.clips_root, dataset.split, windows_key)

            _, val_pairs, _, val_win_keys = split_tuple

            # Build val indices using window keys
            val_indices = [i for i, w in enumerate(dataset.windows) if (w["clip_id"], w["object_uid"], w["t0"]) in val_win_keys]

            if not val_indices:
                raise ValueError(f"No val windows. Check {dataset.clips_root}/split/")

            if rank == 0:
                print(f"[dim]eval shared-scenes split[/dim] val_pairs={len(val_pairs):,} val_windows={len(val_indices):,}")

            return Subset(dataset, val_indices)

        else:
            # ---- Original per-clip split ----
            split_tuple = load_hot3d_split(dataset.clips_root, dataset.split)

            if split_tuple is None:
                # Create split if not present - rank0 only with barrier
                if rank == 0:
                    train_pairs, val_pairs = _build_hot3d_per_clip_split(dataset.windows, base_seed, train_ratio)
                    save_hot3d_split(dataset.clips_root, dataset.split, train_pairs, val_pairs)
                _barrier()
                split_tuple = load_hot3d_split(dataset.clips_root, dataset.split)

            _, val_pairs = split_tuple

            val_indices = [i for i, w in enumerate(dataset.windows) if (w["clip_id"], w["object_uid"]) in val_pairs]

            if not val_indices:
                raise ValueError(f"No val windows. Check {dataset.clips_root}/split/")

            if rank == 0:
                print(f"[dim]eval split[/dim] val_pairs={len(val_pairs):,} val_windows={len(val_indices):,}")

            return Subset(dataset, val_indices)

    # Handle synthetic dataset
    if not isinstance(dataset, SceneSequenceDataset):
        if cfg.train.tiny_overfit:
            tiny_n = max(1, int(cfg.train.tiny_n))
            indices = list(range(min(tiny_n, len(dataset))))
            return Subset(dataset, indices)
        return dataset

    # SceneSequenceDataset logic
    root = str(dataset.dataset_root)

    if cfg.train.tiny_overfit:
        split_tuple = load_split(dataset.dataset_root)
        train_rel = split_tuple[0] if split_tuple else []
        train_set = set(train_rel)
        train_indices = [i for i, w in enumerate(dataset.windows) if _rel_path(w.get("object_dir", ""), root) in train_set]
        if not train_indices:
            train_indices = list(range(len(dataset)))
        return Subset(dataset, train_indices[: max(1, int(cfg.train.tiny_n))])

    split_tuple = load_split(dataset.dataset_root)
    if split_tuple is None:
        per_video: dict[str, list[str]] = {}
        for rec in dataset.records:
            if int(rec.get("n_frames", 0)) >= dataset.min_frames:
                per_video.setdefault(str(rec.get("video_id", "")), []).append(_rel_path(str(rec["object_dir"]), root))
        rng = np.random.default_rng(seed)
        train_ratio = min(1.0, max(0.0, float(cfg.data.train_ratio_per_video)))
        val_rel = []
        for vid in sorted(per_video.keys()):
            uniq = sorted(set(per_video[vid]))
            rng.shuffle(uniq)
            n_train = max(1, min(int(round(train_ratio * len(uniq))), len(uniq)))
            val_rel.extend(uniq[n_train:])
    else:
        _, val_rel = split_tuple

    val_set = set(val_rel)
    val_indices = [i for i, w in enumerate(dataset.windows) if _rel_path(w.get("object_dir", ""), root) in val_set]
    return Subset(dataset, val_indices) if val_indices else Subset(dataset, list(range(len(dataset))))


def _find_best_step_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    best_path, best_step = None, -1
    for fname in os.listdir(ckpt_dir):
        if fname.startswith("step_") and fname.endswith(".pt"):
            step_val = int(fname[5:-3])
            full_path = os.path.join(ckpt_dir, fname)
            if os.path.isfile(full_path) and step_val > best_step:
                best_step, best_path = step_val, full_path
    return best_path


@hydra.main(config_path="../conf", config_name="epic", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Starting evaluation…")
    # Register eval resolver BEFORE apply_config_adapter (it accesses temporal_dit.out_dim which uses ${eval:...})
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: int(eval(expr, {"__builtins__": {}}, {})))
    cfg = apply_config_adapter(cfg)

    device, is_dist, rank, _, seed = init_distributed_env(base_seed=int(cfg.train.seed), timeout_minutes=30)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    val_dataset = build_val_subset(get_dataset(cfg), cfg, seed)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        sampler=DistributedSampler(val_dataset, shuffle=False, drop_last=False) if is_dist else None,
        num_workers=4,
        prefetch_factor=int(cfg.train.prefetch_factor),
        pin_memory=True,
        persistent_workers=True,
        multiprocessing_context=CTX_SPAWN,
        drop_last=False,
        collate_fn=debug_collate,
        worker_init_fn=seed_worker,
        generator=g,
    )

    model: PoserV1 = instantiate(cfg.model, _recursive_=False)
    model.to(device)
    if is_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
            static_graph=True,
        )

    ckpt_dir = os.path.join(cfg.train.out_dir, "checkpoints")
    ckpt_path = str(cfg.eval.ckpt) if cfg.eval.ckpt else os.path.join(ckpt_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, "last.pt") if os.path.exists(os.path.join(ckpt_dir, "last.pt")) else (_find_best_step_checkpoint(ckpt_dir) or ckpt_path)

    if _is_rank0():
        print(f"[bold green]Loading checkpoint[/bold green] → {ckpt_path}")

    prefer_ema = bool(cfg.eval.prefer_ema)
    state_dict, _meta = resolve_and_load_state_dict(ckpt_path, map_location="cpu", prefer_ema=prefer_ema)
    lazy_state = pop_lazy_encoder_proj_state(state_dict)
    core_model = model.module if hasattr(model, "module") else model
    missing, unexpected = core_model.load_state_dict(state_dict, strict=False)
    restore_lazy_encoder_proj(core_model, lazy_state)

    # Optional torch.compile (must happen after state_dict restore)
    maybe_enable_compile(model, cfg, print_fn=(print if _is_rank0() else None))

    if _is_rank0():
        eff_missing = [k for k in (missing or []) if k not in lazy_state]
        print(f"ckpt loaded • params={len(state_dict)} missing={len(eff_missing)} unexpected={len(unexpected)}")
        print(f"[dim]ckpt ema[/dim] • prefer_ema={prefer_ema} has_ema={isinstance((_meta or {}).get('ema_state_dict'), dict)}")

        # Check conditioning mode mismatch
        _meta_cfg = (_meta or {}).get("cfg") or {}
        _meta_temporal = _meta_cfg.get("temporal") if isinstance(_meta_cfg, dict) else None
        ckpt_conditioning = _meta_temporal.get("conditioning") if isinstance(_meta_temporal, dict) else None
        curr_conditioning = getattr(cfg.temporal, "conditioning", "film")
        if ckpt_conditioning is not None and ckpt_conditioning != curr_conditioning:
            print(f"[yellow]WARNING: Config conditioning='{curr_conditioning}' differs from checkpoint conditioning='{ckpt_conditioning}'. Results may be incorrect.[/yellow]")

    if _is_rank0() and cfg.log.show_config_summary:
        print_config_summary(cfg)
    if _is_rank0() and cfg.log.show_model_summary:
        enc_name = cfg.model.encoder._target_.split(".")[-1] if "_target_" in cfg.model.encoder else "encoder"
        temp_name = "AutoregTransformerPose" if str(cfg.model.temporal_kind).lower() == "ar_transformer" else "DiT"
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        print_model_summary(encoder_name=enc_name, temporal_name=temp_name, params_m=params_m, horizon=cfg.data.H, context_len=cfg.data.context_len)

    if str(cfg.eval.eval_mode).lower() == "loss":
        metrics = evaluate_loss(model, val_loader, cfg, device)
    else:
        metrics = evaluate_sampler(model, val_loader, cfg, device)

    if _is_rank0():
        agg_metrics, per_repeat = (metrics, None) if not isinstance(metrics, tuple) else metrics
        if per_repeat:
            for i, m in enumerate(per_repeat):
                epoch_table(title=f"Evaluation Results • Repeat {i + 1}/{len(per_repeat)}", metrics=m, prec=int(cfg.log.precision))
        display = {k: v for k, v in (agg_metrics or {}).items() if isinstance(v, (int, float))}
        epoch_table(title=f"Evaluation Results ({os.path.basename(ckpt_path)})", metrics=display, prec=int(cfg.log.precision))

        out_json = os.path.join(cfg.train.out_dir, "eval_metrics.json")
        os.makedirs(cfg.train.out_dir, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(agg_metrics, f, indent=2)
        print(f"[dim]Saved metrics → {out_json}[/dim]")

    _barrier()
    if dist.is_available() and dist.is_initialized():
        _destroy_pg()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
