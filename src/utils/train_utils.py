"""Training utilities for train/eval/infer entrypoints."""

from __future__ import annotations

import inspect
import json
import os
import re
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from .logger import rprint as print


class IndexDataset(Dataset):
    """Dataset wrapper that injects the sample index into dict samples.

    This is used for debugging rare training crashes by allowing the batch to carry
    per-sample dataset indices (without modifying underlying dataset cache files).
    """

    def __init__(self, base: Dataset, *, key: str = "__dataset_idx") -> None:
        self.base = base
        self.key = str(key)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):  # type: ignore[override]
        sample = self.base[idx]
        if isinstance(sample, dict):
            # Avoid mutating cached dicts in-place (e.g., sample cache returns a shared object)
            out = dict(sample)
            out[self.key] = int(idx)
            return out
        return sample


def _collate_vals(vals):
    if not vals:
        return vals
    v0 = vals[0]
    if isinstance(v0, dict):
        out = {}
        for kk in v0.keys():
            sub = [v.get(kk) for v in vals]
            if any(s is None for s in sub):
                raise RuntimeError(f"Collate failure: nested key='{kk}' has None for items {[i for i, s in enumerate(sub) if s is None]}")
            out[kk] = _collate_vals(sub)
        return out
    if isinstance(v0, np.ndarray):
        try:
            return torch.from_numpy(np.stack([np.ascontiguousarray(v) for v in vals], axis=0))
        except ValueError:
            return [torch.from_numpy(np.ascontiguousarray(v)) for v in vals]
    if isinstance(v0, torch.Tensor):
        try:
            return torch.stack(vals, dim=0)
        except RuntimeError:
            return vals
    if isinstance(v0, (int, float, bool, np.integer, np.floating, np.bool_)):
        return torch.as_tensor(vals)
    if isinstance(v0, str):
        return list(vals)
    return default_collate(vals)


def _to_list(v):
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [float(v)] if v is not None else None


def write_model_io_report(ds: Any, train_loader: DataLoader, cfg: Any, logger_print: Callable[[str], None]) -> None:
    from ..data.datasets import SceneSequenceDataset, SceneSequenceDatasetSynth

    sample0 = ds[0] if isinstance(ds, (SceneSequenceDataset, SceneSequenceDatasetSynth)) else next(iter(train_loader))
    ip = sample0.get("init_pose") if isinstance(sample0, dict) else None
    scene_arr = sample0.get("scene_pcd0", sample0.get("scene_pcd")) if isinstance(sample0, dict) else None
    eval_cfg = getattr(cfg, "eval", {}) or {}
    report = {
        "inputs": {
            "scene_pcd": tuple(scene_arr.shape) if hasattr(scene_arr, "shape") else None,
            "init_pose": {"t0": _to_list(ip.get("t0") if ip else None), "rot6d0": _to_list(ip.get("rot6d0") if ip else None)},
            "K": tuple(sample0["K"].shape) if sample0.get("K") is not None else None,
            "T_cam_anchor_obj": tuple(sample0["T_cam_anchor_obj"].shape) if sample0.get("T_cam_anchor_obj") is not None else None,
        },
        "preprocessing": {
            "context_vec": "concat[t0(3), rot6d0(6), bbox(4)] (+mesh if available) → Linear(context_dim)",
            "targets": f"{getattr(ds, 'output_format', 'abs_in_anchor')} packed as (t[3]+rot6d[6])",
        },
        "fusion": {"pathway": "encoder(scene_pcd + context_vec [+color/normal]) → DiT diffusion"},
        "embeddings": {
            "encoder": cfg.model.encoder._target_.split(".")[-1] if "_target_" in cfg.model.encoder else "encoder",
            "temporal": cfg.model.temporal._target_.split(".")[-1] if "_target_" in cfg.model.temporal else "temporal",
            "context_dim": int(cfg.model.encoder.get("context_dim", cfg.model.get("context_dim", 256))) if hasattr(cfg, "model") else 256,
        },
        "output_head": {"shape": [int(cfg.data.H), 9], "parameterization": "se3(t[3]+rot6d[6])"},
        "loss": "diffusion_v_mse + aux(rot_geodesic, trans_L2)",
        "selection": {
            "metric": "canon_rot_deg_mean (DDIM), lower is better",
            "eval_mode": str(getattr(eval_cfg, "eval_mode", "sampler")),
            "steps": int(getattr(eval_cfg, "steps", getattr(getattr(cfg, "model", {}), "temporal", {}).get("ddim_steps", 50))),
            "eta": float(getattr(eval_cfg, "eta", 0.0)),
            "seed_base": int(getattr(eval_cfg, "seed", getattr(cfg.train, "seed", 42))),
            "repeats": int(getattr(eval_cfg, "repeats", 1)),
        },
    }
    rep_path = os.path.join(cfg.train.out_dir, "model_io.json")
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    logger_print(f"[bold]model_io[/bold] → {rep_path}")


def debug_collate(batch):
    if isinstance(batch[0], dict):
        out = {}
        for k in batch[0].keys():
            vals = [b.get(k) for b in batch]
            if any(v is None for v in vals):
                raise RuntimeError(f"Collate failure: key='{k}' has None for items {[i for i, v in enumerate(vals) if v is None]}")
            out[k] = _collate_vals(vals)
        return out
    return default_collate(batch)


class _BackHook:
    def __init__(self, name: str):
        self.name = name
        self.last = {"has_input": False, "has_output": False, "in_shapes": None, "out_shapes": None, "nan": 0, "inf": 0}

    def __call__(self, module, grad_input, grad_output):
        all_grads = [g for g in list(grad_input) + list(grad_output) if isinstance(g, torch.Tensor)]
        self.last = {
            "has_input": any(g is not None for g in grad_input),
            "has_output": any(g is not None for g in grad_output),
            "in_shapes": [tuple(g.shape) for g in grad_input if g is not None],
            "out_shapes": [tuple(g.shape) for g in grad_output if g is not None],
            "nan": sum(int(torch.isnan(g).sum().item()) for g in all_grads),
            "inf": sum(int(torch.isinf(g).sum().item()) for g in all_grads),
        }


class GradFlowAudit:
    def __init__(self, model: torch.nn.Module, opt: torch.optim.Optimizer, cfg: Any):
        self.model = model.module if hasattr(model, "module") else model
        self.opt = opt
        self.cfg = cfg
        self.enabled = bool(getattr(cfg.train, "grad_audit", False))
        self.max_steps = int(getattr(cfg.train, "grad_audit_steps", 1))
        self.step_count = 0
        self.hooks = []
        self.encoder_hook = _BackHook("encoder")
        self.temporal_kind = str(getattr(self.model, "_temporal_kind", "dit")).lower()
        # Label by backend so grad-audit reports are readable.
        self.temporal_label = "DiTPose" if self.temporal_kind == "dit" else "ar_transformer"
        self.temporal_hook = _BackHook(self.temporal_label)
        self.preclip_norm = None
        self.postclip_norm = None
        self.clip_norm_cfg = float(getattr(cfg.train, "grad_clip_norm", 0.0))
        self.amp_overflow_count = 0
        self.scaler_scale = None
        self.scan_findings = []

    def is_active(self) -> bool:
        return self.enabled and self.step_count < self.max_steps

    def register_module_hooks_once(self) -> None:
        if not self.is_active() or self.hooks:
            return
        enc = getattr(self.model, "encoder", None)
        if isinstance(enc, torch.nn.Module) and str(getattr(self.model, "_encoder_output_type", "")).lower() not in ("dict", "mapping"):
            self.hooks.append(enc.register_full_backward_hook(self.encoder_hook))
        dit = getattr(self.model, "dit", None)
        if isinstance(dit, torch.nn.Module):
            self.hooks.append(dit.register_full_backward_hook(self.temporal_hook))

    def clear_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def begin_step(self) -> None:
        active = self.is_active()
        self.model.set_grad_audit(active)
        if active:
            self.register_module_hooks_once()
        elif self.hooks:
            self.clear_hooks()

    def record_preclip(self, total_norm: float | None) -> None:
        self.preclip_norm = float(total_norm) if total_norm is not None else None

    def record_postclip(self, total_norm: float | None) -> None:
        self.postclip_norm = float(total_norm) if total_norm is not None else None

    def _sum_param_grad_norms(self) -> float:
        return sum(float(torch.linalg.vector_norm(p.grad.detach()).item()) for p in self.model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())

    def _param_coverage(self) -> dict:
        n_total, n_req, n_none, n_zero, n_nan, n_inf = 0, 0, 0, 0, 0, 0
        first_nonfinite, norms = None, []
        for name, p in self.model.named_parameters():
            n_total += p.numel()
            if p.requires_grad:
                n_req += 1
            g = p.grad
            if g is None:
                n_none += 1 if p.requires_grad else 0
                continue
            if torch.isnan(g).any():
                n_nan += 1
                first_nonfinite = first_nonfinite or name
            if torch.isinf(g).any():
                n_inf += 1
                first_nonfinite = first_nonfinite or name
            n = float(torch.linalg.vector_norm(g.detach()).item())
            if n < 1e-12:
                n_zero += 1
            else:
                norms.append((n, name))
        norms.sort()
        return {
            "n_params_total": n_total,
            "n_requires_grad_true": n_req,
            "n_grad_none": n_none,
            "n_grad_zero": n_zero,
            "n_grad_nan": n_nan,
            "n_grad_inf": n_inf,
            "topk": [(v, k) for v, k in norms[-5:][::-1]],
            "bottomk": [(v, k) for v, k in norms[:5]],
            "first_nonfinite": first_nonfinite,
        }

    def _loss_inputs_status(self) -> dict:
        tmap = self.model.get_grad_audit_tensors() if hasattr(self.model, "get_grad_audit_tensors") else {}
        status = {}
        for key, ten in tmap.items():
            has_grad = isinstance(ten, torch.Tensor) and ten.grad is not None
            status[key] = {
                "has_grad": has_grad,
                "grad_norm": float(torch.linalg.vector_norm(ten.grad).item()) if has_grad and torch.isfinite(ten.grad).all() else 0.0,
                "nan": int(torch.isnan(ten.grad).sum().item()) if has_grad else 0,
                "inf": int(torch.isinf(ten.grad).sum().item()) if has_grad else 0,
                "has_grad_fn": getattr(ten, "grad_fn", None) is not None,
                "shape": tuple(ten.shape) if isinstance(ten, torch.Tensor) else None,
            }
        return status

    @staticmethod
    def _find_missing_link(loss_inputs: dict) -> str | None:
        order = ["R_pred", "t_pred", "y0_pred_9d", "v_pred", "dit_out", "ar_out", "encoder_out"]
        have = {k: loss_inputs.get(k, {}).get("has_grad", False) for k in order}
        last_grad = None
        for k in order:
            if have.get(k):
                last_grad = k
            elif last_grad is not None:
                return k
        return None

    def _scan_traps(self) -> None:
        from ..geom import se3_ops

        targets = []
        enc, dit = getattr(self.model, "encoder", None), getattr(self.model, "dit", None)
        if isinstance(enc, torch.nn.Module):
            targets.append(("encoder", enc))
        if isinstance(dit, torch.nn.Module):
            targets.append((self.temporal_label, dit))
        targets.append(("se3_ops.rot6d_to_matrix", se3_ops.rot6d_to_matrix))
        if hasattr(self.model, "encode"):
            targets.append(("PoserV1.encode", self.model.encode))
        patterns = [".detach(", ".data", "with torch.no_grad", ".cpu().numpy(", "inplace=True", "add_(", "mul_(", "copy_(", "scatter_(", "index_add_(", "@torch.no_grad"]
        findings = []
        for name, obj in targets:
            try:
                src = inspect.getsource(obj)
                hits = [p for p in patterns if p in src]
                if hits:
                    findings.append({"module": name, "hits": hits})
            except (OSError, TypeError):
                pass
        self.scan_findings = findings

    def build_and_emit_report(self, step_idx: int, lr: float) -> None:
        os.makedirs(os.path.join(self.cfg.train.out_dir, "gradflow"), exist_ok=True)
        loss_inputs = self._loss_inputs_status()
        missing = self._find_missing_link(loss_inputs)
        cov = self._param_coverage()
        warn_msgs = []
        if cov["n_requires_grad_true"] > 0:
            frac_none = cov["n_grad_none"] / max(1, cov["n_requires_grad_true"])
            if frac_none > 0.05:
                warn_msgs.append(f"WARN: {frac_none:.1%} params have grad=None")
        if cov["n_grad_nan"] > 0 or cov["n_grad_inf"] > 0:
            warn_msgs.append(f"WARN: non-finite grads detected (first: {cov.get('first_nonfinite') or '?'})")
        hooks = {"encoder": self.encoder_hook.last, self.temporal_label: self.temporal_hook.last}
        if step_idx == 1:
            self._scan_traps()
        temporal_key = "dit_out" if self.temporal_kind == "dit" else "ar_out"
        rp_tok, rp_anchor = loss_inputs.get("R_pred", {}), loss_inputs.get("R_pred_anchor", {})
        rp_combined = rp_tok if rp_tok.get("has_grad", False) else (rp_anchor if rp_anchor else rp_tok)
        tp_tok, tp_anchor = loss_inputs.get("t_pred", {}), loss_inputs.get("t_pred_anchor", {})
        tp_combined = tp_tok if tp_tok.get("has_grad", False) else (tp_anchor if tp_anchor else tp_tok)
        report = {
            "global": {
                "grad_norm_preclip": self.preclip_norm,
                "grad_norm_postclip": self.postclip_norm,
                "clip_norm": self.clip_norm_cfg,
                "lr": lr,
                "scaler_scale": self.scaler_scale,
                "amp_overflow_count": self.amp_overflow_count,
            },
            "coverage": cov,
            "per_module": {
                "encoder": loss_inputs.get("encoder_out", {}),
                self.temporal_label: loss_inputs.get(temporal_key, {}),
                "head_pre_se3": loss_inputs.get("y0_pred_9d", {}),
                "rot6d_to_R": rp_combined,
                "t_pred": tp_combined,
                "R_pred_anchor": loss_inputs.get("R_pred_anchor", {}),
                "t_pred_anchor": loss_inputs.get("t_pred_anchor", {}),
            },
            "loss_inputs": loss_inputs,
            "first_missing_link": missing,
            "hooks": hooks,
            "trap_scan": self.scan_findings,
        }
        print("[bold cyan]GradFlow Report[/bold cyan]", f"pre={self.preclip_norm:.3e} post={self.postclip_norm:.3e} clip={self.clip_norm_cfg}", f"lr={lr:.3e}")
        for name, info in report["per_module"].items():
            gn, hz = info.get("grad_norm", 0.0), "yes" if info.get("has_grad") else "no"
            print(f"  {name}: has_grad={hz} gnorm={gn:.3e} nan={info.get('nan', 0)} inf={info.get('inf', 0)} all_zero={(gn < 1e-12) and info.get('has_grad', False)}")
        if missing:
            print(f"missing_link → {missing}")
        topk, bottomk = report["coverage"].get("topk", []), report["coverage"].get("bottomk", [])
        if topk:
            print(f"  topK: {', '.join([f'{n:.2e}:{k}' for n, k in topk[:3]])}")
        if bottomk:
            print(f"  smallK: {', '.join([f'{n:.2e}:{k}' for n, k in bottomk[:3]])}")
        for w in warn_msgs:
            print(w)
        li = report.get("loss_inputs", {})
        keys = ["encoder_out", "dit_out", "ar_out", "v_pred", "y0_pred_9d", "R_pred", "R_pred_anchor", "t_pred", "t_pred_anchor"]
        flags = [f"{k}:{'Y' if li.get(k, {}).get('has_grad', False) else 'N'}" for k in keys if k in li]
        if flags:
            print("  loss_inputs:", ", ".join(flags))
        has_R = li.get("R_pred", {}).get("has_grad", False) or li.get("R_pred_anchor", {}).get("has_grad", False)
        has_t = li.get("t_pred", {}).get("has_grad", False) or li.get("t_pred_anchor", {}).get("has_grad", False)
        enc_ok = li.get("encoder_out", {}).get("has_grad", False)
        tmp_ok = li.get(temporal_key, {}).get("has_grad", False)
        head_ok = li.get("y0_pred_9d", {}).get("has_grad", False) or (has_R and has_t)
        ok = enc_ok and tmp_ok and head_ok
        if ok:
            print("gradflow OK: path connected (encoder/DiT/head and/or R,t)")
        else:
            print(f"gradflow FAIL: missing grads (first missing: {self._find_missing_link(li) or '?'})")
        jpath = os.path.join(self.cfg.train.out_dir, "gradflow", f"step_{step_idx:04d}.json")
        with open(jpath, "w") as f:
            json.dump(report, f, indent=2)
        self.step_count += 1


def compute_grad_weight_norms_if_enabled(model: torch.nn.Module, track: bool) -> tuple[float, float]:
    if not track:
        return 0.0, 0.0
    sqg = sum(float((p.grad.detach() ** 2).sum().item()) for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    sqw = sum(float((p.data.detach() ** 2).sum().item()) for p in model.parameters() if torch.isfinite(p.data).all())
    return sqg**0.5, sqw**0.5


def log_training_anomaly_if_any(loss: torch.Tensor, grad_norm: float, grad_explode_thr: float, track: bool, logger_print: Callable[[str], None]) -> None:
    if not torch.isfinite(loss) or (track and grad_norm > grad_explode_thr):
        logger_print(f"[red]training anomaly[/red] • loss_finite={torch.isfinite(loss).item()} • grad_norm={grad_norm:.2e}")


def grad_probe_if_enabled(model: torch.nn.Module, batch_in: dict, cfg: Any, gaudit: GradFlowAudit, opt: torch.optim.Optimizer, global_step: int, B: int) -> bool:
    from ..geom import se3_ops
    from ..temporal.diffusion import q_sample

    probe_mode = getattr(cfg.train, "grad_probe", False)
    if not (probe_mode and gaudit.is_active()):
        return False
    H = int(getattr(getattr(model, "cfg", cfg), "H", cfg.data.H))
    tgt = batch_in["target_future"]
    if isinstance(tgt, list):
        tgt = torch.from_numpy(np.stack(tgt, axis=0)).float()
    elif hasattr(tgt, "__array__"):
        tgt = torch.as_tensor(tgt).float()
    y0 = tgt.view(B, H * 9).to(next(model.parameters()).device)
    t_total = int(getattr(getattr(getattr(model, "cfg", cfg), "temporal", {}), "T", cfg.data.H))
    t = torch.randint(low=0, high=t_total, size=(B,), device=y0.device)
    y_t = q_sample(y0, t, model.sqrt_alphas_cumprod, model.sqrt_one_minus_alphas_cumprod, torch.randn_like(y0))
    T_cam_anchor_obj = batch_in.get("T_cam_anchor_obj")
    cond_embed = batch_in.get("cond_embed") or model.encode(batch_in["scene_pcd"], batch_in["context_vec"], T_cam_anchor_obj=T_cam_anchor_obj)
    alpha_bar = model.alphas_cumprod[t].unsqueeze(-1)
    v_pred = model.forward(y_t, t, cond_embed)
    y0_pred = (torch.sqrt(alpha_bar) * y_t - torch.sqrt(1.0 - alpha_bar) * v_pred).view(B, H, 9)
    if getattr(model, "_grad_audit", False):
        for name, ten in [("y0_pred_9d", y0_pred), ("v_pred", v_pred), ("encoder_out", cond_embed)]:
            if isinstance(ten, torch.Tensor) and ten.requires_grad:
                ten.retain_grad()
                model._ga_tensors[name] = ten
    mode = str(probe_mode)
    if mode == "se3":
        t_pred, r6_pred = y0_pred[..., :3], y0_pred[..., 3:9]
        R_pred = se3_ops.rot6d_to_matrix(r6_pred.reshape(-1, 6)).view(B, H, 3, 3)
        if getattr(model, "_grad_audit", False):
            assert R_pred.requires_grad and R_pred.grad_fn is not None, "gradflow: R_pred not connected"
            assert t_pred.requires_grad and t_pred.grad_fn is not None, "gradflow: t_pred not connected"
            for name, ten in [("R_pred", R_pred), ("t_pred", t_pred)]:
                ten.retain_grad()
                model._ga_tensors[name] = ten
        y0_gt = y0.view(B, H, 9)
        R_gt = se3_ops.rot6d_to_matrix(y0_gt[..., 3:9].detach().reshape(-1, 6)).view(B, H, 3, 3)
        geod_rad = se3_ops.geodesic_distance_deg(R_pred.view(-1, 3, 3), R_gt.view(-1, 3, 3)).view(B, H) / 57.29577951308232
        probe_loss = geod_rad.mean() + torch.nn.functional.mse_loss(t_pred, y0_gt[..., :3].detach())
    else:
        probe_loss = torch.nn.functional.mse_loss(y0_pred, torch.zeros_like(y0_pred))
    probe_loss.backward()
    clip_norm = float(getattr(cfg.train, "grad_clip_norm", 0.0))
    if clip_norm > 0.0:
        pre = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        gaudit.record_preclip(float(pre))
        gaudit.record_postclip(gaudit._sum_param_grad_norms())
    else:
        cur = gaudit._sum_param_grad_norms()
        gaudit.record_preclip(cur)
        gaudit.record_postclip(cur)
    gaudit.build_and_emit_report(global_step + 1, opt.param_groups[0]["lr"])
    li = model.get_grad_audit_tensors() if hasattr(model, "get_grad_audit_tensors") else {}
    has_R = isinstance(li.get("R_pred"), torch.Tensor) and li["R_pred"].grad is not None
    has_t = isinstance(li.get("t_pred"), torch.Tensor) and li["t_pred"].grad is not None
    if mode == "se3":
        print("gradflow OK: se3 path connected (encoder/DiT/head/R/t)" if has_R and has_t else "gradflow FAIL: se3 path missing grads (check rot6d->R or t path)")
    else:
        print("gradflow OK: v path connected (encoder/DiT/head)")
    return True


def _batch_size_from(batch: dict) -> int:
    tf = batch["target_future"]
    return len(tf) if isinstance(tf, (list, tuple)) else tf.shape[0]


def _scheduled_sampling_prob(cfg: DictConfig, global_step: int) -> float:
    ss_cfg = getattr(cfg.train, "ss", {})
    s0, s1 = float(getattr(ss_cfg, "start_p", 0.0)), float(getattr(ss_cfg, "end_p", 0.2))
    T = int(getattr(ss_cfg, "anneal_steps", 0))
    if T <= 0 or s0 == s1:
        return s1
    return s0 + min(1.0, max(0.0, global_step / T)) * (s1 - s0)


_STEP_PAT = re.compile(r"^step_(\d+)\.pt$")


def _find_latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    step_ckpts = [(int(m.group(1)), os.path.join(ckpt_dir, f)) for f in os.listdir(ckpt_dir) if (m := _STEP_PAT.match(f))]
    if step_ckpts:
        return max(step_ckpts, key=lambda x: x[0])[1]
    for name in ("best.pt", "last.pt"):
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p
    pt_files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    return max(pt_files, key=os.path.getmtime) if pt_files else None


def _resolve_resume_checkpoint_path(cfg: DictConfig) -> str | None:
    resume_flag = getattr(cfg.train, "resume", False)
    explicit_path = getattr(cfg.train, "resume_ckpt", None)
    ckpt_dir = os.path.join(cfg.train.out_dir, "checkpoints")
    if isinstance(explicit_path, str) and explicit_path:
        p = os.path.expanduser(os.path.expandvars(explicit_path))
        return p if os.path.exists(p) else None
    if isinstance(resume_flag, str):
        val = resume_flag.strip().lower()
        if val.endswith(".pt") or "/" in val or "\\" in val:
            p = os.path.expanduser(os.path.expandvars(resume_flag))
            return p if os.path.exists(p) else None
        if val in ("latest", "auto", "true"):
            return _find_latest_checkpoint(ckpt_dir)
        if val in ("best", "last"):
            p = os.path.join(ckpt_dir, f"{val}.pt")
            return p if os.path.exists(p) else None
    return _find_latest_checkpoint(ckpt_dir) if resume_flag else None


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


class _EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in _unwrap(model).named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in _unwrap(model).named_parameters():
            if name in self.shadow and param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)

    @torch.no_grad()
    def store(self, model: torch.nn.Module) -> None:
        self.backup = {n: p.detach().clone() for n, p in _unwrap(model).named_parameters() if n in self.shadow}

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        for name, param in _unwrap(model).named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if not self.backup:
            return
        for name, param in _unwrap(model).named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].to(device=param.device, dtype=param.dtype))
        self.backup = {}


def _ensure_optimizer_tracks_params(optimizer: torch.optim.Optimizer | None, params: list[torch.nn.Parameter]) -> None:
    if not optimizer or not params:
        return
    existing = {id(p) for group in optimizer.param_groups for p in group.get("params", [])}
    new_params = [p for p in params if id(p) not in existing]
    if new_params:
        if optimizer.param_groups:
            optimizer.param_groups[0]["params"].extend(new_params)
        else:
            optimizer.add_param_group({"params": new_params})
