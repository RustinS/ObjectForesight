from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from rich.box import SIMPLE
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config_adapter import compact_view

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_IS_TTY = bool(getattr(sys.stdout, "isatty", lambda: False)())
if _IS_TTY:
    console = Console()
else:
    _width = int(os.environ.get("LOG_CONSOLE_WIDTH", "512"))
    console = Console(force_terminal=False, no_color=True, width=_width, soft_wrap=True, highlight=False)


def rprint(*args, stack_level: int = 1, print_location: bool = False, no_extra: bool = False, **kwargs) -> None:
    """Rich print with timestamp and optional location."""
    if os.environ.get("WORLD_SIZE", "1") != "1" and os.environ.get("RANK", "0") != "0":
        return

    frame = inspect.currentframe()
    for _ in range(max(int(stack_level), 1)):
        if frame is None:
            break
        frame = frame.f_back

    path_disp, lineno = "?", 0
    if frame is not None and print_location:
        info = inspect.getframeinfo(frame)
        path_disp = os.path.relpath(info.filename, _ROOT)
        lineno = info.lineno
    prefix = f"{path_disp}:{lineno}".ljust(25)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if no_extra:
        prefix = ""
    elif print_location:
        prefix = f"[dim]{prefix}[/dim] [[bold green]{ts}[/bold green]] "
    else:
        prefix = f"[[bold green]{ts}[/bold green]] "

    if args and all(isinstance(a, (str, bytes)) for a in args):
        msg = " ".join(a.decode() if isinstance(a, bytes) else a for a in args)
        console.print(f"{prefix}{msg}", **kwargs)
    else:
        console.print(f"{prefix}")
        for obj in args:
            console.print(obj, **kwargs)


@dataclass
class StepAverager:
    ema: float = 0.0
    beta: float = 0.9
    initialized: bool = False

    def update(self, x: float) -> float:
        if not self.initialized:
            self.ema = x
            self.initialized = True
        else:
            self.ema = self.beta * self.ema + (1 - self.beta) * x
        return self.ema


class Throughput:
    """Sliding throughput meter in samples/sec."""

    def __init__(self, window_steps: int = 200):
        self.window = window_steps
        self.buf = []

    def update(self, n: int, dt: float) -> float:
        self.buf.append((n, dt))
        if len(self.buf) > self.window:
            self.buf.pop(0)
        tot_n = sum(n for n, _ in self.buf)
        tot_t = sum(dt for _, dt in self.buf)
        return tot_n / max(tot_t, 1e-8)


def print_config_summary(cfg_dict: Dict[str, Any]):
    flat = compact_view(cfg_dict)
    flat.update(
        {
            "context_len": getattr(getattr(cfg_dict, "data", {}), "context_len", None),
            "batch_size": cfg_dict.train.batch_size,
            "lr": float(cfg_dict.train.lr),
            "lr_schedule": getattr(cfg_dict.train, "lr_schedule", None),
            "weight_decay": float(getattr(cfg_dict.train, "weight_decay", 0.0)),
            "grad_clip_norm": float(getattr(cfg_dict.train, "grad_clip_norm", 0.0)),
            "amp": bool(getattr(cfg_dict.train, "amp", False)),
            "epochs": cfg_dict.train.epochs,
            "encoder": cfg_dict.model.encoder._target_.split(".")[-1] if "_target_" in cfg_dict.model.encoder else "encoder",
            "mesh": getattr(cfg_dict.model, "mesh", "none"),
            "print_every": cfg_dict.log.print_every,
        }
    )
    rprint(Panel.fit("[bold]Run Configuration[/bold]", border_style="cyan"), stack_level=2)
    tbl = Table(box=SIMPLE, show_lines=False)
    tbl.add_column("Key", style="bold cyan")
    tbl.add_column("Value")
    for k, v in flat.items():
        if not isinstance(v, dict):
            tbl.add_row(str(k), str(v))
    rprint(tbl, stack_level=2, no_extra=True)


def print_model_summary(encoder_name: str, temporal_name: str, params_m: float, horizon: int | None = None, context_len: int | None = None):
    extra_bits = []
    if horizon is not None:
        extra_bits.append(f"H={int(horizon)}")
    if context_len is not None:
        extra_bits.append(f"P={int(context_len)}")
    extra_str = " • " + " ".join(extra_bits) if extra_bits else ""
    rprint(
        Panel.fit(
            f"[bold]Model[/bold]: encoder=[green]{encoder_name}[/green], temporal=[green]{temporal_name}[/green], params≈[yellow]{params_m:.2f}M[/yellow]{extra_str}",
            border_style="green",
        ),
        stack_level=2,
        no_extra=True,
    )


def step_line(
    epoch: int,
    step: int,
    total_steps: int,
    loss: float,
    loss_ema: float,
    lr: float,
    grad_norm: Optional[float] = None,
    tput: Optional[float] = None,
    extra: Optional[Dict[str, float]] = None,
    groups: Optional[list] = None,
    prec: int = 2,
):
    """Print a training step line with grouped metrics.

    Args:
        groups: List of dicts, each dict is a group of metrics separated by |.
                Each dict maps metric_name -> (value, unit) or metric_name -> value.
        extra: Legacy flat dict of metrics (used if groups is None).
    """
    # Fixed widths for alignment
    step_w = len(str(total_steps))
    step_bits = [f"[bold]ep[/bold]={epoch:>3}", f"[bold]it[/bold]={step:>{step_w}}/{total_steps}"]

    # Loss group with fixed widths
    loss_bits = [f"[bold]loss[/bold]={loss:>6.{prec}f}", f"[bold]ema[/bold]={loss_ema:>6.{prec}f}", f"[bold]lr[/bold]={lr:.2e}"]
    if grad_norm is not None:
        loss_bits.append(f"[bold]gn[/bold]={grad_norm:>9.2e}")

    all_groups = [" • ".join(step_bits), " • ".join(loss_bits)]

    # Width specs for known metrics: (total_width, decimals)
    metric_widths = {"rot": (6, prec), "t": (5, 3), "main": (5, prec), "aux": (5, prec), "ss": (4, 2), "vel": (5, 3), "acc": (5, 3), "disp": (5, 3)}

    if groups:
        for grp in groups:
            if not grp:
                continue
            bits = []
            for k, v in grp.items():
                w, d = metric_widths.get(k, (6, prec))
                if isinstance(v, tuple):
                    val, unit = v
                    bits.append(f"[bold]{k}[/bold]={val:>{w}.{d}f}{unit}")
                else:
                    bits.append(f"[bold]{k}[/bold]={v:>{w}.{d}f}")
            if bits:
                all_groups.append(" • ".join(bits))
    elif extra:
        bits = [f"[bold]{k}[/bold]={v:>6.{prec}f}" for k, v in extra.items()]
        all_groups.append(" • ".join(bits))

    rprint(" [dim]|[/dim] ".join(all_groups), stack_level=2)


def epoch_table(title: str, metrics: Dict[str, float], prec: int = 4):
    tbl = Table(title=title, box=SIMPLE)
    tbl.add_column("Metric", style="bold magenta")
    tbl.add_column("Value", justify="right")
    for k, v in metrics.items():
        tbl.add_row(k, f"{v:.{prec}f}")
    rprint(tbl, stack_level=2)


def checkpoint_line(path: str, note: str = "saved"):
    rprint(f"[dim]ckpt[/dim] → [bold]{path}[/bold] ({note})", stack_level=2)
