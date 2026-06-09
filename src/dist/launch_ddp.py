from __future__ import annotations

import argparse
import os
import runpy
import sys
from typing import List

from .distrib import (
    allreduce_health_check,
    barrier,
    destroy_pg,
    get_rank,
    get_world_size,
    init_dist_from_slurm,
    infer_master_addr,
    is_rank0,
    register_sigterm_handler,
)


def _parse_cli(argv: List[str]) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Slurm + PyTorch DDP launcher")
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp16", "float32"),
        default=os.environ.get("PRECISION", "bf16"),
        help="Numerical precision to prefer (propagated via env PRECISION)",
    )
    parser.add_argument(
        "--fsdp.enable",
        dest="fsdp_enable",
        type=int,
        choices=(0, 1),
        default=int(os.environ.get("FSDP_ENABLE", "0")),
        help="Enable FSDP wrapping in user code (exported via FSDP_ENABLE)",
    )
    parser.add_argument("--log-dir", type=str, default=os.environ.get("LOG_DIR"))
    parser.add_argument("--ckpt-dir", type=str, default=os.environ.get("CKPT_DIR"))
    parser.add_argument(
        "--eval",
        action="store_true",
        default=os.environ.get("EVAL", "0") == "1",
        help="Run eval_main instead of train_main",
    )

    # Everything after "--" is passed through to training
    args, passthrough = parser.parse_known_args(argv)

    # Support enabling FSDP via passthrough (e.g., from submit --extra-args "--fsdp.enable 1")
    # Strip it from passthrough to avoid confusing user training CLI.
    cleaned: List[str] = []
    i = 0
    # argparse.parse_known_args keeps the literal "--" separator in the unknown
    # list. If forwarded to Hydra it makes argparse treat every following token
    # (e.g. "--config-name") as an override, triggering a LexerNoViableAltException.
    if i < len(passthrough) and passthrough[i] == "--":
        i += 1
    while i < len(passthrough):
        token = passthrough[i]
        if token == "--fsdp.enable":
            if i + 1 < len(passthrough):
                try:
                    args.fsdp_enable = int(passthrough[i + 1])
                except Exception:
                    args.fsdp_enable = 1
                i += 2
                continue
            else:
                # dangling flag; treat as enable=1
                args.fsdp_enable = 1
                i += 1
                continue
        cleaned.append(token)
        i += 1

    return args, cleaned


def _print_topology() -> None:
    if is_rank0():
        master_addr = os.environ.get("MASTER_ADDR", infer_master_addr())
        master_port = os.environ.get("MASTER_PORT", "29500")
        print(
            f"[dist] MASTER_ADDR={master_addr} MASTER_PORT={master_port} "
            f"world={get_world_size()}"
        )
    print(
        f"[rank] global={get_rank()} world={get_world_size()} local={os.environ.get('LOCAL_RANK','0')}"
    )


def _run_user_module(passthrough: List[str], is_eval: bool) -> None:
    # Set argv so Hydra/argparse see only user overrides (no stray tokens)
    # argv[0] is a program name; subsequent tokens are actual CLI args
    module = "src.eval_main" if is_eval else "src.train_main"
    sys.argv = [module, *passthrough]
    runpy.run_module(module, run_name="__main__")


def main(argv: List[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args, passthrough = _parse_cli(argv)

    # Propagate environment for user code
    os.environ["PRECISION"] = args.precision
    os.environ["FSDP_ENABLE"] = str(int(args.fsdp_enable))
    if args.log_dir:
        os.environ["LOG_DIR"] = args.log_dir
    if args.ckpt_dir:
        os.environ["CKPT_DIR"] = args.ckpt_dir

    # Initialize distributed from Slurm; fall back to local single process
    ddp = init_dist_from_slurm(backend="nccl")
    if not ddp:
        print("[warn] Slurm env not detected; running single-process locally.")
        _run_user_module(passthrough, is_eval=args.eval)
        return 0

    register_sigterm_handler(on_terminate=None)
    _print_topology()

    # Quick collective sanity
    if not allreduce_health_check():
        if is_rank0():
            print("[fatal] Health check failed; aborting.")
        destroy_pg()
        return 2

    try:
        _run_user_module(passthrough, is_eval=args.eval)
    finally:
        barrier()
        destroy_pg()
    return 0


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())

