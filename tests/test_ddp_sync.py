"""Minimal test to verify DDP gradient synchronization."""
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def test_ddp_sync():
    dist.init_process_group("nccl", init_method="env://")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    model = torch.nn.Linear(10, 10).to(device)
    ddp_model = DDP(model, device_ids=[local_rank])

    torch.manual_seed(rank)  # Different data per rank
    x = torch.randn(4, 10, device=device)

    # Test 1: Through DDP wrapper (should sync)
    ddp_model.zero_grad()
    out = ddp_model(x)
    loss = out.sum()
    loss.backward()
    grad_ddp = model.weight.grad.clone()
    grad_avg = grad_ddp.clone()
    dist.all_reduce(grad_avg, op=dist.ReduceOp.AVG)
    diff_ddp = (grad_ddp - grad_avg).abs().max().item()

    # Test 2: Bypassing DDP (should NOT sync)
    ddp_model.zero_grad()
    out = ddp_model.module(x)  # Bypass!
    loss = out.sum()
    loss.backward()
    grad_bypass = model.weight.grad.clone()
    grad_avg2 = grad_bypass.clone()
    dist.all_reduce(grad_avg2, op=dist.ReduceOp.AVG)
    diff_bypass = (grad_bypass - grad_avg2).abs().max().item()

    if rank == 0:
        print(f"Through DDP: diff={diff_ddp:.2e} (should be ~0)")
        print(f"Bypass DDP:  diff={diff_bypass:.2e} (should be >0)")
        if diff_ddp < 1e-6 and diff_bypass > 1e-6:
            print("✓ PASS: DDP sync works as expected")
        else:
            print("✗ FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    test_ddp_sync()
