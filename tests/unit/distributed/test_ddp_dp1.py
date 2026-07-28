from __future__ import annotations

import torch
import torch.nn as nn

from nano_megatron.distributed import DistributedDataParallel
from nano_megatron.parallel import (
    ParallelConfig,
    destroy_parallel,
    initialize_parallel,
    is_parallel_initialized,
)


def _init_dp1(monkeypatch, port: str):
    import torch.distributed as dist

    if is_parallel_initialized():
        destroy_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", port)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    return initialize_parallel(ParallelConfig(), dist_backend="gloo")


def test_ddp_dp1_grads_match_bare(monkeypatch):
    ctx = _init_dp1(monkeypatch, "29701")
    torch.manual_seed(0)
    bare = nn.Linear(4, 3, bias=True)
    torch.manual_seed(0)
    wrapped_mod = nn.Linear(4, 3, bias=True)
    ddp = DistributedDataParallel(wrapped_mod, ctx)

    x = torch.randn(2, 4)
    bare_out = bare(x)
    ddp_out = ddp(x)
    assert torch.equal(bare_out, ddp_out)

    bare_out.sum().backward()
    ddp_out.sum().backward()
    ddp.finish_grad_sync()

    for pb, pw in zip(bare.parameters(), ddp.module.parameters()):
        assert pb.grad is not None and pw.grad is not None
        assert torch.equal(pb.grad, pw.grad)

    destroy_parallel()


def test_finish_grad_sync_idempotent_after_backward(monkeypatch):
    ctx = _init_dp1(monkeypatch, "29702")
    torch.manual_seed(1)
    ddp = DistributedDataParallel(nn.Linear(4, 3, bias=True), ctx)

    x = torch.randn(2, 4)
    ddp(x).sum().backward()
    ddp.finish_grad_sync()

    grads_after_first = {
        id(p): p.grad.detach().clone() for p in ddp.module.parameters()
    }
    ddp.finish_grad_sync()  # second call must no-op

    for p in ddp.module.parameters():
        assert p.grad is not None
        assert torch.equal(p.grad, grads_after_first[id(p)])

    destroy_parallel()


def test_finish_grad_sync_noop_without_backward(monkeypatch):
    ctx = _init_dp1(monkeypatch, "29703")
    ddp = DistributedDataParallel(nn.Linear(4, 3, bias=True), ctx)

    # No forward/backward: must not raise.
    ddp.finish_grad_sync()
    for p in ddp.module.parameters():
        assert p.grad is None

    destroy_parallel()
