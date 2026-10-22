from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nano_megatron.distributed.bucket import build_buckets
from nano_megatron.distributed.ddp import DistributedDataParallel


def test_disabled_bucketing_ignores_size_but_preserves_parameter_order():
    model = nn.Sequential(*(nn.Linear(4, 4, bias=False) for _ in range(3)))
    model[1].weight.requires_grad_(False)
    cap = 1 / (1024 * 1024)
    assert len(build_buckets(model, cap)) == 2
    buckets = build_buckets(model, cap, disable_bucketing=True)
    assert len(buckets) == 1
    assert buckets[0].params == [model[2].weight, model[0].weight]
    assert build_buckets(nn.Module(), cap, disable_bucketing=True) == []


def test_disabled_bucketing_preserves_dtype_boundaries():
    model = nn.ParameterList([
        nn.Parameter(torch.zeros(4)),
        nn.Parameter(torch.zeros(4, dtype=torch.float64)),
        nn.Parameter(torch.zeros(4)),
    ])
    buckets = build_buckets(model, disable_bucketing=True)
    assert len(buckets) == 3
    assert [p for b in buckets for p in b.params] == list(reversed(list(model)))
    assert all(len({p.dtype for p in b.params}) == 1 for b in buckets)


def test_disabled_bucketing_still_rejects_invalid_cap_and_mixed_devices():
    with pytest.raises(ValueError, match="bucket_cap_mb"):
        build_buckets(nn.Linear(4, 4), 0, disable_bucketing=True)
    model = nn.ParameterList([
        nn.Parameter(torch.zeros(4)),
        nn.Parameter(torch.zeros(4, device="meta")),
    ])
    with pytest.raises(RuntimeError, match="multiple devices"):
        build_buckets(model, disable_bucketing=True)


class _Work:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class _Backend:
    def __init__(self):
        self.calls = 0
        self.works = []

    def broadcast(self, tensor, src, *, group=None):
        return tensor

    def all_reduce(self, tensor, *, group=None, op="sum", async_op=False):
        self.calls += 1
        tensor.mul_(2)  # Identical peer contribution: mean recovers local grads.
        if async_op:
            work = _Work()
            self.works.append(work)
            return work
        return tensor


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("defer_last_backward", [False, True])
def test_disabled_bucketing_accumulation_wait_and_three_updates(overlap, defer_last_backward):
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 2))
    reference = copy.deepcopy(model)
    backend = _Backend()
    ctx = SimpleNamespace(
        rank=0, backend=backend, data_parallel_size=2,
        context_parallel_size=1, data_context_parallel_group="dp_cp",
        data_context_parallel_src_rank=0,
    )
    ddp = DistributedDataParallel(
        model, ctx, bucket_cap_mb=1 / (1024 * 1024),
        disable_bucketing=True, overlap_grad_reduce=overlap,
    )
    assert len(ddp._buckets) == 1
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    original_buffer = ddp._buckets[0].flat_buffer
    for step in range(3):
        ddp.zero_grad()
        ref_opt.zero_grad(set_to_none=True)
        for micro in range(2):
            x = torch.randn(2, 4)
            if micro == 0 or defer_last_backward:
                with ddp.no_sync():
                    ddp(x).square().sum().backward()
            else:
                ddp(x).square().sum().backward()
            reference(x).square().sum().backward()
            if micro == 0 or defer_last_backward:
                assert backend.calls == step
        if overlap and not defer_last_backward:
            assert backend.calls == step + 1
            assert not backend.works[-1].waited
        ddp.finish_grad_sync()
        ddp.finish_grad_sync()  # Consumer boundary is idempotent.
        assert backend.calls == step + 1
        if overlap:
            assert backend.works[-1].waited
            assert ddp._buckets[0].flat_buffer is original_buffer
        for p, ref in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, ref.grad, rtol=0, atol=0)
        opt.step()
        ref_opt.step()
        for p, ref in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, ref, rtol=0, atol=0)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(opt.state[p][key], ref_opt.state[ref][key], rtol=0, atol=0)
