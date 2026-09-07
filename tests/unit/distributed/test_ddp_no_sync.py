from __future__ import annotations

import torch
import torch.nn as nn

from nano_megatron.distributed.ddp import DistributedDataParallel


class _FakeBackend:
    def __init__(self) -> None:
        self.all_reduce_calls = 0

    def all_reduce(self, tensor, *, group=None, op="sum", async_op=False):
        self.all_reduce_calls += 1
        return tensor

    def broadcast(self, tensor, src, *, group=None):
        return tensor


def _make_ctx(backend, dp_size=2):
    from types import SimpleNamespace
    return SimpleNamespace(
        rank=0,
        data_parallel_rank=0,
        data_parallel_size=dp_size,
        data_parallel_group="dp",
        context_parallel_rank=0,
        context_parallel_size=1,
        data_context_parallel_group="dp_cp",
        data_context_parallel_src_rank=0,
        backend=backend,
    )


def test_no_sync_defers_all_reduce_until_finish():
    backend = _FakeBackend()
    model = nn.Linear(4, 4, bias=False)
    ddp = DistributedDataParallel(model, _make_ctx(backend), bucket_cap_mb=25.0)
    baseline = backend.all_reduce_calls
    x = torch.randn(2, 4)
    with ddp.no_sync():
        ddp(x).sum().backward()
        ddp(x).sum().backward()
    assert backend.all_reduce_calls == baseline
    assert model.weight.grad is not None
    ddp.finish_grad_sync()
    assert backend.all_reduce_calls > baseline


def test_no_sync_nested_restores_flag():
    """Nested no_sync must restore the previous sync flag, not always True."""
    backend = _FakeBackend()
    model = nn.Linear(4, 4, bias=False)
    ddp = DistributedDataParallel(model, _make_ctx(backend), bucket_cap_mb=25.0)

    assert ddp._require_backward_grad_sync is True
    with ddp.no_sync():
        assert ddp._require_backward_grad_sync is False
        with ddp.no_sync():
            assert ddp._require_backward_grad_sync is False
        # After inner exits, still in outer no_sync
        assert ddp._require_backward_grad_sync is False
    assert ddp._require_backward_grad_sync is True


def test_sync_after_no_sync_still_works():
    """After no_sync context exits, normal sync must work again."""
    backend = _FakeBackend()
    model = nn.Linear(4, 4, bias=False)
    ddp = DistributedDataParallel(model, _make_ctx(backend), bucket_cap_mb=25.0)
    x = torch.randn(2, 4)

    # First: no_sync run
    with ddp.no_sync():
        ddp(x).sum().backward()
    ddp.finish_grad_sync()

    calls_after_no_sync = backend.all_reduce_calls

    # Second: normal sync run
    ddp(x).sum().backward()
    ddp.finish_grad_sync()
    assert backend.all_reduce_calls > calls_after_no_sync


def test_overlap_does_not_reduce_when_forward_has_no_backward():
    backend = _FakeBackend()
    model = nn.Linear(4, 4, bias=False)
    ddp = DistributedDataParallel(
        model,
        _make_ctx(backend),
        bucket_cap_mb=25.0,
        overlap_grad_reduce=True,
    )
    baseline = backend.all_reduce_calls
    ddp(torch.randn(2, 4))
    ddp.finish_grad_sync()
    assert backend.all_reduce_calls == baseline


def test_overlap_no_sync_launches_at_finish_boundary():
    backend = _FakeBackend()
    model = nn.Linear(4, 4, bias=False)
    ddp = DistributedDataParallel(
        model,
        _make_ctx(backend),
        bucket_cap_mb=25.0,
        overlap_grad_reduce=True,
    )
    baseline = backend.all_reduce_calls
    with ddp.no_sync():
        ddp(torch.randn(2, 4)).sum().backward()
    assert backend.all_reduce_calls == baseline
    ddp.finish_grad_sync()
    assert backend.all_reduce_calls == baseline + 1


def test_gradient_backend_is_independent_from_parameter_broadcast_backend():
    collective_backend = _FakeBackend()
    gradient_backend = _FakeBackend()
    model = nn.Linear(4, 4, bias=False)
    ddp = DistributedDataParallel(
        model,
        _make_ctx(collective_backend),
        grad_sync_backend=gradient_backend,
    )

    # Constructor parameter synchronization uses broadcast, not a metadata reduction.
    assert collective_backend.all_reduce_calls == 0
    assert gradient_backend.all_reduce_calls == 0

    ddp(torch.randn(2, 4)).sum().backward()
    ddp.finish_grad_sync()
    assert collective_backend.all_reduce_calls == 0
    assert gradient_backend.all_reduce_calls == 1
