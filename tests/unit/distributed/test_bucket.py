from __future__ import annotations

import torch
import torch.nn as nn

from nano_megatron.distributed.bucket import GradBucket, build_buckets


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, torch.Tensor]] = []

    def all_reduce(self, tensor, *, group=None, op="sum", async_op=False):
        self.calls.append(("all_reduce", tensor.detach().clone()))
        # Simulate 2-rank sum: pretend peer contributed the same tensor → *2
        tensor.mul_(2)
        return tensor


def test_build_buckets_reverse_order_and_cap():
    m = nn.Sequential(
        nn.Linear(4, 4, bias=False),  # 16 elems
        nn.Linear(4, 4, bias=False),  # 16
        nn.Linear(4, 4, bias=False),  # 16
    )
    # cap just under 2*16*4 = 128 bytes so each bucket holds at most 1 param of 64B...
    # 16 float32 = 64 bytes. cap_mb such that cap_bytes = 100 → one param per bucket
    buckets = build_buckets(m, bucket_cap_mb=100 / (1024 * 1024))
    flat_params = [p for b in buckets for p in b.params]
    expected = list(reversed(list(m.parameters())))
    assert flat_params == expected
    assert all(len(b.params) == 1 for b in buckets)


def test_build_buckets_packs_small_params_together():
    m = nn.Sequential(
        nn.Linear(2, 2, bias=False),  # 4 * 4 = 16 bytes
        nn.Linear(2, 2, bias=False),
    )
    # large cap → one bucket with both, reverse order
    buckets = build_buckets(m, bucket_cap_mb=25.0)
    assert len(buckets) == 1
    assert buckets[0].params == list(reversed(list(m.parameters())))


def test_build_buckets_splits_mixed_dtypes():
    p_f = nn.Parameter(torch.zeros(4, dtype=torch.float32))
    p_d = nn.Parameter(torch.zeros(4, dtype=torch.float64))
    m = nn.Module()
    m.a = p_f
    m.b = p_d
    buckets = build_buckets(m, bucket_cap_mb=25.0)
    assert len(buckets) == 2
    assert all(len(b.params) == 1 for b in buckets)


def test_build_buckets_rejects_mixed_devices():
    p_cpu = nn.Parameter(torch.zeros(4))
    p_meta = nn.Parameter(torch.zeros(4, device="meta"))
    m = nn.Module()
    m.a = p_cpu
    m.b = p_meta
    try:
        build_buckets(m, bucket_cap_mb=25.0)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "device" in str(e).lower()


def test_mark_ready_and_sync_mean():
    p0 = nn.Parameter(torch.zeros(2))
    p1 = nn.Parameter(torch.zeros(2))
    p0.grad = torch.tensor([1.0, 2.0])
    p1.grad = torch.tensor([3.0, 4.0])
    bucket = GradBucket([p0, p1])
    backend = _FakeBackend()
    assert bucket.mark_ready(p0) is False
    assert bucket.mark_ready(p0) is False  # idempotent
    assert bucket.mark_ready(p1) is True
    bucket.sync(backend, group=None, dp_size=2)
    # fake all_reduce doubled, then /2 → original values
    assert torch.equal(p0.grad, torch.tensor([1.0, 2.0]))
    assert torch.equal(p1.grad, torch.tensor([3.0, 4.0]))
    assert bucket.coalesced is True
    assert len(backend.calls) == 1


def test_sync_dp_size_one_noop():
    p = nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([1.0, 2.0])
    bucket = GradBucket([p])
    backend = _FakeBackend()
    bucket.mark_ready(p)
    bucket.sync(backend, group=None, dp_size=1)
    assert len(backend.calls) == 0
    assert torch.equal(p.grad, torch.tensor([1.0, 2.0]))
    assert bucket.coalesced is True


def test_sync_raises_if_grad_missing():
    p = nn.Parameter(torch.zeros(2))
    bucket = GradBucket([p])
    backend = _FakeBackend()
    try:
        bucket.sync(backend, group=None, dp_size=2)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "grad" in str(e).lower()


def test_reset_clears_state():
    p = nn.Parameter(torch.zeros(2))
    p.grad = torch.ones(2)
    bucket = GradBucket([p])
    bucket.mark_ready(p)
    bucket.sync(_FakeBackend(), group=None, dp_size=1)
    bucket.reset()
    assert bucket.coalesced is False
    assert bucket.mark_ready(p) is True


def test_sync_non_contiguous_grad():
    """Non-contiguous grad must still be updated in-place after sync.

    A _DoublingBackend adds 10 to the flat tensor (no division trick).
    If the unflatten silently drops writes (non-contiguous reshape copy),
    p.grad keeps the original values and the assertion fails.
    """

    class _DoublingBackend:
        """Adds a constant offset so unflatten correctness is visible."""
        def all_reduce(self, tensor, *, group=None, op="sum", async_op=False):
            tensor.add_(10)
            return tensor

    p = nn.Parameter(torch.zeros(4, 2))
    # Non-contiguous 4×2 view via slicing a larger tensor (stride > dim-1 size)
    p.grad = torch.arange(1.0, 17.0).reshape(8, 2)[::2]  # 4×2, non-contiguous
    assert p.grad.shape == (4, 2)
    assert not p.grad.is_contiguous()
    bucket = GradBucket([p])
    bucket.sync(_DoublingBackend(), group=None, dp_size=2)
    # all_reduce added 10 to flat, then /2 → (grad + 10) / 2 per element
    raw = torch.arange(1.0, 17.0).reshape(8, 2)[::2]
    expected = (raw + 10) / 2
    assert torch.equal(p.grad, expected)


def test_reset_clears_flat_buffer():
    p = nn.Parameter(torch.zeros(2))
    p.grad = torch.ones(2)
    bucket = GradBucket([p])
    bucket.sync(_FakeBackend(), group=None, dp_size=2)
    bucket.reset()
    assert bucket._flat is None
