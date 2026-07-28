from __future__ import annotations

import pytest
import torch

from nano_megatron.parallel import (
    ParallelConfig,
    destroy_parallel,
    initialize_parallel,
    is_parallel_initialized,
)
from nano_megatron.parallel.context_parallel import (
    causal_attn_scores_cp,
    gather_from_context_parallel_region,
    local_sequence_range,
    scatter_to_context_parallel_region,
)
from nano_megatron.reference.layers import causal_attn_scores


def _init_cp1(monkeypatch, port: str):
    """Initialize parallel context with cp_size=1 (world_size=1, gloo)."""
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


# ---------------------------------------------------------------------------
# local_sequence_range
# ---------------------------------------------------------------------------


def test_local_sequence_range():
    assert local_sequence_range(0, 2, 8) == (0, 4)
    assert local_sequence_range(1, 2, 8) == (4, 8)


def test_local_sequence_range_cp1():
    assert local_sequence_range(0, 1, 8) == (0, 8)


def test_local_sequence_range_four_way():
    assert local_sequence_range(0, 4, 16) == (0, 4)
    assert local_sequence_range(1, 4, 16) == (4, 8)
    assert local_sequence_range(2, 4, 16) == (8, 12)
    assert local_sequence_range(3, 4, 16) == (12, 16)


def test_local_sequence_range_nondivisible():
    with pytest.raises(ValueError, match="not divisible"):
        local_sequence_range(0, 3, 8)


# ---------------------------------------------------------------------------
# scatter / gather identity for cp_size=1
# ---------------------------------------------------------------------------


def test_scatter_gather_identity_cp1(monkeypatch):
    ctx = _init_cp1(monkeypatch, "29800")
    x = torch.randn(2, 8, 4, requires_grad=True)
    y = scatter_to_context_parallel_region(
        x, ctx.context_parallel_group, ctx.backend, 0, 1
    )
    assert torch.equal(y, x)
    z = gather_from_context_parallel_region(
        y, ctx.context_parallel_group, ctx.backend, 0, 1
    )
    assert torch.equal(z, x)
    z.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


# ---------------------------------------------------------------------------
# scatter / gather with custom seq_dim (e.g. dim=2 for [B, H, S, D])
# ---------------------------------------------------------------------------


def test_scatter_gather_seq_dim_2_cp1(monkeypatch):
    ctx = _init_cp1(monkeypatch, "29801")
    x = torch.randn(1, 2, 8, 4, requires_grad=True)  # [B, H, S, D]
    y = scatter_to_context_parallel_region(
        x, ctx.context_parallel_group, ctx.backend, 0, 1, seq_dim=2
    )
    assert torch.equal(y, x)
    z = gather_from_context_parallel_region(
        y, ctx.context_parallel_group, ctx.backend, 0, 1, seq_dim=2
    )
    assert torch.equal(z, x)
    z.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


# ---------------------------------------------------------------------------
# causal_attn_scores_cp matches full causal scores
# ---------------------------------------------------------------------------


def test_causal_attn_scores_cp_matches_full_slice():
    B, H, S, D = 1, 2, 8, 4
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    full = causal_attn_scores(q, k, scale=0.5)
    for start in (0, 4):
        q_local = q[:, :, start : start + 4, :]
        got = causal_attn_scores_cp(q_local, k, scale=0.5, query_start=start)
        assert torch.allclose(got, full[:, :, start : start + 4, :], atol=0, rtol=0)


# ---------------------------------------------------------------------------
# scatter forward shard math (documents narrow behavior)
# ---------------------------------------------------------------------------


def test_scatter_forward_shard_math():
    """Document that scatter narrows to the expected shard for each cp_rank."""
    x = torch.arange(16, dtype=torch.float32).view(1, 8, 2)
    # cp_size=2 splits seq dim (8) into two chunks of 4
    shard_0 = x[:, 0:4, :]
    shard_1 = x[:, 4:8, :]

    # Verify local_sequence_range produces matching ranges
    assert local_sequence_range(0, 2, 8) == (0, 4)
    assert local_sequence_range(1, 2, 8) == (4, 8)

    # Verify narrow matches expected shards
    chunk = 8 // 2
    assert torch.equal(x.narrow(1, 0 * chunk, chunk), shard_0)
    assert torch.equal(x.narrow(1, 1 * chunk, chunk), shard_1)


# ---------------------------------------------------------------------------
# scatter backward: pad-zeros (reverse of narrow), no collective
# ---------------------------------------------------------------------------


def test_scatter_backward_pad_zeros_math():
    """Scatter backward pads zeros — reverse of narrow, not all-gather.

    Each CP rank embeds the full sequence then narrows.  dL/d(full_embed)
    on rank r is zeros everywhere except the local shard, which holds
    grad_output.  All-gather would incorrectly place peer shards into
    every rank's embed gradient.
    """
    cp_size = 2
    seq_dim = 1
    # Local shard grad on each rank: distinct values.
    grad_rank0 = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    grad_rank1 = torch.arange(100, 108, dtype=torch.float32).view(1, 4, 2)
    full_seq = 8
    chunk = full_seq // cp_size

    for cp_rank, grad_output in ((0, grad_rank0), (1, grad_rank1)):
        full_shape = list(grad_output.shape)
        full_shape[seq_dim] = full_seq
        grad_input = grad_output.new_zeros(full_shape)
        grad_input.narrow(seq_dim, cp_rank * chunk, chunk).copy_(grad_output)
        # Only local shard is non-zero.
        assert torch.equal(
            grad_input.narrow(seq_dim, cp_rank * chunk, chunk), grad_output
        )
        other = 1 - cp_rank
        assert torch.equal(
            grad_input.narrow(seq_dim, other * chunk, chunk),
            torch.zeros_like(grad_output),
        )
        # Contrast: all-gather would cat both shards on every rank (wrong).
        all_gather_wrong = torch.cat([grad_rank0, grad_rank1], dim=seq_dim)
        assert not torch.equal(grad_input, all_gather_wrong)


def test_scatter_backward_pad_zeros_autograd(monkeypatch):
    """Autograd path: scatter backward returns zero-padded full grad."""
    from nano_megatron.parallel.context_parallel import (
        _ScatterToContextParallelRegion,
    )

    ctx = _init_cp1(monkeypatch, "29805")
    # Simulate cp_size=2 rank 0 via direct Function (world is still 1).
    # Forward still needs a real full tensor; we call backward math via apply
    # with cp_size=1 for identity, and unit-test pad path via manual backward
    # of the Function with a fake ctx-equivalent by running apply at cp=1
    # only for wiring — pad-zeros for cp>1 is covered by calling backward
    # through a thin wrapper that sets cp_size without collectives.
    x = torch.arange(16, dtype=torch.float32).view(1, 8, 2).requires_grad_(True)
    # Use Function.forward/backward directly with a stub backend that would
    # fail if all_gather were called.
    class _NoCollectiveBackend:
        def all_gather(self, *args, **kwargs):
            raise AssertionError("scatter backward must not all_gather")

        def all_reduce(self, *args, **kwargs):
            raise AssertionError("scatter backward must not all_reduce")

    backend = _NoCollectiveBackend()
    # rank 1 of cp_size=2: local shard is x[:, 4:8, :]
    y = _ScatterToContextParallelRegion.apply(
        x, ctx.context_parallel_group, backend, 1, 2, 1
    )
    assert y.shape == (1, 4, 2)
    assert torch.equal(y, x.detach()[:, 4:8, :])
    y.sum().backward()
    expected = torch.zeros_like(x)
    expected[:, 4:8, :] = 1.0
    assert torch.equal(x.grad, expected)


# ---------------------------------------------------------------------------
# gather backward identity for cp_size=1 (both grad_op modes)
# ---------------------------------------------------------------------------


def test_gather_backward_identity_cp1_reduce_scatter(monkeypatch):
    """Gather backward with cp_size=1 returns grad_output unchanged."""
    ctx = _init_cp1(monkeypatch, "29802")
    x = torch.randn(2, 8, 4, requires_grad=True)
    y = gather_from_context_parallel_region(
        x, ctx.context_parallel_group, ctx.backend, 0, 1, grad_op="reduce_scatter"
    )
    assert torch.equal(y, x)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_gather_backward_identity_cp1_split(monkeypatch):
    """Gather backward split mode with cp_size=1 is also identity."""
    ctx = _init_cp1(monkeypatch, "29803")
    x = torch.randn(2, 8, 4, requires_grad=True)
    y = gather_from_context_parallel_region(
        x, ctx.context_parallel_group, ctx.backend, 0, 1, grad_op="split"
    )
    assert torch.equal(y, x)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_gather_invalid_grad_op(monkeypatch):
    ctx = _init_cp1(monkeypatch, "29804")
    x = torch.randn(2, 8, 4)
    with pytest.raises(ValueError, match="grad_op"):
        gather_from_context_parallel_region(
            x, ctx.context_parallel_group, ctx.backend, 0, 1, grad_op="mean"
        )


# ---------------------------------------------------------------------------
# gather backward math: reduce_scatter vs split (documents the contract)
# ---------------------------------------------------------------------------


def test_gather_backward_reduce_scatter_math():
    """Document reduce-scatter backward for KV-style partial contributions.

    With cp_size=2, each rank holds a different partial grad_output over the
    full sequence.  reduce-scatter sums the chunks: rank i receives
    sum_r(grad_r[:, shard_i, :]).

    Single-process math: if both ranks contribute the same grad_output G,
    rank 0's local grad is 2 * G[:, 0:4, :] (sum of two identical chunks).
    """
    cp_size = 2
    grad_output = torch.ones(1, 8, 2) * 3.0
    chunks = [c.contiguous() for c in grad_output.chunk(cp_size, dim=1)]
    # Two ranks each contribute the same full G → sum on each shard is 2x.
    rank0_local = chunks[0] + chunks[0]
    rank1_local = chunks[1] + chunks[1]
    assert torch.equal(rank0_local, torch.ones(1, 4, 2) * 6.0)
    assert torch.equal(rank1_local, torch.ones(1, 4, 2) * 6.0)


def test_gather_backward_split_math():
    """Document split backward for identical full-sequence consumers (loss).

    Every CP rank computes the same global-mean CE, so dL/dlogits_full is
    identical on all ranks.  Backward must *narrow* to the local shard —
    not sum — otherwise grads are scaled by cp_size.

    With cp_size=2 and identical G on both ranks:
      split:   rank0 gets G[:, 0:4, :], rank1 gets G[:, 4:8, :]
      reduce_scatter would give 2 * those shards (wrong for loss).
    """
    cp_size = 2
    # Distinct values so shards are distinguishable.
    grad_output = torch.arange(16, dtype=torch.float32).view(1, 8, 2)
    chunk = 8 // cp_size
    for cp_rank in (0, 1):
        split_local = grad_output.narrow(1, cp_rank * chunk, chunk).contiguous()
        assert split_local.shape == (1, 4, 2)
        assert torch.equal(
            split_local, grad_output[:, cp_rank * chunk : (cp_rank + 1) * chunk, :]
        )
    # Contrast: reduce-scatter of two identical G would double each shard.
    chunks = list(grad_output.chunk(cp_size, dim=1))
    rs_rank0 = chunks[0] + chunks[0]
    assert torch.equal(rs_rank0, 2.0 * grad_output[:, 0:4, :])
    assert not torch.equal(rs_rank0, grad_output[:, 0:4, :])


# ---------------------------------------------------------------------------
# local_sequence_range cp_rank validation
# ---------------------------------------------------------------------------


def test_local_sequence_range_invalid_cp_rank():
    """cp_rank must be in [0, cp_size)."""
    with pytest.raises(ValueError, match="cp_rank"):
        local_sequence_range(-1, 2, 8)
    with pytest.raises(ValueError, match="cp_rank"):
        local_sequence_range(2, 2, 8)
    # cp_size < 1 is a separate check
    with pytest.raises(ValueError, match="cp_size must be >= 1"):
        local_sequence_range(0, 0, 8)
