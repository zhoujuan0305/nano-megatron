from __future__ import annotations

import pytest
import torch

from nano_megatron.parallel import (
    ColumnParallelLinear,
    ParallelConfig,
    RowParallelLinear,
    destroy_parallel,
    gather_from_sequence_parallel_region,
    initialize_parallel,
    is_parallel_initialized,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)


def _init_tp1(monkeypatch, port: str):
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


def test_scatter_gather_identity_tp1(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29700")
    x = torch.randn(2, 8, 4, requires_grad=True)
    y = scatter_to_sequence_parallel_region(
        x, ctx.tensor_parallel_group, ctx.backend, 0, 1
    )
    assert torch.equal(y, x)
    z = gather_from_sequence_parallel_region(
        y, ctx.tensor_parallel_group, ctx.backend, 0, 1
    )
    assert torch.equal(z, x)
    z.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_reduce_scatter_identity_tp1(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29701")
    x = torch.randn(2, 8, 4, requires_grad=True)
    y = reduce_scatter_to_sequence_parallel_region(
        x, ctx.tensor_parallel_group, ctx.backend, 0, 1
    )
    assert torch.equal(y, x)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_scatter_rejects_nondivisible_seq(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29702")
    x = torch.randn(2, 7, 4)
    with pytest.raises(ValueError, match="not divisible"):
        scatter_to_sequence_parallel_region(
            x, ctx.tensor_parallel_group, ctx.backend, 0, 2
        )


def test_local_seq_shard_slices():
    # Documents expected shard boundaries for tp=2
    x = torch.arange(16, dtype=torch.float32).view(1, 8, 2)
    chunk = 8 // 2
    r0 = x[:, 0:chunk, :]
    r1 = x[:, chunk : 2 * chunk, :]
    assert r0.shape == (1, 4, 2)
    assert r1.shape == (1, 4, 2)
    assert torch.equal(torch.cat([r0, r1], dim=1), x)


def test_column_row_sp_flag_tp1_matches_nosp(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29710")
    w_col = torch.randn(6, 4)
    b_col = torch.randn(6)
    w_row = torch.randn(4, 6)
    b_row = torch.randn(4)
    x = torch.randn(2, 8, 4, requires_grad=True)
    col = ColumnParallelLinear(
        w_col, b_col, 0, 1, ctx.tensor_parallel_group, ctx.backend,
        sequence_parallel=True,
    )
    row = RowParallelLinear(
        w_row, b_row, 0, 1, ctx.tensor_parallel_group, ctx.backend,
        sequence_parallel=True,
    )
    y = row(col(x))
    y_ref = torch.nn.functional.linear(
        torch.nn.functional.linear(x, w_col, b_col), w_row, b_row
    )
    assert torch.allclose(y, y_ref, atol=0, rtol=0)
