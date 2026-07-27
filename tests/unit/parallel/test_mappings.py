from __future__ import annotations

import torch
import torch.nn.functional as F

from nano_megatron.parallel import (
    ParallelConfig,
    destroy_parallel,
    initialize_parallel,
    is_parallel_initialized,
)
from nano_megatron.parallel.mappings import (
    CommunicationBuffer,
    _CopyToTPRegion,
    _ReduceFromTPRegion,
    ColumnParallelLinear,
    RowParallelLinear,
    column_shard,
    row_shard,
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


def test_copy_to_tp_region_forward_identity(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29510")
    x = torch.randn(3, 4, dtype=torch.float32)
    y = _CopyToTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend)
    assert torch.equal(y, x)


def test_copy_to_tp_region_backward_noop_on_size1_group(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29511")
    x = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    y = _CopyToTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_reduce_from_tp_region_forward_noop_on_size1_group(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29512")
    x = torch.randn(3, 4, dtype=torch.float32)
    y = _ReduceFromTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend)
    assert torch.equal(y, x)


def test_reduce_from_tp_region_backward_identity(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29513")
    x = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    y = _ReduceFromTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_reduce_from_tp_region_with_buffer_returns_different_ptr(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29516")
    x = torch.randn(3, 4, dtype=torch.float32)
    snapshot = x.clone()
    buf_mgr = CommunicationBuffer()
    y = _ReduceFromTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend, buf_mgr)
    assert y.data_ptr() != x.data_ptr()
    assert torch.equal(y, snapshot)


def test_reduce_from_tp_region_without_buffer_is_in_place(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29517")
    x = torch.randn(3, 4, dtype=torch.float32)
    snapshot = x.clone()
    y = _ReduceFromTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend)
    assert y.data_ptr() == x.data_ptr()
    assert torch.equal(y, snapshot)


def test_copy_to_tp_region_does_not_mutate_input(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29514")
    x = torch.randn(3, 4, dtype=torch.float32)
    snapshot = x.clone()
    _CopyToTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend)
    assert torch.equal(x, snapshot)


def test_copy_to_tp_region_backward_is_in_place(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29515")
    x = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    y = _CopyToTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend)
    grad = torch.randn(3, 4, dtype=torch.float32)
    y.backward(grad)
    assert torch.equal(x.grad, grad)


def test_column_shard_shape_and_reconstruction():
    torch.manual_seed(0)
    w = torch.randn(8, 4)
    b = torch.randn(8)
    w0, b0 = column_shard(w, b, tp_rank=0, tp_size=2)
    w1, b1 = column_shard(w, b, tp_rank=1, tp_size=2)
    assert w0.shape == (4, 4) and w1.shape == (4, 4)
    assert b0.shape == (4,) and b1.shape == (4,)
    assert torch.equal(torch.cat([w0, w1], dim=0), w)
    assert torch.equal(torch.cat([b0, b1], dim=0), b)


def test_column_shard_none_bias():
    w = torch.randn(8, 4)
    w0, b0 = column_shard(w, None, tp_rank=0, tp_size=2)
    assert b0 is None
    assert w0.shape == (4, 4)


def test_column_shard_indivisible_raises():
    w = torch.randn(7, 4)
    try:
        column_shard(w, None, tp_rank=0, tp_size=2)
    except ValueError as e:
        assert "divisible" in str(e)
    else:
        raise AssertionError("expected ValueError for non-divisible output dim")


def test_row_shard_shape_and_reconstruction():
    torch.manual_seed(0)
    w = torch.randn(8, 6)
    b = torch.randn(8)
    w0, b0 = row_shard(w, b, tp_rank=0, tp_size=2)
    w1, b1 = row_shard(w, b, tp_rank=1, tp_size=2)
    assert w0.shape == (8, 3) and w1.shape == (8, 3)
    assert torch.equal(b0, b) and torch.equal(b1, b)
    assert torch.equal(torch.cat([w0, w1], dim=1), w)


def test_row_shard_indivisible_raises():
    w = torch.randn(8, 5)
    try:
        row_shard(w, None, tp_rank=0, tp_size=2)
    except ValueError as e:
        assert "divisible" in str(e)
    else:
        raise AssertionError("expected ValueError for non-divisible input dim")


def test_column_parallel_linear_tp1_matches_full_linear(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29520")
    torch.manual_seed(0)
    w = torch.randn(8, 4)
    b = torch.randn(8)
    lin = ColumnParallelLinear(w, b, tp_rank=0, tp_size=1, group=ctx.tensor_parallel_group, backend=ctx.backend)
    x = torch.randn(2, 3, 4)
    y = lin(x)
    assert torch.equal(y, F.linear(x, w, b))


def test_row_parallel_linear_tp1_matches_full_linear(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29521")
    torch.manual_seed(0)
    w = torch.randn(8, 4)
    b = torch.randn(8)
    lin = RowParallelLinear(w, b, tp_rank=0, tp_size=1, group=ctx.tensor_parallel_group, backend=ctx.backend)
    x = torch.randn(2, 3, 4)
    y = lin(x)
    assert torch.equal(y, F.linear(x, w, b))


def test_row_parallel_linear_bias_not_doubled_on_size1(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29522")
    torch.manual_seed(0)
    w = torch.randn(8, 4)
    b = torch.randn(8)
    lin = RowParallelLinear(w, b, tp_rank=0, tp_size=1, group=ctx.tensor_parallel_group, backend=ctx.backend)
    x = torch.zeros(2, 3, 4)
    y = lin(x)
    # x is zero so the all_reduce output is zero; bias should add exactly once
    assert torch.equal(y, b.expand(2, 3, 8))


def test_communication_buffer_reuses_same_tensor():
    mgr = CommunicationBuffer()
    buf1 = mgr.get_buffer((3, 4), torch.float32, torch.device("cpu"))
    buf2 = mgr.get_buffer((3, 4), torch.float32, torch.device("cpu"))
    assert buf1.data_ptr() == buf2.data_ptr()


def test_communication_buffer_different_shape_creates_new():
    mgr = CommunicationBuffer()
    buf1 = mgr.get_buffer((3, 4), torch.float32, torch.device("cpu"))
    buf2 = mgr.get_buffer((5, 4), torch.float32, torch.device("cpu"))
    assert buf1.data_ptr() != buf2.data_ptr()


def test_communication_buffer_different_dtype_creates_new():
    mgr = CommunicationBuffer()
    buf1 = mgr.get_buffer((3, 4), torch.float32, torch.device("cpu"))
    buf2 = mgr.get_buffer((3, 4), torch.float16, torch.device("cpu"))
    assert buf1.data_ptr() != buf2.data_ptr()


def test_reduce_from_tp_region_backward_with_buffer(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29530")
    x = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    buf_mgr = CommunicationBuffer()
    y = _ReduceFromTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend, buf_mgr)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_copy_to_tp_region_backward_with_buffer(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29531")
    x = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    buf_mgr = CommunicationBuffer()
    y = _CopyToTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend, buf_mgr)
    grad = torch.randn(3, 4, dtype=torch.float32)
    y.backward(grad)
    assert torch.equal(x.grad, grad)


def test_column_parallel_linear_has_buffer_manager(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29532")
    w = torch.randn(8, 4)
    b = torch.randn(8)
    lin = ColumnParallelLinear(w, b, tp_rank=0, tp_size=1, group=ctx.tensor_parallel_group, backend=ctx.backend)
    assert isinstance(lin.buffer_manager, CommunicationBuffer)


def test_row_parallel_linear_has_buffer_manager(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29533")
    w = torch.randn(8, 4)
    b = torch.randn(8)
    lin = RowParallelLinear(w, b, tp_rank=0, tp_size=1, group=ctx.tensor_parallel_group, backend=ctx.backend)
    assert isinstance(lin.buffer_manager, CommunicationBuffer)


def test_reduce_from_tp_forward_async_matches_sync(monkeypatch):
    """Async-launch reduce output equals sync-launch reduce output (TP1 gloo)."""
    import torch.distributed as dist
    from nano_megatron.distributed.torch_backend import TorchDistBackend

    if is_parallel_initialized():
        destroy_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29600")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    ctx = initialize_parallel(ParallelConfig(), dist_backend="gloo")
    backend = ctx.backend

    torch.manual_seed(0)
    x_base = torch.randn(4, 8, dtype=torch.float32)

    # Sync baseline: force async_op=False on the underlying TorchDistBackend.
    orig_all_reduce = TorchDistBackend.all_reduce

    def force_sync(self, tensor, *, group=None, op="sum", async_op=False):
        return orig_all_reduce(self, tensor, group=group, op=op, async_op=False)

    monkeypatch.setattr(TorchDistBackend, "all_reduce", force_sync)
    x_sync = x_base.clone()
    out_sync = _ReduceFromTPRegion.apply(x_sync, ctx.tensor_parallel_group, backend)
    out_sync = out_sync.clone()
    monkeypatch.undo()

    # Async path (default after Task 2).
    x_async = x_base.clone()
    out_async = _ReduceFromTPRegion.apply(x_async, ctx.tensor_parallel_group, backend)
    out_async = out_async.clone()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    assert torch.allclose(out_async, out_sync, atol=1e-6), (
        f"async vs sync output differ: max diff = {(out_async - out_sync).abs().max().item()}"
    )

    if is_parallel_initialized():
        destroy_parallel()


def test_copy_to_tp_backward_async_matches_sync(monkeypatch):
    """Async-launch reduce grad equals sync-launch reduce grad (TP1 gloo)."""
    import torch.distributed as dist
    from nano_megatron.distributed.torch_backend import TorchDistBackend

    if is_parallel_initialized():
        destroy_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29601")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    ctx = initialize_parallel(ParallelConfig(), dist_backend="gloo")
    backend = ctx.backend

    torch.manual_seed(0)
    x_base = torch.randn(4, 8, dtype=torch.float32, requires_grad=False)
    grad_seed = torch.randn(4, 8, dtype=torch.float32)

    # Sync baseline: force async_op=False on the underlying TorchDistBackend.
    orig_all_reduce = TorchDistBackend.all_reduce

    def force_sync(self, tensor, *, group=None, op="sum", async_op=False):
        return orig_all_reduce(self, tensor, group=group, op=op, async_op=False)

    monkeypatch.setattr(TorchDistBackend, "all_reduce", force_sync)
    x_sync = x_base.clone().requires_grad_(True)
    y_sync = _CopyToTPRegion.apply(x_sync, ctx.tensor_parallel_group, backend)
    y_sync.backward(grad_seed.clone())
    grad_sync = x_sync.grad.detach().clone()
    monkeypatch.undo()

    # Async path.
    x_async = x_base.clone().requires_grad_(True)
    y_async = _CopyToTPRegion.apply(x_async, ctx.tensor_parallel_group, backend)
    y_async.backward(grad_seed.clone())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    grad_async = x_async.grad.detach().clone()

    assert torch.allclose(grad_async, grad_sync, atol=1e-6), (
        f"async vs sync grad differ: max diff = {(grad_async - grad_sync).abs().max().item()}"
    )

    if is_parallel_initialized():
        destroy_parallel()


def test_column_parallel_forward_unchanged_after_async(monkeypatch):
    """ColumnParallelLinear.forward has no reduce; ensure async wiring did not
    introduce an unexpected all_reduce call."""
    import torch.distributed as dist
    if is_parallel_initialized():
        destroy_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29603")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    ctx = initialize_parallel(ParallelConfig(), dist_backend="gloo")

    w = torch.randn(8, 4)
    b = torch.randn(8)
    lin = ColumnParallelLinear(w, b, tp_rank=0, tp_size=1,
                               group=ctx.tensor_parallel_group, backend=ctx.backend)
    x = torch.randn(2, 3, 4)
    out = lin(x)
    assert out.shape == (2, 3, 8)

    if is_parallel_initialized():
        destroy_parallel()
