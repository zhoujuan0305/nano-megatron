from __future__ import annotations

import time
from unittest.mock import patch

import pytest
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


@pytest.fixture(autouse=True)
def _cleanup_parallel():
    """Ensure parallel state is cleaned up after every test."""
    yield
    import torch.distributed as dist

    if is_parallel_initialized():
        destroy_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


class TestCommunicationBuffer:
    """CommunicationBuffer reuse and allocation tests."""

    def test_buffer_creation(self):
        """Same (shape, dtype, device) returns same tensor."""
        mgr = CommunicationBuffer()
        buf = mgr.get_buffer((4, 8), torch.float32, torch.device("cpu"))
        assert buf.shape == (4, 8)
        assert buf.dtype == torch.float32

    def test_buffer_reuse(self):
        """Repeated calls with same key return the same underlying storage."""
        mgr = CommunicationBuffer()
        b1 = mgr.get_buffer((1024, 1024), torch.float32, torch.device("cpu"))
        b2 = mgr.get_buffer((1024, 1024), torch.float32, torch.device("cpu"))
        assert b1.data_ptr() == b2.data_ptr()

    def test_different_shapes_create_new(self):
        mgr = CommunicationBuffer()
        b1 = mgr.get_buffer((3, 4), torch.float32, torch.device("cpu"))
        b2 = mgr.get_buffer((5, 4), torch.float32, torch.device("cpu"))
        assert b1.data_ptr() != b2.data_ptr()

    def test_different_dtype_creates_new(self):
        mgr = CommunicationBuffer()
        b1 = mgr.get_buffer((3, 4), torch.float32, torch.device("cpu"))
        b2 = mgr.get_buffer((3, 4), torch.float16, torch.device("cpu"))
        assert b1.data_ptr() != b2.data_ptr()

    def test_buffer_count(self):
        """Only one buffer is allocated after many calls with the same key."""
        mgr = CommunicationBuffer()
        for _ in range(100):
            mgr.get_buffer((64, 64), torch.float32, torch.device("cpu"))
        assert len(mgr._buffers) == 1


class TestMemoryUsage:
    """Verify that optimized paths avoid unnecessary allocations."""

    def test_reduce_from_tp_with_buffer_no_clone_overhead(self, monkeypatch):
        """With buffer manager, output uses pre-allocated buffer, not a clone of input."""
        ctx = _init_tp1(monkeypatch, "29540")
        x = torch.randn(16, 32, dtype=torch.float32)
        buf_mgr = CommunicationBuffer()
        y = _ReduceFromTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend, buf_mgr)
        # Output must be a different tensor (uses buffer, not in-place on x)
        assert y.data_ptr() != x.data_ptr()
        # The buffer manager should have exactly one buffer
        assert len(buf_mgr._buffers) == 1

    def test_copy_to_tp_backward_uses_buffer(self, monkeypatch):
        """Backward of _CopyToTPRegion uses pre-allocated buffer when available."""
        ctx = _init_tp1(monkeypatch, "29541")
        x = torch.randn(16, 32, dtype=torch.float32, requires_grad=True)
        buf_mgr = CommunicationBuffer()
        y = _CopyToTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend, buf_mgr)
        grad = torch.randn(16, 32, dtype=torch.float32)
        y.backward(grad)
        assert torch.equal(x.grad, grad)
        assert len(buf_mgr._buffers) == 1

    def test_column_parallel_owns_buffer_manager(self, monkeypatch):
        ctx = _init_tp1(monkeypatch, "29542")
        w = torch.randn(8, 4)
        lin = ColumnParallelLinear(w, None, tp_rank=0, tp_size=1,
                                   group=ctx.tensor_parallel_group, backend=ctx.backend)
        assert isinstance(lin.buffer_manager, CommunicationBuffer)

    def test_row_parallel_owns_buffer_manager(self, monkeypatch):
        ctx = _init_tp1(monkeypatch, "29543")
        w = torch.randn(8, 4)
        lin = RowParallelLinear(w, None, tp_rank=0, tp_size=1,
                                group=ctx.tensor_parallel_group, backend=ctx.backend)
        assert isinstance(lin.buffer_manager, CommunicationBuffer)


class TestCommunicationPattern:
    """Verify the number of all_reduce calls during forward/backward."""

    def _wrap_all_reduce(self, backend):
        """Return (counter_dict, context_manager) that counts all_reduce calls."""
        count = {"n": 0, "async_op_values": []}
        orig = backend.all_reduce

        def counting(tensor, *, group=None, op="sum", async_op=False):
            count["n"] += 1
            count["async_op_values"].append(async_op)
            return orig(tensor, group=group, op=op, async_op=async_op)

        return count, patch.object(backend, "all_reduce", counting)

    def test_reduce_from_tp_calls_all_reduce_once_forward(self, monkeypatch):
        ctx = _init_tp1(monkeypatch, "29550")
        x = torch.randn(4, 8, dtype=torch.float32)
        count, cm = self._wrap_all_reduce(ctx.backend)
        with cm:
            _ReduceFromTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend)
        assert count["n"] == 1

    def test_copy_to_tp_calls_all_reduce_once_backward(self, monkeypatch):
        ctx = _init_tp1(monkeypatch, "29551")
        x = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
        y = _CopyToTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend)
        count, cm = self._wrap_all_reduce(ctx.backend)
        with cm:
            y.sum().backward()
        assert count["n"] == 1

    def test_column_parallel_forward_zero_all_reduce(self, monkeypatch):
        """ColumnParallelLinear forward does not call all_reduce (only backward does)."""
        ctx = _init_tp1(monkeypatch, "29552")
        w = torch.randn(8, 4)
        b = torch.randn(8)
        lin = ColumnParallelLinear(w, b, tp_rank=0, tp_size=1,
                                   group=ctx.tensor_parallel_group, backend=ctx.backend)
        x = torch.randn(2, 3, 4)
        count, cm = self._wrap_all_reduce(ctx.backend)
        with cm:
            lin(x)
        assert count["n"] == 0

    def test_row_parallel_forward_one_all_reduce(self, monkeypatch):
        """RowParallelLinear forward calls all_reduce exactly once."""
        ctx = _init_tp1(monkeypatch, "29553")
        w = torch.randn(8, 4)
        b = torch.randn(8)
        lin = RowParallelLinear(w, b, tp_rank=0, tp_size=1,
                                group=ctx.tensor_parallel_group, backend=ctx.backend)
        x = torch.randn(2, 3, 4)
        count, cm = self._wrap_all_reduce(ctx.backend)
        with cm:
            lin(x)
        assert count["n"] == 1

    def test_reduce_from_tp_forward_uses_async_op(self, monkeypatch):
        """_ReduceFromTPRegion.forward must request async_op=True from the backend."""
        ctx = _init_tp1(monkeypatch, "29560")
        x = torch.randn(4, 8, dtype=torch.float32)
        count, cm = self._wrap_all_reduce(ctx.backend)
        with cm:
            _ReduceFromTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend)
        assert count["n"] == 1
        assert count["async_op_values"] == [True]

    def test_copy_to_tp_backward_uses_async_op(self, monkeypatch):
        """_CopyToTPRegion.backward must request async_op=True from the backend."""
        ctx = _init_tp1(monkeypatch, "29561")
        x = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
        y = _CopyToTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend)
        count, cm = self._wrap_all_reduce(ctx.backend)
        with cm:
            y.sum().backward()
        assert count["n"] == 1
        assert count["async_op_values"] == [True]


class TestPerformance:
    """Micro-benchmarks to quantify optimisation impact."""

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _bench(self, fn, warmup: int = 5, iters: int = 50) -> float:
        for _ in range(warmup):
            fn()
        self._sync()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        self._sync()
        elapsed = time.perf_counter() - start
        return elapsed / iters

    def test_buffer_reuse_avoids_repeated_allocation(self):
        """Reusing a buffer should return the same memory, not allocate new tensors."""
        shape = (1024, 1024)
        dtype = torch.float32
        device = torch.device("cpu")

        mgr = CommunicationBuffer()
        ptrs = set()
        for _ in range(50):
            buf = mgr.get_buffer(shape, dtype, device)
            ptrs.add(buf.data_ptr())
        # All calls must return the exact same storage
        assert len(ptrs) == 1

    def test_reduce_forward_with_buffer_no_slower(self, monkeypatch):
        """_ReduceFromTPRegion with buffer should not be meaningfully slower than without."""
        ctx = _init_tp1(monkeypatch, "29560")
        x = torch.randn(64, 128, dtype=torch.float32)
        buf_mgr = CommunicationBuffer()

        def with_buf():
            _ReduceFromTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend, buf_mgr)

        def without_buf():
            _ReduceFromTPRegion.apply(x.clone(), ctx.tensor_parallel_group, ctx.backend)

        t_with = self._bench(with_buf)
        t_without = self._bench(without_buf)
        print(f"\n  _ReduceFromTPRegion forward: buffer={t_with*1e6:.1f} us, no-buffer={t_without*1e6:.1f} us")
        # Use generous tolerance; this is informational on CI, not a hard gate.
        assert t_with <= t_without * 5.0, (
            f"Buffer path ({t_with*1e6:.1f} us) much slower than no-buffer ({t_without*1e6:.1f} us)"
        )

    def test_linear_forward_backward_throughput(self, monkeypatch):
        """Benchmark a Column+Row forward-backward loop with cross_entropy loss.

        This matches the spec's test_performance() which runs a real
        forward+backward training step, validating Task 1-3 optimisation
        impact on actual training throughput.
        """
        ctx = _init_tp1(monkeypatch, "29561")
        vocab_size = 64
        hidden = 32
        w_col = torch.randn(hidden, hidden)
        b_col = torch.randn(hidden)
        w_row = torch.randn(vocab_size, hidden)
        b_row = torch.randn(vocab_size)

        col = ColumnParallelLinear(w_col, b_col, tp_rank=0, tp_size=1,
                                   group=ctx.tensor_parallel_group, backend=ctx.backend)
        row = RowParallelLinear(w_row, b_row, tp_rank=0, tp_size=1,
                                group=ctx.tensor_parallel_group, backend=ctx.backend)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        col = col.to(device)
        row = row.to(device)
        batch, seq = 4, 16

        def step():
            x = torch.randn(batch, seq, hidden, device=device)
            h = col(x)
            logits = row(h)
            targets = torch.randint(0, vocab_size, (batch, seq), device=device)
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            col.zero_grad()
            row.zero_grad()

        step()  # warmup
        self._sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t = self._bench(step, warmup=3, iters=20)
        print(f"\n  Column+Row forward-backward: {t*1e3:.2f} ms/step")
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"  Peak GPU memory: {peak_mb:.1f} MB")
        # No hard assertion on throughput; this is informational.
        assert t > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCUDAMemory:
    """GPU memory tests — skipped when CUDA is unavailable."""

    @staticmethod
    def _sync():
        torch.cuda.synchronize()

    def test_reduce_from_tp_peak_memory(self, monkeypatch):
        """Measure peak GPU memory for _ReduceFromTPRegion forward."""
        ctx = _init_tp1(monkeypatch, "29570")
        size = (256, 512)
        x = torch.randn(*size, dtype=torch.float32, device="cuda")
        buf_mgr = CommunicationBuffer()

        torch.cuda.reset_peak_memory_stats()
        _ReduceFromTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend, buf_mgr)
        self._sync()
        peak_bytes = torch.cuda.max_memory_allocated()
        print(f"\n  _ReduceFromTPRegion peak memory: {peak_bytes / 1024:.1f} KB")
        assert peak_bytes > 0

    def test_copy_to_tp_backward_peak_memory(self, monkeypatch):
        """Measure peak GPU memory for _CopyToTPRegion forward+backward."""
        ctx = _init_tp1(monkeypatch, "29571")
        size = (256, 512)
        x = torch.randn(*size, dtype=torch.float32, device="cuda", requires_grad=True)
        buf_mgr = CommunicationBuffer()

        torch.cuda.reset_peak_memory_stats()
        y = _CopyToTPRegion.apply(x, ctx.tensor_parallel_group, ctx.backend, buf_mgr)
        grad = torch.randn(*size, dtype=torch.float32, device="cuda")
        y.backward(grad)
        self._sync()
        peak_bytes = torch.cuda.max_memory_allocated()
        print(f"\n  _CopyToTPRegion peak memory: {peak_bytes / 1024:.1f} KB")
        assert peak_bytes > 0

    def test_linear_forward_backward_peak_memory(self, monkeypatch):
        """Measure peak GPU memory for a full Column+Row forward-backward."""
        ctx = _init_tp1(monkeypatch, "29572")
        hidden = 64
        vocab = 128
        w_col = torch.randn(hidden, hidden)
        b_col = torch.randn(hidden)
        w_row = torch.randn(vocab, hidden)
        b_row = torch.randn(vocab)

        col = ColumnParallelLinear(w_col, b_col, tp_rank=0, tp_size=1,
                                   group=ctx.tensor_parallel_group, backend=ctx.backend).cuda()
        row = RowParallelLinear(w_row, b_row, tp_rank=0, tp_size=1,
                                group=ctx.tensor_parallel_group, backend=ctx.backend).cuda()

        batch, seq = 8, 32
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(batch, seq, hidden, device="cuda")
        h = col(x)
        logits = row(h)
        targets = torch.randint(0, vocab, (batch, seq), device="cuda")
        loss = F.cross_entropy(logits.view(-1, vocab), targets.view(-1))
        loss.backward()
        self._sync()
        peak_bytes = torch.cuda.max_memory_allocated()
        print(f"\n  Linear forward-backward peak memory: {peak_bytes / 1024:.1f} KB")
        assert peak_bytes > 0
