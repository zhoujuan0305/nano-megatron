"""Unit tests for pipeline P2P send/recv helpers.

All tests use a fake ParallelContext and a mock CommBackend so that
no real distributed runtime is required.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(
    *,
    pp_rank: int,
    pp_size: int,
    pp_group: Any = "fake-pp-group",
    next_rank: int | None = 1,
    prev_rank: int | None = None,
) -> SimpleNamespace:
    """Build a minimal fake ParallelContext for P2P tests."""
    backend = MagicMock()
    # Mimic real CommBackend.recv which returns the same tensor it received.
    backend.recv.side_effect = lambda tensor, **kw: tensor
    return SimpleNamespace(
        pipeline_parallel_rank=pp_rank,
        pipeline_parallel_size=pp_size,
        pipeline_parallel_group=pp_group,
        backend=backend,
    )


def _patch_neighbors(prev_rank, next_rank):
    """Return a context-manager that patches pipeline_prev_rank / pipeline_next_rank."""
    import nano_megatron.schedules.p2p as mod

    prev_patch = patch.object(mod, "pipeline_prev_rank", return_value=prev_rank)
    next_patch = patch.object(mod, "pipeline_next_rank", return_value=next_rank)
    return prev_patch, next_patch


# ---------------------------------------------------------------------------
# Tests — first stage (pp_rank=0)
# ---------------------------------------------------------------------------

class TestFirstStage:
    """First stage should recv_forward=None, send_backward is no-op."""

    def test_recv_forward_returns_none(self):
        ctx = _make_ctx(pp_rank=0, pp_size=3)
        prev_p, next_p = _patch_neighbors(None, 1)
        with prev_p, next_p:
            from nano_megatron.schedules.p2p import recv_forward
            result = recv_forward(ctx, shape=(2, 4), dtype=torch.float32, device="cpu")
        assert result is None
        ctx.backend.recv.assert_not_called()

    def test_send_backward_is_noop(self):
        ctx = _make_ctx(pp_rank=0, pp_size=3)
        prev_p, next_p = _patch_neighbors(None, 1)
        with prev_p, next_p:
            from nano_megatron.schedules.p2p import send_backward
            t = torch.randn(2, 4)
            send_backward(ctx, t)
        ctx.backend.send.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — last stage (pp_rank=pp_size-1)
# ---------------------------------------------------------------------------

class TestLastStage:
    """Last stage should recv_backward=None, send_forward is no-op."""

    def test_recv_backward_returns_none(self):
        ctx = _make_ctx(pp_rank=2, pp_size=3, prev_rank=1)
        prev_p, next_p = _patch_neighbors(1, None)
        with prev_p, next_p:
            from nano_megatron.schedules.p2p import recv_backward
            result = recv_backward(ctx, shape=(2, 4), dtype=torch.float32, device="cpu")
        assert result is None
        ctx.backend.recv.assert_not_called()

    def test_send_forward_is_noop(self):
        ctx = _make_ctx(pp_rank=2, pp_size=3)
        prev_p, next_p = _patch_neighbors(1, None)
        with prev_p, next_p:
            from nano_megatron.schedules.p2p import send_forward
            t = torch.randn(2, 4)
            send_forward(ctx, t)
        ctx.backend.send.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — middle stage
# ---------------------------------------------------------------------------

class TestMiddleStage:
    """Middle stage should call backend.recv / backend.send correctly."""

    def test_recv_forward_calls_backend(self):
        prev_rank = 0
        ctx = _make_ctx(pp_rank=1, pp_size=3)
        prev_p, next_p = _patch_neighbors(prev_rank, 2)
        with prev_p, next_p:
            from nano_megatron.schedules.p2p import recv_forward
            result = recv_forward(ctx, shape=(2, 4), dtype=torch.float32, device="cpu")
        ctx.backend.recv.assert_called_once()
        call_args = ctx.backend.recv.call_args
        recv_tensor = call_args.args[0]
        assert recv_tensor.shape == (2, 4)
        assert recv_tensor.dtype == torch.float32
        assert call_args.kwargs["src"] == prev_rank
        assert call_args.kwargs["group"] == ctx.pipeline_parallel_group
        assert result is recv_tensor

    def test_send_forward_calls_backend(self):
        next_rank = 2
        ctx = _make_ctx(pp_rank=1, pp_size=3)
        prev_p, next_p = _patch_neighbors(0, next_rank)
        with prev_p, next_p:
            from nano_megatron.schedules.p2p import send_forward
            t = torch.randn(2, 4)
            send_forward(ctx, t)
        ctx.backend.send.assert_called_once_with(
            t, dst=next_rank, group=ctx.pipeline_parallel_group,
        )

    def test_recv_backward_calls_backend(self):
        next_rank = 2
        ctx = _make_ctx(pp_rank=1, pp_size=3)
        prev_p, next_p = _patch_neighbors(0, next_rank)
        with prev_p, next_p:
            from nano_megatron.schedules.p2p import recv_backward
            result = recv_backward(ctx, shape=(4, 8), dtype=torch.float16, device="cpu")
        ctx.backend.recv.assert_called_once()
        call_args = ctx.backend.recv.call_args
        recv_tensor = call_args.args[0]
        assert recv_tensor.shape == (4, 8)
        assert recv_tensor.dtype == torch.float16
        assert call_args.kwargs["src"] == next_rank
        assert call_args.kwargs["group"] == ctx.pipeline_parallel_group
        assert result is recv_tensor

    def test_send_backward_calls_backend(self):
        prev_rank = 0
        ctx = _make_ctx(pp_rank=1, pp_size=3)
        prev_p, next_p = _patch_neighbors(prev_rank, 2)
        with prev_p, next_p:
            from nano_megatron.schedules.p2p import send_backward
            t = torch.randn(4, 8)
            send_backward(ctx, t)
        ctx.backend.send.assert_called_once_with(
            t, dst=prev_rank, group=ctx.pipeline_parallel_group,
        )


# ---------------------------------------------------------------------------
# Tests — single-stage (pp_size=1)
# ---------------------------------------------------------------------------

class TestSingleStage:
    """When pp_size=1, all sends and recvs should be no-ops."""

    def test_recv_forward_none(self):
        ctx = _make_ctx(pp_rank=0, pp_size=1, next_rank=None, prev_rank=None)
        prev_p, next_p = _patch_neighbors(None, None)
        with prev_p, next_p:
            from nano_megatron.schedules.p2p import recv_forward
            assert recv_forward(ctx, shape=(2,), dtype=torch.float32, device="cpu") is None
        ctx.backend.recv.assert_not_called()

    def test_recv_backward_none(self):
        ctx = _make_ctx(pp_rank=0, pp_size=1, next_rank=None, prev_rank=None)
        prev_p, next_p = _patch_neighbors(None, None)
        with prev_p, next_p:
            from nano_megatron.schedules.p2p import recv_backward
            assert recv_backward(ctx, shape=(2,), dtype=torch.float32, device="cpu") is None
        ctx.backend.recv.assert_not_called()

    def test_send_forward_noop(self):
        ctx = _make_ctx(pp_rank=0, pp_size=1, next_rank=None, prev_rank=None)
        prev_p, next_p = _patch_neighbors(None, None)
        with prev_p, next_p:
            from nano_megatron.schedules.p2p import send_forward
            send_forward(ctx, torch.randn(2))
        ctx.backend.send.assert_not_called()

    def test_send_backward_noop(self):
        ctx = _make_ctx(pp_rank=0, pp_size=1, next_rank=None, prev_rank=None)
        prev_p, next_p = _patch_neighbors(None, None)
        with prev_p, next_p:
            from nano_megatron.schedules.p2p import send_backward
            send_backward(ctx, torch.randn(2))
        ctx.backend.send.assert_not_called()
