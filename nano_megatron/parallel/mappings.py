from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F

from nano_megatron.distributed.backend import CommBackend


class CommunicationBuffer:
    """Pre-allocated communication buffer manager.

    Caches buffers by (shape, dtype, device) so repeated all-reduce
    calls reuse the same memory instead of allocating every time.
    """

    def __init__(self) -> None:
        self._buffers: dict[tuple[tuple[int, ...], torch.dtype, torch.device], Tensor] = {}

    def get_buffer(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> Tensor:
        key = (tuple(shape), dtype, device)
        buf = self._buffers.get(key)
        if buf is None:
            buf = torch.empty(shape, dtype=dtype, device=device)
            self._buffers[key] = buf
        return buf


class _CopyToTPRegion(torch.autograd.Function):
    """Input is replicated across TP ranks.

    Forward is identity (the tensor is already on every rank); backward
    all-reduces the partial grad_inputs so the upstream receives the full
    gradient that flows toward the shared/replicated tensor.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, group: Any, backend: CommBackend, buffer_manager: CommunicationBuffer | None = None) -> Tensor:
        ctx.group = group
        ctx.backend = backend
        ctx.buffer_manager = buffer_manager
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None, None]:
        if ctx.buffer_manager is not None:
            buf = ctx.buffer_manager.get_buffer(grad_output.shape, grad_output.dtype, grad_output.device)
            buf.copy_(grad_output)
            work = ctx.backend.all_reduce(buf, group=ctx.group, op="sum", async_op=True)
            # GPU-side fence: ensures NCCL kernel completes before downstream
            # reads on the default stream.  When async_op is honoured, the
            # result is a Work handle; guard for backends that may return the
            # tensor directly (e.g. test stubs forcing sync mode).
            if hasattr(work, "wait"):
                work.wait()
            return buf, None, None, None
        work = ctx.backend.all_reduce(grad_output, group=ctx.group, op="sum", async_op=True)
        if hasattr(work, "wait"):
            work.wait()
        return grad_output, None, None, None


class _ReduceFromTPRegion(torch.autograd.Function):
    """Input is partial (one rank's shard); the full result is the sum.

    Forward all-reduces the partials to the full result on every rank;
    backward is identity because grad_output is already full.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, group: Any, backend: CommBackend, buffer_manager: CommunicationBuffer | None = None) -> Tensor:
        ctx.group = group
        ctx.backend = backend
        if buffer_manager is not None:
            buf = buffer_manager.get_buffer(x.shape, x.dtype, x.device)
            buf.copy_(x)
            work = backend.all_reduce(buf, group=group, op="sum", async_op=True)
            # GPU-side fence: wait for NCCL kernel before returning the buffer.
            if hasattr(work, "wait"):
                work.wait()
            return buf
        work = backend.all_reduce(x, group=group, op="sum", async_op=True)
        if hasattr(work, "wait"):
            work.wait()
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None, None]:
        return grad_output, None, None, None


def column_shard(
    weight: Tensor, bias: Tensor | None, tp_rank: int, tp_size: int
) -> tuple[Tensor, Tensor | None]:
    out_full, _ = weight.shape
    if out_full % tp_size != 0:
        raise ValueError(
            f"column output dim ({out_full}) not divisible by tp_size ({tp_size})"
        )
    chunk = out_full // tp_size
    w_local = weight.data[tp_rank * chunk:(tp_rank + 1) * chunk, :].clone()
    b_local = (
        bias.data[tp_rank * chunk:(tp_rank + 1) * chunk].clone()
        if bias is not None
        else None
    )
    return w_local, b_local


def blockwise_column_shard(
    weight: Tensor,
    bias: Tensor | None,
    tp_rank: int,
    tp_size: int,
    block_sizes: tuple[int, ...],
) -> tuple[Tensor, Tensor | None]:
    """Shard a column-parallel weight whose rows are concatenated logical blocks.

    Contiguous column_shard cuts across block boundaries. For layouts such as
    fused QKV ``[Q; K; V]`` or SwiGLU fc1 ``[gate; up]``, each block must be
    sharded independently so the local rows stay ``[block0_r; block1_r; ...]``.
    """
    out_full, _ = weight.shape
    expected = sum(block_sizes)
    if out_full != expected:
        raise ValueError(
            f"column out dim ({out_full}) != sum(block_sizes) ({expected})"
        )
    for i, dim in enumerate(block_sizes):
        if dim % tp_size != 0:
            raise ValueError(
                f"block_sizes[{i}] ({dim}) not divisible by tp_size ({tp_size})"
            )
    w_parts = weight.data.split(list(block_sizes), dim=0)
    w_local = torch.cat(
        [
            part[tp_rank * (dim // tp_size) : (tp_rank + 1) * (dim // tp_size)]
            for part, dim in zip(w_parts, block_sizes)
        ],
        dim=0,
    ).clone()
    b_local: Tensor | None = None
    if bias is not None:
        b_parts = bias.data.split(list(block_sizes), dim=0)
        b_local = torch.cat(
            [
                part[tp_rank * (dim // tp_size) : (tp_rank + 1) * (dim // tp_size)]
                for part, dim in zip(b_parts, block_sizes)
            ],
            dim=0,
        ).clone()
    return w_local, b_local


def fused_qkv_column_shard(
    weight: Tensor,
    bias: Tensor | None,
    tp_rank: int,
    tp_size: int,
    q_dim: int,
    kv_dim: int,
) -> tuple[Tensor, Tensor | None]:
    """Shard fused QKV weight laid out as [Q; K; V] into [Q_r; K_r; V_r]."""
    return blockwise_column_shard(
        weight, bias, tp_rank, tp_size, (q_dim, kv_dim, kv_dim)
    )


def row_shard(
    weight: Tensor, bias: Tensor | None, tp_rank: int, tp_size: int
) -> tuple[Tensor, Tensor | None]:
    _, in_full = weight.shape
    if in_full % tp_size != 0:
        raise ValueError(
            f"row input dim ({in_full}) not divisible by tp_size ({tp_size})"
        )
    chunk = in_full // tp_size
    w_local = weight.data[:, tp_rank * chunk:(tp_rank + 1) * chunk].clone()
    b_local = bias.data.clone() if bias is not None else None
    return w_local, b_local


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        weight: Tensor,
        bias: Tensor | None,
        tp_rank: int,
        tp_size: int,
        group: Any,
        backend: CommBackend,
        *,
        weight_is_local: bool = False,
    ) -> None:
        super().__init__()
        if weight_is_local:
            w_local = weight.data.clone()
            b_local = bias.data.clone() if bias is not None else None
        else:
            w_local, b_local = column_shard(weight, bias, tp_rank, tp_size)
        self.weight = nn.Parameter(w_local)
        self.bias = nn.Parameter(b_local) if b_local is not None else None
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.group = group
        self.backend = backend
        self.buffer_manager = CommunicationBuffer()

    def forward(self, x: Tensor) -> Tensor:
        x = _CopyToTPRegion.apply(x, self.group, self.backend, self.buffer_manager)
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        weight: Tensor,
        bias: Tensor | None,
        tp_rank: int,
        tp_size: int,
        group: Any,
        backend: CommBackend,
    ) -> None:
        super().__init__()
        w_local, b_local = row_shard(weight, bias, tp_rank, tp_size)
        self.weight = nn.Parameter(w_local)
        self.bias = nn.Parameter(b_local) if b_local is not None else None
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.group = group
        self.backend = backend
        self.buffer_manager = CommunicationBuffer()

    def forward(self, x: Tensor) -> Tensor:
        out = F.linear(x, self.weight, bias=None)
        out = _ReduceFromTPRegion.apply(out, self.group, self.backend, self.buffer_manager)
        if self.bias is not None:
            out = out + self.bias
        return out
