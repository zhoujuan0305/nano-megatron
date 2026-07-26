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
            ctx.backend.all_reduce(buf, group=ctx.group, op="sum")
            return buf, None, None, None
        ctx.backend.all_reduce(grad_output, group=ctx.group, op="sum")
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
            backend.all_reduce(buf, group=group, op="sum")
            return buf
        backend.all_reduce(x, group=group, op="sum")
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
    ) -> None:
        super().__init__()
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
