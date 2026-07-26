from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F

from nano_megatron.distributed.backend import CommBackend


class _CopyToTPRegion(torch.autograd.Function):
    """Input is replicated across TP ranks.

    Forward is identity (the tensor is already on every rank); backward
    all-reduces the partial grad_inputs so the upstream receives the full
    gradient that flows toward the shared/replicated tensor.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, group: Any, backend: CommBackend) -> Tensor:
        ctx.group = group
        ctx.backend = backend
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        grad = grad_output.clone()
        ctx.backend.all_reduce(grad, group=ctx.group, op="sum")
        return grad, None, None


class _ReduceFromTPRegion(torch.autograd.Function):
    """Input is partial (one rank's shard); the full result is the sum.

    Forward all-reduces the partials to the full result on every rank;
    backward is identity because grad_output is already full.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, group: Any, backend: CommBackend) -> Tensor:
        ctx.group = group
        ctx.backend = backend
        out = x.clone()
        ctx.backend.all_reduce(out, group=ctx.group, op="sum")
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        return grad_output, None, None


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

    def forward(self, x: Tensor) -> Tensor:
        x = _CopyToTPRegion.apply(x, self.group, self.backend)
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

    def forward(self, x: Tensor) -> Tensor:
        out = F.linear(x, self.weight, bias=None)
        out = _ReduceFromTPRegion.apply(out, self.group, self.backend)
        if self.bias is not None:
            out = out + self.bias
        return out
