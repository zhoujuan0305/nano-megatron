from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F

from nano_megatron.distributed.backend import CommBackend

SEQ_DIM = 1


def _check_seq_divisible(seq_len: int, tp_size: int) -> None:
    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}")
    if seq_len % tp_size != 0:
        raise ValueError(
            f"sequence length ({seq_len}) not divisible by tp_size ({tp_size})"
        )


def _check_seq_tensor(x: Tensor) -> None:
    if x.dim() < 2:
        raise ValueError(
            f"sequence-parallel tensor must have dim >= 2 (layout [B, S, ...]), got dim={x.dim()}"
        )


def _all_gather_along_sequence(
    x: Tensor,
    *,
    group: Any,
    backend: CommBackend,
    group_size: int,
) -> Tensor:
    """Gather rank-major sequence shards without list staging."""
    x_sequence_first = x.movedim(SEQ_DIM, 0).contiguous()
    output_shape = list(x_sequence_first.shape)
    output_shape[0] *= group_size
    output = x_sequence_first.new_empty(output_shape)
    backend.all_gather_into_tensor(output, x_sequence_first, group=group)
    return output.movedim(0, SEQ_DIM)


def _reduce_scatter_along_sequence(
    x: Tensor,
    *,
    group: Any,
    backend: CommBackend,
    group_size: int,
    async_op: bool = False,
) -> tuple[Tensor, Any]:
    """Pack sequence chunks once and reduce-scatter the contiguous tensor."""
    _check_seq_divisible(x.size(SEQ_DIM), group_size)
    input_sequence_first = x.movedim(SEQ_DIM, 0).contiguous()
    output_shape = list(input_sequence_first.shape)
    output_shape[0] //= group_size
    output_sequence_first = input_sequence_first.new_empty(output_shape)
    result = backend.reduce_scatter_tensor(
        output_sequence_first,
        input_sequence_first,
        group=group,
        op="sum",
        async_op=async_op,
    )
    return output_sequence_first.movedim(0, SEQ_DIM), result


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


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split full sequence along dim=1; backward all-gathers grads."""

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        group: Any,
        backend: CommBackend,
        tp_rank: int,
        tp_size: int,
    ) -> Tensor:
        ctx.group = group
        ctx.backend = backend
        ctx.tp_rank = tp_rank
        ctx.tp_size = tp_size
        if tp_size == 1:
            return x
        _check_seq_tensor(x)
        seq = x.size(SEQ_DIM)
        _check_seq_divisible(seq, tp_size)
        chunk = seq // tp_size
        return x.narrow(SEQ_DIM, tp_rank * chunk, chunk).contiguous()

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> tuple[Tensor, None, None, None, None]:
        if ctx.tp_size == 1:
            return grad_output, None, None, None, None
        grad_input = _all_gather_along_sequence(
            grad_output,
            group=ctx.group,
            backend=ctx.backend,
            group_size=ctx.tp_size,
        )
        return grad_input, None, None, None, None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """All-gather local sequence shards; backward reduce-scatters grads."""

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        group: Any,
        backend: CommBackend,
        tp_rank: int,
        tp_size: int,
    ) -> Tensor:
        ctx.group = group
        ctx.backend = backend
        ctx.tp_rank = tp_rank
        ctx.tp_size = tp_size
        if tp_size == 1:
            return x
        _check_seq_tensor(x)
        return _all_gather_along_sequence(
            x, group=group, backend=backend, group_size=tp_size
        )

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> tuple[Tensor, None, None, None, None]:
        if ctx.tp_size == 1:
            return grad_output, None, None, None, None
        out, _ = _reduce_scatter_along_sequence(
            grad_output,
            group=ctx.group,
            backend=ctx.backend,
            group_size=ctx.tp_size,
        )
        return out, None, None, None, None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce-scatter full sequence to local shard; backward all-gathers grads."""

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        group: Any,
        backend: CommBackend,
        tp_rank: int,
        tp_size: int,
    ) -> Tensor:
        ctx.group = group
        ctx.backend = backend
        ctx.tp_rank = tp_rank
        ctx.tp_size = tp_size
        if tp_size == 1:
            return x
        _check_seq_tensor(x)
        out, _ = _reduce_scatter_along_sequence(
            x, group=group, backend=backend, group_size=tp_size
        )
        return out

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> tuple[Tensor, None, None, None, None]:
        if ctx.tp_size == 1:
            return grad_output, None, None, None, None
        grad_input = _all_gather_along_sequence(
            grad_output,
            group=ctx.group,
            backend=ctx.backend,
            group_size=ctx.tp_size,
        )
        return grad_input, None, None, None, None


_SEQUENCE_PARALLEL_GRAD_ATTR = "_nano_sequence_parallel_grad"


def mark_sequence_parallel_parameter(param: nn.Parameter) -> None:
    """Mark a replicated parameter whose gradient covers one sequence shard."""
    if param.requires_grad:
        setattr(param, _SEQUENCE_PARALLEL_GRAD_ATTR, True)


class SequenceParallelGradSynchronizer:
    """Coalesce replicated SP parameter gradients into one buffer per dtype.

    Gradients accumulate locally across all microbatches. The execution
    schedule calls :meth:`finish` after DP gradient communication completes,
    which is safe because the DP and TP reductions commute but must not access
    the same gradient storage concurrently.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        group: Any,
        backend: CommBackend,
        tp_size: int,
    ) -> None:
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}")
        self._params = [
            param
            for param in params
            if param.requires_grad
            and bool(getattr(param, _SEQUENCE_PARALLEL_GRAD_ATTR, False))
        ]
        self._group = group
        self._backend = backend
        self._tp_size = tp_size
        self._buffers: dict[
            tuple[torch.device, torch.dtype, tuple[int, ...]], Tensor
        ] = {}

    @property
    def num_parameters(self) -> int:
        return len(self._params)

    def finish(self) -> int:
        """Sum marked gradients over TP and return the collective count."""
        if self._tp_size == 1 or not self._params:
            return 0

        grouped: defaultdict[
            tuple[torch.device, torch.dtype], list[nn.Parameter]
        ] = defaultdict(list)
        for param in self._params:
            if param.grad is None:
                raise RuntimeError(
                    "sequence-parallel gradient is missing for parameter "
                    f"with shape={tuple(param.shape)} device={param.device} "
                    f"dtype={param.dtype}"
                )
            grouped[(param.grad.device, param.grad.dtype)].append(param)

        collectives = 0
        for (device, dtype), params in grouped.items():
            sizes = tuple(param.numel() for param in params)
            key = (device, dtype, sizes)
            total = sum(sizes)
            flat = self._buffers.get(key)
            if flat is None:
                flat = torch.empty(total, device=device, dtype=dtype)
                self._buffers[key] = flat

            offset = 0
            for param, size in zip(params, sizes):
                assert param.grad is not None
                flat[offset : offset + size].copy_(param.grad.reshape(-1))
                offset += size

            self._backend.all_reduce(flat, group=self._group, op="sum")
            collectives += 1

            offset = 0
            for param, size in zip(params, sizes):
                assert param.grad is not None
                param.grad.copy_(flat[offset : offset + size].view_as(param.grad))
                offset += size

        return collectives


def scatter_to_sequence_parallel_region(
    x: Tensor,
    group: Any,
    backend: CommBackend,
    tp_rank: int,
    tp_size: int,
) -> Tensor:
    return _ScatterToSequenceParallelRegion.apply(
        x, group, backend, tp_rank, tp_size
    )


def gather_from_sequence_parallel_region(
    x: Tensor,
    group: Any,
    backend: CommBackend,
    tp_rank: int,
    tp_size: int,
) -> Tensor:
    return _GatherFromSequenceParallelRegion.apply(
        x, group, backend, tp_rank, tp_size
    )


def reduce_scatter_to_sequence_parallel_region(
    x: Tensor,
    group: Any,
    backend: CommBackend,
    tp_rank: int,
    tp_size: int,
) -> Tensor:
    return _ReduceScatterToSequenceParallelRegion.apply(
        x, group, backend, tp_rank, tp_size
    )


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


class _LinearWithGradOverlap(torch.autograd.Function):
    """Column-parallel linear with dgrad communication hidden by wgrad GEMM."""

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        group: Any,
        backend: CommBackend,
        tp_size: int,
        sequence_parallel: bool,
    ) -> Tensor:
        if sequence_parallel and tp_size > 1:
            linear_input = _all_gather_along_sequence(
                x, group=group, backend=backend, group_size=tp_size
            )
        else:
            linear_input = x

        ctx.save_for_backward(linear_input, weight)
        ctx.group = group
        ctx.backend = backend
        ctx.tp_size = tp_size
        ctx.sequence_parallel = sequence_parallel
        ctx.has_bias = bias is not None
        return F.linear(linear_input, weight, bias)

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None, None, None, None, None]:
        linear_input, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input_full = grad_output.matmul(weight)

        work = None
        if ctx.tp_size > 1 and ctx.sequence_parallel:
            grad_input, work = _reduce_scatter_along_sequence(
                grad_input_full,
                group=ctx.group,
                backend=ctx.backend,
                group_size=ctx.tp_size,
                async_op=True,
            )
        elif ctx.tp_size > 1:
            grad_input = grad_input_full
            work = ctx.backend.all_reduce(
                grad_input, group=ctx.group, op="sum", async_op=True
            )
        else:
            grad_input = grad_input_full

        grad_output_2d = grad_output.reshape(-1, grad_output.size(-1))
        input_2d = linear_input.reshape(-1, linear_input.size(-1))
        grad_weight = grad_output_2d.t().matmul(input_2d)
        grad_bias = grad_output_2d.sum(dim=0) if ctx.has_bias else None

        if hasattr(work, "wait"):
            work.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None


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
        sequence_parallel: bool = False,
        overlap_dgrad: bool = False,
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
        self.sequence_parallel = sequence_parallel
        self.overlap_dgrad = overlap_dgrad

    def forward(self, x: Tensor) -> Tensor:
        if self.overlap_dgrad:
            return _LinearWithGradOverlap.apply(
                x,
                self.weight,
                self.bias,
                self.group,
                self.backend,
                self.tp_size,
                self.sequence_parallel,
            )
        # SP path: input is sequence-sharded; gather full S then matmul.
        # No CopyToTPRegion — grad reduce-scatter is handled by the gather op.
        if self.sequence_parallel:
            x = gather_from_sequence_parallel_region(
                x, self.group, self.backend, self.tp_rank, self.tp_size
            )
        else:
            x = _CopyToTPRegion.apply(
                x, self.group, self.backend, self.buffer_manager
            )
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
        *,
        sequence_parallel: bool = False,
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
        self.sequence_parallel = sequence_parallel
        # Bias is added after reduce-scatter on the local sequence shard, so
        # its grad is partial and must be summed across the TP group.
        if sequence_parallel and self.bias is not None:
            mark_sequence_parallel_parameter(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        out = F.linear(x, self.weight, bias=None)
        # SP path: reduce-scatter partial outputs to local sequence shard.
        # Bias is full and applied after the collective (Megatron-style).
        if self.sequence_parallel:
            out = reduce_scatter_to_sequence_parallel_region(
                out, self.group, self.backend, self.tp_rank, self.tp_size
            )
        else:
            out = _ReduceFromTPRegion.apply(
                out, self.group, self.backend, self.buffer_manager
            )
        if self.bias is not None:
            out = out + self.bias
        return out
