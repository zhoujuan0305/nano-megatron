"""Non-interleaved 1F1B pipeline schedule (Megatron-style).

Control flow mirrors Megatron
``forward_backward_pipelining_without_interleaving``:

1. Warmup forward passes (depth depends on PP rank)
2. Steady 1F1B (forward next, backward oldest)
3. Cooldown backward passes

P2P uses blocking ``send``/``recv``. To avoid deadlock between adjacent
stages that exchange activation and gradient in the same steady-state
step, communication with the *next* stage is ordered as recv-grad then
send-activation, and with the *prev* stage as send-grad then
recv-activation (peer pairing under blocking collectives).
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from nano_megatron.parallel.context import (
    ParallelContext,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from nano_megatron.schedules.p2p import (
    recv_backward,
    recv_forward,
    send_backward,
    send_forward,
)

if TYPE_CHECKING:
    from nano_megatron.distributed.ddp import DistributedDataParallel
    from nano_megatron.model.pipeline import PipelineStage


def warmup_microbatches(pp_size: int, pp_rank: int, num_microbatches: int) -> int:
    """Number of warmup forward microbatches for this PP rank."""
    if pp_size < 1:
        raise ValueError(f"pp_size must be >= 1, got {pp_size}")
    if not (0 <= pp_rank < pp_size):
        raise ValueError(f"pp_rank ({pp_rank}) out of range for pp_size={pp_size}")
    if num_microbatches < 1:
        raise ValueError(f"num_microbatches must be >= 1, got {num_microbatches}")
    return min(pp_size - pp_rank - 1, num_microbatches)


def _activation_shape(
    micro_batch_size: int, seq_len: int, hidden_size: int
) -> tuple[int, int, int]:
    return (micro_batch_size, seq_len, hidden_size)


def _slice_microbatch(
    tensor: Tensor | None, mb_idx: int, micro_batch_size: int
) -> Tensor | None:
    if tensor is None:
        return None
    start = mb_idx * micro_batch_size
    end = start + micro_batch_size
    return tensor[start:end]


def forward_backward_1f1b(
    *,
    stage: PipelineStage,
    ctx: ParallelContext,
    input_ids: Tensor,
    labels: Tensor,
    positions: Tensor | None = None,
    num_microbatches: int,
    ddp: DistributedDataParallel | None = None,
) -> Tensor | None:
    """Run non-interleaved 1F1B over ``num_microbatches`` microbatches.

    Splits the batch dimension of ``input_ids`` / ``labels`` into equal
    chunks. Returns the mean loss on the last stage; ``None`` elsewhere.
    When ``ddp`` is set, every backward runs under ``ddp.no_sync()`` and
    ``ddp.finish_grad_sync()`` is called before return.
    """
    if num_microbatches < 1:
        raise ValueError(f"num_microbatches must be >= 1, got {num_microbatches}")
    if input_ids.dim() != 2:
        raise ValueError(
            f"input_ids must be [B, S], got shape {tuple(input_ids.shape)}"
        )
    batch_size = input_ids.size(0)
    if batch_size % num_microbatches != 0:
        raise ValueError(
            f"input_ids.size(0) ({batch_size}) must be divisible by "
            f"num_microbatches ({num_microbatches})"
        )
    if labels.shape != input_ids.shape:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must match "
            f"input_ids shape {tuple(input_ids.shape)}"
        )
    if positions is not None and positions.shape != input_ids.shape:
        raise ValueError(
            f"positions shape {tuple(positions.shape)} must match "
            f"input_ids shape {tuple(input_ids.shape)}"
        )

    micro_batch_size = batch_size // num_microbatches
    seq_len = input_ids.size(1)
    hidden_size = stage.config.hidden_size
    act_shape = _activation_shape(micro_batch_size, seq_len, hidden_size)
    # Inter-stage dtype follows stage parameters (FP32 in v1).
    try:
        param_dtype = next(stage.parameters()).dtype
        device = next(stage.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("pipeline stage has no parameters") from exc

    pp_size = ctx.pipeline_parallel_size
    pp_rank = ctx.pipeline_parallel_rank
    is_first = is_pipeline_first_stage(ctx)
    is_last = is_pipeline_last_stage(ctx)

    num_warmup = warmup_microbatches(pp_size, pp_rank, num_microbatches)
    num_remaining = num_microbatches - num_warmup

    # Queues of tensors retained until that microbatch's backward.
    input_tensors: list[Tensor] = []
    output_tensors: list[Tensor] = []
    losses: list[Tensor] = []

    def _forward_step(mb_idx: int, input_tensor: Tensor | None) -> Tensor:
        """One microbatch forward. ``input_tensor`` is tokens (first) or act."""
        pos_mb = _slice_microbatch(positions, mb_idx, micro_batch_size)
        if is_first:
            assert input_tensor is not None
            output = stage(input_tensor, positions=pos_mb)
        else:
            assert input_tensor is not None
            # Recv'd activations are leaves; enable grad for send_backward.
            if not input_tensor.requires_grad:
                input_tensor.requires_grad_(True)
            output = stage(input_tensor, positions=pos_mb)

        if is_last:
            labels_mb = _slice_microbatch(labels, mb_idx, micro_batch_size)
            assert labels_mb is not None
            loss = stage.shifted_cross_entropy(output, labels_mb) / num_microbatches
            losses.append(loss)
            return loss
        return output

    def _recv_input(mb_idx: int) -> Tensor:
        if is_first:
            ids_mb = _slice_microbatch(input_ids, mb_idx, micro_batch_size)
            assert ids_mb is not None
            return ids_mb
        recv = recv_forward(
            ctx, shape=act_shape, dtype=param_dtype, device=device
        )
        assert recv is not None
        return recv

    def _backward_step(
        input_tensor: Tensor,
        output_tensor: Tensor,
        output_tensor_grad: Tensor | None,
    ) -> Tensor | None:
        """Backward one stored microbatch; return grad w.r.t. input (or None)."""
        sync_ctx = ddp.no_sync() if ddp is not None else nullcontext()
        with sync_ctx:
            if is_last:
                # output_tensor is the scaled microbatch loss.
                output_tensor.backward()
            else:
                assert output_tensor_grad is not None
                torch.autograd.backward(
                    tensors=output_tensor, grad_tensors=output_tensor_grad
                )

        if is_first:
            return None
        grad_input = input_tensor.grad
        return grad_input

    def _send_forward_recv_backward(output_tensor: Tensor) -> Tensor | None:
        """Peer-safe exchange with next stage (blocking send/recv).

        Order: recv grad from next, then send activation to next. Pairs with
        the peer's send-grad-then-recv-activation ordering.
        """
        if is_last:
            return None
        # Shape of grad matches the activation we send.
        grad = recv_backward(
            ctx,
            shape=tuple(output_tensor.shape),
            dtype=output_tensor.dtype,
            device=output_tensor.device,
        )
        send_forward(ctx, output_tensor)
        return grad

    def _send_backward_recv_forward(
        input_tensor_grad: Tensor | None, *, recv_next_forward: bool
    ) -> Tensor | None:
        """Peer-safe exchange with previous stage."""
        if not is_first:
            assert input_tensor_grad is not None
            send_backward(ctx, input_tensor_grad)
        if not recv_next_forward or is_first:
            # First stage loads tokens locally; caller handles that.
            return None
        recv = recv_forward(
            ctx, shape=act_shape, dtype=param_dtype, device=device
        )
        return recv

    # ------------------------------------------------------------------
    # Warmup forwards
    # ------------------------------------------------------------------
    for i in range(num_warmup):
        input_tensor = _recv_input(i)
        output_tensor = _forward_step(i, input_tensor)
        if not is_last:
            send_forward(ctx, output_tensor)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    # Prefetch first steady-state input (Megatron pattern).
    if num_remaining > 0:
        if is_first:
            next_input: Tensor | None = _slice_microbatch(
                input_ids, num_warmup, micro_batch_size
            )
        else:
            next_input = recv_forward(
                ctx, shape=act_shape, dtype=param_dtype, device=device
            )
    else:
        next_input = None

    # ------------------------------------------------------------------
    # Steady 1F1B
    # ------------------------------------------------------------------
    for i in range(num_remaining):
        last_iteration = i == (num_remaining - 1)
        mb_idx = i + num_warmup

        input_tensor = next_input
        assert input_tensor is not None or is_first
        if is_first and input_tensor is None:
            input_tensor = _slice_microbatch(input_ids, mb_idx, micro_batch_size)
        assert input_tensor is not None

        output_tensor = _forward_step(mb_idx, input_tensor)

        if is_last:
            output_tensor_grad = None
        else:
            output_tensor_grad = _send_forward_recv_backward(output_tensor)

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

        # Backward oldest pending microbatch.
        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)

        input_tensor_grad = _backward_step(
            input_tensor, output_tensor, output_tensor_grad
        )

        if last_iteration:
            if not is_first and input_tensor_grad is not None:
                send_backward(ctx, input_tensor_grad)
            next_input = None
        else:
            if is_first:
                next_mb = mb_idx + 1
                next_input = _slice_microbatch(
                    input_ids, next_mb, micro_batch_size
                )
                if input_tensor_grad is not None:
                    # first stage: no send_backward
                    pass
            else:
                next_input = _send_backward_recv_forward(
                    input_tensor_grad, recv_next_forward=True
                )

    # ------------------------------------------------------------------
    # Cooldown backwards
    # ------------------------------------------------------------------
    for _ in range(num_warmup):
        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)

        if is_last:
            output_tensor_grad = None
        else:
            output_tensor_grad = recv_backward(
                ctx,
                shape=tuple(output_tensor.shape),
                dtype=output_tensor.dtype,
                device=output_tensor.device,
            )

        input_tensor_grad = _backward_step(
            input_tensor, output_tensor, output_tensor_grad
        )
        if not is_first and input_tensor_grad is not None:
            send_backward(ctx, input_tensor_grad)

    if ddp is not None:
        ddp.finish_grad_sync()

    if not is_last:
        return None
    # Each loss already scaled by 1/M; sum == mean over microbatches.
    assert losses, "last stage produced no microbatch losses"
    return torch.stack(losses).sum()
