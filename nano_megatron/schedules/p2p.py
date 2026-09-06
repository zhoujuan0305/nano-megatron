"""Pipeline-parallel P2P send/recv helpers.

Each helper wraps ``ctx.backend.send`` / ``ctx.backend.recv`` using the
pipeline-neighbor ranks computed by :mod:`nano_megatron.parallel.context`.

First/last stage semantics:

* ``recv_forward`` returns ``None`` on the first stage (no incoming
  activation to receive).
* ``send_forward`` is a no-op on the last stage (no downstream stage).
* ``recv_backward`` returns ``None`` on the last stage (no incoming
  gradient to receive).
* ``send_backward`` is a no-op on the first stage (no upstream stage).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from nano_megatron.distributed.backend import CommWork, P2POperation
from nano_megatron.parallel.context import (
    ParallelContext,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    pipeline_next_rank,
    pipeline_prev_rank,
)


@dataclass
class P2PRequest:
    """Own a P2P buffer until its asynchronous operation has completed."""

    tensor: Tensor
    work: CommWork | None
    _complete: bool = False

    def wait(self) -> Tensor:
        if not self._complete and self.work is not None:
            self.work.wait()
        self._complete = True
        return self.tensor


def recv_forward(
    ctx: ParallelContext,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor | None:
    """Receive an activation tensor from the previous PP stage.

    Returns ``None`` on the first stage where there is no previous stage.
    """
    if is_pipeline_first_stage(ctx):
        return None
    src = pipeline_prev_rank(ctx)
    if src is None:
        return None
    tensor = torch.empty(shape, dtype=dtype, device=device)
    return ctx.backend.recv(tensor, src=src, group=ctx.pipeline_parallel_group)


def send_forward(ctx: ParallelContext, tensor: Tensor) -> None:
    """Send an activation tensor to the next PP stage.

    No-op on the last stage.
    """
    if is_pipeline_last_stage(ctx):
        return
    dst = pipeline_next_rank(ctx)
    if dst is None:
        return
    ctx.backend.send(tensor, dst=dst, group=ctx.pipeline_parallel_group)


def recv_backward(
    ctx: ParallelContext,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor | None:
    """Receive a gradient tensor from the next PP stage.

    Returns ``None`` on the last stage where there is no next stage.
    """
    if is_pipeline_last_stage(ctx):
        return None
    src = pipeline_next_rank(ctx)
    if src is None:
        return None
    tensor = torch.empty(shape, dtype=dtype, device=device)
    return ctx.backend.recv(tensor, src=src, group=ctx.pipeline_parallel_group)


def send_backward(ctx: ParallelContext, tensor: Tensor) -> None:
    """Send a gradient tensor to the previous PP stage.

    No-op on the first stage.
    """
    if is_pipeline_first_stage(ctx):
        return
    dst = pipeline_prev_rank(ctx)
    if dst is None:
        return
    ctx.backend.send(tensor, dst=dst, group=ctx.pipeline_parallel_group)


def recv_forward_async(
    ctx: ParallelContext,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str,
) -> P2PRequest | None:
    src = pipeline_prev_rank(ctx)
    if src is None:
        return None
    tensor = torch.empty(shape, dtype=dtype, device=device)
    works = ctx.backend.batch_p2p(
        [
            P2POperation(
                "recv", tensor, peer=src, group=ctx.pipeline_parallel_group
            )
        ]
    )
    return P2PRequest(tensor, works[0])

def send_forward_async(
    ctx: ParallelContext, tensor: Tensor
) -> P2PRequest | None:
    dst = pipeline_next_rank(ctx)
    if dst is None:
        return None
    works = ctx.backend.batch_p2p(
        [
            P2POperation(
                "send", tensor, peer=dst, group=ctx.pipeline_parallel_group
            )
        ]
    )
    return P2PRequest(tensor, works[0])


def recv_backward_async(
    ctx: ParallelContext,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str,
) -> P2PRequest | None:
    src = pipeline_next_rank(ctx)
    if src is None:
        return None
    tensor = torch.empty(shape, dtype=dtype, device=device)
    works = ctx.backend.batch_p2p(
        [
            P2POperation(
                "recv", tensor, peer=src, group=ctx.pipeline_parallel_group
            )
        ]
    )
    return P2PRequest(tensor, works[0])


def send_backward_async(
    ctx: ParallelContext, tensor: Tensor
) -> P2PRequest | None:
    dst = pipeline_prev_rank(ctx)
    if dst is None:
        return None
    works = ctx.backend.batch_p2p(
        [
            P2POperation(
                "send", tensor, peer=dst, group=ctx.pipeline_parallel_group
            )
        ]
    )
    return P2PRequest(tensor, works[0])
