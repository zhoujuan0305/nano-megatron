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
    """Own P2P buffers until all operations in one batch have completed."""

    tensor: Tensor
    works: tuple[CommWork, ...]
    retained_tensors: tuple[Tensor, ...] = ()
    _complete: bool = False

    def wait(self) -> Tensor:
        if not self._complete:
            for work in self.works:
                work.wait()
        self._complete = True
        self.retained_tensors = ()
        return self.tensor


def _make_request(
    tensor: Tensor,
    works: list[CommWork],
    *,
    retained_tensors: tuple[Tensor, ...] = (),
) -> P2PRequest:
    # Backends may return one aggregate handle for the whole batch or one
    # handle per operation. In either case, waiting every returned handle is
    # sufficient to make all submitted operations complete.
    if not works:
        raise RuntimeError("batch_p2p returned no completion handles")
    return P2PRequest(
        tensor=tensor,
        works=tuple(works),
        retained_tensors=retained_tensors,
    )


def warmup_pipeline_p2p(
    ctx: ParallelContext,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> None:
    """Initialize a two-stage P2P communicator with a matched exchange.

    NCCL requires every rank in a newly created P2P communicator to enter its
    first batched operation. The real overlap schedule has an asymmetric first
    batch, so perform one tiny symmetric exchange once per parallel context.
    """
    if ctx.pipeline_parallel_size == 1:
        return
    if ctx.pipeline_parallel_size != 2:
        raise ValueError(
            "P2P warmup currently supports pipeline_parallel_size <= 2"
        )
    if getattr(ctx, "_pipeline_p2p_warmed", False):
        return

    peer = (
        pipeline_next_rank(ctx)
        if ctx.pipeline_parallel_rank == 0
        else pipeline_prev_rank(ctx)
    )
    assert peer is not None
    send_token = torch.zeros(1, dtype=dtype, device=device)
    recv_token = torch.empty_like(send_token)
    if ctx.pipeline_parallel_rank == 0:
        tensors = (send_token, recv_token)
        operations = (
            P2POperation(
                "send", send_token, peer=peer, group=ctx.pipeline_parallel_group
            ),
            P2POperation(
                "recv", recv_token, peer=peer, group=ctx.pipeline_parallel_group
            ),
        )
    else:
        tensors = (recv_token, send_token)
        operations = (
            P2POperation(
                "recv", recv_token, peer=peer, group=ctx.pipeline_parallel_group
            ),
            P2POperation(
                "send", send_token, peer=peer, group=ctx.pipeline_parallel_group
            ),
        )
    works = ctx.backend.batch_p2p(list(operations))
    _make_request(
        recv_token,
        works,
        retained_tensors=tensors,
    ).wait()
    setattr(ctx, "_pipeline_p2p_warmed", True)


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
    return _make_request(tensor, works)


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
    return _make_request(
        tensor,
        works,
        retained_tensors=(tensor,),
    )


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
    return _make_request(tensor, works)


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
    return _make_request(
        tensor,
        works,
        retained_tensors=(tensor,),
    )


def send_forward_recv_backward_async(
    ctx: ParallelContext,
    tensor: Tensor,
) -> P2PRequest | None:
    """Send activation and receive its gradient in one NCCL P2P batch."""
    peer = pipeline_next_rank(ctx)
    if peer is None:
        return None
    grad = torch.empty_like(tensor)
    works = ctx.backend.batch_p2p(
        [
            P2POperation(
                "send", tensor, peer=peer, group=ctx.pipeline_parallel_group
            ),
            P2POperation(
                "recv", grad, peer=peer, group=ctx.pipeline_parallel_group
            ),
        ]
    )
    return _make_request(
        grad,
        works,
        retained_tensors=(tensor,),
    )


def send_backward_recv_forward_async(
    ctx: ParallelContext,
    tensor: Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str,
) -> P2PRequest | None:
    """Send input gradient and receive the next activation in one P2P batch."""
    peer = pipeline_prev_rank(ctx)
    if peer is None:
        return None
    activation = torch.empty(shape, dtype=dtype, device=device)
    works = ctx.backend.batch_p2p(
        [
            P2POperation(
                "send", tensor, peer=peer, group=ctx.pipeline_parallel_group
            ),
            P2POperation(
                "recv", activation, peer=peer, group=ctx.pipeline_parallel_group
            ),
        ]
    )
    return _make_request(
        activation,
        works,
        retained_tensors=(tensor,),
    )


def send_forward_recv_backward(
    ctx: ParallelContext,
    tensor: Tensor,
) -> Tensor | None:
    request = send_forward_recv_backward_async(ctx, tensor)
    return None if request is None else request.wait()


def send_backward_recv_forward(
    ctx: ParallelContext,
    tensor: Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str,
) -> Tensor | None:
    request = send_backward_recv_forward_async(
        ctx,
        tensor,
        shape=shape,
        dtype=dtype,
        device=device,
    )
    return None if request is None else request.wait()
