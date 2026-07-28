"""Context Parallel (CP) sequence operators and causal mask helper.

Provides scatter/gather operators that split/concatenate along an arbitrary
sequence dimension for context-parallel training, plus a CP-aware causal
attention score function that masks against the *full* key sequence.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from nano_megatron.distributed.backend import CommBackend


# ---------------------------------------------------------------------------
# Sequence range helper
# ---------------------------------------------------------------------------


def local_sequence_range(cp_rank: int, cp_size: int, seq_len: int) -> tuple[int, int]:
    """Return ``(start, end)`` for *cp_rank*'s shard of a sequence of *seq_len*.

    Raises :class:`ValueError` when *seq_len* is not divisible by *cp_size*,
    or when *cp_rank* is out of ``[0, cp_size)``.
    """
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(
            f"cp_rank must be in [0, {cp_size}), got {cp_rank}"
        )
    if seq_len % cp_size != 0:
        raise ValueError(
            f"sequence length ({seq_len}) not divisible by cp_size ({cp_size})"
        )
    chunk = seq_len // cp_size
    start = cp_rank * chunk
    return (start, start + chunk)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_seq_divisible(seq_len: int, cp_size: int) -> None:
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")
    if seq_len % cp_size != 0:
        raise ValueError(
            f"sequence length ({seq_len}) not divisible by cp_size ({cp_size})"
        )


def _check_seq_tensor(x: Tensor, seq_dim: int) -> None:
    if x.dim() < 2:
        raise ValueError(
            f"CP tensor must have dim >= 2, got dim={x.dim()}"
        )
    if seq_dim < 0 or seq_dim >= x.dim():
        raise ValueError(
            f"seq_dim ({seq_dim}) out of range for {x.dim()}-dim tensor"
        )


# ---------------------------------------------------------------------------
# Autograd-aware scatter / gather
# ---------------------------------------------------------------------------


class _ScatterToContextParallelRegion(torch.autograd.Function):
    """Narrow full sequence to local CP shard; backward pads zeros (no collective).

    Each CP rank typically embeds the full sequence then narrows.  The reverse
    of narrow is a zero-padded full-sequence gradient with the local shard
    written in place — not an all-gather (which would incorrectly sum peer
    shards into every rank's embed grad).
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        group: Any,
        backend: CommBackend,
        cp_rank: int,
        cp_size: int,
        seq_dim: int,
    ) -> Tensor:
        ctx.group = group
        ctx.backend = backend
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.seq_dim = seq_dim
        if cp_size == 1:
            return x
        _check_seq_tensor(x, seq_dim)
        seq = x.size(seq_dim)
        _check_seq_divisible(seq, cp_size)
        chunk = seq // cp_size
        ctx.full_seq_len = seq
        return x.narrow(seq_dim, cp_rank * chunk, chunk).contiguous()

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> tuple[Tensor, None, None, None, None, None]:
        if ctx.cp_size == 1:
            return grad_output, None, None, None, None, None
        # Pad zeros: reverse of narrow.  No collective — each rank only owns
        # the gradient for its local sequence shard of the full embed.
        full_shape = list(grad_output.shape)
        full_shape[ctx.seq_dim] = ctx.full_seq_len
        grad_input = grad_output.new_zeros(full_shape)
        chunk = ctx.full_seq_len // ctx.cp_size
        grad_input.narrow(
            ctx.seq_dim, ctx.cp_rank * chunk, chunk
        ).copy_(grad_output)
        return grad_input, None, None, None, None, None


class _GatherFromContextParallelRegion(torch.autograd.Function):
    """All-gather local CP shards; backward reduce-scatter or split."""

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        group: Any,
        backend: CommBackend,
        cp_rank: int,
        cp_size: int,
        seq_dim: int,
        grad_op: str,
    ) -> Tensor:
        ctx.group = group
        ctx.backend = backend
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.seq_dim = seq_dim
        ctx.grad_op = grad_op
        if cp_size == 1:
            return x
        _check_seq_tensor(x, seq_dim)
        # Single output buffer: gather along dim 0 then restore seq_dim.
        # Avoids list all_gather + torch.cat D2D copies.
        x_in = x.movedim(seq_dim, 0).contiguous()
        out_shape = list(x_in.shape)
        out_shape[0] *= cp_size
        out = x_in.new_empty(out_shape)
        backend.all_gather_into_tensor(out, x_in, group=group)
        return out.movedim(0, seq_dim)

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> tuple[Tensor, None, None, None, None, None, None]:
        if ctx.cp_size == 1:
            return grad_output, None, None, None, None, None, None
        _check_seq_divisible(grad_output.size(ctx.seq_dim), ctx.cp_size)
        if ctx.grad_op == "split":
            # Identical full-sequence consumers (e.g. global-mean CE on every
            # CP rank): dL/d(full) is the same on all ranks; each rank only
            # needs its local shard — no cross-rank sum.
            chunk = grad_output.size(ctx.seq_dim) // ctx.cp_size
            out = grad_output.narrow(
                ctx.seq_dim, ctx.cp_rank * chunk, chunk
            ).contiguous()
            return out, None, None, None, None, None, None
        # Default: partial contributions from different consumers (KV all-gather
        # in attention) — sum shards via reduce-scatter.
        chunks = [
            c.contiguous()
            for c in grad_output.chunk(ctx.cp_size, dim=ctx.seq_dim)
        ]
        out = torch.empty_like(chunks[0])
        ctx.backend.reduce_scatter(out, chunks, group=ctx.group, op="sum")
        return out, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------


def scatter_to_context_parallel_region(
    x: Tensor,
    group: Any,
    backend: CommBackend,
    cp_rank: int,
    cp_size: int,
    *,
    seq_dim: int = 1,
) -> Tensor:
    """Scatter full-sequence *x* to the local CP shard along *seq_dim*.

    Forward: narrow.  Backward: pad-zeros into the full sequence (no collective).
    """
    return _ScatterToContextParallelRegion.apply(
        x, group, backend, cp_rank, cp_size, seq_dim
    )


def gather_from_context_parallel_region(
    x: Tensor,
    group: Any,
    backend: CommBackend,
    cp_rank: int,
    cp_size: int,
    *,
    seq_dim: int = 1,
    grad_op: str = "reduce_scatter",
) -> Tensor:
    """All-gather local CP shards into the full sequence along *seq_dim*.

    Forward: all-gather.

    Backward depends on *grad_op*:

    * ``"reduce_scatter"`` (default): sum shards via reduce-scatter.  Correct
      when each CP rank holds a *partial* contribution to the full-sequence
      gradient (e.g. KV all-gather in attention).
    * ``"split"``: narrow ``grad_output`` to the local shard with no
      cross-rank sum.  Correct when every CP rank computed the *same*
      full-sequence gradient (e.g. global-mean CE on gathered logits).
    """
    if grad_op not in ("reduce_scatter", "split"):
        raise ValueError(
            f"grad_op must be 'reduce_scatter' or 'split', got {grad_op!r}"
        )
    return _GatherFromContextParallelRegion.apply(
        x, group, backend, cp_rank, cp_size, seq_dim, grad_op
    )


# ---------------------------------------------------------------------------
# CP-aware causal attention scores
# ---------------------------------------------------------------------------


def causal_attn_scores_cp(
    q: Tensor,
    k_full: Tensor,
    scale: float,
    query_start: int,
) -> Tensor:
    """Compute causal attention scores for a local CP query shard.

    *q* has shape ``[B, H, S_local, D]`` and represents the query shard
    starting at position *query_start* in the full sequence.
    *k_full* has shape ``[B, H, S_full, D]`` and is the **full** key tensor
    (all CP ranks share keys in this layout).

    Returns ``[B, H, S_local, S_full]`` with causal masking applied so that
    each query position ``query_start + i`` may only attend to key positions
    ``<= query_start + i``.
    """
    # [B, H, S_local, S_full]
    scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale
    s_local = q.size(2)
    s_full = k_full.size(2)

    query_pos = torch.arange(query_start, query_start + s_local, device=scores.device).unsqueeze(1)
    key_pos = torch.arange(s_full, device=scores.device).unsqueeze(0)
    causal_mask = key_pos > query_pos  # True where masked

    # Use dtype min (not Python float -inf) so bf16/fp16 scores stay in-dtype.
    neg = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), neg)
    return scores
