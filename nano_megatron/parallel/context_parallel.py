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
    """Scatter full sequence to local CP shard; backward pads zeros (no collective).

    Each CP rank typically embeds the full sequence then selects its shard.
    The reverse is a zero-padded full-sequence gradient with the local shard
    written in place — not an all-gather (which would incorrectly sum peer
    shards into every rank's embed grad).

    *pack*:
      - ``"contiguous"``: narrow to a contiguous chunk (legacy).
      - ``"zigzag"``: ``index_select`` with DualChunkSwap local indices.
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
        pack: str,
    ) -> Tensor:
        ctx.group = group
        ctx.backend = backend
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.seq_dim = seq_dim
        ctx.pack = pack
        if cp_size == 1:
            return x
        _check_seq_tensor(x, seq_dim)
        seq = x.size(seq_dim)
        ctx.full_seq_len = seq
        if pack == "zigzag":
            # DualChunkSwap: select non-contiguous halves owned by this rank.
            indices = zigzag_local_token_indices(
                cp_rank, cp_size, seq, device=x.device
            )
            ctx.zigzag_indices = indices
            return x.index_select(seq_dim, indices).contiguous()
        # contiguous (default): narrow to a contiguous chunk.
        _check_seq_divisible(seq, cp_size)
        chunk = seq // cp_size
        return x.narrow(seq_dim, cp_rank * chunk, chunk).contiguous()

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> tuple[Tensor, None, None, None, None, None, None]:
        if ctx.cp_size == 1:
            return grad_output, None, None, None, None, None, None
        # Pad zeros: reverse of narrow / index_select.  No collective.
        full_shape = list(grad_output.shape)
        full_shape[ctx.seq_dim] = ctx.full_seq_len
        grad_input = grad_output.new_zeros(full_shape)
        if ctx.pack == "zigzag":
            grad_input.index_copy_(ctx.seq_dim, ctx.zigzag_indices, grad_output)
        else:
            chunk = ctx.full_seq_len // ctx.cp_size
            grad_input.narrow(
                ctx.seq_dim, ctx.cp_rank * chunk, chunk
            ).copy_(grad_output)
        return grad_input, None, None, None, None, None, None


class _GatherFromContextParallelRegion(torch.autograd.Function):
    """All-gather local CP shards; backward reduce-scatter or split.

    Forward always returns **AG rank-concat** layout (rank0 local || rank1 || …).
    For ``pack="zigzag"`` the locals are DualChunkSwap shards, so AG is *not*
    global token order — callers that need global order (tests, unfused attn)
    must apply :func:`unpermute_ag_to_global`.  The flash zigzag path keeps AG
    layout and never unpermutes.

    Backward is pack-independent: grads are assumed to match the forward AG
    layout (chunk + RS, or local narrow for ``grad_op="split"``).
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
        grad_op: str,
        pack: str,
    ) -> Tensor:
        ctx.group = group
        ctx.backend = backend
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.seq_dim = seq_dim
        ctx.grad_op = grad_op
        ctx.pack = pack
        if cp_size == 1:
            return x
        _check_seq_tensor(x, seq_dim)
        if pack == "zigzag":
            # Locals are two half-chunks; keep the check so bad shapes fail early.
            local_seq = x.size(seq_dim)
            if local_seq % 2 != 0:
                raise ValueError(
                    f"zigzag gather requires even local sequence length, "
                    f"got local_seq={local_seq}"
                )
        # Single output buffer: gather along dim 0 then restore seq_dim.
        # Avoids list all_gather + torch.cat D2D copies.  No unpermute.
        x_in = x.movedim(seq_dim, 0).contiguous()
        out_shape = list(x_in.shape)
        out_shape[0] *= cp_size
        out = x_in.new_empty(out_shape)
        backend.all_gather_into_tensor(out, x_in, group=group)
        return out.movedim(0, seq_dim)

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> tuple[Tensor, None, None, None, None, None, None, None]:
        if ctx.cp_size == 1:
            return grad_output, None, None, None, None, None, None, None
        _check_seq_divisible(grad_output.size(ctx.seq_dim), ctx.cp_size)
        if ctx.grad_op == "split":
            # Identical full-sequence consumers (e.g. global-mean CE on every
            # CP rank): dL/d(full) is the same on all ranks; each rank only
            # needs its local AG shard — no cross-rank sum.
            chunk = grad_output.size(ctx.seq_dim) // ctx.cp_size
            out = grad_output.narrow(
                ctx.seq_dim, ctx.cp_rank * chunk, chunk
            ).contiguous()
            return out, None, None, None, None, None, None, None
        # Default: partial contributions from different consumers (KV all-gather
        # in attention) — sum AG shards via reduce-scatter.  Zigzag FA writes
        # dk/dv at AG offsets, so no inv-permute is required.
        chunks = [
            c.contiguous()
            for c in grad_output.chunk(ctx.cp_size, dim=ctx.seq_dim)
        ]
        out = torch.empty_like(chunks[0])
        ctx.backend.reduce_scatter(out, chunks, group=ctx.group, op="sum")
        return out, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------


_VALID_CP_PACK = frozenset({"contiguous", "zigzag"})


def _check_pack(pack: str) -> None:
    if pack not in _VALID_CP_PACK:
        raise ValueError(
            f"pack must be 'contiguous' or 'zigzag', got {pack!r}"
        )


def scatter_to_context_parallel_region(
    x: Tensor,
    group: Any,
    backend: CommBackend,
    cp_rank: int,
    cp_size: int,
    *,
    seq_dim: int = 1,
    pack: str = "contiguous",
) -> Tensor:
    """Scatter full-sequence *x* to the local CP shard along *seq_dim*.

    Forward: narrow (``pack="contiguous"``) or zigzag ``index_select``
    (``pack="zigzag"``).  Backward: pad-zeros into the full sequence
    (no collective).
    """
    _check_pack(pack)
    return _ScatterToContextParallelRegion.apply(
        x, group, backend, cp_rank, cp_size, seq_dim, pack
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
    pack: str = "contiguous",
) -> Tensor:
    """All-gather local CP shards into the full sequence along *seq_dim*.

    Forward: all-gather only.  Output is **AG rank-concat** layout for both
    ``pack="contiguous"`` and ``pack="zigzag"``.  Contiguous locals yield
    global order; zigzag DualChunkSwap locals yield AG layout — use
    :func:`unpermute_ag_to_global` when global order is required (tests /
    unfused).  Flash zigzag keeps AG layout end-to-end.

    Backward depends on *grad_op* (pack-independent; grads match AG layout):

    * ``"reduce_scatter"`` (default): sum shards via reduce-scatter.  Correct
      when each CP rank holds a *partial* contribution to the full-sequence
      gradient (e.g. KV all-gather in attention).
    * ``"split"``: take the local AG shard with no cross-rank sum.  Correct
      when every CP rank computed the *same* full-sequence gradient (e.g.
      global-mean CE on gathered logits).
    """
    if grad_op not in ("reduce_scatter", "split"):
        raise ValueError(
            f"grad_op must be 'reduce_scatter' or 'split', got {grad_op!r}"
        )
    _check_pack(pack)
    return _GatherFromContextParallelRegion.apply(
        x, group, backend, cp_rank, cp_size, seq_dim, grad_op, pack
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


# ---------------------------------------------------------------------------
# Zigzag (DualChunkSwap) index helpers
# ---------------------------------------------------------------------------
# Megatron-compatible zigzag layout: split sequence into 2*cp half-chunks
# of length H = S / (2*cp).  Rank r owns half-chunks (r, 2*cp-1-r),
# concatenated as local sequence of length S/cp.
#
# Example CP2 S=8 (H=2):
#   half-chunks: 0:[0,1] 1:[2,3] 2:[4,5] 3:[6,7]
#   rank 0 owns halves (0,3) → local [0,1,6,7]
#   rank 1 owns halves (1,2) → local [2,3,4,5]
# ---------------------------------------------------------------------------


def _check_cp_valid(cp_rank: int, cp_size: int) -> None:
    """Validate cp_rank and cp_size for zigzag helpers."""
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(
            f"cp_rank must be in [0, {cp_size}), got {cp_rank}"
        )


def zigzag_half_ids(cp_rank: int, cp_size: int) -> tuple[int, int]:
    """Return the two half-chunk ids owned by *cp_rank*.

    Megatron layout: rank *r* owns half-chunks ``r`` and ``2*cp_size - 1 - r``.

    Returns ``(first_half_id, second_half_id)``.
    """
    _check_cp_valid(cp_rank, cp_size)
    return (cp_rank, 2 * cp_size - 1 - cp_rank)


def zigzag_local_token_indices(
    cp_rank: int,
    cp_size: int,
    seq_len: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.long,
) -> Tensor:
    """Return global token indices owned by *cp_rank* in zigzag layout.

    Returns a 1-D LongTensor of shape ``[seq_len // cp_size]`` containing
    the global indices of tokens assigned to *cp_rank*.

    Requires ``seq_len % (2 * cp_size) == 0``.
    """
    _check_cp_valid(cp_rank, cp_size)
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")
    if seq_len % (2 * cp_size) != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by 2 * cp_size "
            f"({2 * cp_size}) for zigzag layout"
        )
    H = seq_len // (2 * cp_size)
    first, second = zigzag_half_ids(cp_rank, cp_size)
    first_indices = torch.arange(first * H, (first + 1) * H, device=device, dtype=dtype)
    second_indices = torch.arange(second * H, (second + 1) * H, device=device, dtype=dtype)
    return torch.cat([first_indices, second_indices])


def zigzag_half_ag_token_start(
    half_id: int, cp_size: int, half_len: int
) -> int:
    """Return the AG rank-concat token start index for global half *half_id*.

    After zigzag all-gather, tokens are concatenated in rank order.  Rank *r*
    contributes ``[half_r ; half_{2*cp-1-r}]`` of length ``2 * half_len``.
    Global half ``g`` therefore sits at:

    * ``g < cp_size``: rank ``g``'s first half → ``g * 2 * half_len``
    * ``g >= cp_size``: rank ``2*cp-1-g``'s second half →
      ``(2*cp-1-g) * 2 * half_len + half_len``

    Requires ``0 <= half_id < 2 * cp_size`` and ``half_len >= 1``.
    """
    _check_cp_valid(0, cp_size)
    if half_len < 1:
        raise ValueError(f"half_len must be >= 1, got {half_len}")
    n_halves = 2 * cp_size
    if half_id < 0 or half_id >= n_halves:
        raise ValueError(
            f"half_id must be in [0, {n_halves}), got {half_id}"
        )
    loc_len = 2 * half_len
    if half_id < cp_size:
        return int(half_id * loc_len)
    rank = n_halves - 1 - half_id
    return int(rank * loc_len + half_len)


def zigzag_half_ag_starts(cp_size: int, half_len: int) -> list[int]:
    """AG token starts for every global half id ``0 .. 2*cp_size - 1``.

    Returns a list of length ``2 * cp_size``.  Slice
    ``ag[..., starts[g] : starts[g] + half_len, ...]`` equals global half ``g``.
    """
    n_halves = 2 * cp_size
    return [
        zigzag_half_ag_token_start(g, cp_size, half_len)
        for g in range(n_halves)
    ]


def unpermute_ag_to_global(
    x: Tensor,
    cp_size: int,
    *,
    seq_dim: int = 1,
) -> Tensor:
    """Map zigzag AG rank-concat layout → global token order along *seq_dim*.

    Uses :func:`zigzag_ag_to_global_index`.  Intended for tests, unfused
    attention, and logits comparison — the flash zigzag path keeps AG layout
    and does not call this helper.
    """
    if cp_size == 1:
        return x
    _check_seq_tensor(x, seq_dim)
    seq = x.size(seq_dim)
    if seq % (2 * cp_size) != 0:
        raise ValueError(
            f"sequence length ({seq}) must be divisible by 2 * cp_size "
            f"({2 * cp_size}) for zigzag unpermute"
        )
    half_len = seq // (2 * cp_size)
    perm = zigzag_ag_to_global_index(cp_size, half_len, device=x.device)
    return x.index_select(seq_dim, perm)


def zigzag_ag_to_global_index(
    cp_size: int,
    half_len: int,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Permutation mapping AG rank-concat layout → global half order.

    After an all-gather across CP ranks, tokens are concatenated in rank order
    (rank 0's local tokens, then rank 1's, etc.).  This permutation restores
    global sequential order ``[0, 1, ..., S-1]`` where ``S = cp_size * 2 * half_len``.

    Returns a 1-D LongTensor of shape ``[cp_size * 2 * half_len]``.

    Semantics: ``ag_tensor[perm]`` yields global order.

    Implementation is fully vectorized (no Python token loops).  A scalar
    GPU write loop over ``S`` tokens was measured at ~200 ms/AG on S=2048 and
    dominated e2e zigzag step time.
    """
    _check_cp_valid(0, cp_size)  # just validate cp_size >= 1
    if half_len < 1:
        raise ValueError(f"half_len must be >= 1, got {half_len}")
    n_halves = 2 * cp_size
    loc_len = 2 * half_len  # tokens per rank in AG
    # Half g < cp is rank g's first half (offset 0); half g >= cp is rank
    # (2*cp-1-g)'s second half (offset half_len).
    g = torch.arange(n_halves, device=device, dtype=torch.long)
    r = torch.where(g < cp_size, g, n_halves - 1 - g)
    offset = torch.where(
        g < cp_size,
        torch.zeros((), device=device, dtype=torch.long),
        torch.tensor(half_len, device=device, dtype=torch.long),
    )
    # perm[g, t] = r[g] * loc_len + offset[g] + t  → flatten half-major.
    base = (r * loc_len + offset).unsqueeze(1)  # [n_halves, 1]
    t = torch.arange(half_len, device=device, dtype=torch.long).unsqueeze(0)
    return (base + t).reshape(-1)


def zigzag_global_to_ag_index(
    cp_size: int,
    half_len: int,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Inverse of :func:`zigzag_ag_to_global_index`.

    Maps global half order → AG rank-concat layout.
    ``global_tensor[inverse]`` gives the AG-ordered tensor, or equivalently
    ``ag_tensor[perm] == global_tensor``.

    Returns a 1-D LongTensor of shape ``[cp_size * 2 * half_len]``.
    """
    perm = zigzag_ag_to_global_index(cp_size, half_len, device=device)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return inv
