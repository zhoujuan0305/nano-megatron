"""Attention backend resolution and kernels (unfused / flash / ring)."""

from __future__ import annotations

import os
from typing import Any, Literal

import torch
from torch import Tensor

from nano_megatron.distributed.backend import CommBackend
from nano_megatron.parallel.context_parallel import (
    causal_attn_scores_cp,
    gather_from_context_parallel_region,
    zigzag_half_ag_token_start,
    zigzag_half_ids,
)

CPPackName = Literal["contiguous", "zigzag"]
_VALID_CP_PACK = frozenset({"contiguous", "zigzag"})

AttentionBackendName = Literal["flash", "unfused"]
RequestedAttentionBackend = Literal["auto", "flash", "unfused"]

# CP chunked-FA multi-block strategy:
# - "prefix": cat non-causal KV blocks → 1 FA + 1 causal FA + online combine
# - "legacy": one FA launch per KV block (original loop)
CPFaChunkMode = Literal["prefix", "legacy"]
_VALID_CP_FA_CHUNK = frozenset({"prefix", "legacy"})
_CP_FA_CHUNK_ENV = "NANO_CP_FA_CHUNK"
_DEFAULT_CP_FA_CHUNK: CPFaChunkMode = "prefix"

_FLASH_DTYPES = (torch.float16, torch.bfloat16)


def resolve_cp_fa_chunk_mode(mode: str | None = None) -> CPFaChunkMode:
    """Resolve CP multi-block FA chunk mode.

    Priority: explicit ``mode`` arg → env ``NANO_CP_FA_CHUNK`` → default
    ``"prefix"``.  Valid values: ``"prefix"``, ``"legacy"``.
    """
    if mode is None:
        raw = os.environ.get(_CP_FA_CHUNK_ENV, _DEFAULT_CP_FA_CHUNK)
    else:
        raw = mode
    if raw not in _VALID_CP_FA_CHUNK:
        raise ValueError(
            f"CP FA chunk mode must be one of {sorted(_VALID_CP_FA_CHUNK)}; "
            f"got {raw!r} (env {_CP_FA_CHUNK_ENV} or chunk_mode arg)"
        )
    return raw  # type: ignore[return-value]


def flash_attn_available() -> bool:
    """Return True if the optional ``flash_attn`` package can be imported."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_attention_backend(
    *,
    requested: RequestedAttentionBackend | str,
    dtype: torch.dtype,
    device: torch.device,
    return_activations: bool = False,
) -> AttentionBackendName:
    """Resolve the effective attention backend for a call site.

    Rules (in order):
    - ``return_activations=True`` always forces unfused (needs scores/probs).
    - ``requested="unfused"`` → unfused.
    - ``requested="auto"`` → flash only when CUDA + fp16/bf16 + flash_attn available;
      otherwise unfused. Never silently casts fp32 to half.
    - ``requested="flash"`` → flash when CUDA + fp16/bf16 + package available;
      otherwise raises ``RuntimeError``.
    """
    if return_activations:
        return "unfused"

    if requested not in ("auto", "flash", "unfused"):
        raise ValueError(
            f"requested must be one of 'auto', 'flash', 'unfused'; got {requested!r}"
        )

    if requested == "unfused":
        return "unfused"

    has_flash = flash_attn_available()
    can_flash = (
        device.type == "cuda"
        and dtype in _FLASH_DTYPES
        and has_flash
    )

    if requested == "auto":
        return "flash" if can_flash else "unfused"

    # requested == "flash"
    if can_flash:
        return "flash"

    reasons: list[str] = []
    if device.type != "cuda":
        reasons.append(f"device must be CUDA (got {device})")
    if dtype not in _FLASH_DTYPES:
        reasons.append(f"dtype must be float16 or bfloat16 (got {dtype})")
    if not has_flash:
        reasons.append("flash_attn package is not installed")
    raise RuntimeError(
        "attn_backend='flash' requires CUDA, fp16/bf16, and flash_attn; "
        + "; ".join(reasons)
    )


def unfused_causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    scale: float,
    query_start: int | None = None,
) -> Tensor:
    """Unfused causal attention: scores → softmax → matmul V.

    Args:
        q: Query tensor ``[B, H, S_q, D]``.
        k: Key tensor ``[B, H, S_k, D]``.
        v: Value tensor ``[B, H, S_k, D]``.
        scale: Softmax scale (typically ``1 / sqrt(D)``).
        query_start: If ``None`` and ``S_q == S_k``, apply standard full-sequence
            causal mask. If set, apply the same CP local-query mask as
            ``causal_attn_scores_cp`` (query positions start at ``query_start``).

    Returns:
        Context tensor ``[B, H, S_q, D]``.
    """
    # Lazy imports avoid a circular dependency once layers import attention_backend.
    from nano_megatron.reference.layers import causal_attn_scores, softmax_last

    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"q, k, v must be 4D [B,H,S,D]; got q={tuple(q.shape)}, "
            f"k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if k.shape != v.shape:
        raise ValueError(
            f"k and v must share shape; got k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[3] != k.shape[3]:
        raise ValueError(
            f"q/k batch, heads, and head_dim must match; "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}"
        )

    s_q = q.size(2)
    s_k = k.size(2)

    if query_start is None:
        if s_q != s_k:
            raise ValueError(
                f"query_start is required when S_q != S_k "
                f"(got S_q={s_q}, S_k={s_k})"
            )
        scores = causal_attn_scores(q, k, scale)
    else:
        scores = causal_attn_scores_cp(q, k, scale, query_start)

    probs = softmax_last(scores)
    return torch.matmul(probs, v)


def _to_bshd(x: Tensor) -> Tensor:
    """Convert ``[B, H, S, D]`` → ``[B, S, H, D]`` (flash_attn layout)."""
    return x.transpose(1, 2).contiguous()


def _to_bhsd(x: Tensor) -> Tensor:
    """Convert ``[B, S, H, D]`` → ``[B, H, S, D]`` (project layout)."""
    return x.transpose(1, 2).contiguous()


def flash_causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    scale: float,
    dropout_p: float = 0.0,
) -> Tensor:
    """Causal attention via ``flash_attn.flash_attn_func`` (cp=1, full sequence).

    Args:
        q: Query tensor ``[B, H, S, D]`` on CUDA, fp16 or bf16.
        k: Key tensor ``[B, H, S, D]``, same device/dtype as ``q``.
        v: Value tensor ``[B, H, S, D]``, same device/dtype as ``q``.
        scale: Softmax scale (typically ``1 / sqrt(D)``), passed as ``softmax_scale``.
        dropout_p: Dropout probability (default 0.0).

    Returns:
        Context tensor ``[B, H, S, D]``.

    Raises:
        RuntimeError: If tensors are not CUDA fp16/bf16, or ``flash_attn`` is missing.
        ValueError: If ranks/shapes are invalid.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"q, k, v must be 4D [B,H,S,D]; got q={tuple(q.shape)}, "
            f"k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if k.shape != v.shape or q.shape != k.shape:
        raise ValueError(
            f"q, k, v must share shape for cp=1 flash path; "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )

    device = q.device
    dtype = q.dtype
    if device.type != "cuda":
        raise RuntimeError(
            f"flash_causal_attention requires CUDA tensors (got device={device})"
        )
    if dtype not in _FLASH_DTYPES:
        raise RuntimeError(
            f"flash_causal_attention requires float16 or bfloat16 "
            f"(got dtype={dtype})"
        )
    if k.device != device or v.device != device:
        raise RuntimeError(
            f"q, k, v must be on the same device; "
            f"got q={device}, k={k.device}, v={v.device}"
        )
    if k.dtype != dtype or v.dtype != dtype:
        raise RuntimeError(
            f"q, k, v must share dtype; got q={dtype}, k={k.dtype}, v={v.dtype}"
        )

    try:
        from flash_attn import flash_attn_func
    except ImportError as exc:
        raise RuntimeError(
            "flash_causal_attention requires the flash_attn package"
        ) from exc

    out = flash_attn_func(
        _to_bshd(q),
        _to_bshd(k),
        _to_bshd(v),
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=True,
    )
    return _to_bhsd(out)


def _online_softmax_combine(
    out: Tensor,
    lse: Tensor,
    out_i: Tensor,
    lse_i: Tensor,
) -> tuple[Tensor, Tensor]:
    """Merge two partial attention results via online softmax (float32 accumulate).

    Both ``out`` / ``out_i`` are already *normalized* attention outputs
    (softmax over their own key block).  The merge reweights them by the
    block log-sum-exp values so the result matches full-softmax attention
    over the concatenated key blocks.

    Args:
        out: Running context ``[B, H, S, D]``.
        lse: Running log-sum-exp ``[B, H, S]`` (float).
        out_i: New block context ``[B, H, S, D]``.
        lse_i: New block log-sum-exp ``[B, H, S]`` (float).

    Returns:
        ``(out_new [B, H, S, D], lse_new [B, H, S])``.
    """
    if out.shape != out_i.shape:
        raise ValueError(
            f"out and out_i must share shape; got {tuple(out.shape)} vs {tuple(out_i.shape)}"
        )
    if lse.shape != lse_i.shape:
        raise ValueError(
            f"lse and lse_i must share shape; got {tuple(lse.shape)} vs {tuple(lse_i.shape)}"
        )
    if out.dim() != 4:
        raise ValueError(f"out must be 4D [B,H,S,D]; got {tuple(out.shape)}")
    if lse.dim() != 3:
        raise ValueError(f"lse must be 3D [B,H,S]; got {tuple(lse.shape)}")
    if (
        lse.shape[0] != out.shape[0]
        or lse.shape[1] != out.shape[1]
        or lse.shape[2] != out.shape[2]
    ):
        raise ValueError(
            f"lse [B,H,S] must match out [B,H,S,D] leading dims; "
            f"got lse={tuple(lse.shape)}, out={tuple(out.shape)}"
        )

    # Normalized block outputs: o = Σ_i softmax_i(s) v_i  reweighted by
    #   w_i = exp(lse_i - lse_new),  lse_new = log(Σ_i exp(lse_i)).
    # Using max-stabilized form.  Accumulate in fp32 for numerical stability.
    out_f = out.float()
    out_i_f = out_i.float()
    lse_f = lse.float()
    lse_i_f = lse_i.float()

    m = torch.maximum(lse_f, lse_i_f)
    lse_new = m + torch.log(torch.exp(lse_f - m) + torch.exp(lse_i_f - m))
    w = torch.exp(lse_f - lse_new).unsqueeze(-1)
    w_i = torch.exp(lse_i_f - lse_new).unsqueeze(-1)
    out_new = out_f * w + out_i_f * w_i
    return out_new.to(dtype=out.dtype), lse_new


def _flash_fwd_out_lse(
    q_bshd: Tensor,
    k_bshd: Tensor,
    v_bshd: Tensor,
    *,
    scale: float,
    causal: bool,
    dropout_p: float,
) -> tuple[Tensor, Tensor]:
    """Run flash-attn forward and return ``(out_bshd, lse_bhs)``."""
    from flash_attn import flash_attn_func

    out, lse, _ = flash_attn_func(
        q_bshd,
        k_bshd,
        v_bshd,
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=causal,
        return_attn_probs=True,
    )
    return out, lse


def _flash_bwd_block(
    dout_bshd: Tensor,
    q_bshd: Tensor,
    k_bshd: Tensor,
    v_bshd: Tensor,
    out_bshd: Tensor,
    lse_bhs: Tensor,
    *,
    scale: float,
    causal: bool,
    dropout_p: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Flash-attn block backward with *global* ``out`` / ``lse``.

    Returns ``(dq, dk, dv)`` in BSHD layout.
    """
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_backward

    dq = torch.empty_like(q_bshd)
    dk = torch.empty_like(k_bshd)
    dv = torch.empty_like(v_bshd)
    _wrapped_flash_attn_backward(
        dout_bshd.contiguous(),
        q_bshd.contiguous(),
        k_bshd.contiguous(),
        v_bshd.contiguous(),
        out_bshd.contiguous(),
        lse_bhs.contiguous(),
        dq,
        dk,
        dv,
        dropout_p,
        scale,
        causal,
        -1,  # window_size_left
        -1,  # window_size_right
        0.0,  # softcap
        None,  # alibi_slopes
        False,  # deterministic
        None,  # rng_state (dropout_p==0)
    )
    return dq, dk, dv


def _check_cp_pack(pack: str) -> None:
    if pack not in _VALID_CP_PACK:
        raise ValueError(
            f"pack must be 'contiguous' or 'zigzag', got {pack!r}"
        )


def _chunked_fa_combine_blocks(
    q_bshd: Tensor,
    k_bshd: Tensor,
    v_bshd: Tensor,
    *,
    block_starts: range | list[int],
    block_len: int,
    causal_block: int,
    scale: float,
    dropout_p: float,
    mode: str | None = None,
) -> tuple[Tensor, Tensor]:
    """Run flash over KV blocks and online-softmax-combine.

    Two multi-block strategies (``mode`` / env ``NANO_CP_FA_CHUNK``):

    * ``legacy``: one FA launch per start, sequential online-softmax combine.
    * ``prefix`` (default): concatenate all non-last (non-causal) KV blocks into
      one FA call (``causal=False``), one FA on the last block
      (``causal`` iff that start equals ``causal_block``), then a single
      online-softmax combine.  Caps launches at 2 per Q-shard regardless of
      how many prior blocks are visible (standard CP: causal only on last).

    Args:
        q_bshd: Query in BSHD layout ``[B, S_q, H, D]``.
        k_bshd / v_bshd: Full KV in BSHD (AG or global layout).
        block_starts: Iterable of block start indices (in tokens) to attend.
        block_len: Tokens per KV block (all starts must use the same length).
        causal_block: Block start index that uses ``causal=True`` (the query's
            own block); all earlier blocks use ``causal=False``.
        scale / dropout_p: Passed to flash-attn.
        mode: ``"prefix"`` | ``"legacy"`` | ``None`` (resolve from env).

    Returns:
        ``(out_bhsd [B,H,S_q,D], lse_bhs [B,H,S_q])``.
    """
    chunk_mode = resolve_cp_fa_chunk_mode(mode)
    starts = list(block_starts)
    if not starts:
        raise ValueError("block_starts must be non-empty")
    if block_len < 1:
        raise ValueError(f"block_len must be >= 1, got {block_len}")

    if chunk_mode == "legacy":
        return _chunked_fa_combine_blocks_legacy(
            q_bshd,
            k_bshd,
            v_bshd,
            block_starts=starts,
            block_len=block_len,
            causal_block=causal_block,
            scale=scale,
            dropout_p=dropout_p,
        )

    # --- prefix path ---
    # Standard CP: only the last start may be the causal block.
    if len(starts) >= 2:
        for s in starts[:-1]:
            if s == causal_block:
                raise ValueError(
                    "prefix chunk mode expects causal_block only on the last "
                    f"start; got causal_block={causal_block}, starts={starts}"
                )

    if len(starts) == 1:
        s0 = starts[0]
        k_j = k_bshd[:, s0 : s0 + block_len, :, :].contiguous()
        v_j = v_bshd[:, s0 : s0 + block_len, :, :].contiguous()
        out_i_bshd, lse_i = _flash_fwd_out_lse(
            q_bshd,
            k_j,
            v_j,
            scale=scale,
            causal=(s0 == causal_block),
            dropout_p=dropout_p,
        )
        return _to_bhsd(out_i_bshd), lse_i

    # n >= 2: one non-causal FA on cat(prefix), one FA on last, one combine.
    # Single prefix block: skip cat (same as legacy first launch).
    pref_starts = starts[:-1]
    if len(pref_starts) == 1:
        s0 = pref_starts[0]
        k_pref = k_bshd[:, s0 : s0 + block_len, :, :].contiguous()
        v_pref = v_bshd[:, s0 : s0 + block_len, :, :].contiguous()
    else:
        k_pref = torch.cat(
            [k_bshd[:, s : s + block_len, :, :] for s in pref_starts],
            dim=1,
        ).contiguous()
        v_pref = torch.cat(
            [v_bshd[:, s : s + block_len, :, :] for s in pref_starts],
            dim=1,
        ).contiguous()
    out0_bshd, lse0 = _flash_fwd_out_lse(
        q_bshd,
        k_pref,
        v_pref,
        scale=scale,
        causal=False,
        dropout_p=dropout_p,
    )
    s_last = starts[-1]
    k_last = k_bshd[:, s_last : s_last + block_len, :, :].contiguous()
    v_last = v_bshd[:, s_last : s_last + block_len, :, :].contiguous()
    out1_bshd, lse1 = _flash_fwd_out_lse(
        q_bshd,
        k_last,
        v_last,
        scale=scale,
        causal=(s_last == causal_block),
        dropout_p=dropout_p,
    )
    return _online_softmax_combine(
        _to_bhsd(out0_bshd), lse0, _to_bhsd(out1_bshd), lse1
    )


def _chunked_fa_combine_blocks_legacy(
    q_bshd: Tensor,
    k_bshd: Tensor,
    v_bshd: Tensor,
    *,
    block_starts: list[int],
    block_len: int,
    causal_block: int,
    scale: float,
    dropout_p: float,
) -> tuple[Tensor, Tensor]:
    """Per-block FA + sequential online-softmax combine (original path)."""
    out_bhsd: Tensor | None = None
    lse_bhs: Tensor | None = None
    for start in block_starts:
        k_j = k_bshd[:, start : start + block_len, :, :].contiguous()
        v_j = v_bshd[:, start : start + block_len, :, :].contiguous()
        causal = start == causal_block
        out_i_bshd, lse_i = _flash_fwd_out_lse(
            q_bshd,
            k_j,
            v_j,
            scale=scale,
            causal=causal,
            dropout_p=dropout_p,
        )
        out_i = _to_bhsd(out_i_bshd)
        if out_bhsd is None:
            out_bhsd = out_i
            lse_bhs = lse_i
        else:
            assert lse_bhs is not None
            out_bhsd, lse_bhs = _online_softmax_combine(
                out_bhsd, lse_bhs, out_i, lse_i
            )
    assert out_bhsd is not None and lse_bhs is not None
    return out_bhsd, lse_bhs


def _chunked_fa_bwd_blocks(
    dout_bshd: Tensor,
    q_bshd: Tensor,
    k_bshd: Tensor,
    v_bshd: Tensor,
    out_bshd: Tensor,
    lse_bhs: Tensor,
    *,
    block_starts: range | list[int],
    block_len: int,
    causal_block: int,
    scale: float,
    dropout_p: float,
    dq_bshd: Tensor,
    dk_bshd: Tensor,
    dv_bshd: Tensor,
    mode: str | None = None,
) -> None:
    """Accumulate blockwise flash backward into ``dq`` / ``dk`` / ``dv`` (BSHD).

    Mirrors :func:`_chunked_fa_combine_blocks`:

    * ``legacy``: one bwd per start.
    * ``prefix``: one bwd on cat(prefix KV), scatter ``dK``/``dV`` back to each
      prefix slice with ``add_``; one bwd on the last block; ``dQ`` accumulates.
    """
    chunk_mode = resolve_cp_fa_chunk_mode(mode)
    starts = list(block_starts)
    if not starts:
        raise ValueError("block_starts must be non-empty")
    if block_len < 1:
        raise ValueError(f"block_len must be >= 1, got {block_len}")

    if chunk_mode == "legacy":
        _chunked_fa_bwd_blocks_legacy(
            dout_bshd,
            q_bshd,
            k_bshd,
            v_bshd,
            out_bshd,
            lse_bhs,
            block_starts=starts,
            block_len=block_len,
            causal_block=causal_block,
            scale=scale,
            dropout_p=dropout_p,
            dq_bshd=dq_bshd,
            dk_bshd=dk_bshd,
            dv_bshd=dv_bshd,
        )
        return

    # --- prefix path ---
    if len(starts) >= 2:
        for s in starts[:-1]:
            if s == causal_block:
                raise ValueError(
                    "prefix chunk mode expects causal_block only on the last "
                    f"start; got causal_block={causal_block}, starts={starts}"
                )

    if len(starts) == 1:
        s0 = starts[0]
        k_j = k_bshd[:, s0 : s0 + block_len, :, :].contiguous()
        v_j = v_bshd[:, s0 : s0 + block_len, :, :].contiguous()
        dqi, dki, dvi = _flash_bwd_block(
            dout_bshd,
            q_bshd,
            k_j,
            v_j,
            out_bshd,
            lse_bhs,
            scale=scale,
            causal=(s0 == causal_block),
            dropout_p=dropout_p,
        )
        dq_bshd.add_(dqi)
        dk_bshd[:, s0 : s0 + block_len, :, :].add_(dki)
        dv_bshd[:, s0 : s0 + block_len, :, :].add_(dvi)
        return

    pref_starts = starts[:-1]
    if len(pref_starts) == 1:
        s0 = pref_starts[0]
        k_pref = k_bshd[:, s0 : s0 + block_len, :, :].contiguous()
        v_pref = v_bshd[:, s0 : s0 + block_len, :, :].contiguous()
    else:
        k_pref = torch.cat(
            [k_bshd[:, s : s + block_len, :, :] for s in pref_starts],
            dim=1,
        ).contiguous()
        v_pref = torch.cat(
            [v_bshd[:, s : s + block_len, :, :] for s in pref_starts],
            dim=1,
        ).contiguous()
    dqi0, dki_pref, dvi_pref = _flash_bwd_block(
        dout_bshd,
        q_bshd,
        k_pref,
        v_pref,
        out_bshd,
        lse_bhs,
        scale=scale,
        causal=False,
        dropout_p=dropout_p,
    )
    dq_bshd.add_(dqi0)
    if len(pref_starts) == 1:
        s0 = pref_starts[0]
        dk_bshd[:, s0 : s0 + block_len, :, :].add_(dki_pref)
        dv_bshd[:, s0 : s0 + block_len, :, :].add_(dvi_pref)
    else:
        for i, s in enumerate(pref_starts):
            sl = slice(i * block_len, (i + 1) * block_len)
            dk_bshd[:, s : s + block_len, :, :].add_(dki_pref[:, sl, :, :])
            dv_bshd[:, s : s + block_len, :, :].add_(dvi_pref[:, sl, :, :])

    s_last = starts[-1]
    k_last = k_bshd[:, s_last : s_last + block_len, :, :].contiguous()
    v_last = v_bshd[:, s_last : s_last + block_len, :, :].contiguous()
    dqi1, dki1, dvi1 = _flash_bwd_block(
        dout_bshd,
        q_bshd,
        k_last,
        v_last,
        out_bshd,
        lse_bhs,
        scale=scale,
        causal=(s_last == causal_block),
        dropout_p=dropout_p,
    )
    dq_bshd.add_(dqi1)
    dk_bshd[:, s_last : s_last + block_len, :, :].add_(dki1)
    dv_bshd[:, s_last : s_last + block_len, :, :].add_(dvi1)


def _chunked_fa_bwd_blocks_legacy(
    dout_bshd: Tensor,
    q_bshd: Tensor,
    k_bshd: Tensor,
    v_bshd: Tensor,
    out_bshd: Tensor,
    lse_bhs: Tensor,
    *,
    block_starts: list[int],
    block_len: int,
    causal_block: int,
    scale: float,
    dropout_p: float,
    dq_bshd: Tensor,
    dk_bshd: Tensor,
    dv_bshd: Tensor,
) -> None:
    """Per-block flash backward (original path)."""
    for start in block_starts:
        k_j = k_bshd[:, start : start + block_len, :, :].contiguous()
        v_j = v_bshd[:, start : start + block_len, :, :].contiguous()
        causal = start == causal_block
        dqi, dki, dvi = _flash_bwd_block(
            dout_bshd,
            q_bshd,
            k_j,
            v_j,
            out_bshd,
            lse_bhs,
            scale=scale,
            causal=causal,
            dropout_p=dropout_p,
        )
        dq_bshd.add_(dqi)
        dk_bshd[:, start : start + block_len, :, :] = (
            dk_bshd[:, start : start + block_len, :, :] + dki
        )
        dv_bshd[:, start : start + block_len, :, :] = (
            dv_bshd[:, start : start + block_len, :, :] + dvi
        )


class _FlashChunkedCPCausalAttention(torch.autograd.Function):
    """CP causal attention via chunked FlashAttention (contiguous or zigzag).

    **Contiguous** (``pack="contiguous"``): for KV rank-block ``j <= cp_rank``,
    run flash (``causal`` only when ``j == cp_rank``) and online-softmax-combine.
    KV blocks with ``j > cp_rank`` are fully masked and skipped.  Block size
    is ``S/cp``.  ``K_full``/``V_full`` are AG rank-concat (= global order).

    **Zigzag** (``pack="zigzag"``): local ``Q`` is zigzag ``[half_a; half_b]``;
    ``K_full``/``V_full`` are **AG rank-concat layout** (no unpermute).  For
    each local half with global half-id ``g``, attend AG half-blocks
    ``j = 0..g`` via :func:`zigzag_half_ag_token_start`, with
    ``causal`` only on half ``g``.  Half size ``H = S/(2·cp)``.  ``dk``/``dv``
    are written at AG offsets so gather backward can RS without inv-permute.

    **Multi-block FA chunking** (``chunk_mode`` / env ``NANO_CP_FA_CHUNK``):

    * ``prefix`` (default): cat non-causal KV blocks into one FA + one causal
      FA + one online combine (≤2 launches per Q-shard).
    * ``legacy``: one FA launch per visible KV block.

    Backward: flash blockwise backward with the combined ``out`` and ``lse``
    (standard ring/block FA backward), accumulating ``dq`` and writing
    per-block ``dk``/``dv``.

    Communication is *not* handled here — callers all-gather local K/V first
    (see ``flash_ring_causal_attention``).  This is AG + chunked-FA, not a
    bandwidth-optimal KV ring; grads are exact w.r.t. that compute graph.
    """

    @staticmethod
    def forward(
        ctx: Any,
        q: Tensor,
        k_full: Tensor,
        v_full: Tensor,
        scale: float,
        cp_rank: int,
        cp_size: int,
        dropout_p: float,
        pack: str,
        chunk_mode: str,
    ) -> Tensor:
        if dropout_p != 0.0:
            raise RuntimeError(
                f"_FlashChunkedCPCausalAttention does not support "
                f"dropout_p={dropout_p}.  The chunked backward cannot propagate "
                f"the RNG state required for correct dropout gradients."
            )
        _check_cp_pack(pack)
        resolved_chunk = resolve_cp_fa_chunk_mode(
            None if chunk_mode == "" else chunk_mode
        )
        if q.dim() != 4 or k_full.dim() != 4 or v_full.dim() != 4:
            raise ValueError(
                f"q, k, v must be 4D [B,H,S,D]; got q={tuple(q.shape)}, "
                f"k={tuple(k_full.shape)}, v={tuple(v_full.shape)}"
            )
        if k_full.shape != v_full.shape:
            raise ValueError(
                f"k_full and v_full must share shape; "
                f"got k={tuple(k_full.shape)}, v={tuple(v_full.shape)}"
            )
        if (
            q.shape[0] != k_full.shape[0]
            or q.shape[1] != k_full.shape[1]
            or q.shape[3] != k_full.shape[3]
        ):
            raise ValueError(
                f"q/k batch, heads, head_dim must match; "
                f"got q={tuple(q.shape)}, k={tuple(k_full.shape)}"
            )
        if cp_size < 1:
            raise ValueError(f"cp_size must be >= 1, got {cp_size}")
        if cp_rank < 0 or cp_rank >= cp_size:
            raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")

        s_full = k_full.size(2)
        if s_full % cp_size != 0:
            raise ValueError(
                f"full KV seq len ({s_full}) not divisible by cp_size ({cp_size})"
            )
        s_local = s_full // cp_size
        if q.size(2) != s_local:
            raise ValueError(
                f"q seq len must equal S_full/cp_size ({s_local}); got {q.size(2)}"
            )

        q_bshd = _to_bshd(q)
        k_bshd = _to_bshd(k_full)
        v_bshd = _to_bshd(v_full)

        if pack == "contiguous":
            # Contiguous CP: only KV ranks j <= cp_rank are visible.
            # Block size = S/cp (rank blocks). Shared helper handles prefix/legacy.
            block_starts = [j * s_local for j in range(cp_rank + 1)]
            causal_block = cp_rank * s_local
            out_bhsd, lse_bhs = _chunked_fa_combine_blocks(
                q_bshd,
                k_bshd,
                v_bshd,
                block_starts=block_starts,
                block_len=s_local,
                causal_block=causal_block,
                scale=scale,
                dropout_p=dropout_p,
                mode=resolved_chunk,
            )
            half_len = 0  # unused for contiguous
        else:
            # Zigzag: Q is [half_a; half_b]; K/V full are AG rank-concat layout.
            # Half size H = S / (2 * cp). For each local half with global id g,
            # attend AG half-blocks j=0..g (via half→AG start map) with causal
            # only on half g.
            if s_full % (2 * cp_size) != 0:
                raise ValueError(
                    f"full KV seq len ({s_full}) not divisible by "
                    f"2 * cp_size ({2 * cp_size}) for zigzag pack"
                )
            half_len = s_full // (2 * cp_size)
            if s_local != 2 * half_len:
                raise ValueError(
                    f"zigzag local seq must be 2*H={2 * half_len}; got {s_local}"
                )
            half_ids = zigzag_half_ids(cp_rank, cp_size)
            half_outs: list[Tensor] = []
            half_lses: list[Tensor] = []
            for t, g in enumerate(half_ids):
                q_half = q_bshd[:, t * half_len : (t + 1) * half_len, :, :].contiguous()
                block_starts = [
                    zigzag_half_ag_token_start(j, cp_size, half_len)
                    for j in range(g + 1)
                ]
                causal_block = zigzag_half_ag_token_start(g, cp_size, half_len)
                out_h, lse_h = _chunked_fa_combine_blocks(
                    q_half,
                    k_bshd,
                    v_bshd,
                    block_starts=block_starts,
                    block_len=half_len,
                    causal_block=causal_block,
                    scale=scale,
                    dropout_p=dropout_p,
                    mode=resolved_chunk,
                )
                half_outs.append(out_h)
                half_lses.append(lse_h)
            out_bhsd = torch.cat(half_outs, dim=2)
            lse_bhs = torch.cat(half_lses, dim=2)

        ctx.save_for_backward(q, k_full, v_full, out_bhsd, lse_bhs)
        ctx.scale = scale
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.dropout_p = dropout_p
        ctx.s_local = s_local
        ctx.pack = pack
        ctx.half_len = half_len
        ctx.chunk_mode = resolved_chunk
        return out_bhsd

    @staticmethod
    def backward(
        ctx: Any, dout: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, None, None, None, None, None, None]:
        q, k_full, v_full, out, lse = ctx.saved_tensors
        scale: float = ctx.scale
        cp_rank: int = ctx.cp_rank
        cp_size: int = ctx.cp_size
        s_local: int = ctx.s_local
        dropout_p: float = ctx.dropout_p
        pack: str = ctx.pack
        half_len: int = ctx.half_len
        chunk_mode: str = ctx.chunk_mode

        q_bshd = _to_bshd(q)
        k_bshd = _to_bshd(k_full)
        v_bshd = _to_bshd(v_full)
        out_bshd = _to_bshd(out)
        dout_bshd = _to_bshd(dout)

        # Pad head_dim to multiple of 8 for _wrapped_flash_attn_backward.
        # The forward path uses flash_attn_func which pads internally and
        # returns unpadded output; _wrapped_flash_attn_backward does NOT pad,
        # so we must pad here and unpad the gradients afterward to keep
        # forward and backward head_dim handling consistent.
        head_dim = q_bshd.size(-1)
        if head_dim % 8 != 0:
            pad_size = 8 - head_dim % 8
            q_bshd = torch.nn.functional.pad(q_bshd, [0, pad_size])
            k_bshd = torch.nn.functional.pad(k_bshd, [0, pad_size])
            v_bshd = torch.nn.functional.pad(v_bshd, [0, pad_size])
            out_bshd = torch.nn.functional.pad(out_bshd, [0, pad_size])
            dout_bshd = torch.nn.functional.pad(dout_bshd, [0, pad_size])

        dq_bshd = torch.zeros_like(q_bshd)
        dk_bshd = torch.zeros_like(k_bshd)
        dv_bshd = torch.zeros_like(v_bshd)

        if pack == "contiguous":
            block_starts = [j * s_local for j in range(cp_rank + 1)]
            causal_block = cp_rank * s_local
            _chunked_fa_bwd_blocks(
                dout_bshd,
                q_bshd,
                k_bshd,
                v_bshd,
                out_bshd,
                lse,
                block_starts=block_starts,
                block_len=s_local,
                causal_block=causal_block,
                scale=scale,
                dropout_p=dropout_p,
                dq_bshd=dq_bshd,
                dk_bshd=dk_bshd,
                dv_bshd=dv_bshd,
                mode=chunk_mode,
            )
        else:
            # Zigzag: each local half has its own out/lse slice; accumulate dk/dv
            # at AG offsets (a KV half may be visible to both local Q halves).
            half_ids = zigzag_half_ids(cp_rank, cp_size)
            for t, g in enumerate(half_ids):
                sl = slice(t * half_len, (t + 1) * half_len)
                q_half = q_bshd[:, sl, :, :].contiguous()
                out_half = out_bshd[:, sl, :, :].contiguous()
                dout_half = dout_bshd[:, sl, :, :].contiguous()
                # lse is [B, H, S] (BHSD-style heads), not BSHD.
                lse_half = lse[:, :, sl].contiguous()
                dq_half = torch.zeros_like(q_half)
                block_starts = [
                    zigzag_half_ag_token_start(j, cp_size, half_len)
                    for j in range(g + 1)
                ]
                causal_block = zigzag_half_ag_token_start(g, cp_size, half_len)
                _chunked_fa_bwd_blocks(
                    dout_half,
                    q_half,
                    k_bshd,
                    v_bshd,
                    out_half,
                    lse_half,
                    block_starts=block_starts,
                    block_len=half_len,
                    causal_block=causal_block,
                    scale=scale,
                    dropout_p=dropout_p,
                    dq_bshd=dq_half,
                    dk_bshd=dk_bshd,
                    dv_bshd=dv_bshd,
                    mode=chunk_mode,
                )
                dq_bshd[:, sl, :, :] = dq_half

        # Unpad gradients to original head_dim.
        if head_dim % 8 != 0:
            dq_bshd = dq_bshd[..., :head_dim]
            dk_bshd = dk_bshd[..., :head_dim]
            dv_bshd = dv_bshd[..., :head_dim]

        return (
            _to_bhsd(dq_bshd),
            _to_bhsd(dk_bshd),
            _to_bhsd(dv_bshd),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _flash_chunked_cp_causal_attention(
    q: Tensor,
    k_full: Tensor,
    v_full: Tensor,
    *,
    scale: float,
    cp_rank: int,
    cp_size: int,
    dropout_p: float = 0.0,
    pack: str = "contiguous",
    chunk_mode: str | None = None,
) -> Tensor:
    """Chunked FA over full KV for CP (no collective).  Test / internal helper.

    ``chunk_mode``: ``"prefix"`` | ``"legacy"`` | ``None`` (env / default).
    Empty string is treated as ``None`` for the autograd Function boundary.
    """
    mode_arg = "" if chunk_mode is None else chunk_mode
    return _FlashChunkedCPCausalAttention.apply(
        q, k_full, v_full, scale, cp_rank, cp_size, dropout_p, pack, mode_arg
    )


def flash_ring_causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    scale: float,
    cp_group: Any | None,
    cp_rank: int,
    cp_size: int,
    backend: CommBackend | None,
    dropout_p: float = 0.0,
    pack: str = "contiguous",
    chunk_mode: str | None = None,
) -> Tensor:
    """Causal attention for context-parallel shards (FlashAttention).

    Layout at the API boundary is ``[B, H, S_local, D]`` for ``q`` and local
    ``k``/``v``.

    **Contiguous** (``pack="contiguous"``): rank ``r`` owns
    ``[r * S_local, (r + 1) * S_local)``.  Visibility:

    * KV rank ``j < r``: full attention (``causal=False``)
    * KV rank ``j == r``: local causal (``causal=True``)
    * KV rank ``j > r``: skipped

    **Zigzag** (``pack="zigzag"``): local Q/K/V are DualChunkSwap halves; AG
    keeps rank-concat (AG) layout with **no** full-sequence unpermute.
    Chunked FA indexes KV half-blocks via :func:`zigzag_half_ag_token_start`
    (see ``_FlashChunkedCPCausalAttention``).

    **Multi-block FA** (``chunk_mode`` / env ``NANO_CP_FA_CHUNK``, default
    ``prefix``): prefix-concat non-causal KV into one FA + one causal FA +
    online combine; ``legacy`` keeps one launch per block.

    Implementation note (MVP):
        Uses differentiable all-gather of K/V then **chunked** flash-attn over
        visible blocks with online-softmax combine and a custom backward that
        calls flash blockwise backward with global O/LSE.  This is **not**
        bandwidth-optimal ring exchange of KV; it is AG + chunked-FA with
        correct gradients.  A true P2P ring can replace the gather later
        without changing call sites.

    When ``cp_size == 1``, delegates to ``flash_causal_attention``.

    Raises:
        RuntimeError: If ``dropout_p > 0`` and ``cp_size > 1``.  The chunked
            backward does not propagate the RNG state required for correct
            dropout gradients.  Use ``flash_causal_attention`` (cp_size==1)
            when dropout is needed.
    """
    _check_cp_pack(pack)
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")

    if cp_size == 1:
        return flash_causal_attention(q, k, v, scale=scale, dropout_p=dropout_p)

    if dropout_p != 0.0:
        raise RuntimeError(
            f"flash_ring_causal_attention does not support dropout_p={dropout_p} "
            f"when cp_size={cp_size} > 1.  The chunked backward cannot propagate "
            f"the RNG state required for correct dropout gradients.  "
            f"Use cp_size=1 with flash_causal_attention when dropout is needed."
        )

    if backend is None:
        raise RuntimeError(
            "flash_ring_causal_attention requires a CommBackend when cp_size > 1"
        )
    if cp_group is None:
        raise RuntimeError(
            "flash_ring_causal_attention requires cp_group when cp_size > 1"
        )

    # seq_dim=2 for [B, H, S, D]; pack selects local shard layout (AG out).
    k_full = gather_from_context_parallel_region(
        k,
        cp_group,
        backend,
        cp_rank,
        cp_size,
        seq_dim=2,
        grad_op="reduce_scatter",
        pack=pack,
    )
    v_full = gather_from_context_parallel_region(
        v,
        cp_group,
        backend,
        cp_rank,
        cp_size,
        seq_dim=2,
        grad_op="reduce_scatter",
        pack=pack,
    )
    mode_arg = "" if chunk_mode is None else chunk_mode
    return _FlashChunkedCPCausalAttention.apply(
        q, k_full, v_full, scale, cp_rank, cp_size, dropout_p, pack, mode_arg
    )
