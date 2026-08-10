# tests/unit/parallel/test_attention_backend.py
import pytest
import torch

from nano_megatron.parallel.attention_backend import (
    _online_softmax_combine,
    flash_attn_available,
    flash_ring_causal_attention,
    resolve_attention_backend,
    unfused_causal_attention,
)
from nano_megatron.parallel.context_parallel import causal_attn_scores_cp
from nano_megatron.reference.layers import causal_attn_scores, softmax_last


def test_resolve_unfused_forced():
    assert resolve_attention_backend(
        requested="unfused", dtype=torch.bfloat16, device=torch.device("cpu")
    ) == "unfused"


def test_resolve_auto_fp32_is_unfused():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert resolve_attention_backend(
        requested="auto", dtype=torch.float32, device=dev
    ) == "unfused"


def test_resolve_return_activations_forces_unfused():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert resolve_attention_backend(
        requested="flash",
        dtype=torch.bfloat16,
        device=dev,
        return_activations=True,
    ) == "unfused"


def test_resolve_flash_hard_fail_cpu():
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_attention_backend(
            requested="flash",
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )


def test_resolve_flash_hard_fail_fp32():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with pytest.raises(RuntimeError, match="float16 or bfloat16"):
        resolve_attention_backend(
            requested="flash",
            dtype=torch.float32,
            device=dev,
        )


def test_resolve_invalid_requested():
    with pytest.raises(ValueError, match="requested must be one of"):
        resolve_attention_backend(
            requested="bogus",
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )


def test_unfused_matches_reference_math():
    torch.manual_seed(0)
    b, h, s, d = 2, 4, 16, 8
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    scale = d ** -0.5
    scores = causal_attn_scores(q, k, scale)
    ref = torch.matmul(softmax_last(scores), v)
    out = unfused_causal_attention(q, k, v, scale=scale)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_unfused_query_start_matches_cp_path():
    torch.manual_seed(1)
    b, h, s_k, d = 2, 4, 16, 8
    s_q = 4
    query_start = 8
    q = torch.randn(b, h, s_q, d)
    k = torch.randn(b, h, s_k, d)
    v = torch.randn(b, h, s_k, d)
    scale = d ** -0.5
    scores = causal_attn_scores_cp(q, k, scale, query_start)
    ref = torch.matmul(softmax_last(scores), v)
    out = unfused_causal_attention(q, k, v, scale=scale, query_start=query_start)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_unfused_requires_query_start_when_seq_mismatch():
    b, h, d = 1, 2, 4
    q = torch.randn(b, h, 4, d)
    k = torch.randn(b, h, 8, d)
    v = torch.randn(b, h, 8, d)
    with pytest.raises(ValueError, match="query_start is required"):
        unfused_causal_attention(q, k, v, scale=d ** -0.5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not flash_attn_available(),
    reason="requires CUDA and flash_attn",
)
def test_resolve_auto_bf16_cuda_selects_flash():
    assert resolve_attention_backend(
        requested="auto",
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    ) == "flash"


@pytest.mark.skipif(
    not torch.cuda.is_available() or not flash_attn_available(),
    reason="requires CUDA and flash_attn",
)
def test_flash_matches_unfused_bf16():
    torch.manual_seed(0)
    b, h, s, d = 2, 4, 64, 32
    device = "cuda"
    q = torch.randn(b, h, s, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(b, h, s, d, device=device, dtype=torch.bfloat16)
    v = torch.randn(b, h, s, d, device=device, dtype=torch.bfloat16)
    scale = d ** -0.5
    from nano_megatron.parallel.attention_backend import (
        flash_causal_attention,
        unfused_causal_attention as unfused,
    )

    out_f = flash_causal_attention(q, k, v, scale=scale)
    out_u = unfused(q.float(), k.float(), v.float(), scale=scale).bfloat16()
    torch.testing.assert_close(out_f, out_u, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not flash_attn_available(),
    reason="requires CUDA and flash_attn",
)
def test_flash_rejects_cpu_and_fp32():
    from nano_megatron.parallel.attention_backend import flash_causal_attention

    b, h, s, d = 1, 2, 8, 16
    scale = d ** -0.5
    q_cpu = torch.randn(b, h, s, d, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="CUDA"):
        flash_causal_attention(q_cpu, q_cpu, q_cpu, scale=scale)

    q_fp32 = torch.randn(b, h, s, d, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="float16 or bfloat16"):
        flash_causal_attention(q_fp32, q_fp32, q_fp32, scale=scale)


def test_online_softmax_combine_two_chunks():
    """Combine of two partial attentions must match full softmax attention."""
    torch.manual_seed(0)
    b, h, s_q, d = 2, 3, 5, 4
    s0, s1 = 6, 7
    scale = d ** -0.5

    q = torch.randn(b, h, s_q, d)
    k0 = torch.randn(b, h, s0, d)
    v0 = torch.randn(b, h, s0, d)
    k1 = torch.randn(b, h, s1, d)
    v1 = torch.randn(b, h, s1, d)

    scores0 = torch.matmul(q, k0.transpose(-2, -1)) * scale
    lse0 = torch.logsumexp(scores0, dim=-1)
    out0 = torch.matmul(torch.softmax(scores0, dim=-1), v0)

    scores1 = torch.matmul(q, k1.transpose(-2, -1)) * scale
    lse1 = torch.logsumexp(scores1, dim=-1)
    out1 = torch.matmul(torch.softmax(scores1, dim=-1), v1)

    scores_full = torch.cat([scores0, scores1], dim=-1)
    lse_full = torch.logsumexp(scores_full, dim=-1)
    out_full = torch.matmul(torch.softmax(scores_full, dim=-1), torch.cat([v0, v1], dim=2))

    out_c, lse_c = _online_softmax_combine(out0, lse0, out1, lse1)
    torch.testing.assert_close(out_c, out_full, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(lse_c, lse_full, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not flash_attn_available(),
    reason="requires CUDA and flash_attn",
)
def test_ring_cp1_equals_flash():
    """cp_size==1 ring path must match flash_causal_attention (fwd + grad)."""
    from nano_megatron.parallel.attention_backend import flash_causal_attention

    torch.manual_seed(0)
    b, h, s, d = 2, 4, 64, 32
    device = "cuda"
    dtype = torch.bfloat16
    scale = d ** -0.5

    q = torch.randn(b, h, s, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(b, h, s, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(b, h, s, d, device=device, dtype=dtype, requires_grad=True)

    out_flash = flash_causal_attention(q, k, v, scale=scale)
    out_ring = flash_ring_causal_attention(
        q,
        k,
        v,
        scale=scale,
        cp_group=None,
        cp_rank=0,
        cp_size=1,
        backend=None,
    )
    torch.testing.assert_close(out_ring, out_flash, atol=0.0, rtol=0.0)

    dout = torch.randn_like(out_flash)
    g_flash = torch.autograd.grad(out_flash, (q, k, v), dout, retain_graph=True)
    g_ring = torch.autograd.grad(out_ring, (q, k, v), dout)
    for a, b in zip(g_ring, g_flash):
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not flash_attn_available(),
    reason="requires CUDA and flash_attn",
)
def test_ring_rejects_dropout_when_cp_gt_1():
    """flash_ring_causal_attention must raise on dropout_p>0 when cp_size>1."""
    from nano_megatron.parallel.attention_backend import flash_ring_causal_attention

    b, h, s, d = 1, 2, 16, 32
    device = "cuda"
    dtype = torch.bfloat16
    scale = d ** -0.5

    q = torch.randn(b, h, s, d, device=device, dtype=dtype)
    k = torch.randn(b, h, s, d, device=device, dtype=dtype)
    v = torch.randn(b, h, s, d, device=device, dtype=dtype)

    with pytest.raises(RuntimeError, match="does not support dropout_p"):
        flash_ring_causal_attention(
            q,
            k,
            v,
            scale=scale,
            cp_group=None,
            cp_rank=0,
            cp_size=2,
            backend=None,
            dropout_p=0.1,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not flash_attn_available(),
    reason="requires CUDA and flash_attn",
)
def test_chunked_cp_matches_unfused_single_process():
    """Simulated contiguous CP (AG KV + chunked FA) vs unfused query_start path."""
    torch.manual_seed(1)
    b, h, s_full, d = 2, 4, 64, 32
    cp_size = 2
    cp_rank = 1
    s_local = s_full // cp_size
    device = "cuda"
    dtype = torch.bfloat16
    scale = d ** -0.5
    query_start = cp_rank * s_local

    q = torch.randn(b, h, s_local, d, device=device, dtype=dtype, requires_grad=True)
    k_full = torch.randn(b, h, s_full, d, device=device, dtype=dtype, requires_grad=True)
    v_full = torch.randn(b, h, s_full, d, device=device, dtype=dtype, requires_grad=True)

    # Local KV shard as if this rank owned it; gather is identity when we pass full
    # via the internal chunked path by using cp_size=1-style full tensors through
    # flash_ring with cp_size>1 requires a real process group. Exercise the
    # chunk helper via flash_ring_causal_attention only when cp_size==1 above;
    # here call unfused reference and the public chunked apply via local import.
    from nano_megatron.parallel.attention_backend import _flash_chunked_cp_causal_attention

    out_fa = _flash_chunked_cp_causal_attention(
        q, k_full, v_full, scale=scale, cp_rank=cp_rank, cp_size=cp_size
    )
    out_u = unfused_causal_attention(
        q.float(), k_full.float(), v_full.float(), scale=scale, query_start=query_start
    ).to(dtype)

    torch.testing.assert_close(out_fa, out_u, atol=2e-2, rtol=2e-2)

    dout = torch.randn_like(out_fa)
    g_fa = torch.autograd.grad(out_fa, (q, k_full, v_full), dout, retain_graph=True)
    q_u = q.float().detach().requires_grad_(True)
    k_u = k_full.float().detach().requires_grad_(True)
    v_u = v_full.float().detach().requires_grad_(True)
    out_u2 = unfused_causal_attention(
        q_u, k_u, v_u, scale=scale, query_start=query_start
    )
    g_u = torch.autograd.grad(out_u2, (q_u, k_u, v_u), dout.float())
    for a, b in zip(g_fa, g_u):
        torch.testing.assert_close(a.float(), b, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not flash_attn_available(),
    reason="requires CUDA and flash_attn",
)
@pytest.mark.parametrize("cp_rank", [0, 1])
def test_chunked_cp_zigzag_matches_unfused(cp_rank: int):
    """Zigzag CP chunked FA (AG-layout KV) vs unfused per-half query_start."""
    from nano_megatron.parallel.attention_backend import (
        _flash_chunked_cp_causal_attention,
    )
    from nano_megatron.parallel.context_parallel import (
        zigzag_global_to_ag_index,
        zigzag_half_ids,
    )

    torch.manual_seed(2 + cp_rank)
    b, h, s_full, d = 2, 4, 64, 32
    cp_size = 2
    half_len = s_full // (2 * cp_size)  # H = 16
    s_local = 2 * half_len
    device = "cuda"
    dtype = torch.bfloat16
    scale = d ** -0.5
    g0, g1 = zigzag_half_ids(cp_rank, cp_size)

    # Full global Q/K/V; local Q is zigzag concat of two global halves.
    # FA zigzag consumes AG-layout K/V (rank-concat of zigzag locals).
    q_full = torch.randn(b, h, s_full, d, device=device, dtype=dtype)
    k_global = torch.randn(
        b, h, s_full, d, device=device, dtype=dtype, requires_grad=True
    )
    v_global = torch.randn(
        b, h, s_full, d, device=device, dtype=dtype, requires_grad=True
    )
    inv = zigzag_global_to_ag_index(cp_size, half_len, device=device)
    # ag = global[:, :, inv]  (global_to_ag maps AG pos → global index)
    k_ag = k_global.index_select(2, inv).detach().requires_grad_(True)
    v_ag = v_global.index_select(2, inv).detach().requires_grad_(True)

    q_h0 = q_full[:, :, g0 * half_len : (g0 + 1) * half_len, :].contiguous()
    q_h1 = q_full[:, :, g1 * half_len : (g1 + 1) * half_len, :].contiguous()
    q = torch.cat([q_h0, q_h1], dim=2).detach().requires_grad_(True)

    out_fa = _flash_chunked_cp_causal_attention(
        q,
        k_ag,
        v_ag,
        scale=scale,
        cp_rank=cp_rank,
        cp_size=cp_size,
        pack="zigzag",
    )

    out_u0 = unfused_causal_attention(
        q_h0.float(),
        k_global.float(),
        v_global.float(),
        scale=scale,
        query_start=g0 * half_len,
    )
    out_u1 = unfused_causal_attention(
        q_h1.float(),
        k_global.float(),
        v_global.float(),
        scale=scale,
        query_start=g1 * half_len,
    )
    out_u = torch.cat([out_u0, out_u1], dim=2).to(dtype)
    torch.testing.assert_close(out_fa, out_u, atol=2e-2, rtol=2e-2)

    dout = torch.randn_like(out_fa)
    g_fa = torch.autograd.grad(out_fa, (q, k_ag, v_ag), dout, retain_graph=True)

    q0_u = q_h0.float().detach().requires_grad_(True)
    q1_u = q_h1.float().detach().requires_grad_(True)
    k_u = k_global.float().detach().requires_grad_(True)
    v_u = v_global.float().detach().requires_grad_(True)
    out_u0g = unfused_causal_attention(
        q0_u, k_u, v_u, scale=scale, query_start=g0 * half_len
    )
    out_u1g = unfused_causal_attention(
        q1_u, k_u, v_u, scale=scale, query_start=g1 * half_len
    )
    out_ug = torch.cat([out_u0g, out_u1g], dim=2)
    g_u = torch.autograd.grad(
        out_ug, (q0_u, q1_u, k_u, v_u), dout.float()
    )
    dq_u = torch.cat([g_u[0], g_u[1]], dim=2)
    # Unfused dk/dv are global order; FA writes AG layout — map for compare.
    inv_f = inv.to(device=g_u[2].device)
    dk_u_ag = g_u[2].index_select(2, inv_f)
    dv_u_ag = g_u[3].index_select(2, inv_f)
    torch.testing.assert_close(g_fa[0].float(), dq_u, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(g_fa[1].float(), dk_u_ag, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(g_fa[2].float(), dv_u_ag, atol=5e-2, rtol=5e-2)
    assert out_fa.shape == (b, h, s_local, d)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not flash_attn_available(),
    reason="requires CUDA and flash_attn",
)
def test_prefix_vs_legacy_contiguous_rank1():
    """Contiguous CP rank1: prefix outs/grads match legacy within FA tol."""
    from nano_megatron.parallel.attention_backend import (
        _flash_chunked_cp_causal_attention,
    )

    torch.manual_seed(10)
    b, h, s_full, d = 2, 4, 64, 32
    cp_size, cp_rank = 2, 1
    s_local = s_full // cp_size
    device, dtype = "cuda", torch.bfloat16
    scale = d ** -0.5

    q = torch.randn(b, h, s_local, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(b, h, s_full, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(b, h, s_full, d, device=device, dtype=dtype, requires_grad=True)

    out_leg = _flash_chunked_cp_causal_attention(
        q, k, v, scale=scale, cp_rank=cp_rank, cp_size=cp_size, chunk_mode="legacy"
    )
    out_pre = _flash_chunked_cp_causal_attention(
        q, k, v, scale=scale, cp_rank=cp_rank, cp_size=cp_size, chunk_mode="prefix"
    )
    torch.testing.assert_close(out_pre, out_leg, atol=2e-2, rtol=2e-2)

    dout = torch.randn_like(out_leg)
    g_leg = torch.autograd.grad(out_leg, (q, k, v), dout, retain_graph=True)
    g_pre = torch.autograd.grad(out_pre, (q, k, v), dout)
    for a, b_ in zip(g_pre, g_leg):
        torch.testing.assert_close(a, b_, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not flash_attn_available(),
    reason="requires CUDA and flash_attn",
)
@pytest.mark.parametrize("cp_rank", [0, 1])
def test_prefix_vs_legacy_zigzag(cp_rank: int):
    """Zigzag CP: prefix outs/grads match legacy within FA tol."""
    from nano_megatron.parallel.attention_backend import (
        _flash_chunked_cp_causal_attention,
    )
    from nano_megatron.parallel.context_parallel import (
        zigzag_global_to_ag_index,
        zigzag_half_ids,
    )

    torch.manual_seed(20 + cp_rank)
    b, h, s_full, d = 2, 4, 64, 32
    cp_size = 2
    half_len = s_full // (2 * cp_size)
    s_local = 2 * half_len
    device, dtype = "cuda", torch.bfloat16
    scale = d ** -0.5
    g0, g1 = zigzag_half_ids(cp_rank, cp_size)

    q_full = torch.randn(b, h, s_full, d, device=device, dtype=dtype)
    k_global = torch.randn(b, h, s_full, d, device=device, dtype=dtype)
    v_global = torch.randn(b, h, s_full, d, device=device, dtype=dtype)
    inv = zigzag_global_to_ag_index(cp_size, half_len, device=device)
    k_ag = k_global.index_select(2, inv).detach().requires_grad_(True)
    v_ag = v_global.index_select(2, inv).detach().requires_grad_(True)
    q_h0 = q_full[:, :, g0 * half_len : (g0 + 1) * half_len, :].contiguous()
    q_h1 = q_full[:, :, g1 * half_len : (g1 + 1) * half_len, :].contiguous()
    q = torch.cat([q_h0, q_h1], dim=2).detach().requires_grad_(True)

    kw = dict(
        scale=scale, cp_rank=cp_rank, cp_size=cp_size, pack="zigzag"
    )
    out_leg = _flash_chunked_cp_causal_attention(
        q, k_ag, v_ag, chunk_mode="legacy", **kw
    )
    out_pre = _flash_chunked_cp_causal_attention(
        q, k_ag, v_ag, chunk_mode="prefix", **kw
    )
    torch.testing.assert_close(out_pre, out_leg, atol=2e-2, rtol=2e-2)

    dout = torch.randn_like(out_leg)
    g_leg = torch.autograd.grad(out_leg, (q, k_ag, v_ag), dout, retain_graph=True)
    g_pre = torch.autograd.grad(out_pre, (q, k_ag, v_ag), dout)
    for a, b_ in zip(g_pre, g_leg):
        torch.testing.assert_close(a, b_, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not flash_attn_available(),
    reason="requires CUDA and flash_attn",
)
def test_prefix_launch_count_four_blocks(monkeypatch):
    """Zigzag half with g=3 (4 blocks): prefix → 2 FA fwd launches, not 4."""
    import nano_megatron.parallel.attention_backend as ab

    torch.manual_seed(30)
    b, h, s_q, d = 1, 2, 16, 32
    block_len = 16
    n_blocks = 4
    s_k = n_blocks * block_len
    device, dtype = "cuda", torch.bfloat16
    scale = d ** -0.5

    q = torch.randn(b, s_q, h, d, device=device, dtype=dtype)
    k = torch.randn(b, s_k, h, d, device=device, dtype=dtype)
    v = torch.randn(b, s_k, h, d, device=device, dtype=dtype)
    # Contiguous starts 0,16,32,48 — causal on last (standard CP).
    starts = [i * block_len for i in range(n_blocks)]
    causal_block = starts[-1]

    real_fwd = ab._flash_fwd_out_lse
    calls: list[tuple] = []

    def counting_fwd(*args, **kwargs):
        calls.append((args, kwargs))
        return real_fwd(*args, **kwargs)

    monkeypatch.setattr(ab, "_flash_fwd_out_lse", counting_fwd)

    ab._chunked_fa_combine_blocks(
        q,
        k,
        v,
        block_starts=starts,
        block_len=block_len,
        causal_block=causal_block,
        scale=scale,
        dropout_p=0.0,
        mode="legacy",
    )
    assert len(calls) == 4, f"legacy expected 4 launches, got {len(calls)}"
    calls.clear()

    ab._chunked_fa_combine_blocks(
        q,
        k,
        v,
        block_starts=starts,
        block_len=block_len,
        causal_block=causal_block,
        scale=scale,
        dropout_p=0.0,
        mode="prefix",
    )
    assert len(calls) == 2, f"prefix expected 2 launches, got {len(calls)}"
    # First launch: non-causal prefix cat (seq = 3 * block_len)
    _args0, kwargs0 = calls[0]
    assert kwargs0.get("causal") is False or (
        len(_args0) >= 0 and kwargs0.get("causal", None) is False
    )
    k_pref = _args0[1]
    assert k_pref.shape[1] == 3 * block_len
    # Second launch: last block causal
    _args1, kwargs1 = calls[1]
    assert kwargs1.get("causal") is True
    assert _args1[1].shape[1] == block_len


def test_resolve_cp_fa_chunk_mode_env(monkeypatch):
    from nano_megatron.parallel.attention_backend import resolve_cp_fa_chunk_mode

    monkeypatch.delenv("NANO_CP_FA_CHUNK", raising=False)
    assert resolve_cp_fa_chunk_mode(None) == "prefix"
    assert resolve_cp_fa_chunk_mode("legacy") == "legacy"
    monkeypatch.setenv("NANO_CP_FA_CHUNK", "legacy")
    assert resolve_cp_fa_chunk_mode(None) == "legacy"
    with pytest.raises(ValueError, match="CP FA chunk mode"):
        resolve_cp_fa_chunk_mode("bogus")
