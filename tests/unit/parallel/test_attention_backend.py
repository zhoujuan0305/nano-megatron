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
