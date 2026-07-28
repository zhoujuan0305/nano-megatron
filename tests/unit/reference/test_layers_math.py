import math

import torch

from nano_megatron.reference.layers import (
    causal_attn_scores,
    gelu_erf,
    layer_norm,
)


def test_layer_norm_matches_manual():
    torch.manual_seed(0)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    w = torch.ones(4)
    b = torch.zeros(4)
    y = layer_norm(x, w, b, eps=1e-5)
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    expected = (x - mean) / torch.sqrt(var + 1e-5)
    assert torch.allclose(y, expected, atol=0, rtol=0)


def test_layer_norm_grad_smoke():
    """Gradients through layer_norm must be finite and match manual formula."""
    torch.manual_seed(1)
    x = torch.randn(2, 8, 16, dtype=torch.float32, requires_grad=True)
    w = torch.randn(16, requires_grad=True)
    b = torch.randn(16, requires_grad=True)

    y = layer_norm(x, w, b, eps=1e-5)
    y.sum().backward()

    # All gradients must be finite
    assert torch.isfinite(x.grad).all(), "input grad has non-finite values"
    assert torch.isfinite(w.grad).all(), "weight grad has non-finite values"
    assert torch.isfinite(b.grad).all(), "bias grad has non-finite values"

    # Cross-check: manual formula grads should match
    x2 = x.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    mean = x2.mean(-1, keepdim=True)
    var = x2.var(-1, unbiased=False, keepdim=True)
    y2 = (x2 - mean) / torch.sqrt(var + 1e-5) * w2 + b2
    y2.sum().backward()

    assert torch.allclose(x.grad, x2.grad, atol=1e-5, rtol=1e-5), "input grad mismatch"
    assert torch.allclose(w.grad, w2.grad, atol=1e-5, rtol=1e-5), "weight grad mismatch"
    assert torch.allclose(b.grad, b2.grad, atol=1e-5, rtol=1e-5), "bias grad mismatch"


def test_gelu_erf_at_zero_and_one():
    x = torch.tensor([0.0, 1.0], dtype=torch.float32)
    y = gelu_erf(x)
    # GELU(0)=0; GELU(1)=0.5*(1+erf(1/sqrt(2)))
    assert y[0].item() == 0.0
    expected_1 = 0.5 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
    assert abs(y[1].item() - expected_1) < 1e-7


def test_causal_mask_upper_is_neg_inf():
    q = torch.randn(1, 2, 2, 4)  # B,H,S,D
    k = torch.randn(1, 2, 2, 4)
    scores = causal_attn_scores(q, k, scale=1.0)
    assert scores.shape == (1, 2, 2, 2)
    # dtype.min (not -inf) keeps bf16/fp16 scores in-dtype for matmul.
    neg = torch.finfo(scores.dtype).min
    assert torch.equal(scores[..., 0, 1], torch.full_like(scores[..., 0, 1], neg))
    assert not (scores[..., 1, 0] == neg).any()
