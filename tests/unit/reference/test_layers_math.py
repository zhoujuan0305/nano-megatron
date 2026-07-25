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
    assert torch.isneginf(scores[..., 0, 1]).all()
    assert not torch.isneginf(scores[..., 1, 0]).any()
