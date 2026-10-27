from __future__ import annotations

import pytest
import torch

from nano_megatron.model.tp_gpt import TPCausalSelfAttention
from nano_megatron.reference import ReferenceGPTConfig


@pytest.mark.parametrize("batch", [1, 2])
def test_split_heads_view_matches_contiguous(batch: int) -> None:
    attention = TPCausalSelfAttention.__new__(TPCausalSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.head_dim = 4
    source = torch.randn(batch, 5, 3 * 4, requires_grad=True)
    viewed = attention._split_heads(source, 3, allow_view=True)
    copied = attention._split_heads(source, 3)
    assert viewed.stride(-1) == 1
    assert not viewed.is_contiguous()
    torch.testing.assert_close(viewed, copied)
    viewed.sum().backward(retain_graph=True)
    view_grad = source.grad.clone()
    source.grad = None
    copied.sum().backward()
    torch.testing.assert_close(source.grad, view_grad)


def test_merge_heads_uses_view_when_reshapeable_and_copies_otherwise() -> None:
    attention = TPCausalSelfAttention.__new__(TPCausalSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.local_num_heads = 3
    attention.head_dim = 4
    source = torch.randn(2, 3, 5, 4)
    merged = attention._merge_heads(source, allow_view=True)
    assert merged.shape == (2, 5, 12)
    assert merged._base is not None
    awkward = torch.randn(2, 3, 10, 4)[:, :, ::2, :]
    fallback = attention._merge_heads(awkward, allow_view=True)
    expected = awkward.transpose(1, 2).contiguous().view(2, 5, 12)
    torch.testing.assert_close(fallback, expected)


def test_tp_attention_views_config_defaults_off() -> None:
    assert ReferenceGPTConfig().tp_attention_views is False
