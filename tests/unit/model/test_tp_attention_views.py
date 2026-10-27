from __future__ import annotations

import pytest
import torch

from nano_megatron.model.tp_gpt import (
    TPCausalSelfAttention,
    _allow_tp_attention_views,
)
from nano_megatron.reference import ReferenceGPTConfig


def _attention(heads: int = 3) -> TPCausalSelfAttention:
    attention = TPCausalSelfAttention.__new__(TPCausalSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.head_dim = 4
    attention.local_num_heads = heads
    return attention


@pytest.mark.parametrize("batch", [1, 2])
def test_split_heads_view_handles_fused_qkv_split_and_random_gradients(batch: int) -> None:
    attention = _attention()
    fused = torch.randn(batch, 5, 3 * 3 * 4, requires_grad=True)
    q, k, v = fused.split([12, 12, 12], dim=-1)
    for block in (q, k, v):
        viewed = attention._split_heads(block, 3, allow_view=True)
        copied = attention._split_heads(block, 3)
        assert (
            viewed.untyped_storage().data_ptr()
            == fused.untyped_storage().data_ptr()
        )
        assert viewed.stride(-1) == 1
        torch.testing.assert_close(viewed, copied)

    weights = torch.randn_like(viewed)
    (viewed * weights).sum().backward(retain_graph=True)
    view_grad = fused.grad.clone()
    fused.grad = None
    (copied * weights).sum().backward()
    torch.testing.assert_close(fused.grad, view_grad)


@pytest.mark.parametrize("batch", [1, 2])
def test_merge_heads_view_and_copy_fallback_gradients(batch: int) -> None:
    attention = _attention()
    source_bshd = torch.randn(batch, 5, 3, 4, requires_grad=True)
    context = source_bshd.transpose(1, 2)
    merged = attention._merge_heads(context, allow_view=True)
    assert merged.shape == (batch, 5, 12)
    assert (
        merged.untyped_storage().data_ptr()
        == source_bshd.untyped_storage().data_ptr()
    )

    weights = torch.randn_like(merged)
    (merged * weights).sum().backward(retain_graph=True)
    view_grad = source_bshd.grad.clone()
    source_bshd.grad = None
    (context.transpose(1, 2).reshape(batch, 5, 12) * weights).sum().backward()
    torch.testing.assert_close(source_bshd.grad, view_grad)

    for awkward in (
        torch.randn(batch, 3, 5, 4),
        torch.randn(batch, 3, 10, 4)[:, :, ::2, :],
    ):
        awkward.requires_grad_()
        fallback = attention._merge_heads(awkward, allow_view=True)
        expected = awkward.transpose(1, 2).contiguous().view(batch, 5, 12)
        assert (
            fallback.untyped_storage().data_ptr()
            != awkward.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(fallback, expected)
        grad = torch.randn_like(fallback)
        (fallback * grad).sum().backward(retain_graph=True)
        actual_grad = awkward.grad.clone()
        awkward.grad = None
        (expected * grad).sum().backward()
        torch.testing.assert_close(awkward.grad, actual_grad)


def test_views_only_enabled_for_flash_cp1() -> None:
    config = ReferenceGPTConfig(tp_attention_views=True)
    assert _allow_tp_attention_views(config, 1, "flash")
    assert not _allow_tp_attention_views(config, 1, "unfused")
    assert not _allow_tp_attention_views(config, 2, "flash")
    assert not _allow_tp_attention_views(ReferenceGPTConfig(), 1, "flash")
