"""Unit tests for vocab-parallel range helper and cross-entropy (tp=1)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from nano_megatron.parallel.vocab_parallel import (
    vocab_parallel_cross_entropy,
    vocab_range_from_global,
)
from nano_megatron.reference.loss import shifted_cross_entropy


def test_vocab_range_from_global_boundaries():
    assert vocab_range_from_global(0, 1, 16) == (0, 16)
    assert vocab_range_from_global(0, 2, 16) == (0, 8)
    assert vocab_range_from_global(1, 2, 16) == (8, 16)
    assert vocab_range_from_global(3, 4, 32) == (24, 32)


def test_vocab_range_requires_even_split():
    with pytest.raises(ValueError, match="not divisible"):
        vocab_range_from_global(0, 3, 16)


def test_vocab_range_rejects_bad_rank():
    with pytest.raises(ValueError, match="out of range"):
        vocab_range_from_global(2, 2, 16)


class _IdentityBackend:
    """Single-process stub: all_reduce is a no-op identity."""

    def all_reduce(self, tensor, *, group=None, op="sum", async_op=False):
        return tensor

    def all_gather(self, tensor_list, tensor, *, group=None):
        tensor_list[0].copy_(tensor)
        return tensor_list


def test_vocab_parallel_ce_tp1_matches_shifted_cross_entropy():
    torch.manual_seed(0)
    backend = _IdentityBackend()
    b, s, v = 3, 5, 16
    logits = torch.randn(b, s, v, requires_grad=True)
    labels = torch.randint(0, v, (b, s))

    loss_ref = shifted_cross_entropy(logits, labels)
    loss_vp = vocab_parallel_cross_entropy(
        logits,
        labels,
        vocab_start_index=0,
        vocab_end_index=v,
        group=None,
        backend=backend,
    )
    assert torch.allclose(loss_vp, loss_ref, atol=1e-6, rtol=1e-5)

    loss_vp.backward()
    grad_vp = logits.grad.detach().clone()
    logits.grad = None
    loss_ref.backward()
    assert torch.allclose(grad_vp, logits.grad, atol=1e-6, rtol=1e-5)


def test_vocab_parallel_ce_tp1_matches_f_cross_entropy_flat():
    """Sanity: shifted path equals manual F.cross_entropy on shifted tensors."""
    torch.manual_seed(1)
    backend = _IdentityBackend()
    logits = torch.randn(2, 4, 8, requires_grad=True)
    labels = torch.randint(0, 8, (2, 4))
    loss_vp = vocab_parallel_cross_entropy(
        logits,
        labels,
        vocab_start_index=0,
        vocab_end_index=8,
        group=None,
        backend=backend,
    )
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_f = F.cross_entropy(
        shift_logits.view(-1, 8),
        shift_labels.view(-1),
        reduction="mean",
    )
    assert torch.allclose(loss_vp, loss_f, atol=1e-6, rtol=1e-5)


def test_vocab_parallel_ce_respects_ignore_index():
    torch.manual_seed(2)
    backend = _IdentityBackend()
    logits = torch.randn(2, 5, 8, requires_grad=True)
    labels = torch.randint(0, 8, (2, 5))
    labels[:, 2] = -100
    loss_ref = shifted_cross_entropy(logits, labels, ignore_index=-100)
    loss_vp = vocab_parallel_cross_entropy(
        logits,
        labels,
        vocab_start_index=0,
        vocab_end_index=8,
        group=None,
        backend=backend,
        ignore_index=-100,
    )
    assert torch.allclose(loss_vp, loss_ref, atol=1e-6, rtol=1e-5)
