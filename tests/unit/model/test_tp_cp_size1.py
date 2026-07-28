"""CP size=1 smoke: TPGPT CP wiring is identity and matches reference."""

from __future__ import annotations

import pytest
import torch

from nano_megatron.model import build_tp_gpt_from_reference
from nano_megatron.parallel import (
    ParallelConfig,
    destroy_parallel,
    initialize_parallel,
    is_parallel_initialized,
)
from nano_megatron.reference import ReferenceGPT, ReferenceGPTConfig
from nano_megatron.reference.loss import shifted_cross_entropy


def _cfg() -> ReferenceGPTConfig:
    return ReferenceGPTConfig(
        vocab_size=16,
        max_seq_len=8,
        hidden_size=8,
        num_layers=2,
        num_heads=4,
        ffn_hidden_size=16,
        layernorm_eps=1e-5,
        use_bias=True,
        tie_word_embeddings=False,
    )


def _init_cp1(monkeypatch, port: str, *, context_parallel_size: int = 1):
    import torch.distributed as dist

    if is_parallel_initialized():
        destroy_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", port)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    return initialize_parallel(
        ParallelConfig(context_parallel_size=context_parallel_size),
        dist_backend="gloo",
    )


def _grads(module):
    return {n: p.grad.detach().clone() for n, p in module.named_parameters()}


def test_tp_cp_size1_stores_cp_fields(monkeypatch):
    ctx = _init_cp1(monkeypatch, "29610")
    assert ctx.context_parallel_size == 1
    assert ctx.context_parallel_rank == 0
    torch.manual_seed(0)
    ref = ReferenceGPT(_cfg())
    tp = build_tp_gpt_from_reference(ref, ctx)
    assert tp._cp_size == 1
    assert tp._cp_rank == 0
    assert tp._cp_group is not None
    for block in tp.blocks:
        assert block.attn._cp_size == 1
        assert block.attn._cp_rank == 0
        assert block.attn._cp_group is not None


def test_tp_cp_size1_forward_matches_reference(monkeypatch):
    """cp=1 path must stay numerically identical to reference (no CP ops)."""
    ctx = _init_cp1(monkeypatch, "29611")
    torch.manual_seed(1)
    ref = ReferenceGPT(_cfg())
    tp = build_tp_gpt_from_reference(ref, ctx)
    ids = torch.randint(0, 16, (2, 6))
    ref_logits = ref(ids)
    tp_logits = tp(ids)
    assert tp_logits.shape == ref_logits.shape
    assert tp_logits.dtype == torch.float32
    assert torch.equal(tp_logits, ref_logits)


def test_tp_cp_size1_loss_and_grads_match_reference(monkeypatch):
    ctx = _init_cp1(monkeypatch, "29612")
    torch.manual_seed(2)
    ref = ReferenceGPT(_cfg())
    tp = build_tp_gpt_from_reference(ref, ctx)
    ids = torch.randint(0, 16, (2, 6))
    tp_loss = tp.shifted_cross_entropy(tp(ids), ids)
    ref_loss = shifted_cross_entropy(ref(ids), ids)
    assert torch.allclose(tp_loss, ref_loss, atol=1e-6, rtol=1e-5)

    tp_loss.backward()
    ref_loss.backward()
    tp_grads = _grads(tp)
    ref_grads = _grads(ref)
    assert set(tp_grads) == set(ref_grads)
    for name in tp_grads:
        assert torch.allclose(
            tp_grads[name], ref_grads[name], atol=1e-6, rtol=1e-5
        ), f"grad mismatch on {name}"


def test_tp_cp_rejects_nondivisible_seq_len(monkeypatch):
    """When cp_size > 1 would apply, seq_len must divide; with world=1 only cp=1
    is valid at init, so exercise the forward guard via a patched size."""
    ctx = _init_cp1(monkeypatch, "29613")
    torch.manual_seed(3)
    ref = ReferenceGPT(_cfg())
    tp = build_tp_gpt_from_reference(ref, ctx)
    # Simulate multi-rank CP config on a single process for the divisibility check.
    tp._cp_size = 2
    for block in tp.blocks:
        block.attn._cp_size = 2
    ids = torch.randint(0, 16, (2, 5))  # 5 % 2 != 0
    with pytest.raises(ValueError, match="context_parallel_size"):
        tp(ids)


def test_tp_cp_local_ce_rejects_local_labels(monkeypatch):
    """Under CP, labels must be full-sequence; local-only labels raise."""
    ctx = _init_cp1(monkeypatch, "29614")
    torch.manual_seed(4)
    ref = ReferenceGPT(_cfg())
    tp = build_tp_gpt_from_reference(ref, ctx)
    tp._cp_size = 2
    tp._cp_rank = 0
    # Local logits [B, S/2, V] with full-seq labels would be OK; local labels not.
    logits = torch.randn(2, 4, 16, requires_grad=True)
    local_labels = torch.randint(0, 16, (2, 4))
    with pytest.raises(ValueError, match="full sequence"):
        tp.shifted_cross_entropy(logits, local_labels)


def test_tp_cp_local_ce_pairs_and_scale(monkeypatch):
    """Local CP CE pairs (logit_i, label_{i+1}) and scales by cp/global_valid."""
    ctx = _init_cp1(monkeypatch, "29615")
    torch.manual_seed(5)
    ref = ReferenceGPT(_cfg())
    tp = build_tp_gpt_from_reference(ref, ctx)
    # Simulate cp=2 rank0 on a fixed full sequence without running CP forward.
    tp._cp_size = 2
    tp._cp_rank = 0
    b, s, v = 2, 8, 16
    full_logits = torch.randn(b, s, v, requires_grad=True)
    labels = torch.randint(0, v, (b, s))
    local = full_logits[:, 0:4, :].contiguous()
    loss_local = tp.shifted_cross_entropy(local, labels)

    # Manual: positions i=0..3 → logit[:, i] vs labels[:, i+1], sum * 2 / (B*(S-1))
    from nano_megatron.parallel.vocab_parallel import (
        vocab_parallel_cross_entropy_from_shifted,
    )
    shift_logits = local[:, :4, :]
    shift_labels = labels[:, 1:5]
    local_sum = vocab_parallel_cross_entropy_from_shifted(
        shift_logits,
        shift_labels,
        vocab_start_index=0,
        vocab_end_index=v,
        group=tp._tp_group,
        backend=tp._tp_backend,
        reduction="sum",
    )
    expected = local_sum * 2.0 / (labels[:, 1:] != -100).sum().to(local_sum.dtype)
    assert torch.allclose(loss_local, expected, atol=1e-6, rtol=1e-5)

    # Mean of rank0+rank1 scaled losses equals full mean CE.
    tp._cp_rank = 1
    local1 = full_logits[:, 4:8, :].contiguous()
    loss1 = tp.shifted_cross_entropy(local1, labels)
    ref_loss = shifted_cross_entropy(full_logits, labels)
    assert torch.allclose(
        (loss_local + loss1) * 0.5, ref_loss, atol=1e-5, rtol=1e-4
    )
