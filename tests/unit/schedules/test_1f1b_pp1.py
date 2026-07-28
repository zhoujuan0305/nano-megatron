"""pp=1 1F1B schedule: loss and grads match full-batch reference."""
from __future__ import annotations

import torch

from nano_megatron.model import build_pipeline_stage_from_reference
from nano_megatron.parallel import (
    ParallelConfig,
    destroy_parallel,
    initialize_parallel,
    is_parallel_initialized,
)
from nano_megatron.reference import ReferenceGPT, ReferenceGPTConfig
from nano_megatron.reference.loss import shifted_cross_entropy
from nano_megatron.schedules import forward_backward_1f1b


def _cfg() -> ReferenceGPTConfig:
    return ReferenceGPTConfig(
        vocab_size=64,
        max_seq_len=8,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        ffn_hidden_size=64,
        layernorm_eps=1e-5,
        use_bias=True,
        tie_word_embeddings=False,
    )


def _init_pp1(monkeypatch, port: str):
    import torch.distributed as dist

    if is_parallel_initialized():
        destroy_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", port)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    return initialize_parallel(ParallelConfig(), dist_backend="gloo")


def test_1f1b_pp1_loss_and_grads_match_reference(monkeypatch):
    ctx = _init_pp1(monkeypatch, "29650")
    torch.manual_seed(0)
    ref = ReferenceGPT(_cfg())
    stage = build_pipeline_stage_from_reference(ref, ctx)

    batch, seq = 4, 8
    num_microbatches = 2
    assert batch % num_microbatches == 0
    input_ids = torch.randint(0, 64, (batch, seq))
    labels = input_ids.clone()

    # Reference: single forward + backward on the full batch.
    ref.zero_grad(set_to_none=True)
    ref_logits = ref(input_ids)
    ref_loss = shifted_cross_entropy(ref_logits, labels)
    ref_loss.backward()
    ref_grads = {
        name: p.grad.detach().clone()
        for name, p in ref.named_parameters()
        if p.grad is not None
    }

    # Schedule path (pp=1 degenerates to microbatch loop, no P2P).
    stage.zero_grad(set_to_none=True)
    sched_loss = forward_backward_1f1b(
        stage=stage,
        ctx=ctx,
        input_ids=input_ids,
        labels=labels,
        num_microbatches=num_microbatches,
        ddp=None,
    )

    assert sched_loss is not None
    assert torch.allclose(sched_loss, ref_loss, atol=1e-6, rtol=1e-5)

    stage_grads = {
        name: p.grad.detach().clone()
        for name, p in stage.named_parameters()
        if p.grad is not None
    }
    assert set(stage_grads) == set(ref_grads)
    for name, rg in ref_grads.items():
        assert torch.allclose(stage_grads[name], rg, atol=1e-6, rtol=1e-5), (
            f"grad mismatch on {name}: "
            f"max_abs={(stage_grads[name] - rg).abs().max().item()}"
        )


def test_1f1b_pp1_rejects_non_divisible_batch(monkeypatch):
    ctx = _init_pp1(monkeypatch, "29651")
    torch.manual_seed(1)
    ref = ReferenceGPT(_cfg())
    stage = build_pipeline_stage_from_reference(ref, ctx)
    input_ids = torch.randint(0, 64, (3, 8))
    labels = input_ids.clone()
    try:
        forward_backward_1f1b(
            stage=stage,
            ctx=ctx,
            input_ids=input_ids,
            labels=labels,
            num_microbatches=2,
        )
    except ValueError as e:
        assert "num_microbatches" in str(e) or "divis" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for non-divisible batch")
