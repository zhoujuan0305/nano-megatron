from __future__ import annotations

import torch

from nano_megatron.model import PipelineStage, build_pipeline_stage_from_reference
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


def test_pipeline_stage_pp1_logits_match_reference(monkeypatch):
    ctx = _init_pp1(monkeypatch, "29610")
    torch.manual_seed(0)
    ref = ReferenceGPT(_cfg())
    stage = build_pipeline_stage_from_reference(ref, ctx)

    assert isinstance(stage, PipelineStage)
    assert stage.is_first_stage is True
    assert stage.is_last_stage is True

    ids = torch.randint(0, 64, (2, 8))
    ref_logits = ref(ids)
    stage_logits = stage(ids)
    assert stage_logits.shape == ref_logits.shape
    assert stage_logits.dtype == torch.float32
    assert torch.equal(stage_logits, ref_logits)


def test_pipeline_stage_pp1_loss_matches_reference(monkeypatch):
    ctx = _init_pp1(monkeypatch, "29611")
    torch.manual_seed(1)
    ref = ReferenceGPT(_cfg())
    stage = build_pipeline_stage_from_reference(ref, ctx)
    ids = torch.randint(0, 64, (2, 8))
    assert torch.allclose(
        stage.shifted_cross_entropy(stage(ids), ids),
        shifted_cross_entropy(ref(ids), ids),
        atol=1e-6,
        rtol=1e-5,
    )


def test_pipeline_stage_pp1_param_values_match_reference(monkeypatch):
    ctx = _init_pp1(monkeypatch, "29612")
    torch.manual_seed(2)
    ref = ReferenceGPT(_cfg())
    stage = build_pipeline_stage_from_reference(ref, ctx)
    ref_params = dict(ref.named_parameters())
    stage_params = dict(stage.named_parameters())
    assert set(ref_params) == set(stage_params)
    for name, rp in ref_params.items():
        assert torch.equal(rp.detach(), stage_params[name].detach()), (
            f"param mismatch on {name}"
        )


def test_pipeline_stage_rejects_tied_embeddings_when_pp_gt_1(monkeypatch):
    """tie_word_embeddings + pp>1 is deferred / rejected."""
    import torch.distributed as dist

    if is_parallel_initialized():
        destroy_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
    # Single-process context with pp_size=2 is invalid for world_size=1.
    # Validate the builder guard directly via a minimal fake ctx.
    from types import SimpleNamespace

    cfg = _cfg()
    cfg.tie_word_embeddings = True
    ref = ReferenceGPT(cfg)
    fake_ctx = SimpleNamespace(
        pipeline_parallel_size=2,
        pipeline_parallel_rank=0,
        tensor_parallel_size=1,
        tensor_parallel_rank=0,
        tensor_parallel_group=None,
        backend=None,
        sequence_parallel=False,
    )
    try:
        build_pipeline_stage_from_reference(ref, fake_ctx)
    except ValueError as e:
        assert "tie_word_embeddings" in str(e)
    else:
        raise AssertionError("expected ValueError for tied embeddings with pp>1")
