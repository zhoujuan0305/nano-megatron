"""ReferenceGPT flash attention wiring (cp=1)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_reference_gpt_flash_forward_bf16():
    pytest.importorskip("flash_attn")
    from nano_megatron.reference import ReferenceGPT, ReferenceGPTConfig

    cfg = ReferenceGPTConfig(
        vocab_size=128,
        max_seq_len=64,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        ffn_hidden_size=128,
        use_fused_qkv=True,
        attn_backend="flash",
    )
    model = ReferenceGPT(cfg).cuda().to(torch.bfloat16)
    x = torch.randint(0, 128, (2, 64), device="cuda")

    # Prove the flash kernel is selected (not silent unfused fallback).
    with patch(
        "nano_megatron.reference.layers.flash_causal_attention",
        wraps=__import__(
            "nano_megatron.parallel.attention_backend",
            fromlist=["flash_causal_attention"],
        ).flash_causal_attention,
    ) as flash_spy:
        y = model(x)
    assert flash_spy.called, "expected flash_causal_attention to be used"
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_reference_gpt_flash_matches_unfused_bf16():
    """Optional numerical check: flash vs forced-unfused on same weights."""
    pytest.importorskip("flash_attn")
    from nano_megatron.reference import ReferenceGPT, ReferenceGPTConfig

    torch.manual_seed(0)
    cfg_flash = ReferenceGPTConfig(
        vocab_size=64,
        max_seq_len=32,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        ffn_hidden_size=64,
        use_fused_qkv=True,
        attn_backend="flash",
    )
    cfg_unfused = ReferenceGPTConfig(
        vocab_size=64,
        max_seq_len=32,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        ffn_hidden_size=64,
        use_fused_qkv=True,
        attn_backend="unfused",
    )
    flash_model = ReferenceGPT(cfg_flash).cuda().to(torch.bfloat16)
    unfused_model = ReferenceGPT(cfg_unfused).cuda().to(torch.bfloat16)
    unfused_model.load_state_dict(flash_model.state_dict())

    x = torch.randint(0, 64, (2, 32), device="cuda")
    with torch.no_grad():
        y_flash = flash_model(x)
        y_unfused = unfused_model(x)
    assert torch.allclose(y_flash, y_unfused, atol=2e-2, rtol=2e-2)


def test_return_activations_still_has_scores_probs():
    """return_activations must force unfused and expose scores/probs."""
    from nano_megatron.reference import ReferenceGPT, ReferenceGPTConfig

    cfg = ReferenceGPTConfig(
        vocab_size=16,
        max_seq_len=8,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_hidden_size=16,
        attn_backend="flash",  # resolve must still force unfused
    )
    model = ReferenceGPT(cfg)
    ids = torch.randint(0, 16, (1, 4))
    _, acts = model.forward_with_activations(ids)
    attn = acts["layers"][0]["attn"]
    assert "scores" in attn and "probs" in attn
    assert attn["scores"].shape[-1] == 4
    assert attn["probs"].shape == attn["scores"].shape
