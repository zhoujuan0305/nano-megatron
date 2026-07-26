from __future__ import annotations

import torch

from nano_megatron.model import TPGPT, build_tp_gpt_from_reference
from nano_megatron.parallel import (
    ParallelConfig,
    destroy_parallel,
    initialize_parallel,
    is_parallel_initialized,
)
from nano_megatron.reference import ReferenceGPT, ReferenceGPTConfig
from nano_megatron.reference.loss import shifted_cross_entropy


def _cfg():
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


def _init_tp1(monkeypatch, port):
    import torch.distributed as dist

    from nano_megatron.parallel import destroy_parallel, initialize_parallel

    if is_parallel_initialized():
        destroy_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", port)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    return initialize_parallel(ParallelConfig(), dist_backend="gloo")


def test_tp_size1_forward_matches_reference(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29530")
    torch.manual_seed(0)
    ref = ReferenceGPT(_cfg())
    tp = build_tp_gpt_from_reference(ref, ctx)
    ids = torch.randint(0, 16, (2, 6))
    ref_logits = ref(ids)
    tp_logits = tp(ids)
    assert tp_logits.shape == ref_logits.shape
    assert tp_logits.dtype == torch.float32
    assert torch.equal(tp_logits, ref_logits)


def test_validate_tp_constraints_raises_on_bad_num_heads(monkeypatch):
    # Direct unit test of the constraint validator; no dist needed.
    from nano_megatron.model.tp_gpt import _validate_tp_constraints

    # num_heads=3 not divisible by tp=2; hidden=6 and ffn=12 are divisible,
    # so the failure is reported specifically on num_heads.
    bad_cfg = ReferenceGPTConfig(
        vocab_size=16,
        max_seq_len=8,
        hidden_size=6,
        num_layers=1,
        num_heads=3,
        ffn_hidden_size=12,
        layernorm_eps=1e-5,
        use_bias=True,
    )
    try:
        _validate_tp_constraints(bad_cfg, tp_size=2)
    except ValueError as e:
        assert "num_heads" in str(e)
    else:
        raise AssertionError("expected ValueError for num_heads not divisible by tp_size")


def test_validate_tp_constraints_raises_on_bad_ffn(monkeypatch):
    from nano_megatron.model.tp_gpt import _validate_tp_constraints

    # num_heads=4 and hidden=8 are divisible by tp=2; ffn=13 is not, so the
    # failure is reported specifically on ffn_hidden_size.
    bad_cfg = ReferenceGPTConfig(
        vocab_size=16,
        max_seq_len=8,
        hidden_size=8,
        num_layers=1,
        num_heads=4,
        ffn_hidden_size=13,
        layernorm_eps=1e-5,
        use_bias=True,
    )
    try:
        _validate_tp_constraints(bad_cfg, tp_size=2)
    except ValueError as e:
        assert "ffn_hidden_size" in str(e)
    else:
        raise AssertionError("expected ValueError for ffn not divisible by tp_size")


def test_tp_size1_loss_matches_reference(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29532")
    torch.manual_seed(1)
    ref = ReferenceGPT(_cfg())
    tp = build_tp_gpt_from_reference(ref, ctx)
    ids = torch.randint(0, 16, (2, 6))
    assert torch.allclose(
        shifted_cross_entropy(tp(ids), ids),
        shifted_cross_entropy(ref(ids), ids),
        atol=0,
        rtol=0,
    )


def _grads(module):
    return {n: p.grad.detach().clone() for n, p in module.named_parameters()}


def test_tp_size1_param_grads_match_reference(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29540")
    torch.manual_seed(2)
    ref = ReferenceGPT(_cfg())
    tp = build_tp_gpt_from_reference(ref, ctx)
    ids = torch.randint(0, 16, (2, 6))
    shifted_cross_entropy(tp(ids), ids).backward()
    shifted_cross_entropy(ref(ids), ids).backward()
    tp_grads = _grads(tp)
    ref_grads = _grads(ref)
    assert set(tp_grads) == set(ref_grads)
    for name in tp_grads:
        assert torch.equal(tp_grads[name], ref_grads[name]), f"grad mismatch on {name}"


def test_tp_size1_param_values_match_reference_after_build(monkeypatch):
    ctx = _init_tp1(monkeypatch, "29541")
    torch.manual_seed(3)
    ref = ReferenceGPT(_cfg())
    tp = build_tp_gpt_from_reference(ref, ctx)
    for (rn, rp), (tn, tpp) in zip(ref.named_parameters(), tp.named_parameters()):
        assert rn == tn
        assert torch.equal(rp.detach(), tpp.detach()), f"param mismatch on {rn}"
