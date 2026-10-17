"""CPU checks for opt-in dispatch; real CUDA math/autograd is tested separately."""
from dataclasses import replace

import pytest
import torch

from nano_megatron.model import build_tp_gpt_from_reference
from nano_megatron.parallel import ParallelConfig, destroy_parallel, initialize_parallel, is_parallel_initialized
from nano_megatron.reference import ReferenceGPT, ReferenceGPTConfig
from nano_megatron.reference.layers import apply_rotary_emb, swiglu


@pytest.mark.parametrize("fuse_rope,fuse_swiglu", [(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize("tensor_parallel", [False,True])
def test_pointwise_dispatch_is_opt_in(monkeypatch, fuse_rope, fuse_swiglu, tensor_parallel):
    import nano_megatron.parallel.fused_pointwise as helpers
    calls = dict(rope=0, swiglu=0)
    def rotary(x, freqs):
        calls['rope'] += 1
        return apply_rotary_emb(x, freqs)
    def gate(x):
        calls['swiglu'] += 1
        return swiglu(x)
    monkeypatch.setattr(helpers, 'fused_rotary_emb', rotary)
    monkeypatch.setattr(helpers, 'fused_swiglu', gate)
    cfg = ReferenceGPTConfig(vocab_size=16, max_seq_len=8, hidden_size=8, num_heads=2,
                             num_layers=2, ffn_hidden_size=16, use_fused_qkv=True, attn_backend='unfused')
    baseline = ReferenceGPT(cfg)
    enabled = ReferenceGPT(replace(cfg, fuse_rope=fuse_rope, fuse_swiglu=fuse_swiglu))
    enabled.load_state_dict(baseline.state_dict())
    try:
        if tensor_parallel:
            if is_parallel_initialized():
                destroy_parallel()
            for key,value in dict(MASTER_ADDR='127.0.0.1',MASTER_PORT='29979',RANK='0',WORLD_SIZE='1').items():
                monkeypatch.setenv(key,value)
            ctx = initialize_parallel(ParallelConfig(), dist_backend='gloo')
            enabled = build_tp_gpt_from_reference(enabled, ctx)
        ids = torch.tensor([[1,4,7,3]])
        actual, expected = enabled(ids), baseline(ids)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in enabled.parameters())
        assert calls == dict(rope=4 if fuse_rope else 0, swiglu=2 if fuse_swiglu else 0)
    finally:
        if tensor_parallel and is_parallel_initialized():
            destroy_parallel()


def test_disabled_pointwise_flags_preserve_schema1_model_config():
    from scripts.train_gpt import model_config_dict
    cfg = ReferenceGPTConfig()
    serialized = model_config_dict(cfg)
    assert 'fuse_rope' not in serialized and 'fuse_swiglu' not in serialized
    enabled = model_config_dict(replace(cfg, fuse_rope=True, fuse_swiglu=True))
    assert enabled.pop('fuse_rope') is True
    assert enabled.pop('fuse_swiglu') is True
    assert enabled == serialized
