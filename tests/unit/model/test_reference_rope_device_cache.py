"""Reference (TP1) RoPE cache keeps original FP32 phases and checkpoint keys."""
import pytest
import torch
from nano_megatron.reference.config import ReferenceGPTConfig
from nano_megatron.reference.layers import CausalSelfAttention


def make_layer():
    return CausalSelfAttention(ReferenceGPTConfig(
        vocab_size=32, max_seq_len=16, hidden_size=16, num_layers=1,
        num_heads=2, ffn_hidden_size=32, position_embedding_type='rope',
        rotary_dim=8, rope_fp32_phase=True, attn_backend='unfused',
        hidden_dropout=0., attention_dropout=0.,
    ))


def test_reference_rope_phase_remains_fp32_after_model_cast():
    layer=make_layer();original=layer.freqs.clone();keys=set(layer.state_dict())
    layer.bfloat16()
    assert layer.freqs.dtype==torch.float32
    assert torch.equal(layer.freqs,original)
    assert set(layer.state_dict())==keys
    assert not any('freqs' in name for name in layer.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_reference_rope_reuses_device_allocation_and_exact_phase_values():
    layer=make_layer();original=layer.freqs.clone();keys=set(layer.state_dict())
    layer=layer.to(device='cuda',dtype=torch.bfloat16)
    x=torch.randn(1,8,16,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    positions=torch.arange(8,device='cuda').unsqueeze(0)
    first=layer(x,positions);pointer=layer.freqs.data_ptr()
    second=layer(x,positions)
    assert layer.freqs.device==x.device and layer.freqs.dtype==torch.float32
    assert layer.freqs.data_ptr()==pointer
    assert torch.equal(layer.freqs.cpu(),original)
    assert torch.equal(first,second)
    first.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert set(layer.state_dict())==keys


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_reference_rope_moves_back_with_next_input_without_quantizing():
    layer=make_layer();original=layer.freqs.clone()
    layer=layer.cuda();x=torch.randn(1,8,16,device='cuda')
    layer(x,torch.arange(8,device='cuda').unsqueeze(0))
    layer=layer.cpu();layer(x.cpu(),torch.arange(8).unsqueeze(0))
    assert layer.freqs.device.type=='cpu'
    assert torch.equal(layer.freqs,original)
