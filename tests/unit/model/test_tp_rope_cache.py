from copy import deepcopy
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from nano_megatron.model.tp_gpt import TPCausalSelfAttention
from nano_megatron.reference.layers import precompute_freqs_cis


def _attention_with_freqs() -> TPCausalSelfAttention:
    attention = TPCausalSelfAttention.__new__(TPCausalSelfAttention)
    nn.Module.__init__(attention)
    attention.rotary_emb = True
    attention.freqs = precompute_freqs_cis(8, 16, theta=10000.0)
    attention._rotary_freqs_cache = None
    attention._rotary_freqs_cache_device = None
    return attention


def test_rotary_freqs_cache_reuses_storage_and_keeps_canonical_table():
    attention = _attention_with_freqs()
    canonical = attention.freqs
    cached = attention._get_rotary_freqs("cpu")

    assert attention._get_rotary_freqs(torch.device("cpu")) is cached
    assert attention.freqs is canonical
    assert torch.equal(attention.freqs, canonical)
    assert not cached.requires_grad
    assert "freqs" not in attention.state_dict()
    assert "_rotary_freqs_cache" not in attention.state_dict()


def test_rotary_freqs_cache_is_cleared_by_dtype_and_device_apply():
    attention = _attention_with_freqs()
    canonical = attention.freqs
    attention._get_rotary_freqs("cpu")

    attention.bfloat16()
    assert attention._rotary_freqs_cache is None
    assert attention.freqs is canonical
    assert attention.freqs.dtype == torch.float32
    assert torch.equal(attention.freqs, canonical)
    attention.float()
    assert attention._rotary_freqs_cache is None
    assert attention.freqs.dtype == torch.float32


def test_rotary_positions_index_full_cached_table():
    attention = _attention_with_freqs()
    positions = torch.tensor([[0, 3], [7, 11]])
    selected = attention._get_rotary_freqs("cpu")[positions]
    assert torch.equal(selected, attention.freqs[positions])
    assert selected.shape == (2, 2, attention.freqs.shape[-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_rotary_freqs_cache_moves_to_cuda_and_back():
    attention = _attention_with_freqs()
    canonical = attention.freqs
    cuda_freqs = attention._get_rotary_freqs("cuda")
    assert cuda_freqs.device.type == "cuda"
    assert torch.equal(cuda_freqs.cpu(), canonical)
    assert attention._get_rotary_freqs("cuda") is cuda_freqs

    attention.to("cuda")
    assert attention._rotary_freqs_cache is None
    assert attention.freqs is canonical
    assert torch.equal(attention._get_rotary_freqs("cuda").cpu(), canonical)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_attention_matches_uncached_output_and_gradients(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from nano_megatron.distributed import TorchDistBackend
    from nano_megatron.reference.config import ReferenceGPTConfig
    from nano_megatron.reference.layers import CausalSelfAttention

    config = ReferenceGPTConfig(
        hidden_size=32, num_heads=4, num_layers=1, max_seq_len=16,
        position_embedding_type="rope", rotary_dim=8,
        use_fused_qkv=True, attn_backend="unfused",
    )
    context = SimpleNamespace(
        tensor_parallel_size=1, tensor_parallel_rank=0,
        tensor_parallel_group=None, backend=TorchDistBackend(),
        sequence_parallel=False, context_parallel_rank=0,
        context_parallel_size=1, context_parallel_group=None,
        context_parallel_pack="contiguous",
    )
    torch.manual_seed(19)
    cached = TPCausalSelfAttention(CausalSelfAttention(config), context).to(
        device=device, dtype=dtype
    )
    uncached = deepcopy(cached)
    uncached._get_rotary_freqs = MethodType(
        lambda self, target: self.freqs.to(target), uncached
    )
    positions = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]], device=device)
    x = torch.randn(2, 4, 32, device=device, dtype=dtype, requires_grad=True)
    other_x = x.detach().clone().requires_grad_(True)
    actual = cached(x, positions)
    expected = uncached(other_x, positions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    torch.testing.assert_close(x.grad, other_x.grad, rtol=0, atol=0)
    for left, right in zip(cached.parameters(), uncached.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    table = cached._get_rotary_freqs(x.device)
    assert cached._get_rotary_freqs(x.device) is table
    assert table.dtype == torch.float32
