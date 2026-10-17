from types import SimpleNamespace

import pytest
import torch
from nano_megatron.reference import ReferenceGPT, ReferenceGPTConfig
from nano_megatron.model import build_pipeline_stage_from_reference
from scripts.parity_weights import reference_parameter_shards, pack_qkv, unpack_qkv, megatron_to_nano_layout, copy_reference_parameters


def _reference():
    torch.manual_seed(17)
    return ReferenceGPT(ReferenceGPTConfig(vocab_size=32, max_seq_len=8, hidden_size=16,
        num_layers=4, num_heads=4, ffn_hidden_size=32, use_bias=False, add_qkv_bias=False,
        use_fused_qkv=True, gated_linear_unit=True, activation_func="swiglu", position_embedding_type="rope", rotary_dim=4))


def _context(tp, rank, pp, stage):
    return SimpleNamespace(tensor_parallel_size=tp, tensor_parallel_rank=rank, tensor_parallel_group=None,
        backend=None, sequence_parallel=False, pipeline_parallel_size=pp, pipeline_parallel_rank=stage,
        context_parallel_rank=0, context_parallel_size=1, context_parallel_group=None, context_parallel_pack="contiguous")


@pytest.mark.parametrize("tp,pp", [(1,1),(2,1),(1,2),(2,2)])
def test_all_real_nano_parameter_names_and_values_match_native_builder(tp, pp):
    reference = _reference()
    for rank in range(tp):
        for stage in range(pp):
            built = build_pipeline_stage_from_reference(reference, _context(tp, rank, pp, stage))
            expected = reference_parameter_shards(reference, framework="nano", tp_rank=rank, tp_size=tp,
                                                  pp_rank=stage, pp_size=pp)
            actual = dict(built.named_parameters())
            assert actual.keys() == expected.keys()
            assert all(torch.equal(p, expected[name]) for name, p in actual.items())
            with torch.no_grad():
                for param in actual.values():param.zero_()
            copy_reference_parameters(reference, built, framework="nano", tp_rank=rank, tp_size=tp,
                                      pp_rank=stage, pp_size=pp)
            assert all(torch.equal(p, expected[name]) for name, p in actual.items())


def test_qkv_layout_interleaves_heads_not_individual_channels_and_is_invertible():
    source = torch.arange(24 * 8).reshape(24, 8)
    expected_rows = list(range(0,4))+list(range(8,12))+list(range(16,20))+list(range(4,8))+list(range(12,16))+list(range(20,24))
    packed = pack_qkv(source, head_dim=4)
    assert torch.equal(packed, source[expected_rows])
    assert torch.equal(unpack_qkv(packed, head_dim=4), source)


@pytest.mark.parametrize("tp,pp", [(1,1),(2,1),(1,2),(2,2)])
def test_megatron_layout_roundtrip_preserves_every_norm_mlp_and_stage_parameter(tp, pp):
    reference = _reference()
    for rank in range(tp):
        for stage in range(pp):
            kwargs = dict(tp_rank=rank,tp_size=tp,pp_rank=stage,pp_size=pp)
            nano = reference_parameter_shards(reference, framework="nano", **kwargs)
            meg = reference_parameter_shards(reference, framework="megatron", **kwargs)
            normalized = megatron_to_nano_layout(meg, head_dim=4)
            assert nano.keys() == normalized.keys()
            assert all(torch.equal(value, normalized[name]) for name,value in nano.items())
            assert ("embedding.word_embeddings.weight" in meg) == (stage == 0)
            assert ("output_layer.weight" in meg) == (stage == pp-1)
            assert "decoder.layers.0.mlp.linear_fc1.layer_norm_bias" in meg
            assert "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight" in meg


def test_unknown_names_and_bad_head_layout_rejected():
    with pytest.raises(ValueError):megatron_to_nano_layout({"wrong.weight":torch.zeros(1)},head_dim=4)
    with pytest.raises(ValueError):pack_qkv(torch.zeros(10,4),head_dim=4)
