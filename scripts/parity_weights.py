"""Explicit weight-layout bridge for reference-only Nano/Megatron parity experiments.

No Megatron imports or training mechanisms are used here. Supported mathematics:
untied, bias-free MHA/RoPE/SwiGLU GPT with LayerNorm (including norm beta).
Megatron must use rotary_interleaved=True to match Nano's rotary coordinates.
"""
from __future__ import annotations

import re
import torch
from torch import Tensor, nn


def _validate(cfg, tp_rank: int, tp_size: int, pp_rank: int, pp_size: int) -> None:
    if tp_size < 1 or pp_size < 1 or not 0 <= tp_rank < tp_size or not 0 <= pp_rank < pp_size:
        raise ValueError("invalid parallel coordinates")
    if cfg.num_layers % pp_size or any(n % tp_size for n in (cfg.vocab_size, cfg.num_heads, cfg.ffn_hidden_size)):
        raise ValueError("model dimensions must divide the parallel topology")
    if (cfg.tie_word_embeddings or cfg.position_embedding_type != "rope" or cfg.use_bias
            or cfg.add_qkv_bias or not cfg.use_fused_qkv or cfg.normalization != "layernorm"
            or not cfg.gated_linear_unit or cfg.num_query_groups != cfg.num_heads):
        raise ValueError("unsupported reference mathematics for the narrow parity bridge")


def pack_qkv(value: Tensor, *, head_dim: int) -> Tensor:
    """[Q_all;K_all;V_all] -> [Q_head0;K_head0;V_head0;Q_head1;...]."""
    if value.shape[0] % (3 * head_dim):
        raise ValueError("invalid fused QKV shape")
    heads = value.shape[0] // (3 * head_dim)
    return value.reshape(3, heads, head_dim, *value.shape[1:]).transpose(0, 1).reshape_as(value).contiguous()


def unpack_qkv(value: Tensor, *, head_dim: int) -> Tensor:
    if value.shape[0] % (3 * head_dim):
        raise ValueError("invalid fused QKV shape")
    heads = value.shape[0] // (3 * head_dim)
    return value.reshape(heads, 3, head_dim, *value.shape[1:]).transpose(0, 1).reshape_as(value).contiguous()


def reference_parameter_shards(reference: nn.Module, *, framework: str, tp_rank: int,
                               tp_size: int, pp_rank: int = 0, pp_size: int = 1) -> dict[str, Tensor]:
    cfg = reference.config
    _validate(cfg, tp_rank, tp_size, pp_rank, pp_size)
    if framework not in ("nano", "megatron"):
        raise ValueError("unknown framework")
    h, f = cfg.hidden_size, cfg.ffn_hidden_size
    q = slice(tp_rank * (h // tp_size), (tp_rank + 1) * (h // tp_size))
    ff = slice(tp_rank * (f // tp_size), (tp_rank + 1) * (f // tp_size))
    v = slice(tp_rank * (cfg.vocab_size // tp_size), (tp_rank + 1) * (cfg.vocab_size // tp_size))
    result = {}
    def put(name, value):
        result[name] = value.detach().cpu().clone()
    if pp_rank == 0:
        put("tok_emb.weight", reference.tok_emb.weight[v])
    per_stage = cfg.num_layers // pp_size
    for local, block in enumerate(reference.blocks[pp_rank * per_stage:(pp_rank + 1) * per_stage]):
        p = f"blocks.{local}."
        for name in ("ln1_weight", "ln1_bias", "ln2_weight", "ln2_bias"):
            put(p + name, getattr(block, name))
        qkv = torch.cat([weight[q] for weight in block.attn.qkv_proj.weight.chunk(3, dim=0)], dim=0)
        put(p + "attn.qkv_proj.weight", qkv)
        put(p + "attn.out_proj.weight", block.attn.out_proj.weight[:, q])
        fc1 = torch.cat([weight[ff] for weight in block.mlp.fc1.weight.chunk(2, dim=0)], dim=0)
        put(p + "mlp.fc1.weight", fc1)
        put(p + "mlp.fc2.weight", block.mlp.fc2.weight[:, ff])
    if pp_rank == pp_size - 1:
        put("ln_f_weight", reference.ln_f_weight)
        put("ln_f_bias", reference.ln_f_bias)
        put("lm_head.weight", reference.lm_head.weight[v])
    if framework == "nano":
        return result
    renamed = {}
    for name, value in result.items():
        key = _nano_to_megatron_name(name)
        renamed[key] = pack_qkv(value, head_dim=h // cfg.num_heads) if name.endswith("attn.qkv_proj.weight") else value
    return renamed


def _nano_to_megatron_name(name: str) -> str:
    fixed = {"tok_emb.weight": "embedding.word_embeddings.weight", "lm_head.weight": "output_layer.weight",
             "ln_f_weight": "decoder.final_layernorm.weight", "ln_f_bias": "decoder.final_layernorm.bias"}
    if name in fixed:
        return fixed[name]
    match = re.fullmatch(r"blocks\.(\d+)\.(.+)", name)
    if not match:
        raise ValueError(f"unmapped parameter: {name}")
    layer, suffix = match.groups()
    mapping = {"ln1_weight": "self_attention.linear_qkv.layer_norm_weight",
               "ln1_bias": "self_attention.linear_qkv.layer_norm_bias",
               "ln2_weight": "mlp.linear_fc1.layer_norm_weight", "ln2_bias": "mlp.linear_fc1.layer_norm_bias",
               "attn.qkv_proj.weight": "self_attention.linear_qkv.weight",
               "attn.out_proj.weight": "self_attention.linear_proj.weight",
               "mlp.fc1.weight": "mlp.linear_fc1.weight", "mlp.fc2.weight": "mlp.linear_fc2.weight"}
    if suffix not in mapping:
        raise ValueError(f"unmapped parameter: {name}")
    return f"decoder.layers.{layer}." + mapping[suffix]


def megatron_to_nano_layout(values: dict[str, Tensor], *, head_dim: int) -> dict[str, Tensor]:
    """Normalize real parameter/gradient/master dictionaries, without ignoring keys."""
    fixed = {"embedding.word_embeddings.weight": "tok_emb.weight", "output_layer.weight": "lm_head.weight",
             "decoder.final_layernorm.weight": "ln_f_weight", "decoder.final_layernorm.bias": "ln_f_bias"}
    suffixes = ["ln1_weight", "ln1_bias", "ln2_weight", "ln2_bias", "attn.qkv_proj.weight",
                "attn.out_proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"]
    inverse = {_nano_to_megatron_name("blocks.0." + suffix).split("decoder.layers.0.", 1)[1]: suffix for suffix in suffixes}
    result = {}
    for name, value in values.items():
        if name in fixed:
            result[fixed[name]] = value
            continue
        match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
        if not match or match.group(2) not in inverse:
            raise ValueError(f"unmapped Megatron parameter: {name}")
        local, suffix = match.groups()
        key = f"blocks.{local}." + inverse[suffix]
        result[key] = unpack_qkv(value, head_dim=head_dim) if suffix == "self_attention.linear_qkv.weight" else value
    return result


@torch.no_grad()
def copy_reference_parameters(reference: nn.Module, module: nn.Module, **coordinates) -> None:
    expected = reference_parameter_shards(reference, **coordinates)
    actual = dict(module.named_parameters())
    if expected.keys() != actual.keys():
        raise ValueError(f"parameter names differ: missing={sorted(actual.keys()-expected.keys())}, extra={sorted(expected.keys()-actual.keys())}")
    for name, value in expected.items():
        if actual[name].shape != value.shape:
            raise ValueError(f"parameter shape differs for {name}: {actual[name].shape} vs {value.shape}")
    for name, value in expected.items():
        actual[name].copy_(value.to(device=actual[name].device, dtype=actual[name].dtype))
