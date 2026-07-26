from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import Parameter

from nano_megatron.parallel.context import ParallelContext
from nano_megatron.parallel.mappings import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from nano_megatron.reference.config import ReferenceGPTConfig
from nano_megatron.reference.layers import (
    CausalSelfAttention,
    MLP,
    TransformerBlock,
    causal_attn_scores,
    gelu_erf,
    layer_norm,
    softmax_last,
)
from nano_megatron.reference.model import ReferenceGPT


def _validate_tp_constraints(config: ReferenceGPTConfig, tp_size: int) -> None:
    if tp_size < 1:
        raise ValueError(f"tensor_parallel_size must be >= 1, got {tp_size}")
    if config.num_heads % tp_size != 0:
        raise ValueError(
            f"num_heads ({config.num_heads}) not divisible by tensor_parallel_size ({tp_size})"
        )
    if config.hidden_size % tp_size != 0:
        raise ValueError(
            f"hidden_size ({config.hidden_size}) not divisible by tensor_parallel_size ({tp_size})"
        )
    if config.ffn_hidden_size is None or config.ffn_hidden_size % tp_size != 0:
        raise ValueError(
            f"ffn_hidden_size ({config.ffn_hidden_size}) not divisible by tensor_parallel_size ({tp_size})"
        )


class TPCausalSelfAttention(nn.Module):
    def __init__(self, ref_attn: CausalSelfAttention, ctx: ParallelContext) -> None:
        super().__init__()
        tp = ctx.tensor_parallel_size
        rank = ctx.tensor_parallel_rank
        group = ctx.tensor_parallel_group
        backend = ctx.backend
        self.q_proj = ColumnParallelLinear(
            ref_attn.q_proj.weight, ref_attn.q_proj.bias, rank, tp, group, backend
        )
        self.k_proj = ColumnParallelLinear(
            ref_attn.k_proj.weight, ref_attn.k_proj.bias, rank, tp, group, backend
        )
        self.v_proj = ColumnParallelLinear(
            ref_attn.v_proj.weight, ref_attn.v_proj.bias, rank, tp, group, backend
        )
        self.out_proj = RowParallelLinear(
            ref_attn.out_proj.weight, ref_attn.out_proj.bias, rank, tp, group, backend
        )
        self.local_num_heads = ref_attn.num_heads // tp
        self.head_dim = ref_attn.head_dim
        self.hidden_size = ref_attn.hidden_size
        self.scale = ref_attn.scale

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.local_num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.local_num_heads * self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        scores = causal_attn_scores(q, k, scale=self.scale)
        probs = softmax_last(scores)
        context = torch.matmul(probs, v)
        attn_out = self.out_proj(self._merge_heads(context))
        return attn_out


class TPMLP(nn.Module):
    def __init__(self, ref_mlp: MLP, ctx: ParallelContext) -> None:
        super().__init__()
        tp = ctx.tensor_parallel_size
        rank = ctx.tensor_parallel_rank
        group = ctx.tensor_parallel_group
        backend = ctx.backend
        self.fc1 = ColumnParallelLinear(
            ref_mlp.fc1.weight, ref_mlp.fc1.bias, rank, tp, group, backend
        )
        self.fc2 = RowParallelLinear(
            ref_mlp.fc2.weight, ref_mlp.fc2.bias, rank, tp, group, backend
        )

    def forward(self, x: Tensor) -> Tensor:
        h = gelu_erf(self.fc1(x))
        out = self.fc2(h)
        return out


class TPTransformerBlock(nn.Module):
    def __init__(
        self,
        config: ReferenceGPTConfig,
        ctx: ParallelContext,
        ref_block: TransformerBlock,
    ) -> None:
        super().__init__()
        self.ln1_weight = Parameter(ref_block.ln1_weight.data.clone())
        self.ln1_bias = Parameter(ref_block.ln1_bias.data.clone())
        self.ln2_weight = Parameter(ref_block.ln2_weight.data.clone())
        self.ln2_bias = Parameter(ref_block.ln2_bias.data.clone())
        self.layernorm_eps = config.layernorm_eps
        self.attn = TPCausalSelfAttention(ref_block.attn, ctx)
        self.mlp = TPMLP(ref_block.mlp, ctx)

    def forward(self, x: Tensor) -> Tensor:
        ln1_out = layer_norm(x, self.ln1_weight, self.ln1_bias, self.layernorm_eps)
        attn_out = self.attn(ln1_out)
        resid1 = x + attn_out
        ln2_out = layer_norm(resid1, self.ln2_weight, self.ln2_bias, self.layernorm_eps)
        mlp_out = self.mlp(ln2_out)
        resid2 = resid1 + mlp_out
        return resid2


class TPGPT(nn.Module):
    def __init__(
        self,
        config: ReferenceGPTConfig,
        ctx: ParallelContext,
        ref: ReferenceGPT,
    ) -> None:
        super().__init__()
        _validate_tp_constraints(config, ctx.tensor_parallel_size)
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.tok_emb.weight.data.copy_(ref.tok_emb.weight.data)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.pos_emb.weight.data.copy_(ref.pos_emb.weight.data)
        self.blocks = nn.ModuleList(
            [TPTransformerBlock(config, ctx, rb) for rb in ref.blocks]
        )
        self.ln_f_weight = Parameter(ref.ln_f_weight.data.clone())
        self.ln_f_bias = Parameter(ref.ln_f_bias.data.clone())
        self.layernorm_eps = config.layernorm_eps
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight.data.copy_(ref.lm_head.weight.data)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.to(dtype=torch.float32)

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor | None = None,
    ) -> Tensor:
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_seq_len ({self.config.max_seq_len})"
            )
        if positions is None:
            positions = torch.arange(seq_len, device=input_ids.device)
            positions = positions.unsqueeze(0).expand(batch, -1)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        final_ln = layer_norm(x, self.ln_f_weight, self.ln_f_bias, self.layernorm_eps)
        return self.lm_head(final_ln)


def build_tp_gpt_from_reference(
    ref: ReferenceGPT, ctx: ParallelContext
) -> TPGPT:
    return TPGPT(ref.config, ctx, ref)
