from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import Parameter

from nano_megatron.parallel.context import ParallelContext
from nano_megatron.parallel.mappings import (
    ColumnParallelLinear,
    RowParallelLinear,
    blockwise_column_shard,
    fused_qkv_column_shard,
)
from nano_megatron.parallel.vocab_parallel import (
    VocabParallelEmbedding,
    vocab_parallel_cross_entropy,
    vocab_range_from_global,
)
from nano_megatron.reference.config import ReferenceGPTConfig
from nano_megatron.reference.layers import (
    CausalSelfAttention,
    MLP,
    TransformerBlock,
    apply_rotary_emb,
    causal_attn_scores,
    gelu_erf,
    layer_norm,
    precompute_freqs_cis,
    rms_norm,
    softmax_last,
    swiglu,
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
    if config.vocab_size % tp_size != 0:
        raise ValueError(
            f"vocab_size ({config.vocab_size}) not divisible by tensor_parallel_size ({tp_size})"
        )


class TPCausalSelfAttention(nn.Module):
    def __init__(self, ref_attn: CausalSelfAttention, ctx: ParallelContext) -> None:
        super().__init__()
        tp = ctx.tensor_parallel_size
        rank = ctx.tensor_parallel_rank
        group = ctx.tensor_parallel_group
        backend = ctx.backend
        self.config = ref_attn.config
        
        # Support GQA (Group Query Attention)
        self.num_heads = ref_attn.num_heads
        self.num_kv_heads = ref_attn.num_kv_heads
        self.num_heads_per_group = ref_attn.num_heads_per_group
        self.head_dim = ref_attn.head_dim
        self.hidden_size = ref_attn.hidden_size
        self.scale = ref_attn.scale
        self.local_num_heads = self.num_heads // tp
        self.local_num_kv_heads = self.num_kv_heads // tp

        if self.config.use_fused_qkv:
            # Fused weight is [Q; K; V]; shard each block so local layout is
            # [Q_r; K_r; V_r] matching the forward split below.
            q_dim = self.hidden_size
            kv_dim = self.num_kv_heads * self.head_dim
            w_local, b_local = fused_qkv_column_shard(
                ref_attn.qkv_proj.weight,
                ref_attn.qkv_proj.bias,
                rank,
                tp,
                q_dim,
                kv_dim,
            )
            self.qkv_proj = ColumnParallelLinear(
                w_local,
                b_local,
                rank,
                tp,
                group,
                backend,
                weight_is_local=True,
            )
        else:
            # Separate Q, K, V projections
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

        # RoPE
        if self.config.position_embedding_type == 'rope':
            self.rotary_emb = True
            self.freqs = precompute_freqs_cis(
                self.config.rotary_dim,
                self.config.max_seq_len,
                theta=self.config.rotary_base,
            )
        else:
            self.rotary_emb = False

    def _split_heads(self, x: Tensor, num_heads: int) -> Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.local_num_heads * self.head_dim)

    def _repeat_kv(self, x: Tensor, n_rep: int) -> Tensor:
        """Repeat KV heads for Group Query Attention."""
        if n_rep == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        if self.config.use_fused_qkv:
            # Fused QKV projection
            qkv = self.qkv_proj(x)
            q_size = self.local_num_heads * self.head_dim
            kv_size = self.local_num_kv_heads * self.head_dim
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        q = self._split_heads(q, self.local_num_heads)
        k = self._split_heads(k, self.local_num_kv_heads)
        v = self._split_heads(v, self.local_num_kv_heads)

        # Apply RoPE if configured
        if self.rotary_emb and positions is not None:
            # Move freqs to the same device as x
            freqs = self.freqs.to(x.device)
            freqs = freqs[positions]
            q = apply_rotary_emb(q, freqs)
            k = apply_rotary_emb(k, freqs)

        # Repeat KV heads for GQA
        k = self._repeat_kv(k, self.num_heads_per_group)
        v = self._repeat_kv(v, self.num_heads_per_group)

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
        self.config = ref_mlp.config
        
        if self.config.gated_linear_unit:
            # SwiGLU fc1 is [gate; up]; shard each half so local is [gate_r; up_r]
            # and swiglu's chunk(2) stays correct under TP.
            ffn = ref_mlp.fc1.weight.shape[0] // 2
            w_local, b_local = blockwise_column_shard(
                ref_mlp.fc1.weight, ref_mlp.fc1.bias, rank, tp, (ffn, ffn)
            )
            self.fc1 = ColumnParallelLinear(
                w_local,
                b_local,
                rank,
                tp,
                group,
                backend,
                weight_is_local=True,
            )
        else:
            self.fc1 = ColumnParallelLinear(
                ref_mlp.fc1.weight, ref_mlp.fc1.bias, rank, tp, group, backend
            )
        
        self.fc2 = RowParallelLinear(
            ref_mlp.fc2.weight, ref_mlp.fc2.bias, rank, tp, group, backend
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc1(x)
        
        if self.config.gated_linear_unit:
            # SwiGLU activation
            h = swiglu(h)
        else:
            # GELU activation
            h = gelu_erf(h)
        
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
        self.config = config
        self.ln1_weight = Parameter(ref_block.ln1_weight.data.clone())
        self.ln1_bias = Parameter(ref_block.ln1_bias.data.clone())
        self.ln2_weight = Parameter(ref_block.ln2_weight.data.clone())
        self.ln2_bias = Parameter(ref_block.ln2_bias.data.clone())
        self.layernorm_eps = config.layernorm_eps
        self.attn = TPCausalSelfAttention(ref_block.attn, ctx)
        self.mlp = TPMLP(ref_block.mlp, ctx)

    def _apply_norm(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        """Apply normalization based on config."""
        if self.config.normalization == 'rmsnorm':
            return rms_norm(x, weight, self.layernorm_eps)
        else:
            return layer_norm(x, weight, bias, self.layernorm_eps)

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        ln1_out = self._apply_norm(x, self.ln1_weight, self.ln1_bias)
        attn_out = self.attn(ln1_out, positions=positions)
        resid1 = x + attn_out
        ln2_out = self._apply_norm(resid1, self.ln2_weight, self.ln2_bias)
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
        self._tp_group = ctx.tensor_parallel_group
        self._tp_backend = ctx.backend
        self.vocab_start_index, self.vocab_end_index = vocab_range_from_global(
            ctx.tensor_parallel_rank,
            ctx.tensor_parallel_size,
            config.vocab_size,
        )
        # Vocab-parallel embedding (tp=1 is a full table + identity all-reduce).
        self.tok_emb = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=ctx.tensor_parallel_rank,
            tp_size=ctx.tensor_parallel_size,
            group=ctx.tensor_parallel_group,
            backend=ctx.backend,
            weight=ref.tok_emb.weight,
        )

        # Only use position embedding for learned_absolute type
        if config.position_embedding_type == 'learned_absolute':
            self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_size)
            self.pos_emb.weight.data.copy_(ref.pos_emb.weight.data)
        else:
            self.pos_emb = None

        self.blocks = nn.ModuleList(
            [TPTransformerBlock(config, ctx, rb) for rb in ref.blocks]
        )
        self.ln_f_weight = Parameter(ref.ln_f_weight.data.clone())
        self.ln_f_bias = Parameter(ref.ln_f_bias.data.clone())
        self.layernorm_eps = config.layernorm_eps
        # Column-parallel LM head: local logits [B, S, V/tp], no gather in forward.
        self.lm_head = ColumnParallelLinear(
            ref.lm_head.weight,
            None,
            ctx.tensor_parallel_rank,
            ctx.tensor_parallel_size,
            ctx.tensor_parallel_group,
            ctx.backend,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.to(dtype=torch.float32)

    def shifted_cross_entropy(
        self,
        logits: Tensor,
        labels: Tensor,
        ignore_index: int = -100,
    ) -> Tensor:
        """Vocab-parallel shifted CE matching reference mean reduction."""
        return vocab_parallel_cross_entropy(
            logits,
            labels,
            vocab_start_index=self.vocab_start_index,
            vocab_end_index=self.vocab_end_index,
            group=self._tp_group,
            backend=self._tp_backend,
            ignore_index=ignore_index,
        )

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

        # Token embedding
        x = self.tok_emb(input_ids)

        # Position embedding (only for learned_absolute type)
        if self.pos_emb is not None:
            x = x + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x, positions=positions)

        # Final normalization
        if self.config.normalization == 'rmsnorm':
            final_ln = rms_norm(x, self.ln_f_weight, self.layernorm_eps)
        else:
            final_ln = layer_norm(x, self.ln_f_weight, self.ln_f_bias, self.layernorm_eps)

        return self.lm_head(final_ln)


def build_tp_gpt_from_reference(
    ref: ReferenceGPT, ctx: ParallelContext
) -> TPGPT:
    return TPGPT(ref.config, ctx, ref)
