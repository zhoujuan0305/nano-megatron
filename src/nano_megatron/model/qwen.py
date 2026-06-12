from __future__ import annotations

import torch
from torch import nn

from nano_megatron.config import QwenConfig
from nano_megatron.model.embedding import TokenEmbedding
from nano_megatron.model.loss import cross_entropy_loss
from nano_megatron.model.mlp import SwiGLUMLP
from nano_megatron.model.norm import RMSNorm
from nano_megatron.model.rope import RotaryEmbedding, apply_rotary_pos_emb


class QwenAttention(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = self.head_dim**0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rotary_embedding = RotaryEmbedding(
            self.head_dim, max_seq_len=config.max_position_embeddings
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_embedding(S, device=x.device, dtype=x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        causal_mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(B, S, self.hidden_size)
        out = self.out_proj(out)
        return self.resid_dropout(out)


class QwenTransformerBlock(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.attention = QwenAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = SwiGLUMLP(
            hidden_size=config.hidden_size,
            mlp_hidden_size=config.mlp_hidden_size,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.attention(x)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


class QwenStyleModel(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [QwenTransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_layernorm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.token_embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.final_layernorm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = cross_entropy_loss(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
            )

        return logits, loss
