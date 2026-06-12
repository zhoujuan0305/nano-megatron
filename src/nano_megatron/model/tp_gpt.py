from __future__ import annotations

import torch
from torch import nn

from nano_megatron.config import GPTConfig
from nano_megatron.model.embedding import PositionEmbedding, TokenEmbedding
from nano_megatron.model.loss import cross_entropy_loss
from nano_megatron.model.norm import LayerNorm
from nano_megatron.model.tp_attention import TPCausalSelfAttention
from nano_megatron.model.tp_mlp import TPMLP


class TPTransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.input_layernorm = LayerNorm(config.hidden_size)
        self.attention = TPCausalSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout=config.dropout,
        )
        self.post_attention_layernorm = LayerNorm(config.hidden_size)
        self.mlp = TPMLP(
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


class TPGPTModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.position_embedding = PositionEmbedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TPTransformerBlock(config) for _ in range(config.num_layers)])
        self.final_layernorm = LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        seq_len = input_ids.shape[1]
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(seq_len, device=input_ids.device)
        x = self.dropout(x)

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
