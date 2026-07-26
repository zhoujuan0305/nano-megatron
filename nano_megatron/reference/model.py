from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from nano_megatron.reference.config import ReferenceGPTConfig
from nano_megatron.reference.layers import TransformerBlock, layer_norm, rms_norm


class ReferenceGPT(nn.Module):
    """Reference GPT implementation matching Megatron-LM's GPT-3 architecture.
    
    This implementation supports:
    - RoPE (Rotary Position Embedding)
    - SwiGLU activation function
    - Fused QKV projection
    - Group Query Attention (GQA)
    - Optional bias in linear layers
    - LayerNorm or RMSNorm
    """
    def __init__(self, config: ReferenceGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Only use position embedding for learned_absolute type
        if config.position_embedding_type == 'learned_absolute':
            self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_size)
        else:
            self.pos_emb = None
            
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.ln_f_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.ln_f_bias = nn.Parameter(torch.zeros(config.hidden_size))
        self.layernorm_eps = config.layernorm_eps
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.to(dtype=torch.float32)

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor | None = None,
    ) -> Tensor:
        logits, _ = self._forward_impl(input_ids, positions, return_activations=False)
        return logits

    def forward_with_activations(
        self,
        input_ids: Tensor,
        positions: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        return self._forward_impl(input_ids, positions, return_activations=True)

    def _forward_impl(
        self,
        input_ids: Tensor,
        positions: Tensor | None,
        return_activations: bool,
    ) -> tuple[Tensor, dict[str, Any]]:
        batch, seq_len = input_ids.shape
        max_seq_len = self.config.max_seq_len
        if seq_len > max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_seq_len ({max_seq_len})"
            )
        if positions is None:
            positions = torch.arange(seq_len, device=input_ids.device)
            positions = positions.unsqueeze(0).expand(batch, -1)

        # Token embedding
        emb = self.tok_emb(input_ids)
        
        # Position embedding (only for learned_absolute type)
        if self.pos_emb is not None:
            emb = emb + self.pos_emb(positions)
        
        x = emb

        layer_acts: list[dict[str, Any]] = []
        for block in self.blocks:
            if return_activations:
                x, acts = block(x, positions=positions, return_activations=True)
                layer_acts.append(acts)
            else:
                x = block(x, positions=positions, return_activations=False)

        # Final normalization
        if self.config.normalization == 'rmsnorm':
            final_ln = rms_norm(x, self.ln_f_weight, self.layernorm_eps)
        else:
            final_ln = layer_norm(x, self.ln_f_weight, self.ln_f_bias, self.layernorm_eps)
        
        logits = self.lm_head(final_ln)

        if not return_activations:
            return logits, {}

        acts: dict[str, Any] = {
            "emb": emb,
            "layers": layer_acts,
            "final_ln": final_ln,
            "logits": logits,
        }
        return logits, acts
