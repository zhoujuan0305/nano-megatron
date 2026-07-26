from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class ReferenceGPTConfig:
    """Configuration for ReferenceGPT matching Megatron-LM's GPT-3 architecture.
    
    This configuration is designed to match Megatron-LM's GPT-3 345M model structure:
    - RoPE (Rotary Position Embedding) for position encoding
    - SwiGLU activation function in MLP
    - Fused QKV projection (optional)
    - Optional bias in linear layers
    - LayerNorm with configurable epsilon
    """
    vocab_size: int = 51200
    max_seq_len: int = 1024
    hidden_size: int = 512
    num_layers: int = 12
    num_heads: int = 8
    ffn_hidden_size: int | None = None
    layernorm_eps: float = 1e-5
    use_bias: bool = False  # Megatron-LM default: False
    tie_word_embeddings: bool = False
    dtype: torch.dtype = torch.float32
    
    # Megatron-LM compatible options
    position_embedding_type: Literal['learned_absolute', 'rope'] = 'rope'
    rotary_dim: int | None = None  # Dimension for RoPE, defaults to head_dim
    rotary_base: int = 10000  # Base for RoPE
    activation_func: Literal['gelu', 'swiglu'] = 'swiglu'
    use_fused_qkv: bool = False  # Whether to use fused QKV projection
    num_query_groups: int | None = None  # For Group Query Attention (GQA)
    add_qkv_bias: bool = False  # Add bias to QKV projections only
    gated_linear_unit: bool = True  # Use SwiGLU in MLP
    normalization: Literal['layernorm', 'rmsnorm'] = 'layernorm'
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size
        if self.rotary_dim is None:
            self.rotary_dim = self.hidden_size // self.num_heads
        if self.num_query_groups is None:
            self.num_query_groups = self.num_heads  # Standard MHA
        self.dtype = torch.float32
