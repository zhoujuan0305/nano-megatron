from dataclasses import dataclass

import torch


@dataclass
class ReferenceGPTConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int
    num_layers: int
    num_heads: int
    ffn_hidden_size: int | None = None
    layernorm_eps: float = 1e-5
    use_bias: bool = True
    tie_word_embeddings: bool = False
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size
        self.dtype = torch.float32
