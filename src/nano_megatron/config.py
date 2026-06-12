from __future__ import annotations

from dataclasses import dataclass

DEFAULT_HIDDEN_SIZE = 128
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_ATTENTION_HEADS = 8
DEFAULT_SEQ_LENGTH = 32
DEFAULT_VOCAB_SIZE = 1024
DEFAULT_MLP_RATIO = 4
DEFAULT_DROPOUT = 0.0
DEFAULT_MAX_POSITION_EMBEDDINGS = 512
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_STEPS = 100
DEFAULT_WARMUP_STEPS = 10
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1

DEFAULT_NUM_KEY_VALUE_HEADS = 8
DEFAULT_TIE_WORD_EMBEDDINGS = True

QWEN_0_5B: dict[str, int | bool] = {
    "hidden_size": 1024,
    "num_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "seq_length": 32768,
    "vocab_size": 151936,
    "mlp_ratio": 4,
    "max_position_embeddings": 32768,
    "tie_word_embeddings": True,
}

TINY_QWEN: dict[str, int | bool] = {
    "hidden_size": 128,
    "num_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "seq_length": 32,
    "vocab_size": 1024,
    "mlp_ratio": 4,
    "max_position_embeddings": 512,
    "tie_word_embeddings": True,
}


@dataclass
class GPTConfig:
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_layers: int = DEFAULT_NUM_LAYERS
    num_attention_heads: int = DEFAULT_NUM_ATTENTION_HEADS
    seq_length: int = DEFAULT_SEQ_LENGTH
    vocab_size: int = DEFAULT_VOCAB_SIZE
    mlp_ratio: int = DEFAULT_MLP_RATIO
    dropout: float = DEFAULT_DROPOUT
    max_position_embeddings: int = DEFAULT_MAX_POSITION_EMBEDDINGS
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    batch_size: int = DEFAULT_BATCH_SIZE
    num_steps: int = DEFAULT_NUM_STEPS
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    seed: int = 42
    checkpoint_path: str = ""

    model_type: str = "gpt"

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def mlp_hidden_size(self) -> int:
        return self.hidden_size * self.mlp_ratio

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def num_parameters(self) -> int:
        embed_params = self.vocab_size * self.hidden_size
        pos_params = self.max_position_embeddings * self.hidden_size

        per_layer = 0
        per_layer += self.hidden_size * (3 * self.hidden_size + self.hidden_size)
        per_layer += (
            self.hidden_size * (self.hidden_size + self.mlp_hidden_size)
            + self.mlp_hidden_size * self.hidden_size
        )
        per_layer += 2 * (self.hidden_size + self.hidden_size)
        per_layer += self.hidden_size + self.hidden_size

        lm_head_params = self.vocab_size * self.hidden_size
        return embed_params + pos_params + self.num_layers * per_layer + lm_head_params


@dataclass
class QwenConfig:
    hidden_size: int = TINY_QWEN["hidden_size"]
    num_layers: int = TINY_QWEN["num_layers"]
    num_attention_heads: int = TINY_QWEN["num_attention_heads"]
    num_key_value_heads: int = TINY_QWEN["num_key_value_heads"]
    seq_length: int = TINY_QWEN["seq_length"]
    vocab_size: int = TINY_QWEN["vocab_size"]
    mlp_ratio: int = TINY_QWEN["mlp_ratio"]
    dropout: float = DEFAULT_DROPOUT
    max_position_embeddings: int = TINY_QWEN["max_position_embeddings"]
    tie_word_embeddings: bool = bool(TINY_QWEN["tie_word_embeddings"])
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    batch_size: int = DEFAULT_BATCH_SIZE
    num_steps: int = DEFAULT_NUM_STEPS
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    seed: int = 42
    checkpoint_path: str = ""

    model_type: str = "qwen"

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def mlp_hidden_size(self) -> int:
        multiplier = (self.mlp_ratio * 2) // 3
        remainder = (self.mlp_ratio * 2) % 3
        return (
            self.hidden_size * self.mlp_ratio if remainder == 0 else self.hidden_size * multiplier
        )

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def num_parameters(self) -> int:
        embed_params = self.vocab_size * self.hidden_size
        h = self.hidden_size
        mlp_h = self.mlp_hidden_size
        n_heads = self.num_attention_heads
        n_kv_heads = self.num_key_value_heads
        head_d = self.head_dim

        per_layer = 0
        per_layer += h * (n_heads * head_d) + n_heads * head_d * head_d
        per_layer += h * (n_kv_heads * head_d) + n_kv_heads * head_d * head_d
        per_layer += h * (n_kv_heads * head_d) + n_kv_heads * head_d * head_d
        per_layer += h * h

        per_layer += h * mlp_h + mlp_h * h
        per_layer += h * mlp_h + mlp_h * h
        per_layer += mlp_h * h

        per_layer += h + h
        per_layer += h + h

        lm_head_params = self.vocab_size * h if not self.tie_word_embeddings else 0

        return embed_params + self.num_layers * per_layer + lm_head_params + h
