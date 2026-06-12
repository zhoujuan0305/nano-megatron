import pytest

from nano_megatron.config import GPTConfig, QwenConfig


class TestGPTConfig:
    def test_default_config(self) -> None:
        config = GPTConfig()
        assert config.hidden_size == 128
        assert config.num_layers == 4
        assert config.gradient_accumulation_steps == 1

    def test_override(self) -> None:
        config = GPTConfig(hidden_size=256, num_layers=2)
        assert config.hidden_size == 256
        assert config.num_layers == 2

    def test_effective_batch_size(self) -> None:
        config = GPTConfig(batch_size=4, gradient_accumulation_steps=4)
        assert config.effective_batch_size == 16

    def test_effective_batch_size_no_accum(self) -> None:
        config = GPTConfig(batch_size=4, gradient_accumulation_steps=1)
        assert config.effective_batch_size == 4

    def test_invalid_gradient_accumulation(self) -> None:
        with pytest.raises(ValueError, match="gradient_accumulation_steps"):
            GPTConfig(gradient_accumulation_steps=0)


class TestQwenConfig:
    def test_default_config(self) -> None:
        config = QwenConfig()
        assert config.hidden_size == 128
        assert config.num_layers == 4
        assert config.num_attention_heads == 8
        assert config.num_key_value_heads == 8
        assert config.tie_word_embeddings is True
        assert config.model_type == "qwen"

    def test_override(self) -> None:
        config = QwenConfig(hidden_size=256, num_key_value_heads=4)
        assert config.hidden_size == 256
        assert config.num_key_value_heads == 4

    def test_head_dim(self) -> None:
        config = QwenConfig(hidden_size=128, num_attention_heads=8)
        assert config.head_dim == 16

    def test_num_key_value_groups(self) -> None:
        config = QwenConfig(num_attention_heads=8, num_key_value_heads=4)
        assert config.num_key_value_groups == 2

    def test_kv_heads_divide_heads(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            QwenConfig(num_attention_heads=8, num_key_value_heads=3)

    def test_hidden_size_divide_heads(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            QwenConfig(hidden_size=128, num_attention_heads=5)

    def test_effective_batch_size(self) -> None:
        config = QwenConfig(batch_size=4, gradient_accumulation_steps=2)
        assert config.effective_batch_size == 8

    def test_mlp_hidden_size(self) -> None:
        config = QwenConfig(hidden_size=128, mlp_ratio=4)
        assert config.mlp_hidden_size > 0
