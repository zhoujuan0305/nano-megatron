import pytest

from nano_megatron.config import GPTConfig


class TestGPTConfig:
    def test_default_config(self) -> None:
        config = GPTConfig()
        assert config.hidden_size == 128
        assert config.num_layers == 4
        assert config.num_attention_heads == 8
        assert config.seq_length == 32
        assert config.vocab_size == 1024

    def test_custom_config(self) -> None:
        config = GPTConfig(hidden_size=256, num_layers=2, num_attention_heads=4)
        assert config.hidden_size == 256
        assert config.num_layers == 2
        assert config.num_attention_heads == 4

    def test_head_dim(self) -> None:
        config = GPTConfig(hidden_size=128, num_attention_heads=8)
        assert config.head_dim == 16

    def test_mlp_hidden_size(self) -> None:
        config = GPTConfig(hidden_size=128, mlp_ratio=4)
        assert config.mlp_hidden_size == 512

    def test_invalid_hidden_size_heads(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            GPTConfig(hidden_size=128, num_attention_heads=5)

    def test_invalid_vocab_size(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            GPTConfig(vocab_size=0)

    def test_invalid_num_layers(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            GPTConfig(num_layers=0)

    def test_invalid_hidden_size(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            GPTConfig(hidden_size=0)

    def test_num_parameters(self) -> None:
        config = GPTConfig()
        assert config.num_parameters > 0
