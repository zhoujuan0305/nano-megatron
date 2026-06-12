import pytest
import torch

from nano_megatron.config import GPTConfig
from nano_megatron.model.attention import CausalSelfAttention
from nano_megatron.random import set_seed


class TestCausalSelfAttention:
    def setup_method(self) -> None:
        set_seed(42)
        self.config = GPTConfig()
        self.attn = CausalSelfAttention(
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
        )

    def test_forward_shape(self) -> None:
        B, S = 2, self.config.seq_length
        x = torch.randn(B, S, self.config.hidden_size)
        out = self.attn(x)
        assert out.shape == (B, S, self.config.hidden_size)

    def test_causal_mask_applied(self) -> None:
        B, S = 1, 4
        x = torch.randn(B, S, self.config.hidden_size)
        out = self.attn(x)
        assert not torch.isnan(out).any(), "Output contains NaN"

    def test_gradient_flow(self) -> None:
        B, S = 2, self.config.seq_length
        x = torch.randn(B, S, self.config.hidden_size, requires_grad=True)
        out = self.attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_invalid_hidden_head_ratio(self) -> None:
        with pytest.raises(ValueError):
            CausalSelfAttention(hidden_size=128, num_attention_heads=5)
