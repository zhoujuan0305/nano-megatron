import torch

from nano_megatron.model.mlp import MLP
from nano_megatron.random import set_seed


class TestMLP:
    def setup_method(self) -> None:
        set_seed(42)
        self.hidden_size = 128
        self.mlp = MLP(hidden_size=self.hidden_size, mlp_hidden_size=512)

    def test_forward_shape(self) -> None:
        B, S = 2, 32
        x = torch.randn(B, S, self.hidden_size)
        out = self.mlp(x)
        assert out.shape == (B, S, self.hidden_size)

    def test_gradient_flow(self) -> None:
        B, S = 2, 32
        x = torch.randn(B, S, self.hidden_size, requires_grad=True)
        out = self.mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
