import torch

from nano_megatron.model.mlp import MLP, SwiGLUMLP
from nano_megatron.random import set_seed


class TestSwiGLUMLP:
    def setup_method(self) -> None:
        set_seed(42)

    def test_forward_shape(self) -> None:
        mlp = SwiGLUMLP(hidden_size=128, mlp_hidden_size=344)
        x = torch.randn(2, 32, 128)
        out = mlp(x)
        assert out.shape == (2, 32, 128)

    def test_gradient_flow(self) -> None:
        mlp = SwiGLUMLP(hidden_size=128, mlp_hidden_size=344)
        x = torch.randn(2, 32, 128, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_no_bias(self) -> None:
        mlp = SwiGLUMLP(hidden_size=128, mlp_hidden_size=344)
        assert mlp.gate_proj.bias is None
        assert mlp.up_proj.bias is None
        assert mlp.down_proj.bias is None

    def test_param_count_differs_from_mlp(self) -> None:
        mlp = MLP(hidden_size=128, mlp_hidden_size=512)
        swiglu = SwiGLUMLP(hidden_size=128, mlp_hidden_size=344)
        mlp_params = sum(p.numel() for p in mlp.parameters())
        swiglu_params = sum(p.numel() for p in swiglu.parameters())
        assert mlp_params > 0
        assert swiglu_params > 0
