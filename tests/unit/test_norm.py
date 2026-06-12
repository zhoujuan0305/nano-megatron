import torch

from nano_megatron.model.norm import LayerNorm, RMSNorm
from nano_megatron.random import set_seed


class TestLayerNorm:
    def setup_method(self) -> None:
        set_seed(42)

    def test_forward_shape(self) -> None:
        ln = LayerNorm(128)
        x = torch.randn(2, 32, 128)
        out = ln(x)
        assert out.shape == x.shape

    def test_gradient_flow(self) -> None:
        ln = LayerNorm(128)
        x = torch.randn(2, 32, 128, requires_grad=True)
        out = ln(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestRMSNorm:
    def setup_method(self) -> None:
        set_seed(42)

    def test_forward_shape(self) -> None:
        rms = RMSNorm(128)
        x = torch.randn(2, 32, 128)
        out = rms(x)
        assert out.shape == x.shape

    def test_gradient_flow(self) -> None:
        rms = RMSNorm(128)
        x = torch.randn(2, 32, 128, requires_grad=True)
        out = rms(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert rms.weight.grad is not None

    def test_output_not_nan(self) -> None:
        rms = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        out = rms(x)
        assert not torch.isnan(out).any()

    def test_stabilizes_input(self) -> None:
        rms = RMSNorm(64)
        x = torch.randn(2, 16, 64) * 100
        out = rms(x)
        assert out.abs().max() < 100
