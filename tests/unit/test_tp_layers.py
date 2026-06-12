import pytest
import torch

from nano_megatron.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear


class TestColumnParallelLinearSingleGPU:
    def test_output_shape_gather(self) -> None:
        layer = ColumnParallelLinear(128, 64, bias=True, gather_output=True)
        x = torch.randn(2, 16, 128)
        out = layer(x)
        assert out.shape == (2, 16, 64)

    def test_output_shape_no_gather(self) -> None:
        layer = ColumnParallelLinear(128, 64, bias=True, gather_output=False)
        x = torch.randn(2, 16, 128)
        out = layer(x)
        assert out.shape == (2, 16, 64)

    def test_gradient_flow(self) -> None:
        layer = ColumnParallelLinear(128, 64, bias=True, gather_output=True)
        x = torch.randn(2, 16, 128, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.weight.grad is not None

    def test_without_bias(self) -> None:
        layer = ColumnParallelLinear(128, 64, bias=False, gather_output=True)
        assert layer.bias is None
        x = torch.randn(2, 16, 128)
        out = layer(x)
        assert out.shape == (2, 16, 64)


class TestRowParallelLinearSingleGPU:
    def test_output_shape(self) -> None:
        layer = RowParallelLinear(128, 64, bias=True, input_is_parallel=False)
        x = torch.randn(2, 16, 128)
        out = layer(x)
        assert out.shape == (2, 16, 64)

    def test_output_shape_parallel_input(self) -> None:
        layer = RowParallelLinear(128, 64, bias=True, input_is_parallel=True)
        x = torch.randn(2, 16, 128)
        out = layer(x)
        assert out.shape == (2, 16, 64)

    def test_gradient_flow(self) -> None:
        layer = RowParallelLinear(128, 64, bias=True, input_is_parallel=False)
        x = torch.randn(2, 16, 128, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_without_bias(self) -> None:
        layer = RowParallelLinear(128, 64, bias=False, input_is_parallel=False)
        assert layer.bias is None
        x = torch.randn(2, 16, 128)
        out = layer(x)
        assert out.shape == (2, 16, 64)


class TestDivisibilityCheck:
    def test_column_parallel_invalid(self) -> None:
        with pytest.raises(ValueError, match="not divisible"):
            layer = ColumnParallelLinear(128, 7, bias=True, gather_output=True)
            layer.tp_size = 2
            layer.out_features_per_partition = 7 // 2 + 1
            _ensure_divisibility(7, 2)

    def test_row_parallel_invalid(self) -> None:
        with pytest.raises(ValueError, match="not divisible"):
            _ensure_divisibility(7, 2)


def _ensure_divisibility(numerator: int, denominator: int) -> None:
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} is not divisible by {denominator}.")
