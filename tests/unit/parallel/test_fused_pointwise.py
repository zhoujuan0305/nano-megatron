import pytest
import torch

from nano_megatron.parallel.fused_pointwise import (
    fused_rotary_emb,
    fused_swiglu,
    rotary_reference,
    swiglu_reference,
)
from nano_megatron.reference.layers import apply_rotary_emb, swiglu


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rotary_cpu_math_and_gradient(dtype):
    x = torch.randn(2, 3, 4, 8, dtype=dtype, requires_grad=True)
    freqs = torch.randn(2, 4, 4)
    actual = rotary_reference(x, freqs)
    expected = apply_rotary_emb(x, freqs)
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    grad_actual = x.grad.clone()
    x.grad = None
    expected.backward(grad)
    torch.testing.assert_close(grad_actual, x.grad)


def test_swiglu_cpu_math_and_gradient():
    x = torch.randn(3, 12, requires_grad=True)
    actual = swiglu_reference(x)
    expected = swiglu(x)
    torch.testing.assert_close(actual, expected)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    grad_actual = x.grad.clone()
    x.grad = None
    expected.backward(grad)
    torch.testing.assert_close(grad_actual, x.grad)


def test_invalid_shapes_and_empty_inputs():
    with pytest.raises(ValueError, match="even D"):
        rotary_reference(torch.empty(1, 1, 2, 3), torch.empty(1, 2, 1))
    with pytest.raises(ValueError, match="shape"):
        rotary_reference(torch.empty(1, 1, 2, 4), torch.empty(1, 3, 2))
    with pytest.raises(ValueError, match="even"):
        swiglu_reference(torch.empty(2, 3))
    x = torch.empty(1, 2, 0, 4)
    assert rotary_reference(x, torch.empty(1, 0, 2)).shape == x.shape
    assert swiglu_reference(torch.empty(2, 0)).shape == (2, 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 4, 8), (1, 2, 5, 16)])
def test_cuda_compiled_rotary_output_and_gradient(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    freqs = torch.randn(shape[0], shape[2], shape[3] // 2, device="cuda")
    actual = fused_rotary_emb(x, freqs)
    expected = apply_rotary_emb(x, freqs)
    torch.testing.assert_close(actual, expected)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    grad_actual = x.grad.clone()
    x.grad = None
    expected.backward(grad)
    torch.testing.assert_close(grad_actual, x.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_compiled_swiglu_output_and_gradient(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(4, 32, device="cuda", dtype=dtype, requires_grad=True).t().contiguous().t()
    actual = fused_swiglu(x)
    expected = swiglu(x)
    torch.testing.assert_close(actual, expected)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    grad_actual = x.grad.clone()
    x.grad = None
    expected.backward(grad)
    torch.testing.assert_close(grad_actual, x.grad)
