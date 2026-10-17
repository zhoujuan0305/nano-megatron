import pytest
import torch

from nano_megatron.parallel.fused_pointwise import (
    fused_rotary_emb,
    fused_swiglu,
    rotary_reference,
    swiglu_reference,
)
from nano_megatron.reference.layers import apply_rotary_emb, swiglu


def _compiled_close(actual, expected):
    # Fixed before any CUDA outcome: fusion changes low-precision intermediate rounding.
    atol, rtol = {torch.float32: (2e-6, 2e-5), torch.float16: (2e-3, 2e-3),
                  torch.bfloat16: (2e-2, 2e-2)}[actual.dtype]
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rotary_fp32_phase_cpu_matches_literal_megatron_formula(dtype):
    positions = torch.tensor([255, 256, 511, 512, 513, 1023])
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 8, 2).float() / 8))
    freqs = torch.outer(positions.float(), inv_freq).unsqueeze(0)
    x = torch.randn(1, 2, len(positions), 8, dtype=dtype, requires_grad=True)
    phase = freqs.unsqueeze(1)
    cos = torch.cos(phase).to(dtype)
    sin = torch.sin(phase).to(dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    expected = torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)
    actual = rotary_reference(x, freqs, fp32_phase=True)
    torch.testing.assert_close(actual, expected)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    actual_grad = x.grad.clone()
    x.grad = None
    expected.backward(grad)
    torch.testing.assert_close(actual_grad, x.grad)
    if dtype == torch.bfloat16:
        legacy = rotary_reference(x.detach(), freqs, fp32_phase=False)
        assert not torch.equal(legacy[:, :, 4], actual.detach()[:, :, 4])


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
    _compiled_close(actual, expected)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    grad_actual = x.grad.clone()
    x.grad = None
    expected.backward(grad)
    _compiled_close(grad_actual, x.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_compiled_rotary_fp32_phase_output_and_gradient(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    positions = torch.tensor([255, 256, 511, 512, 513, 1023], device="cuda")
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 8, 2, device="cuda").float() / 8))
    freqs = torch.outer(positions.float(), inv_freq).unsqueeze(0)
    x = torch.randn(1, 2, len(positions), 8, device="cuda", dtype=dtype, requires_grad=True)
    phase = freqs.unsqueeze(1)
    cos, sin = torch.cos(phase).to(dtype), torch.sin(phase).to(dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    expected = torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)
    actual = fused_rotary_emb(x, freqs, fp32_phase=True)
    _compiled_close(actual, expected)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    actual_grad = x.grad.clone()
    x.grad = None
    expected.backward(grad)
    _compiled_close(actual_grad, x.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_compiled_swiglu_output_and_gradient(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(4, 32, device="cuda", dtype=dtype).t().contiguous().t().detach().requires_grad_(True)
    assert x.is_leaf and not x.is_contiguous()
    actual = fused_swiglu(x)
    expected = swiglu(x)
    _compiled_close(actual, expected)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    grad_actual = x.grad.clone()
    x.grad = None
    expected.backward(grad)
    _compiled_close(grad_actual, x.grad)
