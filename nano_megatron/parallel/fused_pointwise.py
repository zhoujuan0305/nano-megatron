"""Small CUDA-compiled pointwise RoPE and SwiGLU helpers."""
from __future__ import annotations

from functools import lru_cache
from types import FunctionType

import torch
from torch import Tensor


def rotary_reference(
    x: Tensor, freqs: Tensor, *, fp32_phase: bool = False
) -> Tensor:
    """Eager RoPE reference matching reference.layers.apply_rotary_emb."""
    if x.ndim != 4 or x.shape[-1] % 2:
        raise ValueError("x must have shape [B, H, S, even D]")
    if freqs.shape != (x.shape[0], x.shape[2], x.shape[-1] // 2):
        raise ValueError("freqs must have shape [B, S, D/2]")
    if freqs.requires_grad:
        raise ValueError("freqs are constants and must not require gradients")
    phase = freqs.float() if fp32_phase else freqs.to(dtype=x.dtype)
    phase = phase.unsqueeze(1)
    cos = torch.cos(phase).to(dtype=x.dtype)
    sin = torch.sin(phase).to(dtype=x.dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    a = even * cos - odd * sin
    b = even * sin + odd * cos
    return torch.stack((a, b), dim=-1).flatten(-2)


def swiglu_reference(projected: Tensor) -> Tensor:
    """Eager SwiGLU reference; projected is concatenated [gate, up]."""
    if projected.ndim < 1 or projected.shape[-1] % 2:
        raise ValueError("projected last dimension must be even")
    gate, up = projected.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def _signature(*tensors: Tensor):
    return tuple((tuple(t.shape), t.stride(), t.dtype, t.device, t.requires_grad) for t in tensors)


@lru_cache(maxsize=32)
def _compiled(name: str, signature):
    template = rotary_reference if name == "rotary" else swiglu_reference
    # Dynamo's specialization limit is per code object. Each static tensor contract
    # gets its own code identity, without resetting/changing global compiler state.
    code = template.__code__.replace(co_name=f"{name}_{hash(signature)}")
    fn = FunctionType(code, template.__globals__, code.co_name, template.__defaults__)
    fn.__kwdefaults__ = template.__kwdefaults__
    return torch.compile(
        fn, fullgraph=True, dynamic=False, backend="inductor",
        options={"triton.cudagraphs": False},
    )


def fused_rotary_emb(
    x: Tensor, freqs: Tensor, *, fp32_phase: bool = False
) -> Tensor:
    """Apply interleaved-pair RoPE with a lazily compiled CUDA pointwise kernel."""
    if not x.is_cuda or not freqs.is_cuda:
        raise ValueError("fused_rotary_emb requires CUDA tensors; use rotary_reference on CPU")
    if x.device != freqs.device:
        raise ValueError("x and freqs must be on the same device")
    return _compiled("rotary", (_signature(x, freqs), fp32_phase))(x, freqs, fp32_phase=fp32_phase)


def fused_swiglu(projected: Tensor) -> Tensor:
    """Apply SiLU(gate) * up with a lazily compiled CUDA pointwise kernel."""
    if not projected.is_cuda:
        raise ValueError("fused_swiglu requires CUDA; use swiglu_reference on CPU")
    return _compiled("swiglu", _signature(projected))(projected)
