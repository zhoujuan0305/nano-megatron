"""Small CUDA-compiled pointwise RoPE and SwiGLU helpers."""
from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor


def rotary_reference(x: Tensor, freqs: Tensor) -> Tensor:
    """Eager RoPE reference matching reference.layers.apply_rotary_emb."""
    if x.ndim != 4 or x.shape[-1] % 2:
        raise ValueError("x must have shape [B, H, S, even D]")
    if freqs.shape != (x.shape[0], x.shape[2], x.shape[-1] // 2):
        raise ValueError("freqs must have shape [B, S, D/2]")
    if freqs.requires_grad:
        raise ValueError("freqs are constants and must not require gradients")
    phase = freqs.to(dtype=x.dtype).unsqueeze(1)
    even, odd = x[..., ::2], x[..., 1::2]
    a = even * torch.cos(phase) - odd * torch.sin(phase)
    b = even * torch.sin(phase) + odd * torch.cos(phase)
    return torch.stack((a, b), dim=-1).flatten(-2)


def swiglu_reference(projected: Tensor) -> Tensor:
    """Eager SwiGLU reference; projected is concatenated [gate, up]."""
    if projected.ndim < 1 or projected.shape[-1] % 2:
        raise ValueError("projected last dimension must be even")
    gate, up = projected.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


@lru_cache(maxsize=2)
def _compiled(name: str):
    fn = rotary_reference if name == "rotary" else swiglu_reference
    return torch.compile(
        fn, fullgraph=True, dynamic=False, backend="inductor",
        options={"triton.cudagraphs": False},
    )


def fused_rotary_emb(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply interleaved-pair RoPE with a lazily compiled CUDA pointwise kernel."""
    if not x.is_cuda or not freqs.is_cuda:
        raise ValueError("fused_rotary_emb requires CUDA tensors; use rotary_reference on CPU")
    if x.device != freqs.device:
        raise ValueError("x and freqs must be on the same device")
    return _compiled("rotary")(x, freqs)


def fused_swiglu(projected: Tensor) -> Tensor:
    """Apply SiLU(gate) * up with a lazily compiled CUDA pointwise kernel."""
    if not projected.is_cuda:
        raise ValueError("fused_swiglu requires CUDA; use swiglu_reference on CPU")
    return _compiled("swiglu")(projected)
