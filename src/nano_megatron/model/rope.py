from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self._cos_cache: torch.Tensor | None = None
        self._sin_cache: torch.Tensor | None = None
        self._cached_seq_len = 0

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if seq_len == self._cached_seq_len and self._cos_cache is not None:
            return
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = self.inv_freq
        if isinstance(inv_freq, torch.Tensor):
            inv_freq = inv_freq.to(device)
        freqs = torch.outer(t, inv_freq)  # type: ignore[arg-type]
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cache = emb.cos().to(dtype)
        self._sin_cache = emb.sin().to(dtype)
        self._cached_seq_len = seq_len

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dtype is None:
            dtype = torch.float32
        self._build_cache(seq_len, device, dtype)
        assert self._cos_cache is not None
        assert self._sin_cache is not None
        cos = self._cos_cache[:seq_len].to(device)
        sin = self._sin_cache[:seq_len].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
