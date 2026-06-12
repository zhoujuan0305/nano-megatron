from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids) * (self.hidden_size**0.5)


class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, seq_len: int, device: torch.device | None = None) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        return self.embedding(positions)
