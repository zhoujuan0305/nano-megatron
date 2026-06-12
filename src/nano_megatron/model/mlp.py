from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_hidden_size)
        self.fc2 = nn.Linear(mlp_hidden_size, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.fc2(h)
        return self.dropout(h)


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.down_proj = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x)))
