from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from nano_megatron.distributed.parallel_state import get_tensor_parallel_size
from nano_megatron.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear


class TPMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        tp_size = (
            get_tensor_parallel_size() if (dist.is_available() and dist.is_initialized()) else 1
        )

        self.fc1 = ColumnParallelLinear(
            hidden_size, mlp_hidden_size, bias=True, gather_output=False
        )
        self.fc2 = RowParallelLinear(
            mlp_hidden_size, hidden_size, bias=True, input_is_parallel=True
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.tp_size = tp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.fc2(h)
        return self.dropout(h)


class TPSwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        tp_size = (
            get_tensor_parallel_size() if (dist.is_available() and dist.is_initialized()) else 1
        )

        if mlp_hidden_size % tp_size != 0 and tp_size > 1:
            raise ValueError(
                f"mlp_hidden_size ({mlp_hidden_size}) must be divisible by tp_size ({tp_size})"
            )

        self.gate_proj = ColumnParallelLinear(
            hidden_size, mlp_hidden_size, bias=False, gather_output=False
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, mlp_hidden_size, bias=False, gather_output=False
        )
        self.down_proj = RowParallelLinear(
            mlp_hidden_size, hidden_size, bias=False, input_is_parallel=True
        )
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        h = gate * up
        return self.dropout(self.down_proj(h))
