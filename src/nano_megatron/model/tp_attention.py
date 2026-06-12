from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from nano_megatron.distributed.parallel_state import get_tensor_parallel_size
from nano_megatron.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear


class TPCausalSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )

        tp_size = (
            get_tensor_parallel_size() if (dist.is_available() and dist.is_initialized()) else 1
        )

        if num_attention_heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"tp_size ({tp_size})"
            )

        self.num_heads = num_attention_heads
        self.num_heads_per_partition = num_attention_heads // tp_size
        self.head_dim = hidden_size // num_attention_heads
        self.hidden_size = hidden_size
        self.scale = self.head_dim**0.5

        self.qkv_proj = ColumnParallelLinear(
            hidden_size, 3 * hidden_size, bias=True, gather_output=False
        )
        self.out_proj = RowParallelLinear(
            hidden_size, hidden_size, bias=True, input_is_parallel=True
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, S, 3, self.num_heads_per_partition, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        causal_mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .reshape(B, S, self.num_heads_per_partition * self.head_dim)
        )
        out = self.out_proj(out)
        return self.resid_dropout(out)
