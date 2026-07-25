from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from nano_megatron.reference.config import ReferenceGPTConfig


def layer_norm(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    eps: float,
) -> Tensor:
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    y = (x - mean) / torch.sqrt(var + eps)
    y = y * weight
    if bias is not None:
        y = y + bias
    return y


def gelu_erf(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def causal_attn_scores(q: Tensor, k: Tensor, scale: float) -> Tensor:
    # q, k: [B, H, S, D] -> scores: [B, H, S, S]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    seq_len = scores.size(-1)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    return scores


def softmax_last(x: Tensor) -> Tensor:
    return torch.softmax(x, dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ReferenceGPTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_heads ({config.num_heads})"
            )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        bias = config.use_bias

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)

        self._force_float32()

    def _force_float32(self) -> None:
        self.to(dtype=torch.float32)

    def _split_heads(self, x: Tensor) -> Tensor:
        # [B, S, H*D] -> [B, H, S, D]
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _merge_heads(self, x: Tensor) -> Tensor:
        # [B, H, S, D] -> [B, S, H*D]
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.hidden_size)

    def forward(
        self,
        x: Tensor,
        return_activations: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = causal_attn_scores(q, k, scale=self.scale)
        probs = softmax_last(scores)
        context = torch.matmul(probs, v)
        attn_out = self.out_proj(self._merge_heads(context))

        if not return_activations:
            return attn_out

        acts: dict[str, Tensor] = {
            "q": q,
            "k": k,
            "v": v,
            "scores": scores,
            "probs": probs,
            "context": context,
            "attn_out": attn_out,
        }
        return attn_out, acts


class MLP(nn.Module):
    def __init__(self, config: ReferenceGPTConfig) -> None:
        super().__init__()
        assert config.ffn_hidden_size is not None
        bias = config.use_bias
        self.fc1 = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=bias)
        self.fc2 = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=bias)
        self.to(dtype=torch.float32)

    def forward(
        self,
        x: Tensor,
        return_activations: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        h = gelu_erf(self.fc1(x))
        out = self.fc2(h)
        if not return_activations:
            return out
        return out, {"fc1_gelu": h, "mlp_out": out}


class TransformerBlock(nn.Module):
    def __init__(self, config: ReferenceGPTConfig) -> None:
        super().__init__()
        self.ln1_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.ln1_bias = nn.Parameter(torch.zeros(config.hidden_size))
        self.ln2_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.ln2_bias = nn.Parameter(torch.zeros(config.hidden_size))
        self.layernorm_eps = config.layernorm_eps
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.to(dtype=torch.float32)

    def forward(
        self,
        x: Tensor,
        return_activations: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        ln1_out = layer_norm(x, self.ln1_weight, self.ln1_bias, self.layernorm_eps)
        if return_activations:
            attn_out, attn_acts = self.attn(ln1_out, return_activations=True)
        else:
            attn_out = self.attn(ln1_out, return_activations=False)
            attn_acts = None
        resid1 = x + attn_out

        ln2_out = layer_norm(
            resid1, self.ln2_weight, self.ln2_bias, self.layernorm_eps
        )
        if return_activations:
            mlp_out, mlp_acts = self.mlp(ln2_out, return_activations=True)
        else:
            mlp_out = self.mlp(ln2_out, return_activations=False)
            mlp_acts = None
        resid2 = resid1 + mlp_out

        if not return_activations:
            return resid2

        acts: dict[str, Any] = {
            "ln1_out": ln1_out,
            "attn_out": attn_out,
            "resid1": resid1,
            "ln2_out": ln2_out,
            "mlp_out": mlp_out,
            "resid2": resid2,
        }
        if attn_acts is not None:
            acts["attn"] = attn_acts
        if mlp_acts is not None:
            acts["mlp"] = mlp_acts
        return resid2, acts
