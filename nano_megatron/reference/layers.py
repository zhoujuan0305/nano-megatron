from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nano_megatron.reference.config import ReferenceGPTConfig


def layer_norm(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    eps: float,
) -> Tensor:
    return F.layer_norm(x, (x.size(-1),), weight=weight, bias=bias, eps=eps)


def rms_norm(
    x: Tensor,
    weight: Tensor,
    eps: float,
) -> Tensor:
    """RMSNorm normalization (Megatron-LM compatible)."""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    y = x / rms * weight
    return y


def gelu_erf(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swiglu(x: Tensor) -> Tensor:
    """SwiGLU activation function (Megatron-LM compatible).
    
    SwiGLU(x) = SiLU(x1) * x2 where x is split into two halves.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(x1) * x2


def apply_rotary_emb(
    x: Tensor,
    freqs: Tensor,
) -> Tensor:
    """Apply rotary embeddings to input tensor.
    
    Args:
        x: Input tensor of shape [B, H, S, D]
        freqs: Frequencies tensor of shape [B, S, D/2] (precomputed frequencies)
    
    Returns:
        Tensor with rotary embeddings applied.
    """
    # x: [B, H, S, D], freqs: [B, S, D/2]
    # Reshape freqs to broadcast with x
    # freqs: [B, S, D/2] -> [B, 1, S, D/2]
    freqs = freqs.unsqueeze(1)
    
    cos = torch.cos(freqs)  # [B, 1, S, D/2]
    sin = torch.sin(freqs)  # [B, 1, S, D/2]
    
    # Split x into pairs
    x1, x2 = x[..., ::2], x[..., 1::2]  # Each [B, H, S, D/2]
    
    # Apply rotation
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    
    # Interleave back
    out = torch.stack([out1, out2], dim=-1).flatten(-2)
    return out


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tensor:
    """Precompute frequencies for rotary embeddings (Megatron-LM compatible).
    
    Args:
        dim: Dimension of the embedding (head_dim)
        max_seq_len: Maximum sequence length
        theta: Base for the frequencies
    
    Returns:
        Frequencies tensor of shape [max_seq_len, dim/2]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return freqs


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
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_query_groups = config.num_query_groups
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        bias = config.use_bias
        qkv_bias = config.add_qkv_bias or config.use_bias

        # Support Group Query Attention (GQA)
        self.num_kv_heads = config.num_query_groups
        self.num_heads_per_group = config.num_heads // config.num_query_groups

        if config.use_fused_qkv:
            # Fused QKV projection (Megatron-LM style)
            self.qkv_proj = nn.Linear(
                config.hidden_size,
                config.hidden_size + 2 * self.num_kv_heads * self.head_dim,
                bias=qkv_bias
            )
        else:
            # Separate Q, K, V projections
            self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=qkv_bias)
            self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)
            self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)

        # RoPE (Rotary Position Embedding)
        if config.position_embedding_type == 'rope':
            self.rotary_emb = True
            self.freqs = precompute_freqs_cis(
                config.rotary_dim,
                config.max_seq_len,
                theta=config.rotary_base,
            )
        else:
            self.rotary_emb = False

        self._force_float32()

    def _force_float32(self) -> None:
        self.to(dtype=torch.float32)

    def _split_heads(self, x: Tensor, num_heads: int) -> Tensor:
        # [B, S, H*D] -> [B, H, S, D]
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _merge_heads(self, x: Tensor) -> Tensor:
        # [B, H, S, D] -> [B, S, H*D]
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.hidden_size)

    def _repeat_kv(self, x: Tensor, n_rep: int) -> Tensor:
        """Repeat KV heads for Group Query Attention."""
        if n_rep == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(
        self,
        x: Tensor,
        positions: Tensor | None = None,
        return_activations: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        seq_len = x.size(1)

        if self.config.use_fused_qkv:
            # Fused QKV projection
            qkv = self.qkv_proj(x)
            q, k, v = qkv.split([
                self.hidden_size,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ], dim=-1)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        q = self._split_heads(q, self.num_heads)
        k = self._split_heads(k, self.num_kv_heads)
        v = self._split_heads(v, self.num_kv_heads)

        # Apply RoPE if configured
        if self.rotary_emb and positions is not None:
            # Move freqs to the same device as x
            freqs = self.freqs.to(x.device)
            freqs = freqs[positions]  # [B, S, D]
            q = apply_rotary_emb(q, freqs)
            k = apply_rotary_emb(k, freqs)

        # Repeat KV heads for GQA
        k = self._repeat_kv(k, self.num_heads_per_group)
        v = self._repeat_kv(v, self.num_heads_per_group)

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
        self.config = config
        
        if config.gated_linear_unit:
            # SwiGLU: fc1 outputs 2x ffn_hidden_size for gating
            self.fc1 = nn.Linear(config.hidden_size, 2 * config.ffn_hidden_size, bias=bias)
        else:
            self.fc1 = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=bias)
        
        self.fc2 = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=bias)
        self.to(dtype=torch.float32)

    def forward(
        self,
        x: Tensor,
        return_activations: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        h = self.fc1(x)
        
        if self.config.gated_linear_unit:
            # SwiGLU activation
            h = swiglu(h)
        else:
            # GELU activation
            h = gelu_erf(h)
        
        out = self.fc2(h)
        if not return_activations:
            return out
        return out, {"fc1_gelu": h, "mlp_out": out}


class TransformerBlock(nn.Module):
    def __init__(self, config: ReferenceGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.ln1_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.ln1_bias = nn.Parameter(torch.zeros(config.hidden_size))
        self.ln2_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.ln2_bias = nn.Parameter(torch.zeros(config.hidden_size))
        self.layernorm_eps = config.layernorm_eps
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.to(dtype=torch.float32)

    def _apply_norm(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        """Apply normalization based on config."""
        if self.config.normalization == 'rmsnorm':
            return rms_norm(x, weight, self.layernorm_eps)
        else:
            return layer_norm(x, weight, bias, self.layernorm_eps)

    def forward(
        self,
        x: Tensor,
        positions: Tensor | None = None,
        return_activations: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        ln1_out = self._apply_norm(x, self.ln1_weight, self.ln1_bias)
        if return_activations:
            attn_out, attn_acts = self.attn(ln1_out, positions=positions, return_activations=True)
        else:
            attn_out = self.attn(ln1_out, positions=positions, return_activations=False)
            attn_acts = None
        resid1 = x + attn_out

        ln2_out = self._apply_norm(resid1, self.ln2_weight, self.ln2_bias)
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
