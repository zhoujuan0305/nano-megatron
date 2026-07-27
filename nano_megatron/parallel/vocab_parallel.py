"""Vocab-parallel embedding, LM-head helpers, and cross-entropy.

Megatron-style even split: ``vocab_size % tp_size == 0`` is required.
Online CE uses all-reduce of max and sum-exp so training never materializes
full-vocab logits.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nano_megatron.distributed.backend import CommBackend
from nano_megatron.parallel.mappings import _ReduceFromTPRegion


def vocab_range_from_global(
    rank: int, tp_size: int, vocab_size: int
) -> tuple[int, int]:
    """Return ``[start, end)`` vocab indices owned by ``rank``.

    Requires an even split: ``vocab_size`` must be divisible by ``tp_size``.
    """
    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}")
    if rank < 0 or rank >= tp_size:
        raise ValueError(f"rank ({rank}) out of range for tp_size ({tp_size})")
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
    if vocab_size % tp_size != 0:
        raise ValueError(
            f"vocab_size ({vocab_size}) not divisible by tp_size ({tp_size}); "
            "v1 requires an even vocab split"
        )
    per_partition = vocab_size // tp_size
    start = rank * per_partition
    end = start + per_partition
    return start, end


class VocabParallelEmbedding(nn.Module):
    """Embedding with vocab dimension sharded across the TP group.

    Forward masks out-of-range token ids, looks up the local table, zeros
    masked rows, then all-reduces so every rank holds the full embedding.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        tp_rank: int,
        tp_size: int,
        group: Any,
        backend: CommBackend,
        weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.group = group
        self.backend = backend
        self.vocab_start_index, self.vocab_end_index = vocab_range_from_global(
            tp_rank, tp_size, num_embeddings
        )
        local_vocab = self.vocab_end_index - self.vocab_start_index
        if weight is not None:
            if weight.shape != (num_embeddings, embedding_dim):
                raise ValueError(
                    f"weight shape {tuple(weight.shape)} != "
                    f"({num_embeddings}, {embedding_dim})"
                )
            w_local = weight.data[
                self.vocab_start_index : self.vocab_end_index, :
            ].clone()
        else:
            w_local = torch.empty(local_vocab, embedding_dim)
            nn.init.normal_(w_local)
        self.weight = nn.Parameter(w_local)

    def forward(self, input_ids: Tensor) -> Tensor:
        # Map global ids into the local table; out-of-range ids become 0 and
        # are zeroed after lookup so only the owning rank contributes.
        input_mask = (input_ids < self.vocab_start_index) | (
            input_ids >= self.vocab_end_index
        )
        masked_input = input_ids - self.vocab_start_index
        masked_input = masked_input.masked_fill(input_mask, 0)
        output_parallel = F.embedding(masked_input, self.weight)
        output_parallel = output_parallel.masked_fill(
            input_mask.unsqueeze(-1), 0.0
        )
        return _ReduceFromTPRegion.apply(
            output_parallel, self.group, self.backend, None
        )


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """Per-token CE over vocab-sharded logits (Megatron online softmax)."""

    @staticmethod
    def forward(
        ctx: Any,
        vocab_parallel_logits: Tensor,
        target: Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        ignore_index: int,
        group: Any,
        backend: CommBackend,
    ) -> Tensor:
        # logits: [N, local_V], target: [N]
        logits_max = vocab_parallel_logits.max(dim=-1).values
        backend.all_reduce(logits_max, group=group, op="max")

        # Stable exp in local partition.
        logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

        # Predicted logit for the label: only the owning partition is non-zero
        # before the sum all-reduce. Out-of-range ids are forced to local index 0
        # for a valid gather, then zeroed so they do not contribute.
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target = masked_target.masked_fill(target_mask, 0)
        predicted_logits = logits.gather(
            dim=-1, index=masked_target.unsqueeze(dim=-1)
        ).squeeze(dim=-1)
        predicted_logits = predicted_logits.masked_fill(target_mask, 0.0)
        ignore_mask = target == ignore_index
        predicted_logits = predicted_logits.masked_fill(ignore_mask, 0.0)
        backend.all_reduce(predicted_logits, group=group, op="sum")

        exp_logits = logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1)
        backend.all_reduce(sum_exp_logits, group=group, op="sum")

        loss = torch.log(sum_exp_logits) - predicted_logits
        loss = loss.masked_fill(ignore_mask, 0.0)

        # Softmax partition for backward; store 1-hot target mask locally.
        softmax = exp_logits / sum_exp_logits.unsqueeze(dim=-1)
        ctx.vocab_start_index = vocab_start_index
        ctx.vocab_end_index = vocab_end_index
        ctx.ignore_index = ignore_index
        ctx.save_for_backward(softmax, target)
        return loss

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> tuple[Tensor, None, None, None, None, None, None]:
        softmax, target = ctx.saved_tensors
        grad_input = softmax * grad_output.unsqueeze(dim=-1)

        # Subtract grad_output from the target class on the owning partition
        # (softmax - one_hot) * grad_output.
        ignore_mask = target == ctx.ignore_index
        in_partition = (
            (target >= ctx.vocab_start_index)
            & (target < ctx.vocab_end_index)
            & (~ignore_mask)
        )
        masked_target = target - ctx.vocab_start_index
        masked_target = masked_target.masked_fill(~in_partition, 0)
        sub = torch.zeros_like(grad_output)
        sub = torch.where(in_partition, grad_output, sub)
        arange = torch.arange(target.size(0), device=target.device)
        grad_input[arange, masked_target] -= sub
        # Ignored positions should not contribute gradient.
        grad_input = grad_input.masked_fill(ignore_mask.unsqueeze(dim=-1), 0.0)
        return grad_input, None, None, None, None, None, None


def vocab_parallel_cross_entropy(
    vocab_parallel_logits: Tensor,
    target: Tensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    group: Any,
    backend: CommBackend,
    ignore_index: int = -100,
) -> Tensor:
    """Shifted token CE over local-vocab logits; mean over non-ignored tokens.

    Matches ``shifted_cross_entropy`` / ``F.cross_entropy(..., reduction="mean")``
    on the equivalent full-vocab logits: predict ``target[:, 1:]`` from
    ``logits[:, :-1]``.
    """
    if vocab_parallel_logits.dim() != 3:
        raise ValueError(
            f"expected logits [B, S, local_V], got shape {tuple(vocab_parallel_logits.shape)}"
        )
    if target.shape != vocab_parallel_logits.shape[:2]:
        raise ValueError(
            f"target shape {tuple(target.shape)} incompatible with logits "
            f"{tuple(vocab_parallel_logits.shape)}"
        )
    shift_logits = vocab_parallel_logits[:, :-1, :].contiguous()
    shift_labels = target[:, 1:].contiguous()
    local_v = shift_logits.size(-1)
    logits_2d = shift_logits.view(-1, local_v)
    labels_1d = shift_labels.view(-1)
    per_token = _VocabParallelCrossEntropy.apply(
        logits_2d,
        labels_1d,
        vocab_start_index,
        vocab_end_index,
        ignore_index,
        group,
        backend,
    )
    valid = labels_1d != ignore_index
    if not valid.any():
        # Keep a connected graph when every label is ignored.
        return per_token.sum() * 0.0
    return per_token[valid].mean()
