"""Pipeline stage: local emb / blocks / head for one PP rank."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import Parameter

from nano_megatron.model.tp_gpt import TPGPT, build_tp_gpt_from_reference
from nano_megatron.parallel.context import ParallelContext
from nano_megatron.parallel.mappings import scatter_to_sequence_parallel_region
from nano_megatron.parallel.vocab_parallel import (
    vocab_parallel_cross_entropy,
    vocab_range_from_global,
)
from nano_megatron.reference.config import ReferenceGPTConfig
from nano_megatron.reference.layers import layer_norm, rms_norm
from nano_megatron.reference.loss import shifted_cross_entropy as ref_shifted_cross_entropy
from nano_megatron.reference.model import ReferenceGPT


def _layer_slice(num_layers: int, pp_rank: int, pp_size: int) -> tuple[int, int]:
    if pp_size < 1:
        raise ValueError(f"pipeline_parallel_size must be >= 1, got {pp_size}")
    if not (0 <= pp_rank < pp_size):
        raise ValueError(
            f"pipeline_parallel_rank ({pp_rank}) out of range for pp_size={pp_size}"
        )
    if num_layers % pp_size != 0:
        raise ValueError(
            f"num_layers ({num_layers}) must be divisible by "
            f"pipeline_parallel_size ({pp_size})"
        )
    layers_per_stage = num_layers // pp_size
    start = pp_rank * layers_per_stage
    end = start + layers_per_stage
    return start, end


class PipelineStage(nn.Module):
    """One pipeline stage: optional emb, local transformer blocks, optional head.

    First stage owns embeddings; last stage owns final LN + lm_head.
    When pp_size == 1 the stage holds the full model.
    """

    def __init__(
        self,
        config: ReferenceGPTConfig,
        *,
        pp_rank: int,
        pp_size: int,
        tok_emb: nn.Module | None,
        pos_emb: nn.Module | None,
        blocks: nn.ModuleList,
        ln_f_weight: Parameter | None,
        ln_f_bias: Parameter | None,
        lm_head: nn.Module | None,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_group: Any = None,
        tp_backend: Any = None,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self._pp_rank = pp_rank
        self._pp_size = pp_size
        self._tp_rank = tp_rank
        self._tp_size = tp_size
        self._tp_group = tp_group
        self._tp_backend = tp_backend
        self._sequence_parallel = sequence_parallel
        self.layernorm_eps = config.layernorm_eps

        self.vocab_start_index, self.vocab_end_index = vocab_range_from_global(
            tp_rank, tp_size, config.vocab_size
        )

        # Register only modules this stage owns (None → not a parameter).
        self.tok_emb = tok_emb
        self.pos_emb = pos_emb
        self.blocks = blocks
        if ln_f_weight is not None:
            self.ln_f_weight = ln_f_weight
        if ln_f_bias is not None:
            self.ln_f_bias = ln_f_bias
        self.lm_head = lm_head

        if self.is_first_stage and self.tok_emb is None:
            raise ValueError("first pipeline stage requires tok_emb")
        if self.is_last_stage and (
            not hasattr(self, "ln_f_weight") or self.lm_head is None
        ):
            raise ValueError("last pipeline stage requires ln_f_* and lm_head")

        self.to(dtype=torch.float32)

    @property
    def is_first_stage(self) -> bool:
        return self._pp_rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self._pp_rank == self._pp_size - 1

    def shifted_cross_entropy(
        self,
        logits: Tensor,
        labels: Tensor,
        ignore_index: int = -100,
    ) -> Tensor:
        """Shifted CE; vocab-parallel when tp>1. Only meaningful on last stage."""
        if not self.is_last_stage:
            raise RuntimeError(
                "shifted_cross_entropy is only valid on the last pipeline stage "
                f"(pp_rank={self._pp_rank}, pp_size={self._pp_size})"
            )
        if self._tp_size == 1:
            return ref_shifted_cross_entropy(
                logits, labels, ignore_index=ignore_index
            )
        return vocab_parallel_cross_entropy(
            logits,
            labels,
            vocab_start_index=self.vocab_start_index,
            vocab_end_index=self.vocab_end_index,
            group=self._tp_group,
            backend=self._tp_backend,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        hidden_or_ids: Tensor,
        *,
        positions: Tensor | None = None,
    ) -> Tensor:
        """
        First stage: ``hidden_or_ids`` is input_ids ``[B, S]`` (long).
        Other stages: ``hidden_or_ids`` is activation ``[B, S, H]`` (float).
        Last stage returns logits; non-last returns activation ``[B, S, H]``.
        """
        if self.is_first_stage:
            input_ids = hidden_or_ids
            if input_ids.dim() != 2:
                raise ValueError(
                    f"first stage expects input_ids [B, S], got shape {tuple(input_ids.shape)}"
                )
            batch, seq_len = input_ids.shape
            if seq_len > self.config.max_seq_len:
                raise ValueError(
                    f"seq_len ({seq_len}) exceeds max_seq_len ({self.config.max_seq_len})"
                )
            if positions is None:
                positions = torch.arange(seq_len, device=input_ids.device)
                positions = positions.unsqueeze(0).expand(batch, -1)

            if self._sequence_parallel and seq_len % self._tp_size != 0:
                raise ValueError(
                    f"seq_len ({seq_len}) must be divisible by tensor_parallel_size "
                    f"({self._tp_size}) when sequence_parallel is enabled"
                )

            assert self.tok_emb is not None
            x = self.tok_emb(input_ids)
            if self.pos_emb is not None:
                x = x + self.pos_emb(positions)

            if self._sequence_parallel:
                x = scatter_to_sequence_parallel_region(
                    x,
                    self._tp_group,
                    self._tp_backend,
                    self._tp_rank,
                    self._tp_size,
                )
        else:
            x = hidden_or_ids
            if x.dim() != 3:
                raise ValueError(
                    f"non-first stage expects hidden [B, S, H], got shape {tuple(x.shape)}"
                )
            batch, seq_len, _hidden = x.shape
            if positions is None:
                # RoPE / learned pos still need absolute positions on middle/last stages.
                positions = torch.arange(seq_len, device=x.device)
                positions = positions.unsqueeze(0).expand(batch, -1)

        for block in self.blocks:
            x = block(x, positions=positions)

        if not self.is_last_stage:
            return x

        if self.config.normalization == "rmsnorm":
            final_ln = rms_norm(x, self.ln_f_weight, self.layernorm_eps)
        else:
            final_ln = layer_norm(
                x, self.ln_f_weight, self.ln_f_bias, self.layernorm_eps
            )
        assert self.lm_head is not None
        return self.lm_head(final_ln)


def _clone_embedding(emb: nn.Embedding) -> nn.Embedding:
    out = nn.Embedding(emb.num_embeddings, emb.embedding_dim, padding_idx=emb.padding_idx)
    out.weight = Parameter(emb.weight.data.clone())
    return out


def _build_from_reference_tp1(
    ref: ReferenceGPT,
    ctx: ParallelContext,
    start: int,
    end: int,
) -> PipelineStage:
    config = ref.config
    pp_rank = ctx.pipeline_parallel_rank
    pp_size = ctx.pipeline_parallel_size
    is_first = pp_rank == 0
    is_last = pp_rank == pp_size - 1

    tok_emb: nn.Module | None = None
    pos_emb: nn.Module | None = None
    if is_first:
        tok_emb = _clone_embedding(ref.tok_emb)
        if ref.pos_emb is not None:
            pos_emb = _clone_embedding(ref.pos_emb)

    # Deep-copy local blocks so the stage owns independent parameters.
    blocks = nn.ModuleList([copy.deepcopy(ref.blocks[i]) for i in range(start, end)])

    ln_f_weight: Parameter | None = None
    ln_f_bias: Parameter | None = None
    lm_head: nn.Module | None = None
    if is_last:
        ln_f_weight = Parameter(ref.ln_f_weight.data.clone())
        ln_f_bias = Parameter(ref.ln_f_bias.data.clone())
        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        lm_head.weight = Parameter(ref.lm_head.weight.data.clone())
        if config.tie_word_embeddings:
            # Only reachable when pp_size == 1 (guarded by builder).
            assert tok_emb is not None
            lm_head.weight = tok_emb.weight  # type: ignore[assignment]

    return PipelineStage(
        config,
        pp_rank=pp_rank,
        pp_size=pp_size,
        tok_emb=tok_emb,
        pos_emb=pos_emb,
        blocks=blocks,
        ln_f_weight=ln_f_weight,
        ln_f_bias=ln_f_bias,
        lm_head=lm_head,
        tp_rank=ctx.tensor_parallel_rank,
        tp_size=ctx.tensor_parallel_size,
        tp_group=ctx.tensor_parallel_group,
        tp_backend=ctx.backend,
        sequence_parallel=ctx.sequence_parallel,
    )


def _build_from_tp_gpt(
    ref: ReferenceGPT,
    ctx: ParallelContext,
    start: int,
    end: int,
) -> PipelineStage:
    full: TPGPT = build_tp_gpt_from_reference(ref, ctx)
    config = ref.config
    pp_rank = ctx.pipeline_parallel_rank
    pp_size = ctx.pipeline_parallel_size
    is_first = pp_rank == 0
    is_last = pp_rank == pp_size - 1

    tok_emb: nn.Module | None = None
    pos_emb: nn.Module | None = None
    if is_first:
        tok_emb = full.tok_emb
        if full.pos_emb is not None:
            pos_emb = full.pos_emb

    # Only keep local blocks — do not register unused blocks as parameters.
    blocks = nn.ModuleList(
        [full.blocks[i] for i in range(start, end)]
    )

    ln_f_weight: Parameter | None = None
    ln_f_bias: Parameter | None = None
    lm_head: nn.Module | None = None
    if is_last:
        ln_f_weight = full.ln_f_weight
        ln_f_bias = full.ln_f_bias
        lm_head = full.lm_head
        if config.tie_word_embeddings:
            assert tok_emb is not None
            # pp_size must be 1 here (builder guard); re-tie local modules.
            lm_head.weight = tok_emb.weight  # type: ignore[assignment]

    # Drop reference to full so unused blocks are not kept alive via stage.
    del full

    return PipelineStage(
        config,
        pp_rank=pp_rank,
        pp_size=pp_size,
        tok_emb=tok_emb,
        pos_emb=pos_emb,
        blocks=blocks,
        ln_f_weight=ln_f_weight,
        ln_f_bias=ln_f_bias,
        lm_head=lm_head,
        tp_rank=ctx.tensor_parallel_rank,
        tp_size=ctx.tensor_parallel_size,
        tp_group=ctx.tensor_parallel_group,
        tp_backend=ctx.backend,
        sequence_parallel=ctx.sequence_parallel,
    )


def build_pipeline_stage_from_reference(
    ref: ReferenceGPT,
    ctx: ParallelContext,
) -> PipelineStage:
    """Build the local pipeline stage from a full reference GPT.

    * ``tp==1``: clone slices of ``ReferenceGPT`` modules into this stage.
    * ``tp>1``: build ``TPGPT`` from ref, then keep only this stage's modules
      (emb on first, head on last, ``blocks[start:end]`` everywhere).
    """
    config = ref.config
    pp_size = ctx.pipeline_parallel_size
    pp_rank = ctx.pipeline_parallel_rank

    if config.tie_word_embeddings and pp_size > 1:
        raise ValueError(
            "tie_word_embeddings is not supported when pipeline_parallel_size > 1"
        )

    start, end = _layer_slice(config.num_layers, pp_rank, pp_size)

    if ctx.tensor_parallel_size == 1:
        return _build_from_reference_tp1(ref, ctx, start, end)
    return _build_from_tp_gpt(ref, ctx, start, end)
