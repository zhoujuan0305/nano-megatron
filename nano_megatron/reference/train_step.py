"""Single-step and multi-step reference training with optional capture."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from nano_megatron.reference.capture import (
    CaptureLevel,
    snapshot_grads,
    snapshot_optimizer,
    snapshot_params,
    snapshot_tree,
)
from nano_megatron.reference.loss import shifted_cross_entropy
from nano_megatron.reference.optimizer import AdamW


@dataclass
class StepResult:
    step: int
    loss: Tensor
    logits: Tensor
    params: dict[str, Tensor]
    grads: dict[str, Tensor] | None = None
    activations: dict[str, Any] | None = None
    optimizer_state: dict[str, dict[str, Any]] | None = None


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reference_train_step(
    model: Module,
    optimizer: AdamW,
    input_ids: Tensor,
    labels: Tensor | None = None,
    capture_level: CaptureLevel = "full",
    step: int = 0,
) -> StepResult:
    if capture_level not in ("minimal", "grads", "full"):
        raise ValueError(f"unknown capture_level: {capture_level!r}")

    target = input_ids if labels is None else labels
    optimizer.zero_grad(set_to_none=True)

    activations_raw: dict[str, Any] | None = None
    if capture_level == "full":
        logits, activations_raw = model.forward_with_activations(input_ids)
    else:
        logits = model.forward(input_ids)

    loss = shifted_cross_entropy(logits, target)
    loss.backward()

    grads: dict[str, Tensor] | None = None
    if capture_level in ("grads", "full"):
        grads = snapshot_grads(model)

    optimizer.step()

    params = snapshot_params(model)
    activations: dict[str, Any] | None = None
    optimizer_state: dict[str, dict[str, Any]] | None = None
    if capture_level == "full":
        activations = snapshot_tree(activations_raw)
        optimizer_state = snapshot_optimizer(optimizer)

    return StepResult(
        step=step,
        loss=loss.detach().cpu().clone(),
        logits=logits.detach().cpu().clone(),
        params=params,
        grads=grads,
        activations=activations,
        optimizer_state=optimizer_state,
    )


def reference_train_loop(
    model: Module,
    optimizer: AdamW,
    batches: Sequence[Tensor],
    steps: int,
    capture_every: int = 1,
    capture_level: CaptureLevel = "full",
) -> list[StepResult]:
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    if capture_every < 1:
        raise ValueError(f"capture_every must be >= 1, got {capture_every}")
    if len(batches) == 0 and steps > 0:
        raise ValueError("batches is empty but steps > 0")

    results: list[StepResult] = []
    for i in range(steps):
        batch = batches[i % len(batches)]
        should_capture = i % capture_every == 0
        level: CaptureLevel = capture_level if should_capture else "minimal"
        result = reference_train_step(
            model,
            optimizer,
            batch,
            labels=None,
            capture_level=level,
            step=i,
        )
        if should_capture:
            results.append(result)
    return results
