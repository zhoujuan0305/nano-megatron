"""Correctness-first GPT reference used as a numerical oracle."""

from nano_megatron.reference.capture import (
    CaptureLevel,
    snapshot_grads,
    snapshot_optimizer,
    snapshot_params,
)
from nano_megatron.reference.config import ReferenceGPTConfig
from nano_megatron.reference.loss import shifted_cross_entropy
from nano_megatron.reference.model import ReferenceGPT
from nano_megatron.reference.optimizer import AdamW
from nano_megatron.reference.train_step import (
    StepResult,
    reference_train_loop,
    reference_train_step,
    seed_all,
)

__all__ = [
    "AdamW",
    "CaptureLevel",
    "ReferenceGPT",
    "ReferenceGPTConfig",
    "StepResult",
    "reference_train_loop",
    "reference_train_step",
    "seed_all",
    "shifted_cross_entropy",
    "snapshot_grads",
    "snapshot_optimizer",
    "snapshot_params",
]
