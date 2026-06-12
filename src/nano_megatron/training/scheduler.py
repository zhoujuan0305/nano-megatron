from __future__ import annotations

import math
from typing import Any


class CosineWarmupScheduler:
    def __init__(
        self, optimizer: Any, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._step_count = 0

    def get_lr(self) -> list[float]:
        if self._step_count < self.warmup_steps:
            scale = self._step_count / max(1, self.warmup_steps)
        else:
            progress = (self._step_count - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            scale = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * scale
        return [base_lr * scale for base_lr in self.base_lrs]

    def step(self) -> None:
        lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, lrs, strict=True):
            group["lr"] = lr
        self._step_count += 1

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "step_count": self._step_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._step_count = state_dict["step_count"]
