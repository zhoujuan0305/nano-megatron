"""Optimizer helpers used by the pipeline benchmark."""

from __future__ import annotations

import torch
from torch import nn


ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8
ADAMW_WEIGHT_DECAY = 0.1


class BenchmarkAdamW:
    """AdamW with FP32 master weights for benchmark model parameters."""

    def __init__(
        self,
        module: nn.Module,
        *,
        learning_rate: float,
        fused: bool,
    ) -> None:
        decay: list[torch.Tensor] = []
        no_decay: list[torch.Tensor] = []
        self._model_params: list[torch.Tensor] = []
        self._masters: list[torch.Tensor] = []
        self.fused = fused
        self.step_count = 0
        for _, param in module.named_parameters():
            if not param.requires_grad:
                continue
            master = param.detach().float().clone()
            self._model_params.append(param)
            self._masters.append(master)
            (decay if param.ndim >= 2 else no_decay).append(master)

        groups = [
            {"params": decay, "weight_decay": ADAMW_WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(
            groups,
            lr=learning_rate,
            betas=ADAMW_BETAS,
            eps=ADAMW_EPS,
            fused=fused,
        )
        self._grad_buffers = [torch.zeros_like(master) for master in self._masters]

    @torch.no_grad()
    def step(self) -> None:
        for param, master, grad_buffer in zip(
            self._model_params, self._masters, self._grad_buffers
        ):
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is None:
                master.grad = None
                continue
            grad_buffer.copy_(grad)
            master.grad = grad_buffer

        self.optimizer.step()
        for param, master in zip(self._model_params, self._masters):
            param.copy_(master)
        self.step_count += 1

    def verify_updated(self, before: list[torch.Tensor] | None = None) -> dict:
        """Validate outside timing; optionally prove model weights changed."""
        if before is not None:
            if len(before) != len(self._model_params):
                raise ValueError("parameter snapshot does not match optimizer parameters")
            if not any(
                not torch.equal(old, param.detach())
                for old, param in zip(before, self._model_params)
            ):
                raise RuntimeError("AdamW validation found no parameter change")
        if not self.optimizer.state:
            raise RuntimeError("AdamW validation found no initialized states")
        for param, master in zip(self._model_params, self._masters):
            if master.dtype != torch.float32 or not torch.isfinite(master).all():
                raise RuntimeError("AdamW master parameters must be finite FP32")
            if not torch.isfinite(param).all():
                raise RuntimeError("AdamW model parameters must be finite")
        for state in self.optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                value = state[key]
                if value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise RuntimeError(f"AdamW {key} must be finite FP32")
        return {
            "optimizer_steps": self.step_count,
            "model_parameter_tensors": len(self._model_params),
            "initialized_state_tensors": len(self.optimizer.state),
            "master_dtype": "float32",
            "moment_dtype": "float32",
            "fused": self.fused,
            "finite": True,
            "model_changed": True if before is not None else None,
        }


def make_benchmark_adamw(
    module: nn.Module, *, learning_rate: float
) -> BenchmarkAdamW:
    """Create the shared CUDA fused / CPU non-fused benchmark implementation."""
    device_types = {param.device.type for param in module.parameters()}
    if len(device_types) > 1:
        raise ValueError("AdamW benchmark module parameters must share a device")
    fused = device_types == {"cuda"}
    return BenchmarkAdamW(module, learning_rate=learning_rate, fused=fused)
