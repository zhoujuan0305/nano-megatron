"""Explicit AdamW with named parameter state (correctness-first reference)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.nn import Parameter


class AdamW:
    """Decoupled AdamW matching ``torch.optim.AdamW`` (bias correction + WD)."""

    def __init__(
        self,
        named_params: Iterable[tuple[str, Parameter]],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.state: dict[str, dict[str, Any]] = {}
        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        seen: set[str] = set()
        params: list[tuple[str, Parameter]] = []
        for name, param in named_params:
            if name in seen:
                raise ValueError(f"duplicate parameter name: {name!r}")
            if not isinstance(param, Parameter):
                raise TypeError(f"expected Parameter for {name!r}, got {type(param)}")
            seen.add(name)
            params.append((name, param))
        self._params = params

    def step(self) -> None:
        beta1, beta2 = self.betas
        lr = self.lr
        eps = self.eps
        weight_decay = self.weight_decay

        for name, param in self._params:
            grad = param.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError(f"AdamW does not support sparse gradients ({name!r})")

            state = self.state.get(name)
            if state is None:
                state = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(param, dtype=torch.float32),
                    "exp_avg_sq": torch.zeros_like(param, dtype=torch.float32),
                }
                self.state[name] = state

            exp_avg: Tensor = state["exp_avg"]
            exp_avg_sq: Tensor = state["exp_avg_sq"]
            state["step"] = int(state["step"]) + 1
            step_t = state["step"]

            p = param.data
            g = grad.detach()
            if p.dtype != torch.float32:
                p_fp32 = p.float()
            else:
                p_fp32 = p

            g_fp32 = g.float() if g.dtype != torch.float32 else g

            # Decoupled weight decay (torch.optim.AdamW)
            if weight_decay != 0.0:
                p_fp32 = p_fp32.mul(1.0 - lr * weight_decay)

            exp_avg.mul_(beta1).add_(g_fp32, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g_fp32, g_fp32, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step_t
            bias_correction2 = 1.0 - beta2**step_t
            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
            p_fp32 = p_fp32.addcdiv(exp_avg, denom, value=-step_size)

            if p.dtype != torch.float32:
                p.copy_(p_fp32.to(dtype=p.dtype))
            else:
                p.copy_(p_fp32)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for _, param in self._params:
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.detach_()
                    param.grad.zero_()

    def state_dict(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name, state in self.state.items():
            out[name] = {
                "exp_avg": state["exp_avg"].clone(),
                "exp_avg_sq": state["exp_avg_sq"].clone(),
                "step": int(state["step"]),
            }
        return out

    def load_state_dict(self, state: dict[str, dict[str, Any]]) -> None:
        param_by_name = {name: param for name, param in self._params}
        new_state: dict[str, dict[str, Any]] = {}
        for name, entry in state.items():
            device = (
                param_by_name[name].device
                if name in param_by_name
                else entry["exp_avg"].device
            )
            new_state[name] = {
                "exp_avg": entry["exp_avg"].detach().to(device=device, dtype=torch.float32).clone(),
                "exp_avg_sq": entry["exp_avg_sq"]
                .detach()
                .to(device=device, dtype=torch.float32)
                .clone(),
                "step": int(entry["step"]),
            }
        self.state = new_state
