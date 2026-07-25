"""Detached CPU snapshots of params, grads, optimizer state, and activations."""

from __future__ import annotations

from typing import Any, Literal

from torch import Tensor
from torch.nn import Module

from nano_megatron.reference.optimizer import AdamW

CaptureLevel = Literal["minimal", "grads", "full"]


def _cpu_clone(t: Tensor) -> Tensor:
    return t.detach().to(device="cpu", dtype=t.dtype).clone()


def snapshot_tree(obj: Any) -> Any:
    """Recursively detach/clone tensors in nested dict/list/tuple structures to CPU."""
    if isinstance(obj, Tensor):
        return _cpu_clone(obj)
    if isinstance(obj, dict):
        return {k: snapshot_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [snapshot_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(snapshot_tree(v) for v in obj)
    return obj


def snapshot_params(model: Module) -> dict[str, Tensor]:
    return {name: _cpu_clone(param) for name, param in model.named_parameters()}


def snapshot_grads(model: Module) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            raise RuntimeError(f"missing grad for parameter {name!r}")
        out[name] = _cpu_clone(param.grad)
    return out


def snapshot_optimizer(opt: AdamW) -> dict[str, dict[str, Any]]:
    raw = opt.state_dict()
    out: dict[str, dict[str, Any]] = {}
    for name, entry in raw.items():
        out[name] = {
            "exp_avg": _cpu_clone(entry["exp_avg"]),
            "exp_avg_sq": _cpu_clone(entry["exp_avg_sq"]),
            "step": int(entry["step"]),
        }
    return out
