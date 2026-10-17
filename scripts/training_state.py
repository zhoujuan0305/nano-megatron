"""Checkpoint helpers for the benchmark FP32-master AdamW optimizer."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import nn


_SCHEMA_VERSION = 1


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def _signature(optimizer: Any, module: nn.Module) -> list[dict[str, Any]]:
    trainable = [(name, param) for name, param in module.named_parameters() if param.requires_grad]
    if len(trainable) != len(optimizer._model_params):
        raise ValueError("module trainable parameter count differs from optimizer")
    signature = []
    for (name, param), model_param in zip(trainable, optimizer._model_params):
        if param is not model_param:
            raise ValueError("module parameter order or identity differs from optimizer")
        signature.append({"name": name, "shape": tuple(param.shape), "dtype": str(param.dtype)})
    return signature


def optimizer_state_dict(optimizer: Any, module: nn.Module) -> dict[str, Any]:
    """Return a detached CPU snapshot; driver owns model/config/data/RNG checkpointing."""
    signature = _signature(optimizer, module)
    if len(optimizer._masters) != len(signature):
        raise ValueError("optimizer master count differs from model parameters")
    masters = []
    for master, param in zip(optimizer._masters, optimizer._model_params):
        if master.dtype != torch.float32 or tuple(master.shape) != tuple(param.shape):
            raise ValueError("optimizer masters must match model shapes and use FP32")
        if not torch.isfinite(master).all():
            raise ValueError("optimizer master contains non-finite values")
        masters.append(_cpu_clone(master))
    groups = []
    for group in optimizer.optimizer.param_groups:
        groups.append({key: _cpu_clone(value) for key, value in group.items() if key != "params"})
    return {
        "schema_version": _SCHEMA_VERSION,
        "parameters": signature,
        "masters": masters,
        "torch_optimizer": _cpu_clone(optimizer.optimizer.state_dict()),
        "step_count": optimizer.step_count,
        "groups": groups,
    }


@torch.no_grad()
def load_optimizer_state_dict(optimizer: Any, module: nn.Module, state: dict[str, Any]) -> None:
    """Restore strictly; saved parameter-group hyperparameters, including LR, take precedence.

    A driver may resume its scheduler from the saved step and then update LR as usual.
    """
    if not isinstance(state, dict) or state.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported or incomplete optimizer checkpoint")
    signature = _signature(optimizer, module)
    if state.get("parameters") != signature:
        raise ValueError("optimizer checkpoint parameter signature mismatch")
    masters = state.get("masters")
    if not isinstance(masters, list) or len(masters) != len(optimizer._masters):
        raise ValueError("optimizer checkpoint master count mismatch")
    for saved, live in zip(masters, optimizer._masters):
        if not isinstance(saved, torch.Tensor) or saved.dtype != torch.float32 or tuple(saved.shape) != tuple(live.shape):
            raise ValueError("optimizer checkpoint master shape/dtype mismatch")
        if not torch.isfinite(saved).all():
            raise ValueError("optimizer checkpoint master contains non-finite values")

    torch_state = state.get("torch_optimizer")
    groups = state.get("groups")
    count = state.get("step_count")
    if not isinstance(torch_state, dict) or not isinstance(groups, list):
        raise ValueError("checkpoint lacks complete torch optimizer state")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("invalid optimizer step counter")
    saved_groups = torch_state.get("param_groups")
    if not isinstance(saved_groups, list) or len(saved_groups) != len(optimizer.optimizer.param_groups) or len(groups) != len(saved_groups):
        raise ValueError("optimizer checkpoint group count mismatch")
    for current, saved, metadata in zip(optimizer.optimizer.param_groups, saved_groups, groups):
        if len(current["params"]) != len(saved.get("params", [])):
            raise ValueError("optimizer checkpoint parameter group topology mismatch")
        for key, value in metadata.items():
            if key != "params" and saved.get(key) != value:
                raise ValueError("optimizer checkpoint hyperparameter metadata mismatch")
        for key in ("lr", "eps", "weight_decay"):
            value = saved.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid optimizer hyperparameter: {key}")
        betas = saved.get("betas")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2 or not all(math.isfinite(float(v)) and 0 <= float(v) < 1 for v in betas):
            raise ValueError("invalid optimizer betas")

    param_ids = [pid for group in saved_groups for pid in group["params"]]
    if len(param_ids) != len(optimizer._masters) or len(set(param_ids)) != len(param_ids):
        raise ValueError("optimizer checkpoint parameter count mismatch")
    saved_state = torch_state.get("state")
    if not isinstance(saved_state, dict) or set(saved_state).difference(param_ids):
        raise ValueError("invalid torch optimizer state mapping")
    if count > 0 and set(saved_state) != set(param_ids):
        raise ValueError("trained checkpoint must initialize every parameter state")
    for entry in saved_state.values():
        if not isinstance(entry, dict) or not all(key in entry for key in ("step", "exp_avg", "exp_avg_sq")):
            raise ValueError("incomplete AdamW moment state")
        for key in ("step", "exp_avg", "exp_avg_sq"):
            value = entry[key]
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise ValueError(f"invalid/non-finite AdamW state {key}")
        if entry["step"].numel() != 1 or entry["step"].item() != count:
            raise ValueError("AdamW per-parameter step disagrees with checkpoint counter")
        for key in ("exp_avg", "exp_avg_sq"):
            if entry[key].dtype != torch.float32:
                raise ValueError(f"invalid AdamW state {key} dtype")
        if (entry["exp_avg_sq"] < 0).any():
            raise ValueError("AdamW second moment cannot be negative")
    # Validate each moment against the parameter associated with its serialized ID.
    for group, live_group in zip(saved_groups, optimizer.optimizer.param_groups):
        for pid, master in zip(group["params"], live_group["params"]):
            entry = saved_state.get(pid)
            if entry is not None and any(tuple(entry[key].shape) != tuple(master.shape) for key in ("exp_avg", "exp_avg_sq")):
                raise ValueError("AdamW moment shape mismatch")

    # Build a detached copy before mutating live state; torch loads moments onto parameter devices.
    prepared = copy.deepcopy(torch_state)
    for live, saved in zip(optimizer._masters, masters):
        live.copy_(saved.to(device=live.device))
    optimizer.optimizer.load_state_dict(prepared)
    optimizer.step_count = count
    for param, master in zip(optimizer._model_params, optimizer._masters):
        param.copy_(master)
        param.grad = None
    for grad_buffer in optimizer._grad_buffers:
        grad_buffer.zero_()
    for model_param, master in zip(optimizer._model_params, optimizer._masters):
        master.grad = None
        main_grad = getattr(model_param, "main_grad", None)
        if main_grad is not None:
            main_grad.zero_()  # Preserve any DDP-owned buffer alias.
