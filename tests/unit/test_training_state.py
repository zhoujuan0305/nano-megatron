from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from scripts.benchmark_optimizer import make_benchmark_adamw
from scripts.training_state import load_optimizer_state_dict, optimizer_state_dict


def _update(module: nn.Module, optimizer: object, index: int) -> None:
    for position, param in enumerate(module.parameters()):
        param.grad = torch.full_like(param, 0.1 + index * 0.03 + position * 0.02)
    optimizer.step()


def _module() -> nn.Module:
    return nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))


def test_optimizer_checkpoint_resumes_identically_and_is_cpu_snapshot() -> None:
    source = _module()
    first = make_benchmark_adamw(source, learning_rate=0.01)
    for index in range(3):
        _update(source, first, index)
    checkpoint = optimizer_state_dict(first, source)
    assert all(t.device.type == "cpu" for t in checkpoint["masters"])
    for master, live in zip(checkpoint["masters"], first._masters):
        assert master.data_ptr() != live.data_ptr()
    for entry in checkpoint["torch_optimizer"]["state"].values():
        assert all(not isinstance(v, torch.Tensor) or v.device.type == "cpu" for v in entry.values())

    resumed_module = copy.deepcopy(source)
    resumed = make_benchmark_adamw(resumed_module, learning_rate=0.2)
    load_optimizer_state_dict(resumed, resumed_module, checkpoint)
    assert all(param.grad is None for param in resumed_module.parameters())
    for module, optimizer in ((source, first), (resumed_module, resumed)):
        _update(module, optimizer, 3)
    for left, right in zip(source.parameters(), resumed_module.parameters()):
        assert torch.equal(left, right)
    for left, right in zip(first._masters, resumed._masters):
        assert torch.equal(left, right)
    assert first.step_count == resumed.step_count == 4
    assert first.optimizer.param_groups[0]["lr"] == resumed.optimizer.param_groups[0]["lr"] == 0.01
    for left, right in zip(first.optimizer.state.values(), resumed.optimizer.state.values()):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(left[key], right[key])
            if isinstance(right[key], torch.Tensor):
                assert right[key].device == resumed._masters[0].device


def test_untrained_optimizer_checkpoint_roundtrip() -> None:
    module = _module()
    optimizer = make_benchmark_adamw(module, learning_rate=0.01)
    checkpoint = optimizer_state_dict(optimizer, module)
    other_module = copy.deepcopy(module)
    other = make_benchmark_adamw(other_module, learning_rate=0.5)
    load_optimizer_state_dict(other, other_module, checkpoint)
    assert other.step_count == 0
    assert not other.optimizer.state
    assert all(torch.equal(a, b) for a, b in zip(optimizer._masters, other._masters))
    assert other.optimizer.param_groups[0]["lr"] == 0.01


@pytest.mark.parametrize("corruption", ["signature", "master_shape", "master_nan", "counter", "counter_mismatch", "missing_state", "moment_nan", "moment_shape"])
def test_optimizer_checkpoint_rejects_corruption(corruption: str) -> None:
    module = _module()
    optimizer = make_benchmark_adamw(module, learning_rate=0.01)
    _update(module, optimizer, 0)
    checkpoint = copy.deepcopy(optimizer_state_dict(optimizer, module))
    if corruption == "signature":
        checkpoint["parameters"][0]["name"] = "wrong"
    elif corruption == "master_shape":
        checkpoint["masters"][0] = torch.zeros(1)
    elif corruption == "master_nan":
        checkpoint["masters"][0].flatten()[0] = float("nan")
    elif corruption == "counter":
        checkpoint["step_count"] = -1
    elif corruption == "counter_mismatch":
        checkpoint["step_count"] += 1
    elif corruption == "missing_state":
        checkpoint["torch_optimizer"]["state"].pop(next(iter(checkpoint["torch_optimizer"]["state"])))
    else:
        item = next(iter(checkpoint["torch_optimizer"]["state"].values()))
        key = "exp_avg" if corruption == "moment_nan" else "exp_avg_sq"
        if corruption == "moment_nan":
            item[key].flatten()[0] = float("nan")
        else:
            item[key] = torch.zeros(1)
    target_module = copy.deepcopy(module)
    target = make_benchmark_adamw(target_module, learning_rate=0.01)
    with pytest.raises(ValueError):
        load_optimizer_state_dict(target, target_module, checkpoint)
