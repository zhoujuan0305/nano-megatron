from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from scripts.benchmark_optimizer import make_benchmark_adamw


def test_benchmark_adamw_matches_torch_adamw_for_multiple_fp32_steps() -> None:
    module = nn.Module()
    module.weight = nn.Parameter(torch.tensor([[1.0, -2.0], [3.0, 4.0]]))
    module.bias = nn.Parameter(torch.tensor([0.5, -0.25]))
    reference = nn.Module()
    reference.weight = nn.Parameter(module.weight.detach().clone())
    reference.bias = nn.Parameter(module.bias.detach().clone())
    benchmark = make_benchmark_adamw(module, learning_rate=0.01)
    torch_optimizer = torch.optim.AdamW(
        [
            {"params": [reference.weight], "weight_decay": 0.1},
            {"params": [reference.bias], "weight_decay": 0.0},
        ],
        lr=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=False,
    )

    for i in range(4):
        grad_weight = torch.full_like(module.weight, 0.2 + i * 0.1)
        grad_bias = torch.full_like(module.bias, -0.3 + i * 0.05)
        module.weight.grad = grad_weight.clone()
        module.bias.grad = grad_bias.clone()
        reference.weight.grad = grad_weight.clone()
        reference.bias.grad = grad_bias.clone()
        benchmark.step()
        torch_optimizer.step()
        torch.testing.assert_close(module.weight, reference.weight)
        torch.testing.assert_close(module.bias, reference.bias)

    assert [group["weight_decay"] for group in benchmark.optimizer.param_groups] == [
        0.1,
        0.0,
    ]
    for state in benchmark.optimizer.state.values():
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32


def test_benchmark_adamw_uses_fp32_master_for_bf16_model() -> None:
    module = nn.Linear(2, 2, bias=True, dtype=torch.bfloat16)
    before = [param.detach().clone() for param in module.parameters()]
    benchmark = make_benchmark_adamw(module, learning_rate=0.01)
    for param in module.parameters():
        param.grad = torch.full_like(param, 0.25)

    benchmark.step()
    benchmark.verify_updated(before)

    assert all(param.dtype == torch.bfloat16 for param in module.parameters())
    assert all(master.dtype == torch.float32 for master in benchmark._masters)


def test_benchmark_adamw_reads_megatron_main_grad() -> None:
    module = nn.Module()
    module.weight = nn.Parameter(torch.tensor([[2.0, 3.0]]))
    module.weight.main_grad = torch.full_like(module.weight, 0.5)
    module.weight.grad = torch.full_like(module.weight, -3.0)
    before = module.weight.detach().clone()
    expected = nn.Parameter(before.clone())
    reference = torch.optim.AdamW(
        [expected], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    expected.grad = module.weight.main_grad.clone()
    reference.step()
    benchmark = make_benchmark_adamw(module, learning_rate=0.01)

    benchmark.step()
    benchmark.verify_updated([before])
    torch.testing.assert_close(module.weight, expected)


def test_benchmark_adamw_skips_parameters_without_gradients() -> None:
    module = nn.Module()
    module.weight = nn.Parameter(torch.tensor([[2.0, 3.0]]))
    module.bias = nn.Parameter(torch.tensor([4.0]))
    module.weight.grad = torch.ones_like(module.weight)
    before_bias = module.bias.detach().clone()
    benchmark = make_benchmark_adamw(module, learning_rate=0.01)

    benchmark.step()

    torch.testing.assert_close(module.bias, before_bias)
    assert benchmark._masters[1] not in benchmark.optimizer.state


def test_master_probes_detect_measured_updates_without_gpu_snapshot() -> None:
    module = nn.Linear(2, 2, bias=False)
    benchmark = make_benchmark_adamw(module, learning_rate=0.01)
    module.weight.grad = torch.ones_like(module.weight)
    benchmark.step()
    probes = benchmark.snapshot_master_samples()
    assert all(probe.device.type == "cpu" for probe in probes)
    with pytest.raises(RuntimeError, match="no sampled master update"):
        benchmark.verify_updated(master_samples=probes)
    benchmark.step()
    stats = benchmark.verify_updated(master_samples=probes)
    assert stats["master_samples_changed"] is True
    assert stats["optimizer_steps"] == 2


def test_benchmark_adamw_ignores_frozen_parameters() -> None:
    module = nn.Linear(2, 2)
    module.bias.requires_grad_(False)
    benchmark = make_benchmark_adamw(module, learning_rate=0.01)
    assert len(benchmark._model_params) == 1
