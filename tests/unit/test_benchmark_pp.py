from __future__ import annotations

import torch
import torch.nn as nn

from scripts.benchmark_pp import _optimizer_step


def test_optimizer_step_uses_grad_and_megatron_main_grad() -> None:
    module = nn.Module()
    module.regular = nn.Parameter(torch.tensor([2.0]))
    module.megatron = nn.Parameter(torch.tensor([3.0]))
    module.regular.grad = torch.tensor([4.0])
    module.megatron.main_grad = torch.tensor([5.0])

    _optimizer_step(module, "sgd", 0.1)

    torch.testing.assert_close(module.regular, torch.tensor([1.6]))
    torch.testing.assert_close(module.megatron, torch.tensor([2.5]))
