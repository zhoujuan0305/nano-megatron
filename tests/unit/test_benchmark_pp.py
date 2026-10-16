from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from scripts.benchmark_pp import _make_input_ids, _mean_next_token_loss, _optimizer_step


def test_optimizer_step_uses_grad_and_megatron_main_grad() -> None:
    module = nn.Module()
    module.regular = nn.Parameter(torch.tensor([2.0]))
    module.megatron = nn.Parameter(torch.tensor([3.0]))
    module.regular.grad = torch.tensor([4.0])
    module.megatron.main_grad = torch.tensor([5.0])

    _optimizer_step(module, "sgd", 0.1)

    torch.testing.assert_close(module.regular, torch.tensor([1.6]))
    torch.testing.assert_close(module.megatron, torch.tensor([2.5]))


def test_benchmark_input_ids_use_reproducible_independent_generator() -> None:
    state = torch.random.get_rng_state()
    first = _make_input_ids(vocab_size=32, shape=(2, 5), device=torch.device("cpu"))
    second = _make_input_ids(vocab_size=32, shape=(2, 5), device=torch.device("cpu"))

    assert torch.equal(first, second)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_megatron_next_token_loss_omits_wrapped_last_target() -> None:
    per_token = torch.tensor([[1.0, 3.0, 1000.0]], requires_grad=True)
    loss = _mean_next_token_loss(per_token)
    assert loss.item() == 2.0
    loss.backward()
    torch.testing.assert_close(per_token.grad, torch.tensor([[0.5, 0.5, 0.0]]))


def test_nano_channel_cli_keeps_four_default_and_accepts_eight(monkeypatch):
    from scripts.benchmark_pp import parse_args
    monkeypatch.setattr("sys.argv", ["benchmark_pp.py"])
    assert parse_args().nano_nccl_channels == 4
    monkeypatch.setattr("sys.argv", ["benchmark_pp.py", "--nano-nccl-channels", "8"])
    assert parse_args().nano_nccl_channels == 8


def test_nano_channel_cli_rejects_nonpositive_value(monkeypatch):
    from scripts.benchmark_pp import parse_args, _validate_args
    monkeypatch.setattr("sys.argv", ["benchmark_pp.py", "--nano-nccl-channels", "0"])
    with pytest.raises(ValueError, match="nano-nccl-channels must be positive"):
        _validate_args(parse_args())
