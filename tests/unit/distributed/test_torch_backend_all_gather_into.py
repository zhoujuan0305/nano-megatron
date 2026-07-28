"""Unit tests for CommBackend.all_gather_into_tensor."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from nano_megatron.distributed.backend import CommBackend
from nano_megatron.distributed.torch_backend import TorchDistBackend


def test_comm_backend_protocol_declares_all_gather_into_tensor() -> None:
    assert hasattr(CommBackend, "all_gather_into_tensor")


def test_torch_dist_backend_implements_all_gather_into_tensor() -> None:
    backend: CommBackend = TorchDistBackend()
    assert hasattr(backend, "all_gather_into_tensor")
    assert callable(backend.all_gather_into_tensor)


def test_all_gather_into_tensor_calls_dist(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch.distributed as dist

    captured: dict[str, Any] = {}

    def fake_all_gather_into_tensor(
        output: Tensor,
        input: Tensor,
        group: Any = None,
        async_op: bool = False,
    ) -> None:
        captured["output"] = output
        captured["input"] = input
        captured["group"] = group
        captured["async_op"] = async_op
        # world_size=1 semantics: copy input into the single output slot
        output.copy_(input.reshape_as(output) if input.numel() == output.numel() else input)

    monkeypatch.setattr(dist, "all_gather_into_tensor", fake_all_gather_into_tensor)

    backend = TorchDistBackend()
    inp = torch.tensor([1.0, 2.0, 3.0])
    out = torch.zeros(3)
    group_sentinel = object()

    result = backend.all_gather_into_tensor(out, inp, group=group_sentinel)

    assert result is out
    assert captured["output"] is out
    assert captured["input"] is inp
    assert captured["group"] is group_sentinel
    assert torch.equal(out, inp)


def test_all_gather_into_tensor_world_size1_gloo(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch.distributed as dist

    if not dist.is_available():
        pytest.skip("torch.distributed not available")

    if dist.is_initialized():
        dist.destroy_process_group()

    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29571")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        backend = TorchDistBackend()
        inp = torch.arange(4, dtype=torch.float32)
        out = torch.empty(4, dtype=torch.float32)
        result = backend.all_gather_into_tensor(out, inp, group=None)
        assert result is out
        assert torch.equal(out, inp)
    finally:
        dist.destroy_process_group()


def test_list_all_gather_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing list all_gather API must remain available and wired."""
    import torch.distributed as dist

    captured: dict[str, Any] = {}

    def fake_all_gather(
        tensor_list: list[Tensor],
        tensor: Tensor,
        group: Any = None,
        async_op: bool = False,
    ) -> None:
        captured["tensor_list"] = tensor_list
        captured["tensor"] = tensor
        captured["group"] = group
        tensor_list[0].copy_(tensor)

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)

    backend = TorchDistBackend()
    inp = torch.ones(2)
    gathered = [torch.zeros(2)]
    result = backend.all_gather(gathered, inp, group=None)
    assert result is gathered
    assert captured["tensor"] is inp
    assert torch.equal(gathered[0], inp)
