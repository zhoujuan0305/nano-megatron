import pytest
import torch
from torch import Tensor
from nano_megatron.distributed.torch_backend import reduce_op_from_string, TorchDistBackend
from nano_megatron.distributed.backend import CommBackend


def test_reduce_op_from_string():
    assert reduce_op_from_string("sum") == torch.distributed.ReduceOp.SUM
    assert reduce_op_from_string("max") == torch.distributed.ReduceOp.MAX
    with pytest.raises(ValueError, match="unsupported"):
        reduce_op_from_string("mean")


def test_torch_backend_is_comm_backend():
    backend: CommBackend = TorchDistBackend()
    assert hasattr(backend, "all_reduce")


def test_all_reduce_sync_returns_tensor(monkeypatch):
    """Default sync path returns the tensor (unchanged behaviour)."""
    import torch.distributed as dist

    captured = {}
    orig = dist.all_reduce

    def fake_all_reduce(tensor, op=None, group=None, async_op=False):
        captured["async_op"] = async_op
        return orig(tensor, op=op, group=group, async_op=async_op)

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    backend = TorchDistBackend()
    t = torch.zeros(4, dtype=torch.float32)

    # Skip the test entirely if dist is not initialised in this run.
    if not dist.is_available() or not dist.is_initialized():
        pytest.skip("distributed not initialised")

    out = backend.all_reduce(t, op="sum")
    assert captured["async_op"] is False
    assert isinstance(out, Tensor)
    monkeypatch.undo()


def test_all_reduce_async_returns_work(monkeypatch):
    """async_op=True causes the backend to return a Work handle, not a Tensor."""
    import torch.distributed as dist

    captured = {}
    orig = dist.all_reduce

    def fake_all_reduce(tensor, op=None, group=None, async_op=False):
        captured["async_op"] = async_op
        return orig(tensor, op=op, group=group, async_op=async_op)

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    backend = TorchDistBackend()
    t = torch.zeros(4, dtype=torch.float32)

    # Skip the test entirely if dist is not initialised in this run.
    if not dist.is_available() or not dist.is_initialized():
        import pytest
        pytest.skip("distributed not initialised")

    out = backend.all_reduce(t, op="sum", async_op=True)
    assert captured["async_op"] is True
    # A Work object is not a Tensor.
    assert not isinstance(out, Tensor)
    # Sanity: the Work object has the documented API surface.
    assert hasattr(out, "wait")
    assert hasattr(out, "is_completed")
    # Drain the async work so other tests start from a clean state.
    out.wait()
    monkeypatch.undo()
