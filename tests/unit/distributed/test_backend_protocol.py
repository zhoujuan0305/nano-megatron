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


def test_broadcast_method_exists():
    backend: CommBackend = TorchDistBackend()
    assert hasattr(backend, "broadcast")
    assert callable(backend.broadcast)


@pytest.mark.parametrize(
    "method_name", ["all_reduce", "reduce_scatter", "all_gather", "send", "recv"]
)
def test_async_operations_are_part_of_backend_contract(method_name):
    backend: CommBackend = TorchDistBackend()
    assert hasattr(backend, method_name)
    assert "async_op" in __import__("inspect").signature(
        getattr(backend, method_name)
    ).parameters


def test_async_send_and_recv_use_nonblocking_primitives(monkeypatch):
    import torch.distributed as dist

    sentinel_send = object()
    sentinel_recv = object()
    monkeypatch.setattr(dist, "isend", lambda *args, **kwargs: sentinel_send)
    monkeypatch.setattr(dist, "irecv", lambda *args, **kwargs: sentinel_recv)
    backend = TorchDistBackend()
    tensor = torch.ones(2)

    assert backend.send(tensor, 1, async_op=True) is sentinel_send
    assert backend.recv(tensor, 1, async_op=True) is sentinel_recv


def test_batch_p2p_maps_backend_neutral_operations(monkeypatch):
    import torch.distributed as dist

    from nano_megatron.distributed import P2POperation

    captured = []
    sentinel = [object(), object()]

    class FakeP2POp:
        def __init__(self, op, tensor, peer, group, tag):
            self.op = op
            self.tensor = tensor
            self.peer = peer
            self.group = group
            self.tag = tag

    def fake_batch(operations):
        captured.extend(operations)
        return sentinel

    monkeypatch.setattr(dist, "batch_isend_irecv", fake_batch)
    monkeypatch.setattr(dist, "P2POp", FakeP2POp)
    backend = TorchDistBackend()
    send = torch.ones(2)
    recv = torch.empty(2)
    result = backend.batch_p2p(
        [
            P2POperation("recv", recv, peer=1),
            P2POperation("send", send, peer=1),
        ]
    )

    assert result is sentinel
    assert [op.op for op in captured] == [dist.irecv, dist.isend]
    assert [op.tensor for op in captured] == [recv, send]


def test_broadcast_calls_dist_broadcast(monkeypatch):
    import torch.distributed as dist

    captured = {}

    def fake_broadcast(tensor, src, group=None):
        captured["src"] = src
        captured["group"] = group
        return None

    monkeypatch.setattr(dist, "broadcast", fake_broadcast)
    backend = TorchDistBackend()
    t = torch.ones(3, dtype=torch.float32)
    out = backend.broadcast(t, src=0, group=None)
    assert captured["src"] == 0
    assert out is t
