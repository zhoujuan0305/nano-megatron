from __future__ import annotations

import torch

from nano_megatron.distributed.backend import P2POperation
from nano_megatron.distributed.routed_backend import (
    CollectiveRoute,
    RoutedCommBackend,
)


class _RecordingBackend:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[str, object | None]] = []
        self.closed = False

    def _record(self, operation: str, group: object | None) -> None:
        self.calls.append((operation, group))

    def all_reduce(self, tensor, *, group=None, op="sum", async_op=False):
        self._record("all_reduce", group)
        return tensor

    def reduce_scatter(
        self, output, input_list, *, group=None, op="sum", async_op=False
    ):
        self._record("reduce_scatter", group)
        return output

    def reduce_scatter_tensor(
        self, output, input, *, group=None, op="sum", async_op=False
    ):
        self._record("reduce_scatter_tensor", group)
        return output

    def all_gather(self, tensor_list, tensor, *, group=None, async_op=False):
        self._record("all_gather", group)
        return tensor_list

    def all_gather_into_tensor(
        self, output, input, *, group=None, async_op=False
    ):
        self._record("all_gather_into_tensor", group)
        return output

    def send(self, tensor, dst, *, group=None, tag=0, async_op=False):
        self._record("send", group)
        return None

    def recv(self, tensor, src, *, group=None, tag=0, async_op=False):
        self._record("recv", group)
        return tensor

    def batch_p2p(self, operations: list[P2POperation]):
        self._record("batch_p2p", operations[0].group if operations else None)
        return []

    def broadcast(self, tensor, src, *, group=None):
        self._record("broadcast", group)
        return tensor

    def barrier(self, *, group=None):
        self._record("barrier", group)

    def close(self):
        self.closed = True


def test_routes_collectives_by_group_identity() -> None:
    fallback = _RecordingBackend("torch")
    tp = _RecordingBackend("nano-tp")
    dp = _RecordingBackend("nano-dp")
    tp_group = object()
    dp_group = object()
    unregistered_group = object()
    backend = RoutedCommBackend(
        fallback,
        [
            CollectiveRoute("tp", tp_group, tp),
            CollectiveRoute("dp-cp", dp_group, dp),
        ],
    )
    tensor = torch.ones(2)

    assert backend.route_names == ("tp", "dp-cp")
    assert tuple(route.backend for route in backend.routes) == (tp, dp)

    backend.all_reduce(tensor, group=tp_group)
    backend.all_gather([tensor.clone(), tensor.clone()], tensor, group=tp_group)
    backend.reduce_scatter(tensor, [tensor, tensor], group=tp_group)
    backend.reduce_scatter_tensor(tensor, tensor, group=tp_group)
    backend.all_reduce(tensor, group=dp_group)
    backend.all_reduce(tensor, group=unregistered_group)

    assert [operation for operation, _ in tp.calls] == [
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "reduce_scatter_tensor",
    ]
    assert [operation for operation, _ in dp.calls] == ["all_reduce"]
    assert [operation for operation, _ in fallback.calls] == ["all_reduce"]


def test_point_to_point_and_control_operations_stay_on_fallback() -> None:
    fallback = _RecordingBackend("torch")
    collective = _RecordingBackend("nano")
    group = object()
    backend = RoutedCommBackend(
        fallback, [CollectiveRoute("tp", group, collective)]
    )
    tensor = torch.ones(2)

    backend.send(tensor, 1, group=group)
    backend.recv(tensor, 1, group=group)
    backend.batch_p2p([P2POperation("send", tensor, 1, group)])
    backend.broadcast(tensor, 0, group=group)
    backend.barrier(group=group)

    assert not collective.calls
    assert [operation for operation, _ in fallback.calls] == [
        "send",
        "recv",
        "batch_p2p",
        "broadcast",
        "barrier",
    ]


def test_owned_routes_close_once() -> None:
    fallback = _RecordingBackend("torch")
    collective = _RecordingBackend("nano")
    backend = RoutedCommBackend(
        fallback,
        [CollectiveRoute("tp", object(), collective)],
        own_routes=True,
    )

    backend.close()
    backend.close()

    assert collective.closed
    assert not fallback.closed
