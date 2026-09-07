from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from torch import Tensor


class CommWork(Protocol):
    """Completion handle returned by an asynchronous communication launch."""

    def wait(self) -> bool: ...

    def is_completed(self) -> bool: ...


class AllReduceBackend(Protocol):
    """The narrow collective interface consumed by DP gradient buckets."""

    def all_reduce(
        self, tensor: Tensor, *, group: Any | None = None, op: str = "sum",
        async_op: bool = False,
    ) -> Tensor | CommWork: ...


class CollectiveBackend(AllReduceBackend, Protocol):
    """Collective operations used by data and model parallelism."""

    def reduce_scatter(
        self,
        output: Tensor,
        input_list: list[Tensor],
        *,
        group: Any | None = None,
        op: str = "sum",
        async_op: bool = False,
    ) -> Tensor | CommWork: ...

    def reduce_scatter_tensor(
        self,
        output: Tensor,
        input: Tensor,
        *,
        group: Any | None = None,
        op: str = "sum",
        async_op: bool = False,
    ) -> Tensor | CommWork: ...

    def all_gather(
        self,
        tensor_list: list[Tensor],
        tensor: Tensor,
        *,
        group: Any | None = None,
        async_op: bool = False,
    ) -> list[Tensor] | CommWork: ...

    def all_gather_into_tensor(
        self,
        output: Tensor,
        input: Tensor,
        *,
        group: Any | None = None,
        async_op: bool = False,
    ) -> Tensor | CommWork: ...


@dataclass(frozen=True)
class P2POperation:
    kind: Literal["send", "recv"]
    tensor: Tensor
    peer: int
    group: Any | None = None
    tag: int = 0


class CommBackend(CollectiveBackend, Protocol):
    def send(
        self,
        tensor: Tensor,
        dst: int,
        *,
        group: Any | None = None,
        tag: int = 0,
        async_op: bool = False,
    ) -> None | CommWork: ...

    def recv(
        self,
        tensor: Tensor,
        src: int,
        *,
        group: Any | None = None,
        tag: int = 0,
        async_op: bool = False,
    ) -> Tensor | CommWork: ...

    def batch_p2p(self, operations: list[P2POperation]) -> list[CommWork]: ...

    def broadcast(
        self, tensor: Tensor, src: int, *, group: Any | None = None
    ) -> Tensor: ...

    def barrier(self, *, group: Any | None = None) -> None: ...
