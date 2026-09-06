from __future__ import annotations

from typing import Any, Protocol

from torch import Tensor


class CommWork(Protocol):
    """Completion handle returned by an asynchronous communication launch."""

    def wait(self) -> bool: ...

    def is_completed(self) -> bool: ...


class CommBackend(Protocol):
    def all_reduce(
        self, tensor: Tensor, *, group: Any | None = None, op: str = "sum",
        async_op: bool = False,
    ) -> Tensor | CommWork: ...

    def reduce_scatter(
        self,
        output: Tensor,
        input_list: list[Tensor],
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

    def broadcast(
        self, tensor: Tensor, src: int, *, group: Any | None = None
    ) -> Tensor: ...

    def barrier(self, *, group: Any | None = None) -> None: ...
