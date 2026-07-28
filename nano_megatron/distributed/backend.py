from __future__ import annotations

from typing import Any, Protocol

from torch import Tensor


class CommBackend(Protocol):
    def all_reduce(
        self, tensor: Tensor, *, group: Any | None = None, op: str = "sum",
        async_op: bool = False,
    ) -> Tensor | Any: ...

    def reduce_scatter(
        self,
        output: Tensor,
        input_list: list[Tensor],
        *,
        group: Any | None = None,
        op: str = "sum",
    ) -> Tensor: ...

    def all_gather(
        self,
        tensor_list: list[Tensor],
        tensor: Tensor,
        *,
        group: Any | None = None,
    ) -> list[Tensor]: ...

    def all_gather_into_tensor(
        self,
        output: Tensor,
        input: Tensor,
        *,
        group: Any | None = None,
    ) -> Tensor: ...

    def send(self, tensor: Tensor, dst: int, *, group: Any | None = None) -> None: ...

    def recv(self, tensor: Tensor, src: int, *, group: Any | None = None) -> Tensor: ...

    def broadcast(
        self, tensor: Tensor, src: int, *, group: Any | None = None
    ) -> Tensor: ...

    def barrier(self, *, group: Any | None = None) -> None: ...
