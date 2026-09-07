from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from torch import Tensor

from nano_megatron.distributed.backend import (
    CollectiveBackend,
    CommBackend,
    CommWork,
    P2POperation,
)


@dataclass(frozen=True)
class CollectiveRoute:
    """Bind one process-group object to its collective implementation."""

    name: str
    group: Any
    backend: CollectiveBackend


class RoutedCommBackend:
    """Route selected process groups to alternate collective backends.

    Point-to-point operations, broadcasts, barriers, and collectives on groups
    without a route remain on the fallback backend. Group matching deliberately
    uses object identity because PyTorch process groups are opaque handles.
    """

    def __init__(
        self,
        fallback: CommBackend,
        routes: Iterable[CollectiveRoute],
        *,
        own_routes: bool = False,
    ) -> None:
        self._fallback = fallback
        self._routes = tuple(routes)
        self._own_routes = own_routes
        self._closed = False
        for index, route in enumerate(self._routes):
            for earlier in self._routes[:index]:
                if route.group is earlier.group:
                    raise ValueError(
                        f"collective group has duplicate routes "
                        f"{earlier.name!r} and {route.name!r}"
                    )

    @property
    def route_names(self) -> tuple[str, ...]:
        return tuple(route.name for route in self._routes)

    @property
    def routes(self) -> tuple[CollectiveRoute, ...]:
        return self._routes

    def _collectives(self, group: Any | None) -> CollectiveBackend:
        if self._closed:
            raise RuntimeError("RoutedCommBackend is closed")
        for route in self._routes:
            if group is route.group:
                return route.backend
        return self._fallback

    def all_reduce(
        self,
        tensor: Tensor,
        *,
        group: Any | None = None,
        op: str = "sum",
        async_op: bool = False,
    ) -> Tensor | CommWork:
        return self._collectives(group).all_reduce(
            tensor, group=group, op=op, async_op=async_op
        )

    def reduce_scatter(
        self,
        output: Tensor,
        input_list: list[Tensor],
        *,
        group: Any | None = None,
        op: str = "sum",
        async_op: bool = False,
    ) -> Tensor | CommWork:
        return self._collectives(group).reduce_scatter(
            output,
            input_list,
            group=group,
            op=op,
            async_op=async_op,
        )

    def reduce_scatter_tensor(
        self,
        output: Tensor,
        input: Tensor,
        *,
        group: Any | None = None,
        op: str = "sum",
        async_op: bool = False,
    ) -> Tensor | CommWork:
        return self._collectives(group).reduce_scatter_tensor(
            output,
            input,
            group=group,
            op=op,
            async_op=async_op,
        )

    def all_gather(
        self,
        tensor_list: list[Tensor],
        tensor: Tensor,
        *,
        group: Any | None = None,
        async_op: bool = False,
    ) -> list[Tensor] | CommWork:
        return self._collectives(group).all_gather(
            tensor_list, tensor, group=group, async_op=async_op
        )

    def all_gather_into_tensor(
        self,
        output: Tensor,
        input: Tensor,
        *,
        group: Any | None = None,
        async_op: bool = False,
    ) -> Tensor | CommWork:
        return self._collectives(group).all_gather_into_tensor(
            output, input, group=group, async_op=async_op
        )

    def send(
        self,
        tensor: Tensor,
        dst: int,
        *,
        group: Any | None = None,
        tag: int = 0,
        async_op: bool = False,
    ) -> None | CommWork:
        return self._fallback.send(
            tensor, dst, group=group, tag=tag, async_op=async_op
        )

    def recv(
        self,
        tensor: Tensor,
        src: int,
        *,
        group: Any | None = None,
        tag: int = 0,
        async_op: bool = False,
    ) -> Tensor | CommWork:
        return self._fallback.recv(
            tensor, src, group=group, tag=tag, async_op=async_op
        )

    def batch_p2p(self, operations: list[P2POperation]) -> list[CommWork]:
        return self._fallback.batch_p2p(operations)

    def broadcast(
        self, tensor: Tensor, src: int, *, group: Any | None = None
    ) -> Tensor:
        return self._fallback.broadcast(tensor, src, group=group)

    def barrier(self, *, group: Any | None = None) -> None:
        self._fallback.barrier(group=group)

    def close(self) -> None:
        if self._closed:
            return
        errors: list[Exception] = []
        if self._own_routes:
            closed: set[int] = set()
            for route in reversed(self._routes):
                backend_id = id(route.backend)
                if backend_id in closed:
                    continue
                closed.add(backend_id)
                close = getattr(route.backend, "close", None)
                if close is None:
                    continue
                try:
                    close()
                except Exception as error:  # Preserve teardown of other communicators.
                    errors.append(error)
        self._closed = True
        if errors:
            raise RuntimeError(
                "collective backend teardown failed: "
                + "; ".join(str(error) for error in errors)
            ) from errors[0]

    def __enter__(self) -> RoutedCommBackend:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
