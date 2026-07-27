from __future__ import annotations

from typing import Any

import torch.distributed as dist
from torch import Tensor

_OP_MAP = {
    "sum": dist.ReduceOp.SUM,
    "max": dist.ReduceOp.MAX,
    "min": dist.ReduceOp.MIN,
    "product": dist.ReduceOp.PRODUCT,
}


def reduce_op_from_string(op: str) -> dist.ReduceOp:
    key = op.lower()
    if key not in _OP_MAP:
        raise ValueError(f"unsupported reduce op: {op!r}")
    return _OP_MAP[key]


class TorchDistBackend:
    def all_reduce(
        self, tensor: Tensor, *, group: Any | None = None, op: str = "sum",
        async_op: bool = False,
    ) -> Tensor | Any:
        work = dist.all_reduce(
            tensor, op=reduce_op_from_string(op), group=group, async_op=async_op,
        )
        return work if async_op else tensor

    def reduce_scatter(
        self,
        output: Tensor,
        input_list: list[Tensor],
        *,
        group: Any | None = None,
        op: str = "sum",
    ) -> Tensor:
        dist.reduce_scatter(
            output, input_list, op=reduce_op_from_string(op), group=group
        )
        return output

    def all_gather(
        self,
        tensor_list: list[Tensor],
        tensor: Tensor,
        *,
        group: Any | None = None,
    ) -> list[Tensor]:
        dist.all_gather(tensor_list, tensor, group=group)
        return tensor_list

    def send(self, tensor: Tensor, dst: int, *, group: Any | None = None) -> None:
        dist.send(tensor, dst, group=group)

    def recv(self, tensor: Tensor, src: int, *, group: Any | None = None) -> Tensor:
        dist.recv(tensor, src, group=group)
        return tensor

    def barrier(self, *, group: Any | None = None) -> None:
        dist.barrier(group=group)
