from __future__ import annotations

from typing import Any

import torch.distributed as dist
from torch import Tensor

from nano_megatron.distributed.backend import P2POperation

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
        async_op: bool = False,
    ) -> Tensor | Any:
        work = dist.reduce_scatter(
            output,
            input_list,
            op=reduce_op_from_string(op),
            group=group,
            async_op=async_op,
        )
        return work if async_op else output

    def all_gather(
        self,
        tensor_list: list[Tensor],
        tensor: Tensor,
        *,
        group: Any | None = None,
        async_op: bool = False,
    ) -> list[Tensor] | Any:
        work = dist.all_gather(
            tensor_list, tensor, group=group, async_op=async_op
        )
        return work if async_op else tensor_list

    def all_gather_into_tensor(
        self,
        output: Tensor,
        input: Tensor,
        *,
        group: Any | None = None,
        async_op: bool = False,
    ) -> Tensor | Any:
        work = dist.all_gather_into_tensor(
            output, input, group=group, async_op=async_op
        )
        return work if async_op else output

    def send(
        self,
        tensor: Tensor,
        dst: int,
        *,
        group: Any | None = None,
        tag: int = 0,
        async_op: bool = False,
    ) -> None | Any:
        if async_op:
            return dist.isend(tensor, dst=dst, group=group, tag=tag)
        dist.send(tensor, dst=dst, group=group, tag=tag)
        return None

    def recv(
        self,
        tensor: Tensor,
        src: int,
        *,
        group: Any | None = None,
        tag: int = 0,
        async_op: bool = False,
    ) -> Tensor | Any:
        if async_op:
            return dist.irecv(tensor, src=src, group=group, tag=tag)
        dist.recv(tensor, src=src, group=group, tag=tag)
        return tensor

    def batch_p2p(self, operations: list[P2POperation]) -> list[Any]:
        if not operations:
            return []
        torch_operations: list[dist.P2POp] = []
        for operation in operations:
            function = dist.isend if operation.kind == "send" else dist.irecv
            torch_operations.append(
                dist.P2POp(
                    function,
                    operation.tensor,
                    operation.peer,
                    operation.group,
                    operation.tag,
                )
            )
        return dist.batch_isend_irecv(torch_operations)

    def broadcast(
        self, tensor: Tensor, src: int, *, group: Any | None = None
    ) -> Tensor:
        dist.broadcast(tensor, src=src, group=group)
        return tensor

    def barrier(self, *, group: Any | None = None) -> None:
        dist.barrier(group=group)
