from __future__ import annotations

import torch
import torch.distributed as dist


def reduce_tensor(
    tensor: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    average: bool = True,
) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    if group is None:
        from nano_megatron.distributed.parallel_state import get_data_parallel_group

        group = get_data_parallel_group()
    world_size = dist.get_world_size(group=group)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    if average and world_size > 1:
        tensor.div_(world_size)
    return tensor


def average_gradients(
    model: torch.nn.Module,
    group: dist.ProcessGroup | None = None,
) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    if group is None:
        from nano_megatron.distributed.parallel_state import get_data_parallel_group

        group = get_data_parallel_group()
    world_size = dist.get_world_size(group=group)
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=group)
            if world_size > 1:
                param.grad.data.div_(world_size)


def broadcast_parameters(
    model: torch.nn.Module,
    src: int = 0,
    group: dist.ProcessGroup | None = None,
) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    if group is None:
        from nano_megatron.distributed.parallel_state import get_data_parallel_group

        group = get_data_parallel_group()
    for param in model.parameters():
        dist.broadcast(param.data, src=src, group=group)
