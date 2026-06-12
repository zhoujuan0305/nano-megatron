from __future__ import annotations

import torch
import torch.distributed as dist


def copy_to_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return input_
    from nano_megatron.distributed.parallel_state import get_tensor_parallel_group

    group = get_tensor_parallel_group()
    dist.all_reduce(input_, group=group)
    return input_


def all_reduce(input_: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return input_
    from nano_megatron.distributed.parallel_state import get_tensor_parallel_group

    group = get_tensor_parallel_group()
    dist.all_reduce(input_, group=group)
    return input_
