from __future__ import annotations

import logging

import torch.distributed as dist

_DP_GROUP: dist.ProcessGroup | None = None
_TP_GROUP: dist.ProcessGroup | None = None

logger = logging.getLogger(__name__)


def create_data_parallel_group() -> dist.ProcessGroup:
    global _DP_GROUP
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Call init_distributed() before creating process groups."
        )
    world_size = dist.get_world_size()
    all_ranks = list(range(world_size))
    _DP_GROUP = dist.new_group(all_ranks)
    rank = dist.get_rank()
    dp_rank = dist.get_rank(group=_DP_GROUP)
    dp_size = dist.get_world_size(group=_DP_GROUP)
    logger.info(f"rank={rank} dp_rank={dp_rank} dp_size={dp_size}")
    return _DP_GROUP


def create_tensor_parallel_group(tp_size: int | None = None) -> dist.ProcessGroup:
    global _TP_GROUP
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Call init_distributed() before creating process groups."
        )
    world_size = dist.get_world_size()
    if tp_size is None:
        tp_size = world_size
    if tp_size > world_size:
        raise ValueError(f"tp_size ({tp_size}) cannot be larger than world_size ({world_size})")
    if world_size % tp_size != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by tp_size ({tp_size})")
    rank = dist.get_rank()
    tp_rank = rank % tp_size
    tp_group_ranks = list(range(rank - tp_rank, rank - tp_rank + tp_size))
    _TP_GROUP = dist.new_group(tp_group_ranks)
    logger.info(f"rank={rank} tp_rank={tp_rank} tp_size={tp_size} tp_group={tp_group_ranks}")
    return _TP_GROUP


def get_data_parallel_group() -> dist.ProcessGroup:
    global _DP_GROUP
    if _DP_GROUP is not None:
        return _DP_GROUP
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Data parallel group not created. Call create_data_parallel_group() first."
        )
    _DP_GROUP = create_data_parallel_group()
    return _DP_GROUP


def get_data_parallel_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank(group=get_data_parallel_group())


def get_data_parallel_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group=get_data_parallel_group())


def get_tensor_parallel_group() -> dist.ProcessGroup:
    global _TP_GROUP
    if _TP_GROUP is not None:
        return _TP_GROUP
    raise RuntimeError(
        "Tensor parallel group not created. Call create_tensor_parallel_group() first."
    )


def get_tensor_parallel_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank(group=get_tensor_parallel_group())


def get_tensor_parallel_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group=get_tensor_parallel_group())


def get_global_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_global_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()
