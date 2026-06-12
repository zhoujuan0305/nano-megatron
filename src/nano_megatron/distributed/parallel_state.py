from __future__ import annotations

import torch.distributed as dist

_DP_GROUP: dist.ProcessGroup | None = None


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
    logger_info_dp_group()
    return _DP_GROUP


def logger_info_dp_group() -> None:
    import logging

    logger = logging.getLogger(__name__)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    logger.info(
        f"rank={rank} world_size={world_size} "
        f"dp_rank={get_data_parallel_rank()} dp_size={get_data_parallel_size()}"
    )


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


def get_global_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_global_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()
