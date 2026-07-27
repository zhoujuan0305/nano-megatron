from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from nano_megatron.distributed.backend import CommBackend
from nano_megatron.distributed.torch_backend import TorchDistBackend
from nano_megatron.parallel.config import ParallelConfig
from nano_megatron.parallel.rank_generator import RankGenerator


@dataclass
class ParallelContext:
    config: ParallelConfig
    rank: int
    world_size: int
    local_rank: int
    backend: CommBackend
    tensor_parallel_size: int
    tensor_parallel_rank: int
    data_parallel_size: int
    data_parallel_rank: int
    pipeline_parallel_size: int
    pipeline_parallel_rank: int
    context_parallel_size: int
    context_parallel_rank: int
    tensor_parallel_group: Any
    data_parallel_group: Any
    pipeline_parallel_group: Any
    context_parallel_group: Any
    sequence_parallel: bool
    data_context_parallel_group: Any


_PARALLEL_CONTEXT: ParallelContext | None = None
_DIST_INITIALIZED_BY_US: bool = False


def _resolve_rank_world(
    rank: int | None, world_size: int | None
) -> tuple[int | None, int | None]:
    if rank is None and "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    if world_size is None and "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    return rank, world_size


def _resolve_local_rank(rank: int) -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        return rank % torch.cuda.device_count()
    return 0


def _create_group_for_rank(rank: int, rank_lists: list[list[int]]) -> Any:
    # Every rank must call new_group with the same lists in the same order.
    mine: Any = None
    for ranks in rank_lists:
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            mine = group
    if mine is None:
        raise RuntimeError(f"rank {rank} not found in any group of {rank_lists}")
    return mine


def initialize_parallel(
    config: ParallelConfig | None = None,
    *,
    backend: CommBackend | None = None,
    init_method: str | None = None,
    dist_backend: str | None = None,
    rank: int | None = None,
    world_size: int | None = None,
) -> ParallelContext:
    global _PARALLEL_CONTEXT, _DIST_INITIALIZED_BY_US

    if _PARALLEL_CONTEXT is not None:
        raise RuntimeError("parallel context already initialized")

    cfg = config if config is not None else ParallelConfig()

    if not dist.is_initialized():
        backend_name = dist_backend or (
            "nccl" if torch.cuda.is_available() else "gloo"
        )
        resolved_rank, resolved_world = _resolve_rank_world(rank, world_size)
        # NCCL binds to the current CUDA device at init_process_group time.
        if backend_name == "nccl" and torch.cuda.is_available():
            if "LOCAL_RANK" in os.environ:
                torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            elif resolved_rank is not None:
                torch.cuda.set_device(_resolve_local_rank(resolved_rank))
        init_kwargs: dict[str, Any] = {"backend": backend_name}
        if init_method is not None:
            init_kwargs["init_method"] = init_method
        if resolved_rank is not None:
            init_kwargs["rank"] = resolved_rank
        if resolved_world is not None:
            init_kwargs["world_size"] = resolved_world
        dist.init_process_group(**init_kwargs)
        _DIST_INITIALIZED_BY_US = True
    else:
        if world_size is not None and world_size != dist.get_world_size():
            raise RuntimeError(
                f"dist already initialized with world_size={dist.get_world_size()}, "
                f"got world_size={world_size}"
            )
        if rank is not None and rank != dist.get_rank():
            raise RuntimeError(
                f"dist already initialized with rank={dist.get_rank()}, got rank={rank}"
            )

    world = dist.get_world_size()
    global_rank = dist.get_rank()
    cfg.validate(world)

    dp = cfg.resolved_data_parallel_size(world)
    local_rank = _resolve_local_rank(global_rank)

    # Reuse-dist path (and post-init safety): device must match local_rank for NCCL.
    if torch.cuda.is_available() and dist.get_backend() == "nccl":
        torch.cuda.set_device(local_rank)

    rg = RankGenerator(
        tp=cfg.tensor_parallel_size,
        dp=dp,
        pp=cfg.pipeline_parallel_size,
        cp=cfg.context_parallel_size,
        order=cfg.order,
    )
    parts = rg.decode(global_rank)

    tp_group = _create_group_for_rank(global_rank, rg.get_ranks("tp"))
    dp_group = _create_group_for_rank(global_rank, rg.get_ranks("dp"))
    pp_group = _create_group_for_rank(global_rank, rg.get_ranks("pp"))
    cp_group = _create_group_for_rank(global_rank, rg.get_ranks("cp"))
    dp_cp_group = _create_group_for_rank(global_rank, rg.get_ranks("dp-cp"))

    comm_backend: CommBackend = backend if backend is not None else TorchDistBackend()

    ctx = ParallelContext(
        config=cfg,
        rank=global_rank,
        world_size=world,
        local_rank=local_rank,
        backend=comm_backend,
        tensor_parallel_size=cfg.tensor_parallel_size,
        tensor_parallel_rank=parts["tp"],
        data_parallel_size=dp,
        data_parallel_rank=parts["dp"],
        pipeline_parallel_size=cfg.pipeline_parallel_size,
        pipeline_parallel_rank=parts["pp"],
        context_parallel_size=cfg.context_parallel_size,
        context_parallel_rank=parts["cp"],
        tensor_parallel_group=tp_group,
        data_parallel_group=dp_group,
        pipeline_parallel_group=pp_group,
        context_parallel_group=cp_group,
        sequence_parallel=cfg.sequence_parallel,
        data_context_parallel_group=dp_cp_group,
    )
    _PARALLEL_CONTEXT = ctx
    return ctx


def get_parallel_context() -> ParallelContext:
    if _PARALLEL_CONTEXT is None:
        raise RuntimeError("parallel context is not initialized")
    return _PARALLEL_CONTEXT


def is_parallel_initialized() -> bool:
    return _PARALLEL_CONTEXT is not None


def destroy_parallel() -> None:
    global _PARALLEL_CONTEXT, _DIST_INITIALIZED_BY_US

    _PARALLEL_CONTEXT = None
    if _DIST_INITIALIZED_BY_US and dist.is_initialized():
        dist.destroy_process_group()
    _DIST_INITIALIZED_BY_US = False
