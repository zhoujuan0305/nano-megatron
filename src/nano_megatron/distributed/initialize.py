from __future__ import annotations

import logging
from datetime import timedelta

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_INITIALIZED = False


def init_distributed(backend: str | None = None, timeout_minutes: int = 30) -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    if not dist.is_available():
        logger.info("torch.distributed not available, running in single-process mode")
        return
    if not dist.is_initialized():
        if backend is None:
            backend = "nccl" if dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))
        _INITIALIZED = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"Initialized distributed: rank={rank} world_size={world_size} "
            f"backend={backend} device={device}"
        )
    else:
        _INITIALIZED = True


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def destroy_distributed() -> None:
    global _INITIALIZED
    if _INITIALIZED and dist.is_initialized():
        dist.destroy_process_group()
        _INITIALIZED = False
        logger.info("Destroyed distributed process group")
