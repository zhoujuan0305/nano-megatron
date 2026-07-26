from __future__ import annotations

import os

import pytest
import torch


def require_nccl_gpus(n: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < n:
        pytest.skip(f"need {n} CUDA devices for NCCL test")
    if not torch.distributed.is_nccl_available():
        pytest.skip("NCCL not available")


def env_rank_info() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world, local
