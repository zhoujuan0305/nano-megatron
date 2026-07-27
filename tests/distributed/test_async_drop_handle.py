"""Stress test: 200 async all_reduces with work.wait() GPU-side fence.

Verifies the async launch + fence pattern used in production code
(mappings.py) is correct under repeated use.

Requires torchrun --nproc_per_node=2 (NCCL on CUDA).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.distributed.common import require_nccl_gpus

REPO = Path(__file__).resolve().parents[2]


def _run_torchrun(nproc: int, test_id: str) -> None:
    require_nccl_gpus(nproc)
    master_port = str(29700 + hash(test_id) % 1000)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "--master_addr=127.0.0.1",
        f"--master_port={master_port}",
        "-m",
        "pytest",
        f"tests/distributed/test_async_drop_handle.py::{test_id}",
        "-v",
        "-s",
        "--import-mode=importlib",
    ]
    env = os.environ.copy()
    env["NANO_ASYNC_TP_WORKER"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = master_port
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


@pytest.mark.skipif(
    os.environ.get("NANO_ASYNC_TP_WORKER") == "1", reason="launcher only"
)
def test_launch_drop_200_async_reduces():
    _run_torchrun(2, "test_worker_drop_200_async_reduces")


@pytest.mark.skipif(
    os.environ.get("NANO_ASYNC_TP_WORKER") != "1", reason="worker only"
)
def test_worker_drop_200_async_reduces():
    """200 async all_reduces with work.wait() fence; verify correctness."""
    import torch.distributed as dist

    from nano_megatron.distributed.torch_backend import TorchDistBackend

    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    assert ws == 2
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=ws)

    backend = TorchDistBackend()
    size = 1024
    expected = float(sum(range(1, ws + 1)))  # 3.0 for ws=2

    # Async + work.wait() pattern (matches production code in mappings.py).
    for i in range(200):
        t = torch.full((size,), float(rank + 1), dtype=torch.float32,
                       device=f"cuda:{rank}")
        work = backend.all_reduce(t, group=dist.group.WORLD, op="sum",
                                  async_op=True)
        work.wait()  # GPU-side fence: default stream waits for NCCL stream
        if i == 199:
            async_final = t.clone()

    # Sync sanity baseline.
    t = torch.full((size,), float(rank + 1), dtype=torch.float32,
                   device=f"cuda:{rank}")
    backend.all_reduce(t, group=dist.group.WORLD, op="sum", async_op=False)
    sync_final = t.clone()

    expected_t = torch.full((size,), expected, dtype=torch.float32,
                             device=f"cuda:{rank}")
    assert torch.allclose(async_final, expected_t, atol=1e-6), (
        f"rank {rank}: async final {async_final[0].item()} != {expected}"
    )
    assert torch.allclose(sync_final, expected_t, atol=1e-6), (
        f"rank {rank}: sync final {sync_final[0].item()} != {expected}"
    )

    dist.destroy_process_group()
