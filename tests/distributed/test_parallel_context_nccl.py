"""Run with:
torchrun --standalone --nproc_per_node=4 -m pytest tests/distributed/test_parallel_context_nccl.py -v
or the helper invocation in the test file via subprocess from a parent test.
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
    # Bind control plane to loopback so tests do not depend on cluster DNS/hostname.
    master_port = str(29500 + nproc + hash(test_id) % 1000)
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
        f"tests/distributed/test_parallel_context_nccl.py::{test_id}",
        "-v",
        "-s",
        "--import-mode=importlib",
    ]
    env = os.environ.copy()
    env["NANO_MEGATRON_NCCL_WORKER"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = master_port
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


@pytest.mark.skipif(os.environ.get("NANO_MEGATRON_NCCL_WORKER") == "1", reason="launcher only")
def test_launch_tp2_dp2_ranks():
    _run_torchrun(4, "test_worker_tp2_dp2_ranks")


@pytest.mark.skipif(os.environ.get("NANO_MEGATRON_NCCL_WORKER") != "1", reason="worker only")
def test_worker_tp2_dp2_ranks():
    from nano_megatron.distributed import TorchDistBackend
    from nano_megatron.parallel import (
        ParallelConfig,
        RankGenerator,
        destroy_parallel,
        initialize_parallel,
        is_parallel_initialized,
    )

    if is_parallel_initialized():
        destroy_parallel()

    cfg = ParallelConfig(tensor_parallel_size=2, data_parallel_size=2)
    ctx = initialize_parallel(cfg, backend=TorchDistBackend(), dist_backend="nccl")
    assert ctx.world_size == 4
    assert ctx.tensor_parallel_size == 2
    assert ctx.data_parallel_size == 2
    assert ctx.pipeline_parallel_size == 1

    t = torch.ones(1, device="cuda") * (ctx.rank + 1)
    ctx.backend.all_reduce(t, group=ctx.data_parallel_group)

    rg = RankGenerator(
        tp=cfg.tensor_parallel_size,
        dp=2,
        pp=cfg.pipeline_parallel_size,
        cp=cfg.context_parallel_size,
        order=cfg.order,
    )
    my_group = next(g for g in rg.get_ranks("dp") if ctx.rank in g)
    expected = float(sum(r + 1 for r in my_group))
    assert t.item() == expected
    destroy_parallel()


@pytest.mark.skipif(os.environ.get("NANO_MEGATRON_NCCL_WORKER") == "1", reason="launcher only")
def test_launch_all_reduce_tp():
    _run_torchrun(2, "test_worker_all_reduce_tp")


@pytest.mark.skipif(os.environ.get("NANO_MEGATRON_NCCL_WORKER") != "1", reason="worker only")
def test_worker_all_reduce_tp():
    from nano_megatron.parallel import (
        ParallelConfig,
        destroy_parallel,
        initialize_parallel,
        is_parallel_initialized,
    )

    if is_parallel_initialized():
        destroy_parallel()

    cfg = ParallelConfig(tensor_parallel_size=2)
    ctx = initialize_parallel(cfg, dist_backend="nccl")
    x = torch.tensor([float(ctx.tensor_parallel_rank + 1)], device="cuda")
    ctx.backend.all_reduce(x, group=ctx.tensor_parallel_group)
    assert x.item() == 3.0  # 1+2
    destroy_parallel()


@pytest.mark.skipif(os.environ.get("NANO_MEGATRON_NCCL_WORKER") == "1", reason="launcher only")
def test_launch_tp_all_gather_reduce_scatter():
    _run_torchrun(2, "test_worker_tp_all_gather_reduce_scatter")


@pytest.mark.skipif(os.environ.get("NANO_MEGATRON_NCCL_WORKER") != "1", reason="worker only")
def test_worker_tp_all_gather_reduce_scatter():
    from nano_megatron.parallel import (
        ParallelConfig,
        destroy_parallel,
        initialize_parallel,
        is_parallel_initialized,
    )

    if is_parallel_initialized():
        destroy_parallel()

    cfg = ParallelConfig(tensor_parallel_size=2)
    ctx = initialize_parallel(cfg, dist_backend="nccl")
    assert ctx.tensor_parallel_size == 2

    value = float(ctx.tensor_parallel_rank + 1)
    x = torch.ones(4, device="cuda") * value
    gathered = [torch.empty_like(x) for _ in range(2)]
    ctx.backend.all_gather(gathered, x, group=ctx.tensor_parallel_group)
    assert torch.equal(gathered[0], torch.ones_like(x))
    assert torch.equal(gathered[1], torch.ones_like(x) * 2)

    # Both ranks hold the same gathered list; reduce_scatter(sum) doubles each chunk.
    out = torch.empty_like(x)
    ctx.backend.reduce_scatter(out, gathered, group=ctx.tensor_parallel_group)
    expected = 2.0 * (ctx.tensor_parallel_rank + 1)
    assert torch.equal(out, torch.full_like(out, expected))
    destroy_parallel()


@pytest.mark.skipif(os.environ.get("NANO_MEGATRON_NCCL_WORKER") == "1", reason="launcher only")
def test_launch_destroy_reinit():
    _run_torchrun(2, "test_worker_destroy_reinit")


@pytest.mark.skipif(os.environ.get("NANO_MEGATRON_NCCL_WORKER") != "1", reason="worker only")
def test_worker_destroy_reinit():
    from nano_megatron.parallel import (
        ParallelConfig,
        destroy_parallel,
        initialize_parallel,
        is_parallel_initialized,
    )

    if is_parallel_initialized():
        destroy_parallel()

    cfg = ParallelConfig(tensor_parallel_size=2)
    ctx = initialize_parallel(cfg, dist_backend="nccl")
    destroy_parallel()

    ctx = initialize_parallel(cfg, dist_backend="nccl")
    x = torch.tensor([float(ctx.tensor_parallel_rank + 1)], device="cuda")
    ctx.backend.all_reduce(x, group=ctx.tensor_parallel_group)
    assert x.item() == 3.0
    destroy_parallel()
