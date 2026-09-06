"""TP2 end-to-end numerical equivalence for dgrad/wgrad overlap.

Verifies that the custom Column Parallel backward which launches dgrad
AllReduce before wgrad GEMM produces the same logits and parameter gradients
as the straightforward autograd path.

Run with torchrun --nproc_per_node=2.
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
    master_port = str(29800 + hash(test_id) % 1000)
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
        f"tests/integration/test_async_overlap_numerical.py::{test_id}",
        "-v",
        "-s",
        "--import-mode=importlib",
    ]
    env = os.environ.copy()
    env["NANO_ASYNC_E2E_WORKER"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = master_port
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


@pytest.mark.skipif(
    os.environ.get("NANO_ASYNC_E2E_WORKER") == "1", reason="launcher only"
)
def test_launch_async_overlap_numerical():
    _run_torchrun(2, "test_worker_async_overlap_numerical")


@pytest.mark.skipif(
    os.environ.get("NANO_ASYNC_E2E_WORKER") != "1", reason="worker only"
)
def test_worker_async_overlap_numerical():
    """TP2 2-layer model: regular vs overlapped Column backward at 1e-6."""
    from dataclasses import replace

    from nano_megatron.model import build_tp_gpt_from_reference
    from nano_megatron.parallel import (
        ParallelConfig,
        destroy_parallel,
        initialize_parallel,
        is_parallel_initialized,
    )
    from nano_megatron.reference import ReferenceGPT, ReferenceGPTConfig
    ws = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    assert ws == 2

    cfg = ReferenceGPTConfig(
        vocab_size=128,
        max_seq_len=16,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        ffn_hidden_size=64,
        layernorm_eps=1e-5,
        use_bias=True,
        tie_word_embeddings=False,
    )

    if is_parallel_initialized():
        destroy_parallel()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    ref_sync = ReferenceGPT(cfg)
    overlap_cfg = replace(cfg, tp_comm_overlap=True)
    ref_overlap = ReferenceGPT(overlap_cfg)
    ref_overlap.load_state_dict(ref_sync.state_dict())
    ctx = initialize_parallel(
        ParallelConfig(tensor_parallel_size=ws), dist_backend="nccl"
    )
    ids = torch.randint(0, 128, (2, 16), device="cuda")
    model_sync = build_tp_gpt_from_reference(ref_sync, ctx).cuda()
    model_overlap = build_tp_gpt_from_reference(ref_overlap, ctx).cuda()

    logits_sync = model_sync(ids)
    loss_sync = model_sync.shifted_cross_entropy(logits_sync, ids)
    loss_sync.backward()
    logits_sync_v = logits_sync.detach().clone()
    grads_sync = {
        n: p.grad.detach().clone()
        for n, p in model_sync.named_parameters()
        if p.grad is not None
    }

    logits_async = model_overlap(ids)
    loss_async = model_overlap.shifted_cross_entropy(logits_async, ids)
    loss_async.backward()
    torch.cuda.synchronize()
    logits_async_v = logits_async.detach().clone()
    grads_async = {
        n: p.grad.detach().clone()
        for n, p in model_overlap.named_parameters()
        if p.grad is not None
    }

    # --- Assertions ---
    logit_diff = (logits_sync_v - logits_async_v).abs().max().item()
    assert logit_diff < 1e-6, (
        f"rank {rank}: logits async-vs-sync max diff = {logit_diff:.3e}"
    )
    for name in grads_sync:
        assert name in grads_async, f"rank {rank}: missing grad {name} in async"
        diff = (grads_sync[name] - grads_async[name]).abs().max().item()
        assert diff < 1e-6, (
            f"rank {rank}: grad {name} async-vs-sync max diff = {diff:.3e}"
        )

    destroy_parallel()
