"""TP2 end-to-end numerical equivalence test for async all_reduce overlap.

Verifies that enabling async_op=True with work.wait() fence in
_ReduceFromTPRegion.forward and _CopyToTPRegion.backward produces identical
logits and per-parameter gradients to a synchronous (monkeypatched) baseline.

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
    """TP2 2-layer nano-megatron model: sync vs async logits/grads at 1e-6.

    Strategy: build one TP model, run forward+backward twice:
    1. Sync baseline (monkeypatch TorchDistBackend.all_reduce to force async_op=False)
    2. Async path (restore original all_reduce with async_op=True support)

    Class-level monkeypatch affects all TorchDistBackend instances, so the same
    model uses different all_reduce behavior in each pass.
    """
    from nano_megatron.distributed.torch_backend import TorchDistBackend
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
    ref = ReferenceGPT(cfg)
    ctx = initialize_parallel(
        ParallelConfig(tensor_parallel_size=ws), dist_backend="nccl"
    )
    ids = torch.randint(0, 128, (2, 16), device="cuda")
    m = build_tp_gpt_from_reference(ref, ctx).cuda()

    # --- Sync baseline: monkeypatch to force async_op=False ---
    orig_all_reduce = TorchDistBackend.all_reduce

    def force_sync(self, tensor, *, group=None, op="sum", async_op=False):
        return orig_all_reduce(self, tensor, group=group, op=op, async_op=False)

    TorchDistBackend.all_reduce = force_sync
    logits_sync = m(ids)
    loss_sync = m.shifted_cross_entropy(logits_sync, ids)
    loss_sync.backward()
    logits_sync_v = logits_sync.detach().clone()
    grads_sync = {
        n: p.grad.detach().clone()
        for n, p in m.named_parameters()
        if p.grad is not None
    }
    m.zero_grad()

    # --- Async path (default async_op=True from Task 2) ---
    TorchDistBackend.all_reduce = orig_all_reduce
    logits_async = m(ids)
    loss_async = m.shifted_cross_entropy(logits_async, ids)
    loss_async.backward()
    torch.cuda.synchronize()
    logits_async_v = logits_async.detach().clone()
    grads_async = {
        n: p.grad.detach().clone()
        for n, p in m.named_parameters()
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
