"""Run with:
torchrun --standalone --nproc_per_node=2 --master_addr=127.0.0.1 \
  -m pytest tests/distributed/test_tp_gloo.py -v -s --import-mode=importlib
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]


def _run_torchrun(nproc: int, test_id: str) -> None:
    master_port = str(29600 + hash(test_id) % 1000)
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
        f"tests/distributed/test_tp_gloo.py::{test_id}",
        "-v",
        "-s",
        "--import-mode=importlib",
    ]
    env = os.environ.copy()
    env["NANO_MEGATRON_TP_WORKER"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = master_port
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


@pytest.mark.skipif(os.environ.get("NANO_MEGATRON_TP_WORKER") == "1", reason="launcher only")
def test_launch_tp2_forward_backward_gloo():
    _run_torchrun(2, "test_worker_tp2_forward_backward_gloo")


@pytest.mark.skipif(os.environ.get("NANO_MEGATRON_TP_WORKER") != "1", reason="worker only")
def test_worker_tp2_forward_backward_gloo():
    from nano_megatron.model import build_tp_gpt_from_reference
    from nano_megatron.parallel import (
        ParallelConfig,
        destroy_parallel,
        initialize_parallel,
        is_parallel_initialized,
    )
    from nano_megatron.reference import ReferenceGPT, ReferenceGPTConfig
    from nano_megatron.reference.loss import shifted_cross_entropy

    if is_parallel_initialized():
        destroy_parallel()
    cfg = ReferenceGPTConfig(
        vocab_size=16,
        max_seq_len=8,
        hidden_size=8,
        num_layers=2,
        num_heads=4,
        ffn_hidden_size=16,
        layernorm_eps=1e-5,
        use_bias=True,
        tie_word_embeddings=False,
    )
    ctx = initialize_parallel(
        ParallelConfig(tensor_parallel_size=2), dist_backend="gloo"
    )
    assert ctx.tensor_parallel_size == 2
    torch.manual_seed(0)
    ref = ReferenceGPT(cfg)
    tp = build_tp_gpt_from_reference(ref, ctx)
    ids = torch.randint(0, 16, (2, 6))

    ref_logits = ref(ids)
    tp_logits = tp(ids)
    # RowParallel forward splits the matmul across TP ranks then all-reduces,
    # which sums in a different order than the reference's single full matmul;
    # the backward _CopyToTPRegion all-reduce has the same property. Both are
    # correct in fp32 but can differ at the ULP level, so the tolerance is
    # principled for fp32 reduce-matmul non-associativity, not a failed-equality
    # cover-up. (gloo CPU all_reduce of fp32 is otherwise deterministic.)
    assert torch.allclose(tp_logits, ref_logits, atol=1e-6, rtol=1e-5)

    shifted_cross_entropy(tp_logits, ids).backward()
    shifted_cross_entropy(ref_logits, ids).backward()
    rank = ctx.tensor_parallel_rank
    tp_sz = ctx.tensor_parallel_size
    for name, tp_p in tp.named_parameters():
        tp_g = tp_p.grad.detach()
        ref_g = {n: p.grad for n, p in ref.named_parameters()}[name]
        if tp_g.shape == ref_g.shape:
            # replicated param (embedding, lm_head, ln) — same shape, same grad
            assert torch.allclose(tp_g, ref_g, atol=1e-6, rtol=1e-5), f"grad mismatch on {name}"
        elif tp_g.shape[0] < ref_g.shape[0] and tp_g.shape[1:] == ref_g.shape[1:]:
            # column-sharded (output rows split)
            chunk = ref_g.shape[0] // tp_sz
            expected = ref_g[rank * chunk:(rank + 1) * chunk]
            assert torch.allclose(tp_g, expected, atol=1e-6, rtol=1e-5), f"grad mismatch on {name}"
        elif tp_g.shape[1] < ref_g.shape[1] and tp_g.shape[0] == ref_g.shape[0]:
            # row-sharded (input columns split)
            chunk = ref_g.shape[1] // tp_sz
            expected = ref_g[:, rank * chunk:(rank + 1) * chunk]
            assert torch.allclose(tp_g, expected, atol=1e-6, rtol=1e-5), f"grad mismatch on {name}"
        else:
            raise AssertionError(f"unexpected shape mismatch for {name}: tp {tp_g.shape} vs ref {ref_g.shape}")
    destroy_parallel()
