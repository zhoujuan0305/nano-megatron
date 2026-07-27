"""Run with:
torchrun --standalone --nproc_per_node=2 --master_addr=127.0.0.1 \
  -m pytest tests/distributed/test_sp_nccl.py -v -s --import-mode=importlib
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
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
        f"tests/distributed/test_sp_nccl.py::{test_id}",
        "-v",
        "-s",
        "--import-mode=importlib",
    ]
    env = os.environ.copy()
    env["NANO_MEGATRON_SP_NCCL_WORKER"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = master_port
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


def _assert_sp_grads_match_tp(tp_on, tp_off) -> None:
    """Both models are TP-sharded the same way; compare grads by name."""
    off_grads = {n: p.grad for n, p in tp_off.named_parameters()}
    for name, p_on in tp_on.named_parameters():
        g_on = p_on.grad
        g_off = off_grads[name]
        assert g_on is not None, f"missing grad on SP model: {name}"
        assert g_off is not None, f"missing grad on non-SP TP model: {name}"
        assert g_on.shape == g_off.shape, (
            f"grad shape mismatch on {name}: SP {g_on.shape} vs TP {g_off.shape}"
        )
        assert torch.allclose(g_on, g_off, atol=1e-6, rtol=1e-5), (
            f"grad mismatch on {name}: "
            f"max_abs={(g_on - g_off).abs().max().item()}"
        )


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_SP_NCCL_WORKER") == "1", reason="launcher only"
)
def test_launch_sp_vs_tp_forward_backward_nccl():
    _run_torchrun(2, "test_worker_sp_vs_tp_forward_backward_nccl")


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_SP_NCCL_WORKER") != "1", reason="worker only"
)
def test_worker_sp_vs_tp_forward_backward_nccl():
    """SP=True TPGPT must match SP=False TPGPT on local vocab logits and grads (NCCL).

    Process groups are initialized once with sequence_parallel=False.
    The SP model is built from dataclasses.replace(ctx, sequence_parallel=True)
    so both models share the same TP groups/ranks.
    """
    from nano_megatron.model import build_tp_gpt_from_reference
    from nano_megatron.parallel import (
        ParallelConfig,
        destroy_parallel,
        initialize_parallel,
        is_parallel_initialized,
    )
    from nano_megatron.reference import ReferenceGPT, ReferenceGPTConfig

    if is_parallel_initialized():
        destroy_parallel()

    # S must be divisible by tp_size when SP is enabled (use 8, not 6).
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
        ParallelConfig(tensor_parallel_size=2, sequence_parallel=False),
        dist_backend="nccl",
    )
    assert ctx.tensor_parallel_size == 2
    assert ctx.sequence_parallel is False

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    ref = ReferenceGPT(cfg)
    tp_off = build_tp_gpt_from_reference(ref, ctx).cuda()

    # Same process groups; only the SP flag differs for linear/scatter wiring.
    ctx_sp = replace(ctx, sequence_parallel=True)
    tp_on = build_tp_gpt_from_reference(ref, ctx_sp).cuda()
    assert tp_on._sequence_parallel is True
    assert tp_off._sequence_parallel is False

    torch.manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (2, 8), device="cuda")

    logits_on = tp_on(ids)
    logits_off = tp_off(ids)
    local_vocab = cfg.vocab_size // ctx.tensor_parallel_size
    assert logits_on.shape == (2, 8, local_vocab)
    assert logits_off.shape == (2, 8, local_vocab)
    assert torch.allclose(logits_on, logits_off, atol=1e-6, rtol=1e-5), (
        f"logits mismatch: max_abs={(logits_on - logits_off).abs().max().item()}"
    )

    # Realistic training path: vocab-parallel shifted CE backward.
    loss_on = tp_on.shifted_cross_entropy(logits_on, ids)
    loss_off = tp_off.shifted_cross_entropy(logits_off, ids)
    assert torch.allclose(loss_on, loss_off, atol=1e-6, rtol=1e-5), (
        f"loss mismatch: on={loss_on.item()} off={loss_off.item()}"
    )

    loss_on.backward()
    loss_off.backward()
    _assert_sp_grads_match_tp(tp_on, tp_off)

    destroy_parallel()


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_SP_NCCL_WORKER") == "1", reason="launcher only"
)
def test_launch_sp_vs_tp_fused_qkv_forward_backward_nccl():
    _run_torchrun(2, "test_worker_sp_vs_tp_fused_qkv_forward_backward_nccl")


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_SP_NCCL_WORKER") != "1", reason="worker only"
)
def test_worker_sp_vs_tp_fused_qkv_forward_backward_nccl():
    """Same SP vs non-SP check with fused QKV path (NCCL)."""
    from nano_megatron.model import build_tp_gpt_from_reference
    from nano_megatron.parallel import (
        ParallelConfig,
        destroy_parallel,
        initialize_parallel,
        is_parallel_initialized,
    )
    from nano_megatron.reference import ReferenceGPT, ReferenceGPTConfig

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
        use_fused_qkv=True,
    )
    ctx = initialize_parallel(
        ParallelConfig(tensor_parallel_size=2, sequence_parallel=False),
        dist_backend="nccl",
    )
    assert ctx.tensor_parallel_size == 2

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    ref = ReferenceGPT(cfg)
    tp_off = build_tp_gpt_from_reference(ref, ctx).cuda()
    ctx_sp = replace(ctx, sequence_parallel=True)
    tp_on = build_tp_gpt_from_reference(ref, ctx_sp).cuda()

    for block in tp_on.blocks:
        assert hasattr(block.attn, "qkv_proj")
        assert not hasattr(block.attn, "q_proj")
        assert block.attn.qkv_proj.sequence_parallel is True
        assert block.attn.out_proj.sequence_parallel is True

    torch.manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (2, 8), device="cuda")

    logits_on = tp_on(ids)
    logits_off = tp_off(ids)
    assert torch.allclose(logits_on, logits_off, atol=1e-6, rtol=1e-5), (
        f"logits mismatch: max_abs={(logits_on - logits_off).abs().max().item()}"
    )

    loss_on = tp_on.shifted_cross_entropy(logits_on, ids)
    loss_off = tp_off.shifted_cross_entropy(logits_off, ids)
    assert torch.allclose(loss_on, loss_off, atol=1e-6, rtol=1e-5)

    loss_on.backward()
    loss_off.backward()
    _assert_sp_grads_match_tp(tp_on, tp_off)

    destroy_parallel()
