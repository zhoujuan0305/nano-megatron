"""CP flash ring path vs unfused AG-KV (NCCL, multi-GPU).

Run with:
torchrun --standalone --nproc_per_node=2 --master_addr=127.0.0.1 \\
  -m pytest tests/distributed/test_flash_cp_nccl.py -v -s --import-mode=importlib
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

# bf16 flash vs fp32-ish unfused: loose numerical tolerance
ATOL = 5e-2
RTOL = 5e-2

WORKER_ENV = "NANO_MEGATRON_FLASH_CP_NCCL_WORKER"


def _flash_attn_ok() -> bool:
    try:
        from nano_megatron.parallel.attention_backend import flash_attn_available

        return flash_attn_available()
    except Exception:
        return False


def _run_torchrun(nproc: int, test_id: str) -> None:
    require_nccl_gpus(nproc)
    if not _flash_attn_ok():
        pytest.skip("flash_attn not available")
    master_port = str(29900 + hash(test_id) % 1000)
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
        f"tests/distributed/test_flash_cp_nccl.py::{test_id}",
        "-v",
        "-s",
        "--import-mode=importlib",
    ]
    env = os.environ.copy()
    env[WORKER_ENV] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = master_port
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


def _small_cfg(*, attn_backend: str):
    from nano_megatron.reference import ReferenceGPTConfig

    # head_dim = 64/4 = 16 (flash-friendly multiple of 8)
    return ReferenceGPTConfig(
        vocab_size=32,
        max_seq_len=32,
        hidden_size=64,
        num_layers=1,
        num_heads=4,
        ffn_hidden_size=128,
        layernorm_eps=1e-5,
        use_bias=True,
        tie_word_embeddings=False,
        attn_backend=attn_backend,
        attention_dropout=0.0,
    )


def _gather_cp_seq(local: torch.Tensor, ctx) -> torch.Tensor:
    from nano_megatron.parallel import gather_from_context_parallel_region

    if ctx.context_parallel_size == 1:
        return local
    return gather_from_context_parallel_region(
        local.detach(),
        ctx.context_parallel_group,
        ctx.backend,
        ctx.context_parallel_rank,
        ctx.context_parallel_size,
        seq_dim=1,
        grad_op="split",
    )


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get(WORKER_ENV) == "1", reason="launcher only")
def test_launch_flash_cp2_vs_unfused_nccl():
    _run_torchrun(2, "test_worker_flash_cp2_vs_unfused_nccl")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get(WORKER_ENV) != "1", reason="worker only")
def test_worker_flash_cp2_vs_unfused_nccl():
    """cp=2 bf16: flash ring path matches unfused AG-KV on same weights/inputs."""
    from nano_megatron.model import build_tp_gpt_from_reference
    from nano_megatron.parallel import (
        ParallelConfig,
        destroy_parallel,
        initialize_parallel,
        is_parallel_initialized,
    )
    from nano_megatron.parallel.attention_backend import flash_attn_available
    from nano_megatron.reference import ReferenceGPT

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("need >=2 CUDA devices")
    if not flash_attn_available():
        pytest.skip("flash_attn not available")

    if is_parallel_initialized():
        destroy_parallel()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    ctx = initialize_parallel(
        ParallelConfig(context_parallel_size=2),
        dist_backend="nccl",
    )
    assert ctx.context_parallel_size == 2
    assert ctx.world_size == 2

    torch.manual_seed(0)
    cfg_unfused = _small_cfg(attn_backend="unfused")
    ref = ReferenceGPT(cfg_unfused).to(device=device, dtype=torch.bfloat16)

    # Same weights; only attn_backend differs.
    model_unfused = build_tp_gpt_from_reference(ref, ctx)
    model_unfused = model_unfused.to(device=device, dtype=torch.bfloat16)

    cfg_flash = _small_cfg(attn_backend="flash")
    # Rebuild from the same ref state so parameters match.
    ref_flash = ReferenceGPT(cfg_flash).to(device=device, dtype=torch.bfloat16)
    ref_flash.load_state_dict(ref.state_dict())
    model_flash = build_tp_gpt_from_reference(ref_flash, ctx)
    model_flash = model_flash.to(device=device, dtype=torch.bfloat16)

    torch.manual_seed(1)
    batch, seq = 2, 32
    ids = torch.randint(0, cfg_unfused.vocab_size, (batch, seq), device=device)

    model_unfused.eval()
    model_flash.eval()
    with torch.no_grad():
        logits_u = model_unfused(ids)
        logits_f = model_flash(ids)

    assert logits_u.shape == (batch, seq // 2, cfg_unfused.vocab_size)
    assert logits_f.shape == logits_u.shape

    full_u = _gather_cp_seq(logits_u, ctx)
    full_f = _gather_cp_seq(logits_f, ctx)
    assert torch.allclose(full_f.float(), full_u.float(), atol=ATOL, rtol=RTOL), (
        f"flash CP logits vs unfused: "
        f"max_abs={(full_f.float() - full_u.float()).abs().max().item()}"
    )

    # Forward + backward on attention-bearing loss (mean of logits).
    model_unfused.train()
    model_flash.train()
    model_unfused.zero_grad(set_to_none=True)
    model_flash.zero_grad(set_to_none=True)

    out_u = model_unfused(ids)
    out_f = model_flash(ids)
    loss_u = out_u.float().sum()
    loss_f = out_f.float().sum()
    loss_u.backward()
    loss_f.backward()

    for (n_u, p_u), (n_f, p_f) in zip(
        model_unfused.named_parameters(), model_flash.named_parameters()
    ):
        assert n_u == n_f
        assert p_u.grad is not None and p_f.grad is not None, f"missing grad on {n_u}"
        gu = p_u.grad.float()
        gf = p_f.grad.float()
        assert torch.allclose(gf, gu, atol=ATOL, rtol=RTOL), (
            f"grad mismatch on {n_u}: max_abs={(gf - gu).abs().max().item()}"
        )

    destroy_parallel()
