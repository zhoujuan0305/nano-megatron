"""Run with:
torchrun --standalone --nproc_per_node=2 --master_addr=127.0.0.1 \
  -m pytest tests/distributed/test_pp_nccl.py -v -s --import-mode=importlib
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

# NCCL float32 collectives are slightly looser than gloo CPU path.
_ATOL = 1e-5
_RTOL = 1e-4


def _run_torchrun(nproc: int, test_id: str) -> None:
    require_nccl_gpus(nproc)
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
        f"tests/distributed/test_pp_nccl.py::{test_id}",
        "-v",
        "-s",
        "--import-mode=importlib",
    ]
    env = os.environ.copy()
    env["NANO_MEGATRON_PP_NCCL_WORKER"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = master_port
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    subprocess.run(cmd, check=True, cwd=REPO, env=env, timeout=180)


def _tiny_cfg(*, num_layers: int = 4):
    from nano_megatron.reference import ReferenceGPTConfig

    return ReferenceGPTConfig(
        vocab_size=64,
        max_seq_len=8,
        hidden_size=32,
        num_layers=num_layers,
        num_heads=4,
        ffn_hidden_size=64,
        layernorm_eps=1e-5,
        use_bias=True,
        tie_word_embeddings=False,
    )


def _stage_to_ref_name(name: str, layer_start: int) -> str:
    if name.startswith("blocks."):
        parts = name.split(".", 2)
        local_idx = int(parts[1])
        rest = parts[2] if len(parts) > 2 else ""
        global_idx = layer_start + local_idx
        if rest:
            return f"blocks.{global_idx}.{rest}"
        return f"blocks.{global_idx}"
    return name


def _assert_stage_grads_match_ref(stage, ref, *, layer_start: int) -> None:
    ref_grads = {n: p.grad for n, p in ref.named_parameters()}
    for name, stage_p in stage.named_parameters():
        assert stage_p.grad is not None, f"missing grad on stage param {name}"
        ref_name = _stage_to_ref_name(name, layer_start)
        assert ref_name in ref_grads, f"{ref_name} not in reference grads (from {name})"
        ref_g = ref_grads[ref_name]
        assert ref_g is not None, f"reference grad is None for {ref_name}"
        stage_g = stage_p.grad.detach()
        assert stage_g.shape == ref_g.shape, (
            f"shape mismatch {name} -> {ref_name}: {stage_g.shape} vs {ref_g.shape}"
        )
        assert torch.allclose(stage_g, ref_g, atol=_ATOL, rtol=_RTOL), (
            f"grad mismatch on {name} -> {ref_name}: "
            f"max_abs={(stage_g - ref_g).abs().max().item()}"
        )


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_PP_NCCL_WORKER") == "1", reason="launcher only"
)
def test_launch_pp2_matches_reference_nccl():
    _run_torchrun(2, "test_worker_pp2_matches_reference_nccl")


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_PP_NCCL_WORKER") != "1", reason="worker only"
)
def test_worker_pp2_matches_reference_nccl():
    from nano_megatron.model import build_pipeline_stage_from_reference
    from nano_megatron.parallel import (
        ParallelConfig,
        destroy_parallel,
        initialize_parallel,
        is_parallel_initialized,
        is_pipeline_last_stage,
    )
    from nano_megatron.reference import ReferenceGPT
    from nano_megatron.reference.loss import shifted_cross_entropy
    from nano_megatron.schedules import forward_backward_1f1b

    if is_parallel_initialized():
        destroy_parallel()

    cfg = _tiny_cfg(num_layers=4)
    try:
        ctx = initialize_parallel(
            ParallelConfig(pipeline_parallel_size=2),
            dist_backend="nccl",
        )
        assert ctx.pipeline_parallel_size == 2
        device = torch.device(f"cuda:{ctx.local_rank}")

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        ref = ReferenceGPT(cfg).to(device)
        stage = build_pipeline_stage_from_reference(ref, ctx).to(device)

        torch.manual_seed(1)
        ids = torch.randint(0, cfg.vocab_size, (4, 8), device=device)
        labels = ids.clone()

        ref.zero_grad(set_to_none=True)
        ref_logits = ref(ids)
        ref_loss = shifted_cross_entropy(ref_logits, labels)
        ref_loss.backward()

        stage.zero_grad(set_to_none=True)
        sched_loss = forward_backward_1f1b(
            stage=stage,
            ctx=ctx,
            input_ids=ids,
            labels=labels,
            num_microbatches=2,
            ddp=None,
        )

        if is_pipeline_last_stage(ctx):
            assert sched_loss is not None
            assert torch.allclose(sched_loss, ref_loss, atol=_ATOL, rtol=_RTOL), (
                f"loss mismatch: sched={sched_loss.item()} ref={ref_loss.item()}"
            )
        else:
            assert sched_loss is None

        layers_per_stage = cfg.num_layers // ctx.pipeline_parallel_size
        layer_start = ctx.pipeline_parallel_rank * layers_per_stage
        _assert_stage_grads_match_ref(stage, ref, layer_start=layer_start)
    finally:
        if is_parallel_initialized():
            destroy_parallel()
