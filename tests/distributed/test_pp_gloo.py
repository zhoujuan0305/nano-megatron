"""Run with:
torchrun --standalone --nproc_per_node=2 --master_addr=127.0.0.1 \
  -m pytest tests/distributed/test_pp_gloo.py -v -s --import-mode=importlib
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]

_ATOL = 1e-6
_RTOL = 1e-5


def _skip_if_cannot_world(nproc: int) -> None:
    """Skip large multi-proc launchers when the env caps gloo world size."""
    max_world = os.environ.get("NANO_MEGATRON_MAX_GLOO_WORLD")
    if max_world is not None and nproc > int(max_world):
        pytest.skip(
            f"need world_size={nproc} but NANO_MEGATRON_MAX_GLOO_WORLD={max_world}"
        )


def _run_torchrun(nproc: int, test_id: str, *, timeout: int = 120) -> None:
    _skip_if_cannot_world(nproc)
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
        f"tests/distributed/test_pp_gloo.py::{test_id}",
        "-v",
        "-s",
        "--import-mode=importlib",
    ]
    env = os.environ.copy()
    env["NANO_MEGATRON_PP_WORKER"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = master_port
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    subprocess.run(cmd, check=True, cwd=REPO, env=env, timeout=timeout)


def _tiny_cfg(*, num_layers: int = 4) -> "ReferenceGPTConfig":
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
    """Map local stage parameter name to full-model reference name."""
    if name.startswith("blocks."):
        parts = name.split(".", 2)
        local_idx = int(parts[1])
        rest = parts[2] if len(parts) > 2 else ""
        global_idx = layer_start + local_idx
        if rest:
            return f"blocks.{global_idx}.{rest}"
        return f"blocks.{global_idx}"
    return name


def _expected_blockwise_grad_shard(
    ref_g: torch.Tensor, rank: int, tp_sz: int, block_sizes: tuple[int, ...]
) -> torch.Tensor:
    parts = ref_g.split(list(block_sizes), dim=0)
    return torch.cat(
        [
            part[rank * (dim // tp_sz) : (rank + 1) * (dim // tp_sz)]
            for part, dim in zip(parts, block_sizes)
        ],
        dim=0,
    )


def _expected_fused_qkv_grad_shard(
    ref_g: torch.Tensor, rank: int, tp_sz: int, q_dim: int, kv_dim: int
) -> torch.Tensor:
    return _expected_blockwise_grad_shard(ref_g, rank, tp_sz, (q_dim, kv_dim, kv_dim))


def _assert_stage_grads_match_ref(
    stage,
    ref,
    cfg,
    *,
    layer_start: int,
    tp_rank: int,
    tp_size: int,
) -> None:
    ref_grads = {n: p.grad for n, p in ref.named_parameters()}
    for name, stage_p in stage.named_parameters():
        assert stage_p.grad is not None, f"missing grad on stage param {name}"
        ref_name = _stage_to_ref_name(name, layer_start)
        assert ref_name in ref_grads, f"{ref_name} not in reference grads (from {name})"
        ref_g = ref_grads[ref_name]
        assert ref_g is not None, f"reference grad is None for {ref_name}"
        stage_g = stage_p.grad.detach()

        if "qkv_proj" in name and cfg.use_fused_qkv:
            q_dim = cfg.hidden_size
            num_kv = getattr(cfg, "num_query_groups", None) or cfg.num_heads
            kv_dim = num_kv * (cfg.hidden_size // cfg.num_heads)
            expected = _expected_fused_qkv_grad_shard(
                ref_g, tp_rank, tp_size, q_dim, kv_dim
            )
            assert torch.allclose(stage_g, expected, atol=_ATOL, rtol=_RTOL), (
                f"grad mismatch on {name} -> {ref_name}"
            )
        elif (
            "mlp.fc1" in name
            and cfg.gated_linear_unit
            and stage_g.shape[0] < ref_g.shape[0]
            and stage_g.shape[1:] == ref_g.shape[1:]
        ):
            ffn = ref_g.shape[0] // 2
            expected = _expected_blockwise_grad_shard(
                ref_g, tp_rank, tp_size, (ffn, ffn)
            )
            assert torch.allclose(stage_g, expected, atol=_ATOL, rtol=_RTOL), (
                f"grad mismatch on {name} -> {ref_name}"
            )
        elif stage_g.shape == ref_g.shape:
            assert torch.allclose(stage_g, ref_g, atol=_ATOL, rtol=_RTOL), (
                f"grad mismatch on {name} -> {ref_name}: "
                f"max_abs={(stage_g - ref_g).abs().max().item()}"
            )
        elif stage_g.shape[0] < ref_g.shape[0] and stage_g.shape[1:] == ref_g.shape[1:]:
            chunk = ref_g.shape[0] // tp_size
            expected = ref_g[tp_rank * chunk : (tp_rank + 1) * chunk]
            assert torch.allclose(stage_g, expected, atol=_ATOL, rtol=_RTOL), (
                f"grad mismatch on {name} -> {ref_name}"
            )
        elif stage_g.shape[1] < ref_g.shape[1] and stage_g.shape[0] == ref_g.shape[0]:
            chunk = ref_g.shape[1] // tp_size
            expected = ref_g[:, tp_rank * chunk : (tp_rank + 1) * chunk]
            assert torch.allclose(stage_g, expected, atol=_ATOL, rtol=_RTOL), (
                f"grad mismatch on {name} -> {ref_name}"
            )
        else:
            raise AssertionError(
                f"unexpected shape mismatch for {name} -> {ref_name}: "
                f"stage {stage_g.shape} vs ref {ref_g.shape}"
            )


def _run_pp_vs_reference(
    *,
    tp: int,
    dp: int,
    pp: int,
    num_layers: int,
    num_microbatches: int,
    batch_per_dp: int,
    seq_len: int,
    seed: int = 0,
    use_ddp: bool = False,
) -> None:
    from nano_megatron.distributed import DistributedDataParallel
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

    cfg = _tiny_cfg(num_layers=num_layers)
    ctx = None
    try:
        ctx = initialize_parallel(
            ParallelConfig(
                tensor_parallel_size=tp,
                data_parallel_size=dp,
                pipeline_parallel_size=pp,
            ),
            dist_backend="gloo",
        )
        assert ctx.tensor_parallel_size == tp
        assert ctx.data_parallel_size == dp
        assert ctx.pipeline_parallel_size == pp

        torch.manual_seed(seed)
        ref = ReferenceGPT(cfg)
        stage = build_pipeline_stage_from_reference(ref, ctx)

        ddp = None
        if use_ddp:
            ddp = DistributedDataParallel(stage, ctx, bucket_cap_mb=25.0)
            stage_mod = ddp.module
        else:
            stage_mod = stage

        # Global batch across DP; each DP rank takes a contiguous shard.
        torch.manual_seed(seed + 1)
        global_bs = batch_per_dp * dp
        ids_global = torch.randint(0, cfg.vocab_size, (global_bs, seq_len))
        labels_global = ids_global.clone()
        local_bs = batch_per_dp
        dp_rank = ctx.data_parallel_rank
        ids_local = ids_global[dp_rank * local_bs : (dp_rank + 1) * local_bs]
        labels_local = labels_global[dp_rank * local_bs : (dp_rank + 1) * local_bs]
        assert ids_local.size(0) % num_microbatches == 0

        # Oracle: full reference on the global batch.
        ref.zero_grad(set_to_none=True)
        ref_logits = ref(ids_global)
        ref_loss = shifted_cross_entropy(ref_logits, labels_global)
        ref_loss.backward()

        if ddp is not None:
            ddp.zero_grad(set_to_none=True)
        else:
            stage_mod.zero_grad(set_to_none=True)

        sched_loss = forward_backward_1f1b(
            stage=stage_mod,
            ctx=ctx,
            input_ids=ids_local,
            labels=labels_local,
            num_microbatches=num_microbatches,
            ddp=ddp,
        )

        is_last = is_pipeline_last_stage(ctx)
        if is_last:
            assert sched_loss is not None, "last stage must return loss"
            if dp == 1:
                assert torch.allclose(
                    sched_loss, ref_loss, atol=_ATOL, rtol=_RTOL
                ), (
                    f"loss mismatch: sched={sched_loss.item()} ref={ref_loss.item()}"
                )
            else:
                # Equal local batches: mean of per-DP local losses == global mean.
                loss_avg = sched_loss.detach().clone()
                ctx.backend.all_reduce(
                    loss_avg, group=ctx.data_parallel_group, op="sum"
                )
                loss_avg = loss_avg / ctx.data_parallel_size
                assert torch.allclose(
                    loss_avg, ref_loss, atol=_ATOL, rtol=_RTOL
                ), (
                    f"DP-averaged loss mismatch: avg={loss_avg.item()} "
                    f"ref={ref_loss.item()}"
                )
        else:
            assert sched_loss is None, "non-last stage must return None"

        layers_per_stage = num_layers // pp
        layer_start = ctx.pipeline_parallel_rank * layers_per_stage
        _assert_stage_grads_match_ref(
            stage_mod,
            ref,
            cfg,
            layer_start=layer_start,
            tp_rank=ctx.tensor_parallel_rank,
            tp_size=ctx.tensor_parallel_size,
        )
    finally:
        if is_parallel_initialized():
            destroy_parallel()


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_PP_WORKER") == "1", reason="launcher only"
)
def test_launch_pp2_matches_reference():
    _run_torchrun(2, "test_worker_pp2_matches_reference")


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_PP_WORKER") == "1", reason="launcher only"
)
def test_launch_tp2_pp2_matches_reference():
    _run_torchrun(4, "test_worker_tp2_pp2_matches_reference")


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_PP_WORKER") == "1", reason="launcher only"
)
def test_launch_dp2_pp2_matches_reference():
    _run_torchrun(4, "test_worker_dp2_pp2_matches_reference")


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_PP_WORKER") == "1", reason="launcher only"
)
def test_launch_tp2_dp2_pp2_matches_reference():
    # world=8 gloo CPU; skip via NANO_MEGATRON_MAX_GLOO_WORLD if needed.
    _run_torchrun(8, "test_worker_tp2_dp2_pp2_matches_reference", timeout=240)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_PP_WORKER") != "1", reason="worker only"
)
def test_worker_pp2_matches_reference():
    _run_pp_vs_reference(
        tp=1,
        dp=1,
        pp=2,
        num_layers=4,
        num_microbatches=2,
        batch_per_dp=4,
        seq_len=8,
        seed=0,
        use_ddp=False,
    )


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_PP_WORKER") != "1", reason="worker only"
)
def test_worker_tp2_pp2_matches_reference():
    _run_pp_vs_reference(
        tp=2,
        dp=1,
        pp=2,
        num_layers=4,
        num_microbatches=2,
        batch_per_dp=4,
        seq_len=8,
        seed=1,
        use_ddp=False,
    )


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_PP_WORKER") != "1", reason="worker only"
)
def test_worker_dp2_pp2_matches_reference():
    _run_pp_vs_reference(
        tp=1,
        dp=2,
        pp=2,
        num_layers=4,
        num_microbatches=2,
        batch_per_dp=4,
        seq_len=8,
        seed=2,
        use_ddp=True,
    )


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_PP_WORKER") != "1", reason="worker only"
)
def test_worker_tp2_dp2_pp2_matches_reference():
    world = int(os.environ.get("WORLD_SIZE", "0"))
    if world < 8:
        pytest.skip(f"need world_size>=8 for TP2×DP2×PP2, got {world}")
    _run_pp_vs_reference(
        tp=2,
        dp=2,
        pp=2,
        num_layers=4,
        num_microbatches=2,
        batch_per_dp=4,
        seq_len=8,
        seed=3,
        use_ddp=True,
    )
