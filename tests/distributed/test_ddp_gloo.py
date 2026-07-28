"""Run with:
torchrun --standalone --nproc_per_node=2 --master_addr=127.0.0.1 \
  -m pytest tests/distributed/test_ddp_gloo.py -v -s --import-mode=importlib
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]


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
    """Full fused weight is [Q; K; V]; each rank holds [Q_r; K_r; V_r]."""
    return _expected_blockwise_grad_shard(ref_g, rank, tp_sz, (q_dim, kv_dim, kv_dim))


def _assert_tp_grads_match_ref(tp, ref, cfg, rank: int, tp_sz: int) -> None:
    for name, tp_p in tp.named_parameters():
        tp_g = tp_p.grad.detach()
        ref_g = {n: p.grad for n, p in ref.named_parameters()}[name]
        if "qkv_proj" in name and cfg.use_fused_qkv:
            q_dim = cfg.hidden_size
            num_kv = getattr(cfg, "num_query_groups", None) or cfg.num_heads
            kv_dim = num_kv * (cfg.hidden_size // cfg.num_heads)
            expected = _expected_fused_qkv_grad_shard(ref_g, rank, tp_sz, q_dim, kv_dim)
            assert torch.allclose(tp_g, expected, atol=1e-6, rtol=1e-5), (
                f"grad mismatch on {name}"
            )
        elif (
            "mlp.fc1" in name
            and cfg.gated_linear_unit
            and tp_g.shape[0] < ref_g.shape[0]
            and tp_g.shape[1:] == ref_g.shape[1:]
        ):
            ffn = ref_g.shape[0] // 2
            expected = _expected_blockwise_grad_shard(ref_g, rank, tp_sz, (ffn, ffn))
            assert torch.allclose(tp_g, expected, atol=1e-6, rtol=1e-5), (
                f"grad mismatch on {name}"
            )
        elif tp_g.shape == ref_g.shape:
            # replicated param (ln, pos_emb) — same shape, same grad
            assert torch.allclose(tp_g, ref_g, atol=1e-6, rtol=1e-5), (
                f"grad mismatch on {name}"
            )
        elif tp_g.shape[0] < ref_g.shape[0] and tp_g.shape[1:] == ref_g.shape[1:]:
            # column-sharded / vocab-sharded (output rows or vocab rows split)
            chunk = ref_g.shape[0] // tp_sz
            expected = ref_g[rank * chunk : (rank + 1) * chunk]
            assert torch.allclose(tp_g, expected, atol=1e-6, rtol=1e-5), (
                f"grad mismatch on {name}"
            )
        elif tp_g.shape[1] < ref_g.shape[1] and tp_g.shape[0] == ref_g.shape[0]:
            # row-sharded (input columns split)
            chunk = ref_g.shape[1] // tp_sz
            expected = ref_g[:, rank * chunk : (rank + 1) * chunk]
            assert torch.allclose(tp_g, expected, atol=1e-6, rtol=1e-5), (
                f"grad mismatch on {name}"
            )
        else:
            raise AssertionError(
                f"unexpected shape mismatch for {name}: tp {tp_g.shape} vs ref {ref_g.shape}"
            )




def _run_torchrun(nproc: int, test_id: str) -> None:
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
        f"tests/distributed/test_ddp_gloo.py::{test_id}",
        "-v",
        "-s",
        "--import-mode=importlib",
    ]
    env = os.environ.copy()
    env["NANO_MEGATRON_DDP_WORKER"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = master_port
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_DDP_WORKER") == "1", reason="launcher only"
)
def test_launch_dp2_grad_matches_reference():
    _run_torchrun(2, "test_worker_dp2_grad_matches_reference")


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_DDP_WORKER") != "1", reason="worker only"
)
def test_worker_dp2_grad_matches_reference():
    from nano_megatron.distributed import DistributedDataParallel
    from nano_megatron.parallel import (
        ParallelConfig,
        destroy_parallel,
        initialize_parallel,
        is_parallel_initialized,
    )

    if is_parallel_initialized():
        destroy_parallel()

    ctx = initialize_parallel(
        ParallelConfig(data_parallel_size=2), dist_backend="gloo"
    )
    assert ctx.data_parallel_size == 2

    torch.manual_seed(0)
    module = nn.Sequential(
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
    )
    ddp = DistributedDataParallel(module, ctx, bucket_cap_mb=25.0)

    # Same global batch on every rank; each rank takes a local shard.
    torch.manual_seed(123)
    x_global = torch.randn(4, 8)
    local_bs = x_global.shape[0] // ctx.data_parallel_size
    r = ctx.data_parallel_rank
    x_local = x_global[r * local_bs : (r + 1) * local_bs]

    # Reference: full-batch mean loss with weights after DDP broadcast.
    ref = nn.Sequential(
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
    )
    ref.load_state_dict(ddp.module.state_dict())

    loss_ref = ref(x_global).pow(2).mean()
    loss_ref.backward()

    loss = ddp(x_local).pow(2).mean()
    loss.backward()
    ddp.finish_grad_sync()

    for (name, p_ddp), p_ref in zip(
        ddp.module.named_parameters(), ref.parameters()
    ):
        assert p_ddp.grad is not None and p_ref.grad is not None, name
        assert torch.allclose(
            p_ddp.grad, p_ref.grad, atol=1e-6, rtol=1e-5
        ), f"grad mismatch on {name}"

    destroy_parallel()


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_DDP_WORKER") == "1", reason="launcher only"
)
def test_launch_tp2_dp2_grad_matches_reference():
    _run_torchrun(4, "test_worker_tp2_dp2_grad_matches_reference")


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_DDP_WORKER") != "1", reason="worker only"
)
def test_worker_tp2_dp2_grad_matches_reference():
    from nano_megatron.distributed import DistributedDataParallel
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
        vocab_size=64,
        max_seq_len=16,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        ffn_hidden_size=64,
        layernorm_eps=1e-5,
        use_bias=True,
        tie_word_embeddings=False,
    )
    ctx = initialize_parallel(
        ParallelConfig(tensor_parallel_size=2, data_parallel_size=2),
        dist_backend="gloo",
    )
    assert ctx.tensor_parallel_size == 2
    assert ctx.data_parallel_size == 2

    torch.manual_seed(0)
    ref = ReferenceGPT(cfg)
    tp = build_tp_gpt_from_reference(ref, ctx)
    ddp = DistributedDataParallel(tp, ctx, bucket_cap_mb=25.0)

    # Global batch [4, seq]; each DP rank takes 2 rows by data_parallel_rank.
    torch.manual_seed(1)
    seq = 8
    ids_global = torch.randint(0, cfg.vocab_size, (4, seq))
    local_bs = ids_global.shape[0] // ctx.data_parallel_size
    dp = ctx.data_parallel_rank
    ids_local = ids_global[dp * local_bs : (dp + 1) * local_bs]

    # Reference: full model on global batch (same seed weights as TP source).
    ref.zero_grad(set_to_none=True)
    ref_logits = ref(ids_global)
    ref_loss = shifted_cross_entropy(ref_logits, ids_global)
    ref_loss.backward()

    ddp.zero_grad(set_to_none=True)
    tp_logits = ddp(ids_local)
    tp_loss = ddp.module.shifted_cross_entropy(tp_logits, ids_local)
    tp_loss.backward()
    ddp.finish_grad_sync()

    _assert_tp_grads_match_ref(
        ddp.module,
        ref,
        cfg,
        ctx.tensor_parallel_rank,
        ctx.tensor_parallel_size,
    )
    destroy_parallel()

