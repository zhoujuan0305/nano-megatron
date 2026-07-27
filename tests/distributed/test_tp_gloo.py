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


def _all_gather_vocab_logits(local_logits: torch.Tensor, ctx) -> torch.Tensor:
    """Concatenate local vocab shards into full-vocab logits for comparison."""
    tp = ctx.tensor_parallel_size
    if tp == 1:
        return local_logits
    gathered = [torch.empty_like(local_logits) for _ in range(tp)]
    ctx.backend.all_gather(
        gathered, local_logits.contiguous(), group=ctx.tensor_parallel_group
    )
    return torch.cat(gathered, dim=-1)


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
    assert tp_logits.shape[-1] == cfg.vocab_size // ctx.tensor_parallel_size

    # Optional full-logits check via all_gather of local vocab shards.
    full_tp_logits = _all_gather_vocab_logits(tp_logits, ctx)
    assert torch.allclose(full_tp_logits, ref_logits, atol=1e-6, rtol=1e-5)

    # Loss path used in training: vocab-parallel CE vs reference shifted CE.
    tp_loss = tp.shifted_cross_entropy(tp_logits, ids)
    ref_loss = shifted_cross_entropy(ref_logits, ids)
    assert torch.allclose(tp_loss, ref_loss, atol=1e-6, rtol=1e-5)

    tp_loss.backward()
    ref_loss.backward()
    _assert_tp_grads_match_ref(
        tp, ref, cfg, ctx.tensor_parallel_rank, ctx.tensor_parallel_size
    )
    destroy_parallel()


@pytest.mark.skipif(os.environ.get("NANO_MEGATRON_TP_WORKER") == "1", reason="launcher only")
def test_launch_tp2_fused_qkv_forward_backward_gloo():
    _run_torchrun(2, "test_worker_tp2_fused_qkv_forward_backward_gloo")


@pytest.mark.skipif(os.environ.get("NANO_MEGATRON_TP_WORKER") != "1", reason="worker only")
def test_worker_tp2_fused_qkv_forward_backward_gloo():
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
        use_fused_qkv=True,
    )
    ctx = initialize_parallel(
        ParallelConfig(tensor_parallel_size=2), dist_backend="gloo"
    )
    assert ctx.tensor_parallel_size == 2
    torch.manual_seed(0)
    ref = ReferenceGPT(cfg)
    tp = build_tp_gpt_from_reference(ref, ctx)

    for block in ref.blocks:
        assert hasattr(block.attn, "qkv_proj")
        assert not hasattr(block.attn, "q_proj")
    for block in tp.blocks:
        assert hasattr(block.attn, "qkv_proj")
        assert not hasattr(block.attn, "q_proj")

    ids = torch.randint(0, 16, (2, 6))
    ref_logits = ref(ids)
    tp_logits = tp(ids)
    assert tp_logits.shape[-1] == cfg.vocab_size // ctx.tensor_parallel_size
    full_tp_logits = _all_gather_vocab_logits(tp_logits, ctx)
    assert torch.allclose(full_tp_logits, ref_logits, atol=1e-6, rtol=1e-5)

    tp_loss = tp.shifted_cross_entropy(tp_logits, ids)
    ref_loss = shifted_cross_entropy(ref_logits, ids)
    assert torch.allclose(tp_loss, ref_loss, atol=1e-6, rtol=1e-5)

    tp_loss.backward()
    ref_loss.backward()
    _assert_tp_grads_match_ref(
        tp, ref, cfg, ctx.tensor_parallel_rank, ctx.tensor_parallel_size
    )
    destroy_parallel()
