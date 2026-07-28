"""Run with:
torchrun --standalone --nproc_per_node=2 --master_addr=127.0.0.1 \
  -m pytest tests/distributed/test_cp_gloo.py -v -s --import-mode=importlib
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]

ATOL = 1e-6
RTOL = 1e-5


def _run_torchrun(nproc: int, test_id: str) -> None:
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
        f"tests/distributed/test_cp_gloo.py::{test_id}",
        "-v",
        "-s",
        "--import-mode=importlib",
    ]
    env = os.environ.copy()
    env["NANO_MEGATRON_CP_WORKER"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = master_port
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


def _small_cfg():
    from nano_megatron.reference import ReferenceGPTConfig

    return ReferenceGPTConfig(
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


def _gather_cp_seq_logits(local_logits: torch.Tensor, ctx) -> torch.Tensor:
    """All-gather CP sequence shards → full-seq local-vocab logits (no grad)."""
    from nano_megatron.parallel import gather_from_context_parallel_region

    if ctx.context_parallel_size == 1:
        return local_logits
    return gather_from_context_parallel_region(
        local_logits.detach(),
        ctx.context_parallel_group,
        ctx.backend,
        ctx.context_parallel_rank,
        ctx.context_parallel_size,
        seq_dim=1,
        grad_op="split",
    )


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


def _assert_tp_grads_match_ref(tp, ref, cfg, rank: int, tp_sz: int) -> None:
    for name, tp_p in tp.named_parameters():
        tp_g = tp_p.grad.detach()
        ref_g = {n: p.grad for n, p in ref.named_parameters()}[name]
        if "qkv_proj" in name and cfg.use_fused_qkv:
            q_dim = cfg.hidden_size
            num_kv = getattr(cfg, "num_query_groups", None) or cfg.num_heads
            kv_dim = num_kv * (cfg.hidden_size // cfg.num_heads)
            expected = _expected_fused_qkv_grad_shard(ref_g, rank, tp_sz, q_dim, kv_dim)
            assert torch.allclose(tp_g, expected, atol=ATOL, rtol=RTOL), (
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
            assert torch.allclose(tp_g, expected, atol=ATOL, rtol=RTOL), (
                f"grad mismatch on {name}"
            )
        elif tp_g.shape == ref_g.shape:
            assert torch.allclose(tp_g, ref_g, atol=ATOL, rtol=RTOL), (
                f"grad mismatch on {name}: "
                f"max_abs={(tp_g - ref_g).abs().max().item()}"
            )
        elif tp_g.shape[0] < ref_g.shape[0] and tp_g.shape[1:] == ref_g.shape[1:]:
            chunk = ref_g.shape[0] // tp_sz
            expected = ref_g[rank * chunk : (rank + 1) * chunk]
            assert torch.allclose(tp_g, expected, atol=ATOL, rtol=RTOL), (
                f"grad mismatch on {name}: "
                f"max_abs={(tp_g - expected).abs().max().item()}"
            )
        elif tp_g.shape[1] < ref_g.shape[1] and tp_g.shape[0] == ref_g.shape[0]:
            chunk = ref_g.shape[1] // tp_sz
            expected = ref_g[:, rank * chunk : (rank + 1) * chunk]
            assert torch.allclose(tp_g, expected, atol=ATOL, rtol=RTOL), (
                f"grad mismatch on {name}: "
                f"max_abs={(tp_g - expected).abs().max().item()}"
            )
        else:
            raise AssertionError(
                f"unexpected shape mismatch for {name}: "
                f"tp {tp_g.shape} vs ref {ref_g.shape}"
            )


def _assert_full_grads_match(model, ref) -> None:
    ref_grads = {n: p.grad for n, p in ref.named_parameters()}
    for name, p in model.named_parameters():
        assert p.grad is not None, f"missing grad on {name}"
        rg = ref_grads[name]
        assert p.grad.shape == rg.shape, (
            f"shape mismatch on {name}: {p.grad.shape} vs {rg.shape}"
        )
        assert torch.allclose(p.grad, rg, atol=ATOL, rtol=RTOL), (
            f"grad mismatch on {name}: "
            f"max_abs={(p.grad - rg).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# CP2 vs reference
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_CP_WORKER") == "1", reason="launcher only"
)
def test_launch_cp2_vs_reference_forward_backward_gloo():
    _run_torchrun(2, "test_worker_cp2_vs_reference_forward_backward_gloo")


def _cp_mean_loss(loss: torch.Tensor, ctx) -> torch.Tensor:
    """Mean scaled local-CE losses over the CP group → global mean CE value."""
    out = loss.detach().clone()
    if ctx.context_parallel_size > 1:
        ctx.backend.all_reduce(out, group=ctx.context_parallel_group, op="sum")
        out = out / ctx.context_parallel_size
    return out


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_CP_WORKER") != "1", reason="worker only"
)
def test_worker_cp2_vs_reference_forward_backward_gloo():
    """CP=2: local CE (no logits gather) matches ref loss/grads after DDP mean.

    Local CE scales ``local_sum * cp_size / global_valid`` so DDP mean over
    DP×CP recovers full-sequence mean-CE gradients.
    """
    from nano_megatron.distributed import DistributedDataParallel
    from nano_megatron.model import build_tp_gpt_from_reference
    from nano_megatron.parallel import (
        ParallelConfig,
        destroy_parallel,
        initialize_parallel,
        is_parallel_initialized,
    )
    from nano_megatron.reference import ReferenceGPT
    from nano_megatron.reference.loss import shifted_cross_entropy

    if is_parallel_initialized():
        destroy_parallel()

    cfg = _small_cfg()
    ctx = initialize_parallel(
        ParallelConfig(context_parallel_size=2),
        dist_backend="gloo",
    )
    assert ctx.context_parallel_size == 2
    assert ctx.data_parallel_size == 1
    assert ctx.tensor_parallel_size == 1

    torch.manual_seed(0)
    ref = ReferenceGPT(cfg)
    model = build_tp_gpt_from_reference(ref, ctx)
    ddp = DistributedDataParallel(model, ctx, bucket_cap_mb=25.0)

    torch.manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))

    # Forward: local seq logits; optional gather only for logits check.
    local_logits = ddp(ids)
    assert local_logits.shape == (2, 4, cfg.vocab_size)
    full_logits = _gather_cp_seq_logits(local_logits, ctx)
    assert full_logits.shape == (2, 8, cfg.vocab_size)

    ref_logits = ref(ids)
    assert torch.allclose(full_logits, ref_logits, atol=ATOL, rtol=RTOL), (
        f"logits mismatch: max_abs={(full_logits - ref_logits).abs().max().item()}"
    )

    # CE must not all-gather logits.
    def _gather_forbidden(*_a, **_k):
        raise AssertionError(
            "shifted_cross_entropy must not call gather_from_context_parallel_region"
        )

    import nano_megatron.model.tp_gpt as tp_gpt_mod

    orig_gather = tp_gpt_mod.gather_from_context_parallel_region
    tp_gpt_mod.gather_from_context_parallel_region = _gather_forbidden
    try:
        cp_loss = ddp.module.shifted_cross_entropy(local_logits, ids)
    finally:
        tp_gpt_mod.gather_from_context_parallel_region = orig_gather

    ref_loss = shifted_cross_entropy(ref_logits, ids)
    cp_loss_mean = _cp_mean_loss(cp_loss, ctx)
    assert torch.allclose(cp_loss_mean, ref_loss, atol=ATOL, rtol=RTOL), (
        f"loss mismatch: cp_mean={cp_loss_mean.item()} ref={ref_loss.item()}"
    )

    # Backward + DDP mean over CP → full grads.
    ref.zero_grad(set_to_none=True)
    ref_loss.backward()

    ddp.zero_grad(set_to_none=True)
    local_logits = ddp(ids)
    tp_gpt_mod.gather_from_context_parallel_region = _gather_forbidden
    try:
        cp_loss = ddp.module.shifted_cross_entropy(local_logits, ids)
    finally:
        tp_gpt_mod.gather_from_context_parallel_region = orig_gather
    cp_loss.backward()
    ddp.finish_grad_sync()

    _assert_full_grads_match(ddp.module, ref)
    destroy_parallel()


# ---------------------------------------------------------------------------
# CP2 × TP2 vs reference
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_CP_WORKER") == "1", reason="launcher only"
)
def test_launch_cp2_tp2_vs_reference_gloo():
    _run_torchrun(4, "test_worker_cp2_tp2_vs_reference_gloo")


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_CP_WORKER") != "1", reason="worker only"
)
def test_worker_cp2_tp2_vs_reference_gloo():
    """world=4, tp=2, cp=2: local CE + DDP mean grads vs ref."""
    from nano_megatron.distributed import DistributedDataParallel
    from nano_megatron.model import build_tp_gpt_from_reference
    from nano_megatron.parallel import (
        ParallelConfig,
        destroy_parallel,
        initialize_parallel,
        is_parallel_initialized,
    )
    from nano_megatron.reference import ReferenceGPT
    from nano_megatron.reference.loss import shifted_cross_entropy
    import nano_megatron.model.tp_gpt as tp_gpt_mod

    if is_parallel_initialized():
        destroy_parallel()

    cfg = _small_cfg()
    ctx = initialize_parallel(
        ParallelConfig(tensor_parallel_size=2, context_parallel_size=2),
        dist_backend="gloo",
    )
    assert ctx.tensor_parallel_size == 2
    assert ctx.context_parallel_size == 2
    assert ctx.data_parallel_size == 1
    assert ctx.world_size == 4

    torch.manual_seed(0)
    ref = ReferenceGPT(cfg)
    model = build_tp_gpt_from_reference(ref, ctx)
    ddp = DistributedDataParallel(model, ctx, bucket_cap_mb=25.0)

    torch.manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    local_vocab = cfg.vocab_size // ctx.tensor_parallel_size
    tp_rank = ctx.tensor_parallel_rank

    local_logits = ddp(ids)
    assert local_logits.shape == (2, 4, local_vocab)
    full_seq_logits = _gather_cp_seq_logits(local_logits, ctx)
    assert full_seq_logits.shape == (2, 8, local_vocab)

    ref_logits = ref(ids)
    expected = ref_logits[
        :, :, tp_rank * local_vocab : (tp_rank + 1) * local_vocab
    ]
    assert torch.allclose(full_seq_logits, expected, atol=ATOL, rtol=RTOL), (
        f"logits mismatch tp={tp_rank}: "
        f"max_abs={(full_seq_logits - expected).abs().max().item()}"
    )

    def _gather_forbidden(*_a, **_k):
        raise AssertionError(
            "shifted_cross_entropy must not call gather_from_context_parallel_region"
        )

    orig_gather = tp_gpt_mod.gather_from_context_parallel_region
    tp_gpt_mod.gather_from_context_parallel_region = _gather_forbidden
    try:
        cp_loss = ddp.module.shifted_cross_entropy(local_logits, ids)
    finally:
        tp_gpt_mod.gather_from_context_parallel_region = orig_gather

    ref_loss = shifted_cross_entropy(ref_logits, ids)
    cp_loss_mean = _cp_mean_loss(cp_loss, ctx)
    assert torch.allclose(cp_loss_mean, ref_loss, atol=ATOL, rtol=RTOL), (
        f"loss mismatch: cp_mean={cp_loss_mean.item()} ref={ref_loss.item()}"
    )

    ref.zero_grad(set_to_none=True)
    ref_loss.backward()

    ddp.zero_grad(set_to_none=True)
    local_logits = ddp(ids)
    tp_gpt_mod.gather_from_context_parallel_region = _gather_forbidden
    try:
        cp_loss = ddp.module.shifted_cross_entropy(local_logits, ids)
    finally:
        tp_gpt_mod.gather_from_context_parallel_region = orig_gather
    cp_loss.backward()
    ddp.finish_grad_sync()

    _assert_tp_grads_match_ref(
        ddp.module,
        ref,
        cfg,
        ctx.tensor_parallel_rank,
        ctx.tensor_parallel_size,
    )
    destroy_parallel()


# ---------------------------------------------------------------------------
# CP2 × DP2 with DDP
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_CP_WORKER") == "1", reason="launcher only"
)
def test_launch_cp2_dp2_ddp_gloo():
    _run_torchrun(4, "test_worker_cp2_dp2_ddp_gloo")


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_CP_WORKER") != "1", reason="worker only"
)
def test_worker_cp2_dp2_ddp_gloo():
    """world=4, cp=2, dp=2: same data within CP group; DDP mean over dp*cp.

    Group size = dp*cp = 4; mean_divisor = 4 (local CE *cp scale).
    Baseline: single-process global-batch mean CE grads.
    """
    from nano_megatron.distributed import DistributedDataParallel
    from nano_megatron.model import build_tp_gpt_from_reference
    from nano_megatron.parallel import (
        ParallelConfig,
        destroy_parallel,
        initialize_parallel,
        is_parallel_initialized,
    )
    from nano_megatron.reference import ReferenceGPT
    from nano_megatron.reference.loss import shifted_cross_entropy
    import nano_megatron.model.tp_gpt as tp_gpt_mod

    if is_parallel_initialized():
        destroy_parallel()

    cfg = _small_cfg()
    ctx = initialize_parallel(
        ParallelConfig(context_parallel_size=2, data_parallel_size=2),
        dist_backend="gloo",
    )
    assert ctx.context_parallel_size == 2
    assert ctx.data_parallel_size == 2
    assert ctx.tensor_parallel_size == 1
    assert ctx.world_size == 4

    torch.manual_seed(0)
    ref = ReferenceGPT(cfg)
    model = build_tp_gpt_from_reference(ref, ctx)
    ddp = DistributedDataParallel(model, ctx, bucket_cap_mb=25.0)

    # Global batch [4, S]; each DP rank takes 2 rows.  Both CP ranks in a DP
    # group see the same local microbatch (CP shards sequence internally).
    torch.manual_seed(1)
    seq = 8
    ids_global = torch.randint(0, cfg.vocab_size, (4, seq))
    local_bs = ids_global.shape[0] // ctx.data_parallel_size
    dp = ctx.data_parallel_rank
    ids_local = ids_global[dp * local_bs : (dp + 1) * local_bs]

    # Reference: full model on global batch (mean over all tokens).
    ref.zero_grad(set_to_none=True)
    ref_logits = ref(ids_global)
    ref_loss = shifted_cross_entropy(ref_logits, ids_global)
    ref_loss.backward()

    ddp.zero_grad(set_to_none=True)
    local_logits = ddp(ids_local)
    assert local_logits.shape == (local_bs, seq // ctx.context_parallel_size, cfg.vocab_size)

    def _gather_forbidden(*_a, **_k):
        raise AssertionError(
            "shifted_cross_entropy must not call gather_from_context_parallel_region"
        )

    orig_gather = tp_gpt_mod.gather_from_context_parallel_region
    tp_gpt_mod.gather_from_context_parallel_region = _gather_forbidden
    try:
        cp_loss = ddp.module.shifted_cross_entropy(local_logits, ids_local)
    finally:
        tp_gpt_mod.gather_from_context_parallel_region = orig_gather
    cp_loss.backward()
    ddp.finish_grad_sync()

    _assert_full_grads_match(ddp.module, ref)
    destroy_parallel()
