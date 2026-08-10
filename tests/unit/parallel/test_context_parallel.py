from __future__ import annotations

import pytest
import torch

from nano_megatron.parallel import (
    ParallelConfig,
    destroy_parallel,
    initialize_parallel,
    is_parallel_initialized,
)
from nano_megatron.parallel.context_parallel import (
    causal_attn_scores_cp,
    gather_from_context_parallel_region,
    local_sequence_range,
    scatter_to_context_parallel_region,
    unpermute_ag_to_global,
    zigzag_ag_to_global_index,
    zigzag_global_to_ag_index,
    zigzag_half_ag_starts,
    zigzag_half_ag_token_start,
    zigzag_half_ids,
    zigzag_local_token_indices,
)
from nano_megatron.reference.layers import causal_attn_scores


def _init_cp1(monkeypatch, port: str):
    """Initialize parallel context with cp_size=1 (world_size=1, gloo)."""
    import torch.distributed as dist

    if is_parallel_initialized():
        destroy_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", port)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    return initialize_parallel(ParallelConfig(), dist_backend="gloo")


# ---------------------------------------------------------------------------
# local_sequence_range
# ---------------------------------------------------------------------------


def test_local_sequence_range():
    assert local_sequence_range(0, 2, 8) == (0, 4)
    assert local_sequence_range(1, 2, 8) == (4, 8)


def test_local_sequence_range_cp1():
    assert local_sequence_range(0, 1, 8) == (0, 8)


def test_local_sequence_range_four_way():
    assert local_sequence_range(0, 4, 16) == (0, 4)
    assert local_sequence_range(1, 4, 16) == (4, 8)
    assert local_sequence_range(2, 4, 16) == (8, 12)
    assert local_sequence_range(3, 4, 16) == (12, 16)


def test_local_sequence_range_nondivisible():
    with pytest.raises(ValueError, match="not divisible"):
        local_sequence_range(0, 3, 8)


# ---------------------------------------------------------------------------
# scatter / gather identity for cp_size=1
# ---------------------------------------------------------------------------


def test_scatter_gather_identity_cp1(monkeypatch):
    ctx = _init_cp1(monkeypatch, "29800")
    x = torch.randn(2, 8, 4, requires_grad=True)
    y = scatter_to_context_parallel_region(
        x, ctx.context_parallel_group, ctx.backend, 0, 1
    )
    assert torch.equal(y, x)
    z = gather_from_context_parallel_region(
        y, ctx.context_parallel_group, ctx.backend, 0, 1
    )
    assert torch.equal(z, x)
    z.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


# ---------------------------------------------------------------------------
# scatter / gather with custom seq_dim (e.g. dim=2 for [B, H, S, D])
# ---------------------------------------------------------------------------


def test_scatter_gather_seq_dim_2_cp1(monkeypatch):
    ctx = _init_cp1(monkeypatch, "29801")
    x = torch.randn(1, 2, 8, 4, requires_grad=True)  # [B, H, S, D]
    y = scatter_to_context_parallel_region(
        x, ctx.context_parallel_group, ctx.backend, 0, 1, seq_dim=2
    )
    assert torch.equal(y, x)
    z = gather_from_context_parallel_region(
        y, ctx.context_parallel_group, ctx.backend, 0, 1, seq_dim=2
    )
    assert torch.equal(z, x)
    z.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


# ---------------------------------------------------------------------------
# causal_attn_scores_cp matches full causal scores
# ---------------------------------------------------------------------------


def test_causal_attn_scores_cp_matches_full_slice():
    B, H, S, D = 1, 2, 8, 4
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    full = causal_attn_scores(q, k, scale=0.5)
    for start in (0, 4):
        q_local = q[:, :, start : start + 4, :]
        got = causal_attn_scores_cp(q_local, k, scale=0.5, query_start=start)
        assert torch.allclose(got, full[:, :, start : start + 4, :], atol=0, rtol=0)


# ---------------------------------------------------------------------------
# scatter forward shard math (documents narrow behavior)
# ---------------------------------------------------------------------------


def test_scatter_forward_shard_math():
    """Document that scatter narrows to the expected shard for each cp_rank."""
    x = torch.arange(16, dtype=torch.float32).view(1, 8, 2)
    # cp_size=2 splits seq dim (8) into two chunks of 4
    shard_0 = x[:, 0:4, :]
    shard_1 = x[:, 4:8, :]

    # Verify local_sequence_range produces matching ranges
    assert local_sequence_range(0, 2, 8) == (0, 4)
    assert local_sequence_range(1, 2, 8) == (4, 8)

    # Verify narrow matches expected shards
    chunk = 8 // 2
    assert torch.equal(x.narrow(1, 0 * chunk, chunk), shard_0)
    assert torch.equal(x.narrow(1, 1 * chunk, chunk), shard_1)


# ---------------------------------------------------------------------------
# scatter backward: pad-zeros (reverse of narrow), no collective
# ---------------------------------------------------------------------------


def test_scatter_backward_pad_zeros_math():
    """Scatter backward pads zeros — reverse of narrow, not all-gather.

    Each CP rank embeds the full sequence then narrows.  dL/d(full_embed)
    on rank r is zeros everywhere except the local shard, which holds
    grad_output.  All-gather would incorrectly place peer shards into
    every rank's embed gradient.
    """
    cp_size = 2
    seq_dim = 1
    # Local shard grad on each rank: distinct values.
    grad_rank0 = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    grad_rank1 = torch.arange(100, 108, dtype=torch.float32).view(1, 4, 2)
    full_seq = 8
    chunk = full_seq // cp_size

    for cp_rank, grad_output in ((0, grad_rank0), (1, grad_rank1)):
        full_shape = list(grad_output.shape)
        full_shape[seq_dim] = full_seq
        grad_input = grad_output.new_zeros(full_shape)
        grad_input.narrow(seq_dim, cp_rank * chunk, chunk).copy_(grad_output)
        # Only local shard is non-zero.
        assert torch.equal(
            grad_input.narrow(seq_dim, cp_rank * chunk, chunk), grad_output
        )
        other = 1 - cp_rank
        assert torch.equal(
            grad_input.narrow(seq_dim, other * chunk, chunk),
            torch.zeros_like(grad_output),
        )
        # Contrast: all-gather would cat both shards on every rank (wrong).
        all_gather_wrong = torch.cat([grad_rank0, grad_rank1], dim=seq_dim)
        assert not torch.equal(grad_input, all_gather_wrong)


def test_scatter_backward_pad_zeros_autograd(monkeypatch):
    """Autograd path: scatter backward returns zero-padded full grad."""
    from nano_megatron.parallel.context_parallel import (
        _ScatterToContextParallelRegion,
    )

    ctx = _init_cp1(monkeypatch, "29805")
    # Simulate cp_size=2 rank 0 via direct Function (world is still 1).
    # Forward still needs a real full tensor; we call backward math via apply
    # with cp_size=1 for identity, and unit-test pad path via manual backward
    # of the Function with a fake ctx-equivalent by running apply at cp=1
    # only for wiring — pad-zeros for cp>1 is covered by calling backward
    # through a thin wrapper that sets cp_size without collectives.
    x = torch.arange(16, dtype=torch.float32).view(1, 8, 2).requires_grad_(True)
    # Use Function.forward/backward directly with a stub backend that would
    # fail if all_gather were called.
    class _NoCollectiveBackend:
        def all_gather(self, *args, **kwargs):
            raise AssertionError("scatter backward must not all_gather")

        def all_reduce(self, *args, **kwargs):
            raise AssertionError("scatter backward must not all_reduce")

    backend = _NoCollectiveBackend()
    # rank 1 of cp_size=2: local shard is x[:, 4:8, :]
    y = _ScatterToContextParallelRegion.apply(
        x, ctx.context_parallel_group, backend, 1, 2, 1, "contiguous"
    )
    assert y.shape == (1, 4, 2)
    assert torch.equal(y, x.detach()[:, 4:8, :])
    y.sum().backward()
    expected = torch.zeros_like(x)
    expected[:, 4:8, :] = 1.0
    assert torch.equal(x.grad, expected)


# ---------------------------------------------------------------------------
# gather backward identity for cp_size=1 (both grad_op modes)
# ---------------------------------------------------------------------------


def test_gather_backward_identity_cp1_reduce_scatter(monkeypatch):
    """Gather backward with cp_size=1 returns grad_output unchanged."""
    ctx = _init_cp1(monkeypatch, "29802")
    x = torch.randn(2, 8, 4, requires_grad=True)
    y = gather_from_context_parallel_region(
        x, ctx.context_parallel_group, ctx.backend, 0, 1, grad_op="reduce_scatter"
    )
    assert torch.equal(y, x)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_gather_backward_identity_cp1_split(monkeypatch):
    """Gather backward split mode with cp_size=1 is also identity."""
    ctx = _init_cp1(monkeypatch, "29803")
    x = torch.randn(2, 8, 4, requires_grad=True)
    y = gather_from_context_parallel_region(
        x, ctx.context_parallel_group, ctx.backend, 0, 1, grad_op="split"
    )
    assert torch.equal(y, x)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_gather_invalid_grad_op(monkeypatch):
    ctx = _init_cp1(monkeypatch, "29804")
    x = torch.randn(2, 8, 4)
    with pytest.raises(ValueError, match="grad_op"):
        gather_from_context_parallel_region(
            x, ctx.context_parallel_group, ctx.backend, 0, 1, grad_op="mean"
        )


# ---------------------------------------------------------------------------
# gather backward math: reduce_scatter vs split (documents the contract)
# ---------------------------------------------------------------------------


def test_gather_backward_reduce_scatter_math():
    """Document reduce-scatter backward for KV-style partial contributions.

    With cp_size=2, each rank holds a different partial grad_output over the
    full sequence.  reduce-scatter sums the chunks: rank i receives
    sum_r(grad_r[:, shard_i, :]).

    Single-process math: if both ranks contribute the same grad_output G,
    rank 0's local grad is 2 * G[:, 0:4, :] (sum of two identical chunks).
    """
    cp_size = 2
    grad_output = torch.ones(1, 8, 2) * 3.0
    chunks = [c.contiguous() for c in grad_output.chunk(cp_size, dim=1)]
    # Two ranks each contribute the same full G → sum on each shard is 2x.
    rank0_local = chunks[0] + chunks[0]
    rank1_local = chunks[1] + chunks[1]
    assert torch.equal(rank0_local, torch.ones(1, 4, 2) * 6.0)
    assert torch.equal(rank1_local, torch.ones(1, 4, 2) * 6.0)


def test_gather_backward_split_math():
    """Document split backward for identical full-sequence consumers (loss).

    Every CP rank computes the same global-mean CE, so dL/dlogits_full is
    identical on all ranks.  Backward must *narrow* to the local shard —
    not sum — otherwise grads are scaled by cp_size.

    With cp_size=2 and identical G on both ranks:
      split:   rank0 gets G[:, 0:4, :], rank1 gets G[:, 4:8, :]
      reduce_scatter would give 2 * those shards (wrong for loss).
    """
    cp_size = 2
    # Distinct values so shards are distinguishable.
    grad_output = torch.arange(16, dtype=torch.float32).view(1, 8, 2)
    chunk = 8 // cp_size
    for cp_rank in (0, 1):
        split_local = grad_output.narrow(1, cp_rank * chunk, chunk).contiguous()
        assert split_local.shape == (1, 4, 2)
        assert torch.equal(
            split_local, grad_output[:, cp_rank * chunk : (cp_rank + 1) * chunk, :]
        )
    # Contrast: reduce-scatter of two identical G would double each shard.
    chunks = list(grad_output.chunk(cp_size, dim=1))
    rs_rank0 = chunks[0] + chunks[0]
    assert torch.equal(rs_rank0, 2.0 * grad_output[:, 0:4, :])
    assert not torch.equal(rs_rank0, grad_output[:, 0:4, :])


# ---------------------------------------------------------------------------
# local_sequence_range cp_rank validation
# ---------------------------------------------------------------------------


def test_local_sequence_range_invalid_cp_rank():
    """cp_rank must be in [0, cp_size)."""
    with pytest.raises(ValueError, match="cp_rank"):
        local_sequence_range(-1, 2, 8)
    with pytest.raises(ValueError, match="cp_rank"):
        local_sequence_range(2, 2, 8)
    # cp_size < 1 is a separate check
    with pytest.raises(ValueError, match="cp_size must be >= 1"):
        local_sequence_range(0, 0, 8)


# ---------------------------------------------------------------------------
# gather forward: single-buffer all_gather_into_tensor (no list+cat)
# ---------------------------------------------------------------------------


class _RecordingGatherBackend:
    """Mock backend: records gather API usage and fills output for cp_size>1."""

    def __init__(self, shards: list[torch.Tensor]):
        self.shards = shards
        self.into_calls: list[dict] = []
        self.list_all_gather_calls: list[dict] = []

    def all_gather_into_tensor(self, output, input, *, group=None):
        self.into_calls.append(
            {"output": output, "input": input, "group": group}
        )
        # Simulate multi-rank gather into a single buffer along dim 0.
        # Avoid torch.cat so tests can assert the production path never cats.
        offset = 0
        for shard in self.shards:
            n = shard.size(0)
            output[offset : offset + n].copy_(shard)
            offset += n
        if offset != output.size(0):
            raise AssertionError(
                f"mock filled {offset} rows but output has {output.size(0)}"
            )
        return output

    def all_gather(self, tensor_list, tensor, *, group=None):
        self.list_all_gather_calls.append(
            {"tensor_list": tensor_list, "tensor": tensor, "group": group}
        )
        raise AssertionError(
            "gather forward must use all_gather_into_tensor, not list all_gather"
        )

    def reduce_scatter(self, output, input_list, *, group=None, op="sum"):
        raise AssertionError("not used in forward gather test")


def test_gather_forward_uses_all_gather_into_tensor_no_cat(monkeypatch):
    """cp_size>1 gather must use single-buffer all_gather_into_tensor, not list+cat."""
    from nano_megatron.parallel.context_parallel import (
        _GatherFromContextParallelRegion,
    )

    cp_size = 2
    seq_dim = 1
    # Local shards as they would appear on each rank before gather (seq on dim 1).
    shard0 = torch.arange(0, 8, dtype=torch.float32).view(1, 4, 2)
    shard1 = torch.arange(100, 108, dtype=torch.float32).view(1, 4, 2)
    # After movedim(seq_dim→0), shards are [S_loc, B, ...] for mock fill along dim 0.
    shards_dim0 = [s.movedim(seq_dim, 0).contiguous() for s in (shard0, shard1)]
    backend = _RecordingGatherBackend(shards_dim0)

    cat_calls: list[tuple] = []
    real_cat = torch.cat

    def tracking_cat(*args, **kwargs):
        cat_calls.append((args, kwargs))
        return real_cat(*args, **kwargs)

    monkeypatch.setattr(torch, "cat", tracking_cat)

    expected = real_cat([shard0, shard1], dim=seq_dim)

    # Rank 0 local input
    x = shard0.clone()
    out = _GatherFromContextParallelRegion.apply(
        x, None, backend, 0, cp_size, seq_dim, "reduce_scatter", "contiguous"
    )

    assert len(backend.list_all_gather_calls) == 0, (
        "list all_gather must not be used on gather forward path"
    )
    assert len(backend.into_calls) == 1, (
        f"expected one all_gather_into_tensor call, got {len(backend.into_calls)}"
    )
    call = backend.into_calls[0]
    # Output buffer first dim is cp_size * local_seq
    assert call["output"].shape[0] == cp_size * shard0.size(seq_dim)
    assert call["input"].shape[0] == shard0.size(seq_dim)
    assert call["input"].is_contiguous()

    assert torch.equal(out, expected)
    assert cat_calls == [], (
        f"torch.cat must not be used on gather forward path, calls={cat_calls!r}"
    )


def test_gather_forward_into_tensor_seq_dim_2(monkeypatch):
    """Single-buffer gather works when sequence is on dim 2 ([B, H, S, D])."""
    from nano_megatron.parallel.context_parallel import (
        _GatherFromContextParallelRegion,
    )

    cp_size = 2
    seq_dim = 2
    shard0 = torch.arange(0, 16, dtype=torch.float32).view(1, 2, 4, 2)
    shard1 = torch.arange(200, 216, dtype=torch.float32).view(1, 2, 4, 2)
    shards_dim0 = [s.movedim(seq_dim, 0).contiguous() for s in (shard0, shard1)]
    backend = _RecordingGatherBackend(shards_dim0)

    real_cat = torch.cat
    cat_calls: list = []

    def tracking_cat(*args, **kwargs):
        cat_calls.append(True)
        return real_cat(*args, **kwargs)

    monkeypatch.setattr(torch, "cat", tracking_cat)

    expected = real_cat([shard0, shard1], dim=seq_dim)
    out = _GatherFromContextParallelRegion.apply(
        shard0.clone(), None, backend, 0, cp_size, seq_dim, "split", "contiguous"
    )
    assert len(backend.into_calls) == 1
    assert len(backend.list_all_gather_calls) == 0
    assert torch.equal(out, expected)
    assert cat_calls == []


# ---------------------------------------------------------------------------
# zigzag_half_ids
# ---------------------------------------------------------------------------


def test_zigzag_half_ids_cp2():
    assert zigzag_half_ids(0, 2) == (0, 3)
    assert zigzag_half_ids(1, 2) == (1, 2)


def test_zigzag_half_ids_cp4():
    # rank r owns halves (r, 2*cp-1-r)
    assert zigzag_half_ids(0, 4) == (0, 7)
    assert zigzag_half_ids(1, 4) == (1, 6)
    assert zigzag_half_ids(2, 4) == (2, 5)
    assert zigzag_half_ids(3, 4) == (3, 4)


def test_zigzag_half_ids_symmetry():
    """Each pair (r, 2*cp-1-r) should be the mirror of each other."""
    cp_size = 4
    for r in range(cp_size):
        first, second = zigzag_half_ids(r, cp_size)
        assert first + second == 2 * cp_size - 1


def test_zigzag_half_ids_invalid():
    with pytest.raises(ValueError):
        zigzag_half_ids(-1, 2)
    with pytest.raises(ValueError):
        zigzag_half_ids(2, 2)
    with pytest.raises(ValueError):
        zigzag_half_ids(0, 0)


# ---------------------------------------------------------------------------
# zigzag_local_token_indices — CP2 S=8
# ---------------------------------------------------------------------------


def test_zigzag_local_token_indices_cp2_s8():
    # S=8 → H=2; rank0 [0,1,6,7]; rank1 [2,3,4,5]
    assert zigzag_local_token_indices(0, 2, 8).tolist() == [0, 1, 6, 7]
    assert zigzag_local_token_indices(1, 2, 8).tolist() == [2, 3, 4, 5]


def test_zigzag_local_token_indices_cp4_s16():
    # S=16, cp=4 → H=2.  Half-chunks: 0:[0,1] 1:[2,3] 2:[4,5] 3:[6,7]
    #                    4:[8,9] 5:[10,11] 6:[12,13] 7:[14,15]
    # rank0 halves (0,7): [0,1,14,15]
    # rank1 halves (1,6): [2,3,12,13]
    # rank2 halves (2,5): [4,5,10,11]
    # rank3 halves (3,4): [6,7,8,9]
    assert zigzag_local_token_indices(0, 4, 16).tolist() == [0, 1, 14, 15]
    assert zigzag_local_token_indices(1, 4, 16).tolist() == [2, 3, 12, 13]
    assert zigzag_local_token_indices(2, 4, 16).tolist() == [4, 5, 10, 11]
    assert zigzag_local_token_indices(3, 4, 16).tolist() == [6, 7, 8, 9]


def test_zigzag_local_token_indices_nondivisible():
    """S % (2*cp) != 0 must raise."""
    with pytest.raises(ValueError, match=r"2 \* cp_size|not divisible"):
        zigzag_local_token_indices(0, 2, 7)
    with pytest.raises(ValueError):
        zigzag_local_token_indices(0, 4, 14)


def test_zigzag_local_token_indices_invalid_rank():
    with pytest.raises(ValueError):
        zigzag_local_token_indices(-1, 2, 8)
    with pytest.raises(ValueError):
        zigzag_local_token_indices(2, 2, 8)


def test_zigzag_local_token_indices_dtype_device():
    t = zigzag_local_token_indices(0, 2, 8, device="cpu", dtype=torch.int32)
    assert t.dtype == torch.int32
    assert t.device.type == "cpu"


def test_zigzag_local_token_indices_covers_all():
    """Every global index [0, S) appears exactly once across all ranks."""
    cp_size, S = 4, 16
    all_ids = []
    for r in range(cp_size):
        all_ids.extend(zigzag_local_token_indices(r, cp_size, S).tolist())
    assert sorted(all_ids) == list(range(S))


# ---------------------------------------------------------------------------
# zigzag_half_ag_token_start / zigzag_half_ag_starts / unpermute_ag_to_global
# ---------------------------------------------------------------------------


def test_zigzag_half_ag_token_start_cp2_s8():
    """CP2 S=8 H=2: AG = [0,1,6,7 | 2,3,4,5]; half starts at AG offsets."""
    cp, H = 2, 2
    # half0@0, half1@4, half2@6, half3@2
    assert zigzag_half_ag_token_start(0, cp, H) == 0
    assert zigzag_half_ag_token_start(1, cp, H) == 4
    assert zigzag_half_ag_token_start(2, cp, H) == 6
    assert zigzag_half_ag_token_start(3, cp, H) == 2
    assert zigzag_half_ag_starts(cp, H) == [0, 4, 6, 2]


def test_zigzag_half_ag_token_start_cp4_s16():
    """CP4 S=16 H=2: AG rank-concat of zigzag locals."""
    cp, S, H = 4, 16, 2
    starts = zigzag_half_ag_starts(cp, H)
    assert len(starts) == 2 * cp
    # Build AG from local indices; each half slice must match global half.
    locals_ = [zigzag_local_token_indices(r, cp, S) for r in range(cp)]
    ag = torch.cat(locals_)
    for g in range(2 * cp):
        start = zigzag_half_ag_token_start(g, cp, H)
        assert start == starts[g]
        assert ag[start : start + H].tolist() == list(range(g * H, (g + 1) * H))


def test_zigzag_half_ag_token_start_matches_ag_to_global_perm():
    """starts[g] equals perm inverse: first AG index of global half g."""
    for cp, H in ((2, 2), (4, 2), (4, 3), (1, 4)):
        perm = zigzag_ag_to_global_index(cp, H)  # global[i] = ag[perm[i]]
        starts = zigzag_half_ag_starts(cp, H)
        for g in range(2 * cp):
            # Global tokens g*H .. (g+1)*H-1 come from AG at perm[g*H + t]
            expected = int(perm[g * H].item())
            assert starts[g] == expected
            for t in range(H):
                assert int(perm[g * H + t].item()) == expected + t


def test_zigzag_half_ag_token_start_invalid():
    with pytest.raises(ValueError):
        zigzag_half_ag_token_start(-1, 2, 2)
    with pytest.raises(ValueError):
        zigzag_half_ag_token_start(4, 2, 2)
    with pytest.raises(ValueError):
        zigzag_half_ag_token_start(0, 2, 0)
    with pytest.raises(ValueError):
        zigzag_half_ag_token_start(0, 0, 2)


def test_unpermute_ag_to_global_cp2_s8():
    cp, S = 2, 8
    H = S // (2 * cp)
    locals_ = [zigzag_local_token_indices(r, cp, S) for r in range(cp)]
    ag = torch.cat(locals_).view(1, S).float()
    # Also as [B, S, D]
    ag_bd = ag.unsqueeze(-1).expand(1, S, 2).contiguous()
    out = unpermute_ag_to_global(ag_bd, cp, seq_dim=1)
    assert out[0, :, 0].tolist() == list(range(S))
    # Roundtrip property via starts
    for g in range(2 * cp):
        start = zigzag_half_ag_token_start(g, cp, H)
        assert ag_bd[0, start : start + H, 0].tolist() == list(
            range(g * H, (g + 1) * H)
        )


def test_unpermute_ag_to_global_cp4_and_seq_dim():
    cp, S = 4, 16
    locals_ = [zigzag_local_token_indices(r, cp, S) for r in range(cp)]
    ag = torch.cat(locals_).float()
    # seq on dim 2: [1, 2, S]
    x = ag.view(1, 1, S).expand(1, 2, S).contiguous()
    out = unpermute_ag_to_global(x, cp, seq_dim=2)
    assert out[0, 0].tolist() == list(range(S))


def test_unpermute_ag_to_global_cp1_identity():
    x = torch.arange(8.0).view(1, 4, 2)
    assert torch.equal(unpermute_ag_to_global(x, 1, seq_dim=1), x)


# ---------------------------------------------------------------------------
# zigzag_ag_to_global_index / zigzag_global_to_ag_index roundtrip
# ---------------------------------------------------------------------------


def test_zigzag_ag_to_global_roundtrip_cp2():
    cp, S = 2, 8
    H = S // (2 * cp)
    locals = [zigzag_local_token_indices(r, cp, S) for r in range(cp)]
    ag = torch.cat(locals)  # rank order
    perm = zigzag_ag_to_global_index(cp, H)
    assert ag[perm].tolist() == list(range(S))
    inv = zigzag_global_to_ag_index(cp, H)
    assert torch.equal(ag, torch.arange(S)[inv])


def test_zigzag_ag_to_global_roundtrip_cp4():
    cp, S = 4, 16
    H = S // (2 * cp)
    locals = [zigzag_local_token_indices(r, cp, S) for r in range(cp)]
    ag = torch.cat(locals)
    perm = zigzag_ag_to_global_index(cp, H)
    assert ag[perm].tolist() == list(range(S))
    inv = zigzag_global_to_ag_index(cp, H)
    assert torch.equal(ag, torch.arange(S)[inv])


def test_zigzag_ag_to_global_shape():
    cp, H = 4, 3
    perm = zigzag_ag_to_global_index(cp, H)
    assert perm.shape == (cp * 2 * H,)
    inv = zigzag_global_to_ag_index(cp, H)
    assert inv.shape == (cp * 2 * H,)


def test_zigzag_ag_to_global_is_permutation():
    """The output must be a valid permutation (unique values, full range)."""
    cp, H = 4, 3
    perm = zigzag_ag_to_global_index(cp, H)
    n = cp * 2 * H
    assert perm.unique().numel() == n
    assert perm.min() == 0
    assert perm.max() == n - 1


def test_zigzag_ag_to_global_roundtrip_cp1():
    """cp=1 degenerates: rank owns everything, AG layout = global order."""
    cp, S = 1, 4
    H = S // (2 * cp)
    loc = zigzag_local_token_indices(0, cp, S)
    assert loc.tolist() == list(range(S))
    perm = zigzag_ag_to_global_index(cp, H)
    assert perm.tolist() == list(range(S))
    inv = zigzag_global_to_ag_index(cp, H)
    assert inv.tolist() == list(range(S))


# ---------------------------------------------------------------------------
# pack mode: validation + zigzag scatter/gather
# ---------------------------------------------------------------------------


class _NoCollectiveBackend:
    """Backend that fails if any collective is invoked."""

    def all_gather(self, *args, **kwargs):
        raise AssertionError("unexpected all_gather")

    def all_gather_into_tensor(self, *args, **kwargs):
        raise AssertionError("unexpected all_gather_into_tensor")

    def all_reduce(self, *args, **kwargs):
        raise AssertionError("unexpected all_reduce")

    def reduce_scatter(self, *args, **kwargs):
        raise AssertionError("unexpected reduce_scatter")


class _RecordingRSBackend(_RecordingGatherBackend):
    """Extends gather mock with reduce_scatter recording."""

    def __init__(self, shards: list[torch.Tensor]):
        super().__init__(shards)
        self.rs_calls: list[dict] = []

    def reduce_scatter(self, output, input_list, *, group=None, op="sum"):
        self.rs_calls.append(
            {"output": output, "input_list": input_list, "group": group, "op": op}
        )
        # Sum corresponding chunks (simulate multi-rank contribution of one rank's view).
        acc = input_list[0].clone()
        for t in input_list[1:]:
            acc = acc + t
        output.copy_(acc)
        return output


def test_scatter_invalid_pack(monkeypatch):
    ctx = _init_cp1(monkeypatch, "29810")
    x = torch.randn(2, 8, 4)
    with pytest.raises(ValueError, match="pack"):
        scatter_to_context_parallel_region(
            x, ctx.context_parallel_group, ctx.backend, 0, 1, pack="round_robin"
        )


def test_gather_invalid_pack(monkeypatch):
    ctx = _init_cp1(monkeypatch, "29811")
    x = torch.randn(2, 8, 4)
    with pytest.raises(ValueError, match="pack"):
        gather_from_context_parallel_region(
            x, ctx.context_parallel_group, ctx.backend, 0, 1, pack="round_robin"
        )


def test_scatter_gather_zigzag_identity_cp1(monkeypatch):
    """cp_size=1 + pack=zigzag is identity (same as contiguous)."""
    ctx = _init_cp1(monkeypatch, "29812")
    x = torch.randn(2, 8, 4, requires_grad=True)
    y = scatter_to_context_parallel_region(
        x, ctx.context_parallel_group, ctx.backend, 0, 1, pack="zigzag"
    )
    assert torch.equal(y, x)
    z = gather_from_context_parallel_region(
        y, ctx.context_parallel_group, ctx.backend, 0, 1, pack="zigzag"
    )
    assert torch.equal(z, x)
    z.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_scatter_zigzag_forward_backward_no_collective():
    """Zigzag scatter uses index_select; backward pads zeros at zigzag indices.

    No collective is required — dummy group/backend are fine for cp_size>1.
    """
    backend = _NoCollectiveBackend()
    x = torch.arange(16.0).view(1, 8, 2).requires_grad_(True)
    # rank 0 of cp2: local [0,1,6,7]
    y = scatter_to_context_parallel_region(
        x, None, backend, 0, 2, seq_dim=1, pack="zigzag"
    )
    expected_y = x.detach()[:, [0, 1, 6, 7], :]
    assert torch.equal(y, expected_y)
    y.sum().backward()
    expected_grad = torch.zeros_like(x)
    expected_grad[:, [0, 1, 6, 7], :] = 1.0
    assert torch.equal(x.grad, expected_grad)


def test_scatter_zigzag_forward_backward_rank1():
    """Zigzag scatter rank 1 of cp2 owns [2,3,4,5]."""
    backend = _NoCollectiveBackend()
    x = torch.arange(16.0).view(1, 8, 2).requires_grad_(True)
    y = scatter_to_context_parallel_region(
        x, None, backend, 1, 2, seq_dim=1, pack="zigzag"
    )
    assert torch.equal(y, x.detach()[:, [2, 3, 4, 5], :])
    (y * torch.arange(1, 9, dtype=y.dtype).view(1, 4, 2)).sum().backward()
    expected_grad = torch.zeros_like(x)
    expected_grad[:, [2, 3, 4, 5], :] = torch.arange(1, 9, dtype=x.dtype).view(1, 4, 2)
    assert torch.equal(x.grad, expected_grad)


def test_scatter_zigzag_seq_dim_2():
    """Zigzag scatter works when sequence is on dim 2."""
    backend = _NoCollectiveBackend()
    x = torch.arange(32.0).view(1, 2, 8, 2).requires_grad_(True)
    y = scatter_to_context_parallel_region(
        x, None, backend, 0, 2, seq_dim=2, pack="zigzag"
    )
    assert y.shape == (1, 2, 4, 2)
    assert torch.equal(y, x.detach()[:, :, [0, 1, 6, 7], :])
    y.sum().backward()
    expected = torch.zeros_like(x)
    expected[:, :, [0, 1, 6, 7], :] = 1.0
    assert torch.equal(x.grad, expected)


def test_scatter_zigzag_math_matches_helper():
    """Document: zigzag scatter shard equals index_select with helper indices."""
    x = torch.arange(32.0).view(1, 16, 2)
    for rank in range(4):
        idx = zigzag_local_token_indices(rank, 4, 16)
        expected = x.index_select(1, idx)
        y = scatter_to_context_parallel_region(
            x, None, _NoCollectiveBackend(), rank, 4, seq_dim=1, pack="zigzag"
        )
        assert torch.equal(y, expected)


def test_scatter_contiguous_default_unchanged(monkeypatch):
    """Default pack=contiguous keeps narrow behavior (bit-identical path)."""
    from nano_megatron.parallel.context_parallel import (
        _ScatterToContextParallelRegion,
    )

    backend = _NoCollectiveBackend()
    x = torch.arange(16.0).view(1, 8, 2).requires_grad_(True)
    # Public API default
    y = scatter_to_context_parallel_region(x, None, backend, 1, 2, seq_dim=1)
    assert torch.equal(y, x.detach()[:, 4:8, :])
    y.sum().backward()
    expected = torch.zeros_like(x)
    expected[:, 4:8, :] = 1.0
    assert torch.equal(x.grad, expected)

    # Direct Function still accepts pack for internal callers
    x2 = torch.arange(16.0).view(1, 8, 2).requires_grad_(True)
    y2 = _ScatterToContextParallelRegion.apply(
        x2, None, backend, 0, 2, 1, "contiguous"
    )
    assert torch.equal(y2, x2.detach()[:, 0:4, :])


def test_gather_zigzag_forward_keeps_ag_layout():
    """Zigzag gather: AG rank-concat only (no full-sequence unpermute)."""
    cp_size = 2
    seq_dim = 1
    S = 8
    # Local zigzag shards
    full = torch.arange(S * 2, dtype=torch.float32).view(1, S, 2)
    shard0 = full.index_select(1, zigzag_local_token_indices(0, cp_size, S))
    shard1 = full.index_select(1, zigzag_local_token_indices(1, cp_size, S))
    # AG layout = rank-concat of locals
    ag = torch.cat([shard0, shard1], dim=seq_dim)
    shards_dim0 = [s.movedim(seq_dim, 0).contiguous() for s in (shard0, shard1)]
    backend = _RecordingGatherBackend(shards_dim0)

    out = gather_from_context_parallel_region(
        shard0.clone(), None, backend, 0, cp_size, seq_dim=seq_dim, pack="zigzag"
    )
    assert len(backend.into_calls) == 1
    assert torch.equal(out, ag)
    # Optional helper restores global order for tests / unfused.
    assert torch.equal(unpermute_ag_to_global(out, cp_size, seq_dim=seq_dim), full)


def test_scatter_gather_zigzag_roundtrip_via_unpermute():
    """scatter zig → gather AG → unpermute_ag_to_global → original."""
    cp_size = 2
    seq_dim = 1
    S = 8
    full = torch.arange(S * 2, dtype=torch.float32).view(1, S, 2)
    locals_ = [
        scatter_to_context_parallel_region(
            full, None, _NoCollectiveBackend(), r, cp_size, seq_dim=seq_dim, pack="zigzag"
        )
        for r in range(cp_size)
    ]
    shards_dim0 = [s.movedim(seq_dim, 0).contiguous() for s in locals_]
    backend = _RecordingGatherBackend(shards_dim0)
    ag = gather_from_context_parallel_region(
        locals_[0].clone(),
        None,
        backend,
        0,
        cp_size,
        seq_dim=seq_dim,
        pack="zigzag",
    )
    restored = unpermute_ag_to_global(ag, cp_size, seq_dim=seq_dim)
    assert torch.equal(restored, full)


def test_gather_zigzag_backward_split_ag_chunk():
    """Zigzag gather bwd split: local AG chunk (same as contiguous)."""
    cp_size = 2
    seq_dim = 1
    S = 8
    full = torch.arange(S * 2, dtype=torch.float32).view(1, S, 2)
    shard0 = full.index_select(1, zigzag_local_token_indices(0, cp_size, S))
    shard1 = full.index_select(1, zigzag_local_token_indices(1, cp_size, S))
    ag = torch.cat([shard0, shard1], dim=seq_dim)
    shards_dim0 = [s.movedim(seq_dim, 0).contiguous() for s in (shard0, shard1)]

    for rank, local in ((0, shard0), (1, shard1)):
        backend = _RecordingGatherBackend(shards_dim0)
        x = local.clone().requires_grad_(True)
        y = gather_from_context_parallel_region(
            x,
            None,
            backend,
            rank,
            cp_size,
            seq_dim=seq_dim,
            grad_op="split",
            pack="zigzag",
        )
        assert torch.equal(y.detach(), ag)
        # Upstream grad in AG layout (matches forward output layout).
        grad_ag = torch.arange(100, 100 + S * 2, dtype=torch.float32).view(1, S, 2)
        y.backward(grad_ag)
        chunk = S // cp_size
        expected_local = grad_ag.narrow(seq_dim, rank * chunk, chunk).contiguous()
        assert torch.equal(x.grad, expected_local)


def test_gather_zigzag_backward_reduce_scatter_no_inv_perm():
    """Zigzag gather bwd RS: chunk AG grad directly (no inv-permute)."""
    cp_size = 2
    seq_dim = 1
    S = 8
    full = torch.arange(S * 2, dtype=torch.float32).view(1, S, 2)
    shard0 = full.index_select(1, zigzag_local_token_indices(0, cp_size, S))
    shard1 = full.index_select(1, zigzag_local_token_indices(1, cp_size, S))
    shards_dim0 = [s.movedim(seq_dim, 0).contiguous() for s in (shard0, shard1)]
    backend = _RecordingRSBackend(shards_dim0)

    x = shard0.clone().requires_grad_(True)
    y = gather_from_context_parallel_region(
        x,
        None,
        backend,
        0,
        cp_size,
        seq_dim=seq_dim,
        grad_op="reduce_scatter",
        pack="zigzag",
    )
    # Grad matches AG forward layout.
    grad_ag = torch.arange(100, 100 + S * 2, dtype=torch.float32).view(1, S, 2)
    y.backward(grad_ag)

    assert len(backend.rs_calls) == 1
    call = backend.rs_calls[0]
    expected_chunks = [c.contiguous() for c in grad_ag.chunk(cp_size, dim=seq_dim)]
    assert len(call["input_list"]) == cp_size
    for got, exp in zip(call["input_list"], expected_chunks):
        assert torch.equal(got, exp)
    expected_local = sum(expected_chunks)
    assert torch.equal(x.grad, expected_local)


def test_gather_contiguous_pack_default_still_ag_order(monkeypatch):
    """Default pack=contiguous gather keeps AG rank-concat order (no unpermute)."""
    cp_size = 2
    seq_dim = 1
    shard0 = torch.arange(0, 8, dtype=torch.float32).view(1, 4, 2)
    shard1 = torch.arange(100, 108, dtype=torch.float32).view(1, 4, 2)
    shards_dim0 = [s.movedim(seq_dim, 0).contiguous() for s in (shard0, shard1)]
    backend = _RecordingGatherBackend(shards_dim0)
    out = gather_from_context_parallel_region(
        shard0.clone(), None, backend, 0, cp_size, seq_dim=seq_dim
    )
    expected = torch.cat([shard0, shard1], dim=seq_dim)
    assert torch.equal(out, expected)
