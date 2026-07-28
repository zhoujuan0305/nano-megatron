from __future__ import annotations

from types import SimpleNamespace

from nano_megatron.parallel.context import (
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    pipeline_next_rank,
    pipeline_prev_rank,
)
from nano_megatron.parallel.rank_generator import RankGenerator


def _ctx_from_parts(tp, dp, pp, cp, parts, rank):
    return SimpleNamespace(
        rank=rank,
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        pipeline_parallel_size=pp,
        context_parallel_size=cp,
        tensor_parallel_rank=parts["tp"],
        data_parallel_rank=parts["dp"],
        pipeline_parallel_rank=parts["pp"],
        context_parallel_rank=parts["cp"],
    )


def test_pp2_neighbors_tp1_dp1():
    rg = RankGenerator(tp=1, dp=1, pp=2, cp=1)
    # ranks 0 and 1 are the two PP stages
    c0 = _ctx_from_parts(1, 1, 2, 1, rg.decode(0), 0)
    c1 = _ctx_from_parts(1, 1, 2, 1, rg.decode(1), 1)
    assert is_pipeline_first_stage(c0) and not is_pipeline_last_stage(c0)
    assert is_pipeline_last_stage(c1) and not is_pipeline_first_stage(c1)
    assert pipeline_prev_rank(c0) is None
    assert pipeline_next_rank(c0) == 1
    assert pipeline_prev_rank(c1) == 0
    assert pipeline_next_rank(c1) is None


def test_pp2_neighbors_with_tp2():
    # world 4: tp-fastest. PP neighbors share tp rank.
    rg = RankGenerator(tp=2, dp=1, pp=2, cp=1)
    # rank0: tp0 pp0; rank1: tp1 pp0; rank2: tp0 pp1; rank3: tp1 pp1
    c0 = _ctx_from_parts(2, 1, 2, 1, rg.decode(0), 0)
    c2 = _ctx_from_parts(2, 1, 2, 1, rg.decode(2), 2)
    assert pipeline_next_rank(c0) == 2
    assert pipeline_prev_rank(c2) == 0


def test_pp1_no_neighbors():
    rg = RankGenerator(tp=1, dp=1, pp=1, cp=1)
    c0 = _ctx_from_parts(1, 1, 1, 1, rg.decode(0), 0)
    assert is_pipeline_first_stage(c0) and is_pipeline_last_stage(c0)
    assert pipeline_prev_rank(c0) is None
    assert pipeline_next_rank(c0) is None
