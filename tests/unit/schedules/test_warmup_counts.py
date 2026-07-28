"""Unit tests for 1F1B warmup microbatch counts."""
from __future__ import annotations

from nano_megatron.schedules.one_f_one_b import warmup_microbatches


def test_warmup_counts():
    assert warmup_microbatches(4, 0, 8) == 3
    assert warmup_microbatches(4, 3, 8) == 0
    assert warmup_microbatches(2, 0, 1) == 1  # min(1, 1) == 1


def test_warmup_clamped_by_num_microbatches():
    assert warmup_microbatches(8, 0, 2) == 2
    assert warmup_microbatches(4, 1, 1) == 1  # min(2, 1) == 1


def test_warmup_pp1_is_zero():
    assert warmup_microbatches(1, 0, 4) == 0
