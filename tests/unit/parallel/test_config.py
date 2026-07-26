import pytest
from nano_megatron.parallel.config import ParallelConfig


def test_infer_dp_from_world_size():
    cfg = ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2)
    assert cfg.resolved_data_parallel_size(8) == 2


def test_explicit_dp_ok():
    cfg = ParallelConfig(tensor_parallel_size=2, data_parallel_size=4)
    cfg.validate(8)


def test_world_size_mismatch_raises():
    cfg = ParallelConfig(tensor_parallel_size=2, data_parallel_size=2)
    with pytest.raises(ValueError, match="world_size"):
        cfg.validate(8)


def test_non_positive_raises():
    with pytest.raises(ValueError):
        ParallelConfig(tensor_parallel_size=0).validate(1)
