import pytest

from nano_megatron.distributed.parallel_state import (
    get_tensor_parallel_rank,
    get_tensor_parallel_size,
)


class TestTPParalleStateNoDist:
    def test_tp_rank_no_dist(self) -> None:
        assert get_tensor_parallel_rank() == 0

    def test_tp_size_no_dist(self) -> None:
        assert get_tensor_parallel_size() == 1

    def test_tp_group_not_created_raises(self) -> None:
        from nano_megatron.distributed.parallel_state import get_tensor_parallel_group

        with pytest.raises(RuntimeError, match="not created"):
            get_tensor_parallel_group()
