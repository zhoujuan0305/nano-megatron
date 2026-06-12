from nano_megatron.distributed.initialize import is_distributed


class TestIsDistributed:
    def test_not_distributed_without_init(self) -> None:
        assert is_distributed() is False
