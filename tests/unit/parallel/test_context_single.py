import torch.distributed as dist

from nano_megatron.parallel import (
    ParallelConfig,
    destroy_parallel,
    get_parallel_context,
    initialize_parallel,
    is_parallel_initialized,
)


def test_initialize_single_process(tmp_path, monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29501")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    if is_parallel_initialized():
        destroy_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
    ctx = initialize_parallel(ParallelConfig(), dist_backend="gloo")
    assert ctx.world_size == 1
    assert ctx.tensor_parallel_rank == 0
    assert ctx.data_parallel_size == 1
    assert get_parallel_context() is ctx
    destroy_parallel()
    assert not is_parallel_initialized()


def test_double_init_raises(monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29502")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    if is_parallel_initialized():
        destroy_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
    initialize_parallel(ParallelConfig(), dist_backend="gloo")
    try:
        import pytest

        with pytest.raises(RuntimeError, match="already initialized"):
            initialize_parallel(ParallelConfig(), dist_backend="gloo")
    finally:
        destroy_parallel()
