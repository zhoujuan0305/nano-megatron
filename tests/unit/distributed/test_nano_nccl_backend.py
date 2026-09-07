import torch

from nano_megatron.distributed.nano_nccl_backend import _RecvBufferPool


def test_recv_buffer_pool_reuses_shape_across_changing_input_addresses():
    pool = _RecvBufferPool()
    first_input = torch.empty(1024, dtype=torch.bfloat16)
    first_output = pool.acquire(first_input)
    pool.release(first_output)

    second_input = torch.empty(1024, dtype=torch.bfloat16)
    second_output = pool.acquire(second_input)

    assert second_input.data_ptr() != first_input.data_ptr()
    assert second_output.data_ptr() == first_output.data_ptr()
    assert pool.cached_buffer_count == 0


def test_recv_buffer_pool_keeps_concurrent_outputs_distinct():
    pool = _RecvBufferPool()
    template = torch.empty(257, dtype=torch.float32)
    first = pool.acquire(template)
    second = pool.acquire(template)

    assert first.data_ptr() != second.data_ptr()
    pool.release(first)
    pool.release(second)
    assert pool.cached_buffer_count == 2
