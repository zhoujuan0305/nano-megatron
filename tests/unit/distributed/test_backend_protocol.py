import pytest
import torch
from nano_megatron.distributed.torch_backend import reduce_op_from_string, TorchDistBackend
from nano_megatron.distributed.backend import CommBackend


def test_reduce_op_from_string():
    assert reduce_op_from_string("sum") == torch.distributed.ReduceOp.SUM
    assert reduce_op_from_string("max") == torch.distributed.ReduceOp.MAX
    with pytest.raises(ValueError, match="unsupported"):
        reduce_op_from_string("mean")


def test_torch_backend_is_comm_backend():
    backend: CommBackend = TorchDistBackend()
    assert hasattr(backend, "all_reduce")
