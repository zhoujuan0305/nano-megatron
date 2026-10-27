from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nano_megatron.distributed.ddp import DistributedDataParallel
from nano_megatron.parallel.mappings import ColumnParallelLinear, RowParallelLinear
from nano_megatron.parallel.wgrad import fused_wgrad_backward
from nano_megatron.reference.config import ReferenceGPTConfig


class Backend:
    def __init__(self):
        self.dp_calls = 0

    def all_reduce(self, tensor, *, group=None, op="sum", async_op=False):
        if group == "dp":
            self.dp_calls += 1
            tensor.mul_(2)  # Two identical replicas; mean reduction is identity.
        return tensor

    def broadcast(self, tensor, src, *, group=None):
        return tensor


def linear(kind, enabled=True, dtype=torch.float32, backend=None):
    torch.manual_seed(321)
    weight = torch.randn(5, 4, dtype=dtype) * .1
    bias = torch.randn(5, dtype=dtype) * .1
    backend = backend or Backend()
    if kind == "row":
        result = RowParallelLinear(weight, bias, 0, 1, "tp", backend,
                                   fuse_wgrad_accum=enabled)
    else:
        result = ColumnParallelLinear(weight, bias, 0, 1, "tp", backend,
                                      overlap_dgrad=kind == "overlap",
                                      fuse_wgrad_accum=enabled)
    return result


@pytest.mark.parametrize("kind", ["column", "overlap", "row"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("set_to_none", [False, True])
def test_multiple_microbatches_and_steps(kind, dtype, set_to_none):
    fused, reference = linear(kind, dtype=dtype), linear(kind, False, dtype)
    atol, rtol = ((.004, .03) if dtype == torch.bfloat16 else (2e-6, 2e-5))
    for _ in range(2):
        fused.zero_grad(set_to_none=set_to_none)
        reference.zero_grad(set_to_none=set_to_none)
        for _ in range(4):
            x = (torch.randn(2, 3, 4, dtype=dtype) * .1).requires_grad_()
            y = x.detach().clone().requires_grad_()
            upstream = torch.randn(2, 3, 5, dtype=dtype) * .1
            fused_wgrad_backward(fused, (fused(x) * upstream).sum())
            (reference(y) * upstream).sum().backward()
            torch.testing.assert_close(x.grad, y.grad, atol=atol, rtol=rtol)
        for p, q in zip(fused.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=atol, rtol=rtol)
            with torch.no_grad():
                p.add_(p.grad, alpha=-.01)
                q.add_(q.grad, alpha=-.01)
    assert fused.weight._nano_fused_wgrad_calls == 8
    assert not hasattr(fused.weight, "_nano_wgrad_scope")


@pytest.mark.parametrize("kind", ["column", "overlap", "row"])
def test_normal_autograd_grad_remains_correct_without_helper(kind):
    fused, reference = linear(kind), linear(kind, False)
    x = torch.randn(2, 4, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a = torch.autograd.grad(fused(x).square().sum(), (x, *fused.parameters()))
    b = torch.autograd.grad(reference(y).square().sum(), (y, *reference.parameters()))
    for p, q in zip(a, b):
        torch.testing.assert_close(p, q)
    assert all(p.grad is None for p in fused.parameters())
    assert getattr(fused.weight, "_nano_fused_wgrad_calls", 0) == 0


def test_normal_higher_order_path_outside_helper():
    fused, reference = linear("column"), linear("column", False)
    values = []
    for module in (fused, reference):
        x = torch.full((2, 4), .1, requires_grad=True)
        first = torch.autograd.grad(module(x).square().sum(), x, create_graph=True)[0]
        values.append(torch.autograd.grad(first.sum(), module.weight)[0])
    torch.testing.assert_close(*values)


def test_helper_rejects_higher_order_option_without_side_effect():
    module = linear("column")
    loss = module(torch.randn(2, 4)).sum()
    with pytest.raises(TypeError):
        fused_wgrad_backward(module, loss, create_graph=True)
    assert module.weight.grad is None


def test_frozen_weight_keeps_input_and_bias_gradients():
    module = linear("column")
    module.weight.requires_grad_(False)
    x = torch.randn(2, 4, requires_grad=True)
    fused_wgrad_backward(module, module(x).sum())
    assert module.weight.grad is None
    assert x.grad is not None and module.bias.grad is not None


def test_dtype_migration_keeps_fusion_and_grad_buffer_valid():
    module = linear("column")
    fused_wgrad_backward(module, module(torch.randn(2, 4)).sum())
    module.to(dtype=torch.bfloat16)
    module.zero_grad(set_to_none=True)
    fused_wgrad_backward(module, module(torch.randn(2, 4, dtype=torch.bfloat16)).sum())
    assert module.weight.grad.dtype == torch.bfloat16
    assert module.weight._nano_fused_wgrad_calls == 2


def test_autocast_is_rejected_explicitly_when_opted_in():
    module = linear("column")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="not autocast"):
            module(torch.randn(2, 4))


def test_shared_parameter_rejected_before_backward():
    a, b = linear("column"), linear("column")
    b.weight = a.weight
    module = nn.ModuleList([a, b])
    loss = a(torch.randn(2, 4)).sum()
    with pytest.raises(ValueError, match="shared/tied"):
        fused_wgrad_backward(module, loss)
    assert a.weight.grad is None
    assert not hasattr(a.weight, "_nano_wgrad_scope")
    with pytest.raises(ValueError, match="tied embeddings"):
        ReferenceGPTConfig(fuse_wgrad_accum=True, tie_word_embeddings=True)


def test_reused_weight_rejected_and_scope_removed():
    module = linear("column")
    loss = module(torch.randn(2, 4)).sum() + module(torch.randn(2, 4)).sum()
    with pytest.raises(RuntimeError, match="reused linear weight"):
        fused_wgrad_backward(module, loss)
    assert not hasattr(module.weight, "_nano_wgrad_scope")


@pytest.mark.parametrize("overlap", [False, True])
def test_ddp_no_sync_ready_and_persistent_views(overlap):
    backend = Backend()
    model = linear("overlap", backend=backend)
    reference = linear("overlap", False)
    ctx = SimpleNamespace(rank=0, data_context_parallel_group="dp",
                          data_context_parallel_src_rank=0,
                          data_parallel_size=2, context_parallel_size=1,
                          backend=backend)
    ddp = DistributedDataParallel(model, ctx, overlap_grad_reduce=overlap)
    for _ in range(2):
        ddp.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        expected_calls = backend.dp_calls
        initial_view = model.weight.grad
        for i in range(3):
            x = torch.randn(2, 4)
            with ddp.no_sync() if i < 2 else nullcontext():
                fused_wgrad_backward(model, ddp(x).sum())
            reference(x).sum().backward()
            if i < 2:
                assert backend.dp_calls == expected_calls
            if overlap:
                assert model.weight.grad is initial_view
        ddp.finish_grad_sync()
        assert backend.dp_calls == expected_calls + 1
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad)
        ddp.finish_grad_sync()
        assert backend.dp_calls == expected_calls + 1
