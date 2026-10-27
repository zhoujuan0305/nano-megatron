"""Explicit first-order training backward with direct GEMM gradient accumulation.

Ordinary backward/autograd.grad retain their normal semantics. Only parameters
scoped by fused_wgrad_backward use the in-place training path; no global autograd
patch or thread-local state is needed by the engine's backward worker threads.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass
class _Accumulation:
    seen: set[int] = field(default_factory=set)


def configure_wgrad_fusion(module: nn.Module) -> None:
    """Enable only TP linear primitives; reject parameter aliases up front."""
    from nano_megatron.parallel.mappings import ColumnParallelLinear, RowParallelLinear

    seen = set()
    for name, param in module.named_parameters(remove_duplicate=False):
        if id(param) in seen:
            raise ValueError(f"fused wgrad does not support shared/tied parameter {name}")
        seen.add(id(param))
    count = 0
    for child in module.modules():
        if isinstance(child, (ColumnParallelLinear, RowParallelLinear)):
            child.fuse_wgrad_accum = True
            child.weight._nano_fuse_wgrad_accum = True
            count += 1
    if count == 0:
        raise ValueError("fused wgrad requires TP linear modules")


def fused_wgrad_backward(
    module: nn.Module,
    tensors: Tensor,
    grad_tensors: Tensor | None = None,
) -> None:
    """Accumulate enabled, unshared linear weights during one microbatch.

    This helper owns a first-order backward; create_graph/inputs are deliberately
    not accepted. Concurrent backwards using the same parameters and reused
    linear weights within one microbatch are unsupported. Calls outside this
    helper do not manually mutate parameter gradients.
    """
    if not hasattr(torch.Tensor, "register_post_accumulate_grad_hook"):
        raise RuntimeError("fused wgrad requires PyTorch post-accumulate gradient hooks")
    params = []
    identities = set()
    for name, param in module.named_parameters(remove_duplicate=False):
        if not param.requires_grad or not getattr(param, "_nano_fuse_wgrad_accum", False):
            continue
        if id(param) in identities:
            raise ValueError(f"fused wgrad does not support shared/tied parameter {name}")
        if getattr(param, "_nano_wgrad_scope", None) is not None:
            raise RuntimeError("concurrent/reentrant fused backwards on the same parameter")
        identities.add(id(param))
        params.append(param)
    scope = _Accumulation()
    for param in params:
        param._nano_wgrad_scope = scope
    try:
        torch.autograd.backward(tensors, grad_tensors=grad_tensors, create_graph=False)
    finally:
        for param in params:
            del param._nano_wgrad_scope


def accumulate_wgrad(weight: Tensor, grad_output: Tensor, linear_input: Tensor) -> bool:
    """Write a wgrad GEMM directly into .grad when the explicit helper is active."""
    scope: _Accumulation | None = getattr(weight, "_nano_wgrad_scope", None)
    if scope is None:
        return False
    if torch.is_grad_enabled():
        raise RuntimeError("fused wgrad supports first-order training backward only")
    if id(weight) in scope.seen:
        raise RuntimeError("fused wgrad does not support a reused linear weight in one backward")
    scope.seen.add(id(weight))
    output = grad_output.reshape(-1, grad_output.shape[-1])
    inputs = linear_input.reshape(-1, linear_input.shape[-1])
    grad = weight.grad
    beta = 1 if grad is not None else 0
    if grad is None:
        grad = torch.empty_like(weight, memory_format=torch.contiguous_format)
        weight.grad = grad
    # AccumulateGrad still visits this leaf when backward returns None. Because
    # .grad is defined, its native post-accumulate hook notifies DDP exactly once
    # AFTER this Function returns, preserving the engine's communication order.
    if (grad.shape != weight.shape or grad.dtype != weight.dtype
            or grad.device != weight.device or grad.requires_grad
            or not grad.is_contiguous()):
        raise RuntimeError("fused wgrad requires a matching contiguous parameter grad buffer")
    if grad.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr():
        raise RuntimeError("parameter data and fused grad buffer must not alias")
    torch.addmm(grad, output.t(), inputs, beta=beta, out=grad)
    weight._nano_fused_wgrad_calls = getattr(weight, "_nano_fused_wgrad_calls", 0) + 1
    return True
