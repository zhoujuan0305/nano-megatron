"""Static-shape reduction: preserve masked loss/grad without host scalar reads."""
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from nano_megatron.parallel import vocab_parallel as vp


class _NoHostScalarOrBooleanIndex(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func != torch.ops.aten._local_scalar_dense.default
        if func == torch.ops.aten.index.Tensor:
            assert not any(
                index is not None and index.dtype == torch.bool
                for index in args[1]
            )
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("mask_kind", ["all", "mixed", "none", "empty"])
def test_static_reduction_matches_masked_loss_and_gradient(monkeypatch, dtype, reduction, mask_kind):
    n = 0 if mask_kind == "empty" else 1024
    labels = torch.zeros((1, n), dtype=torch.long)
    if mask_kind == "mixed":
        labels[:, ::3] = -100
    elif mask_kind == "none":
        labels.fill_(-100)
    values = (torch.arange(n, dtype=torch.float32) % 7 + 100).to(dtype).requires_grad_()
    valid = labels.reshape(-1) != -100
    per_token = values.masked_fill(~valid, 0)
    monkeypatch.setattr(vp._VocabParallelCrossEntropy, "apply", staticmethod(lambda *args: per_token))
    with _NoHostScalarOrBooleanIndex():
        result = vp.vocab_parallel_cross_entropy_from_shifted(
            torch.empty((1, n, 2), dtype=dtype), labels,
            vocab_start_index=0, vocab_end_index=2, group=None, backend=None,
            reduction=reduction,
        )
    if valid.any():
        reference = per_token[valid].mean() if reduction == "mean" else per_token[valid].sum()
    else:
        reference = per_token.sum() * 0
    assert result.dtype == reference.dtype == dtype
    torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-6)
    grad = torch.autograd.grad(result, values, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(reference, values)[0]
    torch.testing.assert_close(grad, expected_grad, rtol=1e-6, atol=1e-6)


class _IdentityBackend:
    def all_reduce(self, tensor, **kwargs):
        return tensor


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_real_ce_full_logits_and_gradients(reduction, all_ignored):
    torch.manual_seed(91)
    logits = torch.randn(2, 5, 11, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 11, (2, 5))
    labels[:, 1::2] = -100
    if all_ignored:
        labels.fill_(-100)
    with _NoHostScalarOrBooleanIndex():
        result = vp.vocab_parallel_cross_entropy_from_shifted(
            logits, labels, vocab_start_index=0, vocab_end_index=11,
            group=None, backend=_IdentityBackend(), reduction=reduction,
        )
    reference = (
        logits.sum() * 0 if all_ignored else
        torch.nn.functional.cross_entropy(logits.reshape(-1, 11), labels.reshape(-1), reduction=reduction)
    )
    torch.testing.assert_close(result, reference, rtol=1e-10, atol=1e-10)
    grad = torch.autograd.grad(result, logits, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(reference, logits)[0]
    torch.testing.assert_close(grad, reference_grad, rtol=1e-10, atol=1e-10)
