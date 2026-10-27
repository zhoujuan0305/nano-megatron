from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from nano_megatron.model.pipeline import PipelineStage
from nano_megatron.model.tp_gpt import TPGPT


class IdentityBackend:
    def all_reduce(self, tensor, *, group=None, op="sum", async_op=False):
        return tensor


@pytest.mark.parametrize("cls", [PipelineStage, TPGPT])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_explicit_fp32_vocab_loss_and_original_logit_gradient_dtype(cls, dtype):
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_ce_fp32=True)
    model._pp_rank = 0
    model._pp_size = 1
    model._tp_size = 2
    model._cp_size = 1
    model.vocab_start_index = 0
    model.vocab_end_index = 13
    model._tp_group = None
    model._tp_backend = IdentityBackend()
    torch.manual_seed(42)
    logits = torch.randn(2, 7, 13, dtype=dtype, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    labels = torch.randint(0, 13, (2, 7))
    loss = model.shifted_cross_entropy(logits, labels)
    expected = F.cross_entropy(reference[:, :-1].float().reshape(-1, 13),
                               labels[:, 1:].reshape(-1))
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, expected)
    loss.backward()
    expected.backward()
    assert logits.grad.dtype == dtype
    torch.testing.assert_close(logits.grad, reference.grad)


def test_default_loss_precision_policy_is_unchanged():
    from nano_megatron.reference.config import ReferenceGPTConfig
    assert ReferenceGPTConfig().vocab_ce_fp32 is False
