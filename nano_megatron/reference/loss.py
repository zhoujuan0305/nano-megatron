from __future__ import annotations

from torch import Tensor
import torch.nn.functional as F


def shifted_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Token-level CE: predict labels[:, 1:] from logits[:, :-1]. Mean over non-ignored.

    Mean vs sum reduction differs only by the constant factor 1/n_tokens.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )
