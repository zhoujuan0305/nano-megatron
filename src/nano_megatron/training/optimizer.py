from __future__ import annotations

from torch import nn, optim


def create_adam_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    no_decay_keys: tuple[str, ...] = ("bias", "LayerNorm.weight", "layernorm.weight"),
) -> optim.AdamW:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd_key in name for nd_key in no_decay_keys):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return optim.AdamW(optimizer_groups, lr=lr)
