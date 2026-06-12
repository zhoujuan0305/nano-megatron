from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    rng_state: dict[str, Any] | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": config,
    }
    if rng_state is not None:
        checkpoint["rng_state"] = rng_state
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint["step"]
    config = checkpoint.get("config", {})
    if "rng_state" in checkpoint:
        rng_state = checkpoint["rng_state"]
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if "cuda" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state["cuda"])
    return step, config


def get_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state
