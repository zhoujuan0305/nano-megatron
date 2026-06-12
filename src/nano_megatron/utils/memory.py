from __future__ import annotations

import torch


def get_peak_memory_mb(device: torch.device | None = None) -> float:
    if device is not None and device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0.0
