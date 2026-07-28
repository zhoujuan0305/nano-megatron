from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from nano_megatron.distributed.backend import CommBackend


class GradBucket:
    def __init__(self, params: list[nn.Parameter]) -> None:
        if not params:
            raise ValueError("GradBucket requires at least one parameter")
        self._params = list(params)
        self._index: dict[nn.Parameter, int] = {p: i for i, p in enumerate(self._params)}
        self._ready: set[int] = set()
        self._coalesced = False
        self._flat: Tensor | None = None

    @property
    def params(self) -> list[nn.Parameter]:
        return list(self._params)

    @property
    def coalesced(self) -> bool:
        return self._coalesced

    @property
    def has_pending_ready(self) -> bool:
        """True if at least one param has been marked ready this iteration."""
        return bool(self._ready)

    def mark_ready(self, param: nn.Parameter) -> bool:
        idx = self._index.get(param)
        if idx is None:
            raise KeyError("parameter not in this bucket")
        if self._coalesced:
            return False
        if idx in self._ready:
            return False
        self._ready.add(idx)
        return len(self._ready) == len(self._params)

    def sync(self, backend: CommBackend, group: Any, dp_size: int) -> None:
        if self._coalesced:
            return
        missing = [i for i, p in enumerate(self._params) if p.grad is None]
        if missing:
            raise RuntimeError(
                f"GradBucket.sync: {len(missing)} parameter(s) have grad=None "
                f"(unused params not supported)"
            )
        if dp_size < 1:
            raise ValueError(f"dp_size must be >= 1, got {dp_size}")
        if dp_size == 1:
            self._coalesced = True
            return

        grads = [p.grad for p in self._params]
        total = sum(g.numel() for g in grads)
        flat = grads[0].new_empty(total)
        offset = 0
        for g in grads:
            n = g.numel()
            flat[offset : offset + n].copy_(g.reshape(-1))
            offset += n
        backend.all_reduce(flat, group=group, op="sum")
        flat.div_(dp_size)
        offset = 0
        for p in self._params:
            g = p.grad
            n = g.numel()
            g.copy_(flat[offset : offset + n].view_as(g))
            offset += n
        self._flat = flat
        self._coalesced = True

    def reset(self) -> None:
        self._ready.clear()
        self._coalesced = False
        self._flat = None


def build_buckets(
    module: nn.Module,
    bucket_cap_mb: float = 25.0,
) -> list[GradBucket]:
    if bucket_cap_mb <= 0:
        raise ValueError(f"bucket_cap_mb must be > 0, got {bucket_cap_mb}")
    cap_bytes = int(bucket_cap_mb * 1024 * 1024)
    params = [p for p in module.parameters() if p.requires_grad]
    params = list(reversed(params))
    if not params:
        return []

    devices = {p.device for p in params}
    if len(devices) > 1:
        raise RuntimeError(
            f"build_buckets: parameters span multiple devices: {devices}"
        )

    buckets: list[GradBucket] = []
    current: list[nn.Parameter] = []
    current_bytes = 0
    current_dtype: torch.dtype | None = None

    def flush() -> None:
        nonlocal current, current_bytes, current_dtype
        if current:
            buckets.append(GradBucket(current))
            current = []
            current_bytes = 0
            current_dtype = None

    for p in params:
        nbytes = p.numel() * p.element_size()
        if current and (
            p.dtype != current_dtype
            or current_bytes + nbytes > cap_bytes
        ):
            flush()
        current.append(p)
        current_bytes += nbytes
        current_dtype = p.dtype
    flush()
    return buckets
