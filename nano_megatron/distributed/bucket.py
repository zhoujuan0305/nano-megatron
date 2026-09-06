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
        self._grad_views: dict[nn.Parameter, Tensor] = {}
        self._work: Any | None = None
        self._sync_started = False

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

    @property
    def all_ready(self) -> bool:
        return len(self._ready) == len(self._params)

    @property
    def sync_started(self) -> bool:
        return self._sync_started

    @property
    def flat_buffer(self) -> Tensor | None:
        return self._flat

    def prepare_grad_buffer(self, *, zero: bool) -> None:
        """Map every parameter gradient onto this bucket's flat storage."""
        total = sum(param.numel() for param in self._params)
        first = self._params[0]
        if self._flat is None:
            self._flat = torch.empty(
                total, dtype=first.dtype, device=first.device
            )
        if zero:
            self._flat.zero_()

        self._grad_views.clear()
        offset = 0
        for param in self._params:
            view = self._flat[offset : offset + param.numel()].view_as(param)
            self._grad_views[param] = view
            param.grad = view
            offset += param.numel()

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

    def sync(
        self,
        backend: CommBackend,
        group: Any,
        dp_size: int,
        *,
        group_size: int | None = None,
    ) -> None:
        """All-reduce grads then mean-divide.

        *dp_size* is the historical name for the mean divisor (typically
        ``data_parallel_size``).  *group_size* is the communication group
        size used only to decide whether to skip the collective.  When
        *group_size* is ``None``, it defaults to *dp_size* (pure-DP
        backward compatible).

        Pure CP (``dp_size=1``, ``group_size=cp_size>1``) still all-reduces
        over the DP×CP group and divides by 1.
        """
        if self._coalesced:
            return
        missing = [i for i, p in enumerate(self._params) if p.grad is None]
        if missing:
            raise RuntimeError(
                f"GradBucket.sync: {len(missing)} parameter(s) have grad=None "
                f"(unused params not supported)"
            )
        mean_divisor = dp_size
        if group_size is None:
            group_size = mean_divisor
        if mean_divisor < 1 or group_size < 1:
            raise ValueError(
                f"mean_divisor and group_size must be >= 1, "
                f"got mean_divisor={mean_divisor}, group_size={group_size}"
            )
        if group_size == 1:
            self._coalesced = True
            return

        grads = [p.grad for p in self._params]
        uses_grad_views = bool(self._grad_views) and all(
            p.grad is self._grad_views[p] for p in self._params
        )
        if uses_grad_views:
            assert self._flat is not None
            flat = self._flat
        else:
            total = sum(g.numel() for g in grads)
            flat = grads[0].new_empty(total)
            offset = 0
            for g in grads:
                n = g.numel()
                flat[offset : offset + n].copy_(g.reshape(-1))
                offset += n
        backend.all_reduce(flat, group=group, op="sum")
        flat.div_(mean_divisor)
        if not uses_grad_views:
            offset = 0
            for p in self._params:
                g = p.grad
                n = g.numel()
                g.copy_(flat[offset : offset + n].view_as(g))
                offset += n
        self._flat = flat
        self._coalesced = True

    def start_sync(
        self,
        backend: CommBackend,
        group: Any,
        mean_divisor: int,
        *,
        group_size: int,
    ) -> None:
        """Launch an in-place asynchronous reduction on the flat grad buffer."""
        if self._sync_started or self._coalesced:
            return
        if mean_divisor < 1 or group_size < 1:
            raise ValueError(
                f"mean_divisor and group_size must be >= 1, "
                f"got mean_divisor={mean_divisor}, group_size={group_size}"
            )
        missing = [p for p in self._params if p.grad is None]
        if missing:
            raise RuntimeError(
                f"GradBucket.start_sync: {len(missing)} parameter(s) have grad=None"
            )
        if not self._grad_views or any(
            p.grad is not self._grad_views[p] for p in self._params
        ):
            raise RuntimeError(
                "GradBucket.start_sync requires prepare_grad_buffer() before backward"
            )

        self._sync_started = True
        if group_size == 1:
            self._coalesced = True
            return

        assert self._flat is not None
        self._flat.div_(mean_divisor)
        work = backend.all_reduce(
            self._flat, group=group, op="sum", async_op=True
        )
        if hasattr(work, "wait"):
            self._work = work
        else:
            self._coalesced = True

    def finish_sync(self) -> None:
        """Wait at the consumer boundary for a previously launched reduction."""
        if not self._sync_started:
            raise RuntimeError("GradBucket.finish_sync called before start_sync")
        if self._work is not None:
            self._work.wait()
            self._work = None
        self._coalesced = True

    def reset(self, *, keep_grad_buffer: bool = False) -> None:
        if self._work is not None:
            raise RuntimeError("cannot reset GradBucket with communication in flight")
        self._ready.clear()
        self._coalesced = False
        self._sync_started = False
        if not keep_grad_buffer:
            self._flat = None
            self._grad_views.clear()


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
