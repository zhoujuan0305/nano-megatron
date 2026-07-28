from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator

import torch
import torch.nn as nn
from torch import Tensor

from nano_megatron.distributed.backend import CommBackend
from nano_megatron.distributed.bucket import GradBucket, build_buckets

if TYPE_CHECKING:
    from nano_megatron.parallel.context import ParallelContext


class DistributedDataParallel(nn.Module):
    """Data-parallel wrapper with gradient bucketing and all-reduce mean sync."""

    def __init__(
        self,
        module: nn.Module,
        ctx: ParallelContext,
        *,
        bucket_cap_mb: float = 25.0,
        broadcast_buffers: bool = False,
    ) -> None:
        super().__init__()
        # broadcast_buffers reserved for a later version.
        _ = broadcast_buffers

        self.add_module("module", module)
        self._ctx = ctx
        self._backend: CommBackend = ctx.backend
        # Sync over DP×CP.  Local-CE under CP scales loss by cp_size so that
        # mean (not sum) over the full DP×CP group recovers full-sequence grads.
        # group_size must be dp*cp so pure CP (dp=1, cp>1) still all-reduces.
        self._dp_group = ctx.data_context_parallel_group
        self._mean_divisor = (
            ctx.data_parallel_size * ctx.context_parallel_size
        )
        self._sync_group_size = self._mean_divisor
        self._buckets: list[GradBucket] = build_buckets(module, bucket_cap_mb)
        self._param_to_bucket: dict[nn.Parameter, GradBucket] = {
            p: bucket for bucket in self._buckets for p in bucket.params
        }
        self._param_names: dict[int, str] = {
            id(p): name for name, p in module.named_parameters()
        }
        # Retain handles so hooks are not garbage-collected.
        self._hook_handles: list[Any] = []
        # True once finish_grad_sync completes (or no-ops) for this iteration.
        # Cleared on forward / first grad-ready mark so a new iteration can sync.
        self._sync_done: bool = True
        # When False, grad hooks skip mark_ready so buckets accumulate without
        # triggering all_reduce.  Managed by no_sync() context manager.
        self._require_backward_grad_sync: bool = True

        self._broadcast_params()
        self._register_grad_hooks()

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        """Defer bucket mark_ready/sync; grads still accumulate on .grad."""
        prev = self._require_backward_grad_sync
        self._require_backward_grad_sync = False
        try:
            yield
        finally:
            self._require_backward_grad_sync = prev

    def _param_device(self) -> torch.device:
        try:
            return next(self.module.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _broadcast_params(self) -> None:
        device = self._param_device()
        # Leader is the rank with dp_rank==0 and cp_rank==0 in this DP×CP group.
        is_leader = (
            self._ctx.data_parallel_rank == 0
            and self._ctx.context_parallel_rank == 0
        )
        leader = torch.tensor(
            [self._ctx.rank if is_leader else -1],
            dtype=torch.long,
            device=device,
        )
        self._backend.all_reduce(leader, group=self._dp_group, op="max")
        dp_src = int(leader.item())

        for param in self._param_to_bucket:
            self._backend.broadcast(
                param.data, src=dp_src, group=self._dp_group
            )

    def _on_param_grad_ready(self, param: nn.Parameter) -> None:
        bucket = self._param_to_bucket.get(param)
        if bucket is None:
            return
        if not self._require_backward_grad_sync:
            self._sync_done = False
            return
        # New grads this iteration — allow finish_grad_sync to run again.
        self._sync_done = False
        if bucket.mark_ready(param):
            bucket.sync(
                self._backend,
                self._dp_group,
                self._mean_divisor,
                group_size=self._sync_group_size,
            )

    def _register_grad_hooks(self) -> None:
        use_post_accumulate = hasattr(
            torch.Tensor, "register_post_accumulate_grad_hook"
        )
        for param in self._param_to_bucket:
            if use_post_accumulate:
                handle = param.register_post_accumulate_grad_hook(
                    self._make_post_accumulate_hook(param)
                )
            else:
                handle = param.register_hook(self._make_grad_hook(param))
            self._hook_handles.append(handle)

    def _make_post_accumulate_hook(
        self, param: nn.Parameter
    ) -> Callable[[nn.Parameter], None]:
        def hook(_param: nn.Parameter) -> None:
            # .grad is already accumulated when this hook runs.
            self._on_param_grad_ready(param)

        return hook

    def _make_grad_hook(
        self, param: nn.Parameter
    ) -> Callable[[Tensor], None]:
        def hook(_grad: Tensor) -> None:
            # register_hook runs before autograd writes .grad. Mutating .grad
            # here risks double-accumulation, so only mark when .grad is already
            # present; finish_grad_sync flushes any remaining buckets.
            if param.grad is not None:
                self._on_param_grad_ready(param)

        return hook

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # Mark iteration open so finish_grad_sync runs even if grad hooks
        # do not fire (register_hook fallback before .grad is written).
        self._sync_done = False
        return self.module(*args, **kwargs)

    def finish_grad_sync(self) -> None:
        """Flush uncoalesced buckets. Idempotent until the next forward/backward."""
        if self._sync_done:
            return

        any_grad = any(p.grad is not None for p in self._param_to_bucket)
        any_activity = any(
            bucket.coalesced or bucket.has_pending_ready for bucket in self._buckets
        )
        # No backward (or grads cleared): nothing to sync.
        if not any_grad and not any_activity:
            self._sync_done = True
            return

        for bucket in self._buckets:
            if bucket.coalesced:
                continue
            missing = [p for p in bucket.params if p.grad is None]
            if missing:
                names = [
                    self._param_names.get(id(p), "<unknown>") for p in missing
                ]
                raise RuntimeError(
                    f"finish_grad_sync rank={self._ctx.rank}: "
                    f"missing grads for {names}"
                )
            bucket.sync(
                self._backend,
                self._dp_group,
                self._mean_divisor,
                group_size=self._sync_group_size,
            )
        for bucket in self._buckets:
            bucket.reset()
        self._sync_done = True
