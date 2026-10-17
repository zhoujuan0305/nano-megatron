"""Experimental static-input full-step CUDA Graph runner for benchmark_pp.

Parameters, inputs, optimizer-state addresses and hyperparameters must stay fixed.
Capture/extra updates are preparation, not measured steps. No eager step is run
on these communicators after capture. This is not an I/O/checkpoint scheduler.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from scripts.benchmark_optimizer import BenchmarkAdamW


class BenchmarkCudaGraphStep:
    def __init__(self, step_fn: Callable[[], None], optimizer: BenchmarkAdamW,
                 losses: list[torch.Tensor], device: torch.device) -> None:
        self.step_fn, self.optimizer, self.losses = step_fn, optimizer, losses
        self.device = torch.device(device)
        if self.device.type != "cuda" or not optimizer.fused:
            raise ValueError("CUDA Graph execution requires CUDA fused BenchmarkAdamW")
        optimizer.optimizer.defaults["capturable"] = True
        for group in optimizer.optimizer.param_groups:
            group["capturable"] = True
        self.captured_losses: tuple[torch.Tensor, ...] = ()
        self.captured_gradients: tuple[torch.Tensor | None, ...] = ()
        self._graph: torch.cuda.CUDAGraph | None = None
        self._attempted = self._closed = self._failed = False

    def _state_steps(self) -> list[torch.Tensor]:
        steps = []
        for master in self.optimizer._masters:
            state = self.optimizer.optimizer.state.get(master, {})
            step = state.get("step")
            if not isinstance(step, torch.Tensor) or not step.is_cuda or step.numel() != 1:
                raise RuntimeError("all optimizer states must have initialized CUDA step tensors")
            steps.append(step)
        if not steps:
            raise RuntimeError("optimizer state must be initialized before graph capture")
        return steps

    def run(self) -> None:
        if self._closed or self._failed:
            raise RuntimeError("CUDA Graph runner is closed or failed")
        if self._graph is None:
            self.step_fn()
        else:
            self._graph.replay()
            self.optimizer.step_count += 1
            self.losses[:] = self.captured_losses

    def prepare(self, warmup_steps: int) -> dict[str, int | float | bool]:
        if self._closed or self._failed or self._attempted:
            raise RuntimeError("CUDA Graph runner is closed, failed, or already prepared")
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be positive")
        steps = self._state_steps()
        self._attempted = True
        started = time.perf_counter()
        before_steps = torch.stack(steps).clone()
        counter_before = self.optimizer.step_count
        try:
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.step_fn()
            # Capture records GPU work but executes the Python bookkeeping once.
            cpu_delta = self.optimizer.step_count - counter_before
            self.optimizer.step_count = counter_before
            if not torch.equal(before_steps, torch.stack(self._state_steps())):
                raise RuntimeError("capture unexpectedly executed optimizer updates")
            if cpu_delta != 1:
                raise RuntimeError(f"capture CPU step counter delta must be one, got {cpu_delta}")
            self.captured_losses = tuple(self.losses)
            self.captured_gradients = tuple(p.grad for p in self.optimizer._model_params)
            self._graph = graph
            for _ in range(warmup_steps):
                self.run()
            torch.cuda.synchronize(self.device)
        except Exception:
            self._failed = True
            self._graph = None
            raise  # Never silently fall back to eager or conceal a GPU update.
        return dict(extra_warmup_updates=warmup_steps,
                    preparation_seconds=time.perf_counter() - started,
                    capture_gpu_update_delta=0, cpu_counter_compensated=True)

    def close(self) -> None:
        """Release the graph before destroying its NCCL process groups."""
        self._graph = None
        self.captured_losses = self.captured_gradients = ()
        self._closed = True
