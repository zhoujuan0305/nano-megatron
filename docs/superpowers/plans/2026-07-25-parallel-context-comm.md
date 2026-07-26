# Parallel Context and Communication Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Megatron-compatible parallel topology (`ParallelContext`) and a sync `CommBackend` with `TorchDistBackend`, verified by pure unit tests and NCCL multi-GPU tests.

**Architecture:** `parallel/` owns config, rank generation, and process-wide context; `distributed/` owns the communication protocol and PyTorch backend. Higher layers query context and call backend methods only.

**Tech Stack:** Python 3.10+, PyTorch distributed (NCCL), pytest, torchrun.

## Global Constraints

- 4 spaces; `snake_case` files/functions; `PascalCase` classes; type annotations on public APIs
- Prefer dataclasses for config/runtime state
- No direct `torch.distributed` calls outside `distributed/torch_backend.py` and context init (group creation may use `torch.distributed.new_group` in `parallel/context.py`)
- Default rank order `"tp-cp-dp-pp"` (Megatron without EP)
- `world_size == tp * cp * dp * pp`
- Sync collectives only; no async Work handles
- Do not implement TP mappings, DDP, ZeRO, PP schedules, or EP
- Do not write hostnames, IPs, GPU UUIDs, or absolute home paths into committed files
- NCCL tests skip when insufficient GPUs; unit topology tests always run
- Comments only for non-obvious why; no narrating comments
- Commit after each task

## File Structure

| Path | Responsibility |
|------|----------------|
| `nano_megatron/parallel/config.py` | `ParallelConfig` dataclass + validation |
| `nano_megatron/parallel/rank_generator.py` | Orthogonal rank group generation (Megatron-compatible) |
| `nano_megatron/parallel/context.py` | `ParallelContext`, initialize/destroy/get |
| `nano_megatron/parallel/__init__.py` | Public exports |
| `nano_megatron/distributed/backend.py` | `CommBackend` Protocol |
| `nano_megatron/distributed/torch_backend.py` | `TorchDistBackend` |
| `nano_megatron/distributed/__init__.py` | Public exports |
| `tests/unit/parallel/test_rank_generator.py` | Topology golden tests |
| `tests/unit/parallel/test_config.py` | Config validation |
| `tests/unit/distributed/test_torch_backend_ops.py` | Op name mapping helpers if pure; else skip |
| `tests/distributed/test_parallel_context_nccl.py` | NCCL multi-process tests |
| `tests/distributed/common.py` | Shared torchrun helpers |
| `pyproject.toml` | Optional: pytest markers |

---

### Task 1: RankGenerator (pure topology)

**Files:**
- Create: `nano_megatron/parallel/__init__.py`
- Create: `nano_megatron/parallel/rank_generator.py`
- Create: `tests/unit/parallel/test_rank_generator.py`

**Interfaces:**
- Produces:
  - `generate_masked_orthogonal_rank_groups(world_size: int, parallel_size: list[int], mask: list[bool]) -> list[list[int]]`
  - `class RankGenerator` with `__init__(self, tp: int, dp: int, pp: int, cp: int, order: str = "tp-cp-dp-pp")`
  - `RankGenerator.get_ranks(self, token: str) -> list[list[int]]`
  - `RankGenerator.decode(self, global_rank: int) -> dict[str, int]` mapping each dim name in order to local rank
  - `RankGenerator.encode(self, ranks: dict[str, int]) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/parallel/test_rank_generator.py
from nano_megatron.parallel.rank_generator import RankGenerator


def test_tp_groups_world8_tp2_dp2_pp2():
    # order tp-cp-dp-pp, cp=1 → same as tp-dp-pp
    rg = RankGenerator(tp=2, dp=2, pp=2, cp=1, order="tp-cp-dp-pp")
    assert rg.get_ranks("tp") == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
    ]


def test_dp_groups_world8_tp2_dp2_pp2():
    rg = RankGenerator(tp=2, dp=2, pp=2, cp=1, order="tp-cp-dp-pp")
    assert rg.get_ranks("dp") == [
        [0, 2],
        [1, 3],
        [4, 6],
        [5, 7],
    ]


def test_pp_groups_world8_tp2_dp2_pp2():
    rg = RankGenerator(tp=2, dp=2, pp=2, cp=1, order="tp-cp-dp-pp")
    assert rg.get_ranks("pp") == [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]


def test_cp_groups_world8_tp2_cp2_dp2():
    rg = RankGenerator(tp=2, dp=2, pp=1, cp=2, order="tp-cp-dp-pp")
    assert rg.get_ranks("cp") == [
        [0, 2],
        [1, 3],
        [4, 6],
        [5, 7],
    ]


def test_dp_cp_groups_world8_tp2_cp2_dp2():
    rg = RankGenerator(tp=2, dp=2, pp=1, cp=2, order="tp-cp-dp-pp")
    assert rg.get_ranks("dp-cp") == [
        [0, 2, 4, 6],
        [1, 3, 5, 7],
    ]


def test_encode_decode_roundtrip():
    rg = RankGenerator(tp=2, dp=2, pp=2, cp=1, order="tp-cp-dp-pp")
    for rank in range(8):
        parts = rg.decode(rank)
        assert rg.encode(parts) == rank


def test_world_size_property():
    rg = RankGenerator(tp=2, dp=2, pp=2, cp=1)
    assert rg.world_size == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/parallel/test_rank_generator.py -v`  
Expected: FAIL (import error or missing symbol)

- [ ] **Step 3: Implement RankGenerator**

Implement Megatron-compatible logic:

```python
# nano_megatron/parallel/rank_generator.py
from __future__ import annotations


def prefix_product(values: list[int], init: int = 1) -> list[int]:
    out = [init]
    cur = init
    for v in values:
        cur *= v
        out.append(cur)
    return out


def generate_masked_orthogonal_rank_groups(
    world_size: int,
    parallel_size: list[int],
    mask: list[bool],
) -> list[list[int]]:
    def inner_product(a: list[int], b: list[int]) -> int:
        return sum(x * y for x, y in zip(a, b))

    def decompose(index: int, shape: list[int], stride: list[int] | None = None) -> list[int]:
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        assert inner_product(idx, stride[:-1]) == index
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]
    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]
    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size
    ranks: list[list[int]] = []
    for group_index in range(num_of_group):
        group_indices = decompose(group_index, unmasked_shape, unmasked_stride)
        rank_group: list[int] = []
        for rank_in_group in range(group_size):
            rank_indices = decompose(rank_in_group, masked_shape, masked_stride)
            combined = []
            mi = ui = 0
            for m in mask:
                if m:
                    combined.append(rank_indices[mi])
                    mi += 1
                else:
                    combined.append(group_indices[ui])
                    ui += 1
            rank = inner_product(combined, global_stride[:-1])
            rank_group.append(rank)
        ranks.append(rank_group)
    return ranks


class RankGenerator:
    def __init__(
        self,
        tp: int,
        dp: int,
        pp: int,
        cp: int = 1,
        order: str = "tp-cp-dp-pp",
    ) -> None:
        if min(tp, dp, pp, cp) < 1:
            raise ValueError("all parallel sizes must be >= 1")
        self.tp = tp
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.world_size = tp * dp * pp * cp
        self.name_to_size = {"tp": tp, "pp": pp, "dp": dp, "cp": cp}
        order = order.lower()
        for name, size in self.name_to_size.items():
            if name not in order:
                if size != 1:
                    raise RuntimeError(
                        f"size of ({name}) is ({size}), but order ({order}) omits it"
                    )
                order = f"{order}-{name}"
        self.order = order
        self.ordered_names = order.split("-")
        self.ordered_size = [self.name_to_size[n] for n in self.ordered_names]

    def get_mask(self, token: str) -> list[bool]:
        tokens = token.split("-")
        mask = [False] * len(self.ordered_names)
        for t in tokens:
            mask[self.ordered_names.index(t)] = True
        return mask

    def get_ranks(self, token: str) -> list[list[int]]:
        return generate_masked_orthogonal_rank_groups(
            self.world_size, self.ordered_size, self.get_mask(token)
        )

    def decode(self, global_rank: int) -> dict[str, int]:
        if not 0 <= global_rank < self.world_size:
            raise ValueError(f"rank {global_rank} out of range [0, {self.world_size})")
        stride = prefix_product(self.ordered_size)
        parts: dict[str, int] = {}
        for name, size, s in zip(self.ordered_names, self.ordered_size, stride):
            parts[name] = (global_rank // s) % size
        return parts

    def encode(self, ranks: dict[str, int]) -> int:
        stride = prefix_product(self.ordered_size)
        total = 0
        for name, s in zip(self.ordered_names, stride):
            total += ranks[name] * s
        return total
```

```python
# nano_megatron/parallel/__init__.py
from nano_megatron.parallel.rank_generator import RankGenerator, generate_masked_orthogonal_rank_groups

__all__ = ["RankGenerator", "generate_masked_orthogonal_rank_groups"]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/unit/parallel/test_rank_generator.py -v`  
Expected: all PASS

If golden vectors fail, re-derive from Megatron formula with order `tp-cp-dp-pp` and fix either test expectations or implementation — prefer matching Megatron algorithm above.

- [ ] **Step 5: Commit**

```bash
git add nano_megatron/parallel tests/unit/parallel/test_rank_generator.py
git commit -m "feat(parallel): add Megatron-compatible RankGenerator"
```

---

### Task 2: ParallelConfig

**Files:**
- Create: `nano_megatron/parallel/config.py`
- Create: `tests/unit/parallel/test_config.py`
- Modify: `nano_megatron/parallel/__init__.py`

**Interfaces:**
- Produces:
```python
@dataclass(frozen=True)
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    data_parallel_size: int | None = None
    order: str = "tp-cp-dp-pp"

    def resolved_data_parallel_size(self, world_size: int) -> int: ...
    def validate(self, world_size: int) -> None: ...
    def product_without_dp(self) -> int:  # tp * pp * cp
```

- [ ] **Step 1: Write failing tests**

```python
import pytest
from nano_megatron.parallel.config import ParallelConfig


def test_infer_dp_from_world_size():
    cfg = ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2)
    assert cfg.resolved_data_parallel_size(8) == 2


def test_explicit_dp_ok():
    cfg = ParallelConfig(tensor_parallel_size=2, data_parallel_size=4)
    cfg.validate(8)


def test_world_size_mismatch_raises():
    cfg = ParallelConfig(tensor_parallel_size=2, data_parallel_size=2)
    with pytest.raises(ValueError, match="world_size"):
        cfg.validate(8)


def test_non_positive_raises():
    with pytest.raises(ValueError):
        ParallelConfig(tensor_parallel_size=0).validate(1)
```

- [ ] **Step 2: Run — expect FAIL**

`python -m pytest tests/unit/parallel/test_config.py -v`

- [ ] **Step 3: Implement ParallelConfig**

```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    data_parallel_size: int | None = None
    order: str = "tp-cp-dp-pp"

    def product_without_dp(self) -> int:
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.context_parallel_size
        )

    def resolved_data_parallel_size(self, world_size: int) -> int:
        self._check_positive_sizes()
        base = self.product_without_dp()
        if world_size % base != 0:
            raise ValueError(
                f"world_size ({world_size}) not divisible by tp*pp*cp ({base})"
            )
        inferred = world_size // base
        if self.data_parallel_size is None:
            return inferred
        if self.data_parallel_size != inferred:
            raise ValueError(
                f"data_parallel_size ({self.data_parallel_size}) inconsistent with "
                f"world_size ({world_size}) / (tp*pp*cp={base}) = {inferred}"
            )
        return self.data_parallel_size

    def validate(self, world_size: int) -> None:
        self._check_positive_sizes()
        dp = self.resolved_data_parallel_size(world_size)
        product = self.product_without_dp() * dp
        if product != world_size:
            raise ValueError(
                f"world_size ({world_size}) != tp*cp*dp*pp ({product})"
            )

    def _check_positive_sizes(self) -> None:
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "context_parallel_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.data_parallel_size is not None and self.data_parallel_size < 1:
            raise ValueError("data_parallel_size must be >= 1")
```

Export `ParallelConfig` from `parallel/__init__.py`.

- [ ] **Step 4: Tests pass**

`python -m pytest tests/unit/parallel/test_config.py -v`

- [ ] **Step 5: Commit**

```bash
git add nano_megatron/parallel/config.py nano_megatron/parallel/__init__.py tests/unit/parallel/test_config.py
git commit -m "feat(parallel): add ParallelConfig validation"
```

---

### Task 3: CommBackend protocol + TorchDistBackend

**Files:**
- Create: `nano_megatron/distributed/__init__.py`
- Create: `nano_megatron/distributed/backend.py`
- Create: `nano_megatron/distributed/torch_backend.py`
- Create: `tests/unit/distributed/test_backend_protocol.py`

**Interfaces:**
- Produces:
```python
class CommBackend(Protocol):
    def all_reduce(self, tensor: Tensor, *, group: Any | None = None, op: str = "sum") -> Tensor: ...
    def reduce_scatter(self, output: Tensor, input_list: list[Tensor], *, group: Any | None = None, op: str = "sum") -> Tensor: ...
    def all_gather(self, tensor_list: list[Tensor], tensor: Tensor, *, group: Any | None = None) -> list[Tensor]: ...
    def send(self, tensor: Tensor, dst: int, *, group: Any | None = None) -> None: ...
    def recv(self, tensor: Tensor, src: int, *, group: Any | None = None) -> Tensor: ...
    def barrier(self, *, group: Any | None = None) -> None: ...

def reduce_op_from_string(op: str) -> torch.distributed.ReduceOp: ...

class TorchDistBackend:
    # implements CommBackend
```

- [ ] **Step 1: Write failing unit test for op mapping + structural protocol**

```python
import pytest
import torch
from nano_megatron.distributed.torch_backend import reduce_op_from_string, TorchDistBackend
from nano_megatron.distributed.backend import CommBackend


def test_reduce_op_from_string():
    assert reduce_op_from_string("sum") == torch.distributed.ReduceOp.SUM
    assert reduce_op_from_string("max") == torch.distributed.ReduceOp.MAX
    with pytest.raises(ValueError, match="unsupported"):
        reduce_op_from_string("mean")


def test_torch_backend_is_comm_backend():
    backend: CommBackend = TorchDistBackend()
    assert hasattr(backend, "all_reduce")
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

```python
# backend.py
from __future__ import annotations
from typing import Any, Protocol
import torch
from torch import Tensor


class CommBackend(Protocol):
    def all_reduce(
        self, tensor: Tensor, *, group: Any | None = None, op: str = "sum"
    ) -> Tensor: ...

    def reduce_scatter(
        self,
        output: Tensor,
        input_list: list[Tensor],
        *,
        group: Any | None = None,
        op: str = "sum",
    ) -> Tensor: ...

    def all_gather(
        self,
        tensor_list: list[Tensor],
        tensor: Tensor,
        *,
        group: Any | None = None,
    ) -> list[Tensor]: ...

    def send(self, tensor: Tensor, dst: int, *, group: Any | None = None) -> None: ...

    def recv(self, tensor: Tensor, src: int, *, group: Any | None = None) -> Tensor: ...

    def barrier(self, *, group: Any | None = None) -> None: ...
```

```python
# torch_backend.py
from __future__ import annotations
from typing import Any
import torch
import torch.distributed as dist
from torch import Tensor

_OP_MAP = {
    "sum": dist.ReduceOp.SUM,
    "max": dist.ReduceOp.MAX,
    "min": dist.ReduceOp.MIN,
    "product": dist.ReduceOp.PRODUCT,
}


def reduce_op_from_string(op: str) -> dist.ReduceOp:
    key = op.lower()
    if key not in _OP_MAP:
        raise ValueError(f"unsupported reduce op: {op!r}")
    return _OP_MAP[key]


class TorchDistBackend:
    def all_reduce(
        self, tensor: Tensor, *, group: Any | None = None, op: str = "sum"
    ) -> Tensor:
        dist.all_reduce(tensor, op=reduce_op_from_string(op), group=group)
        return tensor

    def reduce_scatter(
        self,
        output: Tensor,
        input_list: list[Tensor],
        *,
        group: Any | None = None,
        op: str = "sum",
    ) -> Tensor:
        dist.reduce_scatter(
            output, input_list, op=reduce_op_from_string(op), group=group
        )
        return output

    def all_gather(
        self,
        tensor_list: list[Tensor],
        tensor: Tensor,
        *,
        group: Any | None = None,
    ) -> list[Tensor]:
        dist.all_gather(tensor_list, tensor, group=group)
        return tensor_list

    def send(self, tensor: Tensor, dst: int, *, group: Any | None = None) -> None:
        dist.send(tensor, dst, group=group)

    def recv(self, tensor: Tensor, src: int, *, group: Any | None = None) -> Tensor:
        dist.recv(tensor, src, group=group)
        return tensor

    def barrier(self, *, group: Any | None = None) -> None:
        dist.barrier(group=group)
```

Export from `distributed/__init__.py`: `CommBackend`, `TorchDistBackend`, `reduce_op_from_string`.

- [ ] **Step 4: Tests pass**

- [ ] **Step 5: Commit**

```bash
git add nano_megatron/distributed tests/unit/distributed
git commit -m "feat(distributed): add CommBackend and TorchDistBackend"
```

---

### Task 4: ParallelContext initialize/destroy (single-process + NCCL multiproc)

**Files:**
- Create: `nano_megatron/parallel/context.py`
- Modify: `nano_megatron/parallel/__init__.py`
- Create: `tests/distributed/common.py`
- Create: `tests/distributed/test_parallel_context_nccl.py`
- Create: `tests/unit/parallel/test_context_single.py` (world_size=1 path without multi-GPU)

**Interfaces:**
- Produces:
```python
@dataclass
class ParallelContext:
    config: ParallelConfig
    rank: int
    world_size: int
    local_rank: int
    backend: CommBackend
    tensor_parallel_size: int
    tensor_parallel_rank: int
    data_parallel_size: int
    data_parallel_rank: int
    pipeline_parallel_size: int
    pipeline_parallel_rank: int
    context_parallel_size: int
    context_parallel_rank: int
    tensor_parallel_group: Any
    data_parallel_group: Any
    pipeline_parallel_group: Any
    context_parallel_group: Any
    data_context_parallel_group: Any

def initialize_parallel(
    config: ParallelConfig | None = None,
    *,
    backend: CommBackend | None = None,
    init_method: str | None = None,
    dist_backend: str | None = None,
    rank: int | None = None,
    world_size: int | None = None,
) -> ParallelContext: ...

def get_parallel_context() -> ParallelContext: ...
def destroy_parallel() -> None: ...
def is_parallel_initialized() -> bool: ...
```

Behavior:
1. Default `config = ParallelConfig()` (all 1s).
2. If not `dist.is_initialized()`: call `dist.init_process_group(backend=dist_backend or auto, init_method=init_method or env, rank=..., world_size=...)`. Auto: `nccl` if CUDA else `gloo`.
3. Validate config against `dist.get_world_size()`.
4. Build `RankGenerator` with resolved sizes; create process groups via `dist.new_group(ranks)` for each unique rank list from `get_ranks("tp"|"dp"|"pp"|"cp"|"dp-cp")`. Store the group that contains this rank.
5. Set `torch.cuda.set_device(local_rank)` when CUDA and nccl.
6. `local_rank` from `LOCAL_RANK` env or `rank % device_count` fallback when CUDA.
7. Global `_PARALLEL_CONTEXT`; double-init raises `RuntimeError`.
8. `destroy_parallel`: clear global; call `dist.destroy_process_group()` only if this module initialized dist (track flag). Do not leak context after destroy.

- [ ] **Step 1: Write single-process unit test**

```python
# tests/unit/parallel/test_context_single.py
import os
import torch.distributed as dist
from nano_megatron.parallel import (
    ParallelConfig,
    destroy_parallel,
    get_parallel_context,
    initialize_parallel,
    is_parallel_initialized,
)


def test_initialize_single_process(tmp_path, monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29501")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    if is_parallel_initialized():
        destroy_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
    ctx = initialize_parallel(ParallelConfig(), dist_backend="gloo")
    assert ctx.world_size == 1
    assert ctx.tensor_parallel_rank == 0
    assert ctx.data_parallel_size == 1
    assert get_parallel_context() is ctx
    destroy_parallel()
    assert not is_parallel_initialized()
```

- [ ] **Step 2: Write NCCL multiproc test module**

```python
# tests/distributed/common.py
from __future__ import annotations
import os
import pytest
import torch


def require_nccl_gpus(n: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < n:
        pytest.skip(f"need {n} CUDA devices for NCCL test")
    if not torch.distributed.is_nccl_available():
        pytest.skip("NCCL not available")


def env_rank_info() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world, local
```

```python
# tests/distributed/test_parallel_context_nccl.py
"""Run with:
torchrun --standalone --nproc_per_node=4 -m pytest tests/distributed/test_parallel_context_nccl.py -v
or the helper invocation in the test file via subprocess from a parent test.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from tests.distributed.common import require_nccl_gpus


REPO = Path(__file__).resolve().parents[2]


def _run_torchrun(nproc: int, test_id: str) -> None:
    require_nccl_gpus(nproc)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "-m",
        "pytest",
        f"tests/distributed/test_parallel_context_nccl.py::{test_id}",
        "-v",
        "-s",
    ]
    env = os.environ.copy()
    env["NANO_MEGATRON_NCCL_WORKER"] = "1"
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


@pytest.mark.skipif(os.environ.get("NANO_MEGATRON_NCCL_WORKER") == "1", reason="launcher only")
def test_launch_tp2_dp2_ranks():
    _run_torchrun(4, "test_worker_tp2_dp2_ranks")


@pytest.mark.skipif(os.environ.get("NANO_MEGATRON_NCCL_WORKER") != "1", reason="worker only")
def test_worker_tp2_dp2_ranks():
    from nano_megatron.parallel import (
        ParallelConfig,
        destroy_parallel,
        initialize_parallel,
    )
    from nano_megatron.distributed import TorchDistBackend

    if dist.is_initialized():
        # torchrun already inited? our initialize may reuse
        pass
    cfg = ParallelConfig(tensor_parallel_size=2, data_parallel_size=2)
    ctx = initialize_parallel(cfg, backend=TorchDistBackend(), dist_backend="nccl")
    assert ctx.world_size == 4
    assert ctx.tensor_parallel_size == 2
    assert ctx.data_parallel_size == 2
    assert ctx.pipeline_parallel_size == 1
    # all_reduce on DP group
    t = torch.ones(1, device="cuda") * (ctx.rank + 1)
    ctx.backend.all_reduce(t, group=ctx.data_parallel_group)
    # each DP group has 2 ranks: ranks (0,2), (1,3) for order tp-cp-dp-pp tp=2 dp=2
    # rank0 dp with rank2: sum = (0+1)+(2+1)=4
    # rank1 dp with rank3: sum = (1+1)+(3+1)=6
    expected = {0: 4.0, 1: 6.0, 2: 4.0, 3: 6.0}[ctx.rank]
    assert t.item() == expected
    destroy_parallel()


@pytest.mark.skipif(os.environ.get("NANO_MEGATRON_NCCL_WORKER") == "1", reason="launcher only")
def test_launch_all_reduce_tp():
    _run_torchrun(2, "test_worker_all_reduce_tp")


@pytest.mark.skipif(os.environ.get("NANO_MEGATRON_NCCL_WORKER") != "1", reason="worker only")
def test_worker_all_reduce_tp():
    from nano_megatron.parallel import ParallelConfig, destroy_parallel, initialize_parallel

    cfg = ParallelConfig(tensor_parallel_size=2)
    ctx = initialize_parallel(cfg, dist_backend="nccl")
    x = torch.tensor([float(ctx.tensor_parallel_rank + 1)], device="cuda")
    ctx.backend.all_reduce(x, group=ctx.tensor_parallel_group)
    assert x.item() == 3.0  # 1+2
    destroy_parallel()
```

Note: implementer must fix rank→DP pairing assertions if RankGenerator golden differs; compute expected from `ctx` groups rather than hardcoding wrong pairs if needed. Prefer:

```python
# more robust expectation
group_ranks = ...  # optional helper on context
```

Or derive expected sum as sum of `(r+1)` for r in the DP group containing this rank. Add `ParallelContext` optional helper or use generator:

```python
rg_ranks = RankGenerator(...).get_ranks("dp")
my_group = next(g for g in rg_ranks if ctx.rank in g)
expected = float(sum(r + 1 for r in my_group))
```

- [ ] **Step 2b: Run single-process test — FAIL then implement**

- [ ] **Step 3: Implement context.py** fully per interfaces above

Critical implementation sketch:

```python
_PARALLEL_CONTEXT: ParallelContext | None = None
_DIST_INITIALIZED_BY_US = False


def initialize_parallel(...):
    global _PARALLEL_CONTEXT, _DIST_INITIALIZED_BY_US
    if _PARALLEL_CONTEXT is not None:
        raise RuntimeError("parallel context already initialized")
    cfg = config or ParallelConfig()
    if not dist.is_initialized():
        backend_name = dist_backend or ("nccl" if torch.cuda.is_available() else "gloo")
        # resolve rank/world from args or env
        dist.init_process_group(backend=backend_name, init_method=init_method, ...)
        _DIST_INITIALIZED_BY_US = True
    world = dist.get_world_size()
    rank = dist.get_rank()
    cfg.validate(world)
    dp = cfg.resolved_data_parallel_size(world)
    rg = RankGenerator(tp=cfg.tensor_parallel_size, dp=dp, pp=cfg.pipeline_parallel_size, cp=cfg.context_parallel_size, order=cfg.order)
    parts = rg.decode(rank)
    def _group_for(token: str):
        groups = rg.get_ranks(token)
        for ranks in groups:
            g = dist.new_group(ranks=ranks)
            if rank in ranks:
                mine = g
        return mine
    # create ALL groups on ALL ranks (required by dist.new_group)
    tp_group = _group_for("tp")
    ...
    backend = backend or TorchDistBackend()
    ctx = ParallelContext(...)
    _PARALLEL_CONTEXT = ctx
    return ctx
```

Important: every rank must call `new_group` with the same rank lists in the same order.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/unit/parallel/test_context_single.py -v
python -m pytest tests/distributed/test_parallel_context_nccl.py -v --tb=short
```

Expected: single-process pass; NCCL launcher tests pass on ≥2/4 GPUs.

- [ ] **Step 5: Commit**

```bash
git add nano_megatron/parallel tests/unit/parallel/test_context_single.py tests/distributed
git commit -m "feat(parallel): add ParallelContext with NCCL process groups"
```

---

### Task 5: Public API polish + README blurb + full suite

**Files:**
- Modify: `nano_megatron/parallel/__init__.py`
- Modify: `nano_megatron/distributed/__init__.py`
- Modify: `README.md` and `README_zh.md` (short section only)
- Modify: `pyproject.toml` if markers needed

**Interfaces (final public):**

```python
from nano_megatron.parallel import (
    ParallelConfig,
    ParallelContext,
    RankGenerator,
    destroy_parallel,
    get_parallel_context,
    initialize_parallel,
    is_parallel_initialized,
)
from nano_megatron.distributed import CommBackend, TorchDistBackend
```

- [ ] **Step 1: Ensure exports complete; add README section** describing initialize pattern and NCCL test command

- [ ] **Step 2: Run full relevant suite**

```bash
python -m pytest tests/unit/parallel tests/unit/distributed tests/distributed -v --tb=short
python -m pytest tests/unit/reference -v  # no regressions
```

- [ ] **Step 3: Commit**

```bash
git add README.md README_zh.md nano_megatron pyproject.toml
git commit -m "docs: document parallel context and comm backend API"
```

---

## Self-review checklist

1. Spec coverage: RankGenerator, ParallelConfig, ParallelContext, CommBackend, TorchDistBackend, NCCL tests — all tasked
2. No placeholders
3. Type names consistent across tasks (`data_context_parallel_group`, order `tp-cp-dp-pp`)
4. Group creation collective participation documented
