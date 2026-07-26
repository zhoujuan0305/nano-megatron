# Parallel Context and Communication Abstraction Design

**Date:** 2026-07-25  
**Status:** Approved (approach A; user waived section-by-section review)  
**Scope:** Minimal skeleton only

## Goal

Provide a process-wide `ParallelContext` that owns DP/TP/PP/CP topology and process groups, plus a small `CommBackend` interface with a PyTorch distributed implementation, so later TP/DP/ZeRO/PP code never calls `torch.distributed` directly.

## Non-goals

- Tensor parallel mappings / model sharding
- DDP, ZeRO, pipeline schedules
- Async collectives / communication-computation overlap
- Expert parallelism (EP)
- nano-nccl backend (interface must allow it later)

## Architecture

```text
model / optimizer / schedules
        ↓
  ParallelContext  (ranks, sizes, process groups)
        ↓
  CommBackend      (all_reduce, reduce_scatter, all_gather, send, recv, barrier)
        ↓
  TorchDistBackend → torch.distributed (NCCL/gloo)
```

### Package layout

```text
nano_megatron/
  parallel/
    __init__.py          # public parallel API
    config.py            # ParallelConfig
    rank_generator.py    # Megatron-compatible orthogonal rank groups
    context.py           # ParallelContext + initialize/destroy/get
  distributed/
    __init__.py          # public distributed API
    backend.py           # CommBackend protocol + ProcessGroup type alias
    torch_backend.py     # TorchDistBackend
```

### ParallelConfig

Dataclass fields (all ints ≥ 1):

- `tensor_parallel_size` (tp)
- `pipeline_parallel_size` (pp)
- `context_parallel_size` (cp)
- `data_parallel_size` (dp) — optional; if `None`, inferred as  
  `world_size // (tp * pp * cp)`
- `order: str = "tp-cp-dp-pp"` — Megatron-compatible default (no EP)

Invariants:

- `world_size == tp * cp * dp * pp`
- sequence parallel reuses TP group (flag may exist later; not required now)

### Rank layout

Match Megatron-LM `RankGenerator` with default order `tp-cp-dp-pp` (fastest → slowest):

```text
global_rank = tp + tp_size * (cp + cp_size * (dp + dp_size * pp))
```

Process groups created (NCCL primary):

| Group key | Meaning |
|-----------|---------|
| `tp` | tensor parallel |
| `cp` | context parallel |
| `dp` | data parallel (without CP) |
| `pp` | pipeline parallel |
| `dp_cp` | data + context (for later grad sync / ZeRO) |

Each group is a `torch.distributed.ProcessGroup` (or backend-opaque handle).  
`ParallelContext` stores the **local** group for this rank plus size/rank for each dimension.

### ParallelContext API

```python
@dataclass
class ParallelContext:
    config: ParallelConfig
    rank: int
    world_size: int
    backend: CommBackend

    # per-dimension size / rank
    tensor_parallel_size: int
    tensor_parallel_rank: int
    ...

    # local process groups
    tensor_parallel_group: Any
    data_parallel_group: Any
    pipeline_parallel_group: Any
    context_parallel_group: Any
    data_context_parallel_group: Any  # dp-cp

def initialize_parallel(
    config: ParallelConfig,
    *,
    backend: CommBackend | None = None,
    init_method: str | None = None,
    dist_backend: str = "nccl",
) -> ParallelContext: ...

def get_parallel_context() -> ParallelContext: ...

def destroy_parallel() -> None: ...

def is_parallel_initialized() -> bool: ...
```

Rules:

- Only one process-wide context; second `initialize_parallel` raises.
- `destroy_parallel` clears globals and destroys process groups when safe.
- If `torch.distributed` is not yet initialized, `initialize_parallel` initializes it (env:// or provided `init_method`).
- If already initialized, reuse world and validate `world_size` vs config.
- Caller/model code queries context; does not create groups.

### CommBackend

```python
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

- Sync only; ops complete before return.
- `op` is a string (`"sum"`, `"max"`, `"min"`, `"product"`) mapped inside `TorchDistBackend`.
- In-place on the provided tensors where PyTorch collectives are in-place; return the same tensor for chaining.
- No CUDA stream arguments in v1.

### TorchDistBackend

Wraps `torch.distributed` collectives. Default when `backend=None` in `initialize_parallel`.

## Testing

1. **Unit (no GPU):** `RankGenerator` / rank decode / group membership tables vs golden vectors (Megatron order `tp-cp-dp-pp`).
2. **Unit:** `ParallelConfig` validation errors.
3. **Distributed NCCL (primary):** `torchrun --nproc_per_node=2|4` tests:
   - initialize context with various (tp, dp, pp, cp) factorizations of world_size
   - ranks and group sizes correct
   - `all_reduce` on DP group yields world-consistent sums
   - `all_gather` / `reduce_scatter` on TP group round-trip
   - destroy + re-init once
4. Skip NCCL tests when `torch.cuda.device_count() < world_size` or NCCL unavailable.

## Success criteria

- Public APIs importable from `nano_megatron.parallel` and `nano_megatron.distributed`
- Topology matches Megatron default order for documented golden cases
- NCCL multi-GPU tests pass on this machine (4× A6000)
- No model/optimizer/mapping code in this change
