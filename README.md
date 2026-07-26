# nano-megatron

Compact distributed training framework studying Megatron-style parallelism.

中文文档见 [README_zh.md](README_zh.md).

## ReferenceGPT (numerical oracle)

`nano_megatron.reference` is a single-device FP32 Megatron-style GPT used as a
**numerical oracle** for later parallel and optimized implementations.

Later DP/TP/PP/ZeRO paths should match this reference under the same seed,
config, and synthetic inputs: loss, logits, grads, optimizer state, and
parameter trajectories (strict CPU FP32 equality or agreed tolerances).

### Public API

```python
from nano_megatron.reference import (
    AdamW,
    CaptureLevel,
    ReferenceGPT,
    ReferenceGPTConfig,
    StepResult,
    reference_train_loop,
    reference_train_step,
    seed_all,
    shifted_cross_entropy,
    snapshot_grads,
    snapshot_optimizer,
    snapshot_params,
)
```

### Dump a trajectory

```bash
python scripts/run_reference_gpt.py \
  --seed 0 --steps 3 --device cpu --out ref_traj.pt
```

The file is a `list[dict]` (one entry per captured step). Keys: `step` (int),
and CPU tensors `loss`, `logits`, `params`, `grads`, `activations`,
`optimizer_state`.

### Compare a candidate implementation

1. Fix `seed_all(seed)` and the same `ReferenceGPTConfig` / batch `input_ids`.
2. Run the reference loop (or load a saved trajectory from the CLI).
3. Run the candidate under identical inputs and optimizer hyperparameters.
4. Assert equality (or tight tolerances) on loss, logits, grads, opt state, and params.

Do not loosen numerical tolerances only to make a test pass.

## Parallel context and CommBackend

`nano_megatron.parallel` owns process-wide topology (TP/DP/PP/CP groups).
`nano_megatron.distributed` exposes a small communication backend so model code
does not call `torch.distributed` directly. Default backend is
`TorchDistBackend` (PyTorch collectives); a future path may plug in `nano-nccl`.

### Public API

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

### Initialize pattern

```python
from nano_megatron.distributed import TorchDistBackend
from nano_megatron.parallel import (
    ParallelConfig,
    destroy_parallel,
    initialize_parallel,
)

# world_size must equal tp * cp * dp * pp (dp may be inferred from world_size).
# Rank order default: tp-cp-dp-pp. Every rank participates in every new_group call.
cfg = ParallelConfig(tensor_parallel_size=2, data_parallel_size=2)
ctx = initialize_parallel(cfg, backend=TorchDistBackend())
# ctx.tensor_parallel_group, ctx.data_parallel_group, ...
# ctx.backend.all_reduce(tensor, group=ctx.data_parallel_group)
destroy_parallel()
```

Single-process (CPU/gloo) init is enough for unit tests. Multi-GPU NCCL needs
`torchrun` / env ranks as usual.

### Setup and tests

```bash
python -m pip install -e ".[dev]"
PYTHONPATH=. python -m pytest tests/unit/reference -v
PYTHONPATH=. python -m pytest tests/unit/parallel tests/unit/distributed -v
```

NCCL multi-GPU (needs ≥4 CUDA devices; control plane on loopback):

```bash
PYTHONPATH=. python -m pytest tests/distributed -v --tb=short
# or directly:
torchrun --standalone --nproc_per_node=4 --master_addr=127.0.0.1 \
  -m pytest tests/distributed/test_parallel_context_nccl.py -v
```
