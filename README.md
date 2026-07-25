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

## Setup and tests

```bash
python -m pip install -e .
python -m pytest tests/unit/reference -v
```
