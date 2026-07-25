# AGENTS.md

## Project Positioning

`nano-megatron` is a compact distributed training framework for studying and reproducing the core parallelism techniques used by Megatron-style large language model training systems.

The project plans to support:

* Data Parallelism (DP)
* Tensor Parallelism (TP)
* Sequence Parallelism (SP)
* Pipeline Parallelism (PP)
* Context Parallelism (CP)
* ZeRO-1, ZeRO-2, and ZeRO-3

The project has two long-term goals:

1. provide small, readable, and independently implemented versions of these techniques;
2. approach Megatron throughput under the same model configuration, hardware, software stack, precision, kernels, and communication settings.

PyTorch tensors, autograd, CUDA, and distributed communication primitives may be used directly. PyTorch DDP, FSDP, and related distributed implementations may be used as correctness or performance references, but they should not replace the main implementation of the corresponding nano-megatron feature.

A future communication backend may integrate [`nano-nccl`](https://github.com/zhoujuan0305/nano-nccl). Model, optimizer, and parallelism code must therefore avoid depending directly on a single communication backend.

This project is in an early stage. Prefer correctness, clarity, and testability over broad feature coverage. Do not claim Megatron-equivalent correctness, scalability, or throughput until supported by reproducible results.

## Sensitive Information

Do not write hostnames, IP addresses, physical interface names, GPU UUIDs, absolute user home paths, credentials, tokens, private dataset paths, or internal cluster configuration into files that may be committed to Git.

Use placeholders such as `<host-a>`, `<interface>`, `<dataset-path>`, `<checkpoint-path>`, and `<megatron-path>`.

## Directory Structure

The initial directory layout is expected to follow this structure:

```text
nano-megatron/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── nano_megatron/
│   ├── model/                    # GPT and Transformer components
│   ├── reference/                # simple correctness-first implementations
│   ├── parallel/                 # parallel state, mappings, and parallel operators
│   ├── distributed/              # DDP, buffers, buckets, communication backends
│   ├── optimizer/                # optimizer utilities and ZeRO implementations
│   ├── schedules/                # microbatch and pipeline schedules
│   ├── training/                 # training loop, checkpointing, and metrics
│   └── utils/                    # shared utilities
├── tests/
│   ├── unit/
│   ├── model/
│   ├── distributed/
│   └── training/
├── benchmarks/
└── scripts/
```

Directory responsibilities:

* `model/`: model definitions and Transformer components
* `reference/`: readable implementations used as numerical references
* `parallel/`: DP/TP/SP/PP/CP topology, mappings, and operators
* `distributed/`: gradient synchronization, communication abstraction, buffers, and buckets
* `optimizer/`: optimizer logic and ZeRO state partitioning
* `schedules/`: microbatch and pipeline execution ordering
* `training/`: end-to-end training orchestration
* `tests/`: correctness and regression coverage
* `benchmarks/`: reproducible performance comparisons

The structure may evolve as the implementation matures. Keep ownership clear and avoid premature abstraction.

## Coding Standards

* Use 4 spaces for indentation; do not use tabs.
* Use `snake_case` for files, functions, methods, and variables.
* Use `PascalCase` for classes and dataclasses.
* Use `UPPER_SNAKE_CASE` for module-level constants.
* Add type annotations to public functions, methods, and important interfaces.
* Prefer dataclasses for configuration and small runtime-state descriptions.
* Keep functions focused and make tensor, buffer, process-group, and CUDA-stream ownership explicit.
* Validate important shapes, dtypes, devices, ranks, and group sizes near public boundaries.
* Raise clear exceptions that include relevant rank, group, tensor, or configuration context.
* Comments should explain why a design or synchronization step is needed, not restate obvious code.
* Avoid hidden global state except for an explicitly initialized process-wide parallel context.
* Do not silently move tensors between devices or silently change dtypes in performance-critical code.
* Do not introduce blocking waits or device-wide synchronization unless correctness requires it or the operation defines a benchmark boundary.
* Prefer readable PyTorch implementations before adding custom CUDA, fused kernels, custom autograd functions, or communication overlap.
* Avoid copying large sections of Megatron source code. Reimplement the underlying concepts in a smaller form and document important compatibility decisions.

## Architecture

### Layered design

Keep dependencies flowing in one direction:

```text
training loop
    ↓
model and execution schedules
    ↓
parallel operators and distributed optimizer
    ↓
communication backend abstraction
    ↓
PyTorch distributed / NCCL / nano-nccl
```

Lower layers must not depend on model-specific training code.

### Parallel context

Distributed topology must be created and owned by a central parallel-state component. Model layers and optimizers should query this context instead of creating process groups independently.

The initial topology follows:

```text
world_size = data_parallel_size
           × tensor_parallel_size
           × pipeline_parallel_size
           × context_parallel_size
```

Sequence parallelism reuses the tensor-parallel group and is not an independent world-size dimension.

### Communication abstraction

Communication must be accessed through a small backend interface instead of scattered direct calls to `torch.distributed`.

The initial backend may wrap PyTorch distributed collectives. A later backend may use `nano-nccl` without requiring changes to model, DDP, or ZeRO logic.

The abstraction is expected to cover operations such as:

```text
all_reduce
reduce_scatter
all_gather
send
recv
barrier
```

Asynchronous operations must expose an explicit completion boundary. CUDA stream and synchronization behavior must be clear at the interface.

### Reference and optimized paths

Keep correctness-first implementations separate from optimized implementations.

Reference code should use straightforward PyTorch operations and prioritize mathematical clarity. Optimized code may introduce tensor partitioning, flattened buffers, custom autograd logic, fused operations, and communication-computation overlap, but it must remain testable against the reference path.

### Incremental development

Introduce features in small stages. A feature should first work independently before being composed with additional parallel dimensions.

The intended high-level order is:

```text
single-device GPT reference
→ parallel context and communication abstraction
→ TP and SP
→ custom DP
→ ZeRO-1 and ZeRO-2
→ PP and CP
→ ZeRO-3
→ communication overlap and performance optimization
```

## Early-Stage Development Principles

* Correctness takes priority over throughput.
* Add focused tests with each feature.
* Keep changes small enough to review and debug.
* Do not implement several parallel strategies in one change.
* Prefer a small working interface over a large speculative framework.
* Do not weaken numerical tolerances only to make a test pass.
* Keep benchmark configurations reproducible and separate correctness runs from performance runs.

