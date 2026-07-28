# nano-megatron

Compact distributed training framework for studying Megatron-style parallelism.

中文文档: [README_zh.md](README_zh.md)

## Features

| Parallelism | Status | Notes |
|-------------|--------|--------|
| Data Parallel (DP) | Supported | Custom DDP + gradient buckets |
| Tensor Parallel (TP) | Supported | Column/row parallel + vocab parallel |
| Sequence Parallel (SP) | Supported | Reuses the TP group |
| Pipeline Parallel (PP) | Supported | Non-interleaved 1F1B, sync P2P |
| TP × DP / TP × PP / DP × PP / TP × DP × PP | Supported | Composable via `ParallelContext` |
| Context Parallel (CP) / ZeRO | Planned | Topology reserved in parallel config |

PyTorch tensors, autograd, CUDA, and distributed collectives are used directly. Communication goes through a small `CommBackend` abstraction (default: PyTorch distributed).

## Performance

On 4× RTX A6000 (FP32), under matching GPT configs (345M / 760M / 1.3B):

| Mode | nano / Megatron throughput |
|------|----------------------------|
| TP / TP+SP | **0.93x – 1.01x** |
| DP / TP×DP | **0.97x – 1.05x** |
| PP | **1.00x – 1.04x** |
| TP×PP | **~0.92x** |

Full tables, configs, and reproduction commands: **[performance.md](performance.md)**

## Quick Start

### Installation

```bash
pip install -e ".[dev]"
```

### Run Reference Model

```bash
python scripts/run_reference_gpt.py --seed 0 --steps 3 --device cpu --out ref_traj.pt
```

### Benchmarks (vs Megatron-LM)

Requires Megatron-LM on `PYTHONPATH`. Prefer separate `--framework nano` / `--framework megatron` runs for fair peak memory (especially DP/PP).

```bash
export PYTHONPATH=/path/to/nano-megatron:/path/to/Megatron-LM:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

# TP2 — GPT-3 345M
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_tp.py --framework both --tp-size 2 \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096

# DP2
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_dp.py --framework nano --tp-size 1 --dp-size 2 \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096

# PP2 (1F1B)
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_pp.py --framework nano --pp-size 2 --tp-size 1 \
  --batch-size 8 --num-microbatches 4 --seq-len 1024 \
  --hidden-size 1024 --num-layers 24 --num-heads 16 --ffn-hidden-size 4096
```

More sizes (760M, 1.3B), TP+SP, TP×DP, PP4, TP×PP: [performance.md](performance.md).

### Verify Architecture

```bash
python scripts/verify_architecture.py
```

### Run Tests

```bash
# Unit tests
PYTHONPATH=. python -m pytest tests/unit -v

# Distributed tests (multi-process; some need multi-GPU / NCCL)
PYTHONPATH=. python -m pytest tests/distributed tests/integration -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
