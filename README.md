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
| Context Parallel (CP) | Supported | All-gather KV; no PP/SP combo; wrap with DDP when `cp>1` |
| TP × DP / TP × PP / DP × PP / TP × DP × PP | Supported | Composable via `ParallelContext` |
| TP × CP / CP × DP | Supported | Composable via `ParallelContext` |
| ZeRO | Planned | — |

PyTorch tensors, autograd, CUDA, and distributed collectives are used directly. Communication goes through a small `CommBackend` abstraction (default: PyTorch distributed).

## Performance

On 4× RTX A6000, under matching GPT configs (345M / 760M / 1.3B):

| Mode | Precision | nano / Megatron throughput |
|------|-----------|----------------------------|
| TP / TP+SP | FP32 | **0.93x – 1.01x** |
| DP / TP×DP | FP32 | **0.97x – 1.05x** |
| PP | FP32 | **1.00x – 1.04x** |
| TP×PP | FP32 | **~0.92x** |
| CP / CP×TP / CP×DP | BF16 | **0.49x – 0.85x** |

CP uses BF16 on both sides (Megatron TE FlashAttention CP requires half precision). nano CP is contiguous all-gather KV; Megatron uses zigzag + TE kernels.

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

Requires Megatron-LM on `PYTHONPATH`. Prefer separate `--framework nano` / `--framework megatron` runs for fair peak memory (DP/PP/CP).

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

# CP2 (BF16; seq_len % (2*cp) == 0)
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_cp.py --framework nano --cp-size 2 --precision bf16 \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096
```

More sizes (760M, 1.3B), TP+SP, TP×DP, PP4, TP×PP, CP4, TP×CP, CP×DP: [performance.md](performance.md).

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
