# nano-megatron

Compact distributed training framework for studying Megatron-style parallelism.

中文文档: [README_zh.md](README_zh.md)

## Performance

On 4× RTX A6000 (FP32), nano-megatron reaches **0.93x–1.01x** of Megatron-LM throughput under matching GPT configs (345M / 760M / 1.3B, TP2 & TP4, with and without sequence parallel).

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

### Run TP Benchmark (vs Megatron-LM)

Requires Megatron-LM on `PYTHONPATH`. Example: GPT-3 345M, TP2 / TP4.

```bash
export PYTHONPATH=/path/to/nano-megatron:/path/to/Megatron-LM:$PYTHONPATH

# TP2
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_tp.py --framework both --tp-size 2 \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096

# TP4
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/benchmark_tp.py --framework both --tp-size 4 \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096
```

More model sizes (760M, 1.3B) and flags: [performance.md](performance.md).

### Verify Architecture

```bash
python scripts/verify_architecture.py
```

### Run Tests

```bash
# Unit tests
PYTHONPATH=. python -m pytest tests/unit -v

# Distributed tests (requires multi-GPU / NCCL for some cases)
PYTHONPATH=. python -m pytest tests/distributed tests/integration -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
