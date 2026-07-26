# nano-megatron

Compact distributed training framework for studying Megatron-style parallelism.

中文文档: [README_zh.md](README_zh.md)

## Performance

Benchmark results comparing nano-megatron with Megatron-LM:

| TP Size | nano-megatron | Megatron-LM | Ratio |
|---------|---------------|-------------|-------|
| TP2 | 27,991 tokens/sec | 39,078 tokens/sec | 0.72x |
| TP4 | 29,502 tokens/sec | 39,010 tokens/sec | 0.76x |

Full benchmark details: [performance.md](performance.md)

## Quick Start

### Installation

```bash
pip install -e ".[dev]"
```

### Run Reference Model

```bash
python scripts/run_reference_gpt.py --seed 0 --steps 3 --device cpu --out ref_traj.pt
```

### Run TP Benchmark

```bash
# TP2
torchrun --standalone --nproc_per_node=2 scripts/benchmark_tp.py \
    --framework both --tp-size 2 --batch-size 2 --seq-len 1024 \
    --hidden-size 512 --num-layers 12 --num-heads 8

# TP4
torchrun --standalone --nproc_per_node=4 scripts/benchmark_tp.py \
    --framework both --tp-size 4 --batch-size 2 --seq-len 1024 \
    --hidden-size 512 --num-layers 12 --num-heads 8
```

### Verify Architecture

```bash
python scripts/verify_architecture.py
```

### Run Tests

```bash
# Unit tests
PYTHONPATH=. python -m pytest tests/unit -v

# Distributed tests (requires NCCL)
PYTHONPATH=. python -m pytest tests/distributed -v
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
