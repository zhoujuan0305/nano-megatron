# nano-megatron

面向研究与复现的紧凑分布式训练框架，覆盖 Megatron 风格大模型并行技术。

English README: [README.md](README.md)

## 性能

nano-megatron 与 Megatron-LM 的基准测试结果：

| TP Size | nano-megatron | Megatron-LM | Ratio |
|---------|---------------|-------------|-------|
| TP2 | 27,991 tokens/sec | 39,078 tokens/sec | 0.72x |
| TP4 | 29,502 tokens/sec | 39,010 tokens/sec | 0.76x |

完整基准测试详情：[performance.md](performance.md)

## 快速开始

### 安装

```bash
pip install -e ".[dev]"
```

### 运行参考模型

```bash
python scripts/run_reference_gpt.py --seed 0 --steps 3 --device cpu --out ref_traj.pt
```

### 运行 TP 基准测试

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

### 验证模型结构

```bash
python scripts/verify_architecture.py
```

### 运行测试

```bash
# 单元测试
PYTHONPATH=. python -m pytest tests/unit -v

# 分布式测试（需要 NCCL）
PYTHONPATH=. python -m pytest tests/distributed -v
```

## 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。
