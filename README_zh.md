# nano-megatron

面向研究与复现的紧凑分布式训练框架，覆盖 Megatron 风格大模型并行技术。

English README: [README.md](README.md)

## 性能

在 4× RTX A6000（FP32）上，相同 GPT 配置（345M / 760M / 1.3B，TP2 与 TP4）下，nano-megatron 吞吐约为 Megatron-LM 的 **0.96x–1.04x**。

完整表格、模型配置与复现命令见：**[performance.md](performance.md)**

## 快速开始

### 安装

```bash
pip install -e ".[dev]"
```

### 运行参考模型

```bash
python scripts/run_reference_gpt.py --seed 0 --steps 3 --device cpu --out ref_traj.pt
```

### 运行 TP 基准测试（对比 Megatron-LM）

需将 Megatron-LM 加入 `PYTHONPATH`。示例：GPT-3 345M，TP2 / TP4。

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

更多模型规模（760M、1.3B）与参数见 [performance.md](performance.md)。

### 验证模型结构

```bash
python scripts/verify_architecture.py
```

### 运行测试

```bash
# 单元测试
PYTHONPATH=. python -m pytest tests/unit -v

# 分布式测试（部分用例需要多卡 / NCCL）
PYTHONPATH=. python -m pytest tests/distributed tests/integration -v
```

## 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。
