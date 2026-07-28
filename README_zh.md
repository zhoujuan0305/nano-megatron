# nano-megatron

面向研究与复现的紧凑分布式训练框架，覆盖 Megatron 风格大模型并行技术。

English README: [README.md](README.md)

## 功能

| 并行 | 状态 | 说明 |
|------|------|------|
| 数据并行 (DP) | 已支持 | 自定义 DDP + 梯度 bucket |
| 张量并行 (TP) | 已支持 | 列/行并行 + vocab 并行 |
| 序列并行 (SP) | 已支持 | 复用 TP process group |
| 流水线并行 (PP) | 已支持 | 非交错 1F1B，同步 P2P |
| TP×DP / TP×PP / DP×PP / TP×DP×PP | 已支持 | 通过 `ParallelContext` 组合 |
| 上下文并行 (CP) / ZeRO | 规划中 | 拓扑已在 parallel config 预留 |

可直接使用 PyTorch tensor、autograd、CUDA 与分布式通信原语。通信经小型 `CommBackend` 抽象（默认包装 PyTorch distributed）。

## 性能

在 4× RTX A6000（FP32）、相同 GPT 配置（345M / 760M / 1.3B）下：

| 模式 | nano / Megatron 吞吐比 |
|------|------------------------|
| TP / TP+SP | **0.93x – 1.01x** |
| DP / TP×DP | **0.97x – 1.05x** |
| PP | **1.00x – 1.04x** |
| TP×PP | **约 0.92x** |

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

### 基准测试（对比 Megatron-LM）

需将 Megatron-LM 加入 `PYTHONPATH`。为公平对比峰值显存，建议 DP/PP 分别用 `--framework nano` 与 `--framework megatron` 各跑一次。

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

# PP2（1F1B）
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_pp.py --framework nano --pp-size 2 --tp-size 1 \
  --batch-size 8 --num-microbatches 4 --seq-len 1024 \
  --hidden-size 1024 --num-layers 24 --num-heads 16 --ffn-hidden-size 4096
```

更多规模（760M、1.3B）、TP+SP、TP×DP、PP4、TP×PP 见 [performance.md](performance.md)。

### 验证模型结构

```bash
python scripts/verify_architecture.py
```

### 运行测试

```bash
# 单元测试
PYTHONPATH=. python -m pytest tests/unit -v

# 分布式测试（多进程；部分用例需要多卡 / NCCL）
PYTHONPATH=. python -m pytest tests/distributed tests/integration -v
```

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。
