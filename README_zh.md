# nano-megatron

面向研究与复现的紧凑分布式训练框架，覆盖 Megatron 风格大模型并行技术。

English README: [README.md](README.md)

## 功能

| 并行 / 能力 | 状态 | 说明 |
|-------------|------|------|
| 数据并行 (DP) | 已支持 | 自定义 DDP + 梯度 bucket |
| 张量并行 (TP) | 已支持 | 列/行并行 + vocab 并行 |
| 序列并行 (SP) | 已支持 | 复用 TP process group |
| 流水线并行 (PP) | 已支持 | 非交错 1F1B，同步 P2P |
| 上下文并行 (CP) | 已支持 | 连续 AG-KV；FA 路径为 AG + 分块 flash（非 P2P ring）；不与 PP/SP 组合；`cp>1` 时包 DDP |
| FlashAttention | 已支持 | 可选 `flash-attn`；`attn_backend=auto\|flash\|unfused`；覆盖 TP + CP |
| TP×DP / TP×PP / DP×PP / TP×DP×PP | 已支持 | 通过 `ParallelContext` 组合 |
| TP×CP / CP×DP | 已支持 | 通过 `ParallelContext` 组合 |
| ZeRO | 规划中 | — |

可直接使用 PyTorch tensor、autograd、CUDA 与分布式通信原语。通信经小型 `CommBackend` 抽象（默认包装 PyTorch distributed）。

## 性能

在 4× RTX A6000、相同 GPT 配置 **345M / 760M / 1.3B** 对比 Megatron-LM（TE）：

| 模式 | 精度 | nano / Megatron 吞吐比 | 说明 |
|------|------|------------------------|------|
| TP / TP+SP | FP32 | **0.93x – 1.01x** | unfused attention（FA 需半精度） |
| DP / TP×DP | FP32 | **0.97x – 1.05x** | |
| PP | FP32 | **1.00x – 1.04x** | |
| TP×PP | FP32 | **约 0.92x** | |
| TP / TP+SP | **BF16 + FA** | **0.87x – 0.92x** | 三个规模均覆盖 |
| DP2 | **BF16 + FA** | **0.95x – 1.01x** | 显存 **0.77x – 0.81x** |
| CP2 | **BF16 + FA** | **0.81x – 0.83x** | unfused 约 0.52x–0.73x；显存常更低 |

完整表格（按规模）：**[performance.md](performance.md)** §2.1（345M）· §3.1（760M）· §4.1（1.3B）。

## 快速开始

### 安装

```bash
pip install -e ".[dev]"
```

### 可选：FlashAttention

`flash-attn` 是可选依赖。单独安装以加速半精度注意力：

```bash
pip install flash-attn
```

在 `ReferenceGPTConfig` 中设置注意力后端：

```python
config = ReferenceGPTConfig(attn_backend="auto")  # 默认
```

| `attn_backend` | 行为 |
|----------------|------|
| `"auto"` | 当 CUDA + fp16/bf16 + `flash-attn` 已安装时使用 FlashAttention；否则回退到 unfused |
| `"flash"` | 要求 CUDA + fp16/bf16 + `flash-attn`；不可用时抛出 `RuntimeError` |
| `"unfused"` | 始终使用参考实现：scores→softmax→matmul |

**上下文并行（CP）说明：** CP flash 路径使用连续 all-gather + 分块 FA（非 TE zigzag ring）。`attention_dropout > 0` 且 `cp > 1` 时，flash CP 路径**不支持**（分块反向传播无法传递 dropout RNG 状态）。

### 运行参考模型

```bash
python scripts/run_reference_gpt.py --seed 0 --steps 3 --device cpu --out ref_traj.pt
```

### 基准测试（对比 Megatron-LM）

需将 Megatron-LM 加入 `PYTHONPATH`。为公平对比峰值显存，建议 DP/PP/CP 分别用 `--framework nano` 与 `--framework megatron` 各跑一次。

```bash
export PYTHONPATH=/path/to/nano-megatron:/path/to/Megatron-LM:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- BF16 + FlashAttention（345M TP2；需安装 flash-attn）---
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_tp.py --framework nano --tp-size 2 --precision bf16 --attn-backend flash \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_tp.py --framework megatron --tp-size 2 --precision bf16 \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096

# 760M:  --hidden-size 1536 --ffn-hidden-size 6144 --batch-size 2
# 1.3B:  --hidden-size 2048 --ffn-hidden-size 8192 --batch-size 1（TP2/DP/CP；TP4 用 batch=2）
# DP2 / CP2: scripts/benchmark_dp.py | benchmark_cp.py  + 同样 --precision bf16 [--attn-backend flash]

# --- FP32 基线（TP2 345M）---
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_tp.py --framework both --tp-size 2 \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096

# PP2（1F1B，FP32）
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_pp.py --framework nano --pp-size 2 --tp-size 1 \
  --batch-size 8 --num-microbatches 4 --seq-len 1024 \
  --hidden-size 1024 --num-layers 24 --num-heads 16 --ffn-hidden-size 4096
```

全部规模与并行组合见 [performance.md](performance.md)。

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
