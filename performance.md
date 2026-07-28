# Performance Report

## 1. Test Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX A6000 (4x, 48GB each) |
| Python env | `/workspace/envs/megatron` |
| PyTorch | 2.10.0a0+a36e1d39eb.nv26.01.42222806 |
| CUDA | 13.1 |
| Python | 3.12.3 |
| OS | Linux |
| Precision | FP32 |
| Batch / Seq | see each model section |
| Warmup / Measure | 3 / 10 steps |
| Parallel modes | TP, TP+SP, DP, TP×DP |
| SP note | SP reuses the TP group; `seq_len % tp_size == 0` required |
| DP note | micro-batch per DP rank; global tok/s = local × dp_size |
| DP memory | nano and Megatron run in **separate torchrun processes** (no in-process `--framework both`) |
| Env | `CUDA_DEVICE_MAX_CONNECTIONS=1` (recommended for Megatron TP/SP/DP) |

Measured with `scripts/benchmark_tp.py` (TP/SP, `--framework both`) and `scripts/benchmark_dp.py` (DP / TP×DP, isolated `--framework nano` then `--framework megatron`).

---

## 2. GPT-3 345M

### Model Configuration

| Parameter | nano-megatron | Megatron-LM |
|-----------|---------------|-------------|
| vocab_size | 51200 | 51200 |
| hidden_size | 1024 | 1024 |
| num_layers | 24 | 24 |
| num_heads | 16 | 16 |
| ffn_hidden_size | 4096 | 4096 |
| max_seq_len | 2048 | 2048 |
| batch_size | 2 | 2 |
| position_embedding | RoPE | RoPE |
| activation | SwiGLU (silu) | SwiGLU (silu) |
| normalization | LayerNorm | LayerNorm |
| bias | False | False |
| fused QKV | True | True (TE) |
| Params / rank (TP2) | 253.9M | 253.9M |
| Params / rank (TP4) | 127.0M | 127.0M |
| Params / rank (DP, tp=1) | 507.6M | 507.6M |

### TP2

| Framework | SP | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|----|------------|-------------|-------------------|
| nano-megatron | off | 7,410 | 16,234 | 552.77 |
| Megatron-LM | off | 7,310 | 14,775 | 560.35 |
| nano-megatron | on | 7,154 | 14,290 | 572.55 |
| Megatron-LM | on | 7,219 | 14,014 | 567.37 |

**Throughput Ratio** (nano / Megatron): TP **1.01x**, TP+SP **0.99x**

### TP4

| Framework | SP | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|----|------------|-------------|-------------------|
| nano-megatron | off | 10,691 | 9,772 | 383.11 |
| Megatron-LM | off | 10,806 | 8,288 | 379.03 |
| nano-megatron | on | 10,302 | 7,631 | 397.58 |
| Megatron-LM | on | 10,313 | 7,123 | 397.17 |

**Throughput Ratio** (nano / Megatron): TP **0.99x**, TP+SP **1.00x**

### DP2 (tp=1)

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron | 3,991 | 7,982 | 26,834 | 1,026.31 |
| Megatron-LM | 3,791 | 7,582 | 29,731 | 1,080.50 |

**Throughput Ratio** (nano / Megatron, global): **1.05x**  
**Memory Ratio** (nano / Megatron): **0.90x**

### DP4 (tp=1)

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron | 3,707 | 14,828 | 26,834 | 1,104.93 |
| Megatron-LM | 3,533 | 14,134 | 29,731 | 1,159.23 |

**Throughput Ratio** (nano / Megatron, global): **1.05x**  
**Memory Ratio** (nano / Megatron): **0.90x**

### TP2×DP2

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron | 6,459 | 12,919 | 16,234 | 634.12 |
| Megatron-LM | 6,421 | 12,842 | 15,743 | 637.90 |

**Throughput Ratio** (nano / Megatron, global): **1.01x**  
**Memory Ratio** (nano / Megatron): **1.03x**

---

## 3. GPT 760M

### Model Configuration

| Parameter | nano-megatron | Megatron-LM |
|-----------|---------------|-------------|
| vocab_size | 51200 | 51200 |
| hidden_size | 1536 | 1536 |
| num_layers | 24 | 24 |
| num_heads | 16 | 16 |
| ffn_hidden_size | 6144 | 6144 |
| max_seq_len | 2048 | 2048 |
| batch_size | 2 | 2 |
| position_embedding | RoPE | RoPE |
| activation | SwiGLU (silu) | SwiGLU (silu) |
| normalization | LayerNorm | LayerNorm |
| bias | False | False |
| fused QKV | True | True (TE) |
| Params / rank (TP2) | 531.8M | 531.8M |
| Params / rank (TP4) | 266.0M | 266.0M |
| Params / rank (DP, tp=1) | 1063.4M | 1063.4M |

### TP2

| Framework | SP | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|----|------------|-------------|-------------------|
| nano-megatron | off | 5,598 | 20,799 | 731.66 |
| Megatron-LM | off | 5,709 | 19,006 | 717.52 |
| nano-megatron | on | 5,419 | 17,882 | 755.85 |
| Megatron-LM | on | 5,621 | 17,865 | 728.64 |

**Throughput Ratio** (nano / Megatron): TP **0.98x**, TP+SP **0.96x**

### TP4

| Framework | SP | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|----|------------|-------------|-------------------|
| nano-megatron | off | 7,829 | 12,836 | 523.17 |
| Megatron-LM | off | 8,185 | 10,792 | 500.42 |
| nano-megatron | on | 7,512 | 9,625 | 545.24 |
| Megatron-LM | on | 7,814 | 9,056 | 524.17 |

**Throughput Ratio** (nano / Megatron): TP **0.96x**, TP+SP **0.96x**

### DP2 (tp=1)

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron | 2,952 | 5,904 | 33,626 | 1,387.53 |
| Megatron-LM | 2,868 | 5,737 | 39,520 | 1,427.94 |

**Throughput Ratio** (nano / Megatron, global): **1.03x**  
**Memory Ratio** (nano / Megatron): **0.85x**

### DP4 (tp=1)

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron | 2,634 | 10,537 | 33,626 | 1,554.88 |
| Megatron-LM | 2,576 | 10,304 | 39,520 | 1,590.09 |

**Throughput Ratio** (nano / Megatron, global): **1.02x**  
**Memory Ratio** (nano / Megatron): **0.85x**

### TP2×DP2

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron | 4,548 | 9,096 | 20,799 | 900.64 |
| Megatron-LM | 4,657 | 9,313 | 21,034 | 879.59 |

**Throughput Ratio** (nano / Megatron, global): **0.98x**  
**Memory Ratio** (nano / Megatron): **0.99x**

---

## 4. GPT 1.3B

### Model Configuration

| Parameter | nano-megatron | Megatron-LM |
|-----------|---------------|-------------|
| vocab_size | 51200 | 51200 |
| hidden_size | 2048 | 2048 |
| num_layers | 24 | 24 |
| num_heads | 16 | 16 |
| ffn_hidden_size | 8192 | 8192 |
| max_seq_len | 2048 | 2048 |
| batch_size (TP2) | 1 | 1 |
| batch_size (TP4) | 2 | 2 |
| position_embedding | RoPE | RoPE |
| activation | SwiGLU (silu) | SwiGLU (silu) |
| normalization | LayerNorm | LayerNorm |
| bias | False | False |
| fused QKV | True | True (TE) |
| Params / rank (TP2) | 910.4M | 910.4M |
| Params / rank (TP4) | 455.3M | 455.3M |
| Params / rank (DP, tp=1) | 1820.5M | 1820.5M |

> TP2 uses `batch_size=1` to fit A6000 48GB; TP4 uses `batch_size=2`. DP and TP2×DP2 use `batch_size=1` (full or half replica). Tokens/sec already normalizes by batch size.

### TP2

| Framework | SP | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|----|------------|-------------|-------------------|
| nano-megatron | off | 4,204 | 14,466 | 487.16 |
| Megatron-LM | off | 4,350 | 13,611 | 470.81 |
| nano-megatron | on | 4,100 | 12,529 | 499.52 |
| Megatron-LM | on | 4,257 | 12,866 | 481.05 |

**Throughput Ratio** (nano / Megatron): TP **0.97x**, TP+SP **0.96x**

### TP4

| Framework | SP | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|----|------------|-------------|-------------------|
| nano-megatron | off | 5,879 | 16,099 | 696.68 |
| Megatron-LM | off | 6,348 | 13,504 | 645.24 |
| nano-megatron | on | 5,665 | 11,818 | 723.05 |
| Megatron-LM | on | 6,005 | 11,190 | 682.15 |

**Throughput Ratio** (nano / Megatron): TP **0.93x**, TP+SP **0.94x**

### DP2 (tp=1, batch_size=1)

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron | 1,901 | 3,803 | 24,562 | 1,077.11 |
| Megatron-LM | 1,920 | 3,841 | 32,252 | 1,066.47 |

**Throughput Ratio** (nano / Megatron, global): **0.99x**  
**Memory Ratio** (nano / Megatron): **0.76x**

### DP4 (tp=1, batch_size=1)

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron | 1,503 | 6,014 | 24,562 | 1,362.17 |
| Megatron-LM | 1,524 | 6,095 | 32,252 | 1,343.95 |

**Throughput Ratio** (nano / Megatron, global): **0.99x**  
**Memory Ratio** (nano / Megatron): **0.76x**

### TP2×DP2 (batch_size=1)

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron | 2,658 | 5,316 | 14,674 | 770.54 |
| Megatron-LM | 2,752 | 5,504 | 17,028 | 744.16 |

**Throughput Ratio** (nano / Megatron, global): **0.97x**  
**Memory Ratio** (nano / Megatron): **0.86x**

---

## 5. Reproduction

```bash
source /workspace/envs/megatron/bin/activate
export PYTHONPATH=/path/to/nano-megatron:/path/to/Megatron-LM:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- TP / SP (scripts/benchmark_tp.py) ---

# 345M TP2
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_tp.py --framework both --tp-size 2 \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096

# 345M TP2 + SP
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_tp.py --framework both --tp-size 2 --sequence-parallel \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096

# 345M TP4
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/benchmark_tp.py --framework both --tp-size 4 \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096

# 345M TP4 + SP
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/benchmark_tp.py --framework both --tp-size 4 --sequence-parallel \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096

# 760M TP2 / TP2+SP
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_tp.py --framework both --tp-size 2 \
  --batch-size 2 --seq-len 2048 --hidden-size 1536 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 6144
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_tp.py --framework both --tp-size 2 --sequence-parallel \
  --batch-size 2 --seq-len 2048 --hidden-size 1536 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 6144

# 760M TP4 / TP4+SP
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/benchmark_tp.py --framework both --tp-size 4 \
  --batch-size 2 --seq-len 2048 --hidden-size 1536 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 6144
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/benchmark_tp.py --framework both --tp-size 4 --sequence-parallel \
  --batch-size 2 --seq-len 2048 --hidden-size 1536 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 6144

# 1.3B TP2 / TP2+SP (batch_size=1)
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_tp.py --framework both --tp-size 2 \
  --batch-size 1 --seq-len 2048 --hidden-size 2048 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 8192
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_tp.py --framework both --tp-size 2 --sequence-parallel \
  --batch-size 1 --seq-len 2048 --hidden-size 2048 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 8192

# 1.3B TP4 / TP4+SP (batch_size=2)
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/benchmark_tp.py --framework both --tp-size 4 \
  --batch-size 2 --seq-len 2048 --hidden-size 2048 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 8192
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/benchmark_tp.py --framework both --tp-size 4 --sequence-parallel \
  --batch-size 2 --seq-len 2048 --hidden-size 2048 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 8192

# --- DP / TP×DP (scripts/benchmark_dp.py) ---
# Fair peak memory: run nano and megatron in separate processes (not --framework both).

# 345M DP2 example (repeat with megatron; same pattern for DP4 / TP2×DP2 / other sizes)
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_dp.py --framework nano --tp-size 1 --dp-size 2 \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_dp.py --framework megatron --tp-size 1 --dp-size 2 \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096

# 345M DP4 / TP2×DP2: nproc=4, --dp-size 4 or --tp-size 2 --dp-size 2
# 760M: --hidden-size 1536 --ffn-hidden-size 6144 --batch-size 2
# 1.3B: --hidden-size 2048 --ffn-hidden-size 8192 --batch-size 1
```
