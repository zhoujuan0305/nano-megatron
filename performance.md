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
| Params / rank (TP2) | 253.9M | 253.9M |
| Params / rank (TP4) | 127.0M | 127.0M |

### TP2

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 7,426 | 16,234 | 551.56 |
| Megatron-LM | 7,169 | 14,775 | 571.35 |

**Throughput Ratio**: nano-megatron / Megatron-LM = **1.04x**

### TP4

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 10,735 | 9,772 | 381.56 |
| Megatron-LM | 10,650 | 8,288 | 384.58 |

**Throughput Ratio**: nano-megatron / Megatron-LM = **1.01x**

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
| Params / rank (TP2) | 531.8M | 531.8M |
| Params / rank (TP4) | 266.0M | 266.0M |

### TP2

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 5,624 | 20,799 | 728.34 |
| Megatron-LM | 5,605 | 19,006 | 730.78 |

**Throughput Ratio**: nano-megatron / Megatron-LM = **1.00x**

### TP4

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 7,841 | 12,836 | 522.40 |
| Megatron-LM | 8,103 | 10,792 | 505.48 |

**Throughput Ratio**: nano-megatron / Megatron-LM = **0.97x**

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
| Params / rank (TP2) | 910.4M | 910.4M |
| Params / rank (TP4) | 455.3M | 455.3M |

> TP2 uses `batch_size=1` to fit A6000 48GB; TP4 uses `batch_size=2`. Tokens/sec already normalizes by batch size.

### TP2

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 4,209 | 14,466 | 486.60 |
| Megatron-LM | 4,239 | 13,611 | 483.17 |

**Throughput Ratio**: nano-megatron / Megatron-LM = **0.99x**

### TP4

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 5,909 | 16,099 | 693.17 |
| Megatron-LM | 6,125 | 13,504 | 668.72 |

**Throughput Ratio**: nano-megatron / Megatron-LM = **0.96x**

---

## 5. Reproduction

```bash
source /workspace/envs/megatron/bin/activate
export PYTHONPATH=/path/to/nano-megatron:/path/to/Megatron-LM:$PYTHONPATH

# 345M TP2
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_tp.py --framework both --tp-size 2 \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096

# 345M TP4
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/benchmark_tp.py --framework both --tp-size 4 \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096

# 760M TP2
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_tp.py --framework both --tp-size 2 \
  --batch-size 2 --seq-len 2048 --hidden-size 1536 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 6144

# 760M TP4
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/benchmark_tp.py --framework both --tp-size 4 \
  --batch-size 2 --seq-len 2048 --hidden-size 1536 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 6144

# 1.3B TP2 (batch_size=1)
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_tp.py --framework both --tp-size 2 \
  --batch-size 1 --seq-len 2048 --hidden-size 2048 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 8192

# 1.3B TP4 (batch_size=2)
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/benchmark_tp.py --framework both --tp-size 4 \
  --batch-size 2 --seq-len 2048 --hidden-size 2048 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 8192
```
