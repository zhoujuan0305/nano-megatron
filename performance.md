# Performance Report

## 1. Test Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX A6000 (4x, 48GB each) |
| PyTorch | 2.10.0a0+a36e1d39eb.nv26.01.42222806 |
| CUDA | 13.1 |
| Python | 3.12.3 |
| OS | Linux |
| Precision | FP32 |
| Batch / Seq | see each model section |
| Warmup / Measure | 3 / 10 steps |

Both frameworks use the same architecture settings per model (RoPE, SwiGLU, LayerNorm, no bias) for a fair comparison.

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
| Total Parameters | ~345M | ~345M |

### TP2

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 6,345 | 19,953 | 645.56 |
| Megatron-LM | 6,761 | 22,598 | 605.81 |

**Throughput Ratio**: nano-megatron / Megatron-LM = **0.94x**

### TP4

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 7,851 | 14,577 | 521.72 |
| Megatron-LM | 10,075 | 12,322 | 406.55 |

**Throughput Ratio**: nano-megatron / Megatron-LM = **0.78x**

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
| Total Parameters | ~760M | ~760M |

### TP2

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 4,695 | 25,778 | 872.41 |
| Megatron-LM | 5,409 | 26,849 | 757.25 |

**Throughput Ratio**: nano-megatron / Megatron-LM = **0.87x**

### TP4

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 5,683 | 18,962 | 720.70 |
| Megatron-LM | 7,806 | 14,868 | 524.73 |

**Throughput Ratio**: nano-megatron / Megatron-LM = **0.73x**

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
| Total Parameters | ~1.3B | ~1.3B |

> TP2 uses `batch_size=1` to fit A6000 48GB; TP4 uses `batch_size=2`. Tokens/sec already normalizes by batch size.

### TP2

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 3,555 | 17,985 | 576.01 |
| Megatron-LM | 4,182 | 17,410 | 489.69 |

**Throughput Ratio**: nano-megatron / Megatron-LM = **0.85x**

### TP4

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 4,327 | 23,538 | 946.57 |
| Megatron-LM | 5,974 | 17,627 | 685.60 |

**Throughput Ratio**: nano-megatron / Megatron-LM = **0.72x**

---

## 5. Summary

| Model | Parallel | nano-megatron | Megatron-LM | Ratio |
|-------|----------|---------------|-------------|-------|
| 345M | TP2 | 6,345 | 6,761 | **0.94x** |
| 345M | TP4 | 7,851 | 10,075 | **0.78x** |
| 760M | TP2 | 4,695 | 5,409 | **0.87x** |
| 760M | TP4 | 5,683 | 7,806 | **0.73x** |
| 1.3B | TP2 | 3,555 | 4,182 | **0.85x** |
| 1.3B | TP4 | 4,327 | 5,974 | **0.72x** |

Gap widens with larger TP size (TP4 vs TP2), consistent with communication and kernel-fusion differences.

---

## 6. Reproduction

```bash
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
