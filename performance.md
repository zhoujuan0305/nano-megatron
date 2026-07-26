# Performance Report

## 1. Test Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX A6000 (4x, 48GB each) |
| PyTorch | 2.10.0a0+a36e1d39eb.nv26.01.42222806 |
| CUDA | 13.1 |
| Python | 3.12.3 |
| OS | Linux |
| Date | 2026-07-26 |

## 2. Model Configuration

**Architecture**: GPT-3 345M (Decoder-only Transformer)

| Parameter | Value |
|-----------|-------|
| vocab_size | 51200 |
| hidden_size | 512 |
| num_layers | 12 |
| num_heads | 8 |
| ffn_hidden_size | 2048 |
| max_seq_len | 1024 |
| position_embedding | RoPE |
| activation | SwiGLU |
| normalization | LayerNorm |
| bias | False |
| Total Parameters | 102,786,048 |

### TP2 Performance

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 27,991 | 3,200 | 73.17 |
| Megatron-LM | 39,078 | 1,103 | 52.41 |

**Throughput Ratio**: nano-megatron / Megatron-LM = **0.72x**

### TP4 Performance

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 29,502 | 2,720 | 69.42 |
| Megatron-LM | 39,010 | 1,107 | 52.50 |

**Throughput Ratio**: nano-megatron / Megatron-LM = **0.76x**

## 3. How to Reproduce

### TP2 Benchmark

```bash
torchrun --standalone --nproc_per_node=2 \
    scripts/benchmark_tp.py \
    --framework both \
    --tp-size 2 \
    --batch-size 2 \
    --seq-len 1024 \
    --hidden-size 512 \
    --num-layers 12 \
    --num-heads 8
```

### TP4 Benchmark

```bash
torchrun --standalone --nproc_per_node=4 \
    scripts/benchmark_tp.py \
    --framework both \
    --tp-size 4 \
    --batch-size 2 \
    --seq-len 1024 \
    --hidden-size 512 \
    --num-layers 12 \
    --num-heads 8
```
