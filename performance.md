# Performance Report

## 1. Test Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX A6000 (4x, 48GB each) |
| PyTorch | 2.10.0a0+a36e1d39eb.nv26.01.42222806 |
| CUDA | 13.1 |
| Python | 3.12.3 |
| OS | Linux |

## 2. Model Configuration

**Architecture**: GPT-3 345M (Decoder-only Transformer)

| Parameter | nano-megatron | Megatron-LM |
|-----------|---------------|-------------|
| vocab_size | 51200 | 51200 |
| hidden_size | 1024 | 1024 |
| num_layers | 24 | 24 |
| num_heads | 16 | 16 |
| ffn_hidden_size | 4096 | 4096 |
| max_seq_len | 2048 | 2048 |
| position_embedding | RoPE | RoPE |
| activation | SwiGLU (silu) | SwiGLU (silu) |
| normalization | LayerNorm | LayerNorm |
| bias | False | False |
| Total Parameters | ~345M | ~345M |

### TP2 Performance

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 6,344 | 19,953 | 645.64 |
| Megatron-LM | 6,760 | 22,598 | 605.92 |

**Throughput Ratio**: nano-megatron / Megatron-LM = **0.94x**

### TP4 Performance

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 7,857 | 14,577 | 521.32 |
| Megatron-LM | 10,067 | 12,312 | 406.86 |

**Throughput Ratio**: nano-megatron / Megatron-LM = **0.78x**

## 4. Reproduction

### TP2 Benchmark

```bash
PYTHONPATH=/workspace/src/nano-megatron:/workspace/src/Megatron-LM:$PYTHONPATH \
/workspace/envs/megatron/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    scripts/benchmark_tp.py \
    --framework both \
    --tp-size 2 \
    --batch-size 2 \
    --seq-len 2048 \
    --hidden-size 1024 \
    --num-layers 24 \
    --num-heads 16
```

### TP4 Benchmark

```bash
PYTHONPATH=/workspace/src/nano-megatron:/workspace/src/Megatron-LM:$PYTHONPATH \
/workspace/envs/megatron/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node=4 \
    scripts/benchmark_tp.py \
    --framework both \
    --tp-size 4 \
    --batch-size 2 \
    --seq-len 2048 \
    --hidden-size 1024 \
    --num-layers 24 \
    --num-heads 16
```
