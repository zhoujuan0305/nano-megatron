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
| Precision | FP32 (TP/DP/PP); **BF16 (CP only)** |
| Batch / Seq | see each model section |
| Warmup / Measure | 3 / 10 steps |
| Parallel modes | TP, TP+SP, DP, TP×DP, PP, TP×PP, **CP, CP×TP, CP×DP** |
| SP note | SP reuses the TP group; `seq_len % tp_size == 0` required |
| DP note | micro-batch per DP rank; global tok/s = local × dp_size |
| PP note | non-interleaved 1F1B; local_bs = sum of microbatches; tok/s = local_bs × seq / wall (× dp if DP) |
| PP P2P | nano: sync `send`/`recv`; Megatron: schedule P2P (TE kernels on Megatron path) |
| CP note | nano: contiguous all-gather KV; Megatron: TE FlashAttention + zigzag pack; `seq_len % (2·cp) == 0`; tok/s = batch × seq × dp / wall (CP does not multiply data) |
| CP precision | Megatron TE CP requires bf16/fp16; CP tables use **BF16 on both sides** for a fair comparison |
| DP / PP / CP memory | nano and Megatron run in **separate torchrun processes** (no in-process `--framework both`) |
| Env | `CUDA_DEVICE_MAX_CONNECTIONS=1` (recommended for Megatron TP/SP/DP/PP/CP) |

Measured with `scripts/benchmark_tp.py` (TP/SP, `--framework both`), `scripts/benchmark_dp.py` (DP / TP×DP, isolated runs), `scripts/benchmark_pp.py` (PP / TP×PP, isolated runs), and `scripts/benchmark_cp.py` (CP / CP×TP / CP×DP, isolated runs, BF16).

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
| Params / rank (PP2) | 253.8M | 253.8M |
| Params / rank (PP4) | 153.1M | 153.1M |
| Params / rank (TP2×PP2) | 126.9M | 126.9M |
| Params / rank (CP, tp=1) | 507.6M | 507.6M |
| Params / rank (TP2×CP2) | 253.9M | 253.9M |

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

### PP2 (tp=1, local_bs=8, seq=1024, microbatches=4)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 9,318 | 9,945 | 879.17 |
| Megatron-LM | 9,084 | 10,687 | 901.85 |

**Throughput Ratio** (nano / Megatron): **1.03x**  
**Memory Ratio** (nano / Megatron): **0.93x**

### PP4 (tp=1, local_bs=8, seq=1024, microbatches=8)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 16,416 | 5,201 | 499.02 |
| Megatron-LM | 15,792 | 5,586 | 518.75 |

**Throughput Ratio** (nano / Megatron): **1.04x**  
**Memory Ratio** (nano / Megatron): **0.93x**

### TP2×PP2 (local_bs=8, seq=1024, microbatches=4)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 15,669 | 5,782 | 522.83 |
| Megatron-LM | 16,994 | 5,771 | 482.06 |

**Throughput Ratio** (nano / Megatron): **0.92x**  
**Memory Ratio** (nano / Megatron): **1.00x**

> PP runs use `seq_len=1024` and `local_bs=8` (sum of microbatches). Schedule is non-interleaved 1F1B on both sides. Megatron path uses Transformer Engine layers; nano uses sync P2P.

### CP2 (tp=1, BF16, batch=2, seq=2048)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 12,663 | 9,209 | 323.45 |
| Megatron-LM | 25,678 | 5,765 | 159.52 |

**Throughput Ratio** (nano / Megatron): **0.49x**  
**Memory Ratio** (nano / Megatron): **1.60x**

### CP4 (tp=1, BF16, batch=2, seq=2048)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 15,056 | 6,289 | 272.04 |
| Megatron-LM | 19,515 | 3,863 | 209.89 |

**Throughput Ratio** (nano / Megatron): **0.77x**  
**Memory Ratio** (nano / Megatron): **1.63x**

### TP2×CP2 (BF16, batch=2, seq=2048)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 18,138 | 5,047 | 225.83 |
| Megatron-LM | 30,458 | 3,109 | 134.48 |

**Throughput Ratio** (nano / Megatron): **0.60x**  
**Memory Ratio** (nano / Megatron): **1.62x**

### CP2×DP2 (BF16, batch=2, seq=2048)

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron | 11,303 | 22,607 | 9,209 | 362.37 |
| Megatron-LM | 20,773 | 41,546 | 5,765 | 197.18 |

**Throughput Ratio** (nano / Megatron, global): **0.54x**  
**Memory Ratio** (nano / Megatron): **1.60x**

> CP section is **BF16** (Megatron TE FlashAttention CP requires half precision). nano uses contiguous all-gather KV; Megatron uses zigzag CP packing + TE kernels. Gap is expected until nano adds ring/Flash CP kernels.

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
| Params / rank (PP2) | 531.7M | 531.7M |
| Params / rank (PP4) | 305.2M | 305.2M |
| Params / rank (TP2×PP2) | 265.9M | 265.9M |
| Params / rank (CP, tp=1) | 1063.4M | 1063.4M |
| Params / rank (TP2×CP2) | 531.8M | 531.8M |

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

### PP2 (tp=1, local_bs=8, seq=1024, microbatches=4)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 6,372 | 14,334 | 1,285.64 |
| Megatron-LM | 6,321 | 15,443 | 1,296.00 |

**Throughput Ratio** (nano / Megatron): **1.01x**  
**Memory Ratio** (nano / Megatron): **0.93x**

### PP4 (tp=1, local_bs=8, seq=1024, microbatches=8)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 10,735 | 7,505 | 763.13 |
| Megatron-LM | 10,458 | 8,081 | 783.32 |

**Throughput Ratio** (nano / Megatron): **1.03x**  
**Memory Ratio** (nano / Megatron): **0.93x**

### TP2×PP2 (local_bs=8, seq=1024, microbatches=4)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 10,846 | 8,364 | 755.32 |
| Megatron-LM | 11,792 | 8,345 | 694.74 |

**Throughput Ratio** (nano / Megatron): **0.92x**  
**Memory Ratio** (nano / Megatron): **1.00x**

### CP2 (tp=1, BF16, batch=2, seq=2048)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 9,126 | 11,723 | 448.85 |
| Megatron-LM | 15,003 | 9,287 | 273.01 |

**Throughput Ratio** (nano / Megatron): **0.61x**  
**Memory Ratio** (nano / Megatron): **1.26x**

### CP4 (tp=1, BF16, batch=2, seq=2048)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 10,013 | 8,195 | 409.06 |
| Megatron-LM | 13,446 | 6,735 | 304.63 |

**Throughput Ratio** (nano / Megatron): **0.74x**  
**Memory Ratio** (nano / Megatron): **1.22x**

### TP2×CP2 (BF16, batch=2, seq=2048)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 12,629 | 6,518 | 324.33 |
| Megatron-LM | 18,991 | 4,961 | 215.68 |

**Throughput Ratio** (nano / Megatron): **0.67x**  
**Memory Ratio** (nano / Megatron): **1.31x**

### CP2×DP2 (BF16, batch=2, seq=2048)

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron | 7,729 | 15,458 | 11,724 | 529.94 |
| Megatron-LM | 11,558 | 23,116 | 9,287 | 354.38 |

**Throughput Ratio** (nano / Megatron, global): **0.67x**  
**Memory Ratio** (nano / Megatron): **1.26x**

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
| Params / rank (PP2) | 910.3M | 910.3M |
| Params / rank (PP4) | 507.6M | 507.6M |
| Params / rank (TP2×PP2) | 455.2M | 455.2M |
| Params / rank (CP, tp=1) | 1820.5M | 1820.5M |
| Params / rank (TP2×CP2) | 910.4M | 910.4M |

> TP2 uses `batch_size=1` to fit A6000 48GB; TP4 uses `batch_size=2`. DP and TP2×DP2 use `batch_size=1` (full or half replica). PP / TP×PP use `local_bs=4`, `seq_len=1024`. CP tables use `batch_size=1`, BF16. Tokens/sec already normalizes by batch size.

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

### PP2 (tp=1, local_bs=4, seq=1024, microbatches=4)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 4,245 | 13,273 | 964.94 |
| Megatron-LM | 4,227 | 14,071 | 969.02 |

**Throughput Ratio** (nano / Megatron): **1.00x**  
**Memory Ratio** (nano / Megatron): **0.94x**

### PP4 (tp=1, local_bs=4, seq=1024, microbatches=4)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 6,235 | 8,661 | 656.96 |
| Megatron-LM | 6,146 | 9,261 | 666.45 |

**Throughput Ratio** (nano / Megatron): **1.01x**  
**Memory Ratio** (nano / Megatron): **0.94x**

### TP2×PP2 (local_bs=4, seq=1024, microbatches=4)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 7,285 | 7,449 | 562.25 |
| Megatron-LM | 7,900 | 7,479 | 518.50 |

**Throughput Ratio** (nano / Megatron): **0.92x**  
**Memory Ratio** (nano / Megatron): **1.00x**

### CP2 (tp=1, BF16, batch=1, seq=2048)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 5,532 | 10,922 | 370.18 |
| Megatron-LM | 7,754 | 10,092 | 264.11 |

**Throughput Ratio** (nano / Megatron): **0.71x**  
**Memory Ratio** (nano / Megatron): **1.08x**

### CP4 (tp=1, BF16, batch=1, seq=2048)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 4,638 | 10,677 | 441.54 |
| Megatron-LM | 5,432 | 8,628 | 377.05 |

**Throughput Ratio** (nano / Megatron): **0.85x**  
**Memory Ratio** (nano / Megatron): **1.24x**

### TP2×CP2 (BF16, batch=1, seq=2048)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 6,709 | 5,663 | 305.25 |
| Megatron-LM | 8,497 | 5,267 | 241.03 |

**Throughput Ratio** (nano / Megatron): **0.79x**  
**Memory Ratio** (nano / Megatron): **1.08x**

### CP2×DP2 (BF16, batch=1, seq=2048)

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron | 3,979 | 7,959 | 10,922 | 514.66 |
| Megatron-LM | 5,076 | 10,152 | 10,092 | 403.48 |

**Throughput Ratio** (nano / Megatron, global): **0.78x**  
**Memory Ratio** (nano / Megatron): **1.08x**

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

# --- PP / TP×PP (scripts/benchmark_pp.py) ---
# Fair peak memory: run nano and megatron in separate processes.

# 345M PP2
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_pp.py --framework nano --pp-size 2 --tp-size 1 \
  --batch-size 8 --num-microbatches 4 --seq-len 1024 \
  --hidden-size 1024 --num-layers 24 --num-heads 16 --ffn-hidden-size 4096
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_pp.py --framework megatron --pp-size 2 --tp-size 1 \
  --batch-size 8 --num-microbatches 4 --seq-len 1024 \
  --hidden-size 1024 --num-layers 24 --num-heads 16 --ffn-hidden-size 4096

# 345M PP4
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/benchmark_pp.py --framework nano --pp-size 4 --tp-size 1 \
  --batch-size 8 --num-microbatches 8 --seq-len 1024 \
  --hidden-size 1024 --num-layers 24 --num-heads 16 --ffn-hidden-size 4096
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/benchmark_pp.py --framework megatron --pp-size 4 --tp-size 1 \
  --batch-size 8 --num-microbatches 8 --seq-len 1024 \
  --hidden-size 1024 --num-layers 24 --num-heads 16 --ffn-hidden-size 4096

# 345M TP2×PP2
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/benchmark_pp.py --framework nano --pp-size 2 --tp-size 2 \
  --batch-size 8 --num-microbatches 4 --seq-len 1024 \
  --hidden-size 1024 --num-layers 24 --num-heads 16 --ffn-hidden-size 4096
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/benchmark_pp.py --framework megatron --pp-size 2 --tp-size 2 \
  --batch-size 8 --num-microbatches 4 --seq-len 1024 \
  --hidden-size 1024 --num-layers 24 --num-heads 16 --ffn-hidden-size 4096

# 760M: same PP knobs as 345M with --hidden-size 1536 --ffn-hidden-size 6144
# 1.3B: --hidden-size 2048 --ffn-hidden-size 8192 --batch-size 4
#   (PP2/PP4/TP2×PP2 all use local_bs=4, seq=1024; PP4 uses --num-microbatches 4)

# --- CP / CP×TP / CP×DP (scripts/benchmark_cp.py, BF16) ---
# Fair peak memory: separate processes. seq_len must be divisible by 2*cp.

# 345M CP2
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_cp.py --framework nano --cp-size 2 --precision bf16 \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_cp.py --framework megatron --cp-size 2 --precision bf16 \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096

# 345M CP4 / TP2×CP2 / CP2×DP2: nproc=4, --cp-size 4 | --tp-size 2 --cp-size 2 | --cp-size 2 --dp-size 2
# 760M: --hidden-size 1536 --ffn-hidden-size 6144 --batch-size 2
# 1.3B: --hidden-size 2048 --ffn-hidden-size 8192 --batch-size 1
```
