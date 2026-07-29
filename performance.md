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
| Precision | FP32 (TP/DP/PP baseline); **BF16 + FlashAttention** (FA section + CP) |
| Batch / Seq | see each model section |
| Warmup / Measure | 3 / 10 steps |
| Parallel modes | TP, TP+SP, DP, TP×DP, PP, TP×PP, **CP, CP×TP, CP×DP** |
| SP note | SP reuses the TP group; `seq_len % tp_size == 0` required |
| DP note | micro-batch per DP rank; global tok/s = local × dp_size |
| PP note | non-interleaved 1F1B; local_bs = sum of microbatches; tok/s = local_bs × seq / wall (× dp if DP) |
| PP P2P | nano: sync `send`/`recv`; Megatron: schedule P2P (TE kernels on Megatron path) |
| Attention | nano: optional `flash-attn` via `attn_backend=auto\|flash\|unfused` (default auto); Megatron: TE DotProductAttention (Flash in bf16/fp16) |
| CP note | nano FA path: contiguous **AG-KV + chunked FlashAttention** (not P2P ring); unfused fallback: AG-KV + matmul softmax; Megatron: TE FlashAttention + zigzag pack; `seq_len % (2·cp) == 0`; tok/s = batch × seq × dp / wall |
| CP precision | Megatron TE CP requires bf16/fp16; FA/CP fair tables use **BF16 on both sides** |
| DP / PP / CP memory | nano and Megatron run in **separate torchrun processes** (no in-process `--framework both`) |
| Env | `CUDA_DEVICE_MAX_CONNECTIONS=1` (recommended for Megatron TP/SP/DP/PP/CP) |

Measured with `scripts/benchmark_tp.py` (TP/SP; FP32 or `--precision bf16 --attn-backend flash`), `scripts/benchmark_dp.py` (DP / TP×DP, isolated), `scripts/benchmark_pp.py` (PP / TP×PP, isolated), and `scripts/benchmark_cp.py` (CP / CP×TP / CP×DP, isolated, BF16).

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

### CP2 (tp=1, BF16, batch=2, seq=2048) — legacy unfused nano

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron (unfused AG-KV) | 13,408 | 7,809 | 305.49 |
| Megatron-LM | 25,678 | 5,765 | 159.52 |

**Throughput Ratio** (nano / Megatron): **0.52x**  
**Memory Ratio** (nano / Megatron): **1.35x**

### CP4 (tp=1, BF16, batch=2, seq=2048) — legacy unfused nano

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron (unfused AG-KV) | 17,018 | 4,691 | 240.69 |
| Megatron-LM | 19,515 | 3,863 | 209.89 |

**Throughput Ratio** (nano / Megatron): **0.87x**  
**Memory Ratio** (nano / Megatron): **1.21x**

### TP2×CP2 (BF16, batch=2, seq=2048) — legacy unfused nano

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron (unfused AG-KV) | 19,656 | 4,347 | 208.38 |
| Megatron-LM | 30,458 | 3,109 | 134.48 |

**Throughput Ratio** (nano / Megatron): **0.65x**  
**Memory Ratio** (nano / Megatron): **1.40x**

### CP2×DP2 (BF16, batch=2, seq=2048) — legacy unfused nano

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron (unfused AG-KV) | 11,976 | 23,953 | 7,809 | 342.00 |
| Megatron-LM | 20,773 | 41,546 | 5,765 | 197.18 |

**Throughput Ratio** (nano / Megatron, global): **0.58x**  
**Memory Ratio** (nano / Megatron): **1.35x**

> Legacy CP rows above are **BF16** with nano **unfused** AG-KV (pre–flash-attn). Prefer **§2.1 BF16 + FlashAttention** for current FA numbers. Megatron path is zigzag CP + TE Flash.

### 2.1 BF16 + FlashAttention (345M)

Fair compare with FA on both sides: nano `--precision bf16 --attn-backend flash`; Megatron TE BF16 (FlashAttention). Same model knobs as §2 (`batch=2`, `seq=2048` unless noted). Measured 2026-07-29 on the same A6000 node.

#### TP2 (BF16 + FA)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 28,580 | 5,119 | 143.32 |
| Megatron-LM | 31,152 | 4,532 | 131.48 |

**Throughput Ratio** (nano / Megatron): **0.92x**  
**Memory Ratio** (nano / Megatron): **1.13x**

#### TP2 + SP (BF16 + FA)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 26,725 | 4,146 | 153.26 |
| Megatron-LM | 30,085 | 4,147 | 136.15 |

**Throughput Ratio** (nano / Megatron): **0.89x**  
**Memory Ratio** (nano / Megatron): **1.00x**

#### TP4 (BF16 + FA)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 31,655 | 3,360 | 129.40 |
| Megatron-LM | 35,699 | 2,681 | 114.74 |

**Throughput Ratio** (nano / Megatron): **0.89x**  
**Memory Ratio** (nano / Megatron): **1.25x**

#### DP2 (tp=1, BF16 + FA)

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron | 16,747 | 33,494 | 7,432 | 244.58 |
| Megatron-LM | 16,553 | 33,105 | 9,173 | 247.45 |

**Throughput Ratio** (nano / Megatron, global): **1.01x**  
**Memory Ratio** (nano / Megatron): **0.81x**

#### CP2 (tp=1, BF16 + FA)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron (AG-KV + chunked FA) | 20,718 | 4,788 | 197.70 |
| Megatron-LM (TE zigzag + FA) | 25,594 | 5,765 | 160.04 |

**Throughput Ratio** (nano / Megatron): **0.81x**  
**Memory Ratio** (nano / Megatron): **0.83x**

> FA lift on nano CP2 vs legacy unfused: ~13.4k → **20.7k** tok/s (~1.5×); mem 7.8 GB → **4.8 GB**. Remaining CP gap vs Megatron is mostly **AG+chunked FA vs TE ring/zigzag**, not missing Flash kernels. TP still trails TE fused Linear/LN (~8–11%). DP stays ~parity thr with lower nano DDP memory.

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
| nano-megatron | 9,539 | 10,401 | 429.39 |
| Megatron-LM | 15,003 | 9,287 | 273.01 |

**Throughput Ratio** (nano / Megatron): **0.64x**  
**Memory Ratio** (nano / Megatron): **1.12x**

### CP4 (tp=1, BF16, batch=2, seq=2048)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 10,892 | 6,740 | 376.07 |
| Megatron-LM | 13,446 | 6,735 | 304.63 |

**Throughput Ratio** (nano / Megatron): **0.81x**  
**Memory Ratio** (nano / Megatron): **1.00x**

### TP2×CP2 (BF16, batch=2, seq=2048)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 13,224 | 5,854 | 309.74 |
| Megatron-LM | 18,991 | 4,961 | 215.68 |

**Throughput Ratio** (nano / Megatron): **0.70x**  
**Memory Ratio** (nano / Megatron): **1.18x**

### CP2×DP2 (BF16, batch=2, seq=2048)

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron | 8,007 | 16,015 | 10,400 | 511.53 |
| Megatron-LM | 11,558 | 23,116 | 9,287 | 354.38 |

**Throughput Ratio** (nano / Megatron, global): **0.69x**  
**Memory Ratio** (nano / Megatron): **1.12x**

> Legacy CP rows: nano **unfused** AG-KV. Current FA numbers: **§3.1**.

### 3.1 BF16 + FlashAttention (760M)

Same knobs as §3 (`batch=2`, `seq=2048`). nano `--precision bf16 --attn-backend flash`; Megatron TE BF16.

#### TP2 (BF16 + FA)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 17,630 | 7,438 | 232.33 |
| Megatron-LM | 19,070 | 6,654 | 214.78 |

**Throughput Ratio** (nano / Megatron): **0.92x**  
**Memory Ratio** (nano / Megatron): **1.12x**

#### TP2 + SP (BF16 + FA)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 16,516 | 5,978 | 248.00 |
| Megatron-LM | 18,544 | 6,077 | 220.89 |

**Throughput Ratio** (nano / Megatron): **0.89x**  
**Memory Ratio** (nano / Megatron): **0.98x**

#### TP4 (BF16 + FA)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 20,956 | 4,930 | 195.45 |
| Megatron-LM | 23,556 | 3,958 | 173.89 |

**Throughput Ratio** (nano / Megatron): **0.89x**  
**Memory Ratio** (nano / Megatron): **1.25x**

#### DP2 (tp=1, BF16 + FA)

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron | 9,583 | 19,167 | 10,913 | 427.41 |
| Megatron-LM | 9,774 | 19,549 | 14,097 | 419.06 |

**Throughput Ratio** (nano / Megatron, global): **0.98x**  
**Memory Ratio** (nano / Megatron): **0.77x**

#### CP2 (tp=1, BF16 + FA)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron (AG-KV + chunked FA) | 12,422 | 7,349 | 329.75 |
| Megatron-LM (TE zigzag + FA) | 14,997 | 9,287 | 273.12 |

**Throughput Ratio** (nano / Megatron): **0.83x**  
**Memory Ratio** (nano / Megatron): **0.79x**

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
| nano-megatron | 5,665 | 10,922 | 361.55 |
| Megatron-LM | 7,754 | 10,092 | 264.11 |

**Throughput Ratio** (nano / Megatron): **0.73x**  
**Memory Ratio** (nano / Megatron): **1.08x**

### CP4 (tp=1, BF16, batch=1, seq=2048)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 4,797 | 10,677 | 426.90 |
| Megatron-LM | 5,432 | 8,628 | 377.05 |

**Throughput Ratio** (nano / Megatron): **0.88x**  
**Memory Ratio** (nano / Megatron): **1.24x**

### TP2×CP2 (BF16, batch=1, seq=2048)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 6,912 | 5,663 | 296.29 |
| Megatron-LM | 8,497 | 5,267 | 241.03 |

**Throughput Ratio** (nano / Megatron): **0.81x**  
**Memory Ratio** (nano / Megatron): **1.08x**

### CP2×DP2 (BF16, batch=1, seq=2048)

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron | 4,063 | 8,127 | 10,922 | 504.02 |
| Megatron-LM | 5,076 | 10,152 | 10,092 | 403.48 |

**Throughput Ratio** (nano / Megatron, global): **0.80x**  
**Memory Ratio** (nano / Megatron): **1.08x**

> Legacy CP rows: nano **unfused** AG-KV. Current FA numbers: **§4.1**.

### 4.1 BF16 + FlashAttention (1.3B)

Same knobs as §4 (`TP2 batch=1`, `TP4 batch=2`, `DP/CP batch=1`, `seq=2048`). nano `--precision bf16 --attn-backend flash`; Megatron TE BF16.

#### TP2 (BF16 + FA, batch=1)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 11,289 | 5,755 | 181.41 |
| Megatron-LM | 12,594 | 5,296 | 162.62 |

**Throughput Ratio** (nano / Megatron): **0.90x**  
**Memory Ratio** (nano / Megatron): **1.09x**

#### TP2 + SP (BF16 + FA, batch=1)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 10,852 | 4,787 | 188.73 |
| Megatron-LM | 12,181 | 4,912 | 168.14 |

**Throughput Ratio** (nano / Megatron): **0.89x**  
**Memory Ratio** (nano / Megatron): **0.97x**

#### TP4 (BF16 + FA, batch=2)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron | 14,457 | 6,572 | 283.31 |
| Megatron-LM | 16,584 | 5,309 | 246.98 |

**Throughput Ratio** (nano / Megatron): **0.87x**  
**Memory Ratio** (nano / Megatron): **1.24x**

#### DP2 (tp=1, BF16 + FA, batch=1)

| Framework | Tokens/sec (local) | Tokens/sec (global) | Memory (MB) | Avg Step Time (ms) |
|-----------|--------------------|---------------------|-------------|-------------------|
| nano-megatron | 5,106 | 10,211 | 10,633 | 401.13 |
| Megatron-LM | 5,400 | 10,799 | 13,210 | 379.29 |

**Throughput Ratio** (nano / Megatron, global): **0.95x**  
**Memory Ratio** (nano / Megatron): **0.80x**

#### CP2 (tp=1, BF16 + FA, batch=1)

| Framework | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|------------|-------------|-------------------|
| nano-megatron (AG-KV + chunked FA) | 6,420 | 10,921 | 318.98 |
| Megatron-LM (TE zigzag + FA) | 7,702 | 10,092 | 265.90 |

**Throughput Ratio** (nano / Megatron): **0.83x**  
**Memory Ratio** (nano / Megatron): **1.08x**

---

## 5. Reproduction

```bash
source /workspace/envs/megatron/bin/activate
export PYTHONPATH=/path/to/nano-megatron:/path/to/Megatron-LM:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- BF16 + FlashAttention (345M; §2.1) ---
# Optional: pip install flash-attn. Fair peak memory: separate nano / megatron jobs.

# TP2 BF16+FA
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_tp.py --framework nano --tp-size 2 --precision bf16 --attn-backend flash \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/benchmark_tp.py --framework megatron --tp-size 2 --precision bf16 \
  --batch-size 2 --seq-len 2048 --hidden-size 1024 --num-layers 24 \
  --num-heads 16 --ffn-hidden-size 4096

# DP2 / CP2 BF16+FA: add --precision bf16; nano also --attn-backend flash
#   scripts/benchmark_dp.py --framework nano|megatron --dp-size 2 ...
#   scripts/benchmark_cp.py --framework nano|megatron --cp-size 2 ...
# 760M: --hidden-size 1536 --ffn-hidden-size 6144 --batch-size 2
# 1.3B: --hidden-size 2048 --ffn-hidden-size 8192 --batch-size 1 (TP2/DP/CP); TP4 batch=2

# --- TP / SP FP32 baseline (scripts/benchmark_tp.py) ---

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
