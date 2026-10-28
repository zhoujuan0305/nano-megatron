# Performance

## Environment

Measured on 2026-10-28.

| Item | Configuration |
| --- | --- |
| Hardware | 2 nodes × 4 NVIDIA RTX A6000, 48 GB per GPU |
| Parallelism | TP2 × PP2 × DP2, sequence parallelism enabled |
| Placement | TP and DP groups within each node; PP crosses nodes |
| Runtime | Linux, Python 3.12, PyTorch 2.10.0a0+a36e1d39eb.nv26.01.42222806 |
| CUDA / NCCL | 13.1 / 2.29.2 |
| Communication | PyTorch/NCCL; inter-node NCCL Socket, Ring/Simple, IB disabled |
| CPU allocation | Fixed shared pool of 10 logical CPUs per node for 4 ranks; tensor-library thread budgets set to 1 |
| Model | 1.424B GPT; 24 layers, hidden size 2048, 16 heads, FFN size 5504, vocabulary 51200; RoPE, SwiGLU, LayerNorm |
| Input | Fixed synthetic tokens; sequence length 1024 |
| Batch | Global batch 8; batch 4 per DP replica; 4 microbatches of size 1 |
| Precision / execution | BF16 parameters and gradients, FP32 loss computation, eager execution, FlashAttention |
| Optimizer | Same FP32-master fused AdamW in both frameworks; lr 1e-4, betas (0.9, 0.95), eps 1e-8, weight decay 0.1 on matrices |
| Nano options | Wgrad accumulation fusion, fused RoPE/SwiGLU, TP dgrad and DP gradient overlap; size-based DP bucketing disabled after the first PP stage; attention layout views off |
| Megatron reference | Transformer Engine layers, FlashAttention, RoPE and gradient accumulation fusion enabled; native DDP with overlap; batched PP communication without the extra synchronization |
| Measurement | 50 warmup + 100 measured updates, one independent run per implementation; maximum elapsed time across 8 ranks divided by 100 |

## Results

| Implementation | Full step (ms) | Throughput relative to Megatron |
| --- | ---: | ---: |
| nano-megatron, wgrad accumulation fusion enabled | 266.5746 | 96.55% |
| Megatron-Core + Transformer Engine | 257.3770 | 100.00% |

A measured step includes forward, backward, TP/SP and DP collectives, PP
communication, and the AdamW update. Warmup, validation, checkpointing, data I/O,
and profiler overhead are excluded.

Nano's observed step time was 3.57% higher. These are single-run observations on
shared CPUs, not statistically established or CPU-isolated results. SMT sibling
load differed between runs, so this table is not a causal measurement of any
individual optimization. Results apply to this configuration only.
