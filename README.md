# nano-megatron

[中文](README_zh.md)

## Background

A compact PyTorch/CUDA framework for understanding Megatron-style distributed
LLM training through independent implementation and comparison with Megatron.
It implements data, tensor, sequence, pipeline, and context parallelism, including
custom gradient buckets, non-interleaved 1F1B scheduling, and communication overlap.
PyTorch provides tensors, autograd, and distributed primitives; Megatron is a
reference, not the implementation of Nano's parallel layers or DDP.

FlashAttention is optional. Collectives use PyTorch/NCCL by default, with an
optional [nano-nccl](https://github.com/zhoujuan0305/nano-nccl) backend.

## Performance

TP2 × PP2 × DP2 with sequence parallelism, 2 nodes × 4 RTX A6000, a 1.424B GPT,
BF16 + FlashAttention, and FP32-master AdamW:

| Implementation | Full training step |
| --- | ---: |
| nano-megatron, wgrad accumulation fusion enabled | 266.57 ms |
| Megatron-Core + Transformer Engine | 257.38 ms |

Nano reached **96.55% of Megatron's throughput** in this single-run observation.
Each run used 50 warmup and 100 measured updates. CPUs were shared, with a fixed
10-logical-CPU pool per node; SMT contention varied between runs. This is not an
isolated or statistically established performance guarantee. Environment and
measurement details: [performance.md](performance.md).

## Usage

Install a PyTorch build suitable for your CUDA environment, then:

```bash
python -m pip install -e ".[dev]"
# Optional; required for the FlashAttention example below.
python -m pip install flash-attn --no-build-isolation
```

Run a small CPU reference model:

```bash
python scripts/run_reference_gpt.py --seed 0 --steps 3 --device cpu --out ref_traj.pt
```

For TP2 × SP × PP2 × DP2, run the following on **both nodes**, with four GPUs per
node. Set `NODE_RANK` to 0 or 1 and use the same reachable `MASTER_ADDR` and port.

```bash
export MASTER_ADDR="<node-0-address>"
export MASTER_PORT=29500
export NODE_RANK=0  # 1 on the second node
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

torchrun --nnodes=2 --nproc-per-node=4 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  scripts/benchmark_pp.py --framework nano \
  --tp-size 2 --pp-size 2 --dp-size 2 --sequence-parallel \
  --hidden-size 2048 --num-layers 24 --num-heads 16 --ffn-hidden-size 5504 \
  --batch-size 4 --num-microbatches 4 --seq-len 1024 \
  --precision bf16 --attn-backend flash --optimizer adamw \
  --warmup-steps 50 --benchmark-steps 100 \
  --fuse-rope --rope-fp32-phase --fuse-swiglu --vocab-ce-fp32 \
  --overlap-grad-reduce --overlap-tp-dgrad \
  --disable-bucketing-after-first-stage --fuse-wgrad-accum
```

`--batch-size` is per DP replica: this example has global batch 8 and microbatch
size 1. Select CPU affinity and a network interface appropriate for your machines;
the command does not reserve CPUs or reproduce the measured CPU allocation.

Wgrad accumulation fusion is opt-in and uses the explicit training backward
scope. It does not support tied/reused weights, concurrent use of the same
parameter, autocast, higher-order gradients, or full-step CUDA Graphs. See
[TEXT_TRAINING.md](TEXT_TRAINING.md) for text training and checkpoints.

Unit tests: `python -m pytest tests/unit`. Megatron comparison additionally
requires Megatron-Core and Transformer Engine; the Nano path does not depend on
TE.

## License

[MIT](LICENSE).
