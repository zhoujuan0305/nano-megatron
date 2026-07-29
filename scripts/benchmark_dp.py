#!/usr/bin/env python3
"""Benchmark Data Parallelism: nano-megatron DDP vs Megatron-LM DDP.

Same model knobs as scripts/benchmark_tp.py (RoPE, SwiGLU, LayerNorm, FP32).
Measures per-GPU peak memory and global tokens/sec (local_batch * dp_size).
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class BenchmarkResult:
    framework: str
    model_name: str
    tp_size: int
    dp_size: int
    batch_size: int
    seq_len: int
    hidden_size: int
    num_layers: int
    num_heads: int
    tokens_per_sec_local: float
    tokens_per_sec_global: float
    memory_mb: float
    avg_step_time_ms: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--framework",
        type=str,
        choices=["nano", "megatron", "both"],
        default="nano",
        help="Single framework per process. `both` exits with instructions "
        "(run nano and megatron as separate torchrun jobs for fair memory).",
    )
    p.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    p.add_argument(
        "--dp-size",
        type=int,
        default=None,
        help="Data parallel size (default: world_size // tp_size)",
    )
    p.add_argument("--batch-size", type=int, default=2, help="Micro-batch per DP rank")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--hidden-size", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--ffn-hidden-size", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--benchmark-steps", type=int, default=10)
    p.add_argument("--bucket-cap-mb", type=float, default=25.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16"],
        default="fp32",
        help="Compute dtype. bf16 enables FlashAttention on both sides.",
    )
    p.add_argument(
        "--attn-backend",
        type=str,
        choices=["auto", "flash", "unfused"],
        default="auto",
        help="nano attention backend (ignored by Megatron).",
    )
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def _dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.bfloat16 if args.precision == "bf16" else torch.float32


def _resolve_dp(args: argparse.Namespace) -> int:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world % args.tp_size != 0:
        raise ValueError(f"world_size ({world}) not divisible by tp_size ({args.tp_size})")
    inferred = world // args.tp_size
    if args.dp_size is None:
        return inferred
    if args.dp_size != inferred:
        raise ValueError(
            f"dp_size ({args.dp_size}) != world_size/tp_size ({inferred})"
        )
    return args.dp_size


def _ensure_dist(local_rank: int) -> None:
    if dist.is_initialized():
        torch.cuda.set_device(local_rank)
        return
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        rank=int(os.environ.get("RANK", 0)),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
    )


def _time_loop(
    step_fn,
    *,
    warmup: int,
    steps: int,
    device: torch.device,
) -> tuple[float, float]:
    for _ in range(warmup):
        step_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        step_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    mem = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    return elapsed, mem


def benchmark_nano(args: argparse.Namespace, dp_size: int) -> BenchmarkResult:
    from nano_megatron.distributed import DistributedDataParallel
    from nano_megatron.model import build_tp_gpt_from_reference
    from nano_megatron.parallel import (
        ParallelConfig,
        destroy_parallel,
        initialize_parallel,
        is_parallel_initialized,
    )
    from nano_megatron.reference import ReferenceGPT, ReferenceGPTConfig
    from nano_megatron.reference.loss import shifted_cross_entropy

    if is_parallel_initialized():
        destroy_parallel()

    ffn = args.ffn_hidden_size or 4 * args.hidden_size
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_rank0 = int(os.environ.get("RANK", 0)) == 0
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else args.device)

    _ensure_dist(local_rank)
    ctx = initialize_parallel(
        ParallelConfig(
            tensor_parallel_size=args.tp_size,
            data_parallel_size=dp_size,
        ),
        dist_backend="nccl" if device.type == "cuda" else "gloo",
    )

    cfg = ReferenceGPTConfig(
        vocab_size=51200,
        max_seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_hidden_size=ffn,
        layernorm_eps=1e-5,
        use_bias=False,
        position_embedding_type="rope",
        rotary_dim=args.hidden_size // args.num_heads,
        rotary_base=10000,
        activation_func="swiglu",
        gated_linear_unit=True,
        normalization="layernorm",
        use_fused_qkv=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attn_backend=args.attn_backend,
    )

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    dtype = _dtype(args)
    ref = ReferenceGPT(cfg)
    if args.tp_size > 1:
        model = build_tp_gpt_from_reference(ref, ctx).to(device=device, dtype=dtype)
    else:
        model = ref.to(device=device, dtype=dtype)
    ddp = DistributedDataParallel(model, ctx, bucket_cap_mb=args.bucket_cap_mb)
    ddp.train()

    if is_rank0:
        n_params = sum(p.numel() for p in ddp.module.parameters())
        print(
            f"[nano] params/rank={n_params/1e6:.1f}M tp={args.tp_size} dp={dp_size} "
            f"bucket_cap_mb={args.bucket_cap_mb} precision={args.precision} "
            f"attn_backend={cfg.attn_backend}",
            flush=True,
        )

    input_ids = torch.randint(
        0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device
    )

    def _loss(logits, labels):
        if hasattr(ddp.module, "shifted_cross_entropy"):
            return ddp.module.shifted_cross_entropy(logits, labels)
        return shifted_cross_entropy(logits, labels)

    def step() -> None:
        logits = ddp(input_ids)
        loss = _loss(logits, input_ids)
        loss.backward()
        ddp.finish_grad_sync()
        ddp.zero_grad(set_to_none=True)

    elapsed, memory_mb = _time_loop(
        step, warmup=args.warmup_steps, steps=args.benchmark_steps, device=device
    )
    destroy_parallel()

    local_tokens = args.benchmark_steps * args.batch_size * args.seq_len
    tps_local = local_tokens / elapsed
    tps_global = tps_local * dp_size
    tag = "nano-megatron (DP)" if args.tp_size == 1 else "nano-megatron (TP×DP)"
    return BenchmarkResult(
        framework=tag,
        model_name="GPT",
        tp_size=args.tp_size,
        dp_size=dp_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        tokens_per_sec_local=tps_local,
        tokens_per_sec_global=tps_global,
        memory_mb=memory_mb,
        avg_step_time_ms=(elapsed / args.benchmark_steps) * 1000,
    )


def benchmark_megatron(args: argparse.Namespace, dp_size: int) -> BenchmarkResult:
    ffn = args.ffn_hidden_size or 4 * args.hidden_size
    vocab_size = 51200
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_rank0 = int(os.environ.get("RANK", 0)) == 0

    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel as MegatronDDP
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    parallel_state.destroy_model_parallel()
    _ensure_dist(local_rank)

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(42)

    if parallel_state.get_data_parallel_world_size() != dp_size:
        raise RuntimeError(
            f"Megatron dp_size={parallel_state.get_data_parallel_world_size()} "
            f"!= expected {dp_size}"
        )

    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        ffn_hidden_size=ffn,
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        add_qkv_bias=False,
        activation_func=torch.nn.functional.silu,
        gated_linear_unit=True,
        normalization="LayerNorm",
        sequence_parallel=False,
        tp_comm_overlap=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        apply_rope_fusion=False,
        bias_activation_fusion=False,
        masked_softmax_fusion=False,
        bias_dropout_fusion=False,
        fp16=False,
        bf16=(args.precision == "bf16"),
        params_dtype=_dtype(args),
    )

    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=vocab_size,
        max_sequence_length=args.seq_len,
        position_embedding_type="rope",
        rotary_base=10000,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
    ).cuda(local_rank)
    if args.precision == "bf16":
        model = model.bfloat16()

    # Align with nano: mean grads (average_in_collective), no overlap, no ZeRO.
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
        average_in_collective=True,
        bucket_size=None,
    )
    ddp = MegatronDDP(config=config, ddp_config=ddp_config, module=model)
    ddp.train()

    if is_rank0:
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[megatron] params/rank={n_params/1e6:.1f}M tp={args.tp_size} dp={dp_size} "
            f"precision={args.precision}",
            flush=True,
        )

    device = torch.device(f"cuda:{local_rank}")
    input_ids = torch.randint(
        0, vocab_size, (args.batch_size, args.seq_len), device=device
    )
    position_ids = (
        torch.arange(args.seq_len, device=device)
        .unsqueeze(0)
        .expand(args.batch_size, -1)
    )

    def _loss(logits, labels):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return vocab_parallel_cross_entropy(shift_logits, shift_labels).mean()

    def step() -> None:
        logits = ddp(input_ids, position_ids, None)
        loss = _loss(logits, input_ids)
        loss.backward()
        ddp.finish_grad_sync()
        ddp.zero_grad_buffer()
        model.zero_grad(set_to_none=True)

    elapsed, memory_mb = _time_loop(
        step,
        warmup=args.warmup_steps,
        steps=args.benchmark_steps,
        device=device,
    )
    parallel_state.destroy_model_parallel()

    local_tokens = args.benchmark_steps * args.batch_size * args.seq_len
    tps_local = local_tokens / elapsed
    tps_global = tps_local * dp_size
    tag = "Megatron-LM (DP)" if args.tp_size == 1 else "Megatron-LM (TP×DP)"
    return BenchmarkResult(
        framework=tag,
        model_name="GPT",
        tp_size=args.tp_size,
        dp_size=dp_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        tokens_per_sec_local=tps_local,
        tokens_per_sec_global=tps_global,
        memory_mb=memory_mb,
        avg_step_time_ms=(elapsed / args.benchmark_steps) * 1000,
    )


def _print_result(r: BenchmarkResult) -> None:
    print(f"  Tokens/sec (local/GPU):  {r.tokens_per_sec_local:.2f}")
    print(f"  Tokens/sec (global):     {r.tokens_per_sec_global:.2f}")
    print(f"  Memory (MB/GPU):         {r.memory_mb:.2f}")
    print(f"  Avg step time (ms):      {r.avg_step_time_ms:.2f}")


def write_markdown(results: list[BenchmarkResult], path: str) -> None:
    import datetime

    rows = ""
    for r in results:
        rows += (
            f"| {r.framework} | TP{r.tp_size}×DP{r.dp_size} | {r.batch_size} | "
            f"{r.seq_len} | {r.hidden_size} | {r.num_layers} | {r.num_heads} | "
            f"{r.tokens_per_sec_local:.2f} | {r.tokens_per_sec_global:.2f} | "
            f"{r.memory_mb:.2f} | {r.avg_step_time_ms:.2f} |\n"
        )
    analysis = ""
    if len(results) >= 2:
        a, b = results[0], results[1]
        if b.tokens_per_sec_global > 0:
            analysis = (
                f"- Throughput ratio (nano/megatron global): "
                f"{a.tokens_per_sec_global / b.tokens_per_sec_global:.3f}x\n"
                f"- Memory ratio (nano/megatron): "
                f"{a.memory_mb / b.memory_mb:.3f}x\n"
            )
    content = f"""# Data Parallelism Performance Comparison

- **Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **PyTorch**: {torch.__version__}
- **CUDA**: {torch.version.cuda}
- **Precision**: FP32
- **Tokens**: global = local_batch × seq × dp_size / wall_time

| Framework | Parallel | Micro-BS | Seq | Hidden | Layers | Heads | Tok/s local | Tok/s global | Mem MB | Step ms |
|-----------|----------|----------|-----|--------|--------|-------|-------------|--------------|--------|---------|
{rows}
## Analysis

{analysis}
"""
    with open(path, "w") as f:
        f.write(content)


def _cuda_reset() -> None:
    """Drop cached blocks between isolated framework runs."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()


def main() -> None:
    args = parse_args()
    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = 4 * args.hidden_size
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for DP benchmark")

    # Fair peak memory: `both` is not supported in-process. Launch two
    # separate torchrun jobs (`--framework nano` then `--framework megatron`).
    if args.framework == "both":
        raise SystemExit(
            "benchmark_dp.py: use separate processes for fair memory comparison:\n"
            "  torchrun ... scripts/benchmark_dp.py --framework nano ...\n"
            "  torchrun ... scripts/benchmark_dp.py --framework megatron ...\n"
        )

    dp_size = _resolve_dp(args)
    is_rank0 = int(os.environ.get("RANK", 0)) == 0
    results: list[BenchmarkResult] = []

    _cuda_reset()
    if args.framework == "nano":
        if is_rank0:
            print("Benchmarking nano-megatron DDP...", flush=True)
        r = benchmark_nano(args, dp_size)
        results.append(r)
        if is_rank0:
            _print_result(r)
    elif args.framework == "megatron":
        if is_rank0:
            print("Benchmarking Megatron-LM DDP...", flush=True)
        r = benchmark_megatron(args, dp_size)
        results.append(r)
        if is_rank0:
            _print_result(r)

    if is_rank0 and results and args.output:
        write_markdown(results, args.output)
        print(f"\nResults written to {args.output}", flush=True)

    _cuda_reset()
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
