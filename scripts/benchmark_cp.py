#!/usr/bin/env python3
"""Benchmark Context Parallelism: nano-megatron vs Megatron-LM.

Same model knobs as scripts/benchmark_tp.py / benchmark_dp.py
(RoPE, SwiGLU, LayerNorm, FP32, fused QKV).

Tokens accounting: CP splits sequence work, not data replicas.
  global_tok/s = (benchmark_steps * batch_size * seq_len * dp_size) / wall_time

Fair peak memory: run nano and megatron as separate torchrun jobs
(`--framework both` is rejected).

Megatron CP uses zigzag load-balancing, so seq_len %% (2 * cp_size) == 0.
nano CP uses contiguous shards; seq_len %% cp_size == 0 is enough (still
enforce 2*cp for head-to-head comparison with Megatron).
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
    cp_size: int
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
        help="Single framework per process. `both` exits with instructions.",
    )
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--cp-size", type=int, default=2)
    p.add_argument(
        "--dp-size",
        type=int,
        default=None,
        help="Data parallel size (default: world // (tp * cp))",
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
    p.add_argument(
        "--precision",
        type=str,
        choices=["bf16", "fp32"],
        default="bf16",
        help="Compute dtype. Megatron TE CP requires bf16/fp16; default bf16 "
        "for a fair nano vs Megatron CP comparison.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def _dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.bfloat16 if args.precision == "bf16" else torch.float32


def _resolve_dp(args: argparse.Namespace) -> int:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    base = args.tp_size * args.cp_size
    if world % base != 0:
        raise ValueError(
            f"world_size ({world}) not divisible by tp*cp ({base})"
        )
    inferred = world // base
    if args.dp_size is None:
        return inferred
    if args.dp_size != inferred:
        raise ValueError(
            f"dp_size ({args.dp_size}) != world_size/(tp*cp) ({inferred})"
        )
    return args.dp_size


def _validate_seq(args: argparse.Namespace) -> None:
    if args.cp_size < 1:
        raise ValueError("cp_size must be >= 1")
    # Align with Megatron zigzag requirement for fair comparison.
    if args.seq_len % (2 * args.cp_size) != 0:
        raise ValueError(
            f"seq_len ({args.seq_len}) must be divisible by 2*cp_size "
            f"({2 * args.cp_size}) for Megatron-compatible CP"
        )


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


def _tag(framework: str, tp: int, cp: int, dp: int) -> str:
    parts = []
    if tp > 1:
        parts.append(f"TP{tp}")
    if cp > 1:
        parts.append(f"CP{cp}")
    if dp > 1:
        parts.append(f"DP{dp}")
    if not parts:
        parts.append("baseline")
    return f"{framework} ({'×'.join(parts)})"


def benchmark_nano(
    args: argparse.Namespace, dp_size: int, cp_size: int
) -> BenchmarkResult:
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
            context_parallel_size=cp_size,
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
    )

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    dtype = _dtype(args)
    ref = ReferenceGPT(cfg)
    model = build_tp_gpt_from_reference(ref, ctx).to(device=device, dtype=dtype)
    # cp>1 requires DDP over data_context_parallel_group for correct grads.
    ddp = DistributedDataParallel(model, ctx, bucket_cap_mb=args.bucket_cap_mb)
    ddp.train()

    if is_rank0:
        n_params = sum(p.numel() for p in ddp.module.parameters())
        print(
            f"[nano] params/rank={n_params/1e6:.1f}M "
            f"tp={args.tp_size} cp={cp_size} dp={dp_size} precision={args.precision}",
            flush=True,
        )

    # Full sequence on every CP rank; model scatters internally.
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

    # Tokens counted on full sequence; CP does not multiply data volume.
    tokens = args.benchmark_steps * args.batch_size * args.seq_len
    tps_local = tokens / elapsed
    tps_global = tps_local * dp_size
    return BenchmarkResult(
        framework=_tag("nano-megatron", args.tp_size, cp_size, dp_size),
        model_name="GPT",
        tp_size=args.tp_size,
        cp_size=cp_size,
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


def benchmark_megatron(
    args: argparse.Namespace, dp_size: int, cp_size: int
) -> BenchmarkResult:
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
    from megatron.core.utils import get_batch_on_this_cp_rank

    parallel_state.destroy_model_parallel()
    _ensure_dist(local_rank)

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
    )
    model_parallel_cuda_manual_seed(42)

    got_dp = parallel_state.get_data_parallel_world_size()
    got_cp = parallel_state.get_context_parallel_world_size()
    if got_dp != dp_size or got_cp != cp_size:
        raise RuntimeError(
            f"Megatron dp/cp={got_dp}/{got_cp} != expected {dp_size}/{cp_size}"
        )

    dtype = _dtype(args)
    if dtype == torch.float32:
        raise RuntimeError(
            "Megatron TE context parallel requires bf16/fp16 "
            "(FlashAttention/FusedAttention). Use --precision bf16."
        )

    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        ffn_hidden_size=ffn,
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
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
        bf16=True,
        params_dtype=torch.bfloat16,
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
    ).cuda(local_rank).to(dtype=torch.bfloat16)

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
            f"[megatron] params/rank={n_params/1e6:.1f}M "
            f"tp={args.tp_size} cp={cp_size} dp={dp_size} precision={args.precision}",
            flush=True,
        )

    device = torch.device(f"cuda:{local_rank}")
    full_ids = torch.randint(
        0, vocab_size, (args.batch_size, args.seq_len), device=device
    )
    full_pos = (
        torch.arange(args.seq_len, device=device)
        .unsqueeze(0)
        .expand(args.batch_size, -1)
        .contiguous()
    )

    def step() -> None:
        # Zigzag CP slice of tokens/positions (Megatron load-balance).
        batch = get_batch_on_this_cp_rank(
            {"tokens": full_ids, "labels": full_ids, "position_ids": full_pos}
        )
        tokens = batch["tokens"]
        labels = batch["labels"]
        position_ids = batch["position_ids"]
        logits = ddp(tokens, position_ids, None)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = vocab_parallel_cross_entropy(shift_logits, shift_labels).mean()
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

    tokens = args.benchmark_steps * args.batch_size * args.seq_len
    tps_local = tokens / elapsed
    tps_global = tps_local * dp_size
    return BenchmarkResult(
        framework=_tag("Megatron-LM", args.tp_size, cp_size, dp_size),
        model_name="GPT",
        tp_size=args.tp_size,
        cp_size=cp_size,
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
    print(f"  Framework:               {r.framework}")
    print(f"  Parallel:                TP{r.tp_size}×CP{r.cp_size}×DP{r.dp_size}")
    print(f"  Tokens/sec (local):      {r.tokens_per_sec_local:.2f}")
    print(f"  Tokens/sec (global):     {r.tokens_per_sec_global:.2f}")
    print(f"  Memory (MB/GPU):         {r.memory_mb:.2f}")
    print(f"  Avg step time (ms):      {r.avg_step_time_ms:.2f}")


def write_markdown(results: list[BenchmarkResult], path: str) -> None:
    import datetime

    rows = ""
    for r in results:
        rows += (
            f"| {r.framework} | TP{r.tp_size}×CP{r.cp_size}×DP{r.dp_size} | "
            f"{r.batch_size} | {r.seq_len} | {r.hidden_size} | {r.num_layers} | "
            f"{r.num_heads} | {r.tokens_per_sec_local:.2f} | "
            f"{r.tokens_per_sec_global:.2f} | {r.memory_mb:.2f} | "
            f"{r.avg_step_time_ms:.2f} |\n"
        )
    content = f"""# Context Parallelism Performance Comparison

- **Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **PyTorch**: {torch.__version__}
- **CUDA**: {torch.version.cuda}
- **Precision**: FP32
- **Tokens**: global = batch × seq × dp / wall (CP does not multiply data)

| Framework | Parallel | Micro-BS | Seq | Hidden | Layers | Heads | Tok/s local | Tok/s global | Mem MB | Step ms |
|-----------|----------|----------|-----|--------|--------|-------|-------------|--------------|--------|---------|
{rows}
"""
    with open(path, "w") as f:
        f.write(content)


def _cuda_reset() -> None:
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
        raise RuntimeError("CUDA required for CP benchmark")
    if args.framework == "both":
        raise SystemExit(
            "benchmark_cp.py: use separate processes for fair memory comparison:\n"
            "  torchrun ... scripts/benchmark_cp.py --framework nano ...\n"
            "  torchrun ... scripts/benchmark_cp.py --framework megatron ...\n"
        )

    _validate_seq(args)
    dp_size = _resolve_dp(args)
    cp_size = args.cp_size
    is_rank0 = int(os.environ.get("RANK", 0)) == 0
    results: list[BenchmarkResult] = []

    _cuda_reset()
    if args.framework == "nano":
        if is_rank0:
            print("Benchmarking nano-megatron CP...", flush=True)
        r = benchmark_nano(args, dp_size, cp_size)
        results.append(r)
        if is_rank0:
            _print_result(r)
    elif args.framework == "megatron":
        if is_rank0:
            print("Benchmarking Megatron-LM CP...", flush=True)
        r = benchmark_megatron(args, dp_size, cp_size)
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
