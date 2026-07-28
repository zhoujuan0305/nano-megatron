#!/usr/bin/env python3
"""Benchmark Pipeline Parallelism: nano-megatron 1F1B vs optional Megatron-LM.

Same model knobs as scripts/benchmark_dp.py (RoPE, SwiGLU, LayerNorm, FP32).
Measures per-GPU peak memory and tokens/sec (local batch is the full
microbatch sum on each DP rank; global = local × dp_size).
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
    pp_size: int
    tp_size: int
    dp_size: int
    num_microbatches: int
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
        help="Framework(s) to benchmark. Megatron path is best-effort "
        "(skipped if import/setup fails).",
    )
    p.add_argument("--pp-size", type=int, default=2, help="Pipeline parallel size")
    p.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    p.add_argument(
        "--dp-size",
        type=int,
        default=None,
        help="Data parallel size (default: world_size // (tp_size * pp_size))",
    )
    p.add_argument(
        "--num-microbatches",
        type=int,
        default=4,
        help="Number of microbatches (batch-size must be divisible)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Local batch per DP rank (sum of microbatches)",
    )
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--hidden-size", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--ffn-hidden-size", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--benchmark-steps", type=int, default=10)
    p.add_argument("--bucket-cap-mb", type=float, default=25.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def _resolve_dp(args: argparse.Namespace) -> int:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    base = args.tp_size * args.pp_size
    if world % base != 0:
        raise ValueError(
            f"world_size ({world}) not divisible by tp*pp ({base})"
        )
    inferred = world // base
    if args.dp_size is None:
        return inferred
    if args.dp_size != inferred:
        raise ValueError(
            f"dp_size ({args.dp_size}) != world_size/(tp*pp) ({inferred})"
        )
    return args.dp_size


def _validate_args(args: argparse.Namespace) -> None:
    if args.pp_size < 1:
        raise ValueError(f"pp-size must be >= 1, got {args.pp_size}")
    if args.tp_size < 1:
        raise ValueError(f"tp-size must be >= 1, got {args.tp_size}")
    if args.num_microbatches < 1:
        raise ValueError(
            f"num-microbatches must be >= 1, got {args.num_microbatches}"
        )
    if args.batch_size % args.num_microbatches != 0:
        raise ValueError(
            f"batch-size ({args.batch_size}) must be divisible by "
            f"num-microbatches ({args.num_microbatches})"
        )
    if args.num_layers % args.pp_size != 0:
        raise ValueError(
            f"num-layers ({args.num_layers}) must be divisible by "
            f"pp-size ({args.pp_size})"
        )


def _ensure_dist(local_rank: int) -> None:
    if dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return
    if torch.cuda.is_available():
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


def _make_result(
    *,
    framework: str,
    args: argparse.Namespace,
    dp_size: int,
    elapsed: float,
    memory_mb: float,
) -> BenchmarkResult:
    local_tokens = args.benchmark_steps * args.batch_size * args.seq_len
    tps_local = local_tokens / elapsed if elapsed > 0 else 0.0
    tps_global = tps_local * dp_size
    return BenchmarkResult(
        framework=framework,
        model_name="GPT",
        pp_size=args.pp_size,
        tp_size=args.tp_size,
        dp_size=dp_size,
        num_microbatches=args.num_microbatches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        tokens_per_sec_local=tps_local,
        tokens_per_sec_global=tps_global,
        memory_mb=memory_mb,
        avg_step_time_ms=(elapsed / args.benchmark_steps) * 1000
        if args.benchmark_steps > 0
        else 0.0,
    )


def benchmark_nano(args: argparse.Namespace, dp_size: int) -> BenchmarkResult:
    from nano_megatron.distributed import DistributedDataParallel
    from nano_megatron.model import build_pipeline_stage_from_reference
    from nano_megatron.parallel import (
        ParallelConfig,
        destroy_parallel,
        initialize_parallel,
        is_parallel_initialized,
    )
    from nano_megatron.reference import ReferenceGPT, ReferenceGPTConfig
    from nano_megatron.schedules import forward_backward_1f1b

    if is_parallel_initialized():
        destroy_parallel()

    ffn = args.ffn_hidden_size or 4 * args.hidden_size
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_rank0 = int(os.environ.get("RANK", 0)) == 0
    device = torch.device(
        f"cuda:{local_rank}" if args.device == "cuda" else args.device
    )

    _ensure_dist(local_rank)
    ctx = initialize_parallel(
        ParallelConfig(
            tensor_parallel_size=args.tp_size,
            data_parallel_size=dp_size,
            pipeline_parallel_size=args.pp_size,
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
        tie_word_embeddings=False,
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

    ref = ReferenceGPT(cfg)
    stage = build_pipeline_stage_from_reference(ref, ctx).to(device)

    ddp = None
    if dp_size > 1:
        ddp = DistributedDataParallel(stage, ctx, bucket_cap_mb=args.bucket_cap_mb)
        stage_mod = ddp.module
        train_mod = ddp
    else:
        stage_mod = stage
        train_mod = stage
    train_mod.train()

    if is_rank0:
        n_params = sum(p.numel() for p in stage_mod.parameters())
        print(
            f"[nano] params/rank={n_params/1e6:.1f}M "
            f"pp={args.pp_size} tp={args.tp_size} dp={dp_size} "
            f"microbatches={args.num_microbatches} "
            f"bucket_cap_mb={args.bucket_cap_mb}",
            flush=True,
        )

    input_ids = torch.randint(
        0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device
    )
    labels = input_ids.clone()

    def step() -> None:
        if ddp is not None:
            ddp.zero_grad(set_to_none=True)
        else:
            stage_mod.zero_grad(set_to_none=True)
        forward_backward_1f1b(
            stage=stage_mod,
            ctx=ctx,
            input_ids=input_ids,
            labels=labels,
            num_microbatches=args.num_microbatches,
            ddp=ddp,
        )

    elapsed, memory_mb = _time_loop(
        step, warmup=args.warmup_steps, steps=args.benchmark_steps, device=device
    )
    destroy_parallel()

    tag_parts = ["nano-megatron"]
    dims = []
    if args.pp_size > 1:
        dims.append("PP")
    if args.tp_size > 1:
        dims.append("TP")
    if dp_size > 1:
        dims.append("DP")
    if dims:
        tag = f"{tag_parts[0]} ({'×'.join(dims)})"
    else:
        tag = f"{tag_parts[0]} (single)"
    return _make_result(
        framework=tag,
        args=args,
        dp_size=dp_size,
        elapsed=elapsed,
        memory_mb=memory_mb,
    )


def benchmark_megatron(args: argparse.Namespace, dp_size: int) -> BenchmarkResult:
    """Megatron-LM non-interleaved 1F1B PP path (same structural knobs as nano).

    Requires Megatron-LM on ``PYTHONPATH``. Uses
    ``get_forward_backward_func`` / ``forward_backward_pipelining_without_interleaving``.
    """
    try:
        from megatron.core import parallel_state
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.pipeline_parallel import get_forward_backward_func
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.transformer_config import TransformerConfig
    except ImportError as exc:
        raise ImportError(
            "Megatron-LM not importable (is it on PYTHONPATH?)"
        ) from exc

    ffn = args.ffn_hidden_size or 4 * args.hidden_size
    vocab_size = 51200
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_rank0 = int(os.environ.get("RANK", 0)) == 0
    micro_batch_size = args.batch_size // args.num_microbatches
    device = torch.device(f"cuda:{local_rank}")

    try:
        parallel_state.destroy_model_parallel()
    except Exception:
        pass
    _ensure_dist(local_rank)

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(42)

    if parallel_state.get_data_parallel_world_size() != dp_size:
        raise RuntimeError(
            f"Megatron dp_size={parallel_state.get_data_parallel_world_size()} "
            f"!= expected {dp_size}"
        )
    if parallel_state.get_pipeline_model_parallel_world_size() != args.pp_size:
        raise RuntimeError(
            f"Megatron pp_size="
            f"{parallel_state.get_pipeline_model_parallel_world_size()} "
            f"!= expected {args.pp_size}"
        )

    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        ffn_hidden_size=ffn,
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        context_parallel_size=1,
        pipeline_dtype=torch.float32,
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
        bf16=False,
        params_dtype=torch.float32,
        deallocate_pipeline_outputs=False,
    )

    is_first = parallel_state.is_pipeline_first_stage()
    is_last = parallel_state.is_pipeline_last_stage()
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=vocab_size,
        max_sequence_length=args.seq_len,
        pre_process=is_first,
        post_process=is_last,
        position_embedding_type="rope",
        rotary_base=10000,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
    ).cuda(local_rank)
    model.train()

    if is_rank0:
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[megatron] params/rank={n_params/1e6:.1f}M "
            f"pp={args.pp_size} tp={args.tp_size} dp={dp_size} "
            f"microbatches={args.num_microbatches} "
            f"micro_bs={micro_batch_size}",
            flush=True,
        )

    # One full local batch; iterator yields microbatch slices each schedule.
    full_ids = torch.randint(
        0, vocab_size, (args.batch_size, args.seq_len), device=device
    )
    full_pos = (
        torch.arange(args.seq_len, device=device)
        .unsqueeze(0)
        .expand(args.batch_size, -1)
        .contiguous()
    )
    full_labels = full_ids.clone()

    forward_backward_func = get_forward_backward_func()

    def _make_iterator():
        """Yield microbatches for one global step (num_microbatches items)."""

        def gen():
            for i in range(args.num_microbatches):
                s = i * micro_batch_size
                e = s + micro_batch_size
                yield {
                    "tokens": full_ids[s:e],
                    "position_ids": full_pos[s:e],
                    "labels": full_labels[s:e],
                }

        return gen()

    def forward_step_func(data_iterator, model_chunk):
        batch = next(data_iterator)
        tokens = batch["tokens"]
        position_ids = batch["position_ids"]
        # Intermediate stages ignore token embeddings (set_input_tensor);
        # last stage uses labels for LM loss.
        labels = batch["labels"] if is_last else None
        output_tensor = model_chunk(tokens, position_ids, None, labels=labels)

        def loss_func(output_tensor):
            # With labels, GPTModel returns per-token loss; mean matches nano CE.
            loss = output_tensor.float().mean()
            return loss, {"lm loss": loss.detach()}

        return output_tensor, loss_func

    def step() -> None:
        model.zero_grad(set_to_none=True)
        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=_make_iterator(),
            model=model,
            num_microbatches=args.num_microbatches,
            seq_length=args.seq_len,
            micro_batch_size=micro_batch_size,
            forward_only=False,
        )

    elapsed, memory_mb = _time_loop(
        step,
        warmup=args.warmup_steps,
        steps=args.benchmark_steps,
        device=device,
    )
    parallel_state.destroy_model_parallel()

    dims = []
    if args.pp_size > 1:
        dims.append("PP")
    if args.tp_size > 1:
        dims.append("TP")
    if dp_size > 1:
        dims.append("DP")
    tag = (
        f"Megatron-LM ({'×'.join(dims)})" if dims else "Megatron-LM (single)"
    )
    return _make_result(
        framework=tag,
        args=args,
        dp_size=dp_size,
        elapsed=elapsed,
        memory_mb=memory_mb,
    )


def _print_result(r: BenchmarkResult) -> None:
    print(
        f"  Parallel: PP{r.pp_size}×TP{r.tp_size}×DP{r.dp_size} "
        f"M={r.num_microbatches}"
    )
    print(f"  Tokens/sec (local/GPU):  {r.tokens_per_sec_local:.2f}")
    print(f"  Tokens/sec (global):     {r.tokens_per_sec_global:.2f}")
    print(f"  Memory (MB/GPU):         {r.memory_mb:.2f}")
    print(f"  Avg step time (ms):      {r.avg_step_time_ms:.2f}")


def write_markdown(results: list[BenchmarkResult], path: str) -> None:
    import datetime

    rows = ""
    for r in results:
        rows += (
            f"| {r.framework} | PP{r.pp_size}×TP{r.tp_size}×DP{r.dp_size} | "
            f"{r.num_microbatches} | {r.batch_size} | {r.seq_len} | "
            f"{r.hidden_size} | {r.num_layers} | {r.num_heads} | "
            f"{r.tokens_per_sec_local:.2f} | {r.tokens_per_sec_global:.2f} | "
            f"{r.memory_mb:.2f} | {r.avg_step_time_ms:.2f} |\n"
        )
    analysis = ""
    if len(results) >= 2:
        a, b = results[0], results[1]
        if b.tokens_per_sec_global > 0:
            analysis = (
                f"- Throughput ratio (first/second global): "
                f"{a.tokens_per_sec_global / b.tokens_per_sec_global:.3f}x\n"
                f"- Memory ratio (first/second): "
                f"{a.memory_mb / max(b.memory_mb, 1e-9):.3f}x\n"
            )
    content = f"""# Pipeline Parallelism Performance Comparison

- **Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **PyTorch**: {torch.__version__}
- **CUDA**: {torch.version.cuda}
- **Precision**: FP32
- **Tokens**: global = local_batch × seq × dp_size / wall_time
- **Schedule**: non-interleaved 1F1B

| Framework | Parallel | Microbatches | Local-BS | Seq | Hidden | Layers | Heads | Tok/s local | Tok/s global | Mem MB | Step ms |
|-----------|----------|--------------|----------|-----|--------|--------|-------|-------------|--------------|--------|---------|
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
    _validate_args(args)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for PP benchmark (use torchrun on GPU hosts). "
            "CPU/gloo path is covered by unit/distributed tests, not this script."
        )

    dp_size = _resolve_dp(args)
    is_rank0 = int(os.environ.get("RANK", 0)) == 0
    results: list[BenchmarkResult] = []

    if args.framework in ("nano", "both"):
        _cuda_reset()
        if is_rank0:
            print("Benchmarking nano-megatron PP (1F1B)...", flush=True)
        r = benchmark_nano(args, dp_size)
        results.append(r)
        if is_rank0:
            _print_result(r)

    if args.framework in ("megatron", "both"):
        _cuda_reset()
        if is_rank0:
            print("Benchmarking Megatron-LM PP (best-effort)...", flush=True)
        try:
            r = benchmark_megatron(args, dp_size)
            results.append(r)
            if is_rank0:
                _print_result(r)
        except Exception as exc:  # noqa: BLE001 — best-effort path
            if is_rank0:
                print(
                    f"  Megatron PP path skipped: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            if args.framework == "megatron":
                raise SystemExit(
                    "Megatron PP benchmark failed and --framework=megatron "
                    "was requested (no nano fallback)."
                ) from exc

    if is_rank0 and results and args.output:
        write_markdown(results, args.output)
        print(f"\nResults written to {args.output}", flush=True)

    _cuda_reset()
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
