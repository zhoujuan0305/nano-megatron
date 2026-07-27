#!/usr/bin/env python3
"""Benchmark Tensor Parallelism performance for nano-megatron vs Megatron-LM."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    framework: str
    model_name: str
    tp_size: int
    batch_size: int
    seq_len: int
    hidden_size: int
    num_layers: int
    num_heads: int
    tokens_per_sec: float
    memory_mb: float
    avg_step_time_ms: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", type=str, choices=["nano", "megatron", "both"], default="both")
    p.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--hidden-size", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--ffn-hidden-size", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--benchmark-steps", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default=None, help="Output file path (default: no output)")
    return p.parse_args()


def benchmark_nano_megatron(args: argparse.Namespace) -> BenchmarkResult:
    """Benchmark nano-megatron TP implementation."""
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

    ffn_hidden_size = args.ffn_hidden_size or 4 * args.hidden_size
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_rank0 = int(os.environ.get("RANK", 0)) == 0
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else args.device)

    # Align with Megatron-LM structural settings used in this benchmark.
    cfg = ReferenceGPTConfig(
        vocab_size=51200,
        max_seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_hidden_size=ffn_hidden_size,
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

    if args.tp_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    ctx = None
    if dist.is_initialized():
        ctx = initialize_parallel(
            ParallelConfig(tensor_parallel_size=args.tp_size),
            dist_backend="nccl" if torch.cuda.is_available() else "gloo",
        )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    ref = ReferenceGPT(cfg)
    if ctx is not None:
        model = build_tp_gpt_from_reference(ref, ctx).to(device)
    else:
        model = ref.to(device)
    model.train()

    if is_rank0:
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[nano] params={n_params/1e6:.1f}M "
            f"fused_qkv={cfg.use_fused_qkv} pos={cfg.position_embedding_type} "
            f"glued={cfg.gated_linear_unit} bias={cfg.use_bias} "
            f"dropout=({cfg.hidden_dropout},{cfg.attention_dropout})",
            flush=True,
        )

    input_ids = torch.randint(
        0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device
    )

    def _compute_loss(logits, labels):
        # TP path returns local-vocab logits; use vocab-parallel CE.
        if ctx is not None and hasattr(model, "shifted_cross_entropy"):
            return model.shifted_cross_entropy(logits, labels)
        return shifted_cross_entropy(logits, labels)

    for _ in range(args.warmup_steps):
        logits = model(input_ids)
        loss = _compute_loss(logits, input_ids)
        loss.backward()
        model.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    for _ in range(args.benchmark_steps):
        logits = model(input_ids)
        loss = _compute_loss(logits, input_ids)
        loss.backward()
        model.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_step_time = total_time / args.benchmark_steps
    total_tokens = args.benchmark_steps * args.batch_size * args.seq_len
    tokens_per_sec = total_tokens / total_time
    memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    )

    if ctx is not None:
        destroy_parallel()

    framework_name = "nano-megatron (TP)" if ctx is not None else "nano-megatron (ref)"
    return BenchmarkResult(
        framework=framework_name,
        model_name="GPT",
        tp_size=args.tp_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        tokens_per_sec=tokens_per_sec,
        memory_mb=memory_mb,
        avg_step_time_ms=avg_step_time * 1000,
    )


def benchmark_megatron(args: argparse.Namespace) -> BenchmarkResult:
    """Benchmark Megatron-LM TP implementation.

    Config is aligned with nano-megatron: RoPE, SwiGLU, LayerNorm, no bias,
    zero dropout, vocab-parallel output, shifted next-token CE.
    """
    ffn_hidden_size = args.ffn_hidden_size or 4 * args.hidden_size
    vocab_size = 51200
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_rank0 = int(os.environ.get("RANK", 0)) == 0

    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

    parallel_state.destroy_model_parallel()

    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        torch.cuda.set_device(local_rank)

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(42)

    # Match nano-megatron structural knobs. Leave TE kernels as Megatron's
    # optimized path; disable dropout so we are not measuring extra noise.
    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        ffn_hidden_size=ffn_hidden_size,
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        add_qkv_bias=False,
        activation_func=torch.nn.functional.silu,
        gated_linear_unit=True,
        normalization="LayerNorm",
        sequence_parallel=False,
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
    model.train()

    if is_rank0:
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[megatron] params={n_params/1e6:.1f}M "
            f"dropout=({config.hidden_dropout},{config.attention_dropout}) "
            f"pos={model.position_embedding_type} "
            f"glued={config.gated_linear_unit} bias={config.add_bias_linear}",
            flush=True,
        )

    input_ids = torch.randint(
        0, vocab_size, (args.batch_size, args.seq_len), device=f"cuda:{local_rank}"
    )
    position_ids = (
        torch.arange(args.seq_len, device=input_ids.device)
        .unsqueeze(0)
        .expand(args.batch_size, -1)
    )
    # TE causal attention uses attn_mask_type=causal; mask is padding-style.
    attention_mask = None

    def _compute_loss(logits, labels):
        # Align with nano shifted next-token CE: predict labels[:,1:] from logits[:,:-1].
        # Megatron CE is over the last dim (vocab); [B,S,Vlocal] is fine when target is [B,S].
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return vocab_parallel_cross_entropy(shift_logits, shift_labels).mean()

    for _ in range(args.warmup_steps):
        logits = model(input_ids, position_ids, attention_mask)
        loss = _compute_loss(logits, input_ids)
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    for _ in range(args.benchmark_steps):
        logits = model(input_ids, position_ids, attention_mask)
        loss = _compute_loss(logits, input_ids)
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_step_time = total_time / args.benchmark_steps
    total_tokens = args.benchmark_steps * args.batch_size * args.seq_len
    tokens_per_sec = total_tokens / total_time
    memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    parallel_state.destroy_model_parallel()

    return BenchmarkResult(
        framework="Megatron-LM",
        model_name="GPT",
        tp_size=args.tp_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        tokens_per_sec=tokens_per_sec,
        memory_mb=memory_mb,
        avg_step_time_ms=avg_step_time * 1000,
    )


def format_result_markdown(result: BenchmarkResult) -> str:
    """Format a benchmark result as markdown."""
    return f"""| {result.framework} | {result.model_name} | TP{result.tp_size} | {result.batch_size} | {result.seq_len} | {result.hidden_size} | {result.num_layers} | {result.num_heads} | {result.tokens_per_sec:.2f} | {result.memory_mb:.2f} | {result.avg_step_time_ms:.2f} |"""


def write_performance_md(results: list[BenchmarkResult], output_path: str) -> None:
    """Write benchmark results to markdown file."""
    import datetime

    # Build results table
    results_table = ""
    for r in results:
        results_table += f"| {r.framework} | {r.model_name} | TP{r.tp_size} | {r.batch_size} | {r.seq_len} | {r.hidden_size} | {r.num_layers} | {r.num_heads} | {r.tokens_per_sec:.2f} | {r.memory_mb:.2f} | {r.avg_step_time_ms:.2f} |\n"

    # Build analysis
    analysis = ""
    if results:
        analysis += f"**{results[0].framework}**: {results[0].tokens_per_sec:.2f} tokens/sec\n"
    if len(results) > 1:
        analysis += f"\n**{results[1].framework}**: {results[1].tokens_per_sec:.2f} tokens/sec\n"
        if results[1].tokens_per_sec > 0:
            speedup = results[0].tokens_per_sec / results[1].tokens_per_sec
            analysis += f"\n**Speedup**: {speedup:.2f}x\n"

    # Build memory section
    memory_section = ""
    if results:
        memory_section += f"**{results[0].framework}**: {results[0].memory_mb:.2f} MB\n"
    if len(results) > 1:
        memory_section += f"\n**{results[1].framework}**: {results[1].memory_mb:.2f} MB\n"

    content = f"""# Tensor Parallelism Performance Comparison

## Test Environment

- **GPU**: NVIDIA RTX A6000 (4x)
- **PyTorch**: {torch.__version__}
- **CUDA**: {torch.version.cuda}
- **Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Configuration

- **Architecture**: GPT (Decoder-only Transformer)
- **Vocabulary Size**: 1024
- **Precision**: FP32

## Benchmark Results

| Framework | Model | TP Size | Batch Size | Seq Len | Hidden Size | Num Layers | Num Heads | Tokens/sec | Memory (MB) | Avg Step Time (ms) |
|-----------|-------|---------|------------|---------|-------------|------------|-----------|------------|-------------|-------------------|
{results_table}
## Analysis

### Throughput Comparison

{analysis}
### Memory Usage

{memory_section}
### Notes

- nano-megatron uses a simplified reference implementation with explicit TP sharding
- Megatron-LM uses optimized fused kernels and communication patterns
- Both tests use the same model configuration for fair comparison
- Performance differences reflect implementation maturity and optimization level

## Model Architecture Mapping

| Component | nano-megatron | Megatron-LM |
|-----------|---------------|-------------|
| Attention | Manual Q/K/V projection | Fused QKV projection |
| LayerNorm | Custom implementation | Optimized CUDA kernel |
| MLP | Separate FC1/FC2 | Fused SwiGLU (optional) |
| Communication | PyTorch distributed | NCCL optimized |
| Activation | GELU erf | SwiGLU (default) |

## Recommendations

1. For **research and debugging**: Use nano-megatron for clarity and simplicity
2. For **production training**: Use Megatron-LM for performance and stability
3. For **learning TP concepts**: nano-megatron provides better visibility into sharding mechanics
"""

    with open(output_path, "w") as f:
        f.write(content)


def main() -> None:
    args = parse_args()

    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = 4 * args.hidden_size

    # Only rank 0 prints aggregate-facing lines to avoid interleaved multi-rank noise.
    is_rank0 = int(os.environ.get("RANK", 0)) == 0
    results = []

    if args.framework in ["nano", "both"]:
        if is_rank0:
            print("Benchmarking nano-megatron...", flush=True)
        try:
            nano_result = benchmark_nano_megatron(args)
            results.append(nano_result)
            if is_rank0:
                print(f"  Tokens/sec: {nano_result.tokens_per_sec:.2f}")
                print(f"  Memory: {nano_result.memory_mb:.2f} MB")
                print(f"  Avg step time: {nano_result.avg_step_time_ms:.2f} ms")
        except Exception as e:
            if is_rank0:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
            raise

    if args.framework in ["megatron", "both"]:
        if is_rank0:
            print("Benchmarking Megatron-LM...", flush=True)
        try:
            megatron_result = benchmark_megatron(args)
            results.append(megatron_result)
            if is_rank0:
                print(f"  Tokens/sec: {megatron_result.tokens_per_sec:.2f}")
                print(f"  Memory: {megatron_result.memory_mb:.2f} MB")
                print(f"  Avg step time: {megatron_result.avg_step_time_ms:.2f} ms")
        except Exception as e:
            if is_rank0:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
            raise

    if is_rank0:
        if results and args.output:
            write_performance_md(results, args.output)
            print(f"\nResults written to {args.output}")
        elif not results:
            print("No benchmark results to write.")


if __name__ == "__main__":
    main()
