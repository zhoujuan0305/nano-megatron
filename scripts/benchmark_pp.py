#!/usr/bin/env python3
"""Benchmark Pipeline Parallelism: nano-megatron 1F1B vs optional Megatron-LM.

Same model knobs as scripts/benchmark_dp.py (RoPE, SwiGLU, LayerNorm).
Measures per-GPU peak memory and tokens/sec (local batch is the full
microbatch sum on each DP rank; global = local × dp_size).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass
from functools import wraps

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
    cuda_graphs: bool = False
    deterministic: bool = False
    extra_warmup_updates: int = 0


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
    p.add_argument(
        "--deterministic", action="store_true",
        help="Process-local deterministic PyTorch/FlashAttention execution (nano only).",
    )
    p.add_argument(
        "--cuda-graphs", action="store_true",
        help="Experimental full-step replay; requires --deterministic and CUDA TP2/PP2/SP BF16 flash.",
    )
    p.add_argument(
        "--cuda-graph-warmup-steps", type=int, default=50,
        help="Additional real optimizer updates after capture, outside timing; reported separately.",
    )
    p.add_argument("--bucket-cap-mb", type=float, default=25.0)
    p.add_argument(
        "--fuse-wgrad-accum", action="store_true",
        help="Nano TP only: direct wgrad accumulation during explicit training backward.",
    )
    p.add_argument(
        "--tp-attention-views", action="store_true",
        help="Nano only: allow TP attention layout views on the CP1 Flash path.",
    )
    p.add_argument(
        "--disable-bucketing-after-first-stage",
        action="store_true",
        help=(
            "Nano only: use one DP gradient bucket per contiguous dtype run on "
            "pipeline stages after stage 0. Keeps final-backward async reduction; "
            "may reduce communication/PP contention at the cost of DP overlap."
        ),
    )
    p.add_argument(
        "--optimizer",
        choices=["none", "sgd", "adamw"],
        default="none",
        help="Optional parameter update included in each measured step.",
    )
    p.add_argument("--learning-rate", type=float, default=1.0e-4)
    p.add_argument(
        "--collective-backend",
        "--dp-backend",
        dest="collective_backend",
        choices=["torch", "nano-nccl"],
        default="torch",
        help=(
            "Backend for active TP/SP and DP collectives. --dp-backend is a "
            "deprecated alias. PP point-to-point remains on PyTorch distributed."
        ),
    )
    p.add_argument(
        "--nano-nccl-library",
        type=str,
        default=None,
        help="Path to ABI-v2 libnano_nccl_mpi_c.so for Nano collectives.",
    )
    p.add_argument(
        "--nano-nccl-transport",
        choices=["auto", "shm", "p2p", "socket", "rdma"],
        default="rdma",
    )
    p.add_argument(
        "--nano-nccl-channels",
        type=int,
        default=4,
        help="Expected channel count of the supplied Nano NCCL library.",
    )
    p.add_argument(
        "--overlap-grad-reduce",
        action="store_true",
        help="Overlap DP gradient bucket reductions with the final backward.",
    )
    p.add_argument(
        "--overlap-tp-dgrad",
        action="store_true",
        help="Overlap TP dgrad communication with each wgrad GEMM.",
    )
    p.add_argument(
        "--overlap-p2p-comm",
        action="store_true",
        help="Prepost pipeline receives and defer waits across computation.",
    )
    p.add_argument(
        "--sequence-parallel",
        action="store_true",
        help="Shard eligible activations over the TP group.",
    )
    p.add_argument(
        "--precision", choices=["fp32", "bf16"], default="fp32"
    )
    p.add_argument(
        "--attn-backend",
        choices=["auto", "flash", "unfused"],
        default="auto",
    )
    p.add_argument("--fuse-rope", action="store_true", help="Nano CUDA pointwise RoPE via lazy Inductor (no CUDA Graphs).")
    p.add_argument("--rope-fp32-phase", action="store_true", help="Compute RoPE cosine/sine from FP32 phases before casting to activation dtype.")
    p.add_argument("--fuse-swiglu", action="store_true", help="Nano CUDA pointwise SwiGLU via lazy Inductor (no CUDA Graphs).")
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


def _disable_stage_bucketing(args: argparse.Namespace, pipeline_rank: int) -> bool:
    return (
        getattr(args, "disable_bucketing_after_first_stage", False)
        and pipeline_rank > 0
    )


def _validate_args(args: argparse.Namespace) -> None:
    if getattr(args, "fuse_wgrad_accum", False):
        if args.framework != "nano" or args.tp_size <= 1:
            raise ValueError("--fuse-wgrad-accum requires Nano with TP>1")
        if getattr(args, "cuda_graphs", False):
            raise ValueError("fused wgrad CUDA Graph capture is not validated")
    if (getattr(args, "tp_attention_views", False)
            and args.framework != "nano"):
        raise ValueError("--tp-attention-views supports nano only")
    if (getattr(args, "disable_bucketing_after_first_stage", False)
            and args.framework != "nano"):
        raise ValueError("--disable-bucketing-after-first-stage supports nano only")
    if getattr(args, "deterministic", False) and args.framework != "nano":
        raise ValueError("--deterministic currently supports --framework=nano only")
    if getattr(args, "cuda_graph_warmup_steps", 50) < 1:
        raise ValueError("cuda-graph-warmup-steps must be positive")
    if getattr(args, "cuda_graphs", False):
        if (getattr(args, "fuse_rope", False) or getattr(args, "fuse_swiglu", False)
                or getattr(args, "rope_fp32_phase", False)):
            raise ValueError("CUDA Graphs with fused pointwise kernels or FP32-phase RoPE are not validated")
        if not getattr(args, "deterministic", False):
            raise ValueError("--cuda-graphs requires --deterministic")
        supported = (
            args.framework == "nano" and args.device == "cuda"
            and args.collective_backend == "torch" and args.optimizer == "adamw"
            and args.tp_size == 2 and args.pp_size == 2 and args.dp_size in (None, 1)
            and args.sequence_parallel and args.precision == "bf16"
            and args.attn_backend == "flash" and not args.overlap_grad_reduce
            and not args.overlap_tp_dgrad and not args.overlap_p2p_comm
        )
        if not supported:
            raise ValueError("CUDA graphs currently require nano CUDA Torch/NCCL, TP2/PP2/DP1/SP, BF16 flash, AdamW, and no overlap flags")
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
    if args.sequence_parallel and args.tp_size <= 1:
        raise ValueError("sequence-parallel requires tp-size > 1")
    if args.sequence_parallel and args.seq_len % args.tp_size != 0:
        raise ValueError("seq-len must be divisible by tp-size with sequence parallel")
    if args.overlap_p2p_comm and args.pp_size > 2:
        raise ValueError("overlap-p2p-comm currently supports pp-size <= 2")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.nano_nccl_channels < 1:
        raise ValueError("nano-nccl-channels must be positive")
    if args.collective_backend == "nano-nccl":
        if not args.nano_nccl_library:
            raise ValueError(
                "--nano-nccl-library is required with "
                "--collective-backend=nano-nccl"
            )
        if args.framework == "megatron":
            raise ValueError(
                "--collective-backend=nano-nccl applies to the nano framework"
            )


def _dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.bfloat16 if args.precision == "bf16" else torch.float32


def _make_input_ids(
    *, vocab_size: int, shape: tuple[int, int], device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(1234)
    return torch.randint(0, vocab_size, shape, device=device, generator=generator)


@torch.no_grad()
def _optimizer_step(module: torch.nn.Module, optimizer: str, learning_rate: float) -> None:
    if optimizer == "none":
        return
    if optimizer == "adamw":
        raise RuntimeError("AdamW steps require the persistent benchmark optimizer")
    if optimizer != "sgd":
        raise ValueError(f"unsupported optimizer: {optimizer}")
    params = []
    grads = []
    for param in module.parameters():
        grad = param.grad
        if grad is None:
            grad = getattr(param, "main_grad", None)
        if grad is not None:
            params.append(param)
            grads.append(grad)
    if params:
        torch._foreach_add_(params, grads, alpha=-learning_rate)


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


def _mean_next_token_loss(per_token_loss: torch.Tensor) -> torch.Tensor:
    """Megatron returns [batch, sequence]; omit the target after the sequence."""
    return per_token_loss[:, :-1].float().mean()


def _validate_training_step(
    adamw, losses, phase: str, before=None, *, master_samples=None
) -> None:
    """Correctness evidence outside the timed region on every participating rank."""
    import json
    import math

    stats = adamw.verify_updated(before, master_samples=master_samples)
    device_steps = torch.stack([
        state["step"].detach().reshape(())
        for state in adamw.optimizer.state.values()
    ]).cpu()
    if not bool((device_steps == adamw.step_count).all()):
        raise RuntimeError(f"optimizer state step counts disagree with CPU counter after {phase}")
    stats["optimizer_device_steps"] = sorted(set(device_steps.tolist()))
    values = [float(loss.float().item()) for loss in losses]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"non-finite benchmark loss after {phase}")
    stats.update(
        phase=phase,
        rank=int(os.environ.get("RANK", 0)),
        loss=sum(values) / len(values) if values else None,
        loss_observations=len(values),
    )
    print("OPTIMIZER_VALIDATION " + json.dumps(stats, sort_keys=True), flush=True)


def _time_loop(
    step_fn,
    *,
    warmup: int,
    steps: int,
    device: torch.device,
    after_warmup=None,
) -> tuple[float, float]:
    """Measure steady-state steps after one full GC outside the timed region.

    This aligns the synthetic benchmark's post-warmup state without disabling
    automatic or explicit GC during measurement; it is not a production GC
    optimization. Distributed ranks rendezvous after collection, outside timing.
    """
    for _ in range(warmup):
        step_fn()
    if after_warmup is not None:
        after_warmup()
    gc.collect(2)
    if device.type == "cuda":
        torch.cuda.synchronize()
    if dist.is_initialized() and dist.get_world_size() > 1:
        if device.type == "cuda":
            index = (
                device.index if device.index is not None else torch.cuda.current_device()
            )
            dist.barrier(device_ids=[index])
        else:
            dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.nvtx.range_push("nano_megatron_benchmark_loop")
    t0 = time.perf_counter()
    for _ in range(steps):
        step_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    elapsed = time.perf_counter() - t0
    mem = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    return elapsed, mem


def _global_max_metrics(
    elapsed: float, memory_mb: float, *, device: torch.device
) -> tuple[float, float]:
    """Use the slowest rank for throughput and the largest per-rank allocation."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return elapsed, memory_mb
    metrics = torch.tensor(
        [elapsed, memory_mb], dtype=torch.float64, device=device
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    return float(metrics[0].item()), float(metrics[1].item())


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
        cuda_graphs=getattr(args, "cuda_graphs", False),
        deterministic=getattr(args, "deterministic", False),
        extra_warmup_updates=getattr(args, "cuda_graph_warmup_steps", 50) if getattr(args, "cuda_graphs", False) else 0,
        avg_step_time_ms=(elapsed / args.benchmark_steps) * 1000
        if args.benchmark_steps > 0
        else 0.0,
    )


def _scoped_determinism(function):
    """Scope process-local policy; CUDA libraries can still cache handle settings.

    Run isolated processes for performance comparisons. Concurrent unrelated
    work in the same Python process is not supported by this benchmark.
    """
    @wraps(function)
    def wrapped(args, *pargs, **kwargs):
        if not getattr(args, "deterministic", False):
            return function(args, *pargs, **kwargs)
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        try:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            return function(args, *pargs, **kwargs)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
            if workspace is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    return wrapped


@_scoped_determinism
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

    if getattr(args, "cuda_graphs", False):
        _validate_args(args)
        if dp_size != 1:
            raise ValueError("CUDA graphs currently require DP1")
    _ensure_dist(local_rank)
    ctx = initialize_parallel(
        ParallelConfig(
            tensor_parallel_size=args.tp_size,
            data_parallel_size=dp_size,
            pipeline_parallel_size=args.pp_size,
            sequence_parallel=args.sequence_parallel,
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
        attn_backend=args.attn_backend,
        tp_attention_views=getattr(args, "tp_attention_views", False),
        fuse_rope=getattr(args, "fuse_rope", False),
        fuse_swiglu=getattr(args, "fuse_swiglu", False),
        rope_fp32_phase=getattr(args, "rope_fp32_phase", False),
        tp_comm_overlap=args.overlap_tp_dgrad,
        fuse_wgrad_accum=getattr(args, "fuse_wgrad_accum", False),
    )

    collective_backend = None
    if args.collective_backend == "nano-nccl":
        from nano_megatron.distributed import create_nano_nccl_training_backend

        collective_backend = create_nano_nccl_training_backend(
            ctx,
            args.nano_nccl_library,
            transport=args.nano_nccl_transport,
            expected_channels=args.nano_nccl_channels,
        )
        ctx.backend = collective_backend

    try:
        torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42)

        ref = ReferenceGPT(cfg)
        dtype = _dtype(args)
        stage = build_pipeline_stage_from_reference(ref, ctx).to(
            device=device, dtype=dtype
        )
    except Exception:
        if collective_backend is not None:
            collective_backend.close()
        destroy_parallel()
        raise

    graph_step = None
    try:
        from scripts.benchmark_optimizer import make_benchmark_adamw

        ddp = None
        if dp_size > 1:
            ddp = DistributedDataParallel(
                stage,
                ctx,
                bucket_cap_mb=args.bucket_cap_mb,
                disable_bucketing=_disable_stage_bucketing(
                    args, ctx.pipeline_parallel_rank
                ),
                overlap_grad_reduce=args.overlap_grad_reduce,
            )
            stage_mod = ddp.module
            train_mod = ddp
        else:
            stage_mod = stage
            train_mod = stage
        train_mod.train()
        adamw = (
            make_benchmark_adamw(stage_mod, learning_rate=args.learning_rate)
            if args.optimizer == "adamw"
            else None
        )

        if is_rank0:
            n_params = sum(p.numel() for p in stage_mod.parameters())
            backend_detail = args.collective_backend
            if collective_backend is not None:
                routes = ";".join(
                    f"{route.name}:{route.backend.transport}"
                    f"[{','.join(route.backend.edge_transports)}]"
                    for route in collective_backend.routes
                )
                backend_detail += (
                    f"/{args.nano_nccl_transport}/{args.nano_nccl_channels}ch/{routes}"
                )
            print(
                f"[nano] params/rank={n_params/1e6:.1f}M "
                f"pp={args.pp_size} tp={args.tp_size} dp={dp_size} "
                f"microbatches={args.num_microbatches} "
                f"bucket_cap_mb={args.bucket_cap_mb} "
                f"disable_bucketing_after_first_stage={getattr(args, 'disable_bucketing_after_first_stage', False)} "
                f"precision={args.precision} sp={args.sequence_parallel} "
                f"collective_backend={backend_detail} "
                f"optimizer={args.optimizer} "
                f"deterministic={torch.are_deterministic_algorithms_enabled()} "
                f"cuda_graphs={getattr(args, 'cuda_graphs', False)} "
                + (
                    f"optimizer_config=lr:{args.learning_rate:g},betas:(0.9,0.95),"
                    "eps:1e-8,weight_decay:0.1,decay:ndim>=2,clip:none "
                    if args.optimizer == "adamw"
                    else ""
                )
                + f"fuse_wgrad_accum={getattr(args, 'fuse_wgrad_accum', False)} "
                + f"overlap_dp={args.overlap_grad_reduce} "
                f"overlap_tp={args.overlap_tp_dgrad} "
                f"overlap_pp={args.overlap_p2p_comm}",
                flush=True,
            )

        input_ids = _make_input_ids(
            vocab_size=cfg.vocab_size,
            shape=(args.batch_size, args.seq_len),
            device=device,
        )
        labels = input_ids.clone()
        last_losses: list[torch.Tensor] = []

        def step() -> None:
            last_losses.clear()
            if ddp is not None:
                ddp.zero_grad(set_to_none=True)
            else:
                stage_mod.zero_grad(set_to_none=True)
            loss = forward_backward_1f1b(
                stage=stage_mod,
                ctx=ctx,
                input_ids=input_ids,
                labels=labels,
                num_microbatches=args.num_microbatches,
                ddp=ddp,
                overlap_p2p_comm=args.overlap_p2p_comm,
            )
            if loss is not None:
                last_losses.append(loss.detach().clone())
            if adamw is not None:
                adamw.step()
            else:
                _optimizer_step(stage_mod, args.optimizer, args.learning_rate)

        if getattr(args, "cuda_graphs", False):
            from scripts.benchmark_cuda_graph import BenchmarkCudaGraphStep

            graph_step = BenchmarkCudaGraphStep(step, adamw, last_losses, device)

        before = (
            [param.detach().clone() for param in stage_mod.parameters() if param.requires_grad]
            if adamw is not None
            else None
        )
        master_samples = None

        def validate_warmup() -> None:
            nonlocal before, master_samples
            if adamw is not None:
                _validate_training_step(adamw, last_losses, "warmup", before)
                master_samples = adamw.snapshot_master_samples()
            before = None  # Do not retain validation snapshots in timed memory.

        def prepare_measurement() -> None:
            validate_warmup()
            if graph_step is not None:
                preparation = graph_step.prepare(args.cuda_graph_warmup_steps)
                print("CUDA_GRAPH_PREPARATION " + json.dumps(preparation), flush=True)
                validate_warmup()  # Refresh probes after additional real updates.

        elapsed, memory_mb = _time_loop(
            graph_step.run if graph_step is not None else step,
            warmup=max(args.warmup_steps, 1) if adamw is not None else args.warmup_steps,
            steps=args.benchmark_steps,
            device=device,
            after_warmup=prepare_measurement,
        )
        if adamw is not None:
            _validate_training_step(
                adamw, last_losses, "measured", master_samples=master_samples
            )
        elapsed, memory_mb = _global_max_metrics(
            elapsed, memory_mb, device=device
        )
    finally:
        if graph_step is not None:
            graph_step.close()
        if collective_backend is not None:
            collective_backend.close()
        destroy_parallel()

    tag_parts = [
        "nano-megatron+nano-nccl"
        if args.collective_backend == "nano-nccl"
        else "nano-megatron"
    ]
    if getattr(args, "deterministic", False):
        tag_parts[0] += "+deterministic"
    if getattr(args, "cuda_graphs", False):
        tag_parts[0] += "+cuda-graph"
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
    if args.overlap_p2p_comm:
        raise ValueError(
            "Megatron-LM's non-interleaved pipeline schedule rejects "
            "overlap_p2p_comm; run the Megatron reference without "
            "--overlap-p2p-comm"
        )
    try:
        from megatron.core import parallel_state
        from megatron.core.distributed import (
            DistributedDataParallel as MegatronDDP,
        )
        from megatron.core.distributed import (
            DistributedDataParallelConfig,
            finalize_model_grads,
        )
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
        pipeline_dtype=_dtype(args),
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        add_qkv_bias=False,
        activation_func=torch.nn.functional.silu,
        gated_linear_unit=True,
        normalization="LayerNorm",
        sequence_parallel=args.sequence_parallel,
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
        deallocate_pipeline_outputs=False,
        overlap_p2p_comm=False,
        batch_p2p_comm=True,
        # The pipeline schedule owns final gradient synchronization in a
        # normal Megatron training step.  This also coalesces replicated
        # sequence-parallel parameter grads across the TP group; calling only
        # DDP.finish_grad_sync() would omit that reduction.
        finalize_model_grads_func=finalize_model_grads,
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
    if args.precision == "bf16":
        model = model.bfloat16()
    model.train()

    from scripts.benchmark_optimizer import make_benchmark_adamw

    bucket_size = int(
        args.bucket_cap_mb * 1024 * 1024 / _dtype(args).itemsize
    )
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=args.overlap_grad_reduce,
        use_distributed_optimizer=False,
        average_in_collective=True,
        bucket_size=bucket_size,
    )
    ddp = MegatronDDP(config=config, ddp_config=ddp_config, module=model)
    if dp_size > 1:
        ddp.broadcast_params()
    train_model = ddp
    if args.overlap_grad_reduce:
        config.no_sync_func = ddp.no_sync
    adamw = (
        make_benchmark_adamw(model, learning_rate=args.learning_rate)
        if args.optimizer == "adamw"
        else None
    )

    if is_rank0:
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[megatron] params/rank={n_params/1e6:.1f}M "
            f"pp={args.pp_size} tp={args.tp_size} dp={dp_size} "
            f"microbatches={args.num_microbatches} "
            f"micro_bs={micro_batch_size} precision={args.precision} "
            f"sp={args.sequence_parallel} "
            f"optimizer={args.optimizer} "
            + (
                f"optimizer_config=lr:{args.learning_rate:g},betas:(0.9,0.95),"
                "eps:1e-8,weight_decay:0.1,decay:ndim>=2,clip:none "
                if args.optimizer == "adamw"
                else ""
            )
            + f"overlap_dp={args.overlap_grad_reduce} "
            "overlap_tp=native overlap_pp=False",
            flush=True,
        )

    # One full local batch; iterator yields microbatch slices each schedule.
    full_ids = _make_input_ids(
        vocab_size=vocab_size,
        shape=(args.batch_size, args.seq_len),
        device=device,
    )
    full_pos = (
        torch.arange(args.seq_len, device=device)
        .unsqueeze(0)
        .expand(args.batch_size, -1)
        .contiguous()
    )
    # Nano shifts labels in shifted_cross_entropy; Megatron expects callers
    # to supply next-token targets. The wrapped final target is excluded below.
    full_labels = torch.roll(full_ids, shifts=-1, dims=1)

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

    last_losses: list[torch.Tensor] = []

    def forward_step_func(data_iterator, model_chunk):
        batch = next(data_iterator)
        tokens = batch["tokens"]
        position_ids = batch["position_ids"]
        # Intermediate stages ignore token embeddings (set_input_tensor);
        # last stage uses labels for LM loss.
        labels = batch["labels"] if is_last else None
        output_tensor = model_chunk(tokens, position_ids, None, labels=labels)

        def loss_func(output_tensor):
            loss = _mean_next_token_loss(output_tensor)
            # The schedule may scale loss in place by the microbatch count.
            last_losses.append(loss.detach().clone())
            return loss, {"lm loss": loss.detach()}

        return output_tensor, loss_func

    def step() -> None:
        last_losses.clear()
        if ddp is not None:
            ddp.zero_grad_buffer()
        model.zero_grad(set_to_none=True)
        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=_make_iterator(),
            model=train_model,
            num_microbatches=args.num_microbatches,
            seq_length=args.seq_len,
            micro_batch_size=micro_batch_size,
            forward_only=False,
        )
        if adamw is not None:
            adamw.step()
        else:
            _optimizer_step(model, args.optimizer, args.learning_rate)

    before = (
        [param.detach().clone() for param in model.parameters() if param.requires_grad]
        if adamw is not None
        else None
    )
    master_samples = None

    def validate_warmup() -> None:
        nonlocal before, master_samples
        if adamw is not None:
            _validate_training_step(adamw, last_losses, "warmup", before)
            master_samples = adamw.snapshot_master_samples()
        before = None

    elapsed, memory_mb = _time_loop(
        step,
        warmup=max(args.warmup_steps, 1) if adamw is not None else args.warmup_steps,
        steps=args.benchmark_steps,
        device=device,
        after_warmup=validate_warmup,
    )
    if adamw is not None:
        _validate_training_step(
            adamw, last_losses, "measured", master_samples=master_samples
        )
    elapsed, memory_mb = _global_max_metrics(
        elapsed, memory_mb, device=device
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
    graph_notes = "".join(
        f"- **CUDA Graph ({r.framework})**: fixed-shape full forward/backward and AdamW replay; {r.extra_warmup_updates} additional real warmup updates outside timing. Per-rank costs are logged as CUDA_GRAPH_PREPARATION. The automatic GC policy is unchanged. Not an I/O/checkpoint or dynamic-input benchmark.\n"
        for r in results if r.cuda_graphs
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
- **Precision**: see run command
- **Tokens**: global = local_batch × seq × dp_size / wall_time
- **Schedule**: non-interleaved 1F1B
- **Timing boundary**: synthetic steady-state benchmark; one full generation-2 GC runs after warmup and its validation callback, before GPU synchronization/reset and the timer. Distributed ranks rendezvous after collection, outside timing. The caller's automatic GC policy is unchanged, and any subsequent GC remains part of measured steps. This boundary aligns framework benchmark state; it is not a production GC optimization or a claim about production training throughput.

| Framework | Parallel | Microbatches | Local-BS | Seq | Hidden | Layers | Heads | Tok/s local | Tok/s global | Mem MB | Step ms |
|-----------|----------|--------------|----------|-----|--------|--------|-------|-------------|--------------|--------|---------|
{rows}
{graph_notes}
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

    try:
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
                    raise

        if is_rank0 and results and args.output:
            write_markdown(results, args.output)
            print(f"\nResults written to {args.output}", flush=True)
    finally:
        _cuda_reset()
        if dist.is_initialized():
            dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
