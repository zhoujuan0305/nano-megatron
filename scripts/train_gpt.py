#!/usr/bin/env python3
"""Small reproducible text-training health check, not a production training engine.

Uses nano's native TP/SP/PP schedule, dynamic document-bounded batches, heldout
forward-only evaluation, complete FP32-master AdamW checkpoints (same topology).
No CUDA Graph or throughput-only shortcut is used in this driver.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import time

import torch
import torch.distributed as dist

from nano_megatron.model import build_pipeline_stage_from_reference
from nano_megatron.parallel import ParallelConfig, initialize_parallel, destroy_parallel
from nano_megatron.reference import ReferenceGPT, ReferenceGPTConfig
from nano_megatron.schedules import forward_backward_1f1b
from nano_megatron.schedules.p2p import recv_forward, send_forward
from scripts.benchmark_optimizer import make_benchmark_adamw
from scripts.benchmark_pp import _validate_training_step
from scripts.training_data import DocumentTokenSampler
from scripts.training_state import optimizer_state_dict, load_optimizer_state_dict


def learning_rate(update: int, total: int, warmup: int, peak: float, minimum: float) -> float:
    if update < 1 or not 0 <= warmup < total or not 0 < minimum <= peak:
        raise ValueError("invalid learning-rate schedule")
    if update <= warmup:
        return peak * update / warmup
    fraction = min((update - warmup) / (total - warmup), 1.0)
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * fraction)) / 2


def model_config_dict(config: ReferenceGPTConfig) -> dict:
    def encode(value):
        if isinstance(value, torch.dtype):
            return str(value)
        raise TypeError(f"unsupported model configuration value: {type(value)}")
    values = asdict(config)
    # Preserve schema-1 eager checkpoint semantics; enabled numerical paths remain explicit.
    for name in ("fuse_rope", "fuse_swiglu", "rope_fp32_phase"):
        if not values[name]:
            values.pop(name)
    return json.loads(json.dumps(values, default=encode))


def advance_data_hash(previous: str, batch_id: str) -> str:
    return hashlib.sha256(bytes.fromhex(previous) + bytes.fromhex(batch_id)).hexdigest()


def reduce_scalar(value: torch.Tensor) -> float:
    value = value.detach().float().clone()
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return float(value.item())


@torch.no_grad()
def evaluate(stage, ctx, sampler, *, batch_size: int, microbatches: int, batches: int, device) -> float:
    stage.eval()
    local_loss = torch.zeros((), device=device)
    micro = batch_size // microbatches
    shape = (micro, sampler.sequence_length // ctx.tensor_parallel_size if ctx.sequence_parallel else sampler.sequence_length, stage.config.hidden_size)
    dtype = next(stage.parameters()).dtype
    positions = torch.arange(sampler.sequence_length, device=device).unsqueeze(0).expand(micro, -1)
    for batch_index in range(batches):
        ids, _ = sampler.batch(batch_index, batch_size)
        ids = ids.to(device)
        for mb in ids.split(micro):
            hidden = mb if stage.is_first_stage else recv_forward(ctx, shape=shape, dtype=dtype, device=device)
            output = stage(hidden, positions=positions)
            if stage.is_last_stage:
                loss = stage.shifted_cross_entropy(output, mb)
                if ctx.tensor_parallel_rank == 0:
                    local_loss.add_(loss.float() / (batches * microbatches))
            else:
                send_forward(ctx, output)
    result = reduce_scalar(local_loss)
    stage.train()
    if not math.isfinite(result):
        raise RuntimeError("non-finite heldout loss")
    return result


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def save_checkpoint(directory: Path, *, stage, optimizer, semantic: dict, step: int,
                    data_hash: str, rank: int, local_rank: int, device) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(schema_version=1, semantic=semantic, step=step, data_hash=data_hash,
                   model={name: tensor.detach().cpu().clone() for name, tensor in stage.state_dict().items()},
                   optimizer=optimizer_state_dict(optimizer, stage),
                   rng=dict(python=random.getstate(), torch=torch.get_rng_state(),
                            cuda=torch.cuda.get_rng_state(device)))
    path = directory / f"rank{rank:04d}.pt"
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    atomic_json(directory / f"rank{rank:04d}.json", dict(step=step, data_hash=data_hash, bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])
    if local_rank == 0:
        atomic_json(directory / "metadata.json", dict(schema_version=1, semantic=semantic, step=step, data_hash=data_hash))
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])


def restore_checkpoint(directory: Path, *, stage, optimizer, semantic: dict, rank: int, device) -> tuple[int, str]:
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["semantic"] != semantic:
        raise ValueError("checkpoint training/data/scheduler/topology configuration mismatch")
    path = directory / f"rank{rank:04d}.pt"
    integrity = json.loads(path.with_suffix(".json").read_text())
    if path.stat().st_size != integrity["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != integrity["sha256"]:
        raise ValueError("checkpoint shard size/checksum mismatch")
    if integrity["step"] != metadata["step"] or integrity["data_hash"] != metadata["data_hash"]:
        raise ValueError("checkpoint shard index metadata mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)  # Only this driver's trusted local output.
    if payload["schema_version"] != 1 or payload["semantic"] != semantic or payload["step"] != metadata["step"]:
        raise ValueError("checkpoint shard metadata mismatch")
    if payload["optimizer"]["step_count"] != payload["step"] or payload["data_hash"] != metadata["data_hash"]:
        raise ValueError("checkpoint update/data cursor mismatch")
    stage.load_state_dict(payload["model"], strict=True)
    load_optimizer_state_dict(optimizer, stage, payload["optimizer"])
    random.setstate(payload["rng"]["python"])
    torch.set_rng_state(payload["rng"]["torch"])
    torch.cuda.set_rng_state(payload["rng"]["cuda"], device)
    return payload["step"], payload["data_hash"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--pp-size", type=int, default=1)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--ffn-hidden-size", type=int, default=704)
    p.add_argument("--vocab-size", type=int, default=51200)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--microbatches", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-seed", type=int, default=1234)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--min-learning-rate", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--check-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--attn-backend", choices=["flash", "unfused"], default="flash")
    p.add_argument("--memory-cap-gib", type=float, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.tp_size, args.pp_size, args.batch_size, args.microbatches, args.seq_len) < 1:
        raise ValueError("parallel, batch, and sequence sizes must be positive")
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        raise ValueError("stop-after must be within the training schedule")
    if args.batch_size % args.microbatches or args.num_layers % args.pp_size:
        raise ValueError("batch/layer partition is not divisible")
    if args.tp_size > 1 and args.seq_len % args.tp_size:
        raise ValueError("SP requires sequence divisible by TP")
    if min(args.steps, args.log_every, args.eval_every, args.eval_batches, args.check_every, args.save_every) < 1:
        raise ValueError("step/interval counts must be positive")
    if not 0 <= args.warmup < args.steps:
        raise ValueError("warmup must be shorter than total schedule")
    rank, local = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != args.tp_size * args.pp_size:
        raise ValueError("this health-check driver supports DP1 only")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib * 1024**3 / torch.cuda.get_device_properties(local).total_memory, local)
    if world > 1:
        dist.init_process_group("nccl")
    ctx = initialize_parallel(ParallelConfig(tensor_parallel_size=args.tp_size, pipeline_parallel_size=args.pp_size,
                             data_parallel_size=1, sequence_parallel=args.tp_size > 1), dist_backend="nccl")
    output = Path(args.output)
    output.mkdir(exist_ok=True, parents=True)
    path = output / f"metrics.rank{rank}.jsonl"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = ReferenceGPTConfig(vocab_size=args.vocab_size, max_seq_len=args.seq_len, hidden_size=args.hidden_size,
        num_layers=args.num_layers, num_heads=args.num_heads, ffn_hidden_size=args.ffn_hidden_size,
        layernorm_eps=1e-5, use_bias=False, tie_word_embeddings=False, position_embedding_type="rope",
        rotary_dim=args.hidden_size // args.num_heads, rotary_base=10000, activation_func="swiglu",
        gated_linear_unit=True, normalization="layernorm", use_fused_qkv=True, hidden_dropout=0.,
        attention_dropout=0., attn_backend=args.attn_backend)
    reference = ReferenceGPT(cfg)
    logical_parameters = sum(p.numel() for p in reference.parameters())
    stage = build_pipeline_stage_from_reference(reference, ctx).to(device=device, dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32)
    del reference
    stage.train()
    optimizer = make_benchmark_adamw(stage, learning_rate=args.learning_rate)
    train = DocumentTokenSampler(args.data_dir, "train", args.seq_len, args.data_seed)
    heldout = DocumentTokenSampler(args.data_dir, "validation", args.seq_len, args.data_seed)
    if train.dataset_id != heldout.dataset_id:
        raise ValueError("split provenance differs")
    semantic = dict(model=model_config_dict(cfg), world_size=world, tp_size=args.tp_size, pp_size=args.pp_size,
        batch_size=args.batch_size, microbatches=args.microbatches, precision=args.precision,
        seed=args.seed, data_seed=args.data_seed, dataset_id=train.dataset_id, sampler="document-window-v1",
        total_schedule_steps=args.steps, warmup=args.warmup, learning_rate=args.learning_rate, min_learning_rate=args.min_learning_rate)
    semantic = json.loads(json.dumps(semantic))  # Canonical JSON types for shard/metadata equality.
    step, data_hash = 0, "00" * 32
    if args.resume:
        step, data_hash = restore_checkpoint(Path(args.resume), stage=stage, optimizer=optimizer, semantic=semantic, rank=rank, device=device)
    start_step, start_wall = step, time.perf_counter()
    stop = min(args.steps, args.stop_after or args.steps)
    if stop <= step:
        raise ValueError("stop must be after saved update")
    latest_eval = None
    try:
        with path.open("x") as log:
            def record(kind, **values):
                row = dict(kind=kind, rank=rank, step=step, epoch_ns=time.time_ns(), elapsed_seconds=time.perf_counter() - start_wall, **values)
                log.write(json.dumps(row) + "\n")
                log.flush()
                if rank == 0:
                    print(json.dumps(row), flush=True)
            record("metadata", semantic=semantic, logical_parameters=logical_parameters, local_parameters=sum(p.numel() for p in stage.parameters()), pid=os.getpid(), resume=args.resume)
            latest_eval = evaluate(stage, ctx, heldout, batch_size=args.batch_size, microbatches=args.microbatches,
                                   batches=args.eval_batches, device=device)
            record("evaluation", validation_loss=latest_eval, optimizer_steps=optimizer.step_count)
            accumulation = torch.zeros((), device=device)
            count = 0
            previous_log = time.perf_counter()
            while step < stop:
                ids, batch = train.batch(step, args.batch_size)
                ids = ids.to(device)
                lr = learning_rate(step + 1, args.steps, args.warmup, args.learning_rate, args.min_learning_rate)
                for group in optimizer.optimizer.param_groups:
                    group["lr"] = lr
                stage.zero_grad(set_to_none=True)
                loss = forward_backward_1f1b(stage=stage, ctx=ctx, input_ids=ids, labels=ids, num_microbatches=args.microbatches)
                if stage.is_last_stage and ctx.tensor_parallel_rank == 0:
                    accumulation.add_(loss.detach().float())
                optimizer.step()
                step += 1
                count += 1
                data_hash = advance_data_hash(data_hash, batch["batch_id"])
                if step % args.log_every == 0 or step == stop:
                    mean_loss = reduce_scalar(accumulation) / count
                    if not math.isfinite(mean_loss):
                        raise RuntimeError("non-finite training loss")
                    now = time.perf_counter()
                    record("train", training_loss=mean_loss, learning_rate=lr, updates_in_window=count,
                           window_wall_seconds=now - previous_log, batch_id=batch["batch_id"], data_hash=data_hash,
                           allocated_mib=torch.cuda.memory_allocated() / 1024**2,
                           reserved_mib=torch.cuda.memory_reserved() / 1024**2,
                           max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    accumulation.zero_()
                    count = 0
                    previous_log = now
                if step % args.check_every == 0 or step == stop:
                    _validate_training_step(optimizer, [], "longrun")
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in stage.parameters()):
                        raise RuntimeError("non-finite gradients")
                if step % args.eval_every == 0 or step == stop:
                    stage.zero_grad(set_to_none=True)
                    latest_eval = evaluate(stage, ctx, heldout, batch_size=args.batch_size, microbatches=args.microbatches,
                                           batches=args.eval_batches, device=device)
                    record("evaluation", validation_loss=latest_eval, optimizer_steps=optimizer.step_count)
                if step % args.save_every == 0 or step == stop:
                    directory = output / f"checkpoint-{step:08d}"
                    save_checkpoint(directory, stage=stage, optimizer=optimizer, semantic=semantic, step=step,
                                    data_hash=data_hash, rank=rank, local_rank=local, device=device)
                    record("checkpoint", directory=directory.name, data_hash=data_hash)
            summary = dict(completed=True, step=step, start_step=start_step, updates_this_invocation=step - start_step,
                nominal_tokens_total=step * args.batch_size * args.seq_len,
                supervised_targets_total=step * args.batch_size * (args.seq_len - 1), data_hash=data_hash,
                validation_loss=latest_eval, cpu_optimizer_steps=optimizer.step_count,
                device_optimizer_steps=torch.stack([s["step"] for s in optimizer.optimizer.state.values()]).cpu().tolist(),
                wall_seconds=time.perf_counter() - start_wall, logical_parameters=logical_parameters,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2, semantic=semantic)
            atomic_json(output / f"summary.rank{rank}.json", summary)
            record("completed", **{k: v for k, v in summary.items() if k != "step"})
    finally:
        destroy_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Also cover failures during construction/restore before the logging loop.
        destroy_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
