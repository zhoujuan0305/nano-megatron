#!/usr/bin/env python3
"""Compare Nano NCCL and torch.distributed over the generated DP groups."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from nano_megatron.distributed import NanoNcclBackend
from nano_megatron.parallel import ParallelConfig, destroy_parallel, initialize_parallel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--transport", choices=["auto", "socket", "rdma"], default="rdma")
    parser.add_argument("--dtype", choices=["float", "bf16"], default="bf16")
    parser.add_argument(
        "--sizes",
        default="257,65539,12582912",
        help="Comma-separated element counts, including non-aligned cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_size = int(os.environ["WORLD_SIZE"])
    expected_world = args.tp_size * args.pp_size * args.dp_size
    if world_size != expected_world:
        raise ValueError(f"world size {world_size} != tp*pp*dp {expected_world}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("launch with one visible GPU per process")

    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", rank=int(os.environ["RANK"]), world_size=world_size
    )
    ctx = initialize_parallel(
        ParallelConfig(
            tensor_parallel_size=args.tp_size,
            pipeline_parallel_size=args.pp_size,
            data_parallel_size=args.dp_size,
        )
    )
    backend = NanoNcclBackend.from_parallel_context(
        ctx, args.library, transport=args.transport, expected_channels=4
    )
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    sizes = [int(value) for value in args.sizes.split(",")]
    worst_error = 0.0
    try:
        for count in sizes:
            indices = torch.arange(count, device="cuda", dtype=torch.float32)
            source = ((indices.remainder(31) + 1 + ctx.rank) / 64).to(dtype)
            torch_result = source.clone()
            dist.all_reduce(torch_result, group=ctx.data_context_parallel_group)

            nano_result = source.clone()
            work = backend.all_reduce(
                nano_result,
                group=ctx.data_context_parallel_group,
                op="sum",
                async_op=True,
            )
            work.wait()
            torch.cuda.synchronize()
            error = float(
                (nano_result.float() - torch_result.float()).abs().max().item()
            )
            worst_error = max(worst_error, error)
            torch.testing.assert_close(nano_result, torch_result, rtol=0, atol=0)

        metric = torch.tensor([worst_error], device="cuda", dtype=torch.float64)
        dist.all_reduce(metric, op=dist.ReduceOp.MAX)
        if ctx.rank == 0:
            print(
                f"PASS groups=DP{args.dp_size} dtype={args.dtype} "
                f"transport={backend.transport} channels={backend.channel_count} "
                f"sizes={sizes} max_abs_error={metric.item():.6g}",
                flush=True,
            )
        dist.barrier(device_ids=[0])
    finally:
        backend.close()
        destroy_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
