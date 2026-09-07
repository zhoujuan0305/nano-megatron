#!/usr/bin/env python3
"""Check Nano NCCL TP/SP and DP collectives against PyTorch NCCL.

Launch every process under the same Open MPI world and provide the usual
RANK/WORLD_SIZE/LOCAL_RANK variables expected by torch.distributed.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

from nano_megatron.distributed import (
    TorchDistBackend,
    create_nano_nccl_training_backend,
)
from nano_megatron.parallel import (
    ParallelConfig,
    destroy_parallel,
    initialize_parallel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument(
        "--transport",
        choices=["auto", "shm", "p2p", "socket", "rdma"],
        default="auto",
    )
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _phase(message: str, *, enabled: bool) -> None:
    if enabled:
        rank = os.environ.get("RANK", "?")
        print(f"[rank {rank}] {message}", file=sys.stderr, flush=True)


def _assert_equal(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    try:
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0, msg=name)
    except AssertionError:
        mismatch = (actual != expected).nonzero().flatten()
        first = int(mismatch[0].item()) if mismatch.numel() else -1
        raise AssertionError(
            f"{name}: mismatches={mismatch.numel()} first={first} "
            f"actual={actual[first].item() if first >= 0 else 'n/a'} "
            f"expected={expected[first].item() if first >= 0 else 'n/a'}"
        ) from None


def _check_all_reduce(
    nano,
    torch_backend: TorchDistBackend,
    *,
    group,
    group_rank: int,
    name: str,
) -> None:
    for op in ("sum", "max"):
        source = torch.arange(65539, device="cuda", dtype=torch.float32)
        source.add_(float(group_rank + 1))
        expected = source.clone()
        actual = source.clone()
        torch_backend.all_reduce(expected, group=group, op=op)
        work = nano.all_reduce(actual, group=group, op=op, async_op=True)
        work.wait()
        _assert_equal(actual, expected, f"{name} all-reduce {op}")


def _check_gather_scatter(
    nano,
    torch_backend: TorchDistBackend,
    *,
    group,
    group_rank: int,
    group_size: int,
    name: str,
) -> None:
    rank = group_rank
    source = torch.arange(65539, device="cuda", dtype=torch.bfloat16)
    source.add_(rank * 3)

    expected_gather = [torch.empty_like(source) for _ in range(group_size)]
    actual_gather = [torch.empty_like(source) for _ in range(group_size)]
    torch_backend.all_gather(expected_gather, source, group=group)
    gather_work = nano.all_gather(
        actual_gather, source, group=group, async_op=True
    )
    gather_work.wait()
    for source_rank, (actual, expected) in enumerate(
        zip(actual_gather, expected_gather)
    ):
        _assert_equal(actual, expected, f"{name} all-gather rank {source_rank}")

    expected_into = torch.empty(
        source.numel() * group_size,
        device=source.device,
        dtype=source.dtype,
    )
    actual_into = torch.empty_like(expected_into)
    torch_backend.all_gather_into_tensor(expected_into, source, group=group)
    into_work = nano.all_gather_into_tensor(
        actual_into, source, group=group, async_op=True
    )
    into_work.wait()
    _assert_equal(actual_into, expected_into, f"{name} all-gather-into-tensor")

    inputs = [
        torch.full_like(source, float(rank * 10 + destination + 1))
        for destination in range(group_size)
    ]
    expected_scatter = torch.empty_like(source)
    actual_scatter = torch.empty_like(source)
    torch_backend.reduce_scatter(expected_scatter, inputs, group=group, op="sum")
    scatter_work = nano.reduce_scatter(
        actual_scatter,
        inputs,
        group=group,
        op="sum",
        async_op=True,
    )
    scatter_work.wait()
    _assert_equal(actual_scatter, expected_scatter, f"{name} reduce-scatter")


def main() -> None:
    args = parse_args()
    world_size = int(os.environ["WORLD_SIZE"])
    expected_world = args.tp_size * args.pp_size * args.dp_size
    if world_size != expected_world:
        raise ValueError(
            f"WORLD_SIZE={world_size}, expected "
            f"tp_size*pp_size*dp_size={expected_world}"
        )

    _phase("initialize_parallel: begin", enabled=args.verbose)
    ctx = initialize_parallel(
        ParallelConfig(
            tensor_parallel_size=args.tp_size,
            pipeline_parallel_size=args.pp_size,
            data_parallel_size=args.dp_size,
        ),
        dist_backend="nccl",
    )
    _phase("initialize_parallel: complete", enabled=args.verbose)
    torch_backend = TorchDistBackend()
    _phase("Nano communicators: begin", enabled=args.verbose)
    nano = create_nano_nccl_training_backend(
        ctx,
        args.library,
        transport=args.transport,
    )
    _phase("Nano communicators: complete", enabled=args.verbose)
    for route in nano.routes:
        _phase(
            f"{route.name}: transport={route.backend.transport} "
            f"edges={','.join(route.backend.edge_transports)}",
            enabled=args.verbose,
        )
    try:
        ctx.backend = nano
        _phase("TP all-reduce: begin", enabled=args.verbose)
        _check_all_reduce(
            nano,
            torch_backend,
            group=ctx.tensor_parallel_group,
            group_rank=ctx.tensor_parallel_rank,
            name="TP",
        )
        _phase("TP gather/scatter: begin", enabled=args.verbose)
        _check_gather_scatter(
            nano,
            torch_backend,
            group=ctx.tensor_parallel_group,
            group_rank=ctx.tensor_parallel_rank,
            group_size=ctx.tensor_parallel_size,
            name="TP",
        )
        _phase("DP all-reduce: begin", enabled=args.verbose)
        _check_all_reduce(
            nano,
            torch_backend,
            group=ctx.data_context_parallel_group,
            group_rank=ctx.data_parallel_rank,
            name="DP",
        )
        _phase("DP gather/scatter: begin", enabled=args.verbose)
        _check_gather_scatter(
            nano,
            torch_backend,
            group=ctx.data_context_parallel_group,
            group_rank=ctx.data_parallel_rank,
            group_size=ctx.data_parallel_size * ctx.context_parallel_size,
            name="DP",
        )
        _phase("collectives: complete", enabled=args.verbose)
        torch.cuda.synchronize()
        dist.barrier()
        if ctx.rank == 0:
            print(
                "Nano NCCL TP and DP AllReduce/AllGather/ReduceScatter "
                "match PyTorch NCCL",
                flush=True,
            )
    finally:
        _phase("destroy: begin", enabled=args.verbose)
        nano.close()
        destroy_parallel()
        _phase("destroy: complete", enabled=args.verbose)


if __name__ == "__main__":
    main()
