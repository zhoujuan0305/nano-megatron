#!/usr/bin/env python3
"""Measure communication time not overlapped by GPU compute in an Nsight trace.

The benchmark emits one ``nano_megatron_benchmark_loop`` NVTX range per rank.
For every CUDA context, this script clips kernels to that rank's range and
reports communication time that does not intersect compute. NCCL kernels and
Nano NCCL's Ring Simple kernel are classified as communication. The maximum
per-rank value is the critical-rank exposed communication time used by the
experiment report.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


Interval = tuple[int, int]
NVTX_RANGE = "nano_megatron_benchmark_loop"
COMMUNICATION_PATTERNS = ("nccl", "ring_simple_kernel")


@dataclass(frozen=True)
class ContextResult:
    process_id: int
    device_id: int
    context_id: int
    steps: int
    range_ms: float
    communication_ms: float
    communication_overlapped_ms: float
    communication_exposed_ms: float
    exposed_ms_per_step: float
    overlap_fraction: float
    communication_kernels: int
    compute_kernels: int


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_length(intervals: Iterable[Interval]) -> int:
    return sum(end - start for start, end in merge_intervals(intervals))


def intersection_length(left: Iterable[Interval], right: Iterable[Interval]) -> int:
    a = merge_intervals(left)
    b = merge_intervals(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _sqlite_path(trace: Path) -> Path:
    if trace.suffix == ".sqlite":
        return trace
    if trace.suffix != ".nsys-rep":
        raise ValueError("trace must end in .nsys-rep or .sqlite")
    sqlite_path = trace.with_suffix(".sqlite")
    subprocess.run(
        [
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--force-overwrite=true",
            "--output",
            str(sqlite_path),
            str(trace),
        ],
        check=True,
    )
    return sqlite_path


def analyze(sqlite_path: Path, *, steps: int) -> list[ContextResult]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    connection = sqlite3.connect(sqlite_path)
    try:
        strings = dict(connection.execute("SELECT id, value FROM StringIds"))
        ranges: dict[int, Interval] = {}
        process_ids: dict[int, int] = {}
        for start, end, text, text_id, global_tid in connection.execute(
            "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS "
            "WHERE end IS NOT NULL"
        ):
            label = text if text is not None else strings.get(text_id)
            if label == NVTX_RANGE:
                # Nsight encodes process identity in the upper bits. CUDA
                # activity stores the same value with the thread bits cleared.
                global_pid = global_tid & ~0xFFFFFF
                current = ranges.get(global_pid)
                if current is not None:
                    raise RuntimeError(
                        f"multiple {NVTX_RANGE!r} ranges for globalPid={global_pid}"
                    )
                ranges[global_pid] = (start, end)
                process_ids[global_pid] = global_tid & 0xFFFFFF
        if not ranges:
            raise RuntimeError(f"NVTX range {NVTX_RANGE!r} not found")

        grouped: dict[tuple[int, int, int], tuple[list[Interval], list[Interval], int, int]] = {}
        query = """
            SELECT k.start, k.end, k.deviceId, k.contextId, k.globalPid, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON s.id = k.shortName
            ORDER BY k.start
        """
        for start, end, device_id, context_id, global_pid, name in connection.execute(query):
            window = ranges.get(global_pid)
            if window is None or end <= window[0] or start >= window[1]:
                continue
            clipped = (max(start, window[0]), min(end, window[1]))
            key = (global_pid, device_id, context_id)
            comm, compute, comm_count, compute_count = grouped.setdefault(
                key, ([], [], 0, 0)
            )
            if any(pattern in name.lower() for pattern in COMMUNICATION_PATTERNS):
                comm.append(clipped)
                grouped[key] = (comm, compute, comm_count + 1, compute_count)
            else:
                compute.append(clipped)
                grouped[key] = (comm, compute, comm_count, compute_count + 1)

        results = []
        for (global_pid, device_id, context_id), (
            communication,
            compute,
            communication_kernels,
            compute_kernels,
        ) in sorted(grouped.items()):
            if not communication:
                continue
            range_start, range_end = ranges[global_pid]
            communication_ns = interval_length(communication)
            overlapped_ns = intersection_length(communication, compute)
            exposed_ns = communication_ns - overlapped_ns
            process_id = process_ids[global_pid]
            results.append(
                ContextResult(
                    process_id=process_id,
                    device_id=device_id,
                    context_id=context_id,
                    steps=steps,
                    range_ms=(range_end - range_start) / 1e6,
                    communication_ms=communication_ns / 1e6,
                    communication_overlapped_ms=overlapped_ns / 1e6,
                    communication_exposed_ms=exposed_ns / 1e6,
                    exposed_ms_per_step=exposed_ns / 1e6 / steps,
                    overlap_fraction=(overlapped_ns / communication_ns),
                    communication_kernels=communication_kernels,
                    compute_kernels=compute_kernels,
                )
            )
        if not results:
            raise RuntimeError("no communication kernels found inside benchmark NVTX ranges")
        return results
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    sqlite_path = _sqlite_path(args.trace.resolve())
    results = analyze(sqlite_path, steps=args.steps)
    for result in results:
        print(
            f"pid={result.process_id} device={result.device_id} "
            f"comm={result.communication_ms:.3f} ms "
            f"overlapped={result.communication_overlapped_ms:.3f} ms "
            f"exposed={result.communication_exposed_ms:.3f} ms "
            f"exposed/step={result.exposed_ms_per_step:.3f} ms "
            f"overlap={100 * result.overlap_fraction:.1f}%"
        )
    critical = max(result.exposed_ms_per_step for result in results)
    print(f"critical_rank_exposed_per_step={critical:.3f} ms")

    if args.json is not None:
        payload = {
            "trace": str(args.trace.resolve()),
            "sqlite": str(sqlite_path),
            "nvtx_range": NVTX_RANGE,
            "steps": args.steps,
            "contexts": [asdict(result) for result in results],
            "critical_rank_exposed_ms_per_step": critical,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
