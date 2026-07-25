#!/usr/bin/env python3
"""Run ReferenceGPT and save a multi-step FP32 trajectory for oracle comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from nano_megatron.reference import (
    AdamW,
    ReferenceGPT,
    ReferenceGPTConfig,
    StepResult,
    reference_train_loop,
    seed_all,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--out", type=str, required=True, help="torch.save path for trajectory")
    p.add_argument("--vocab-size", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--hidden-size", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--ffn-hidden-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    return p.parse_args()


def step_result_to_dict(result: StepResult) -> dict[str, Any]:
    return {
        "step": result.step,
        "loss": result.loss,
        "logits": result.logits,
        "params": result.params,
        "grads": result.grads,
        "activations": result.activations,
        "optimizer_state": result.optimizer_state,
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if args.seq_len > args.max_seq_len:
        raise ValueError(
            f"seq_len ({args.seq_len}) must be <= max_seq_len ({args.max_seq_len})"
        )
    if args.steps < 0:
        raise ValueError(f"steps must be non-negative, got {args.steps}")

    device = torch.device(args.device)
    seed_all(args.seed)

    cfg = ReferenceGPTConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_hidden_size=args.ffn_hidden_size,
    )
    model = ReferenceGPT(cfg).to(device)
    optimizer = AdamW(
        list(model.named_parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    input_ids = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.seq_len),
        device=device,
        dtype=torch.long,
    )
    batches = [input_ids]

    results = reference_train_loop(
        model,
        optimizer,
        batches,
        steps=args.steps,
        capture_every=1,
        capture_level="full",
    )
    trajectory = [step_result_to_dict(r) for r in results]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trajectory, out_path)
    print(f"saved {len(trajectory)} steps to {out_path}")


if __name__ == "__main__":
    main()
