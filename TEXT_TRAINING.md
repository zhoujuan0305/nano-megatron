# Small real-text training health check

`python -m scripts.train_gpt` exercises nano's native TP/SP/PP training with
changing text batches, forward-only heldout evaluation, and complete
FP32-master AdamW checkpoints. It is an experimental health-check driver,
not a production trainer or evidence of large-model pretraining convergence.
CUDA Graph is not used. Checkpoint restore requires the same topology,
model/data/scheduler configuration and optimizer policy.

## Prepared data

Supply a directory containing `dataset.json` and, for each of `train` and
`validation`, a one-dimensional integer NPY token array and an int64 NPY
array of document token offsets. Offsets begin at zero, are monotonic and
end at the token-array length. Example manifest entry:

```json
{"splits": {"train": {
  "tokens_file": "train.tokens.npy", "tokens_sha256": "<sha256>",
  "offsets_file": "train.offsets.npy", "offsets_sha256": "<sha256>"
}, "validation": {
  "tokens_file": "validation.tokens.npy", "tokens_sha256": "<sha256>",
  "offsets_file": "validation.offsets.npy", "offsets_sha256": "<sha256>"
}}}
```

Retain corpus/tokenizer attribution, revision and preparation provenance in
additional manifest fields. Do not mix validation documents into training.
The sampler verifies file hashes, chooses legal document-bounded starts
uniformly with replacement, and deterministically derives each batch from
its split, seed and update index. It is not a sequential epoch traversal.
The fixed heldout panel is not full-dataset or word-level perplexity.

## Run and resume

```bash
torchrun --standalone --nproc_per_node=1 -m scripts.train_gpt \
  --data-dir ./prepared --output ./runs/text-single
```

For four GPUs, use a valid rendezvous and `--nproc_per_node`/`--nnodes`
launch appropriate for the hardware, adding `--tp-size 2 --pp-size 2`.
Global batch is `--batch-size`; it must divide into `--microbatches`.
This driver supports DP1 only.

```bash
# Same schedule (10000 total updates), stop at the persisted midpoint.
torchrun --standalone --nproc_per_node=1 -m scripts.train_gpt \
  --data-dir ./prepared --output ./runs/text-first --stop-after 5000
# Fresh process; performs only the remaining updates, not another 10000.
torchrun --standalone --nproc_per_node=1 -m scripts.train_gpt \
  --data-dir ./prepared --output ./runs/text-resumed \
  --resume ./runs/text-first/checkpoint-00005000
```

Outputs must be new directories. Model/master weights, AdamW moments,
device and CPU counters, RNG, data hash-chain and semantic configuration
are saved. Only load trusted local checkpoints. Current LR is restored;
the next update derives LR from the saved step and unchanged schedule.
Per-rank metrics distinguish nominal input tokens and supervised next-token
targets; long-run wall time includes sampling, validation and checkpoint IO.
