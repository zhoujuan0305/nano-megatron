import argparse
import logging
import time

import torch

from nano_megatron.config import GPTConfig, QwenConfig
from nano_megatron.data.synthetic import SyntheticDataset
from nano_megatron.model.gpt import GPTModel
from nano_megatron.model.qwen import QwenStyleModel
from nano_megatron.random import set_seed
from nano_megatron.utils.device import get_device
from nano_megatron.utils.memory import get_peak_memory_mb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a tiny model")
    parser.add_argument("--model-type", type=str, default="gpt", choices=["gpt", "qwen"])
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-attention-heads", type=int, default=None)
    parser.add_argument("--num-key-value-heads", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.model_type == "qwen":
        config_kwargs: dict = {}
        if args.hidden_size is not None:
            config_kwargs["hidden_size"] = args.hidden_size
        if args.num_layers is not None:
            config_kwargs["num_layers"] = args.num_layers
        if args.num_attention_heads is not None:
            config_kwargs["num_attention_heads"] = args.num_attention_heads
        if args.num_key_value_heads is not None:
            config_kwargs["num_key_value_heads"] = args.num_key_value_heads
        if args.seq_length is not None:
            config_kwargs["seq_length"] = args.seq_length
        if args.vocab_size is not None:
            config_kwargs["vocab_size"] = args.vocab_size
        if args.batch_size is not None:
            config_kwargs["batch_size"] = args.batch_size
        config = QwenConfig(**config_kwargs)
    else:
        config_kwargs = {}
        if args.hidden_size is not None:
            config_kwargs["hidden_size"] = args.hidden_size
        if args.num_layers is not None:
            config_kwargs["num_layers"] = args.num_layers
        if args.num_attention_heads is not None:
            config_kwargs["num_attention_heads"] = args.num_attention_heads
        if args.seq_length is not None:
            config_kwargs["seq_length"] = args.seq_length
        if args.vocab_size is not None:
            config_kwargs["vocab_size"] = args.vocab_size
        if args.batch_size is not None:
            config_kwargs["batch_size"] = args.batch_size
        config = GPTConfig(**config_kwargs)

    set_seed(config.seed)
    device = get_device()

    if args.model_type == "qwen":
        model = QwenStyleModel(config)
        model_type_str = "qwen"
    else:
        model = GPTModel(config)
        model_type_str = "gpt"

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Benchmark config: model={model_type_str} params={num_params:,}")
    logger.info(
        f"  hidden_size={config.hidden_size} layers={config.num_layers} "
        f"heads={config.num_attention_heads}"
    )
    logger.info(
        f"  seq_length={config.seq_length} vocab={config.vocab_size} batch_size={config.batch_size}"
    )
    logger.info(f"  device={device} dtype=float32")

    dataset = SyntheticDataset(
        vocab_size=config.vocab_size,
        seq_length=config.seq_length,
        num_samples=config.batch_size * args.num_steps * 2,
        seed=config.seed,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    model.train()
    data_iter = iter(dataloader)

    logger.info("Warming up...")
    for _ in range(args.warmup_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        _, loss = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

    logger.info(f"Benchmarking {args.num_steps} steps...")
    total_tokens = 0
    total_time = 0.0

    data_iter = iter(dataloader)

    for i in range(args.num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        start = time.perf_counter()
        optimizer.zero_grad()
        _, loss = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_tokens += input_ids.shape[0] * input_ids.shape[1]
        total_time += elapsed

        if (i + 1) % 5 == 0:
            logger.info(
                f"  step {i + 1}/{args.num_steps} loss={loss.item():.4f} "
                f"time={elapsed * 1000:.1f}ms"
            )

    peak_mem_after = get_peak_memory_mb(device)
    avg_step_time = total_time / args.num_steps * 1000
    tokens_per_sec = total_tokens / total_time
    samples_per_sec = config.batch_size * args.num_steps / total_time

    logger.info("=" * 50)
    logger.info("Benchmark Results:")
    logger.info(f"  model_name: {model_type_str}")
    logger.info("  parallel_mode: single_gpu")
    logger.info("  world_size: 1")
    logger.info("  dp=1 tp=1 pp=1")
    logger.info(f"  global_batch_size: {config.batch_size}")
    logger.info(f"  micro_batch_size: {config.batch_size}")
    logger.info(f"  seq_length: {config.seq_length}")
    logger.info(f"  hidden_size: {config.hidden_size}")
    logger.info(f"  num_layers: {config.num_layers}")
    logger.info(f"  num_attention_heads: {config.num_attention_heads}")
    logger.info("  dtype: float32")
    logger.info(f"  steps: {args.num_steps}")
    logger.info(f"  avg_step_time_ms: {avg_step_time:.2f}")
    logger.info(f"  tokens_per_sec: {tokens_per_sec:.1f}")
    logger.info(f"  samples_per_sec: {samples_per_sec:.2f}")
    logger.info(f"  peak_memory_mb: {peak_mem_after:.1f}")
    logger.info(f"  final_loss: {loss.item():.4f}")
    logger.info("  recompute: False")
    logger.info("  mixed_precision: False")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
