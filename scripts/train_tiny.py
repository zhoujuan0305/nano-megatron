import argparse
import logging

import torch

from nano_megatron.config import GPTConfig, QwenConfig
from nano_megatron.data.synthetic import SyntheticDataset
from nano_megatron.model.gpt import GPTModel
from nano_megatron.model.qwen import QwenStyleModel
from nano_megatron.random import set_seed
from nano_megatron.training.checkpointing import load_checkpoint
from nano_megatron.training.trainer import Trainer
from nano_megatron.utils.device import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny model (GPT or Qwen-style)")
    parser.add_argument("--model-type", type=str, default="gpt", choices=["gpt", "qwen"])
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-attention-heads", type=int, default=None)
    parser.add_argument("--num-key-value-heads", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", type=str, default=None, help="Path to save checkpoint")
    parser.add_argument("--load", type=str, default=None, help="Path to load checkpoint")
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
        if args.num_steps is not None:
            config_kwargs["num_steps"] = args.num_steps
        if args.learning_rate is not None:
            config_kwargs["learning_rate"] = args.learning_rate
        if args.weight_decay is not None:
            config_kwargs["weight_decay"] = args.weight_decay
        if args.warmup_steps is not None:
            config_kwargs["warmup_steps"] = args.warmup_steps
        if args.gradient_accumulation_steps is not None:
            config_kwargs["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        if args.seed is not None:
            config_kwargs["seed"] = args.seed
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
        if args.num_steps is not None:
            config_kwargs["num_steps"] = args.num_steps
        if args.learning_rate is not None:
            config_kwargs["learning_rate"] = args.learning_rate
        if args.weight_decay is not None:
            config_kwargs["weight_decay"] = args.weight_decay
        if args.warmup_steps is not None:
            config_kwargs["warmup_steps"] = args.warmup_steps
        if args.gradient_accumulation_steps is not None:
            config_kwargs["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        if args.seed is not None:
            config_kwargs["seed"] = args.seed
        config = GPTConfig(**config_kwargs)

    set_seed(config.seed)
    device = get_device()
    logger.info(f"Device: {device}")

    model = QwenStyleModel(config) if args.model_type == "qwen" else GPTModel(config)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"Config: {config}")
    logger.info(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {config.effective_batch_size}")

    dataset = SyntheticDataset(
        vocab_size=config.vocab_size,
        seq_length=config.seq_length,
        num_samples=1024,
        seed=config.seed,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    trainer = Trainer(config=config, model=model, train_dataloader=dataloader)

    if args.load is not None:
        start_step, _ = load_checkpoint(args.load, trainer.model, trainer.optimizer)
        trainer.global_step = start_step
        logger.info(f"Resumed from step {start_step}")

    results = trainer.train()

    if results:
        logger.info(f"Final loss: {results[-1].loss:.4f}")

    if args.save is not None:
        trainer.save_checkpoint(args.save)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
