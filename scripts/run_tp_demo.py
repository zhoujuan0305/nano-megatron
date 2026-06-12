import argparse
import logging

import torch
import torch.distributed as dist

from nano_megatron.config import GPTConfig
from nano_megatron.data.synthetic import SyntheticDataset
from nano_megatron.distributed.initialize import destroy_distributed, init_distributed
from nano_megatron.distributed.parallel_state import (
    create_tensor_parallel_group,
    get_tensor_parallel_rank,
    get_tensor_parallel_size,
)
from nano_megatron.model.tp_gpt import TPGPTModel
from nano_megatron.random import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tensor Parallel training demo")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    init_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    create_tensor_parallel_group(tp_size=world_size)
    tp_rank = get_tensor_parallel_rank()
    tp_size = get_tensor_parallel_size()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    logger.info(f"rank={rank} tp_rank={tp_rank} tp_size={tp_size} device={device}")

    config = GPTConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    if config.hidden_size % tp_size != 0:
        raise ValueError(
            f"hidden_size ({config.hidden_size}) must be divisible by tp_size ({tp_size})"
        )
    if config.num_attention_heads % tp_size != 0:
        raise ValueError(
            f"num_attention_heads ({config.num_attention_heads}) must be divisible by "
            f"tp_size ({tp_size})"
        )

    set_seed(config.seed + tp_rank)
    model = TPGPTModel(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"rank={rank} tp_rank={tp_rank} params_per_rank={num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    dataset = SyntheticDataset(
        vocab_size=config.vocab_size,
        seq_length=config.seq_length,
        num_samples=256,
        seed=config.seed,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model.train()
    data_iter = iter(dataloader)

    losses: list[float] = []
    for step in range(1, config.num_steps + 1):
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

        loss_val = loss.item()
        losses.append(loss_val)

        if step % 5 == 0 or step == 1:
            logger.info(f"step={step} loss={loss_val:.4f} rank={rank} tp_rank={tp_rank}")

    if len(losses) >= 2:
        logger.info(f"First 5 avg loss: {sum(losses[:5]) / min(5, len(losses)):.4f}")
        logger.info(f"Last 5 avg loss: {sum(losses[-5:]) / min(5, len(losses)):.4f}")

    logger.info(f"rank={rank} tp_rank={tp_rank} training complete")

    destroy_distributed()


if __name__ == "__main__":
    main()
