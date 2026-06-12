import argparse
import logging

import torch
import torch.distributed as dist

from nano_megatron.config import GPTConfig
from nano_megatron.data.synthetic import SyntheticDataset
from nano_megatron.data_parallel.grad_sync import average_gradients, broadcast_parameters
from nano_megatron.distributed.initialize import destroy_distributed, init_distributed
from nano_megatron.distributed.parallel_state import (
    create_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_size,
    get_global_rank,
)
from nano_megatron.model.gpt import GPTModel
from nano_megatron.random import set_seed
from nano_megatron.training.optimizer import create_adam_optimizer
from nano_megatron.training.scheduler import CosineWarmupScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Data Parallel training demo")
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

    rank = get_global_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    dp_group = create_data_parallel_group()
    dp_rank = get_data_parallel_rank()
    dp_size = get_data_parallel_size()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{dp_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    logger.info(
        f"rank={rank} dp_rank={dp_rank} dp_size={dp_size} world_size={world_size} device={device}"
    )

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

    set_seed(config.seed + dp_rank)
    model = GPTModel(config).to(device)
    broadcast_parameters(model, src=0, group=dp_group)

    optimizer = create_adam_optimizer(model, lr=config.learning_rate)
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer, warmup_steps=config.warmup_steps, max_steps=config.num_steps
    )

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

        average_gradients(model, group=dp_group)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_tensor = torch.tensor([loss.item()], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM, group=dp_group)
        avg_loss = loss_tensor.item() / dp_size

        losses.append(avg_loss)

        if step % 5 == 0 or step == 1:
            logger.info(
                f"step={step} loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f} "
                f"dp_rank={dp_rank}/{dp_size}"
            )

    if len(losses) >= 2:
        logger.info(f"First 5 avg loss: {sum(losses[:5]) / min(5, len(losses)):.4f}")
        logger.info(f"Last 5 avg loss: {sum(losses[-5:]) / min(5, len(losses)):.4f}")

    logger.info(f"rank={rank} dp_rank={dp_rank} training complete")

    destroy_distributed()


if __name__ == "__main__":
    main()
