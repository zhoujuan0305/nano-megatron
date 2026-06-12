from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from nano_megatron.config import GPTConfig, QwenConfig
from nano_megatron.data_parallel.grad_sync import average_gradients, broadcast_parameters
from nano_megatron.distributed.parallel_state import get_data_parallel_rank, get_data_parallel_size
from nano_megatron.training.checkpointing import get_rng_state, save_checkpoint
from nano_megatron.training.optimizer import create_adam_optimizer
from nano_megatron.training.scheduler import CosineWarmupScheduler
from nano_megatron.utils.device import get_device
from nano_megatron.utils.memory import get_peak_memory_mb

logger = logging.getLogger(__name__)


@dataclass
class TrainStepResult:
    step: int
    loss: float
    lr: float
    step_time_ms: float
    tokens_per_sec: float
    peak_memory_mb: float
    micro_batches: int


class Trainer:
    def __init__(
        self,
        config: GPTConfig | QwenConfig,
        model: nn.Module,
        train_dataloader: DataLoader,
        dp_group: torch.distributed.ProcessGroup | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.dp_group = dp_group

        if dp_group is not None:
            dp_rank = get_data_parallel_rank()
            self.device = torch.device(f"cuda:{dp_rank}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = get_device()

        self.model = self.model.to(self.device)

        if dp_group is not None:
            broadcast_parameters(self.model, src=0, group=dp_group)

        self.optimizer = create_adam_optimizer(
            self.model, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.num_steps,
        )
        self.global_step = 0

    def train_step(self, micro_batches: list[tuple[torch.Tensor, torch.Tensor]]) -> TrainStepResult:
        self.model.train()
        start_time = time.perf_counter()

        accumulated_loss = 0.0
        total_tokens = 0
        num_micro = len(micro_batches)

        for input_ids, labels in micro_batches:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            _, loss = self.model(input_ids, labels=labels)
            scaled_loss = loss / num_micro
            scaled_loss.backward()

            accumulated_loss += loss.item()
            total_tokens += input_ids.shape[0] * input_ids.shape[1]

        if self.dp_group is not None:
            average_gradients(self.model, group=self.dp_group)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        self.global_step += 1
        elapsed = time.perf_counter() - start_time

        peak_mem = get_peak_memory_mb(self.device)

        if self.dp_group is not None:
            dp_size = get_data_parallel_size()
            loss_tensor = torch.tensor([accumulated_loss / num_micro], device=self.device)
            torch.distributed.all_reduce(
                loss_tensor, op=torch.distributed.ReduceOp.SUM, group=self.dp_group
            )
            avg_loss = loss_tensor.item() / dp_size
        else:
            avg_loss = accumulated_loss / num_micro

        return TrainStepResult(
            step=self.global_step,
            loss=avg_loss,
            lr=self.scheduler.get_last_lr()[0],
            step_time_ms=elapsed * 1000,
            tokens_per_sec=total_tokens / elapsed,
            peak_memory_mb=peak_mem,
            micro_batches=num_micro,
        )

    def train(self, num_steps: int | None = None) -> list[TrainStepResult]:
        if num_steps is None:
            num_steps = self.config.num_steps

        results: list[TrainStepResult] = []
        data_iter = iter(self.train_dataloader)

        for _ in range(num_steps):
            micro_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
            for _ in range(self.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)
                micro_batches.append(batch)

            result = self.train_step(micro_batches)
            results.append(result)

            if result.step % 10 == 0 or result.step == 1:
                dp_info = ""
                if self.dp_group is not None:
                    dp_info = f" dp_rank={get_data_parallel_rank()}/{get_data_parallel_size()}"
                logger.info(
                    f"step={result.step} loss={result.loss:.4f} "
                    f"lr={result.lr:.6f} time={result.step_time_ms:.1f}ms "
                    f"tok/s={result.tokens_per_sec:.0f} "
                    f"micro={result.micro_batches}{dp_info} "
                    f"mem={result.peak_memory_mb:.1f}MB"
                )

        return results

    def save_checkpoint(self, path: str) -> None:
        if isinstance(self.config, QwenConfig):
            config_dict = {
                "model_type": "qwen",
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "num_attention_heads": self.config.num_attention_heads,
                "num_key_value_heads": self.config.num_key_value_heads,
                "seq_length": self.config.seq_length,
                "vocab_size": self.config.vocab_size,
                "mlp_ratio": self.config.mlp_ratio,
                "dropout": self.config.dropout,
                "max_position_embeddings": self.config.max_position_embeddings,
                "tie_word_embeddings": self.config.tie_word_embeddings,
            }
        else:
            config_dict = {
                "model_type": "gpt",
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "num_attention_heads": self.config.num_attention_heads,
                "seq_length": self.config.seq_length,
                "vocab_size": self.config.vocab_size,
                "mlp_ratio": self.config.mlp_ratio,
                "dropout": self.config.dropout,
                "max_position_embeddings": self.config.max_position_embeddings,
            }
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            step=self.global_step,
            config=config_dict,
            rng_state=get_rng_state(),
        )
        logger.info(f"Checkpoint saved to {path} at step {self.global_step}")
