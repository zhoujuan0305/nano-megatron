import tempfile
from pathlib import Path

import torch

from nano_megatron.config import GPTConfig
from nano_megatron.data.synthetic import SyntheticDataset
from nano_megatron.model.gpt import GPTModel
from nano_megatron.random import set_seed
from nano_megatron.training.checkpointing import load_checkpoint
from nano_megatron.training.optimizer import create_adam_optimizer
from nano_megatron.training.trainer import Trainer


class TestTrainTinyGPT:
    def test_trainer_runs(self) -> None:
        set_seed(42)
        config = GPTConfig(num_steps=5)
        model = GPTModel(config)
        dataset = SyntheticDataset(
            vocab_size=config.vocab_size,
            seq_length=config.seq_length,
            num_samples=32,
            seed=config.seed,
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
        trainer = Trainer(config=config, model=model, train_dataloader=dataloader)
        results = trainer.train(num_steps=5)
        assert len(results) == 5
        assert results[-1].loss > 0

    def test_loss_decreases_over_training(self) -> None:
        set_seed(42)
        config = GPTConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            seq_length=16,
            vocab_size=128,
            num_steps=50,
        )
        model = GPTModel(config)
        dataset = SyntheticDataset(
            vocab_size=config.vocab_size,
            seq_length=config.seq_length,
            num_samples=128,
            seed=config.seed,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True
        )
        trainer = Trainer(config=config, model=model, train_dataloader=dataloader)
        results = trainer.train()

        first_5_avg = sum(r.loss for r in results[:5]) / 5
        last_5_avg = sum(r.loss for r in results[-5:]) / 5
        assert last_5_avg < first_5_avg, (
            f"Loss did not decrease: first5={first_5_avg:.4f}, last5={last_5_avg:.4f}"
        )

    def test_checkpoint_save_and_resume(self) -> None:
        set_seed(42)
        config = GPTConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            seq_length=16,
            vocab_size=128,
            num_steps=10,
        )
        model = GPTModel(config)
        dataset = SyntheticDataset(
            vocab_size=config.vocab_size,
            seq_length=config.seq_length,
            num_samples=32,
            seed=config.seed,
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
        trainer = Trainer(config=config, model=model, train_dataloader=dataloader)
        trainer.train(num_steps=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = str(Path(tmpdir) / "ckpt.pt")
            trainer.save_checkpoint(ckpt_path)

            model2 = GPTModel(config)
            optimizer2 = create_adam_optimizer(model2)
            step, _ = load_checkpoint(ckpt_path, model2, optimizer2)
            assert step == 5

            dataset2 = SyntheticDataset(
                vocab_size=config.vocab_size,
                seq_length=config.seq_length,
                num_samples=32,
                seed=config.seed,
            )
            dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=config.batch_size)
            trainer2 = Trainer(config=config, model=model2, train_dataloader=dataloader2)
            trainer2.global_step = step
            results2 = trainer2.train(num_steps=5)
            assert len(results2) == 5
