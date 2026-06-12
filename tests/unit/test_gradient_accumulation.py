import torch

from nano_megatron.config import GPTConfig, QwenConfig
from nano_megatron.data.synthetic import SyntheticDataset
from nano_megatron.model.gpt import GPTModel
from nano_megatron.model.qwen import QwenStyleModel
from nano_megatron.random import set_seed
from nano_megatron.training.trainer import Trainer


class TestGradientAccumulation:
    def test_gpt_grad_accum_equivalent(self) -> None:
        set_seed(42)
        config = GPTConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            seq_length=16,
            vocab_size=128,
            batch_size=2,
            num_steps=1,
            gradient_accumulation_steps=1,
        )
        config_accum = GPTConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            seq_length=16,
            vocab_size=128,
            batch_size=2,
            num_steps=1,
            gradient_accumulation_steps=4,
        )

        set_seed(42)
        model_single = GPTModel(config)
        dataset = SyntheticDataset(
            vocab_size=config.vocab_size, seq_length=config.seq_length, num_samples=64, seed=42
        )
        dataloader_single = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False
        )

        set_seed(123)
        model_accum = GPTModel(config_accum)
        model_accum.load_state_dict(model_single.state_dict())

        dataloader_accum = torch.utils.data.DataLoader(
            dataset, batch_size=config_accum.batch_size, shuffle=False
        )

        trainer_single = Trainer(
            config=config, model=model_single, train_dataloader=dataloader_single
        )
        trainer_accum = Trainer(
            config=config_accum, model=model_accum, train_dataloader=dataloader_accum
        )

        result_single = trainer_single.train_step([next(iter(dataloader_single))])
        accum_batches = [next(iter(dataloader_accum)) for _ in range(4)]
        result_accum = trainer_accum.train_step(accum_batches)

        assert result_single.loss > 0
        assert result_accum.loss > 0

    def test_qwen_grad_accum_runs(self) -> None:
        set_seed(42)
        config = QwenConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            seq_length=16,
            vocab_size=128,
            batch_size=2,
            num_steps=3,
            gradient_accumulation_steps=2,
        )
        model = QwenStyleModel(config)
        dataset = SyntheticDataset(
            vocab_size=config.vocab_size, seq_length=config.seq_length, num_samples=64, seed=42
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
        trainer = Trainer(config=config, model=model, train_dataloader=dataloader)
        results = trainer.train(num_steps=3)
        assert len(results) == 3
        for r in results:
            assert r.loss > 0
            assert r.micro_batches == 2

    def test_grad_accum_micro_batches_count(self) -> None:
        set_seed(42)
        config = GPTConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            seq_length=16,
            vocab_size=128,
            batch_size=2,
            gradient_accumulation_steps=3,
        )
        model = GPTModel(config)
        dataset = SyntheticDataset(
            vocab_size=config.vocab_size, seq_length=config.seq_length, num_samples=64, seed=42
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
        trainer = Trainer(config=config, model=model, train_dataloader=dataloader)
        data_iter = iter(dataloader)
        batches = [next(data_iter) for _ in range(3)]
        result = trainer.train_step(batches)
        assert result.micro_batches == 3
