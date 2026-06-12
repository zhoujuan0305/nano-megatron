import tempfile
from pathlib import Path

import torch

from nano_megatron.config import GPTConfig
from nano_megatron.model.gpt import GPTModel
from nano_megatron.random import set_seed
from nano_megatron.training.checkpointing import load_checkpoint, save_checkpoint
from nano_megatron.training.optimizer import create_adam_optimizer


class TestSaveLoadCheckpoint:
    def test_save_and_load_preserves_weights(self) -> None:
        set_seed(42)
        config = GPTConfig()
        model = GPTModel(config)
        optimizer = create_adam_optimizer(model)

        B = 2
        input_ids = torch.randint(0, config.vocab_size, (B, config.seq_length))
        labels = input_ids.clone()
        _, loss = model(input_ids, labels=labels)
        assert loss is not None
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ckpt.pt")
            save_checkpoint(path, model, optimizer, step=1, config={"test": True})

            model2 = GPTModel(config)
            optimizer2 = create_adam_optimizer(model2)
            step, cfg = load_checkpoint(path, model2, optimizer2)

            assert step == 1
            assert cfg["test"] is True

            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), model2.named_parameters(), strict=True
            ):
                assert n1 == n2
                assert torch.equal(p1, p2), f"Parameter {n1} mismatch after load"

    def test_step_counter_preserved(self) -> None:
        set_seed(42)
        config = GPTConfig()
        model = GPTModel(config)
        optimizer = create_adam_optimizer(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ckpt.pt")
            save_checkpoint(path, model, optimizer, step=42, config={})

            model2 = GPTModel(config)
            optimizer2 = create_adam_optimizer(model2)
            step, _ = load_checkpoint(path, model2, optimizer2)
            assert step == 42
