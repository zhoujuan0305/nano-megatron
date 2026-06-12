import torch

from nano_megatron.config import QwenConfig
from nano_megatron.model.qwen import QwenStyleModel
from nano_megatron.random import set_seed


class TestQwenCorrectness:
    def test_forward_shape_various_sizes(self) -> None:
        configs = [
            QwenConfig(
                hidden_size=64,
                num_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                seq_length=16,
                vocab_size=512,
            ),
            QwenConfig(
                hidden_size=128, num_layers=4, num_attention_heads=8, seq_length=32, vocab_size=1024
            ),
            QwenConfig(
                hidden_size=128,
                num_layers=2,
                num_attention_heads=8,
                num_key_value_heads=4,
                seq_length=32,
                vocab_size=1024,
            ),
        ]
        for config in configs:
            model = QwenStyleModel(config)
            input_ids = torch.randint(0, config.vocab_size, (2, config.seq_length))
            logits, loss = model(input_ids, labels=input_ids)
            assert logits.shape == (2, config.seq_length, config.vocab_size)
            assert loss is not None
            assert loss.item() > 0

    def test_loss_decreases(self) -> None:
        set_seed(42)
        config = QwenConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            seq_length=16,
            vocab_size=128,
        )
        model = QwenStyleModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        set_seed(42)
        input_ids = torch.randint(0, config.vocab_size, (4, config.seq_length))
        labels = input_ids.clone()

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            _, loss = model(input_ids, labels=labels)
            assert loss is not None
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_seed_reproducibility(self) -> None:
        for seed in [42, 123]:
            set_seed(seed)
            config = QwenConfig(
                hidden_size=64,
                num_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                seq_length=16,
                vocab_size=128,
            )
            model1 = QwenStyleModel(config)
            set_seed(seed)
            model2 = QwenStyleModel(config)

            input_ids = torch.randint(0, config.vocab_size, (2, config.seq_length))
            with torch.no_grad():
                logits1, _ = model1(input_ids)
                logits2, _ = model2(input_ids)
            assert torch.allclose(logits1, logits2, atol=1e-6), f"Seed {seed} not reproducible"

    def test_backward_gradient_numerics(self) -> None:
        set_seed(42)
        config = QwenConfig(hidden_size=128, num_layers=2, num_attention_heads=8)
        model = QwenStyleModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, config.seq_length))
        labels = input_ids.clone()
        _, loss = model(input_ids, labels=labels)
        assert loss is not None
        loss.backward()

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        assert total_norm > 0
        assert not torch.isnan(torch.tensor(total_norm))

    def test_causal_mask_effect(self) -> None:
        config = QwenConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            seq_length=32,
            vocab_size=128,
        )
        model = QwenStyleModel(config)
        model.eval()

        set_seed(42)
        input_ids = torch.randint(0, config.vocab_size, (1, config.seq_length))

        with torch.no_grad():
            logits_full, _ = model(input_ids)
            prefix_len = 16
            input_prefix = input_ids[:, :prefix_len]
            logits_prefix, _ = model(input_prefix)

        assert torch.allclose(logits_full[:, :prefix_len, :], logits_prefix, atol=1e-4), (
            "Causal mask violated: prefix outputs differ"
        )

    def test_no_nan_in_output(self) -> None:
        config = QwenConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            seq_length=16,
            vocab_size=128,
        )
        model = QwenStyleModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, config.seq_length))
        logits, loss = model(input_ids, labels=input_ids)
        assert not torch.isnan(logits).any()
        assert loss is not None
        assert not torch.isnan(loss)

    def test_gqa_loss_decreases(self) -> None:
        set_seed(42)
        config = QwenConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            seq_length=16,
            vocab_size=128,
        )
        model = QwenStyleModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        set_seed(42)
        input_ids = torch.randint(0, config.vocab_size, (4, config.seq_length))
        labels = input_ids.clone()

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            _, loss = model(input_ids, labels=labels)
            assert loss is not None
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        assert losses[-1] < losses[0]
