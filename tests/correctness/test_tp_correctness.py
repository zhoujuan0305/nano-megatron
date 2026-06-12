from __future__ import annotations

import pytest
import torch

from nano_megatron.config import GPTConfig
from nano_megatron.model.gpt import GPTModel
from nano_megatron.model.tp_gpt import TPGPTModel
from nano_megatron.random import set_seed
from nano_megatron.tensor_parallel.layers import _ensure_divisibility


class TestTPAlignsWithBaseline:
    def test_tp1_matches_gpt_forward(self) -> None:
        set_seed(42)
        config = GPTConfig(
            hidden_size=64, num_layers=2, num_attention_heads=4, seq_length=16, vocab_size=128
        )
        model_gpt = GPTModel(config)
        model_tp = TPGPTModel(config)

        model_gpt_state = model_gpt.state_dict()
        tp_state = model_tp.state_dict()

        for key in tp_state:
            assert key in model_gpt_state, f"Key {key} not found in GPT model"
            assert tp_state[key].shape == model_gpt_state[key].shape, (
                f"Shape mismatch for {key}: TP={tp_state[key].shape}, GPT={model_gpt_state[key].shape}"
            )

        model_tp.load_state_dict(model_gpt_state, strict=False)

        set_seed(99)
        input_ids = torch.randint(0, config.vocab_size, (2, config.seq_length))

        model_gpt.eval()
        model_tp.eval()
        with torch.no_grad():
            logits_gpt, _ = model_gpt(input_ids)
            logits_tp, _ = model_tp(input_ids)

        assert logits_gpt.shape == logits_tp.shape, (
            f"Shape mismatch: GPT={logits_gpt.shape}, TP={logits_tp.shape}"
        )
        assert not torch.isnan(logits_tp).any(), "TP model logits contain NaN"

    def test_tp1_loss_positive(self) -> None:
        set_seed(42)
        config = GPTConfig(
            hidden_size=64, num_layers=2, num_attention_heads=4, seq_length=16, vocab_size=128
        )
        model = TPGPTModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, config.seq_length))
        labels = input_ids.clone()
        _, loss = model(input_ids, labels=labels)
        assert loss is not None
        assert loss.item() > 0

    def test_tp1_backward_gradient_exists(self) -> None:
        set_seed(42)
        config = GPTConfig(
            hidden_size=64, num_layers=2, num_attention_heads=4, seq_length=16, vocab_size=128
        )
        model = TPGPTModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, config.seq_length))
        labels = input_ids.clone()
        _, loss = model(input_ids, labels=labels)
        assert loss is not None
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_tp1_loss_decreases(self) -> None:
        set_seed(42)
        config = GPTConfig(
            hidden_size=64, num_layers=2, num_attention_heads=4, seq_length=16, vocab_size=128
        )
        model = TPGPTModel(config)
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

    def test_hidden_size_not_divisible_by_tp2_raises(self) -> None:
        with pytest.raises(ValueError, match="not divisible"):
            _ensure_divisibility(65, 2)

    def test_heads_not_divisible_by_tp2_raises(self) -> None:
        with pytest.raises(ValueError, match="not divisible"):
            _ensure_divisibility(5, 2)


class TestEnsureDivisibility:
    def test_divisible(self) -> None:
        _ensure_divisibility(128, 2)
        _ensure_divisibility(128, 4)
        _ensure_divisibility(64, 8)

    def test_not_divisible(self) -> None:
        with pytest.raises(ValueError, match="not divisible"):
            _ensure_divisibility(7, 2)
        with pytest.raises(ValueError, match="not divisible"):
            _ensure_divisibility(127, 4)
