import torch

from nano_megatron.config import QwenConfig
from nano_megatron.model.qwen import QwenAttention, QwenStyleModel, QwenTransformerBlock
from nano_megatron.random import set_seed


class TestQwenAttention:
    def setup_method(self) -> None:
        set_seed(42)
        self.config = QwenConfig()

    def test_forward_shape(self) -> None:
        attn = QwenAttention(self.config)
        x = torch.randn(2, self.config.seq_length, self.config.hidden_size)
        out = attn(x)
        assert out.shape == (2, self.config.seq_length, self.config.hidden_size)

    def test_gradient_flow(self) -> None:
        attn = QwenAttention(self.config)
        x = torch.randn(2, self.config.seq_length, self.config.hidden_size, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_gqa_forward_shape(self) -> None:
        config = QwenConfig(num_attention_heads=8, num_key_value_heads=2, hidden_size=128)
        attn = QwenAttention(config)
        x = torch.randn(2, config.seq_length, config.hidden_size)
        out = attn(x)
        assert out.shape == (2, config.seq_length, config.hidden_size)

    def test_gqa_gradient_flow(self) -> None:
        config = QwenConfig(num_attention_heads=8, num_key_value_heads=2, hidden_size=128)
        attn = QwenAttention(config)
        x = torch.randn(2, config.seq_length, config.hidden_size, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestQwenTransformerBlock:
    def setup_method(self) -> None:
        set_seed(42)
        self.config = QwenConfig()

    def test_forward_shape(self) -> None:
        block = QwenTransformerBlock(self.config)
        x = torch.randn(2, self.config.seq_length, self.config.hidden_size)
        out = block(x)
        assert out.shape == (2, self.config.seq_length, self.config.hidden_size)

    def test_gradient_flow(self) -> None:
        block = QwenTransformerBlock(self.config)
        x = torch.randn(2, self.config.seq_length, self.config.hidden_size, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestQwenStyleModel:
    def setup_method(self) -> None:
        set_seed(42)
        self.config = QwenConfig()

    def test_forward_shape(self) -> None:
        model = QwenStyleModel(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (2, self.config.seq_length))
        logits, loss = model(input_ids)
        assert logits.shape == (2, self.config.seq_length, self.config.vocab_size)
        assert loss is None

    def test_forward_with_labels(self) -> None:
        model = QwenStyleModel(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (2, self.config.seq_length))
        labels = input_ids.clone()
        logits, loss = model(input_ids, labels=labels)
        assert logits.shape == (2, self.config.seq_length, self.config.vocab_size)
        assert loss is not None
        assert loss.item() > 0

    def test_backward_gradient_exists(self) -> None:
        model = QwenStyleModel(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (2, self.config.seq_length))
        labels = input_ids.clone()
        _, loss = model(input_ids, labels=labels)
        assert loss is not None
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_optimizer_step_changes_params(self) -> None:
        set_seed(42)
        config = QwenConfig()
        model = QwenStyleModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, config.seq_length))
        labels = input_ids.clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        before = {n: p.data.clone() for n, p in model.named_parameters()}
        _, loss = model(input_ids, labels=labels)
        assert loss is not None
        loss.backward()
        optimizer.step()

        changed = any(not torch.equal(before[n], p.data) for n, p in model.named_parameters())
        assert changed

    def test_tied_embeddings(self) -> None:
        config = QwenConfig(tie_word_embeddings=True)
        model = QwenStyleModel(config)
        assert model.lm_head.weight is model.token_embedding.embedding.weight

    def test_untied_embeddings(self) -> None:
        config = QwenConfig(tie_word_embeddings=False)
        model = QwenStyleModel(config)
        assert model.lm_head.weight is not model.token_embedding.embedding.weight

    def test_gqa_model_forward(self) -> None:
        config = QwenConfig(num_attention_heads=8, num_key_value_heads=2, hidden_size=128)
        model = QwenStyleModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, config.seq_length))
        logits, loss = model(input_ids, labels=input_ids)
        assert logits.shape == (2, config.seq_length, config.vocab_size)
        assert loss is not None
        assert not torch.isnan(loss)

    def test_no_nan_in_output(self) -> None:
        config = QwenConfig()
        model = QwenStyleModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, config.seq_length))
        logits, loss = model(input_ids, labels=input_ids)
        assert not torch.isnan(logits).any()
        assert loss is not None
        assert not torch.isnan(loss)

    def test_different_batch_sizes(self) -> None:
        config = QwenConfig()
        model = QwenStyleModel(config)
        for B in [1, 2, 4]:
            input_ids = torch.randint(0, config.vocab_size, (B, config.seq_length))
            logits, loss = model(input_ids, labels=input_ids)
            assert logits.shape[0] == B
            assert loss is not None
