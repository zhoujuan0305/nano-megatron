import torch

from nano_megatron.config import GPTConfig
from nano_megatron.model.gpt import GPTModel, TransformerBlock
from nano_megatron.random import set_seed


class TestTransformerBlock:
    def setup_method(self) -> None:
        set_seed(42)
        self.config = GPTConfig()

    def test_forward_shape(self) -> None:
        block = TransformerBlock(self.config)
        B, S = 2, self.config.seq_length
        x = torch.randn(B, S, self.config.hidden_size)
        out = block(x)
        assert out.shape == (B, S, self.config.hidden_size)

    def test_gradient_flow(self) -> None:
        block = TransformerBlock(self.config)
        B, S = 2, self.config.seq_length
        x = torch.randn(B, S, self.config.hidden_size, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestGPTModel:
    def setup_method(self) -> None:
        set_seed(42)
        self.config = GPTConfig()

    def test_forward_shape(self) -> None:
        model = GPTModel(self.config)
        B = 2
        input_ids = torch.randint(0, self.config.vocab_size, (B, self.config.seq_length))
        logits, loss = model(input_ids)
        assert logits.shape == (B, self.config.seq_length, self.config.vocab_size)
        assert loss is None

    def test_forward_with_labels(self) -> None:
        model = GPTModel(self.config)
        B = 2
        input_ids = torch.randint(0, self.config.vocab_size, (B, self.config.seq_length))
        labels = input_ids.clone()
        logits, loss = model(input_ids, labels=labels)
        assert logits.shape == (B, self.config.seq_length, self.config.vocab_size)
        assert loss is not None
        assert loss.item() > 0

    def test_backward_gradient_exists(self) -> None:
        model = GPTModel(self.config)
        B = 2
        input_ids = torch.randint(0, self.config.vocab_size, (B, self.config.seq_length))
        labels = input_ids.clone()
        _, loss = model(input_ids, labels=labels)
        assert loss is not None
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_optimizer_step_changes_params(self) -> None:
        set_seed(42)
        model = GPTModel(self.config)
        B = 2
        input_ids = torch.randint(0, self.config.vocab_size, (B, self.config.seq_length))
        labels = input_ids.clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        before = {n: p.data.clone() for n, p in model.named_parameters()}
        _, loss = model(input_ids, labels=labels)
        assert loss is not None
        loss.backward()
        optimizer.step()

        changed = False
        for n, p in model.named_parameters():
            if not torch.equal(before[n], p.data):
                changed = True
                break
        assert changed, "No parameter changed after optimizer step"

    def test_causal_mask_effect(self) -> None:
        model = GPTModel(self.config)
        model.eval()
        B = 1
        input_ids = torch.randint(0, self.config.vocab_size, (B, self.config.seq_length))

        with torch.no_grad():
            logits_full, _ = model(input_ids)

            input_ids_short = input_ids[:, :16]
            logits_short, _ = model(input_ids_short)

        assert torch.allclose(logits_full[:, :16, :], logits_short, atol=1e-5), (
            "Causal mask violated: outputs differ for same prefix"
        )

    def test_different_batch_sizes(self) -> None:
        model = GPTModel(self.config)
        for B in [1, 2, 4]:
            input_ids = torch.randint(0, self.config.vocab_size, (B, self.config.seq_length))
            logits, loss = model(input_ids, labels=input_ids)
            assert logits.shape[0] == B
            assert loss is not None
