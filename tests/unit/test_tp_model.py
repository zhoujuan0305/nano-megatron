import torch

from nano_megatron.config import GPTConfig
from nano_megatron.model.tp_attention import TPCausalSelfAttention
from nano_megatron.model.tp_gpt import TPGPTModel
from nano_megatron.model.tp_mlp import TPMLP
from nano_megatron.random import set_seed


class TestTPMLPSingleGPU:
    def setup_method(self) -> None:
        set_seed(42)

    def test_forward_shape(self) -> None:
        mlp = TPMLP(hidden_size=128, mlp_hidden_size=512)
        x = torch.randn(2, 32, 128)
        out = mlp(x)
        assert out.shape == (2, 32, 128)

    def test_gradient_flow(self) -> None:
        mlp = TPMLP(hidden_size=128, mlp_hidden_size=512)
        x = torch.randn(2, 32, 128, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestTPCausalSelfAttentionSingleGPU:
    def setup_method(self) -> None:
        set_seed(42)

    def test_forward_shape(self) -> None:
        config = GPTConfig()
        attn = TPCausalSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
        )
        x = torch.randn(2, config.seq_length, config.hidden_size)
        out = attn(x)
        assert out.shape == (2, config.seq_length, config.hidden_size)

    def test_gradient_flow(self) -> None:
        config = GPTConfig()
        attn = TPCausalSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
        )
        x = torch.randn(2, config.seq_length, config.hidden_size, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_no_nan(self) -> None:
        config = GPTConfig()
        attn = TPCausalSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
        )
        x = torch.randn(1, 8, config.hidden_size)
        out = attn(x)
        assert not torch.isnan(out).any()


class TestTPGPTModelSingleGPU:
    def setup_method(self) -> None:
        set_seed(42)

    def test_forward_shape(self) -> None:
        config = GPTConfig()
        model = TPGPTModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, config.seq_length))
        logits, loss = model(input_ids)
        assert logits.shape == (2, config.seq_length, config.vocab_size)
        assert loss is None

    def test_forward_with_labels(self) -> None:
        config = GPTConfig()
        model = TPGPTModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, config.seq_length))
        labels = input_ids.clone()
        logits, loss = model(input_ids, labels=labels)
        assert logits.shape == (2, config.seq_length, config.vocab_size)
        assert loss is not None
        assert loss.item() > 0

    def test_backward_gradient_exists(self) -> None:
        config = GPTConfig()
        model = TPGPTModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, config.seq_length))
        labels = input_ids.clone()
        _, loss = model(input_ids, labels=labels)
        assert loss is not None
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_optimizer_step_changes_params(self) -> None:
        set_seed(42)
        config = GPTConfig()
        model = TPGPTModel(config)
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

    def test_different_batch_sizes(self) -> None:
        config = GPTConfig()
        model = TPGPTModel(config)
        for B in [1, 2, 4]:
            input_ids = torch.randint(0, config.vocab_size, (B, config.seq_length))
            logits, _ = model(input_ids, labels=input_ids)
            assert logits.shape[0] == B
