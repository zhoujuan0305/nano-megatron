import torch

from nano_megatron.model.embedding import PositionEmbedding, TokenEmbedding
from nano_megatron.random import set_seed


class TestTokenEmbedding:
    def setup_method(self) -> None:
        set_seed(42)

    def test_forward_shape(self) -> None:
        emb = TokenEmbedding(vocab_size=1024, hidden_size=128)
        input_ids = torch.randint(0, 1024, (2, 32))
        out = emb(input_ids)
        assert out.shape == (2, 32, 128)

    def test_gradient_flow(self) -> None:
        emb = TokenEmbedding(vocab_size=1024, hidden_size=128)
        input_ids = torch.randint(0, 1024, (2, 32))
        out = emb(input_ids)
        loss = out.sum()
        loss.backward()
        assert emb.embedding.weight.grad is not None


class TestPositionEmbedding:
    def setup_method(self) -> None:
        set_seed(42)

    def test_forward_shape(self) -> None:
        emb = PositionEmbedding(max_seq_len=512, hidden_size=128)
        out = emb(32)
        assert out.shape == (32, 128)

    def test_gradient_flow(self) -> None:
        emb = PositionEmbedding(max_seq_len=512, hidden_size=128)
        out = emb(32)
        loss = out.sum()
        loss.backward()
        assert emb.embedding.weight.grad is not None
