import torch

from nano_megatron.model.rope import RotaryEmbedding, apply_rotary_pos_emb
from nano_megatron.random import set_seed


class TestRotaryEmbedding:
    def setup_method(self) -> None:
        set_seed(42)

    def test_output_shape(self) -> None:
        head_dim = 16
        rope = RotaryEmbedding(head_dim, max_seq_len=128)
        cos, sin = rope(32, device=torch.device("cpu"))
        assert cos.shape == (32, head_dim)
        assert sin.shape == (32, head_dim)

    def test_cache_reuse(self) -> None:
        head_dim = 16
        rope = RotaryEmbedding(head_dim, max_seq_len=128)
        cos1, sin1 = rope(32, device=torch.device("cpu"))
        cos2, sin2 = rope(32, device=torch.device("cpu"))
        assert torch.allclose(cos1, cos2)
        assert torch.allclose(sin1, sin2)

    def test_cache_invalidate_on_different_len(self) -> None:
        head_dim = 16
        rope = RotaryEmbedding(head_dim, max_seq_len=128)
        cos1, _sin1 = rope(16, device=torch.device("cpu"))
        cos2, _sin2 = rope(32, device=torch.device("cpu"))
        assert cos1.shape[0] == 16
        assert cos2.shape[0] == 32


class TestApplyRotaryPosEmb:
    def setup_method(self) -> None:
        set_seed(42)

    def test_shape_preserved(self) -> None:
        B, H, S, D = 2, 4, 32, 16
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        rope = RotaryEmbedding(D, max_seq_len=128)
        cos, sin = rope(S, device=torch.device("cpu"))
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_outputs_not_nan(self) -> None:
        B, H, S, D = 2, 4, 32, 16
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        rope = RotaryEmbedding(D, max_seq_len=128)
        cos, sin = rope(S, device=torch.device("cpu"))
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        assert not torch.isnan(q_out).any()
        assert not torch.isnan(k_out).any()

    def test_gradient_flow(self) -> None:
        B, H, S, D = 1, 2, 16, 8
        q = torch.randn(B, H, S, D, requires_grad=True)
        k = torch.randn(B, H, S, D, requires_grad=True)
        rope = RotaryEmbedding(D, max_seq_len=128)
        cos, sin = rope(S, device=torch.device("cpu"))
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        loss = q_out.sum() + k_out.sum()
        loss.backward()
        assert q.grad is not None
        assert k.grad is not None
