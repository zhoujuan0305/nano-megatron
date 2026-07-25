import torch

from nano_megatron.reference.config import ReferenceGPTConfig
from nano_megatron.reference.model import ReferenceGPT
from nano_megatron.reference.loss import shifted_cross_entropy


def _tiny_cfg(**kw):
    base = dict(
        vocab_size=8,
        max_seq_len=8,
        hidden_size=4,
        num_layers=1,
        num_heads=2,
        ffn_hidden_size=8,
        layernorm_eps=1e-5,
        use_bias=True,
        tie_word_embeddings=False,
    )
    base.update(kw)
    return ReferenceGPTConfig(**base)


def test_logits_shape_and_dtype():
    torch.manual_seed(0)
    m = ReferenceGPT(_tiny_cfg())
    ids = torch.randint(0, 8, (2, 5))
    logits = m(ids)
    assert logits.shape == (2, 5, 8)
    assert logits.dtype == torch.float32


def test_seq_len_exceeds_max_raises():
    m = ReferenceGPT(_tiny_cfg(max_seq_len=4))
    ids = torch.zeros(1, 5, dtype=torch.long)
    try:
        m(ids)
    except ValueError as e:
        msg = str(e)
        assert "seq_len" in msg and "5" in msg
        assert "max_seq_len" in msg and "4" in msg
    else:
        raise AssertionError("expected ValueError for seq_len > max_seq_len")


def test_forward_with_activations_keys():
    torch.manual_seed(0)
    m = ReferenceGPT(_tiny_cfg(num_layers=2))
    ids = torch.arange(6).view(1, 6) % 8
    logits, acts = m.forward_with_activations(ids)
    assert "emb" in acts and "final_ln" in acts and "logits" in acts
    assert "layers" in acts and len(acts["layers"]) == 2
    for layer in acts["layers"]:
        for k in ("ln1_out", "attn_out", "resid1", "ln2_out", "mlp_out", "resid2"):
            assert k in layer


def test_shifted_ce_matches_manual_mean():
    logits = torch.randn(1, 3, 4, dtype=torch.float32)
    labels = torch.tensor([[1, 2, 3]])
    loss = shifted_cross_entropy(logits, labels)
    # manual on positions 0,1 predicting labels 2,3 from logits[:,0:2]
    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    tgt = labels[:, 1:]
    manual = -log_probs[0, 0, tgt[0, 0]] - log_probs[0, 1, tgt[0, 1]]
    manual = manual / 2
    assert torch.allclose(loss, manual)


def test_backward_grad_matches_autograd_grad():
    torch.manual_seed(1)
    m = ReferenceGPT(_tiny_cfg())
    ids = torch.randint(0, 8, (1, 4))
    logits = m(ids)
    loss = shifted_cross_entropy(logits, ids)
    loss.backward()
    g1 = {n: p.grad.detach().clone() for n, p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    logits2 = m(ids)
    loss2 = shifted_cross_entropy(logits2, ids)
    grads = torch.autograd.grad(loss2, list(m.parameters()))
    for (n, p), g in zip(m.named_parameters(), grads):
        assert torch.allclose(g1[n], g, atol=0, rtol=0)
