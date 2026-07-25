from pathlib import Path

import torch

from nano_megatron.reference.config import ReferenceGPTConfig
from nano_megatron.reference.model import ReferenceGPT
from nano_megatron.reference.optimizer import AdamW
from nano_megatron.reference.train_step import seed_all, reference_train_loop

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_TINY_TRAJ_T3 = _FIXTURE_DIR / "tiny_traj_t3.pt"


def _tiny_cfg() -> ReferenceGPTConfig:
    return ReferenceGPTConfig(
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


def _run_tiny_traj_t3():
    """Same seed/config/batch as fixtures/tiny_traj_t3.pt."""
    seed_all(123)
    m = ReferenceGPT(_tiny_cfg())
    opt = AdamW(list(m.named_parameters()), lr=1e-3, weight_decay=0.01)
    batches = [torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])]
    return reference_train_loop(m, opt, batches * 3, steps=3, capture_level="full")


def test_multistep_trajectory_deterministic():
    t1 = _run_tiny_traj_t3()
    t2 = _run_tiny_traj_t3()
    assert len(t1) == 3
    for a, b in zip(t1, t2):
        assert torch.equal(a.loss, b.loss)
        assert torch.equal(a.logits, b.logits)
        for k in a.params:
            assert torch.equal(a.params[k], b.params[k])
            assert torch.equal(a.grads[k], b.grads[k])
        for name in a.optimizer_state:
            for field in ("exp_avg", "exp_avg_sq"):
                assert torch.equal(
                    a.optimizer_state[name][field],
                    b.optimizer_state[name][field],
                )


def test_params_change_after_steps():
    seed_all(0)
    m = ReferenceGPT(_tiny_cfg())
    init_params = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = AdamW(list(m.named_parameters()), lr=1e-2, weight_decay=0.0)
    batches = [torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])]
    traj = reference_train_loop(m, opt, batches * 3, steps=3, capture_level="minimal")
    assert len(traj) == 3
    final = traj[-1].params
    changed = any(not torch.equal(final[n], init_params[n]) for n in init_params)
    assert changed


def test_golden_trajectory_fixture_strict():
    """Regression vs committed fixture (CPU FP32, atol=0/rtol=0).

    Regenerate if numerics intentionally change (from repo root, package installed):

      python -c "
      import torch
      from tests.unit.reference.test_train_trajectory import _run_tiny_traj_t3, _TINY_TRAJ_T3
      traj = _run_tiny_traj_t3()
      torch.save({
          'meta': {'seed': 123, 'steps': 3, 'lr': 1e-3, 'weight_decay': 0.01,
                   'batch': [[0,1,2,3,4,5,6,7]]},
          'steps': [{'step': int(r.step),
                     'loss': r.loss.detach().cpu().float().clone(),
                     'logits': r.logits.detach().cpu().float().clone()}
                    for r in traj],
      }, _TINY_TRAJ_T3)
      "
    """
    assert _TINY_TRAJ_T3.is_file(), f"missing fixture {_TINY_TRAJ_T3}"
    fixture = torch.load(_TINY_TRAJ_T3, map_location="cpu", weights_only=True)
    traj = _run_tiny_traj_t3()
    assert len(traj) == len(fixture["steps"]) == 3
    for live, gold in zip(traj, fixture["steps"]):
        assert int(live.step) == int(gold["step"])
        assert torch.allclose(
            live.loss.detach().cpu().float(),
            gold["loss"].float(),
            atol=0,
            rtol=0,
        )
        assert torch.allclose(
            live.logits.detach().cpu().float(),
            gold["logits"].float(),
            atol=0,
            rtol=0,
        )
