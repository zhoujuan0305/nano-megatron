import torch
from torch import nn

from nano_megatron.reference.optimizer import AdamW


def test_adamw_matches_torch_optim():
    torch.manual_seed(0)
    p1 = nn.Parameter(torch.randn(4, 4))
    p2 = nn.Parameter(p1.detach().clone())
    p1.grad = torch.randn_like(p1)
    p2.grad = p1.grad.detach().clone()
    opt_ref = AdamW([("w", p1)], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    opt_torch = torch.optim.AdamW([p2], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    for _ in range(3):
        opt_ref.step()
        opt_torch.step()
        p1.grad = torch.randn_like(p1)
        p2.grad = p1.grad.detach().clone()
    assert torch.allclose(p1, p2, atol=1e-7, rtol=1e-6)


def test_adamw_hand_step_scalar():
    # single param, t=1, hand formula
    w = nn.Parameter(torch.tensor([2.0]))
    w.grad = torch.tensor([0.5])
    opt = AdamW([("w", w)], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    opt.step()
    m = 0.1 * 0.5  # (1-b1)*g
    v = 0.001 * 0.25
    mhat = m / (1 - 0.9)
    vhat = v / (1 - 0.999)
    expected = 2.0 - 0.1 * (mhat / (vhat**0.5 + 1e-8))
    assert abs(w.item() - expected) < 1e-6


def test_adamw_load_state_dict_casts_to_param_device_fp32():
    w = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
    opt = AdamW([("w", w)], lr=1e-3)
    sd = {
        "w": {
            "exp_avg": torch.tensor([0.1, 0.2], dtype=torch.float64),
            "exp_avg_sq": torch.tensor([0.01, 0.02], dtype=torch.float64),
            "step": 3,
        }
    }
    opt.load_state_dict(sd)
    st = opt.state["w"]
    assert st["exp_avg"].dtype == torch.float32
    assert st["exp_avg_sq"].dtype == torch.float32
    assert st["exp_avg"].device == w.device
    assert st["exp_avg_sq"].device == w.device
    assert st["step"] == 3
    assert torch.allclose(st["exp_avg"], torch.tensor([0.1, 0.2]))
