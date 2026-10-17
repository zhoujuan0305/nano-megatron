from types import SimpleNamespace

import torch
import scripts.train_gpt as driver


def test_sp_heldout_uses_global_positions_on_nonfirst_stage(monkeypatch):
    class Stage(torch.nn.Module):
        is_first_stage = False
        is_last_stage = True
        config = SimpleNamespace(hidden_size=8)

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0

        def forward(self, hidden, *, positions):
            assert hidden.shape == (1, 16, 8)
            assert torch.equal(positions, torch.arange(32))
            assert not torch.is_grad_enabled()
            self.calls += 1
            return hidden

        def shifted_cross_entropy(self, output, labels):
            assert labels.shape == (1, 32)
            return torch.tensor(2.0)

    sampler = SimpleNamespace(sequence_length=32, batch=lambda index, size: (torch.zeros(size, 32, dtype=torch.long), {}))
    stage = Stage()
    ctx = SimpleNamespace(tensor_parallel_size=2, sequence_parallel=True, tensor_parallel_rank=0)
    monkeypatch.setattr(driver, "recv_forward", lambda ctx, **kwargs: torch.zeros(kwargs["shape"]))
    loss = driver.evaluate(stage, ctx, sampler, batch_size=4, microbatches=4, batches=2, device=torch.device("cpu"))
    assert loss == 2.0 and stage.calls == 8 and stage.training
