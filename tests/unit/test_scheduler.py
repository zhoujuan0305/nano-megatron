from torch import optim

from nano_megatron.training.scheduler import CosineWarmupScheduler


class TestCosineWarmupScheduler:
    def test_warmup_lr(self) -> None:
        model = __import__("torch").nn.Linear(10, 10)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10, max_steps=100)

        lrs = []
        for _ in range(10):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1] or i == 0, "LR should increase during warmup"

    def test_state_dict_round_trip(self) -> None:
        import torch

        model = torch.nn.Linear(10, 10)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps=5, max_steps=50)

        for _ in range(10):
            scheduler.step()

        state = scheduler.state_dict()
        scheduler2 = CosineWarmupScheduler(optimizer, warmup_steps=5, max_steps=50)
        scheduler2.load_state_dict(state)
        assert scheduler2._step_count == 10
