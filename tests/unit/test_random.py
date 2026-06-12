import torch

from nano_megatron.random import get_torch_generator, set_seed


class TestSetSeed:
    def test_reproducibility(self) -> None:
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self) -> None:
        set_seed(42)
        a = torch.randn(10)
        set_seed(123)
        b = torch.randn(10)
        assert not torch.equal(a, b)

    def test_torch_generator(self) -> None:
        gen = get_torch_generator(42, torch.device("cpu"))
        a = torch.randn(10, generator=gen)
        gen2 = get_torch_generator(42, torch.device("cpu"))
        b = torch.randn(10, generator=gen2)
        assert torch.equal(a, b)
