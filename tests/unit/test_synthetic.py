import torch

from nano_megatron.data.synthetic import SyntheticDataset
from nano_megatron.random import set_seed


class TestSyntheticDataset:
    def setup_method(self) -> None:
        set_seed(42)

    def test_length(self) -> None:
        ds = SyntheticDataset(vocab_size=1024, seq_length=32, num_samples=100)
        assert len(ds) == 100

    def test_getitem_shape(self) -> None:
        ds = SyntheticDataset(vocab_size=1024, seq_length=32, num_samples=100)
        input_ids, labels = ds[0]
        assert input_ids.shape == (32,)
        assert labels.shape == (32,)

    def test_labels_equal_input(self) -> None:
        ds = SyntheticDataset(vocab_size=1024, seq_length=32, num_samples=100)
        input_ids, labels = ds[0]
        assert torch.equal(input_ids, labels)

    def test_vocab_range(self) -> None:
        ds = SyntheticDataset(vocab_size=1024, seq_length=32, num_samples=100)
        input_ids, _ = ds[0]
        assert input_ids.min() >= 0
        assert input_ids.max() < 1024

    def test_deterministic_with_seed(self) -> None:
        ds1 = SyntheticDataset(vocab_size=1024, seq_length=32, num_samples=50, seed=42)
        ds2 = SyntheticDataset(vocab_size=1024, seq_length=32, num_samples=50, seed=42)
        assert torch.equal(ds1[0][0], ds2[0][0])
