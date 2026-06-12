from __future__ import annotations

from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(
        self, vocab_size: int, seq_length: int, num_samples: int = 1024, seed: int = 42
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.seed = seed

        import numpy as np

        rng = np.random.default_rng(seed)
        self._data = rng.integers(0, vocab_size, size=(num_samples, seq_length), dtype=np.int64)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple:
        import torch

        token_ids = self._data[idx]
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels = input_ids.clone()
        return input_ids, labels
