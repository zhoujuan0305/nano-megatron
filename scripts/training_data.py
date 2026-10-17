"""Stateless, document-bounded token windows from pre-exported public NPY data."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


class DocumentTokenSampler:
    """Uniform over all legal within-document starts; never cross a document/split."""

    def __init__(self, directory: str, split: str, sequence_length: int, seed: int) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        root = Path(directory)
        raw = (root / "dataset.json").read_bytes()
        self.dataset_id = hashlib.sha256(raw).hexdigest()
        info = json.loads(raw)["splits"][split]
        for kind in ("tokens", "offsets"):
            path = root / info[kind + "_file"]
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024**2), b""):
                    digest.update(block)
            if digest.hexdigest() != info[kind + "_sha256"]:
                raise ValueError(f"{split} {kind} checksum mismatch")
        self.tokens = np.load(root / info["tokens_file"], mmap_mode="r")
        self.offsets = np.load(root / info["offsets_file"], mmap_mode="r")
        if (self.tokens.ndim != 1 or self.offsets.ndim != 1 or len(self.offsets) < 2
                or not np.issubdtype(self.tokens.dtype, np.integer)
                or not np.issubdtype(self.offsets.dtype, np.integer)
                or self.offsets[0] != 0 or self.offsets[-1] != len(self.tokens)):
            raise ValueError("invalid token/document arrays")
        if np.any(np.diff(self.offsets) < 0):
            raise ValueError("document offsets must be monotonic")
        self.sequence_length, self.seed, self.split = sequence_length, seed, split
        legal = np.maximum(np.diff(self.offsets) - sequence_length + 1, 0)
        self.cumulative = np.cumsum(legal, dtype=np.int64)
        self.total_starts = int(self.cumulative[-1])
        if self.total_starts == 0:
            raise ValueError("no document is long enough for a complete window")

    def batch(self, update: int, batch_size: int) -> tuple[torch.Tensor, dict]:
        if update < 0 or batch_size < 1:
            raise ValueError("invalid update or batch size")
        rng = random.Random(f"document-window-v1:{self.seed}:{self.split}:{update}")
        starts, documents, data = [], [], []
        for _ in range(batch_size):
            draw = rng.randrange(self.total_starts)
            document = int(np.searchsorted(self.cumulative, draw, side="right"))
            previous = int(self.cumulative[document - 1]) if document else 0
            start = int(self.offsets[document]) + draw - previous
            assert start + self.sequence_length <= self.offsets[document + 1]
            starts.append(start)
            documents.append(document)
            data.append(np.array(self.tokens[start:start + self.sequence_length], dtype=np.int64))
        identity = dict(algorithm="document-window-v1", dataset_id=self.dataset_id, split=self.split,
                        seed=self.seed, update=update, starts=starts, documents=documents)
        identity["batch_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return torch.from_numpy(np.stack(data)), identity
