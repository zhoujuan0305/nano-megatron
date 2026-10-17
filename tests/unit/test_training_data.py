import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scripts.training_data import DocumentTokenSampler
from scripts.train_gpt import learning_rate, advance_data_hash, model_config_dict
from nano_megatron.reference import ReferenceGPTConfig


def _data(root: Path):
    manifest = {"splits": {}}
    for split in ["train", "validation"]:
        tokens = np.arange(100, dtype=np.uint16) + (0 if split == "train" else 1000)
        arrays = {"tokens": tokens, "offsets": np.array([0, 10, 40, 100], dtype=np.int64)}
        info = {}
        for kind, value in arrays.items():
            path = root / f"{split}.{kind}.npy"
            np.save(path, value)
            info[kind + "_file"] = path.name
            info[kind + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["splits"][split] = info
    (root / "dataset.json").write_text(json.dumps(manifest))


def test_document_windows_are_dynamic_repeatable_and_bounded(tmp_path):
    _data(tmp_path)
    sampler = DocumentTokenSampler(str(tmp_path), "train", 16, 42)
    first, meta = sampler.batch(0, 16)
    again, same = sampler.batch(0, 16)
    later, changed = sampler.batch(1, 16)
    assert torch.equal(first, again) and meta == same
    assert not torch.equal(first, later) and meta["batch_id"] != changed["batch_id"]
    assert 0 not in meta["documents"]  # Ten-token document is too short.
    for start, document in zip(meta["starts"], meta["documents"]):
        assert sampler.offsets[document] <= start
        assert start + 16 <= sampler.offsets[document + 1]
    # A fresh sampler at restored update 1 produces the same next batch.
    restored = DocumentTokenSampler(str(tmp_path), "train", 16, 42)
    assert torch.equal(restored.batch(1, 16)[0], later)
    valid = DocumentTokenSampler(str(tmp_path), "validation", 16, 42)
    assert int(valid.batch(0, 4)[0].min()) >= 1000
    assert valid.batch(0, 4)[1]["batch_id"] != sampler.batch(0, 4)[1]["batch_id"]


def test_dataset_checksum_and_too_short_documents_rejected(tmp_path):
    _data(tmp_path)
    with pytest.raises(ValueError, match="long enough"):
        DocumentTokenSampler(str(tmp_path), "train", 101, 42)
    path = tmp_path / "train.tokens.npy"
    path.write_bytes(path.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        DocumentTokenSampler(str(tmp_path), "train", 16, 42)


def test_model_config_is_json_serializable_including_torch_dtypes():
    config = ReferenceGPTConfig(vocab_size=51200, max_seq_len=32, hidden_size=64,
                                num_layers=2, num_heads=4, ffn_hidden_size=176)
    canonical = model_config_dict(config)
    assert json.loads(json.dumps(canonical)) == canonical
    assert any(value == "torch.float32" for value in canonical.values())


def test_schedule_and_data_trace_are_resumable():
    rates = [learning_rate(i, 10, 2, 1e-3, 1e-4) for i in range(1, 11)]
    assert rates[:2] == [5e-4, 1e-3]
    assert rates[-1] == pytest.approx(1e-4)
    assert rates[2:] == sorted(rates[2:], reverse=True)
    initial = "00" * 32
    middle = advance_data_hash(initial, "11" * 32)
    final = advance_data_hash(middle, "22" * 32)
    assert final == advance_data_hash(middle, "22" * 32)
    assert final != advance_data_hash(initial, "22" * 32)
    with pytest.raises(ValueError):
        learning_rate(1, 10, 10, 1e-3, 1e-4)
