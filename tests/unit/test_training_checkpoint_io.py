import json

import pytest
import torch
from scripts.benchmark_optimizer import make_benchmark_adamw
from scripts.train_gpt import save_checkpoint, restore_checkpoint


def _checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: torch.zeros(8, dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: None)
    module = torch.nn.Linear(3, 2)
    optimizer = make_benchmark_adamw(module, learning_rate=0.01)
    for param in module.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()
    config = {"model": "small", "scheduler": {"steps": 10}}
    save_checkpoint(tmp_path / "checkpoint", stage=module, optimizer=optimizer, semantic=config,
                    step=1, data_hash="12" * 32, rank=0, local_rank=0, device=torch.device("cpu"))
    return module, optimizer, config


def test_checkpoint_file_roundtrip_cpu_with_cuda_rng_mocked(tmp_path, monkeypatch):
    module, optimizer, config = _checkpoint(tmp_path, monkeypatch)
    restored = torch.nn.Linear(3, 2)
    other = make_benchmark_adamw(restored, learning_rate=0.9)
    step, trace = restore_checkpoint(tmp_path / "checkpoint", stage=restored, optimizer=other,
                                    semantic=config, rank=0, device=torch.device("cpu"))
    assert step == 1 and trace == "12" * 32
    assert optimizer.step_count == other.step_count == 1
    assert all(torch.equal(a, b) for a, b in zip(module.parameters(), restored.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(optimizer._masters, other._masters))


@pytest.mark.parametrize("field", ["bytes", "step", "data_hash"])
def test_checkpoint_rejects_inconsistent_shard_index(tmp_path, monkeypatch, field):
    module, optimizer, config = _checkpoint(tmp_path, monkeypatch)
    path = tmp_path / "checkpoint/rank0000.json"
    info = json.loads(path.read_text())
    info[field] = "34" * 32 if field == "data_hash" else info[field] + 1
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="checkpoint shard"):
        restore_checkpoint(path.parent, stage=module, optimizer=optimizer, semantic=config,
                           rank=0, device=torch.device("cpu"))
