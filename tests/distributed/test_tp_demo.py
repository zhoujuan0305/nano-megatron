from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

HELPER_SCRIPT = textwrap.dedent("""\
import sys
import torch
import torch.distributed as dist

from nano_megatron.config import GPTConfig
from nano_megatron.data.synthetic import SyntheticDataset
from nano_megatron.distributed.initialize import init_distributed, destroy_distributed
from nano_megatron.distributed.parallel_state import create_tensor_parallel_group, get_tensor_parallel_rank, get_tensor_parallel_size
from nano_megatron.model.tp_gpt import TPGPTModel
from nano_megatron.random import set_seed

def main():
    init_distributed()
    tp_group = create_tensor_parallel_group(tp_size=2)
    tp_rank = get_tensor_parallel_rank()
    tp_size = get_tensor_parallel_size()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{dist.get_rank()}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    config = GPTConfig(hidden_size=64, num_layers=2, num_attention_heads=4, seq_length=16, vocab_size=128)

    set_seed(42)
    model = TPGPTModel(config).to(device)

    dataset = SyntheticDataset(vocab_size=config.vocab_size, seq_length=config.seq_length, num_samples=32, seed=42)
    input_ids, labels = dataset[0]
    input_ids = input_ids.unsqueeze(0).to(device)
    labels = labels.unsqueeze(0).to(device)

    model.train()
    _, loss = model(input_ids, labels=labels)
    loss.backward()

    for n, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"rank {tp_rank}: NaN gradient for {n}"

    print(f"CHECK_OK rank={tp_rank} tp_size={tp_size} loss={loss.item():.6f}", flush=True)

    destroy_distributed()

if __name__ == "__main__":
    main()
""")


@pytest.mark.distributed
@pytest.mark.gpu
class TestTPDemo:
    def test_tp_demo_runs(self) -> None:
        script = Path(__file__).resolve().parents[2] / "scripts" / "run_tp_demo.py"
        if not script.exists():
            pytest.skip(f"Script not found: {script}")
        nproc = 2
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={nproc}",
                str(script),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0, f"TP demo failed:\n{combined}"
        assert "training complete" in combined

    def test_tp_correctness_forward_backward(self) -> None:
        tmp_script = Path("/tmp/nano_megatron_tp_test_helper.py")
        tmp_script.write_text(HELPER_SCRIPT)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(tmp_script),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0, f"TP correctness test failed:\n{combined}"
        assert "CHECK_OK" in combined
