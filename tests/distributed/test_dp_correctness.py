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
from nano_megatron.data_parallel.grad_sync import average_gradients, broadcast_parameters
from nano_megatron.distributed.initialize import init_distributed, destroy_distributed
from nano_megatron.distributed.parallel_state import (
    create_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_size,
)
from nano_megatron.model.gpt import GPTModel
from nano_megatron.random import set_seed

def main():
    init_distributed()
    dp_group = create_data_parallel_group()
    dp_rank = get_data_parallel_rank()
    dp_size = get_data_parallel_size()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{dp_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    config = GPTConfig(hidden_size=64, num_layers=2, num_attention_heads=4, seq_length=16, vocab_size=128)

    set_seed(42)
    model = GPTModel(config).to(device)
    broadcast_parameters(model, src=0, group=dp_group)

    params_rank0 = {n: p.data.clone() for n, p in model.named_parameters()}

    dataset = SyntheticDataset(vocab_size=config.vocab_size, seq_length=config.seq_length, num_samples=32, seed=42)
    input_ids, labels = dataset[0]
    input_ids = input_ids.unsqueeze(0).to(device)
    labels = labels.unsqueeze(0).to(device)

    model.train()
    _, loss = model(input_ids, labels=labels)
    loss.backward()

    average_gradients(model, group=dp_group)

    for n, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"rank {dp_rank}: NaN gradient for {n}"
            assert not torch.isinf(p.grad).any(), f"rank {dp_rank}: Inf gradient for {n}"

    dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=dp_group)
    avg_loss = loss.item() / dp_size

    print(f"CHECK_OK rank={dp_rank} dp_size={dp_size} loss={avg_loss:.6f}", flush=True)

    destroy_distributed()

if __name__ == "__main__":
    main()
""")


@pytest.mark.distributed
@pytest.mark.gpu
class TestDPCorrectness:
    def test_dp_gradient_sync(self) -> None:
        tmp_script = Path("/tmp/nano_megatron_dp_test_helper.py")
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
        assert result.returncode == 0, f"DP test failed:\n{combined}"
        assert "CHECK_OK" in combined
