from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _run_torchrun(script_path: Path, nproc: int, timeout: int = 120) -> tuple[int, str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc}",
            str(script_path),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout + result.stderr


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


@pytest.mark.distributed
@pytest.mark.gpu
class TestDPDemo:
    def test_dp_demo_runs(self) -> None:
        script = SCRIPTS_DIR / "run_dp_demo.py"
        if not script.exists():
            pytest.skip(f"Script not found: {script}")
        nproc = 2
        returncode, output = _run_torchrun(script, nproc)
        assert returncode == 0, f"DP demo failed:\n{output}"
        assert "training complete" in output
