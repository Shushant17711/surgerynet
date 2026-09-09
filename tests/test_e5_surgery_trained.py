import subprocess
from pathlib import Path

import pytest

from experiments.e5_surgery_trained import run

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TQEC_PYTHON = PROJECT_ROOT / ".venv-tqec" / "bin" / "python"

pytestmark = pytest.mark.skipif(not TQEC_PYTHON.exists(), reason=".venv-tqec not set up")


def _generate_surgery_circuit(tmp_path: Path, k: int = 1) -> Path:
    out = tmp_path / f"k{k}_spacelike.stim"
    subprocess.run(
        [
            str(TQEC_PYTHON),
            str(PROJECT_ROOT / "circuits" / "tqec_backend.py"),
            "--distance-k", str(k),
            "--p", "0.01",
            "--kind", "spacelike",
            "--out", str(out),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return out


def test_e5_trains_and_evaluates_end_to_end(tmp_path):
    circuit_path = _generate_surgery_circuit(tmp_path)
    result, model = run(
        str(circuit_path), patch_dimension=3.0, observable_kind="spacelike",
        train_shots=200, test_shots=200, epochs=1, batch_size=64, seed=0,
    )
    assert result.experiment == "E5" and result.decoder == "gnn"
    assert 0.0 <= result.logical_error_rate <= 1.0
    assert result.extra["surgery_trained"] is True
