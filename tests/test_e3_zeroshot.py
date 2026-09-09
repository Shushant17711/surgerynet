import subprocess
from pathlib import Path

import pytest

from experiments.e3_zeroshot import evaluate_fixed_input_infeasibility, evaluate_gnn_zero_shot
from models.gnn_decoder import GNNDecoder

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


def test_gnn_zero_shot_evaluation_produces_valid_result(tmp_path):
    circuit_path = _generate_surgery_circuit(tmp_path)
    model = GNNDecoder(num_layers=4, hidden_dim=32)  # untrained -- smoke test on wiring, not accuracy
    result = evaluate_gnn_zero_shot(
        model, str(circuit_path), patch_dimension=3.0, observable_kind="spacelike", shots=200, seed=0
    )
    assert result.experiment == "E3" and result.decoder == "gnn"
    assert result.status == "ok"
    assert 0.0 <= result.logical_error_rate <= 1.0
    assert result.extra["zero_shot"] is True


def test_mlp_and_cnn_report_infeasible_on_surgery_shaped_input():
    mlp_result, cnn_result = evaluate_fixed_input_infeasibility(
        memory_num_detectors=24,
        memory_grid_shape=(3, 4, 4),
        surgery_num_detectors=60,
        surgery_grid_shape=(3, 4, 9),
        observable_kind="spacelike",
    )
    assert mlp_result.status == "infeasible"
    assert cnn_result.status == "infeasible"
    assert mlp_result.decoder == "mlp" and cnn_result.decoder == "cnn"
