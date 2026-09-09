from pathlib import Path

import pytest

from experiments.e6_config_randomization import run

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TQEC_PYTHON = PROJECT_ROOT / ".venv-tqec" / "bin" / "python"

pytestmark = pytest.mark.skipif(not TQEC_PYTHON.exists(), reason=".venv-tqec not set up")


def test_e6_runs_end_to_end_with_disjoint_configs_and_valid_result():
    result = run(
        distances=(3, 5),
        p=0.02,
        shots_per_config=100,
        epochs=1,
        batch_size=32,
        train_fraction=0.5,
        seed=0,
    )
    assert result.experiment == "E6" and result.decoder == "gnn"
    assert 0.0 <= result.logical_error_rate <= 1.0
    assert set(result.extra["train_distances"]).isdisjoint(result.extra["held_out_distances"])
    assert "generalization_gap" in result.extra
