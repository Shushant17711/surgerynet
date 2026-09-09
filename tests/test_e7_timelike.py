from pathlib import Path

import pytest

from experiments.e7_timelike import run

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TQEC_PYTHON = PROJECT_ROOT / ".venv-tqec" / "bin" / "python"

pytestmark = pytest.mark.skipif(not TQEC_PYTHON.exists(), reason=".venv-tqec not set up")


def test_e7_runs_and_reports_both_observable_kinds_for_both_decoders():
    results = run(k=1, p=0.02, train_shots=150, test_shots=300, epochs=1, batch_size=32, seed=0)
    kinds_decoders = {(r.observable_kind, r.decoder) for r in results}
    assert kinds_decoders == {
        ("spacelike", "gnn"),
        ("spacelike", "mwpm"),
        ("timelike", "gnn"),
        ("timelike", "mwpm"),
    }
    for r in results:
        assert r.experiment == "E7"
        assert 0.0 <= r.logical_error_rate <= 1.0
