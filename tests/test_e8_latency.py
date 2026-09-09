from pathlib import Path

import pytest

from experiments.e8_latency import run_latency_sweep

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TQEC_PYTHON = PROJECT_ROOT / ".venv-tqec" / "bin" / "python"

pytestmark = pytest.mark.skipif(not TQEC_PYTHON.exists(), reason=".venv-tqec not set up")


def test_latency_sweep_returns_one_record_per_size_with_nonnegative_duration():
    results = run_latency_sweep(ks=(1, 2), p=0.02, shots=100, devices=("cpu",), seed=0)

    # one gnn + one mwpm record per k, on cpu
    assert len(results) == 2 * 2
    for r in results:
        assert r.experiment == "E8"
        assert r.latency_seconds is not None and r.latency_seconds >= 0.0
        assert r.extra["device"] in ("cpu", "cuda")
    ks_seen = {r.k for r in results}
    assert ks_seen == {1, 2}
