"""Exercises circuits/tqec_backend.py through the actual cross-venv path:
shell out to .venv-tqec's Python (the way real data generation will use
it), then load the resulting .stim file with the main venv's stim install.
Skipped if .venv-tqec hasn't been set up (see circuits/tqec_backend.py's
module docstring for setup instructions) — it isn't a hard project
dependency, just the preferred backend for anyone who has it.
"""

import subprocess
import sys
from pathlib import Path

import pymatching
import pytest
import stim

from circuits.lattice_surgery import load_generated_circuit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TQEC_PYTHON = PROJECT_ROOT / ".venv-tqec" / "bin" / "python"

pytestmark = pytest.mark.skipif(
    not TQEC_PYTHON.exists(), reason=".venv-tqec not set up (see tqec_backend.py docstring)"
)


def _generate(tmp_path: Path, kind: str, k: int, p: float) -> Path:
    out = tmp_path / f"{kind}_k{k}.stim"
    subprocess.run(
        [
            str(TQEC_PYTHON),
            str(PROJECT_ROOT / "circuits" / "tqec_backend.py"),
            "--distance-k", str(k),
            "--p", str(p),
            "--kind", kind,
            "--out", str(out),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return out


@pytest.mark.parametrize("kind", ["spacelike", "timelike"])
def test_generated_circuit_loads_and_has_deterministic_detectors(tmp_path, kind):
    path = _generate(tmp_path, kind, k=1, p=0.005)
    circuit = load_generated_circuit(path)
    dem = circuit.detector_error_model(decompose_errors=True)
    assert len(dem) > 0


@pytest.mark.parametrize("kind", ["spacelike", "timelike"])
def test_decoded_logical_error_rate_improves_with_distance(tmp_path, kind):
    """The real validation gate (design doc §5.1, week-5 gate): a properly
    *decoded* (not raw-physical-flip) logical error rate must go down as
    distance goes up, for p below threshold. p=0.00259 and k=1 vs k=2 were
    both picked from empirically running tqec's own binary_search_threshold
    on this block graph shape."""
    rates = {}
    for k in (1, 2):
        path = _generate(tmp_path, kind, k=k, p=0.00259)
        circuit = load_generated_circuit(path)
        dem = circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        sampler = circuit.compile_detector_sampler()
        dets, obs = sampler.sample(shots=20000, separate_observables=True)
        preds = matcher.decode_batch(dets)
        rates[k] = (preds[:, 0] != obs[:, 0]).mean()
    assert rates[2] < rates[1]
