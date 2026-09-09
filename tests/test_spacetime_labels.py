import subprocess
from pathlib import Path

import pytest
import stim

from data.spacetime_labels import infer_detector_labels

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TQEC_PYTHON = PROJECT_ROOT / ".venv-tqec" / "bin" / "python"

pytestmark = pytest.mark.skipif(not TQEC_PYTHON.exists(), reason=".venv-tqec not set up")


def _generate(tmp_path: Path, k: int) -> stim.Circuit:
    out = tmp_path / f"k{k}.stim"
    subprocess.run(
        [
            str(TQEC_PYTHON),
            str(PROJECT_ROOT / "circuits" / "tqec_backend.py"),
            "--distance-k", str(k),
            "--p", "0.001",
            "--kind", "spacelike",
            "--out", str(out),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return stim.Circuit.from_file(str(out))


@pytest.mark.parametrize("k", [1, 2])
def test_every_detector_gets_a_label_with_valid_values(tmp_path, k):
    circuit = _generate(tmp_path, k)
    labels = infer_detector_labels(circuit)
    assert len(labels) == circuit.num_detectors
    for lbl in labels.values():
        assert lbl.region in ("patch_A", "patch_B", "routing_region")
        assert lbl.phase in ("pre_merge", "merged", "post_split")
        assert lbl.stabilizer_type in ("X", "Z")


@pytest.mark.parametrize("k", [1, 2])
def test_routing_region_detectors_only_occur_during_merged_phase(tmp_path, k):
    circuit = _generate(tmp_path, k)
    labels = infer_detector_labels(circuit)
    routing = [lbl for lbl in labels.values() if lbl.region == "routing_region"]
    assert routing, "expected at least one routing-region detector"
    assert all(lbl.phase == "merged" for lbl in routing)


@pytest.mark.parametrize("k", [1, 2])
def test_all_three_phases_are_present(tmp_path, k):
    circuit = _generate(tmp_path, k)
    labels = infer_detector_labels(circuit)
    phases = {lbl.phase for lbl in labels.values()}
    assert phases == {"pre_merge", "merged", "post_split"}


def test_merge_window_grows_with_distance(tmp_path):
    """Sanity check that the inferred merge window scales with k, rather
    than being some fixed artifact of one specific circuit size."""
    windows = {}
    for k in (1, 2):
        circuit = _generate(tmp_path, k)
        labels = infer_detector_labels(circuit)
        merged_ts = {lbl.t for lbl in labels.values() if lbl.phase == "merged"}
        windows[k] = len(merged_ts)
    assert windows[2] > windows[1]
