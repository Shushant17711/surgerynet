import subprocess
from pathlib import Path

import pytest

from data.generate import sample_shots
from data.to_graph import GraphBuildContext, batch_to_graphs
from experiments.e4_localization import compute_localization_heatmap, render_heatmap
from circuits.lattice_surgery import load_generated_circuit
from models.gnn_decoder import GNNDecoder
from schema import REGION_LABELS

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
            "--p", "0.02",
            "--kind", "spacelike",
            "--out", str(out),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return out


def test_heatmap_shape_and_value_range(tmp_path):
    circuit_path = _generate_surgery_circuit(tmp_path)
    circuit = load_generated_circuit(str(circuit_path))
    ctx = GraphBuildContext.build(circuit, patch_dimension=3.0)
    batch = sample_shots(circuit, shots=500, seed=0)
    graphs = batch_to_graphs(batch, ctx, observable_kind="spacelike")

    model = GNNDecoder(num_layers=4, hidden_dim=32)
    heatmap, rounds = compute_localization_heatmap(model, ctx, batch, graphs)

    num_rounds_expected = len({int(lbl.t) for lbl in ctx.labels.values()})
    assert heatmap.shape == (num_rounds_expected, len(REGION_LABELS))
    assert len(rounds) == num_rounds_expected
    valid = heatmap[~__import__("numpy").isnan(heatmap)]
    assert ((valid >= 0.0) & (valid <= 1.0)).all()


def test_render_heatmap_writes_a_file(tmp_path):
    circuit_path = _generate_surgery_circuit(tmp_path)
    circuit = load_generated_circuit(str(circuit_path))
    ctx = GraphBuildContext.build(circuit, patch_dimension=3.0)
    batch = sample_shots(circuit, shots=200, seed=0)
    graphs = batch_to_graphs(batch, ctx, observable_kind="spacelike")
    model = GNNDecoder(num_layers=4, hidden_dim=32)
    heatmap, _ = compute_localization_heatmap(model, ctx, batch, graphs)

    out_path = tmp_path / "heatmap.png"
    render_heatmap(heatmap, str(out_path))
    assert out_path.exists() and out_path.stat().st_size > 0
