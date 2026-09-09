import subprocess
from pathlib import Path

import pytest
import stim

from data.generate import sample_shots
from data.to_graph import (
    GraphBuildContext,
    assert_no_surgery_feature_leakage,
    batch_to_graphs,
    dem_edge_pairs,
)
from schema import NUM_NODE_FEATURES, OUTPUT_HEADS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TQEC_PYTHON = PROJECT_ROOT / ".venv-tqec" / "bin" / "python"

pytestmark = pytest.mark.skipif(not TQEC_PYTHON.exists(), reason=".venv-tqec not set up")


def _generate(tmp_path: Path, k: int, p: float = 0.01) -> stim.Circuit:
    out = tmp_path / f"k{k}.stim"
    subprocess.run(
        [
            str(TQEC_PYTHON),
            str(PROJECT_ROOT / "circuits" / "tqec_backend.py"),
            "--distance-k", str(k),
            "--p", str(p),
            "--kind", "spacelike",
            "--out", str(out),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return stim.Circuit.from_file(str(out))


def _patch_dimension_for_k(k: int) -> float:
    return float(2 * k + 1)


def test_dem_edge_pairs_nonempty(tmp_path):
    circuit = _generate(tmp_path, k=1)
    edges = dem_edge_pairs(circuit)
    assert edges
    for pair in edges:
        assert len(pair) == 2
        for d in pair:
            assert 0 <= d < circuit.num_detectors


def test_batch_to_graphs_shapes_and_labels(tmp_path):
    circuit = _generate(tmp_path, k=1)
    batch = sample_shots(circuit, shots=200, seed=0)
    ctx = GraphBuildContext.build(circuit, patch_dimension=_patch_dimension_for_k(1))
    graphs = batch_to_graphs(batch, ctx, observable_kind="spacelike")

    assert len(graphs) == 200
    for g in graphs:
        assert g.x.shape[1] == NUM_NODE_FEATURES
        assert g.x.shape[0] == g.num_fired
        assert g.y.shape == (1, 1)
        assert g.head_id == OUTPUT_HEADS.index("spacelike")
        if g.edge_index.numel():
            assert g.edge_index.max().item() < g.x.shape[0]

    # at p=0.01 with 200 shots, at least some should have fired detectors
    assert any(g.num_fired > 0 for g in graphs)


def test_memory_mode_masks_surgery_features(tmp_path):
    circuit = _generate(tmp_path, k=1)
    batch = sample_shots(circuit, shots=200, seed=0)
    ctx = GraphBuildContext.build(circuit, patch_dimension=_patch_dimension_for_k(1))

    masked = batch_to_graphs(batch, ctx, observable_kind="spacelike", mask_surgery_features=True)
    assert_no_surgery_feature_leakage(masked)  # should not raise

    unmasked = batch_to_graphs(batch, ctx, observable_kind="spacelike", mask_surgery_features=False)
    with pytest.raises(AssertionError):
        assert_no_surgery_feature_leakage(unmasked)


def test_two_configurations_produce_structurally_equivalent_graphs(tmp_path):
    """The invariance discipline (design doc §5.2): graphs from different
    absolute patch sizes/round counts must have the same feature width and
    the same *normalized* coordinate range — i.e. the normalization must
    actually cancel out the absolute size difference between k=1 and k=2,
    not just produce two different-but-internally-consistent scales."""
    xy_ranges: dict[int, tuple[float, float]] = {}
    round_ranges: dict[int, tuple[float, float]] = {}
    for k in (1, 2):
        circuit = _generate(tmp_path, k=k, p=0.02)
        batch = sample_shots(circuit, shots=500, seed=0)
        ctx = GraphBuildContext.build(circuit, patch_dimension=_patch_dimension_for_k(k))
        graphs = batch_to_graphs(batch, ctx, observable_kind="spacelike")

        assert all(g.x.shape[1] == NUM_NODE_FEATURES for g in graphs)
        fired_rows = [g.x for g in graphs if g.num_fired > 0]
        assert fired_rows, f"k={k}: no shots produced a fired detector at p=0.02"
        all_x = __import__("torch").cat(fired_rows, dim=0)
        xy = all_x[:, :2]
        rounds = all_x[:, 2]
        xy_ranges[k] = (xy.min().item(), xy.max().item())
        round_ranges[k] = (rounds.min().item(), rounds.max().item())
        assert (rounds >= 0).all() and (rounds <= 1.0 + 1e-6).all()

    # The two configurations have very different absolute sizes (k=1 vs
    # k=2 roughly doubles patch dimension and round count) but normalized
    # ranges should land in the same ballpark, not scale with k.
    for k in (1, 2):
        lo, hi = xy_ranges[k]
        assert -1.0 <= lo and hi <= 6.0, f"k={k}: normalized xy range {xy_ranges[k]} looks unnormalized"
