from pathlib import Path

import pytest

from experiments.e9_ablations import BASELINE_ABLATION_CONFIG, ablation_grid, leave_one_out_variants, run_one

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TQEC_PYTHON = PROJECT_ROOT / ".venv-tqec" / "bin" / "python"


def test_ablation_grid_has_no_duplicate_combinations():
    grid = ablation_grid()
    assert len(grid) == len(set(grid))
    assert grid  # nonempty


def test_ablation_grid_excludes_the_edgeless_combination():
    grid = ablation_grid()
    assert not any(not c.use_dem_edges and not c.use_radius_edges for c in grid)


def test_leave_one_out_variants_each_differ_from_baseline_by_exactly_one_field():
    variants = leave_one_out_variants()
    assert variants["baseline (all components on)"] == BASELINE_ABLATION_CONFIG
    for name, config in variants.items():
        if name == "baseline (all components on)":
            continue
        diffs = sum(
            1
            for field in (
                "mask_surgery_features", "use_dem_edges", "use_radius_edges", "num_layers", "use_norm",
                "use_edge_features",
            )
            if getattr(config, field) != getattr(BASELINE_ABLATION_CONFIG, field)
        )
        assert diffs == 1, f"{name} differs from baseline in {diffs} fields, expected exactly 1"


@pytest.mark.skipif(not TQEC_PYTHON.exists(), reason=".venv-tqec not set up")
def test_run_one_executes_a_single_ablation_variant_end_to_end():
    grid = ablation_grid()
    result = run_one(grid[0], k=1, p=0.02, train_shots=100, test_shots=100, epochs=1, batch_size=32, seed=0)
    assert result.experiment == "E9" and result.decoder == "gnn"
    assert 0.0 <= result.logical_error_rate <= 1.0
    assert result.extra["num_layers"] == grid[0].num_layers


@pytest.mark.skipif(not TQEC_PYTHON.exists(), reason=".venv-tqec not set up")
def test_run_one_records_variant_name():
    result = run_one(
        BASELINE_ABLATION_CONFIG, k=1, p=0.02, train_shots=100, test_shots=100, epochs=1, batch_size=32,
        seed=0, variant_name="baseline (all components on)",
    )
    assert result.extra["variant_name"] == "baseline (all components on)"
