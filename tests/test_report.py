from experiments.common import ExperimentResult, append_result
from experiments.report import _COLUMNS, aggregate, group_by_config, render_main_table, write_main_table


def test_group_by_config_groups_only_by_seed_difference():
    rows = [
        ExperimentResult(experiment="E2", decoder="gnn", observable_kind="spacelike", distance=3, p=0.005, seed=0, logical_error_rate=0.05),
        ExperimentResult(experiment="E2", decoder="gnn", observable_kind="spacelike", distance=3, p=0.005, seed=1, logical_error_rate=0.06),
        ExperimentResult(experiment="E2", decoder="gnn", observable_kind="spacelike", distance=3, p=0.005, seed=2, logical_error_rate=0.04),
        ExperimentResult(experiment="E2", decoder="mwpm", observable_kind="spacelike", distance=3, p=0.005, seed=0, logical_error_rate=0.03),
    ]
    groups = group_by_config(rows)
    assert len(groups) == 2  # gnn group and mwpm group
    gnn_group = [g for g in groups.values() if len(g) == 3][0]
    assert {r.seed for r in gnn_group} == {0, 1, 2}


def test_render_shows_mean_and_std_with_three_or_more_seeds():
    rows = [
        ExperimentResult(experiment="E2", decoder="gnn", observable_kind="spacelike", distance=3, p=0.005, seed=s, logical_error_rate=r)
        for s, r in enumerate([0.10, 0.12, 0.08])
    ]
    table = render_main_table(rows)
    assert "±" in table
    assert "(n=3)" in table
    assert "INSUFFICIENT" not in table


def test_render_flags_insufficient_seeds_below_threshold():
    rows = [
        ExperimentResult(experiment="E2", decoder="gnn", observable_kind="spacelike", distance=3, p=0.005, seed=0, logical_error_rate=0.10),
        ExperimentResult(experiment="E2", decoder="gnn", observable_kind="spacelike", distance=3, p=0.005, seed=1, logical_error_rate=0.12),
    ]
    table = render_main_table(rows)
    assert "INSUFFICIENT SEEDS" in table
    assert "(n=2" in table


def test_single_row_with_no_seed_is_flagged_insufficient():
    rows = [ExperimentResult(experiment="E2", decoder="mwpm", observable_kind="spacelike", logical_error_rate=0.02)]
    table = render_main_table(rows)
    assert "INSUFFICIENT SEEDS" in table


def test_infeasible_rows_render_as_na_regardless_of_seed_count():
    rows = [
        ExperimentResult(experiment="E3", decoder="mlp", observable_kind="spacelike", status="infeasible", seed=0),
        ExperimentResult(experiment="E3", decoder="mlp", observable_kind="spacelike", status="infeasible", seed=1),
    ]
    table = render_main_table(rows)
    lines = [l for l in table.splitlines() if "mlp" in l]
    assert len(lines) == 1
    assert "N/A" in lines[0]
    assert "±" not in lines[0]


def test_aggregate_reads_all_experiment_files(tmp_path):
    append_result(
        tmp_path / "e2_memory_baseline.jsonl",
        ExperimentResult(experiment="E2", decoder="gnn", observable_kind="spacelike", logical_error_rate=0.05),
    )
    rows = aggregate(tmp_path)
    assert len(rows) == 1


def test_write_main_table_creates_file(tmp_path):
    ok_row = ExperimentResult(experiment="E2", decoder="mwpm", observable_kind="spacelike", logical_error_rate=0.01)
    append_result(tmp_path / "e2_memory_baseline.jsonl", ok_row)

    out_path = write_main_table(results_dir=tmp_path, out_path=tmp_path / "main_table.md")
    assert out_path.exists()
    content = out_path.read_text()
    assert "mwpm" in content
    assert list(_COLUMNS) == [
        "experiment", "decoder", "observable_kind", "status", "distance", "k", "p",
        "logical_error_rate", "latency_seconds", "n_seeds",
    ]
