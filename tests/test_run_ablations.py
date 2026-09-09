from experiments.common import ExperimentResult, append_result
from experiments.report import RESULT_FILENAMES
from experiments.run_ablations import render_report
from experiments.run_ablations import run as run_ablations


def _row(variant_name: str, rate: float, seed: int) -> ExperimentResult:
    return ExperimentResult(
        experiment="E9", decoder="gnn", observable_kind="spacelike", seed=seed,
        logical_error_rate=rate, extra={"variant_name": variant_name},
    )


def test_report_shows_delta_vs_baseline_and_flags_low_seed_count(tmp_path):
    path = tmp_path / "e9_ablations.jsonl"
    for seed, rate in enumerate([0.10, 0.11, 0.09]):
        append_result(path, _row("baseline (all components on)", rate, seed))
    for seed, rate in enumerate([0.30, 0.32]):  # only 2 seeds -- should be flagged
        append_result(path, _row("region/phase features removed", rate, seed))

    report = render_report(str(path), min_seeds=3)
    assert "baseline (all components on)" in report
    assert "region/phase features removed" in report
    assert "INSUFFICIENT SEEDS" in report
    assert "+0.2" in report  # region/phase removal is worse than baseline by ~0.2


def test_report_handles_missing_baseline_gracefully(tmp_path):
    path = tmp_path / "empty.jsonl"
    report = render_report(str(path))
    assert "No baseline" in report


def test_default_results_path_does_not_collide_with_main_table_e9_file():
    """Regression test for a real bug: run_ablations.py's leave-one-out
    variants and run_all.py's own single-config E9 stage must not write to
    the same file, or main_table.md's one E9 summary row silently averages
    together configs that are deliberately different (that's the whole
    point of the ablation study)."""
    import inspect

    default_results_path = inspect.signature(run_ablations).parameters["results_path"].default
    basename = default_results_path.rsplit("/", 1)[-1]
    assert basename not in RESULT_FILENAMES
