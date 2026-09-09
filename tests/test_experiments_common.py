import pytest

from experiments.common import ExperimentResult, append_result, read_results


def test_result_requires_error_rate_when_ok():
    with pytest.raises(ValueError):
        ExperimentResult(experiment="E2", decoder="gnn", observable_kind="spacelike", status="ok")


def test_infeasible_row_does_not_need_error_rate():
    r = ExperimentResult(experiment="E3", decoder="mlp", observable_kind="spacelike", status="infeasible")
    assert r.logical_error_rate is None


def test_append_and_read_round_trips(tmp_path):
    path = tmp_path / "e2.jsonl"
    r1 = ExperimentResult(
        experiment="E2", decoder="gnn", observable_kind="spacelike", distance=3, p=0.005, logical_error_rate=0.01
    )
    r2 = ExperimentResult(experiment="E2", decoder="mlp", observable_kind="spacelike", status="infeasible")
    append_result(path, r1)
    append_result(path, r2)

    rows = read_results(path)
    assert len(rows) == 2
    assert rows[0] == r1
    assert rows[1] == r2
