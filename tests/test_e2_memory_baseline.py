from experiments.e2_memory_baseline import run
from models.gnn_decoder import GNNDecoder


def test_e2_runs_end_to_end_and_emits_valid_result_rows():
    """Smoke test at a tiny scale — not expected to actually beat MWPM
    with this few shots/epochs, just checks the pipeline wiring and result
    schema are correct. Real runs use scripts/reproduce.sh-scale shots."""
    gnn_result, mwpm_result, model = run(
        distance=3, p=0.01, train_shots=200, test_shots=200, epochs=1, batch_size=64, seed=0
    )
    assert gnn_result.experiment == "E2" and gnn_result.decoder == "gnn"
    assert mwpm_result.experiment == "E2" and mwpm_result.decoder == "mwpm"
    assert 0.0 <= gnn_result.logical_error_rate <= 1.0
    assert 0.0 <= mwpm_result.logical_error_rate <= 1.0
    assert "beats_mwpm" in gnn_result.extra
    assert isinstance(model, GNNDecoder)
