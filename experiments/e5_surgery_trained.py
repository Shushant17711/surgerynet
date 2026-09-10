"""E5 — surgery-trained model (design doc §6 E5, H3 setup): train a GNN
directly on merge/split data (full region/phase features, not masked),
then evaluate with the exact same harness as E3
(experiments.common.evaluate_gnn) so "how much of the E3 zero-shot gap
closes" is a like-for-like comparison, not an artifact of two different
eval codepaths.
"""

from __future__ import annotations

from circuits.lattice_surgery import load_generated_circuit
from data.generate import sample_shots
from data.to_graph import GraphBuildContext, batch_to_graphs
from experiments.common import ExperimentResult, evaluate_gnn, train_gnn
from models.gnn_decoder import GNNDecoder
from schema import NUM_EDGE_FEATURES


def run(
    surgery_circuit_path: str,
    patch_dimension: float,
    observable_kind: str,
    train_shots: int = 5000,
    test_shots: int = 5000,
    epochs: int = 3,
    batch_size: int = 128,
    seed: int = 0,
    p: float | None = None,
    k: int | None = None,
    hidden_dim: int = 128,
    num_layers: int = 6,
    conv_type: str = "transformer",
    heads: int = 2,
    use_norm: bool = True,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
    edge_dim: int | None = NUM_EDGE_FEATURES,
    use_sum_pool: bool = False,
) -> tuple[ExperimentResult, GNNDecoder]:
    """`p`/`k` are recorded on the result but not used to build the
    circuit — record the same (k, p) the caller generated
    `surgery_circuit_path` with, so results/report.py doesn't merge rows
    from different physical error rates into one aggregate."""
    circuit = load_generated_circuit(surgery_circuit_path)
    ctx = GraphBuildContext.build(circuit, patch_dimension=patch_dimension)

    train_batch = sample_shots(circuit, shots=train_shots, seed=seed)
    train_graphs = batch_to_graphs(train_batch, ctx, observable_kind=observable_kind, mask_surgery_features=False)

    model = train_gnn(
        train_graphs,
        hidden_dim=hidden_dim, num_layers=num_layers, conv_type=conv_type, heads=heads, use_norm=use_norm,
        lr=lr, weight_decay=weight_decay, epochs=epochs, batch_size=batch_size, seed=seed, device=device,
        edge_dim=edge_dim, use_sum_pool=use_sum_pool,
    )

    test_batch = sample_shots(circuit, shots=test_shots, seed=seed + 1)
    test_graphs = batch_to_graphs(test_batch, ctx, observable_kind=observable_kind, mask_surgery_features=False)
    rate = evaluate_gnn(model, test_graphs)

    result = ExperimentResult(
        experiment="E5",
        decoder="gnn",
        observable_kind=observable_kind,
        seed=seed,
        p=p,
        k=k,
        logical_error_rate=rate,
        extra={"surgery_trained": True, "train_shots": train_shots, "epochs": epochs},
    )
    return result, model
