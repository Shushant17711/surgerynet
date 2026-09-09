"""E2 — memory baseline (design doc §6 E2): train the GNN on plain
`distance`-round memory data, compare against MWPM. "Beat MWPM there, as
prior work does. If you cannot, the problem is your GNN, not lattice
surgery." Emits a result row regardless of outcome (experiments/common.py)
so a "GNN did not beat MWPM" result is captured, not swallowed.

Uses a plain stim-generated memory circuit (not the TQEC lattice-surgery
backend) — data/to_graph.py's region/phase inference degrades gracefully
here (no merge seam to find) and is masked off anyway via
mask_surgery_features=True, so the same pipeline serves both.
"""

from __future__ import annotations

import argparse

import stim

from baselines.mwpm import MWPMBaseline
from data.generate import sample_shots
from data.to_graph import GraphBuildContext, batch_to_graphs
from experiments.common import ExperimentResult, append_result, evaluate_gnn, train_gnn
from models.gnn_decoder import GNNDecoder
from schema import NUM_EDGE_FEATURES


def run(
    distance: int = 3,
    p: float = 0.005,
    train_shots: int = 5000,
    test_shots: int = 5000,
    epochs: int = 3,
    batch_size: int = 128,
    seed: int = 0,
    hidden_dim: int = 128,
    num_layers: int = 6,
    conv_type: str = "transformer",
    heads: int = 2,
    use_norm: bool = True,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
    edge_dim: int | None = NUM_EDGE_FEATURES,
) -> tuple[ExperimentResult, ExperimentResult, GNNDecoder]:
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=distance, rounds=distance, after_clifford_depolarization=p
    )
    ctx = GraphBuildContext.build(circuit, patch_dimension=float(distance))

    train_batch = sample_shots(circuit, shots=train_shots, seed=seed)
    train_graphs = batch_to_graphs(train_batch, ctx, observable_kind="spacelike", mask_surgery_features=True)

    model = train_gnn(
        train_graphs,
        hidden_dim=hidden_dim, num_layers=num_layers, conv_type=conv_type, heads=heads, use_norm=use_norm,
        lr=lr, weight_decay=weight_decay, epochs=epochs, batch_size=batch_size, seed=seed, device=device,
        edge_dim=edge_dim,
    )

    test_batch = sample_shots(circuit, shots=test_shots, seed=seed + 1)
    test_graphs = batch_to_graphs(test_batch, ctx, observable_kind="spacelike", mask_surgery_features=True)
    gnn_rate = evaluate_gnn(model, test_graphs)

    mwpm = MWPMBaseline(circuit)
    mwpm_preds = mwpm.decode(test_batch.detections)
    mwpm_rate = float((mwpm_preds != test_batch.observables[:, 0]).mean())

    gnn_result = ExperimentResult(
        experiment="E2",
        decoder="gnn",
        observable_kind="spacelike",
        distance=distance,
        p=p,
        seed=seed,
        logical_error_rate=gnn_rate,
        extra={"beats_mwpm": gnn_rate < mwpm_rate, "train_shots": train_shots, "epochs": epochs},
    )
    mwpm_result = ExperimentResult(
        experiment="E2", decoder="mwpm", observable_kind="spacelike", distance=distance, p=p, seed=seed,
        logical_error_rate=mwpm_rate,
    )
    return gnn_result, mwpm_result, model


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--train-shots", type=int, default=5000)
    parser.add_argument("--test-shots", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/e2_memory_baseline.jsonl")
    args = parser.parse_args()

    gnn_result, mwpm_result, _ = run(
        distance=args.distance,
        p=args.p,
        train_shots=args.train_shots,
        test_shots=args.test_shots,
        epochs=args.epochs,
        seed=args.seed,
    )
    append_result(args.out, gnn_result)
    append_result(args.out, mwpm_result)
    print(f"GNN:  {gnn_result.logical_error_rate:.4f}")
    print(f"MWPM: {mwpm_result.logical_error_rate:.4f}")
    print("GNN beats MWPM:", gnn_result.extra["beats_mwpm"])


if __name__ == "__main__":
    _cli()
