"""E7 — timelike errors (design doc §6 E7, H4): spacelike vs timelike
logical error rate, separately, at a given physical error rate. Includes
the sanity check design §11 explicitly asks for: cross-check the GNN's
timelike rate against MWPM's timelike rate at the same p before trusting
it ("timelike errors are confusing... sanity-check against MWPM's
timelike rate").
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from baselines.mwpm import MWPMBaseline
from circuits.configurations import k_to_distance
from circuits.lattice_surgery import load_generated_circuit
from data.generate import sample_shots
from data.to_graph import GraphBuildContext, batch_to_graphs
from experiments.common import ExperimentResult, evaluate_gnn, generate_surgery_circuit, train_gnn
from schema import NUM_EDGE_FEATURES


def _run_one_kind(
    k: int,
    p: float,
    kind: str,
    train_shots: int,
    test_shots: int,
    epochs: int,
    batch_size: int,
    seed: int,
    hidden_dim: int = 128,
    num_layers: int = 6,
    conv_type: str = "transformer",
    heads: int = 2,
    use_norm: bool = True,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
    edge_dim: int | None = NUM_EDGE_FEATURES,
) -> tuple[float, float]:
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / f"k{k}_{kind}.stim"
        generate_surgery_circuit(k, p, kind, out_path)
        circuit = load_generated_circuit(out_path)
        ctx = GraphBuildContext.build(circuit, patch_dimension=float(k_to_distance(k)))

        train_batch = sample_shots(circuit, shots=train_shots, seed=seed)
        train_graphs = batch_to_graphs(train_batch, ctx, observable_kind=kind, mask_surgery_features=False)

        model = train_gnn(
            train_graphs,
            hidden_dim=hidden_dim, num_layers=num_layers, conv_type=conv_type, heads=heads, use_norm=use_norm,
            lr=lr, weight_decay=weight_decay, epochs=epochs, batch_size=batch_size, seed=seed, device=device,
            edge_dim=edge_dim,
        )

        test_batch = sample_shots(circuit, shots=test_shots, seed=seed + 1)
        test_graphs = batch_to_graphs(test_batch, ctx, observable_kind=kind, mask_surgery_features=False)
        gnn_rate = evaluate_gnn(model, test_graphs)

        mwpm = MWPMBaseline(circuit)
        mwpm_preds = mwpm.decode(test_batch.detections)
        mwpm_rate = float((mwpm_preds != test_batch.observables[:, 0]).mean())

    return gnn_rate, mwpm_rate


def run(
    k: int = 1,
    p: float = 0.01,
    train_shots: int = 1000,
    test_shots: int = 1000,
    epochs: int = 1,
    batch_size: int = 64,
    seed: int = 0,
    sanity_tolerance: float = 0.3,
    hidden_dim: int = 128,
    num_layers: int = 6,
    conv_type: str = "transformer",
    heads: int = 2,
    use_norm: bool = True,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
    edge_dim: int | None = NUM_EDGE_FEATURES,
) -> list[ExperimentResult]:
    """`sanity_tolerance` is deliberately loose (design §11's check is a
    coarse "did something go wildly wrong", not a precision comparison —
    GNN and MWPM are different decoders and are not expected to match
    closely, only to agree on rough order of magnitude)."""
    results: list[ExperimentResult] = []
    rates: dict[str, dict[str, float]] = {}
    for kind in ("spacelike", "timelike"):
        gnn_rate, mwpm_rate = _run_one_kind(
            k, p, kind, train_shots, test_shots, epochs, batch_size, seed,
            hidden_dim=hidden_dim, num_layers=num_layers, conv_type=conv_type, heads=heads, use_norm=use_norm,
            lr=lr, weight_decay=weight_decay, device=device, edge_dim=edge_dim,
        )
        rates[kind] = {"gnn": gnn_rate, "mwpm": mwpm_rate}
        results.append(
            ExperimentResult(
                experiment="E7", decoder="gnn", observable_kind=kind, k=k, p=p, seed=seed, logical_error_rate=gnn_rate
            )
        )
        results.append(
            ExperimentResult(
                experiment="E7", decoder="mwpm", observable_kind=kind, k=k, p=p, seed=seed, logical_error_rate=mwpm_rate
            )
        )

    timelike_gap = abs(rates["timelike"]["gnn"] - rates["timelike"]["mwpm"])
    if timelike_gap > sanity_tolerance:
        raise AssertionError(
            f"GNN timelike rate ({rates['timelike']['gnn']:.4f}) and MWPM timelike rate "
            f"({rates['timelike']['mwpm']:.4f}) disagree by more than {sanity_tolerance} at p={p} — "
            "per design doc §11, sanity-check the timelike circuit/labels before trusting either number"
        )
    return results
