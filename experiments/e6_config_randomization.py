"""E6 — configuration randomization (design doc §6 E6, H3): train across
the Task 2.3 configuration sweep, evaluate on a held-out configuration
subset, report the generalization gap (design §7's own named metric).

"Configuration" here means TQEC's scaling parameter k, since the TQEC
backend ties patch distance and merge/round duration to k together (no
independent routing-width knob yet — see circuits/configurations.py).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from torch_geometric.data import Data

from circuits.configurations import SweptConfig, held_out_split, sweep_configurations
from circuits.lattice_surgery import load_generated_circuit
from data.generate import sample_shots
from data.to_graph import GraphBuildContext, batch_to_graphs
from experiments.common import ExperimentResult, evaluate_gnn, generate_surgery_circuit, train_gnn


def _graphs_for_config(
    sc: SweptConfig, p: float, kind: str, shots: int, seed: int, tmp_dir: Path
) -> list[Data]:
    out_path = tmp_dir / f"k{sc.k}_{kind}.stim"
    generate_surgery_circuit(sc.k, p, kind, out_path)
    circuit = load_generated_circuit(out_path)
    ctx = GraphBuildContext.build(circuit, patch_dimension=float(sc.config.patch_a.distance))
    batch = sample_shots(circuit, shots=shots, seed=seed)
    return batch_to_graphs(batch, ctx, observable_kind=kind, mask_surgery_features=False)


def run(
    distances: tuple[int, ...] = (3, 5, 7),
    p: float = 0.01,
    observable_kind: str = "spacelike",
    shots_per_config: int = 500,
    epochs: int = 2,
    batch_size: int = 64,
    train_fraction: float = 0.7,
    seed: int = 0,
    hidden_dim: int = 64,
    num_layers: int = 6,
    conv_type: str = "transformer",
    heads: int = 2,
    use_norm: bool = True,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    device: str | None = None,
) -> ExperimentResult:
    swept = sweep_configurations(distances=distances, merge_round_multipliers=(1.0,))
    train_configs, held_out_configs = held_out_split(swept, train_fraction=train_fraction, seed=seed)

    train_ks = {sc.k for sc in train_configs}
    held_out_ks = {sc.k for sc in held_out_configs}
    if not train_ks.isdisjoint(held_out_ks):
        raise AssertionError("train and held-out configuration sets are not disjoint")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        train_graphs: list[Data] = []
        for sc in train_configs:
            train_graphs.extend(
                _graphs_for_config(sc, p, observable_kind, shots_per_config, seed, tmp_dir)
            )

        model = train_gnn(
            train_graphs,
            hidden_dim=hidden_dim, num_layers=num_layers, conv_type=conv_type, heads=heads, use_norm=use_norm,
            lr=lr, weight_decay=weight_decay, epochs=epochs, batch_size=batch_size, seed=seed, device=device,
        )

        train_eval_graphs: list[Data] = []
        for sc in train_configs:
            train_eval_graphs.extend(
                _graphs_for_config(sc, p, observable_kind, shots_per_config, seed + 1, tmp_dir)
            )
        held_out_graphs: list[Data] = []
        for sc in held_out_configs:
            held_out_graphs.extend(
                _graphs_for_config(sc, p, observable_kind, shots_per_config, seed + 1, tmp_dir)
            )

    train_rate = evaluate_gnn(model, train_eval_graphs)
    held_out_rate = evaluate_gnn(model, held_out_graphs)

    return ExperimentResult(
        experiment="E6",
        decoder="gnn",
        observable_kind=observable_kind,
        seed=seed,
        p=p,
        logical_error_rate=held_out_rate,
        extra={
            "train_configuration_error_rate": train_rate,
            "generalization_gap": held_out_rate - train_rate,
            "train_distances": sorted({sc.config.patch_a.distance for sc in train_configs}),
            "held_out_distances": sorted({sc.config.patch_a.distance for sc in held_out_configs}),
        },
    )
