"""E9 — ablations (design doc §6 E9): region/phase features on/off,
DEM-derived vs radius edges, message-passing depth, normalization on/off,
plus a post-hoc addition: DEM edge *weights* (the log-odds feature added
to close some of the E2 gap — see LIMITATIONS.md) on/off. Each variant
runs through the same E3/E5 evaluation function (experiments.common.evaluate_gnn).
"""

from __future__ import annotations

import itertools
import tempfile
from dataclasses import dataclass
from pathlib import Path

from circuits.configurations import k_to_distance
from circuits.lattice_surgery import load_generated_circuit
from data.generate import sample_shots
from data.to_graph import GraphBuildContext, batch_to_graphs
from experiments.common import ExperimentResult, evaluate_gnn, generate_surgery_circuit, train_gnn
from schema import NUM_EDGE_FEATURES


@dataclass(frozen=True, slots=True)
class AblationConfig:
    mask_surgery_features: bool
    use_dem_edges: bool
    use_radius_edges: bool
    num_layers: int
    use_norm: bool
    use_edge_features: bool = True


def ablation_grid(
    mask_surgery_features_options: tuple[bool, ...] = (False, True),
    edge_source_options: tuple[tuple[bool, bool], ...] = ((True, True), (True, False), (False, True)),
    num_layers_options: tuple[int, ...] = (4, 6),
    use_norm_options: tuple[bool, ...] = (True, False),
    use_edge_features_options: tuple[bool, ...] = (True, False),
) -> list[AblationConfig]:
    """`edge_source_options` is (use_dem_edges, use_radius_edges) pairs —
    (False, False) is excluded, since it would leave every fired detector
    edgeless."""
    return [
        AblationConfig(mask, dem, radius, layers, norm, edge_feat)
        for mask, (dem, radius), layers, norm, edge_feat in itertools.product(
            mask_surgery_features_options, edge_source_options, num_layers_options, use_norm_options,
            use_edge_features_options,
        )
    ]


BASELINE_ABLATION_CONFIG = AblationConfig(
    mask_surgery_features=False, use_dem_edges=True, use_radius_edges=True, num_layers=6, use_norm=True,
    use_edge_features=True,
)
"""`num_layers=6` matches the tuned config from `experiments/hparam_search.py`
(HPARAM_SEARCH.md's winner) — the ablation baseline should be the model this
repo actually uses, not an arbitrary point in the design doc's 4-6 layer range.
`use_edge_features=True` matches the DEM-log-odds-as-edge-attr addition."""


def leave_one_out_variants(baseline: AblationConfig = BASELINE_ABLATION_CONFIG) -> dict[str, AblationConfig]:
    """One variant per component, each flipping exactly one knob relative
    to `baseline` — "what breaks when each component is removed", not the
    full factorial grid. Named so the ablation report can label rows by
    what changed rather than by the raw config tuple.

    `baseline` now defaults to the tuned 6-layer config (the deepest the
    design doc's 4-6 layer range allows), so the depth ablation direction
    flips from the pre-tuning version of this file: it's a genuine
    "less depth" removal (6 -> 4 layers) rather than the old baseline=4
    version's "closest available direction is deeper, not shallower."
    """
    return {
        "baseline (all components on)": baseline,
        "region/phase features removed": AblationConfig(
            mask_surgery_features=True, use_dem_edges=baseline.use_dem_edges,
            use_radius_edges=baseline.use_radius_edges, num_layers=baseline.num_layers, use_norm=baseline.use_norm,
            use_edge_features=baseline.use_edge_features,
        ),
        "DEM edges removed (radius-only)": AblationConfig(
            mask_surgery_features=baseline.mask_surgery_features, use_dem_edges=False,
            use_radius_edges=True, num_layers=baseline.num_layers, use_norm=baseline.use_norm,
            use_edge_features=baseline.use_edge_features,
        ),
        "radius edges removed (DEM-only)": AblationConfig(
            mask_surgery_features=baseline.mask_surgery_features, use_dem_edges=True,
            use_radius_edges=False, num_layers=baseline.num_layers, use_norm=baseline.use_norm,
            use_edge_features=baseline.use_edge_features,
        ),
        "shallower (4 layers vs 6)": AblationConfig(
            mask_surgery_features=baseline.mask_surgery_features, use_dem_edges=baseline.use_dem_edges,
            use_radius_edges=baseline.use_radius_edges, num_layers=4, use_norm=baseline.use_norm,
            use_edge_features=baseline.use_edge_features,
        ),
        "normalization removed": AblationConfig(
            mask_surgery_features=baseline.mask_surgery_features, use_dem_edges=baseline.use_dem_edges,
            use_radius_edges=baseline.use_radius_edges, num_layers=baseline.num_layers, use_norm=False,
            use_edge_features=baseline.use_edge_features,
        ),
        "edge weights removed (topology only)": AblationConfig(
            mask_surgery_features=baseline.mask_surgery_features, use_dem_edges=baseline.use_dem_edges,
            use_radius_edges=baseline.use_radius_edges, num_layers=baseline.num_layers, use_norm=baseline.use_norm,
            use_edge_features=False,
        ),
    }


def run_one(
    config: AblationConfig,
    k: int = 1,
    p: float = 0.01,
    observable_kind: str = "spacelike",
    train_shots: int = 500,
    test_shots: int = 500,
    epochs: int = 1,
    batch_size: int = 64,
    seed: int = 0,
    variant_name: str | None = None,
    hidden_dim: int = 128,
    conv_type: str = "transformer",
    heads: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
) -> ExperimentResult:
    """`num_layers`/`use_norm` come from `config` (that's what this
    experiment ablates) — every other hyperparameter is a passthrough to
    `train_gnn` so a tuned config from `experiments/hparam_search.py` can
    be used as the ablation baseline without duplicating the training
    loop here."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / f"k{k}_{observable_kind}.stim"
        generate_surgery_circuit(k, p, observable_kind, out_path)
        circuit = load_generated_circuit(out_path)
        ctx = GraphBuildContext.build(circuit, patch_dimension=float(k_to_distance(k)))

        train_batch = sample_shots(circuit, shots=train_shots, seed=seed)
        train_graphs = batch_to_graphs(
            train_batch, ctx, observable_kind,
            mask_surgery_features=config.mask_surgery_features,
            use_dem_edges=config.use_dem_edges,
            use_radius_edges=config.use_radius_edges,
        )

        model = train_gnn(
            train_graphs,
            hidden_dim=hidden_dim, num_layers=config.num_layers, conv_type=conv_type, heads=heads,
            use_norm=config.use_norm, lr=lr, weight_decay=weight_decay, epochs=epochs, batch_size=batch_size,
            seed=seed, device=device, edge_dim=(NUM_EDGE_FEATURES if config.use_edge_features else None),
        )

        test_batch = sample_shots(circuit, shots=test_shots, seed=seed + 1)
        test_graphs = batch_to_graphs(
            test_batch, ctx, observable_kind,
            mask_surgery_features=config.mask_surgery_features,
            use_dem_edges=config.use_dem_edges,
            use_radius_edges=config.use_radius_edges,
        )
        rate = evaluate_gnn(model, test_graphs)

    return ExperimentResult(
        experiment="E9",
        decoder="gnn",
        observable_kind=observable_kind,
        k=k,
        p=p,
        seed=seed,
        logical_error_rate=rate,
        extra={
            "variant_name": variant_name,
            "mask_surgery_features": config.mask_surgery_features,
            "use_dem_edges": config.use_dem_edges,
            "use_radius_edges": config.use_radius_edges,
            "num_layers": config.num_layers,
            "use_norm": config.use_norm,
            "use_edge_features": config.use_edge_features,
        },
    )
