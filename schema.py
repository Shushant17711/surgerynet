"""Shared contracts for SurgeryNet.

Every module that builds or consumes a graph (data/to_graph.py,
models/gnn_decoder.py, models/mlp_decoder.py, models/cnn_decoder.py,
baselines/*.py) imports the feature layout from here instead of
redefining it. This is what keeps the "no absolute coordinates, no
fixed patch size baked into a dense layer" invariance discipline
(design doc §5.2) enforceable in code review rather than by convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

STABILIZER_TYPES: tuple[str, ...] = ("X", "Z")
REGION_LABELS: tuple[str, ...] = ("patch_A", "patch_B", "routing_region")
PHASE_LABELS: tuple[str, ...] = ("pre_merge", "merged", "post_split")

Basis = Literal["X", "Z"]
RegionLabel = Literal["patch_A", "patch_B", "routing_region"]
PhaseLabel = Literal["pre_merge", "merged", "post_split"]

# Node feature vector layout. Fixed order, fixed width — anything that reads
# or writes node features indexes through these slices, never a bare int.
NODE_FEATURE_NAMES: tuple[str, ...] = (
    "x",
    "y",
    "round",
    "stab_X",
    "stab_Z",
    "region_patch_A",
    "region_patch_B",
    "region_routing_region",
    "phase_pre_merge",
    "phase_merged",
    "phase_post_split",
    "density",
)
NUM_NODE_FEATURES: int = len(NODE_FEATURE_NAMES)

XY_SLICE = slice(0, 2)
ROUND_SLICE = slice(2, 3)
STABILIZER_TYPE_SLICE = slice(3, 5)
REGION_SLICE = slice(5, 8)
PHASE_SLICE = slice(8, 11)
DENSITY_SLICE = slice(11, 12)

assert REGION_SLICE.stop - REGION_SLICE.start == len(REGION_LABELS)
assert PHASE_SLICE.stop - PHASE_SLICE.start == len(PHASE_LABELS)
assert STABILIZER_TYPE_SLICE.stop - STABILIZER_TYPE_SLICE.start == len(STABILIZER_TYPES)
assert DENSITY_SLICE.stop == NUM_NODE_FEATURES

# Region/phase features must be zeroable independently of everything else,
# so H1 (zero-shot transfer, design §5.2's fairness paragraph) can mask them
# out at memory-training time without touching spatial/round/type features.
SURGERY_ONLY_SLICES: tuple[slice, ...] = (REGION_SLICE, PHASE_SLICE)

# The two decoder output heads (design §5.3) — spacelike for the logical
# observable, timelike for the parity-measurement outcome. H4 depends on
# these staying separate all the way through training and reporting.
OUTPUT_HEADS: tuple[str, ...] = ("spacelike", "timelike")

# Edge feature vector layout (added post-hoc: the GNN originally saw only
# graph *topology* — which detectors are connected — never the DEM's own
# per-edge error probability, which is exactly what MWPM's matching weight
# comes from. `dem_log_odds` is that same information (log((1-p)/p), the
# standard matching-decoder edge weight), not a topology proxy.
EDGE_FEATURE_NAMES: tuple[str, ...] = (
    "dem_log_odds",  # 0 for edges with no DEM-derived weight (radius-only edges)
    "is_dem_edge",
    "is_radius_edge",
    "spatial_distance",  # normalized by patch_dimension
    "temporal_distance",  # normalized by total_rounds
)
NUM_EDGE_FEATURES: int = len(EDGE_FEATURE_NAMES)


@dataclass(frozen=True, slots=True)
class PatchConfig:
    """A single rotated surface code patch."""

    distance: int

    def __post_init__(self) -> None:
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError(f"distance must be odd and >= 3, got {self.distance}")


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    """The routing region joining two patches during a merge."""

    width: int

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError(f"routing width must be >= 1, got {self.width}")


@dataclass(frozen=True, slots=True)
class SurgeryConfig:
    """One merge/split configuration: two patches, a routing region, a schedule.

    This is the shared unit that circuits/lattice_surgery.py builds a circuit
    from, circuits/configurations.py enumerates sweeps over, and
    data/to_graph.py reads back to label nodes with region/phase.
    """

    patch_a: PatchConfig
    patch_b: PatchConfig
    routing: RoutingConfig
    merge_rounds: int
    basis: Basis = "Z"

    def __post_init__(self) -> None:
        if self.merge_rounds < 1:
            raise ValueError(f"merge_rounds must be >= 1, got {self.merge_rounds}")

    @property
    def max_distance(self) -> int:
        return max(self.patch_a.distance, self.patch_b.distance)
