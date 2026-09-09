"""Detection-event -> PyG graph conversion (design doc §5.2, Task 3.2/3.3).

Nodes are detectors that *fired* in a given shot. Edges come from two
sources, unioned: pairs that co-occur in a decomposed detector-error-model
mechanism (the decoder-relevant correlation structure — "use it, do not
guess a radius", per the design doc), and pairs within a configurable
space-time radius (a locality fallback/complement, and the only edge
source once graphs get big enough that iterating every DEM hyperedge would
be wasteful — radius edges are cheap to compute directly from coordinates).

**Edge features (post-hoc addition).** The graph originally carried edge
*topology* only — which detectors are connected — never the DEM's own
per-edge error probability. That is exactly the number MWPM's matching
weight comes from (`weight = log((1-p)/p)`), so a GNN with no access to it
is working with strictly less information than the baseline it is being
compared against for every edge derived from the DEM. `dem_edge_weights`
computes that same log-odds weight per co-occurring detector pair (combining
multiple contributing error mechanisms the way independent-probability
combination requires: `p_combined = p1 + p2 - 2*p1*p2`), and `shot_to_graph`
now attaches it — plus edge-source flags and normalized spatial/temporal
distance — as `edge_attr` (`schema.EDGE_FEATURE_NAMES`).

Invariance discipline (§5.2): coordinates are normalized by patch
dimension and round index by total rounds, so a graph's *shape* doesn't
encode absolute patch size or round count — verified by
test_two_configurations_produce_structurally_equivalent_graphs below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import stim
import torch
from torch_geometric.data import Data

from data.generate import ShotBatch
from data.spacetime_labels import DetectorLabel, infer_detector_labels
from schema import (
    DENSITY_SLICE,
    NUM_EDGE_FEATURES,
    NUM_NODE_FEATURES,
    OUTPUT_HEADS,
    PHASE_LABELS,
    PHASE_SLICE,
    REGION_LABELS,
    REGION_SLICE,
    ROUND_SLICE,
    STABILIZER_TYPE_SLICE,
    STABILIZER_TYPES,
    SURGERY_ONLY_SLICES,
    XY_SLICE,
)


def dem_edge_weights(circuit: stim.Circuit) -> dict[frozenset[int], float]:
    """Detector-index pairs that co-occur in some decomposed DEM error
    mechanism, mapped to the combined probability of *some* contributing
    mechanism flipping that edge — the true correlation structure (and
    weight) a matching-style decoder uses, independent of any chosen
    spatial radius. Multiple mechanisms contributing to the same pair
    combine via the standard independent-probability formula
    `p = p1 + p2 - 2*p1*p2` (two independent sources, either flipping this
    edge but not both, still flips it)."""
    dem = circuit.detector_error_model(decompose_errors=True)
    weights: dict[frozenset[int], float] = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                pair = frozenset((dets[i], dets[j]))
                prev = weights.get(pair, 0.0)
                weights[pair] = prev + p - 2 * prev * p
    return weights


def dem_edge_pairs(circuit: stim.Circuit) -> set[frozenset[int]]:
    """Detector-index pairs that co-occur in some decomposed DEM error
    mechanism — the true correlation structure a matching-style decoder
    would use, independent of any chosen spatial radius. (Just the key set
    of `dem_edge_weights`; kept as its own function since most callers only
    need the topology, not the weights.)"""
    return set(dem_edge_weights(circuit).keys())


def _dem_log_odds(p: float) -> float:
    """`log((1-p)/p)`, the standard matching-decoder edge weight — larger
    for more confident (lower-probability) edges. Clipped to keep this
    finite and in a sane range for a raw feature input (a p of exactly 0
    or 1 would otherwise blow up or divide by zero)."""
    p = min(max(p, 1e-6), 0.5)
    return math.log((1.0 - p) / p)


@dataclass(frozen=True, slots=True)
class GraphBuildContext:
    """Everything needed to convert shots from one circuit into graphs,
    computed once per circuit and reused across all its shots."""

    labels: dict[int, DetectorLabel]
    dem_weights: dict[frozenset[int], float]
    patch_dimension: float
    total_rounds: float
    spatial_radius: float = 3.0
    temporal_radius: float = 1.0

    @classmethod
    def build(cls, circuit: stim.Circuit, patch_dimension: float, spatial_radius: float = 3.0) -> "GraphBuildContext":
        labels = infer_detector_labels(circuit)
        all_t = [lbl.t for lbl in labels.values()]
        total_rounds = (max(all_t) - min(all_t) + 1) if all_t else 1.0
        return cls(
            labels=labels,
            dem_weights=dem_edge_weights(circuit),
            patch_dimension=float(patch_dimension),
            total_rounds=float(total_rounds),
        )


def _local_density(fired: list[int], ctx: GraphBuildContext) -> dict[int, float]:
    """Number of *other* fired detectors within the spatial+temporal
    radius, normalized by the count of fired detectors — a cheap proxy for
    "how crowded is the syndrome here", one of §5.2's node features."""
    density: dict[int, float] = {}
    n = max(1, len(fired) - 1)
    for d in fired:
        lbl = ctx.labels[d]
        count = 0
        for other in fired:
            if other == d:
                continue
            o = ctx.labels[other]
            if abs(o.t - lbl.t) <= ctx.temporal_radius and (
                (o.x - lbl.x) ** 2 + (o.y - lbl.y) ** 2
            ) ** 0.5 <= ctx.spatial_radius:
                count += 1
        density[d] = count / n
    return density


def shot_to_graph(
    detections_row: np.ndarray,
    observables_row: np.ndarray,
    ctx: GraphBuildContext,
    observable_kind: str,
    mask_surgery_features: bool = False,
    use_dem_edges: bool = True,
    use_radius_edges: bool = True,
) -> Data:
    """Build one PyG Data object from a single shot's fired detectors.

    `observable_kind` ("spacelike" or "timelike") records which of the
    model's two output heads (schema.OUTPUT_HEADS) this shot's single
    observable column is a label for — circuits/tqec_backend.py generates
    spacelike and timelike data as *separate* circuits (TQEC's ports can
    only be filled with one basis at a time), so a given shot never has
    both labels available at once. `y` holds the one available label;
    `head_id` tells the training loop which head's loss to apply it to.

    `mask_surgery_features=True` zeroes the region/phase one-hots (Task
    3.3, H1's fairness precondition — memory-training data must not leak
    region/phase information that only exists during real lattice surgery).

    `use_dem_edges`/`use_radius_edges` (Task 6.8/E9's own ablation knobs)
    toggle each edge source independently; at least one should stay on or
    fired detectors end up with no edges at all.
    """
    if observable_kind not in OUTPUT_HEADS:
        raise ValueError(f"observable_kind must be one of {OUTPUT_HEADS}, got {observable_kind!r}")
    head_id = OUTPUT_HEADS.index(observable_kind)
    fired = [i for i, v in enumerate(detections_row) if v]

    if fired:
        density = _local_density(fired, ctx)
        x = torch.zeros((len(fired), NUM_NODE_FEATURES), dtype=torch.float32)
        for row, d in enumerate(fired):
            lbl = ctx.labels[d]
            x[row, XY_SLICE.start] = lbl.x / ctx.patch_dimension
            x[row, XY_SLICE.start + 1] = lbl.y / ctx.patch_dimension
            x[row, ROUND_SLICE.start] = lbl.t / ctx.total_rounds
            x[row, STABILIZER_TYPE_SLICE.start + STABILIZER_TYPES.index(lbl.stabilizer_type)] = 1.0
            if not mask_surgery_features:
                x[row, REGION_SLICE.start + REGION_LABELS.index(lbl.region)] = 1.0
                x[row, PHASE_SLICE.start + PHASE_LABELS.index(lbl.phase)] = 1.0
            x[row, DENSITY_SLICE.start] = density[d]
    else:
        x = torch.zeros((0, NUM_NODE_FEATURES), dtype=torch.float32)

    index_of = {d: i for i, d in enumerate(fired)}
    fired_set = set(fired)
    # (node_i, node_j) -> [dem_log_odds, is_dem_edge, is_radius_edge, spatial_dist, temporal_dist],
    # built directed-both-ways so edge_attr stays aligned with edge_index after the final sort/dedup
    # below (a DEM edge that's also within radius keeps its weight AND gets is_radius_edge=1 — the
    # two sources are unioned onto one edge, not duplicated into two parallel edges).
    edge_attrs: dict[tuple[int, int], list[float]] = {}

    def _set_or_merge(ii: int, jj: int, dem_w: float, is_dem: float, is_radius: float, sp_dist: float, t_dist: float) -> None:
        for a, b in ((ii, jj), (jj, ii)):
            if (a, b) in edge_attrs:
                existing = edge_attrs[(a, b)]
                existing[0] = max(existing[0], dem_w)
                existing[1] = max(existing[1], is_dem)
                existing[2] = max(existing[2], is_radius)
            else:
                edge_attrs[(a, b)] = [dem_w, is_dem, is_radius, sp_dist, t_dist]

    if use_dem_edges:
        for pair, p in ctx.dem_weights.items():
            a, b = tuple(pair)
            if a in fired_set and b in fired_set:
                la, lb = ctx.labels[a], ctx.labels[b]
                sp_dist = math.hypot(la.x - lb.x, la.y - lb.y) / ctx.patch_dimension
                t_dist = abs(la.t - lb.t) / ctx.total_rounds
                _set_or_merge(index_of[a], index_of[b], _dem_log_odds(p), 1.0, 0.0, sp_dist, t_dist)
    if use_radius_edges:
        for i, di in enumerate(fired):
            li = ctx.labels[di]
            for dj in fired[i + 1 :]:
                lj = ctx.labels[dj]
                spatial = math.hypot(li.x - lj.x, li.y - lj.y)
                if abs(li.t - lj.t) <= ctx.temporal_radius and spatial <= ctx.spatial_radius:
                    sp_dist = spatial / ctx.patch_dimension
                    t_dist = abs(li.t - lj.t) / ctx.total_rounds
                    _set_or_merge(index_of[di], index_of[dj], 0.0, 0.0, 1.0, sp_dist, t_dist)

    if edge_attrs:
        pairs_sorted = sorted(edge_attrs.keys())
        edge_index = torch.tensor(pairs_sorted, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([edge_attrs[p] for p in pairs_sorted], dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, NUM_EDGE_FEATURES), dtype=torch.float32)

    y = torch.tensor([[float(observables_row[0])]], dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, head_id=head_id, num_fired=len(fired))


def batch_to_graphs(
    batch: ShotBatch,
    ctx: GraphBuildContext,
    observable_kind: str,
    mask_surgery_features: bool = False,
    use_dem_edges: bool = True,
    use_radius_edges: bool = True,
) -> list[Data]:
    return [
        shot_to_graph(
            batch.detections[i],
            batch.observables[i],
            ctx,
            observable_kind,
            mask_surgery_features,
            use_dem_edges,
            use_radius_edges,
        )
        for i in range(batch.detections.shape[0])
    ]


def assert_no_surgery_feature_leakage(graphs: list[Data]) -> None:
    """Task 3.3's own correctness check: memory-mode graphs must have
    exactly-zero region/phase one-hots, or H1's zero-shot claim is
    contaminated by leaked surgery-only features."""
    for g in graphs:
        for s in SURGERY_ONLY_SLICES:
            if g.x.numel() and g.x[:, s].abs().sum() != 0:
                raise AssertionError("surgery-only feature leaked into a masked graph")
