"""Detection-event -> PyG graph conversion (design doc §5.2, Task 3.2/3.3).

Nodes are detectors that *fired* in a given shot. Edges come from two
sources, unioned: pairs that co-occur in a decomposed detector-error-model
mechanism (the decoder-relevant correlation structure — "use it, do not
guess a radius", per the design doc), and pairs within a configurable
space-time radius (a locality fallback/complement, and the only edge
source once graphs get big enough that iterating every DEM hyperedge would
be wasteful — radius edges are cheap to compute directly from coordinates).

Invariance discipline (§5.2): coordinates are normalized by patch
dimension and round index by total rounds, so a graph's *shape* doesn't
encode absolute patch size or round count — verified by
test_two_configurations_produce_structurally_equivalent_graphs below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim
import torch
from torch_geometric.data import Data

from data.generate import ShotBatch
from data.spacetime_labels import DetectorLabel, infer_detector_labels
from schema import (
    DENSITY_SLICE,
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


def dem_edge_pairs(circuit: stim.Circuit) -> set[frozenset[int]]:
    """Detector-index pairs that co-occur in some decomposed DEM error
    mechanism — the true correlation structure a matching-style decoder
    would use, independent of any chosen spatial radius."""
    dem = circuit.detector_error_model(decompose_errors=True)
    edges: set[frozenset[int]] = set()
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                edges.add(frozenset((dets[i], dets[j])))
    return edges


@dataclass(frozen=True, slots=True)
class GraphBuildContext:
    """Everything needed to convert shots from one circuit into graphs,
    computed once per circuit and reused across all its shots."""

    labels: dict[int, DetectorLabel]
    dem_edges: set[frozenset[int]]
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
            dem_edges=dem_edge_pairs(circuit),
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
    edge_pairs: list[tuple[int, int]] = []
    fired_set = set(fired)
    if use_dem_edges:
        for pair in ctx.dem_edges:
            a, b = tuple(pair)
            if a in fired_set and b in fired_set:
                edge_pairs.append((index_of[a], index_of[b]))
    if use_radius_edges:
        for i, di in enumerate(fired):
            li = ctx.labels[di]
            for dj in fired[i + 1 :]:
                lj = ctx.labels[dj]
                if abs(li.t - lj.t) <= ctx.temporal_radius and (
                    (li.x - lj.x) ** 2 + (li.y - lj.y) ** 2
                ) ** 0.5 <= ctx.spatial_radius:
                    edge_pairs.append((index_of[di], index_of[dj]))

    if edge_pairs:
        undirected = edge_pairs + [(b, a) for a, b in edge_pairs]
        edge_index = torch.tensor(sorted(set(undirected)), dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    y = torch.tensor([[float(observables_row[0])]], dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, y=y, head_id=head_id, num_fired=len(fired))


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
