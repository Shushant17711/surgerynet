"""A union-find-style baseline decoder (design doc §5.4, union-find row:
"fast classical baseline"). Same decode(detections) -> predictions
interface as baselines/mwpm.py, for drop-in comparison and E8's latency
benchmark.

Implements the cluster-growth-then-peel structure of Delfosse & Nickerson
(2017), with two simplifications made for auditability given how much
correctness risk this project already absorbed getting the circuit itself
right (see circuits/lattice_surgery.py's module docstring — a lesson
taken seriously here): growth proceeds by whole edges per round rather
than scheduled half-edges (a valid, if slightly less optimal, decoder —
see test_union_find.py's hand-checkable tiny cases), and only weight-1
("boundary") and weight-2 DEM error mechanisms become graph edges — the
weight>=3 hyperedges `decompose_errors=True` sometimes can't fully reduce
(a real but minority fraction of mechanisms, e.g. correlated two-qubit
errors) are dropped rather than handled via a hypergraph union-find
variant.

Algorithm per shot:
  1. Each fired detector starts as its own singleton cluster.
  2. Repeatedly grow every unsatisfied ("odd", not boundary-touching)
     cluster by all edges leaving it, unioning clusters that touch;
     unioning across an edge adds that edge to the running spanning
     forest. Stop once every cluster is satisfied (even parity, or has
     reached the virtual boundary node).
  3. Peel each cluster's spanning tree from the leaves inward, deciding
     which edges are part of the correction by propagating leftover
     parity toward the root.
  4. The predicted observable flip is the XOR of the (precomputed)
     observable-flip masks of every edge used in the correction.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import stim

BOUNDARY = -1


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self._parent[ra] = rb
        return True


class UnionFindBaseline:
    def __init__(self, circuit: stim.Circuit) -> None:
        dem = circuit.detector_error_model(decompose_errors=True)
        self.num_detectors = circuit.num_detectors
        self.adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.edge_endpoints: list[tuple[int, int]] = []
        self.edge_obs_mask: list[frozenset[int]] = []

        seen_pairs: set[tuple[int, int]] = set()
        for inst in dem.flattened():
            if inst.type != "error":
                continue
            dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
            obs = frozenset(t.val for t in inst.targets_copy() if t.is_logical_observable_id())
            if len(dets) == 1:
                u, v = dets[0], BOUNDARY
            elif len(dets) == 2:
                u, v = dets[0], dets[1]
            else:
                continue
            key = (min(u, v), max(u, v))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            edge_id = len(self.edge_endpoints)
            self.edge_endpoints.append((u, v))
            self.edge_obs_mask.append(obs)
            self.adjacency[u].append((v, edge_id))
            self.adjacency[v].append((u, edge_id))

    def decode(self, detections: np.ndarray) -> np.ndarray:
        single = detections.ndim == 1
        rows = detections[None, :] if single else detections
        preds = np.array([self._decode_one(row) for row in rows], dtype=bool)
        return preds[0] if single else preds

    def _decode_one(self, detection_row: np.ndarray) -> bool:
        fired = [i for i, v in enumerate(detection_row) if v]
        if not fired:
            return False

        dsu = _UnionFind()
        cluster_nodes: dict[int, set[int]] = {d: {d} for d in fired}
        cluster_parity: dict[int, int] = {d: 1 for d in fired}
        cluster_boundary: dict[int, bool] = {d: False for d in fired}
        spanning_edges: list[int] = []
        max_rounds = max(1, self.num_detectors)

        def root_state(node: int) -> int:
            return dsu.find(node)

        for _ in range(max_rounds):
            roots = {root_state(d) for d in fired}
            unsatisfied = [
                r for r in roots if not cluster_boundary.get(r, False) and cluster_parity.get(r, 0) % 2 == 1
            ]
            if not unsatisfied:
                break

            frontier_nodes: set[int] = set()
            for r in unsatisfied:
                frontier_nodes |= cluster_nodes[r]

            merges: list[tuple[int, int, int]] = []  # (node_a, node_b, edge_id)
            for node in frontier_nodes:
                for neighbor, edge_id in self.adjacency.get(node, []):
                    if dsu.find(node) != dsu.find(neighbor):
                        merges.append((node, neighbor, edge_id))

            for a, b, edge_id in merges:
                ra, rb = dsu.find(a), dsu.find(b)
                if ra == rb:
                    continue
                spanning_edges.append(edge_id)
                nodes_a = cluster_nodes.pop(ra, {a})
                nodes_b = cluster_nodes.pop(rb, {b})
                parity_a = cluster_parity.pop(ra, 0)
                parity_b = cluster_parity.pop(rb, 0)
                boundary_a = cluster_boundary.pop(ra, ra == BOUNDARY)
                boundary_b = cluster_boundary.pop(rb, rb == BOUNDARY)
                dsu.union(ra, rb)
                new_root = dsu.find(ra)
                cluster_nodes[new_root] = nodes_a | nodes_b | {a, b}
                cluster_parity[new_root] = parity_a ^ parity_b
                cluster_boundary[new_root] = boundary_a or boundary_b or new_root == BOUNDARY

        return self._peel(fired, spanning_edges)

    def _peel(self, fired: list[int], spanning_edges: list[int]) -> bool:
        tree_adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for edge_id in spanning_edges:
            u, v = self.edge_endpoints[edge_id]
            tree_adjacency[u].append((v, edge_id))
            tree_adjacency[v].append((u, edge_id))

        defect = {d: True for d in fired}
        degree = {node: len(neighbors) for node, neighbors in tree_adjacency.items()}
        removed_edges: set[int] = set()

        leaves = [n for n, deg in degree.items() if deg == 1 and n != BOUNDARY]
        predicted_flip = False

        while leaves:
            leaf = leaves.pop()
            if degree.get(leaf, 0) != 1:
                continue
            remaining = [
                (nb, eid) for nb, eid in tree_adjacency[leaf] if eid not in removed_edges
            ]
            if not remaining:
                continue
            parent, edge_id = remaining[0]
            removed_edges.add(edge_id)
            degree[leaf] -= 1
            degree[parent] = degree.get(parent, 1) - 1

            if defect.get(leaf, False):
                if 0 in self.edge_obs_mask[edge_id]:
                    predicted_flip = not predicted_flip
                defect[parent] = not defect.get(parent, False)

            if degree.get(parent, 0) == 1 and parent != BOUNDARY:
                leaves.append(parent)

        return predicted_flip
