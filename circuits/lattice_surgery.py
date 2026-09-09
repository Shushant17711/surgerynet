"""Manual construction of a d=3 (extensible) rotated-surface-code lattice
surgery (merge/split) circuit, per design doc §5.1 route 3.

Coordinate and CNOT-schedule conventions are copied from stim's own
``surface_code:rotated_memory_z`` generator (data qubits at (odd, odd),
ancillas at (even, even), ancilla type by (x//2 + y//2) parity, and the
SE/SW/NE/NW-style four-step CNOT order) — verified to reproduce it exactly
in tests/test_lattice_surgery.py::test_solo_patch_matches_stim_reference.
Building on a verified-identical base for the non-merged case is what
keeps the merge-specific code the only place new bugs can hide.

Merge/split is implemented as "grow the code, hold, then split by directly
measuring the routing column" (Horsman et al. 2012): during the
``pre_merge`` phase patch A and patch B run as two independent d x d
patches; at merge time the routing column(s) are reset and the whole
region becomes one bigger d x (2d+w) rotated-code rectangle for
``merge_rounds`` rounds; splitting measures the routing data qubits
directly in the merge basis (their parity is exactly Z_A(x)Z_B, the
timelike observable) and the two patches resume independently.

Only supports two patches of equal distance in v1 — see SurgeryConfig.

STATUS (see tqec_backend.py for the preferred path): the spacelike
observable (`left_a` below) is fully validated — deterministic under
`detector_error_model()` and shows correct sub-threshold scaling. The
timelike (routing-column parity) observable's Pauli-frame correction is
NOT yet fully closed — a brute-force search over the obvious seam-ancilla
corrections did not find a working combination, and it needs either a more
systematic stabilizer-tableau derivation or is better obtained from
tqec_backend.py's "timelike" circuit kind, which sidesteps the issue by
using a different (TQEC-verified) block-graph observable definition.
Prefer `tqec_backend.py` + `load_generated_circuit` below for real data
generation; this hand-rolled builder remains for the validated spacelike
path and as a from-scratch reference implementation (design doc §5.1
route 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import stim

from schema import Basis, SurgeryConfig

Coord = tuple[int, int]

# Offsets tried in order for each ancilla type when looking for a CX partner
# during one of the four scheduling ticks. Order matters (hook-error
# avoidance) and was read directly off stim's reference circuit.
_X_OFFSETS: tuple[Coord, ...] = ((1, 1), (-1, 1), (1, -1), (-1, -1))
_Z_OFFSETS: tuple[Coord, ...] = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _ancilla_type(x: int, y: int) -> str:
    return "X" if ((x // 2 + y // 2) % 2 == 1) else "Z"


@dataclass(frozen=True)
class _Region:
    """A rotated-code rectangle: data qubits + the ancillas they support."""

    data: frozenset[Coord]
    ancillas: dict[Coord, tuple[str, tuple[Coord, ...]]] = field(default_factory=dict)


def _rect_region(width: int, height: int, x_offset: int = 0, y_offset: int = 0) -> _Region:
    """Data + ancilla coordinates for a `width` x `height` (in data qubits)
    rotated-code rectangle. width == height == d reproduces a normal patch;
    a bigger width (patch_a + routing + patch_b) reproduces the merged code.
    """
    data = {
        (x_offset + 1 + 2 * c, y_offset + 1 + 2 * r)
        for r in range(height)
        for c in range(width)
    }
    x_min, x_max = x_offset, x_offset + 2 * width
    y_min, y_max = y_offset, y_offset + 2 * height
    ancillas: dict[Coord, tuple[str, tuple[Coord, ...]]] = {}
    for gy in range(y_min, y_max + 1, 2):
        for gx in range(x_min, x_max + 1, 2):
            atype = _ancilla_type(gx, gy)
            offsets = _X_OFFSETS if atype == "X" else _Z_OFFSETS
            neighbors = tuple((gx + dx, gy + dy) for dx, dy in offsets if (gx + dx, gy + dy) in data)
            if len(neighbors) < 2:
                continue  # corner: not a real stabilizer
            on_top_or_bottom = gy in (y_min, y_max)
            on_left_or_right = gx in (x_min, x_max)
            if on_top_or_bottom and not on_left_or_right and atype != "X":
                continue  # top/bottom boundary is X-type only, by convention
            if on_left_or_right and not on_top_or_bottom and atype != "Z":
                continue  # left/right boundary is Z-type only, by convention
            ancillas[(gx, gy)] = (atype, neighbors)
    return _Region(data=frozenset(data), ancillas=ancillas)


def _left_column(height: int, x_offset: int, y_offset: int) -> tuple[Coord, ...]:
    """The X_L chain for a patch built by `_rect_region`: its own left column
    (verified against stim's own "surface_code:rotated_memory_x" reference).
    Used as the spacelike observable specifically *because* it sits on the
    boundary the merge never touches — see the long comment in
    build_lattice_surgery_circuit about why the top-row Z_L doesn't survive
    a merge along the left/right (Z-type) boundary."""
    return tuple((x_offset + 1, y_offset + 1 + 2 * r) for r in range(height))


class _Emitter:
    """Accumulates a stim.Circuit while tracking measurement-record offsets
    and each ancilla's last-measured neighbor set, so detectors can be built
    generically for bulk, boundary, freshly-appeared, and about-to-vanish
    ancillas alike."""

    def __init__(self) -> None:
        self.circuit = stim.Circuit()
        self._qubit_index: dict[Coord, int] = {}
        self._mcount = 0
        # coord -> (absolute record index, neighbor tuple it was measured with)
        self._last_ancilla: dict[Coord, tuple[int, tuple[Coord, ...]]] = {}

    def qi(self, coord: Coord) -> int:
        idx = self._qubit_index.get(coord)
        if idx is None:
            idx = len(self._qubit_index)
            self._qubit_index[coord] = idx
            self.circuit.append("QUBIT_COORDS", [idx], list(coord))
        return idx

    def reset(self, coords: frozenset[Coord] | set[Coord], basis: Basis = "Z") -> None:
        if not coords:
            return
        op = "R" if basis == "Z" else "RX"
        self.circuit.append(op, [self.qi(c) for c in sorted(coords)])

    def tick(self) -> None:
        self.circuit.append("TICK")

    def h(self, coords: list[Coord]) -> None:
        if coords:
            self.circuit.append("H", [self.qi(c) for c in coords])

    def cx_pairs(self, pairs: list[tuple[Coord, Coord]]) -> None:
        if not pairs:
            return
        idxs: list[int] = []
        for a, b in pairs:
            idxs.append(self.qi(a))
            idxs.append(self.qi(b))
        self.circuit.append("CX", idxs)

    def _rec(self, absolute_index: int):
        return stim.target_rec(absolute_index - self._mcount)

    def measure_round(
        self, ancillas: dict[Coord, tuple[str, tuple[Coord, ...]]], deterministic_type: str = "X"
    ) -> None:
        """One syndrome-extraction round: H / 4xCX / H / MR, plus detectors,
        for exactly the ancillas given (their neighbor sets define the
        current region shape — pre-merge, merged, or post-split).

        `deterministic_type` names which ancilla type has a known (round-one
        deterministic) outcome when its neighbor set is entirely fresh —
        "X" when data qubits are RX-reset (|+>_L, our convention), "Z" when
        they're R-reset (|0>_L)."""
        x_ancillas = sorted(c for c, (t, _) in ancillas.items() if t == "X")
        z_ancillas = sorted(c for c, (t, _) in ancillas.items() if t == "Z")
        ordered = sorted(ancillas.keys())

        self.tick()
        self.h(x_ancillas)
        for offset_idx in range(4):
            pairs: list[tuple[Coord, Coord]] = []
            for c in ordered:
                atype, neighbors = ancillas[c]
                offsets = _X_OFFSETS if atype == "X" else _Z_OFFSETS
                # find which neighbor corresponds to this offset slot
                target = None
                dx_dy = offsets[offset_idx]
                candidate = (c[0] + dx_dy[0], c[1] + dx_dy[1])
                if candidate in neighbors:
                    target = candidate
                if target is None:
                    continue
                pairs.append((c, target) if atype == "X" else (target, c))
            self.tick()
            self.cx_pairs(pairs)
        self.tick()
        self.h(x_ancillas)

        self.tick()
        self.circuit.append("MR", [self.qi(c) for c in ordered])
        this_round: dict[Coord, int] = {}
        for i, c in enumerate(ordered):
            this_round[c] = self._mcount + i
        self._mcount += len(ordered)

        for c in ordered:
            atype, neighbors = ancillas[c]
            here = self._rec(this_round[c])
            prev = self._last_ancilla.get(c)
            if prev is not None and prev[1] == neighbors:
                self.circuit.append(
                    "DETECTOR", [here, self._rec(prev[0])], list(c) + [0]
                )
            elif atype == deterministic_type:
                # Freshly formed ancilla of the "matching" type: all its data
                # neighbors are in a known eigenstate (either untouched since
                # patch reset, or just-reset routing qubits), so round one is
                # deterministic — exactly like round 1 of a plain memory
                # experiment for whichever basis the data qubits started in.
                self.circuit.append("DETECTOR", [here], list(c) + [0])
            # Freshly formed ancilla of the *other* type: first outcome is
            # genuinely random, exactly like round 1 of a plain memory
            # experiment's off-basis stabilizers. No detector.
            self._last_ancilla[c] = (this_round[c], neighbors)

    def measure_data_destructive(self, coords: tuple[Coord, ...], basis: Basis = "Z") -> dict[Coord, int]:
        """Destructive measurement of data qubits (used for the surgery
        split, and for the very end of the circuit)."""
        if not coords:
            return {}
        op = "M" if basis == "Z" else "MX"
        self.circuit.append(op, [self.qi(c) for c in coords])
        offsets = {c: self._mcount + i for i, c in enumerate(coords)}
        self._mcount += len(coords)
        return offsets

    def close_out_ancillas(
        self,
        ancillas: dict[Coord, tuple[str, tuple[Coord, ...]]],
        data_offsets: dict[Coord, int],
        closeable_type: str = "X",
    ) -> None:
        """Final-round style detector: recompute each ancilla's expected
        value from the just-measured data qubits and compare to its last
        live syndrome-round outcome. Only the ancilla type matching the
        final measurement's basis can be recomputed this way (an X-type
        stabilizer can't be recomputed from Z-basis data, and vice versa)."""
        for c, (atype, neighbors) in sorted(ancillas.items()):
            if atype != closeable_type:
                continue
            prev = self._last_ancilla.get(c)
            if prev is None or not all(n in data_offsets for n in neighbors):
                continue
            terms = [self._rec(data_offsets[n]) for n in neighbors]
            terms.append(self._rec(prev[0]))
            self.circuit.append("DETECTOR", terms, list(c) + [1])

    def last_ancilla_record(self, coord: Coord) -> int:
        return self._last_ancilla[coord][0]

    def observable_include(
        self,
        index: int,
        coords: tuple[Coord, ...],
        data_offsets: dict[Coord, int],
        extra_absolute_records: tuple[int, ...] = (),
    ) -> None:
        terms = [self._rec(data_offsets[c]) for c in coords]
        terms += [self._rec(i) for i in extra_absolute_records]
        self.circuit.append("OBSERVABLE_INCLUDE", terms, [index])

    def noise(self, p: float) -> None:
        if p <= 0:
            return
        # Applied as a blanket single/two-qubit depolarizing pass is left to
        # the caller via stim's circuit-level noise channels for simplicity;
        # kept as a hook for future per-gate noise injection.
        raise NotImplementedError


def build_lattice_surgery_circuit(
    config: SurgeryConfig,
    pre_merge_rounds: int = 1,
    post_split_rounds: int = 1,
    p: float = 0.001,
) -> stim.Circuit:
    """Build the merge/split circuit for `config`.

    Schedule (design doc §5.1): measure individual stabilizers -> prepare
    routing qubits -> measure joint stabilizers for `config.merge_rounds`
    rounds -> split (measure routing qubits) -> resume individual.
    """
    if config.patch_a.distance != config.patch_b.distance:
        raise NotImplementedError(
            "unequal patch distances not yet supported — the merged region "
            "must currently be a clean rectangle; see design doc §5.1/H3"
        )
    d = config.patch_a.distance
    w = config.routing.width

    region_a = _rect_region(width=d, height=d, x_offset=0, y_offset=0)
    b_x_offset = 2 * (d + w)
    region_b = _rect_region(width=d, height=d, x_offset=b_x_offset, y_offset=0)
    region_merged = _rect_region(width=2 * d + w, height=d, x_offset=0, y_offset=0)
    routing_data = tuple(sorted(region_merged.data - region_a.data - region_b.data))
    assert len(routing_data) == w * d

    # Patches are prepared in |+>_L (X eigenstate), not |0>_L. This matters:
    # Z_L (a horizontal chain touching the left/right boundary) is exactly
    # the boundary this merge operates on, so it does *not* survive the
    # merge as a well-defined observable — measuring the new joint ancillas
    # that appear at the seam (e.g. one spanning one A-boundary qubit and
    # one routing qubit) anticommutes with it. X_L (the left/right-column
    # chain touching the untouched top/bottom boundary) is disjoint from
    # every qubit the merge ever touches, so it stays well-defined
    # throughout and is what we use as the spacelike observable. The
    # merge/split itself still reads out Z_A(x)Z_B via a Z-basis
    # measurement of the routing qubits — that's what "Z-type merge" means,
    # independent of the data qubits' own preparation basis.
    left_a = _left_column(d, x_offset=0, y_offset=0)

    em = _Emitter()
    em.reset(set(region_a.data) | set(region_b.data), basis="X")
    em.reset(set(region_a.ancillas) | set(region_b.ancillas), basis="Z")

    for _ in range(pre_merge_rounds):
        em.measure_round({**region_a.ancillas, **region_b.ancillas}, deterministic_type="X")

    em.reset(set(routing_data), basis="X")
    for _ in range(config.merge_rounds):
        em.measure_round(region_merged.ancillas, deterministic_type="X")

    # Newly-formed X-type ancillas at the seam only need to correct the raw
    # routing-qubit parity if their own stabilizer *anticommutes* with it —
    # i.e. touches an odd number of routing qubits (a weight-2 ancilla
    # spanning exactly one old-patch qubit and one routing qubit, like the
    # (6,0)/(8,6) corners for d=3). A seam ancilla with even overlap (a bulk
    # weight-4 one straddling two routing qubits) already commutes and must
    # NOT be included — its own first-round outcome is still unclosed/random,
    # so folding it in would inject fresh noise rather than remove it.
    routing_set = set(routing_data)
    new_x_boundary_ancillas = [
        c
        for c, (atype, neighbors) in region_merged.ancillas.items()
        if atype == "X"
        and c not in region_a.ancillas
        and c not in region_b.ancillas
        and len(set(neighbors) & routing_set) % 2 == 1
    ]
    correction = tuple(em.last_ancilla_record(c) for c in new_x_boundary_ancillas)

    split_offsets = em.measure_data_destructive(routing_data, basis="Z")
    em.observable_include(1, routing_data, split_offsets, extra_absolute_records=correction)

    for _ in range(post_split_rounds):
        em.measure_round({**region_a.ancillas, **region_b.ancillas}, deterministic_type="X")

    final_data = tuple(sorted(region_a.data | region_b.data))
    final_offsets = em.measure_data_destructive(final_data, basis="X")
    em.close_out_ancillas({**region_a.ancillas, **region_b.ancillas}, final_offsets, closeable_type="X")
    em.observable_include(0, left_a, final_offsets)

    return _add_noise(em.circuit, p)


def _add_noise(circuit: stim.Circuit, p: float) -> stim.Circuit:
    """Apply uniform depolarizing noise: DEPOLARIZE1 after single-qubit
    gates/idles, DEPOLARIZE2 after two-qubit gates, and measurement/reset
    flip noise — by rebuilding the circuit instruction-by-instruction."""
    if p <= 0:
        return circuit
    noisy = stim.Circuit()
    for inst in circuit:
        noisy.append(inst)
        name = inst.name
        targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if name == "CX":
            noisy.append("DEPOLARIZE2", targets, p)
        elif name in ("H",):
            noisy.append("DEPOLARIZE1", targets, p)
        elif name in ("R", "RX"):
            noisy.append("X_ERROR" if name == "R" else "Z_ERROR", targets, p)
        elif name in ("M", "MR"):
            pass  # measurement error is injected via the M/MR instruction's own arg instead
    return _rebuild_with_measurement_noise(noisy, p)


def _rebuild_with_measurement_noise(circuit: stim.Circuit, p: float) -> stim.Circuit:
    out = stim.Circuit()
    for inst in circuit:
        if inst.name in ("M", "MR"):
            targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            out.append(inst.name, targets, p)
        else:
            out.append(inst)
    return out


def load_generated_circuit(path: str | Path) -> stim.Circuit:
    """Load a `.stim` circuit produced by `tqec_backend.py` under the
    separate `.venv-tqec` environment. Stim's text format is portable
    across Stim installs/Python versions, so this is a plain file read —
    no cross-venv call needed at data-generation time."""
    return stim.Circuit.from_file(str(path))
