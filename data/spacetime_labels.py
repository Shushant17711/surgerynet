"""Recovers region/phase/stabilizer-type labels for each detector in a
TQEC-generated circuit (design doc §5.2's region/phase features).

TQEC's compiled circuit carries no explicit "this ancilla is patch A vs
routing" annotation — that structure has to be inferred from the circuit
itself. Two things make this tractable and (empirically checked, see
tests/test_spacetime_labels.py) robust across code distances:

- **Phase / routing region, from time extent.** Ancillas that exist only
  because of the merge (spanning one patch's boundary qubit and the seam)
  are active for a short, *contiguous, interior* span of rounds — they
  don't touch the very first or very last round of the circuit, unlike
  every "always there" bulk/boundary ancilla of a standalone patch. That
  interior span *is* the "merged" phase; timesteps before/after it are
  "pre_merge"/"post_split", and the (x, y) positions that are only active
  during it are the "routing_region" — verified against two different
  code distances (k=1 and k=2) in tests/test_spacetime_labels.py.
- **Stabilizer type, from lattice geometry.** Ancilla (x, y) with
  (x//2 + y//2) odd is X-type, even is Z-type — the same checkerboard rule
  verified against stim's own rotated-memory reference in
  tests/test_lattice_surgery.py. This is a property of the rotated
  surface code's geometry, not of TQEC's specific gate-level
  implementation (CZ/CX vs H/CX), so it should hold regardless of which
  gates TQEC happens to compile a given stabilizer measurement into.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import stim

from schema import PhaseLabel, RegionLabel


@dataclass(frozen=True, slots=True)
class DetectorLabel:
    region: RegionLabel
    phase: PhaseLabel
    stabilizer_type: str  # "X" or "Z"
    x: float
    y: float
    t: float


def _stabilizer_type(x: float, y: float) -> str:
    return "X" if ((int(x) // 2 + int(y) // 2) % 2 == 1) else "Z"


def infer_detector_labels(circuit: stim.Circuit) -> dict[int, DetectorLabel]:
    det_coords = circuit.get_detector_coordinates()
    by_xy: dict[tuple[float, float], set[float]] = defaultdict(set)
    for coord in det_coords.values():
        by_xy[(coord[0], coord[1])].add(coord[2])

    all_t = [t for ts in by_xy.values() for t in ts]
    t_min, t_max = min(all_t), max(all_t)

    seam_xy = {xy for xy, ts in by_xy.items() if min(ts) != t_min and max(ts) != t_max}
    if seam_xy:
        seam_t_min = min(t for xy in seam_xy for t in by_xy[xy])
        seam_t_max = max(t for xy in seam_xy for t in by_xy[xy])
        seam_x_mid = (min(xy[0] for xy in seam_xy) + max(xy[0] for xy in seam_xy)) / 2
    else:
        seam_t_min = seam_t_max = None
        all_x = [xy[0] for xy in by_xy]
        seam_x_mid = (min(all_x) + max(all_x)) / 2

    labels: dict[int, DetectorLabel] = {}
    for idx, coord in det_coords.items():
        x, y, t = coord[0], coord[1], coord[2]
        if (x, y) in seam_xy:
            region: RegionLabel = "routing_region"
        elif x < seam_x_mid:
            region = "patch_A"
        else:
            region = "patch_B"

        if seam_t_min is not None and seam_t_min <= t <= seam_t_max:
            phase: PhaseLabel = "merged"
        elif seam_t_min is not None and t < seam_t_min:
            phase = "pre_merge"
        else:
            phase = "post_split"

        labels[idx] = DetectorLabel(
            region=region, phase=phase, stabilizer_type=_stabilizer_type(x, y), x=x, y=y, t=t
        )
    return labels
