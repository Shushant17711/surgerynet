"""Shot sampling and detection-event extraction (design doc §5.1, Task 3.1).

Samples shots from a pre-generated `.stim` circuit (via
circuits/tqec_backend.py, run separately under `.venv-tqec`), and persists
raw per-shot detection events plus the spacelike/timelike logical outcomes
to a single `.npz` file per (configuration, physical error rate) pair.

Note on `sinter`: it's the right tool for *aggregate* Monte Carlo (adaptive
shot/error stopping, parallel workers across many circuits) as used in
circuits/validate.py's threshold checks, but it reports statistics, not raw
per-shot detector bits — and raw per-shot syndromes are exactly what graph
construction (data/to_graph.py) needs. So this module samples directly via
`stim.Circuit.compile_detector_sampler()`, which is more than fast enough
at the shot counts this project needs on a laptop (design doc §10).

Usage:
    .venv/bin/python -m data.generate --circuit circuits/generated/k1_spacelike.stim \\
        --shots 100000 --out data/raw/k1_spacelike.npz
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim

# If the raw (undecoded) observable flip rate is this close to 0.5, no
# decoder can possibly do much better than chance on the data -- almost
# always a sign the circuit is being run well above its own threshold
# (e.g. too many rounds for the chosen p, or p chosen for a different,
# shorter circuit) rather than a property of the decoder being evaluated.
# Caught the hard way: three multi-seed "credibility" runs were produced,
# and reported, at p=0.008 on a ~15-round lattice-surgery circuit whose
# actual threshold (measured earlier, separately, and never cross-checked
# against this choice) was ~0.0035 -- every surgery-side number in that
# run was uninterpretable as a result, not because any code was broken.
NEAR_CHANCE_THRESHOLD = 0.4


@dataclass(frozen=True, slots=True)
class ShotBatch:
    """One sampled batch: `detections` is (shots, num_detectors) bool,
    `observables` is (shots, num_observables) bool. Kept together (rather
    than as separate return values) so callers can't accidentally
    transpose or misalign them when saving/loading."""

    detections: np.ndarray
    observables: np.ndarray
    detector_coordinates: dict[int, tuple[float, ...]]

    def __post_init__(self) -> None:
        if self.detections.shape[0] != self.observables.shape[0]:
            raise ValueError(
                f"shot count mismatch: {self.detections.shape[0]} detection rows "
                f"vs {self.observables.shape[0]} observable rows"
            )

    def near_chance_rate(self) -> float | None:
        """Max over observable columns of |mean flip rate - 0.5|, inverted
        so a *small* return value means *close to chance* — or None if
        there are no observables to check."""
        if self.observables.shape[1] == 0:
            return None
        return float(np.abs(self.observables.mean(axis=0) - 0.5).min())


def sample_shots(
    circuit: stim.Circuit, shots: int, seed: int | None = None, warn_near_chance: bool = True
) -> ShotBatch:
    """Direct stim sampling of raw per-shot detector/observable bits —
    see the module docstring for why this isn't done via `sinter` here.

    `warn_near_chance=True` (default) prints a loud warning, not an
    exception, if any observable's raw flip rate sits within
    `NEAR_CHANCE_THRESHOLD` of 0.5 — training/decoding will proceed, but
    nothing can learn from labels indistinguishable from a coin flip, so
    silence here would hide the single most diagnostic number available
    before spending any compute on training."""
    sampler = circuit.compile_detector_sampler(seed=seed)
    detections, observables = sampler.sample(shots=shots, separate_observables=True)
    batch = ShotBatch(
        detections=detections,
        observables=observables,
        detector_coordinates=circuit.get_detector_coordinates(),
    )
    if warn_near_chance:
        gap = batch.near_chance_rate()
        if gap is not None and gap < (0.5 - NEAR_CHANCE_THRESHOLD):
            flip_rates = observables.mean(axis=0).tolist()
            warnings.warn(
                f"observable flip rate(s) {flip_rates} are within "
                f"{0.5 - NEAR_CHANCE_THRESHOLD:.2f} of chance (0.5) at this p — "
                "no decoder can learn much from labels this close to a coin flip; "
                "check that p is below this circuit's actual threshold before "
                "trusting any downstream result",
                stacklevel=2,
            )
    return batch


def save_shots(batch: ShotBatch, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # detector_coordinates keys are ints, values are tuples of floats of
    # possibly-varying length (2 or 3) -- store as a padded float array
    # plus the index list, rather than relying on np.savez's object pickling.
    det_indices = np.array(sorted(batch.detector_coordinates), dtype=np.int64)
    max_dims = max((len(v) for v in batch.detector_coordinates.values()), default=0)
    coords = np.zeros((len(det_indices), max_dims), dtype=np.float64)
    for row, idx in enumerate(det_indices):
        v = batch.detector_coordinates[int(idx)]
        coords[row, : len(v)] = v
    np.savez_compressed(
        path,
        detections=batch.detections,
        observables=batch.observables,
        detector_indices=det_indices,
        detector_coordinates=coords,
    )


def load_shots(path: str | Path) -> ShotBatch:
    with np.load(path) as f:
        det_indices = f["detector_indices"]
        coords = f["detector_coordinates"]
        detector_coordinates = {
            int(idx): tuple(coords[row]) for row, idx in enumerate(det_indices)
        }
        return ShotBatch(
            detections=f["detections"],
            observables=f["observables"],
            detector_coordinates=detector_coordinates,
        )


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit", type=Path, required=True, help="path to a .stim file")
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    circuit = stim.Circuit.from_file(str(args.circuit))
    batch = sample_shots(circuit, shots=args.shots, seed=args.seed)
    save_shots(batch, args.out)
    print(
        f"wrote {args.out}: {batch.detections.shape[0]} shots, "
        f"{batch.detections.shape[1]} detectors, {batch.observables.shape[1]} observables"
    )


if __name__ == "__main__":
    _cli()
