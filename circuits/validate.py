"""THE week-5 gate (design doc §5.1 "Validation checkpoint, non-negotiable",
§9 week 5, §11 "Circuit construction eats the semester").

Runs PyMatching against the circuit's detector error model across a small
distance sweep at a fixed physical error rate, and checks the one thing
that actually matters: does the *decoded* logical error rate go down as
distance goes up? (Not the raw physical-error observable-flip rate — that
one increases with distance/qubit count regardless of code quality, and
conflating the two was a real bug caught during development here; see
tests/test_tqec_backend.py's docstring and the module comment below.)

Two checks, in order:
  1. Calibration control: reproduce the expected sub-threshold trend on a
     *plain memory* circuit generated directly by stim (no lattice surgery
     involved). If this fails, the validation *methodology* itself is
     broken and the surgery check below is meaningless.
  2. The actual gate: the same check applied to the lattice surgery
     circuit, for both the spacelike and timelike observables, generated
     via circuits/tqec_backend.py (which needs the separate .venv-tqec
     environment — see that module's docstring).

Usage (run with the main venv, which has stim + pymatching):
    .venv/bin/python circuits/validate.py
    .venv/bin/python circuits/validate.py --calibration-only
    .venv/bin/python circuits/validate.py --p 0.003 --ks 1 2 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pymatching
import stim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TQEC_PYTHON = PROJECT_ROOT / ".venv-tqec" / "bin" / "python"


def decoded_logical_error_rates(circuit: stim.Circuit, shots: int) -> np.ndarray:
    """The metric that actually matters: fraction of shots where a real
    decoder's correction disagrees with the true observable flip. One rate
    per OBSERVABLE_INCLUDE index in the circuit."""
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    preds = matcher.decode_batch(dets)
    return (preds != obs).mean(axis=0)


def calibration_control(
    distances: tuple[int, ...] = (3, 5), p: float = 0.005, shots: int = 20_000
) -> dict[int, float]:
    """Known-good reference: a plain rotated-memory circuit from stim's own
    generator. Proves the *validation methodology* works before trusting it
    on the surgery circuit."""
    rates: dict[int, float] = {}
    for d in distances:
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d,
            rounds=d,
            after_clifford_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p,
        )
        rates[d] = float(decoded_logical_error_rates(circuit, shots)[0])
    return rates


def _generate_surgery_circuit(k: int, p: float, kind: str, out_path: Path) -> None:
    if not TQEC_PYTHON.exists():
        raise RuntimeError(
            f"{TQEC_PYTHON} not found — set up the TQEC backend first "
            "(see circuits/tqec_backend.py's module docstring)"
        )
    subprocess.run(
        [
            str(TQEC_PYTHON),
            str(PROJECT_ROOT / "circuits" / "tqec_backend.py"),
            "--distance-k", str(k),
            "--p", str(p),
            "--kind", kind,
            "--out", str(out_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def surgery_gate(
    ks: tuple[int, ...] = (1, 2), p: float = 0.00259, kind: str = "spacelike", shots: int = 20_000
) -> dict[int, float]:
    rates: dict[int, float] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for k in ks:
            out_path = Path(tmp) / f"{kind}_k{k}.stim"
            _generate_surgery_circuit(k, p, kind, out_path)
            circuit = stim.Circuit.from_file(str(out_path))
            rates[k] = float(decoded_logical_error_rates(circuit, shots)[0])
    return rates


def _is_strictly_decreasing(values: list[float]) -> bool:
    return all(a > b for a, b in zip(values, values[1:]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--p", type=float, default=0.00259)
    parser.add_argument("--shots", type=int, default=20_000)
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()

    print(f"[1/3] Calibration control: plain memory circuit, p={args.p}, distances={args.distances}")
    calib = calibration_control(tuple(args.distances), p=args.p, shots=args.shots)
    for d, rate in calib.items():
        print(f"      d={d}: decoded logical error rate = {rate:.4f}")
    if not _is_strictly_decreasing(list(calib.values())):
        print("FAIL: calibration control did not show decreasing error rate with distance.")
        print("      The validation methodology itself is broken — do not trust anything below.")
        return 1
    print("      OK — methodology validated against a known-good circuit.")

    if args.calibration_only:
        return 0

    overall_ok = True
    for kind in ("spacelike", "timelike"):
        print(f"\n[2/3] Surgery gate ({kind}): TQEC-backed circuit, p={args.p}, ks={args.ks}")
        try:
            rates = surgery_gate(tuple(args.ks), p=args.p, kind=kind, shots=args.shots)
        except RuntimeError as e:
            print(f"FAIL: {e}")
            return 1
        for k, rate in rates.items():
            print(f"      k={k} (d={2 * k + 1}): decoded logical error rate = {rate:.4f}")
        if _is_strictly_decreasing(list(rates.values())):
            print(f"      OK — {kind} observable improves with distance at p={args.p}.")
        else:
            print(f"      FAIL — {kind} observable does NOT improve with distance at p={args.p}.")
            print("      Either p is above threshold for this observable, or the circuit is wrong.")
            overall_ok = False

    print("\n[3/3] Gate result:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
