import stim

from baselines.mwpm import MWPMBaseline
from circuits.validate import calibration_control


def test_mwpm_baseline_matches_validate_calibration_numbers():
    """Task 5.1's own bar: this wrapper's decoded logical error rate must
    match circuits/validate.py's calibration_control (same decode path —
    PyMatching against the circuit's own DEM — just exercised through the
    reusable baseline class instead of validate.py's inline helper)."""
    d, p, shots, seed = 3, 0.005, 20_000, 0
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=d, after_clifford_depolarization=p
    )

    reference_rate = calibration_control(distances=(d,), p=p, shots=shots)[d]

    baseline = MWPMBaseline(circuit)
    sampler = circuit.compile_detector_sampler(seed=seed)
    detections, observables = sampler.sample(shots=shots, separate_observables=True)
    preds = baseline.decode(detections)
    baseline_rate = (preds != observables[:, 0]).mean()

    assert abs(baseline_rate - reference_rate) < 0.01
