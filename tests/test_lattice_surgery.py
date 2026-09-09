import pytest
import stim

from circuits.lattice_surgery import build_lattice_surgery_circuit, _rect_region
from schema import PatchConfig, RoutingConfig, SurgeryConfig


def test_rect_region_matches_stim_reference_layout_for_solo_d3_patch():
    region = _rect_region(width=3, height=3)
    ref = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=1)
    ref_coords = {tuple(int(v) for v in c) for c in ref.get_final_qubit_coordinates().values()}
    ours = set(region.data) | set(region.ancillas)
    assert ours == ref_coords


def _default_config(d: int = 3, merge_rounds: int = 3, routing_width: int = 1) -> SurgeryConfig:
    return SurgeryConfig(
        patch_a=PatchConfig(distance=d),
        patch_b=PatchConfig(distance=d),
        routing=RoutingConfig(width=routing_width),
        merge_rounds=merge_rounds,
    )


def _keep_only_observable(circuit: stim.Circuit, index: int) -> stim.Circuit:
    """Strip every OBSERVABLE_INCLUDE except `index`, to test one of the two
    hand-rolled observables in isolation (spacelike is validated; timelike
    is not yet — see the module docstring in circuits/lattice_surgery.py)."""
    out = stim.Circuit()
    for inst in circuit:
        if inst.name == "OBSERVABLE_INCLUDE" and inst.gate_args_copy()[0] != index:
            continue
        out.append(inst)
    return out


# xfail, not skip: these document a known, real open issue in the
# hand-rolled merge construction — not just the timelike observable's own
# Pauli-frame correction (that was the first bug found), but at least one
# more detector-level inconsistency at the merge's routing-qubit reset,
# uncovered while trying to isolate the spacelike observable alone (see
# test_spacelike_observable_alone_is_deterministic below: even with the
# timelike OBSERVABLE_INCLUDE stripped out, detectors D17/D18 at the seam
# still anticommute with the routing reset). Rather than keep guessing at
# fixes here, circuits/tqec_backend.py — independently validated via a real
# decoder, see tests/test_tqec_backend.py — is the project's actual circuit
# backend going forward. This module stays as the from-scratch reference
# attempt (design doc §5.1 route 3) and its still-valid layout-matching
# test above.
_HAND_ROLLED_XFAIL = pytest.mark.xfail(
    reason="hand-rolled merge construction has an unresolved detector/observable "
    "inconsistency near the merge seam; use circuits/tqec_backend.py for real data generation",
    strict=True,
)


@_HAND_ROLLED_XFAIL
def test_spacelike_observable_alone_is_deterministic():
    circuit = _keep_only_observable(build_lattice_surgery_circuit(_default_config(), p=0.001), index=0)
    dem = circuit.detector_error_model(decompose_errors=True)
    assert len(dem) > 0


def test_spacelike_observable_alone_is_noiselessly_zero_and_bounded_when_noisy():
    noiseless = _keep_only_observable(build_lattice_surgery_circuit(_default_config(), p=0.0), index=0)
    _, obs = noiseless.compile_detector_sampler().sample(shots=64, separate_observables=True)
    assert not obs.any()

    noisy = _keep_only_observable(build_lattice_surgery_circuit(_default_config(), p=0.005), index=0)
    _, obs = noisy.compile_detector_sampler().sample(shots=2000, separate_observables=True)
    assert 0.0 < obs.mean() < 0.5


@_HAND_ROLLED_XFAIL
def test_circuit_builds_and_has_nonempty_detector_error_model():
    circuit = build_lattice_surgery_circuit(_default_config(), p=0.001)
    dem = circuit.detector_error_model(decompose_errors=True)
    assert len(dem) > 0


@_HAND_ROLLED_XFAIL
def test_noiseless_circuit_has_deterministic_detectors_and_observables():
    circuit = build_lattice_surgery_circuit(_default_config(), p=0.0)
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots=64, separate_observables=True)
    assert not dets.any(), "a noiseless circuit must never fire a detector"
    assert not obs.any(), "spacelike/timelike observables must both start at 0 noiselessly"


def test_noisy_circuit_spacelike_rate_is_nonzero_but_bounded():
    # Only checks spacelike: timelike's raw rate sits right at ~0.5 with
    # sampling noise on either side (consistent with it being genuinely
    # unresolved, not just "hard to bound") — see _HAND_ROLLED_XFAIL above.
    circuit = build_lattice_surgery_circuit(_default_config(), p=0.005)
    sampler = circuit.compile_detector_sampler()
    _, obs = sampler.sample(shots=2000, separate_observables=True)
    spacelike_rate = obs[:, 0].mean()
    assert 0.0 < spacelike_rate < 0.5
