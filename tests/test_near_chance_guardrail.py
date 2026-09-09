import stim
import warnings

from data.generate import NEAR_CHANCE_THRESHOLD, sample_shots


def test_warns_when_observable_is_near_chance():
    """Regression test for a real, expensive-to-discover bug: p chosen
    above a circuit's actual threshold produces labels indistinguishable
    from a coin flip, and nothing decodes that. This must be surfaced
    loudly, not discovered only after a full multi-seed run and manual
    diagnosis."""
    # A circuit run for many more rounds than its distance can tolerate at
    # this p -- reliably drives the raw flip rate close to 0.5.
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=60, after_clifford_depolarization=0.05
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sample_shots(circuit, shots=2000, seed=0)
    assert any("chance" in str(w.message) for w in caught)


def test_no_warning_for_a_healthy_sub_threshold_circuit():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.001
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sample_shots(circuit, shots=2000, seed=0)
    assert not any("chance" in str(w.message) for w in caught)


def test_warn_can_be_disabled():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=60, after_clifford_depolarization=0.05
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sample_shots(circuit, shots=2000, seed=0, warn_near_chance=False)
    assert not any("chance" in str(w.message) for w in caught)


def test_near_chance_threshold_is_sane():
    assert 0.0 < NEAR_CHANCE_THRESHOLD < 0.5
