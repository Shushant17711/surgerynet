import numpy as np
import stim
from collections import defaultdict

from baselines.mwpm import MWPMBaseline
from baselines.union_find import BOUNDARY, UnionFindBaseline


def _hand_built_chain_decoder() -> UnionFindBaseline:
    """0 -- 1 --(flips obs 0)-- 2 -- BOUNDARY, hand-verified in the
    module's own development notes: firing only detector 1 has a unique
    correction path (1->2->boundary) that must flip the observable."""
    decoder = UnionFindBaseline.__new__(UnionFindBaseline)
    decoder.num_detectors = 3
    decoder.edge_endpoints = [(0, 1), (1, 2), (2, BOUNDARY)]
    decoder.edge_obs_mask = [frozenset(), frozenset({0}), frozenset()]
    decoder.adjacency = defaultdict(list)
    for edge_id, (u, v) in enumerate(decoder.edge_endpoints):
        decoder.adjacency[u].append((v, edge_id))
        decoder.adjacency[v].append((u, edge_id))
    return decoder


def test_hand_built_chain_firing_middle_detector_flips_observable():
    decoder = _hand_built_chain_decoder()
    row = np.array([False, True, False])
    assert decoder.decode(row) == True


def test_hand_built_chain_no_defects_never_flips():
    decoder = _hand_built_chain_decoder()
    row = np.array([False, False, False])
    assert decoder.decode(row) == False


def test_hand_built_chain_firing_endpoint_matches_directly_to_boundary():
    """Firing only detector 2: the unique short correction is 2->boundary
    (edge with no observable flip)."""
    decoder = _hand_built_chain_decoder()
    row = np.array([False, False, True])
    assert decoder.decode(row) == False


def _small_circuit(distance: int = 3, p: float = 0.01) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=distance, rounds=distance, after_clifford_depolarization=p
    )


def test_interface_parity_with_mwpm_same_shapes():
    circuit = _small_circuit()
    mwpm = MWPMBaseline(circuit)
    uf = UnionFindBaseline(circuit)

    sampler = circuit.compile_detector_sampler(seed=0)
    detections, _ = sampler.sample(shots=20, separate_observables=True)

    mwpm_preds = mwpm.decode(detections)
    uf_preds = uf.decode(detections)
    assert mwpm_preds.shape == uf_preds.shape == (20,)
    assert mwpm_preds.dtype == uf_preds.dtype == bool

    single_mwpm = mwpm.decode(detections[0])
    single_uf = uf.decode(detections[0])
    assert isinstance(single_mwpm, (bool, np.bool_))
    assert isinstance(single_uf, (bool, np.bool_))


def test_union_find_decoded_error_rate_is_reasonable_and_improves_with_distance():
    """Not expected to beat MWPM, but should be a *valid* decoder: bounded
    error rate, and the same qualitative below-threshold improvement with
    distance that circuits/validate.py checks for MWPM."""
    rates = {}
    for d in (3, 5):
        circuit = _small_circuit(distance=d, p=0.002)
        uf = UnionFindBaseline(circuit)
        sampler = circuit.compile_detector_sampler(seed=1)
        detections, observables = sampler.sample(shots=4000, separate_observables=True)
        preds = uf.decode(detections)
        rates[d] = (preds != observables[:, 0]).mean()

    assert 0.0 <= rates[3] < 0.5
    assert 0.0 <= rates[5] < 0.5
    assert rates[5] < rates[3]
