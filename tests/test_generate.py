import stim

from data.generate import ShotBatch, load_shots, sample_shots, save_shots


def _small_circuit() -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=2, after_clifford_depolarization=0.005
    )


def test_sample_shots_shape_matches_circuit():
    circuit = _small_circuit()
    batch = sample_shots(circuit, shots=50, seed=0)
    assert batch.detections.shape == (50, circuit.num_detectors)
    assert batch.observables.shape == (50, circuit.num_observables)
    assert len(batch.detector_coordinates) == circuit.num_detectors


def test_shot_batch_rejects_mismatched_shot_counts():
    import numpy as np

    try:
        ShotBatch(detections=np.zeros((5, 3)), observables=np.zeros((4, 1)), detector_coordinates={})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_save_and_load_round_trips(tmp_path):
    circuit = _small_circuit()
    batch = sample_shots(circuit, shots=30, seed=1)
    path = tmp_path / "shots.npz"
    save_shots(batch, path)
    loaded = load_shots(path)

    assert (loaded.detections == batch.detections).all()
    assert (loaded.observables == batch.observables).all()
    assert loaded.detector_coordinates.keys() == batch.detector_coordinates.keys()
    for idx, coord in batch.detector_coordinates.items():
        assert tuple(float(v) for v in coord) == tuple(float(v) for v in loaded.detector_coordinates[idx])
