"""PyMatching-based MWPM baseline (design doc §5.4, MWPM row: "Yes, given a
correct DEM"). Same decode(detections) -> predictions interface as
baselines/union_find.py, so experiment code can swap between them."""

from __future__ import annotations

import numpy as np
import pymatching
import stim


class MWPMBaseline:
    def __init__(self, circuit: stim.Circuit) -> None:
        self.dem = circuit.detector_error_model(decompose_errors=True)
        self.matcher = pymatching.Matching.from_detector_error_model(self.dem)

    def decode(self, detections: np.ndarray) -> np.ndarray:
        """detections: (shots, num_detectors) bool array (or a single
        (num_detectors,) row). Returns predicted flips for observable 0 —
        this project's circuits carry exactly one observable per circuit
        (see circuits/tqec_backend.py's module docstring)."""
        single = detections.ndim == 1
        rows = detections[None, :] if single else detections
        preds = self.matcher.decode_batch(rows)[:, 0].astype(bool)
        return preds[0] if single else preds
