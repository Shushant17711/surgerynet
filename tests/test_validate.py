from circuits.validate import _is_strictly_decreasing, calibration_control


def test_is_strictly_decreasing():
    assert _is_strictly_decreasing([0.1, 0.05, 0.01])
    assert not _is_strictly_decreasing([0.1, 0.1, 0.01])
    assert not _is_strictly_decreasing([0.01, 0.05, 0.1])
    assert _is_strictly_decreasing([0.1])
    assert _is_strictly_decreasing([])


def test_calibration_control_reproduces_below_threshold_trend():
    """The validation methodology's own self-check (design doc §5.1): a
    plain memory circuit well below threshold must show decoded logical
    error rate decreasing with distance."""
    rates = calibration_control(distances=(3, 5), p=0.002, shots=20_000)
    assert rates[5] < rates[3]
