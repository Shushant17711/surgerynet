import pytest

from schema import (
    DENSITY_SLICE,
    NUM_NODE_FEATURES,
    PHASE_SLICE,
    REGION_SLICE,
    STABILIZER_TYPE_SLICE,
    SURGERY_ONLY_SLICES,
    XY_SLICE,
    OUTPUT_HEADS,
    PatchConfig,
    RoutingConfig,
    SurgeryConfig,
)


def test_slices_are_contiguous_and_cover_the_feature_vector():
    slices = [XY_SLICE, slice(2, 3), STABILIZER_TYPE_SLICE, REGION_SLICE, PHASE_SLICE, DENSITY_SLICE]
    cursor = 0
    for s in slices:
        assert s.start == cursor
        cursor = s.stop
    assert cursor == NUM_NODE_FEATURES


def test_surgery_only_slices_are_region_and_phase_only():
    assert SURGERY_ONLY_SLICES == (REGION_SLICE, PHASE_SLICE)


def test_output_heads_are_spacelike_and_timelike():
    assert OUTPUT_HEADS == ("spacelike", "timelike")


def test_patch_config_rejects_even_or_small_distance():
    PatchConfig(distance=3)
    with pytest.raises(ValueError):
        PatchConfig(distance=4)
    with pytest.raises(ValueError):
        PatchConfig(distance=1)


def test_routing_config_rejects_nonpositive_width():
    RoutingConfig(width=1)
    with pytest.raises(ValueError):
        RoutingConfig(width=0)


def test_surgery_config_max_distance():
    cfg = SurgeryConfig(
        patch_a=PatchConfig(distance=3),
        patch_b=PatchConfig(distance=5),
        routing=RoutingConfig(width=1),
        merge_rounds=3,
    )
    assert cfg.max_distance == 5
