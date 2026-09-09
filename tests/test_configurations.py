from pathlib import Path

import pytest

from circuits.configurations import (
    distance_to_k,
    held_out_split,
    k_to_distance,
    sweep_configurations,
)

TQEC_PYTHON = Path(__file__).resolve().parent.parent / ".venv-tqec" / "bin" / "python"


def test_distance_k_roundtrip():
    for k in range(1, 6):
        d = k_to_distance(k)
        assert distance_to_k(d) == k


def test_distance_to_k_rejects_even_or_small():
    with pytest.raises(ValueError):
        distance_to_k(4)
    with pytest.raises(ValueError):
        distance_to_k(1)


def test_sweep_produces_full_cross_product():
    swept = sweep_configurations(
        distances=(3, 5), merge_round_multipliers=(1.0, 2.0), routing_widths=(1, 2)
    )
    assert len(swept) == 2 * 2 * 2
    for sc in swept:
        assert sc.config.patch_a.distance == k_to_distance(sc.k)


def test_held_out_split_is_disjoint_and_nonempty():
    swept = sweep_configurations(distances=(3, 5, 7), merge_round_multipliers=(1.0, 1.5))
    train, held_out = held_out_split(swept, train_fraction=0.7, seed=0)
    assert train and held_out
    train_ids = {id(sc) for sc in train}
    held_out_ids = {id(sc) for sc in held_out}
    assert train_ids.isdisjoint(held_out_ids)
    assert len(train) + len(held_out) == len(swept)


@pytest.mark.skipif(not TQEC_PYTHON.exists(), reason=".venv-tqec not set up")
def test_every_swept_configuration_builds_a_tqec_block_graph(tmp_path):
    import subprocess

    swept = sweep_configurations(distances=(3, 5), merge_round_multipliers=(1.0,))
    for sc in swept:
        out = tmp_path / f"k{sc.k}.stim"
        subprocess.run(
            [
                str(TQEC_PYTHON),
                str(Path(__file__).resolve().parent.parent / "circuits" / "tqec_backend.py"),
                "--distance-k", str(sc.k),
                "--p", "0.001",
                "--kind", "spacelike",
                "--out", str(out),
            ],
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        assert out.exists()
