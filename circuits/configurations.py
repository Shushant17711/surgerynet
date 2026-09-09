"""Configuration sweep generator (design doc §5.1, E6/H3).

Enumerates `SurgeryConfig` instances across patch distance, merge duration,
and routing geometry, and splits them into disjoint train/held-out sets for
E6 (configuration randomization). Every sampled configuration must be
buildable by circuits/tqec_backend.py's block-graph construction without
raising — that's the correctness bar this module's own test checks.

Note: `circuits/tqec_backend.py`'s `k` parameter scales patch distance *and*
merge/round duration together (`d = 2k + 1`), matching how TQEC's own
`block_temporal_height` ties round count to `k`. `RoutingConfig.width` is
only meaningful to the hand-rolled builder (`circuits/lattice_surgery.py`) —
TQEC's block graph has no separate routing-width knob in this version, so
sweeping it here is forward-looking (useful once/if the hand-rolled builder
or a future TQEC convention exposes it) rather than immediately exercised
by the TQEC backend.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from schema import PatchConfig, RoutingConfig, SurgeryConfig


def distance_to_k(distance: int) -> int:
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance must be odd and >= 3, got {distance}")
    return (distance - 1) // 2


def k_to_distance(k: int) -> int:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return 2 * k + 1


@dataclass(frozen=True, slots=True)
class SweptConfig:
    """A SurgeryConfig plus the TQEC scaling parameter it maps to, so
    downstream code doesn't have to re-derive `k` from `config.max_distance`."""

    config: SurgeryConfig
    k: int


def sweep_configurations(
    distances: tuple[int, ...] = (3, 5),
    merge_round_multipliers: tuple[float, ...] = (1.0,),
    routing_widths: tuple[int, ...] = (1,),
) -> list[SweptConfig]:
    """Enumerate the full cross-product of distance x merge-duration x
    routing-width as SurgeryConfigs. merge_round_multipliers scales
    merge_rounds relative to distance (e.g. 1.0 -> merge_rounds == distance,
    matching the design doc's "hold for d rounds" default)."""
    out: list[SweptConfig] = []
    for d in distances:
        k = distance_to_k(d)
        for mult in merge_round_multipliers:
            merge_rounds = max(1, round(d * mult))
            for w in routing_widths:
                cfg = SurgeryConfig(
                    patch_a=PatchConfig(distance=d),
                    patch_b=PatchConfig(distance=d),
                    routing=RoutingConfig(width=w),
                    merge_rounds=merge_rounds,
                )
                out.append(SweptConfig(config=cfg, k=k))
    return out


def held_out_split(
    configs: list[SweptConfig], train_fraction: float = 0.7, seed: int = 0
) -> tuple[list[SweptConfig], list[SweptConfig]]:
    """Disjoint train/held-out split for E6's configuration-generalization
    gap. Splits by *distinct configuration*, not by shot, so held-out
    configs are genuinely unseen during training."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    shuffled = list(configs)
    random.Random(seed).shuffle(shuffled)
    n_train = max(1, round(len(shuffled) * train_fraction))
    n_train = min(n_train, len(shuffled) - 1) if len(shuffled) > 1 else n_train
    return shuffled[:n_train], shuffled[n_train:]
