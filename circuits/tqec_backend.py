"""TQEC-backed lattice surgery circuit construction (design doc §5.1, route 1).

IMPORTANT: this module requires the `tqec` package, which only supports
Python <3.14. It must be run under a *separate* virtualenv
(`.venv-tqec`, built against Python 3.12) from the rest of this project
(`.venv`, Python 3.14, running stim/pymatching/sinter/torch). The two talk
to each other through plain `.stim` circuit files on disk — Stim's text
format is portable across both Stim installs and both Python versions.

Workflow:
    .venv-tqec/bin/python circuits/tqec_backend.py --k 2 --p 0.005 \\
        --kind spacelike --out circuits/generated/k2_spacelike.stim

Then, from the main environment:
    circuit = circuits.lattice_surgery.load_generated_circuit(path)

Why two circuit "kinds" instead of one circuit with both observables:
TQEC's ports (the open ends of the block graph, here In_A/In_B/Out_A/Out_B)
must be capped with a single logical basis to produce a concrete, sampleable
circuit. Capping with X gives a circuit where each patch's own X_L survives
the merge untouched (a clean spacelike observable — verified empirically:
`find_correlation_surfaces()` returns a 4-edge surface spanning exactly one
patch's own time-line). Capping with Z gives a single combined correlation
surface spanning the whole graph (both patches' worth of the merge/split
channel) — a well-defined, decodable observable tied to whether the merge
itself was read out correctly, which is what design doc §5/§7 calls the
timelike/parity-measurement quantity. Both were validated against PyMatching
with a real decoder (not just `detector_error_model()` accepting the
circuit) to show the expected sub-threshold, distance-improves-things trend
before being adopted here — see the session notes / commit history for the
`binary_search_threshold`-based check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema import SurgeryConfig

ObservableKind = Literal["spacelike", "timelike"]


def build_block_graph(config: SurgeryConfig):
    """A two-patch Z-type merge/split block graph: two vertical patch
    columns (pre-merge / merged / post-split cubes) joined by one spatial
    pipe during the merged time slice.

    Only supports equal-distance patches in v1, matching the hand-rolled
    circuit's own limitation (see circuits/lattice_surgery.py) — TQEC's `k`
    parameter scales both patches together.
    """
    from tqec.computation.block_graph import BlockGraph
    from tqec.utils.position import Position3D

    if config.patch_a.distance != config.patch_b.distance:
        raise NotImplementedError(
            "unequal patch distances not yet supported by the TQEC backend"
        )

    g = BlockGraph("Lattice surgery ZZ merge/split")
    nodes = [
        (Position3D(0, 0, 0), "P", "In_A"),
        (Position3D(0, 0, 1), "ZXZ", ""),  # patch A: pre-merge
        (Position3D(0, 0, 2), "ZXZ", ""),  # patch A: merged window
        (Position3D(0, 0, 3), "ZXZ", ""),  # patch A: post-split
        (Position3D(0, 0, 4), "P", "Out_A"),
        (Position3D(1, 0, 0), "P", "In_B"),
        (Position3D(1, 0, 1), "ZXZ", ""),  # patch B: pre-merge
        (Position3D(1, 0, 2), "ZXZ", ""),  # patch B: merged window
        (Position3D(1, 0, 3), "ZXZ", ""),  # patch B: post-split
        (Position3D(1, 0, 4), "P", "Out_B"),
    ]
    for pos, kind, label in nodes:
        g.add_cube(pos, kind, label)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (8, 9)]:
        g.add_pipe(nodes[i][0], nodes[j][0])
    g.add_pipe(Position3D(0, 0, 2), Position3D(1, 0, 2))  # the merge itself
    return g


def _select_observables(g, kind: ObservableKind):
    from tqec.computation.cube import ZXCube
    from tqec.utils.enums import Basis

    if kind == "spacelike":
        g.fill_ports(ZXCube.from_str("ZXX"))
        surfaces = g.find_correlation_surfaces()
        clean = [s for s in surfaces if len(s.span) == 4]
        if len(clean) != 1:
            raise RuntimeError(
                f"expected exactly one 4-edge spacelike surface, found {len(clean)}"
            )
        return clean
    elif kind == "timelike":
        g.fill_ports(ZXCube.from_str("ZXZ"))
        return g.find_correlation_surfaces()
    else:
        raise ValueError(f"unknown observable kind: {kind!r}")


def compile_circuit(config: SurgeryConfig, k: int, p: float, kind: ObservableKind):
    """Build and compile the noisy stim.Circuit for one observable kind."""
    import tqec

    g = build_block_graph(config)
    surfaces = _select_observables(g, kind)
    compiled = tqec.compile_block_graph(g, observables=surfaces)
    noiseless = compiled.generate_stim_circuit(k=k)
    if p <= 0:
        return noiseless
    noise = tqec.NoiseModel.uniform_depolarizing(p=p)
    return noise.noisy_circuit(noiseless)


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance-k", type=int, default=1, help="TQEC scaling parameter k")
    parser.add_argument("--p", type=float, default=0.005, help="uniform depolarizing rate")
    parser.add_argument("--kind", choices=["spacelike", "timelike"], required=True)
    parser.add_argument("--routing-width", type=int, default=1)
    parser.add_argument("--merge-rounds", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from schema import PatchConfig, RoutingConfig

    d = 2 * args.distance_k + 1
    config = SurgeryConfig(
        patch_a=PatchConfig(distance=d),
        patch_b=PatchConfig(distance=d),
        routing=RoutingConfig(width=args.routing_width),
        merge_rounds=args.merge_rounds,
    )
    circuit = compile_circuit(config, k=args.distance_k, p=args.p, kind=args.kind)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(str(circuit))
    print(f"wrote {args.out} ({len(str(circuit).splitlines())} lines, d={d}, kind={args.kind})")


if __name__ == "__main__":
    _cli()
