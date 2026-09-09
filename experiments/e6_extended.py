"""Extended E6 (H3, design doc §6 E6): `experiments/run_all.py` calls
`e6_config_randomization.run` with `distances=(3, 5)` — only two
configurations, so the 70/30 train/held-out split (`held_out_split`) lands
on exactly 1 train config and 1 held-out config. That is about as thin as
a "configuration generalization" claim can get, and it shows: E6's
reported generalization gap in `results/main_table.md` has a std (±0.10)
nearly half its own mean, i.e. genuinely unstable across seeds, not just
under-sampled (LIMITATIONS.md).

This script reruns E6 with `distances=(3, 5, 7)` (k=1,2,3) instead, giving
`held_out_split` 3 configurations to work with (2 train, 1 held-out at the
default 70% split) — still thin by the standards of a real configuration
sweep, but a real improvement over 2. Reported as a supplementary result
(`results/e6_extended.jsonl`, `results/E6_EXTENDED.md`), not a replacement
for the main table's E6 row (which stays as `run_all.py` produced it, for
a stable comparison point against the rest of that table).
"""

from __future__ import annotations

import argparse
import statistics

from experiments.common import ExperimentResult, append_result
from experiments.e6_config_randomization import run as run_e6


def run(
    p: float = 0.002,
    shots_per_config: int = 15000,
    epochs: int = 20,
    num_seeds: int = 8,
    results_path: str = "results/e6_extended.jsonl",
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    for seed in range(num_seeds):
        print(f"[E6 extended] seed={seed}: distances=(3,5,7)")
        r = run_e6(distances=(3, 5, 7), p=p, shots_per_config=shots_per_config, epochs=epochs, seed=seed)
        append_result(results_path, r)
        results.append(r)
        print(f"    held-out rate={r.logical_error_rate:.4f}  train rate={r.extra['train_configuration_error_rate']:.4f}"
              f"  gap={r.extra['generalization_gap']:.4f}  train_d={r.extra['train_distances']}"
              f"  held_out_d={r.extra['held_out_distances']}")
    return results


def render_summary(results: list[ExperimentResult]) -> str:
    held_out_rates = [r.logical_error_rate for r in results]
    gaps = [r.extra["generalization_gap"] for r in results]
    lines = [
        "# E6 extended: configuration generalization with 3 configs (k=1,2,3)",
        "",
        "Supplementary to `results/main_table.md`'s E6 row, which uses only "
        "distances=(3,5) (1 train config, 1 held-out config -- as thin as this "
        "metric can get). This run uses distances=(3,5,7): 2 train configs, 1 "
        "held-out, still not many, but a real improvement.",
        "",
        f"| metric | mean ± std (n={len(results)}) |",
        "|---|---|",
        f"| held-out logical error rate | {statistics.fmean(held_out_rates):.4f} ± "
        f"{statistics.stdev(held_out_rates) if len(held_out_rates) > 1 else float('nan'):.4f} |",
        f"| generalization gap (held-out minus train) | {statistics.fmean(gaps):.4f} ± "
        f"{statistics.stdev(gaps) if len(gaps) > 1 else float('nan'):.4f} |",
        "",
        "For comparison, `results/main_table.md`'s E6 row (distances=(3,5), "
        "n=8): held-out rate 0.2224 ± 0.1014.",
    ]
    return "\n".join(lines) + "\n"


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=float, default=0.002)
    parser.add_argument("--shots-per-config", type=int, default=15000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--results-path", default="results/e6_extended.jsonl")
    parser.add_argument("--summary-path", default="results/E6_EXTENDED.md")
    args = parser.parse_args()

    results = run(
        p=args.p, shots_per_config=args.shots_per_config, epochs=args.epochs,
        num_seeds=args.num_seeds, results_path=args.results_path,
    )
    summary = render_summary(results)
    with open(args.summary_path, "w") as f:
        f.write(summary)
    print(f"\nwrote {args.summary_path}")
    print(summary)


if __name__ == "__main__":
    _cli()
