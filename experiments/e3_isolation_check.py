"""Follow-up to the E3 zero-shot-got-worse-after-tuning finding
(LIMITATIONS.md, "A third pass"): after `experiments/hparam_search.py`'s
tuned hyperparameters improved every surgery-*trained* metric (E5, E7, E9)
but made E3's zero-shot transfer measurably *worse* (32% -> 42% error), the
most plausible reading offered was "the tuned architecture specializes
harder to memory-graph statistics" — but that was never isolated from the
other thing that changed at the same time: training scale (3000 shots/8
epochs -> 30000 shots/20 epochs).

This script isolates the two: train at the OLD (untuned) hyperparameters
but the NEW (large) scale, across the same 8 seeds, and evaluate zero-shot
on the same surgery circuit the main run used. Three points to compare:

  old hyperparams, old scale (archived)   -> results/archive_pre_tuning/
  old hyperparams, new scale (this script) -> results/e3_isolation_check.jsonl
  new hyperparams, new scale (main run)    -> results/e3_zeroshot.jsonl

If "old hp, new scale" lands near the old scale's ~32% zero-shot rate, the
regression is attributable to the tuned hyperparameters (architecture), not
scale. If it lands near the new run's ~42%, scale itself is implicated
instead, and the "specializes harder to memory statistics" story from
LIMITATIONS.md needs revising.
"""

from __future__ import annotations

import argparse
import statistics

from experiments.common import ExperimentResult, append_result
from experiments.e2_memory_baseline import run as run_e2
from experiments.e3_zeroshot import evaluate_gnn_zero_shot

# The untuned, design-doc-default hyperparameters used everywhere before
# experiments/hparam_search.py (see LIMITATIONS.md's "first version" story).
OLD_HP = dict(hidden_dim=64, num_layers=4, conv_type="gat", heads=4, lr=1e-3, weight_decay=0.0)


def run(
    distance: int = 3,
    p: float = 0.002,
    train_shots: int = 30000,
    test_shots: int = 30000,
    epochs: int = 20,
    num_seeds: int = 8,
    surgery_circuit_path: str = "results/_k1_spacelike.stim",
    k: int = 1,
    results_path: str = "results/e3_isolation_check.jsonl",
) -> list[ExperimentResult]:
    patch_dimension = float(2 * k + 1)
    results: list[ExperimentResult] = []
    for seed in range(num_seeds):
        print(f"[isolation] seed={seed}: training at OLD hyperparams, NEW scale")
        e2_gnn, _, model = run_e2(
            distance=distance, p=p, train_shots=train_shots, test_shots=test_shots, epochs=epochs,
            seed=seed, **OLD_HP,
        )
        e2_gnn.extra["isolation_variant"] = "old_hp_new_scale"
        append_result(results_path, e2_gnn)
        results.append(e2_gnn)

        e3_result = evaluate_gnn_zero_shot(
            model, surgery_circuit_path, patch_dimension, "spacelike",
            shots=test_shots, seed=seed, p=p, k=k,
        )
        e3_result.extra["isolation_variant"] = "old_hp_new_scale"
        append_result(results_path, e3_result)
        results.append(e3_result)
        print(f"    E2 (memory) rate={e2_gnn.logical_error_rate:.4f}  E3 (zero-shot) rate={e3_result.logical_error_rate:.4f}")
    return results


def render_summary(results: list[ExperimentResult]) -> str:
    e2_rates = [r.logical_error_rate for r in results if r.experiment == "E2"]
    e3_rates = [r.logical_error_rate for r in results if r.experiment == "E3"]
    lines = [
        "# E3 isolation check: architecture (tuning) vs. scale",
        "",
        "Three points, same p=0.002, same 8 seeds where applicable, same surgery circuit:",
        "",
        "| condition | E2 (memory) rate | E3 (zero-shot) rate |",
        "|---|---|---|",
        "| old hyperparams, OLD scale (3000 shots/8 epochs/3 seeds, archived) | 0.0091 ± 0.0042 | 0.3206 ± 0.0356 |",
        f"| old hyperparams, NEW scale (30000 shots/20 epochs/{len(e2_rates)} seeds, this script) "
        f"| {statistics.fmean(e2_rates):.4f} ± {statistics.stdev(e2_rates) if len(e2_rates) > 1 else float('nan'):.4f} "
        f"| {statistics.fmean(e3_rates):.4f} ± {statistics.stdev(e3_rates) if len(e3_rates) > 1 else float('nan'):.4f} |",
        "| new (tuned) hyperparams, NEW scale (30000 shots/20 epochs/8 seeds, main run) | 0.0013 ± 0.0002 | 0.4207 ± 0.0305 |",
        "",
    ]
    old_scale_e3, new_run_e3 = 0.3206, 0.4207
    this_e3 = statistics.fmean(e3_rates)
    dist_to_old = abs(this_e3 - old_scale_e3)
    dist_to_new = abs(this_e3 - new_run_e3)
    verdict = (
        "closer to the OLD-hyperparameter result -> the zero-shot regression is attributable to the "
        "TUNED HYPERPARAMETERS (architecture/capacity), not training scale"
        if dist_to_old < dist_to_new
        else "closer to the NEW-hyperparameter result -> training SCALE itself (not the tuned "
        "hyperparameters) is implicated in the zero-shot regression -- the 'specializes harder to "
        "memory statistics' story in LIMITATIONS.md needs revising"
    )
    lines.append(f"**Verdict**: old-hp/new-scale E3 rate ({this_e3:.4f}) is {verdict}.")
    return "\n".join(lines) + "\n"


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--p", type=float, default=0.002)
    parser.add_argument("--train-shots", type=int, default=30000)
    parser.add_argument("--test-shots", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--surgery-circuit-path", default="results/_k1_spacelike.stim")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--results-path", default="results/e3_isolation_check.jsonl")
    parser.add_argument("--summary-path", default="results/E3_ISOLATION_CHECK.md")
    args = parser.parse_args()

    results = run(
        distance=args.distance, p=args.p, train_shots=args.train_shots, test_shots=args.test_shots,
        epochs=args.epochs, num_seeds=args.num_seeds, surgery_circuit_path=args.surgery_circuit_path,
        k=args.k, results_path=args.results_path,
    )
    summary = render_summary(results)
    with open(args.summary_path, "w") as f:
        f.write(summary)
    print(f"\nwrote {args.summary_path}")
    print(summary)


if __name__ == "__main__":
    _cli()
