"""Runs the leave-one-out ablation variants (experiments.e9_ablations)
across >=3 seeds each, appends to results/e9_ablations_loo.jsonl (a
*different* file from run_all.py's own single-config E9 stage, which
writes results/e9_ablations.jsonl — mixing the two would silently merge
the 5 "component removed" variants into main_table.md's one E9 summary
row, a real bug caught by inspecting the raw jsonl after a full run), then
writes a human-readable "what breaks" report to results/ABLATIONS.md.
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from experiments.common import ExperimentResult, append_result, read_results
from experiments.e9_ablations import leave_one_out_variants, run_one


def run(
    k: int = 1,
    p: float = 0.01,
    train_shots: int = 1000,
    test_shots: int = 1000,
    epochs: int = 2,
    num_seeds: int = 3,
    results_path: str = "results/e9_ablations_loo.jsonl",
    hidden_dim: int = 64,
    conv_type: str = "transformer",
    heads: int = 2,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    device: str | None = None,
) -> list[ExperimentResult]:
    variants = leave_one_out_variants()
    results: list[ExperimentResult] = []
    for name, config in variants.items():
        for seed in range(num_seeds):
            print(f"[ablation] {name} (seed={seed})")
            r = run_one(
                config, k=k, p=p, train_shots=train_shots, test_shots=test_shots,
                epochs=epochs, seed=seed, variant_name=name,
                hidden_dim=hidden_dim, conv_type=conv_type, heads=heads, lr=lr, weight_decay=weight_decay,
                device=device,
            )
            append_result(results_path, r)
            results.append(r)
    return results


def render_report(results_path: str = "results/e9_ablations_loo.jsonl", min_seeds: int = 3) -> str:
    rows = [r for r in read_results(results_path) if r.extra.get("variant_name")]
    by_variant: dict[str, list[ExperimentResult]] = defaultdict(list)
    for r in rows:
        by_variant[r.extra["variant_name"]].append(r)

    baseline_name = "baseline (all components on)"
    if baseline_name not in by_variant:
        return "No baseline ablation results found — run experiments/run_ablations.py first.\n"

    baseline_rates = [r.logical_error_rate for r in by_variant[baseline_name]]
    baseline_mean = statistics.fmean(baseline_rates)

    lines = ["# Ablation report", "", "What breaks when each component is removed, relative to baseline.", ""]
    lines.append("| variant | mean logical error rate | std | n seeds | delta vs baseline |")
    lines.append("|---|---|---|---|---|")

    for name, group in sorted(by_variant.items(), key=lambda kv: kv[0] != baseline_name):
        rates = [r.logical_error_rate for r in group]
        n = len(rates)
        mean = statistics.fmean(rates)
        std = statistics.stdev(rates) if n > 1 else float("nan")
        delta = mean - baseline_mean
        flag = "" if n >= min_seeds else " (INSUFFICIENT SEEDS)"
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {name} | {mean:.4f} | {std:.4f} | {n}{flag} | {sign}{delta:.4f} |")

    lines.append("")
    lines.append("Positive delta = worse (higher logical error rate) than baseline when that component is removed.")
    return "\n".join(lines) + "\n"


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--train-shots", type=int, default=1000)
    parser.add_argument("--test-shots", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--results-path", default="results/e9_ablations_loo.jsonl")
    parser.add_argument("--report-path", default="results/ABLATIONS.md")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--conv-type", default="transformer")
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run(
        k=args.k, p=args.p, train_shots=args.train_shots, test_shots=args.test_shots,
        epochs=args.epochs, num_seeds=args.num_seeds, results_path=args.results_path,
        hidden_dim=args.hidden_dim, conv_type=args.conv_type, heads=args.heads, lr=args.lr,
        weight_decay=args.weight_decay, device=args.device,
    )
    report = render_report(args.results_path, min_seeds=args.num_seeds)
    with open(args.report_path, "w") as f:
        f.write(report)
    print(f"wrote {args.report_path}")


if __name__ == "__main__":
    _cli()
