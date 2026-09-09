"""Evidence for the edge-feature addition (LIMITATIONS.md's "fourth pass"):
the GNN originally saw only graph *topology*, never the DEM's own per-edge
error probability — exactly the number MWPM's own matching weight comes
from. `data/to_graph.py` now attaches that (as `log((1-p)/p)`, plus
edge-source flags and normalized distance) to every edge as `edge_attr`,
and `models/gnn_decoder.py` accepts an `edge_dim` to use it.

This script is the reproducible record of two things:
  1. A/B: does `edge_dim=None` (topology only) vs `edge_dim=NUM_EDGE_FEATURES`
     (topology + DEM weight) actually change the result, at the *previously
     tuned* hyperparameters, on both E2 (memory) and E5 (surgery)?
  2. A small follow-up hyperparameter re-check *with* edge features on —
     not a full re-run of `hparam_search.py`'s 58-trial search, but enough
     configs around the previous winner to check whether edge features
     shift which hyperparameters are best (they do: more hidden_dim and a
     higher learning rate both help once edge weights are available).
"""

from __future__ import annotations

import argparse
import statistics

from experiments.common import ExperimentResult, append_result, generate_surgery_circuit
from experiments.e2_memory_baseline import run as run_e2
from experiments.e5_surgery_trained import run as run_e5

PREVIOUS_TUNED = dict(hidden_dim=64, num_layers=6, conv_type="transformer", heads=2, lr=3e-4, weight_decay=1e-4)
CANDIDATE_RETUNE = dict(hidden_dim=128, num_layers=6, conv_type="transformer", heads=2, lr=1e-3, weight_decay=1e-4)


def ab_test(
    p: float = 0.002,
    train_shots: int = 20000,
    test_shots: int = 20000,
    epochs: int = 25,
    num_seeds: int = 3,
    k: int = 1,
    surgery_circuit_path: str = "results/_k1_spacelike.stim",
    results_path: str = "results/edge_feature_check.jsonl",
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []

    for label, hp, edge_dim in (
        ("previous_tuned_no_edge_feat", PREVIOUS_TUNED, None),
        ("previous_tuned_with_edge_feat", PREVIOUS_TUNED, 5),
        ("retuned_with_edge_feat", CANDIDATE_RETUNE, 5),
    ):
        for seed in range(num_seeds):
            print(f"[E2] {label} seed={seed}")
            e2_gnn, e2_mwpm, _ = run_e2(
                distance=3, p=p, train_shots=train_shots, test_shots=test_shots, epochs=epochs, seed=seed,
                edge_dim=edge_dim, **hp,
            )
            e2_gnn.extra["variant"] = label
            append_result(results_path, e2_gnn)
            results.append(e2_gnn)

            print(f"[E5] {label} seed={seed}")
            e5_result, _ = run_e5(
                surgery_circuit_path, float(2 * k + 1), "spacelike", train_shots=train_shots, test_shots=test_shots,
                epochs=epochs, seed=seed, p=p, k=k, edge_dim=edge_dim, **hp,
            )
            e5_result.extra["variant"] = label
            append_result(results_path, e5_result)
            results.append(e5_result)
            print(
                f"    E2 gnn={e2_gnn.logical_error_rate:.4f} mwpm={e2_mwpm.logical_error_rate:.4f}  "
                f"E5 gnn={e5_result.logical_error_rate:.4f}"
            )
    return results


def render_summary(results: list[ExperimentResult]) -> str:
    by_variant_e2: dict[str, list[float]] = {}
    by_variant_e5: dict[str, list[float]] = {}
    for r in results:
        variant = r.extra.get("variant", "unknown")
        if r.experiment == "E2":
            by_variant_e2.setdefault(variant, []).append(r.logical_error_rate)
        elif r.experiment == "E5":
            by_variant_e5.setdefault(variant, []).append(r.logical_error_rate)

    lines = ["# Edge-feature addition: A/B check + small re-tune", "",
              "| variant | E2 (memory) mean rate | E5 (surgery) mean rate |", "|---|---|---|"]
    order = ["previous_tuned_no_edge_feat", "previous_tuned_with_edge_feat", "retuned_with_edge_feat"]
    for variant in order:
        e2_rates = by_variant_e2.get(variant, [])
        e5_rates = by_variant_e5.get(variant, [])
        e2_str = f"{statistics.fmean(e2_rates):.4f} (n={len(e2_rates)})" if e2_rates else "N/A"
        e5_str = f"{statistics.fmean(e5_rates):.4f} (n={len(e5_rates)})" if e5_rates else "N/A"
        lines.append(f"| {variant} | {e2_str} | {e5_str} |")
    lines += [
        "",
        "`previous_tuned_*` uses `results/HPARAM_SEARCH.md`'s winner "
        f"({PREVIOUS_TUNED}); `retuned_with_edge_feat` is a small follow-up check "
        f"around it once edge features are on ({CANDIDATE_RETUNE}), not a full re-run "
        "of `hparam_search.py`'s 58-trial search.",
    ]
    return "\n".join(lines) + "\n"


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=float, default=0.002)
    parser.add_argument("--train-shots", type=int, default=20000)
    parser.add_argument("--test-shots", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--surgery-circuit-path", default="results/_k1_spacelike.stim")
    parser.add_argument("--results-path", default="results/edge_feature_check.jsonl")
    parser.add_argument("--summary-path", default="results/EDGE_FEATURE_CHECK.md")
    args = parser.parse_args()

    generate_surgery_circuit(args.k, args.p, "spacelike", args.surgery_circuit_path)
    results = ab_test(
        p=args.p, train_shots=args.train_shots, test_shots=args.test_shots, epochs=args.epochs,
        num_seeds=args.num_seeds, k=args.k, surgery_circuit_path=args.surgery_circuit_path,
        results_path=args.results_path,
    )
    summary = render_summary(results)
    with open(args.summary_path, "w") as f:
        f.write(summary)
    print(f"\nwrote {args.summary_path}")
    print(summary)


if __name__ == "__main__":
    _cli()
