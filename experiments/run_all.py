"""Chains E2 through E9 into one run, handling the cross-stage handoffs a
shell script can't express cleanly (E2's trained memory model feeds E3's
zero-shot evaluation and E4's heatmap). Writes every stage's result into
results/*.jsonl via experiments/common.append_result, then calls
experiments/report.py.

Runs each stochastic stage across `--num-seeds` seeds (default 3) —
"single-seed numbers are not evidence" — so experiments/report.py's
aggregation has enough rows per configuration to report a real mean +/-
std rather than a lone number. E3's fixed-input-infeasibility check and E4's
heatmap are deterministic-shape / illustrative respectively and are not
looped across seeds (see their own docstrings).

This is the "real run" driver scripts/reproduce.sh delegates to. Defaults
are still smoke-test-sized (matching this project's own test suite) — pass
larger --*-shots values for an actual publication-scale run.
"""

from __future__ import annotations

import argparse

from circuits.configurations import k_to_distance
from circuits.lattice_surgery import load_generated_circuit
from data.generate import sample_shots
from data.to_graph import GraphBuildContext, batch_to_graphs
from experiments.common import append_result, generate_surgery_circuit
from experiments.e2_memory_baseline import run as run_e2
from experiments.e3_zeroshot import evaluate_fixed_input_infeasibility, evaluate_gnn_zero_shot
from experiments.e4_localization import compute_localization_heatmap, render_heatmap
from experiments.e5_surgery_trained import run as run_e5
from experiments.e6_config_randomization import run as run_e6
from experiments.e7_timelike import run as run_e7
from experiments.e8_latency import run_latency_sweep
from experiments.e9_ablations import ablation_grid, run_one
from experiments.report import write_main_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num-seeds", type=int, default=3, help="seeds per stochastic stage; <3 is flagged in the report")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--full-ablation-grid", action="store_true", help="run every E9 variant, not just one")
    parser.add_argument("--hidden-dim", type=int, default=64, help="tuned default — see results/HPARAM_SEARCH.md")
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--conv-type", default="transformer")
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None, help="cuda/cpu; default auto-detects")
    args = parser.parse_args()

    results_dir = args.results_dir
    seeds = list(range(args.num_seeds))
    hp = dict(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, conv_type=args.conv_type, heads=args.heads,
        lr=args.lr, weight_decay=args.weight_decay, device=args.device,
    )

    surgery_path = f"{results_dir}/_k{args.k}_spacelike.stim"
    generate_surgery_circuit(args.k, args.p, "spacelike", surgery_path)
    patch_dimension = float(k_to_distance(args.k))
    circuit = load_generated_circuit(surgery_path)

    first_memory_model = None
    for seed in seeds:
        print(f"[E2] memory baseline (seed={seed})")
        e2_gnn, e2_mwpm, memory_trained_model = run_e2(
            distance=args.distance, p=args.p, train_shots=args.shots, test_shots=args.shots,
            epochs=args.epochs, seed=seed, **hp,
        )
        append_result(f"{results_dir}/e2_memory_baseline.jsonl", e2_gnn)
        append_result(f"{results_dir}/e2_memory_baseline.jsonl", e2_mwpm)
        if first_memory_model is None:
            first_memory_model = memory_trained_model

        print(f"[E3] zero-shot transfer (seed={seed})")
        e3_result = evaluate_gnn_zero_shot(
            memory_trained_model, surgery_path, patch_dimension, "spacelike",
            shots=args.shots, seed=seed, p=args.p, k=args.k,
        )
        append_result(f"{results_dir}/e3_zeroshot.jsonl", e3_result)

        print(f"[E5] surgery-trained model (seed={seed})")
        e5_result, _ = run_e5(
            surgery_path, patch_dimension, "spacelike",
            train_shots=args.shots, test_shots=args.shots, epochs=args.epochs, seed=seed,
            p=args.p, k=args.k, **hp,
        )
        append_result(f"{results_dir}/e5_surgery_trained.jsonl", e5_result)

        print(f"[E6] configuration randomization (seed={seed})")
        e6_result = run_e6(
            distances=(3, 5), p=args.p, shots_per_config=args.shots // 2, epochs=args.epochs, seed=seed, **hp,
        )
        append_result(f"{results_dir}/e6_config_randomization.jsonl", e6_result)

        print(f"[E7] timelike errors (seed={seed})")
        e7_results = run_e7(
            k=args.k, p=args.p, train_shots=args.shots, test_shots=args.shots, epochs=args.epochs, seed=seed, **hp,
        )
        for r in e7_results:
            append_result(f"{results_dir}/e7_timelike.jsonl", r)

        print(f"[E8] latency (seed={seed})")
        # Capped independent of --shots: E8 measures single-batch forward
        # latency, so the batch should stay a realistic size regardless of
        # how large a scale-up run's training/test shot counts get (a
        # 40000-graph single batch is neither a meaningful latency number
        # nor safe to allocate at once on an 8GB card at k=2 with the
        # tuned 6-layer TransformerConv model).
        e8_results = run_latency_sweep(ks=(1, 2), p=args.p, shots=min(args.shots, 500), seed=seed)
        for r in e8_results:
            append_result(f"{results_dir}/e8_latency.jsonl", r)

        print(f"[E9] ablations (seed={seed})")
        grid = ablation_grid() if args.full_ablation_grid else ablation_grid()[:1]
        e9_hp = {k: v for k, v in hp.items() if k != "num_layers"}
        for config in grid:
            r = run_one(
                config, k=args.k, p=args.p, train_shots=args.shots, test_shots=args.shots, epochs=args.epochs,
                seed=seed, **e9_hp,
            )
            append_result(f"{results_dir}/e9_ablations.jsonl", r)

    print("[E3] fixed-input infeasibility (deterministic, run once)")
    mlp_result, cnn_result = evaluate_fixed_input_infeasibility(
        memory_num_detectors=circuit.num_detectors // 2,  # deliberately mismatched -- memory != surgery size
        memory_grid_shape=(args.distance, args.distance, args.distance),
        surgery_num_detectors=circuit.num_detectors,
        surgery_grid_shape=(args.distance, args.distance, args.distance + 2),
        observable_kind="spacelike",
    )
    append_result(f"{results_dir}/e3_zeroshot.jsonl", mlp_result)
    append_result(f"{results_dir}/e3_zeroshot.jsonl", cnn_result)

    print("[E4] localization heatmap (illustrative, seed=0's model)")
    ctx = GraphBuildContext.build(circuit, patch_dimension=patch_dimension)
    batch = sample_shots(circuit, shots=args.shots, seed=0)
    graphs = batch_to_graphs(batch, ctx, observable_kind="spacelike")
    heatmap, _ = compute_localization_heatmap(first_memory_model, ctx, batch, graphs)
    render_heatmap(heatmap, f"{results_dir}/spacetime_heatmap.png")

    print("[report] aggregating into main_table.md")
    out = write_main_table(results_dir=results_dir, out_path=f"{results_dir}/main_table.md")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
