"""H4 threshold curves (design doc §6 E7, §7 "both thresholds, separately",
§8's own named `results/timelike_threshold.png`, never produced before
this script — see LIMITATIONS.md).

`experiments/e7_timelike.py` reports one (spacelike, timelike) rate pair
at one `p` — a sanity check, not a threshold. A real threshold needs a
`p` sweep at (at least) two code distances and the crossing point where
the larger distance stops helping: below threshold, more distance means
*fewer* logical errors (k=2 curve below k=1 curve); above threshold it's
the reverse. The crossing is the pseudo-threshold estimate.

MWPM is cheap (no training) — sweep many `p` points densely at k=1 and
k=2 for both observable kinds, straight from the DEM.

The GNN is expensive — train a fresh model per (kind, k, p) rather than
reuse one model across the sweep, since a decoder trained at one `p`
implicitly bakes in an assumption about the noise level (its default
edge weights / classification boundary are all fit to that training
distribution). Evaluated at fewer, representative `p` points along the
same curve.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from baselines.mwpm import MWPMBaseline
from circuits.configurations import k_to_distance
from circuits.lattice_surgery import load_generated_circuit
from data.generate import sample_shots
from data.to_graph import GraphBuildContext, batch_to_graphs
from experiments.common import ExperimentResult, append_result, evaluate_gnn, generate_surgery_circuit, train_gnn
from schema import NUM_EDGE_FEATURES


def mwpm_sweep(
    kind: str, ks: tuple[int, ...], ps: tuple[float, ...], shots: int, seed: int = 0,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in ks:
            for p in ps:
                out_path = Path(tmp) / f"k{k}_{kind}_p{p}.stim"
                generate_surgery_circuit(k, p, kind, out_path)
                circuit = load_generated_circuit(out_path)
                batch = sample_shots(circuit, shots=shots, seed=seed, warn_near_chance=False)
                mwpm = MWPMBaseline(circuit)
                preds = mwpm.decode(batch.detections)
                rate = float((preds != batch.observables[:, 0]).mean())
                results.append(
                    ExperimentResult(
                        experiment="E7_THRESHOLD", decoder="mwpm", observable_kind=kind, k=k, p=p, seed=seed,
                        logical_error_rate=rate, extra={"shots": shots},
                    )
                )
                print(f"  [mwpm] {kind} k={k} p={p:.5f}: rate={rate:.4f}")
    return results


def gnn_sweep(
    kind: str,
    ks: tuple[int, ...],
    ps: tuple[float, ...],
    train_shots: int,
    test_shots: int,
    epochs: int,
    seed: int,
    hidden_dim: int,
    num_layers: int,
    conv_type: str,
    heads: int,
    lr: float,
    weight_decay: float,
    device: str | None,
    batch_size: int = 64,
    edge_dim: int | None = NUM_EDGE_FEATURES,
) -> list[ExperimentResult]:
    """`batch_size` defaults lower than `train_gnn`'s own default (128) —
    surgery graphs at k=2 (d=5, more rounds, more detectors) are
    substantially larger than the k=1 or plain-memory graphs the higher
    default was fine for, and a 128-graph batch of 6-layer TransformerConv
    at k=2 was enough to CUDA-OOM an 8GB card mid-sweep (confirmed by
    actually running this at batch_size=128 first)."""
    results: list[ExperimentResult] = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in ks:
            for p in ps:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                out_path = Path(tmp) / f"k{k}_{kind}_p{p}.stim"
                generate_surgery_circuit(k, p, kind, out_path)
                circuit = load_generated_circuit(out_path)
                ctx = GraphBuildContext.build(circuit, patch_dimension=float(k_to_distance(k)))

                train_batch = sample_shots(circuit, shots=train_shots, seed=seed, warn_near_chance=False)
                train_graphs = batch_to_graphs(train_batch, ctx, observable_kind=kind, mask_surgery_features=False)
                model = train_gnn(
                    train_graphs, hidden_dim=hidden_dim, num_layers=num_layers, conv_type=conv_type, heads=heads,
                    lr=lr, weight_decay=weight_decay, epochs=epochs, seed=seed, device=device,
                    batch_size=batch_size, edge_dim=edge_dim,
                )

                test_batch = sample_shots(circuit, shots=test_shots, seed=seed + 1, warn_near_chance=False)
                test_graphs = batch_to_graphs(test_batch, ctx, observable_kind=kind, mask_surgery_features=False)
                rate = evaluate_gnn(model, test_graphs)
                results.append(
                    ExperimentResult(
                        experiment="E7_THRESHOLD", decoder="gnn", observable_kind=kind, k=k, p=p, seed=seed,
                        logical_error_rate=rate, extra={"train_shots": train_shots, "epochs": epochs},
                    )
                )
                print(f"  [gnn]  {kind} k={k} p={p:.5f}: rate={rate:.4f}")
    return results


def estimate_crossing(
    ps: list[float], rate_low_k: list[float], rate_high_k: list[float]
) -> float | None:
    """Pseudo-threshold: the `p` where the higher-`k` curve stops being
    below the lower-`k` curve (linear interpolation between the two
    bracketing sample points). Returns None if the curves never cross in
    the swept range — a genuine possible outcome (e.g. always sub- or
    always super-threshold across the whole sweep), reported as such
    rather than extrapolated."""
    diffs = [high - low for low, high in zip(rate_low_k, rate_high_k)]  # >0 once high-k is worse
    for i in range(len(diffs) - 1):
        if diffs[i] <= 0 < diffs[i + 1] or diffs[i] < 0 <= diffs[i + 1]:
            p0, p1 = ps[i], ps[i + 1]
            d0, d1 = diffs[i], diffs[i + 1]
            if d1 == d0:
                continue
            frac = -d0 / (d1 - d0)
            return p0 + frac * (p1 - p0)
    return None


def render_threshold_plot(all_results: list[ExperimentResult], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, kind in zip(axes, ("spacelike", "timelike")):
        for decoder, marker, ls in (("mwpm", "o", "-"), ("gnn", "^", "--")):
            rows = [r for r in all_results if r.observable_kind == kind and r.decoder == decoder]
            ks = sorted({r.k for r in rows})
            for k in ks:
                pts = sorted(((r.p, r.logical_error_rate) for r in rows if r.k == k))
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker=marker, linestyle=ls, label=f"{decoder} k={k} (d={2 * k + 1})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("physical error rate p")
        ax.set_ylabel("decoded logical error rate")
        ax.set_title(f"{kind} threshold")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_summary(all_results: list[ExperimentResult]) -> str:
    lines = ["# Threshold sweep (H4)", "", "Pseudo-threshold = the p where the k=2 (d=5) MWPM curve stops",
              "beating the k=1 (d=3) curve (linear interpolation between bracketing swept p values;", "'not found in range' if the curves never cross within the sweep).", ""]
    lines.append("| observable_kind | decoder | pseudo-threshold p |")
    lines.append("|---|---|---|")
    for kind in ("spacelike", "timelike"):
        for decoder in ("mwpm", "gnn"):
            rows = [r for r in all_results if r.observable_kind == kind and r.decoder == decoder]
            ks = sorted({r.k for r in rows})
            if len(ks) < 2:
                lines.append(f"| {kind} | {decoder} | N/A (only k={ks} swept) |")
                continue
            k_lo, k_hi = ks[0], ks[1]
            pts_lo = sorted(((r.p, r.logical_error_rate) for r in rows if r.k == k_lo))
            pts_hi = sorted(((r.p, r.logical_error_rate) for r in rows if r.k == k_hi))
            common_ps = sorted(set(p for p, _ in pts_lo) & set(p for p, _ in pts_hi))
            if len(common_ps) < 2:
                lines.append(f"| {kind} | {decoder} | N/A (insufficient overlapping p points) |")
                continue
            rate_lo = [dict(pts_lo)[p] for p in common_ps]
            rate_hi = [dict(pts_hi)[p] for p in common_ps]
            crossing = estimate_crossing(common_ps, rate_lo, rate_hi)
            lines.append(f"| {kind} | {decoder} | {f'{crossing:.5f}' if crossing is not None else 'not found in swept range'} |")
    return "\n".join(lines) + "\n"


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--mwpm-ps", type=float, nargs="+",
                         default=[0.0008, 0.0012, 0.0018, 0.0025, 0.0035, 0.005, 0.007, 0.01, 0.015, 0.02])
    parser.add_argument("--gnn-ps", type=float, nargs="+", default=[0.0012, 0.0025, 0.005, 0.01, 0.02])
    parser.add_argument("--mwpm-shots", type=int, default=20000)
    parser.add_argument("--gnn-train-shots", type=int, default=8000)
    parser.add_argument("--gnn-test-shots", type=int, default=5000)
    parser.add_argument("--gnn-epochs", type=int, default=15)
    parser.add_argument("--gnn-batch-size", type=int, default=64,
                         help="kept below E2's 128 default — k=2 surgery graphs are much larger; see gnn_sweep's docstring. "
                              "Note training-time OOM was never actually the failure mode (evaluate_gnn's un-chunked "
                              "eval batch was) — going much lower than this just makes training slower for no safety benefit.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128, help="tuned default — see results/HPARAM_SEARCH.md")
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--conv-type", default="transformer")
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--results-path", default="results/e7_threshold_sweep.jsonl")
    parser.add_argument("--summary-path", default="results/THRESHOLD_SWEEP.md")
    parser.add_argument("--plot-path", default="results/timelike_threshold.png")
    parser.add_argument("--skip-gnn", action="store_true", help="MWPM curves only (fast) — skip the expensive GNN training sweep")
    args = parser.parse_args()

    all_results: list[ExperimentResult] = []
    for kind in ("spacelike", "timelike"):
        print(f"[mwpm sweep] {kind}")
        rows = mwpm_sweep(kind, tuple(args.ks), tuple(args.mwpm_ps), args.mwpm_shots, seed=args.seed)
        for r in rows:
            append_result(args.results_path, r)
        all_results.extend(rows)

        if not args.skip_gnn:
            print(f"[gnn sweep] {kind}")
            rows = gnn_sweep(
                kind, tuple(args.ks), tuple(args.gnn_ps), args.gnn_train_shots, args.gnn_test_shots,
                args.gnn_epochs, args.seed, args.hidden_dim, args.num_layers, args.conv_type, args.heads,
                args.lr, args.weight_decay, args.device, batch_size=args.gnn_batch_size,
            )
            for r in rows:
                append_result(args.results_path, r)
            all_results.extend(rows)

    render_threshold_plot(all_results, args.plot_path)
    summary = render_summary(all_results)
    with open(args.summary_path, "w") as f:
        f.write(summary)
    print(f"\nwrote {args.plot_path}")
    print(f"wrote {args.summary_path}")
    print(summary)


if __name__ == "__main__":
    _cli()
