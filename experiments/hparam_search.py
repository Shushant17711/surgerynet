"""Hyperparameter search for closing the E2 gate (design doc §6 E2: "beat
MWPM on memory, or the problem is your GNN, not lattice surgery").

Before this script existed, every experiment used the same
hardcoded (hidden_dim=64, num_layers=4, conv_type="gat", heads=4, lr=1e-3)
defaults from the design doc's own §5.3 ranges, never tuned against
anything — and the GNN lost to MWPM by ~15x on the plain memory circuit at
the validated sub-threshold p=0.002 (see LIMITATIONS.md's "first version
was wrong" section). This was also all CPU: an RTX 4060 sat idle the
whole time (see `experiments/common.py::default_device`), which is the
actual reason no search had been run before — a single E2 config took
long enough on CPU that trying 40+ of them wasn't practical.

Two-stage search:
  1. **Cheap ranking pass** — many randomly sampled configs (from
     `SEARCH_SPACE`, respecting `hidden_dim % heads == 0`), one seed each,
     modest shots/epochs, to rank candidates fast.
  2. **Confirmation pass** — the top `--confirm-top` configs from stage 1,
     rerun at the real target scale (more shots, more epochs) across
     `--confirm-seeds` seeds, so the winner is picked on real multi-seed
     evidence, not a single lucky draw.

Writes every trial (both stages) to `results/hparam_search.jsonl`
(`experiment="HPARAM"` rows, reusing the shared `ExperimentResult`
schema) and a human-readable `results/HPARAM_SEARCH.md` summary naming
the winning config and stating plainly whether it actually beat MWPM.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

from experiments.common import ExperimentResult, append_result
from experiments.e2_memory_baseline import run as run_e2

SEARCH_SPACE: dict[str, tuple] = {
    "hidden_dim": (64, 128, 256),
    "num_layers": (4, 5, 6),
    "conv_type": ("gat", "transformer"),
    "heads": (2, 4),
    "lr": (3e-4, 1e-3, 3e-3),
    "batch_size": (64, 128, 256),
    "weight_decay": (0.0, 1e-5, 1e-4),
}


@dataclass(frozen=True, slots=True)
class HparamConfig:
    hidden_dim: int
    num_layers: int
    conv_type: str
    heads: int
    lr: float
    batch_size: int
    weight_decay: float

    def label(self) -> str:
        return (
            f"hd{self.hidden_dim}_L{self.num_layers}_{self.conv_type}_h{self.heads}"
            f"_lr{self.lr:g}_bs{self.batch_size}_wd{self.weight_decay:g}"
        )


def sample_configs(n: int, seed: int = 0) -> list[HparamConfig]:
    """Random sample of `n` distinct configs from `SEARCH_SPACE`, skipping
    any (hidden_dim, heads) pair where hidden_dim isn't divisible by
    heads — `GNNDecoder` requires that (each head gets an equal slice of
    the hidden dimension)."""
    rng = random.Random(seed)
    seen: set[HparamConfig] = set()
    out: list[HparamConfig] = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        cfg = HparamConfig(
            hidden_dim=rng.choice(SEARCH_SPACE["hidden_dim"]),
            num_layers=rng.choice(SEARCH_SPACE["num_layers"]),
            conv_type=rng.choice(SEARCH_SPACE["conv_type"]),
            heads=rng.choice(SEARCH_SPACE["heads"]),
            lr=rng.choice(SEARCH_SPACE["lr"]),
            batch_size=rng.choice(SEARCH_SPACE["batch_size"]),
            weight_decay=rng.choice(SEARCH_SPACE["weight_decay"]),
        )
        if cfg.hidden_dim % cfg.heads != 0 or cfg in seen:
            continue
        seen.add(cfg)
        out.append(cfg)
    return out


def run_trial(
    config: HparamConfig,
    stage: str,
    distance: int,
    p: float,
    train_shots: int,
    test_shots: int,
    epochs: int,
    seed: int,
    device: str | None,
) -> ExperimentResult:
    start = time.perf_counter()
    gnn_result, mwpm_result, _ = run_e2(
        distance=distance, p=p, train_shots=train_shots, test_shots=test_shots, epochs=epochs, seed=seed,
        hidden_dim=config.hidden_dim, num_layers=config.num_layers, conv_type=config.conv_type,
        heads=config.heads, lr=config.lr, batch_size=config.batch_size, weight_decay=config.weight_decay,
        device=device,
    )
    elapsed = time.perf_counter() - start
    return ExperimentResult(
        experiment="HPARAM",
        decoder="gnn",
        observable_kind="spacelike",
        distance=distance,
        p=p,
        seed=seed,
        logical_error_rate=gnn_result.logical_error_rate,
        extra={
            "stage": stage,
            "config": asdict(config),
            "config_label": config.label(),
            "mwpm_rate": mwpm_result.logical_error_rate,
            "beats_mwpm": gnn_result.logical_error_rate < mwpm_result.logical_error_rate,
            "train_shots": train_shots,
            "epochs": epochs,
            "elapsed_seconds": elapsed,
        },
    )


def run_search(
    distance: int = 3,
    p: float = 0.002,
    num_candidates: int = 40,
    cheap_train_shots: int = 5000,
    cheap_test_shots: int = 3000,
    cheap_epochs: int = 10,
    confirm_top: int = 6,
    confirm_train_shots: int = 20000,
    confirm_test_shots: int = 10000,
    confirm_epochs: int = 25,
    confirm_seeds: int = 3,
    results_path: str = "results/hparam_search.jsonl",
    device: str | None = None,
    search_seed: int = 0,
) -> tuple[HparamConfig, list[ExperimentResult]]:
    configs = sample_configs(num_candidates, seed=search_seed)
    print(f"[hparam_search] stage 1: ranking {len(configs)} candidates "
          f"({cheap_train_shots} train shots, {cheap_epochs} epochs, 1 seed each)")

    stage1_results: list[ExperimentResult] = []
    for i, cfg in enumerate(configs):
        r = run_trial(cfg, "rank", distance, p, cheap_train_shots, cheap_test_shots, cheap_epochs, seed=0, device=device)
        append_result(results_path, r)
        stage1_results.append(r)
        print(
            f"  [{i + 1}/{len(configs)}] {cfg.label()}: gnn={r.logical_error_rate:.4f} "
            f"mwpm={r.extra['mwpm_rate']:.4f} beats_mwpm={r.extra['beats_mwpm']} "
            f"({r.extra['elapsed_seconds']:.1f}s)"
        )

    stage1_results.sort(key=lambda r: r.logical_error_rate)
    top_configs = [HparamConfig(**r.extra["config"]) for r in stage1_results[:confirm_top]]

    print(f"\n[hparam_search] stage 2: confirming top {len(top_configs)} candidates "
          f"({confirm_train_shots} train shots, {confirm_epochs} epochs, {confirm_seeds} seeds each)")

    stage2_results: list[ExperimentResult] = []
    for i, cfg in enumerate(top_configs):
        for seed in range(confirm_seeds):
            r = run_trial(
                cfg, "confirm", distance, p, confirm_train_shots, confirm_test_shots, confirm_epochs,
                seed=seed, device=device,
            )
            append_result(results_path, r)
            stage2_results.append(r)
            print(
                f"  [{i + 1}/{len(top_configs)}, seed {seed}] {cfg.label()}: "
                f"gnn={r.logical_error_rate:.4f} mwpm={r.extra['mwpm_rate']:.4f} "
                f"({r.extra['elapsed_seconds']:.1f}s)"
            )

    by_label: dict[str, list[ExperimentResult]] = defaultdict(list)
    for r in stage2_results:
        by_label[r.extra["config_label"]].append(r)

    def mean_rate(rows: list[ExperimentResult]) -> float:
        return statistics.fmean(r.logical_error_rate for r in rows)

    winner_label = min(by_label, key=lambda lbl: mean_rate(by_label[lbl]))
    winner_config = HparamConfig(**by_label[winner_label][0].extra["config"])

    return winner_config, stage1_results + stage2_results


def render_summary(all_results: list[ExperimentResult], winner: HparamConfig, p: float) -> str:
    confirm_rows = [r for r in all_results if r.extra.get("stage") == "confirm"]
    by_label: dict[str, list[ExperimentResult]] = defaultdict(list)
    for r in confirm_rows:
        by_label[r.extra["config_label"]].append(r)

    lines = [
        "# Hyperparameter search results",
        "",
        f"Search target: beat MWPM on plain memory (E2) at p={p}. "
        f"{len(all_results)} total trials logged to `results/hparam_search.jsonl` "
        "(stage 1 ranking + stage 2 multi-seed confirmation).",
        "",
        "## Stage 2 (confirmation) — mean +/- std over seeds",
        "",
        "| config | mean GNN rate | std | n seeds | mean MWPM rate | beats MWPM |",
        "|---|---|---|---|---|---|",
    ]
    rows_sorted = sorted(by_label.items(), key=lambda kv: statistics.fmean(r.logical_error_rate for r in kv[1]))
    for label, rows in rows_sorted:
        rates = [r.logical_error_rate for r in rows]
        mwpm_rates = [r.extra["mwpm_rate"] for r in rows]
        mean = statistics.fmean(rates)
        std = statistics.stdev(rates) if len(rates) > 1 else float("nan")
        mwpm_mean = statistics.fmean(mwpm_rates)
        beats = "yes" if mean < mwpm_mean else "no"
        marker = " **<- winner**" if label == winner.label() else ""
        lines.append(f"| {label}{marker} | {mean:.4f} | {std:.4f} | {len(rates)} | {mwpm_mean:.4f} | {beats} |")

    winner_rows = by_label[winner.label()]
    winner_mean = statistics.fmean(r.logical_error_rate for r in winner_rows)
    winner_mwpm_mean = statistics.fmean(r.extra["mwpm_rate"] for r in winner_rows)
    ratio_str = (
        f"{winner_mean / winner_mwpm_mean:.2f}x MWPM's rate"
        if winner_mwpm_mean > 0
        else "MWPM's rate was 0 at this shot count — ratio undefined, not zero-inflated"
    )
    lines += [
        "",
        "## Winning configuration",
        "",
        f"```\n{winner}\n```",
        "",
        f"Mean GNN logical error rate: {winner_mean:.4f}, mean MWPM: {winner_mwpm_mean:.4f} "
        f"({ratio_str}, {'GNN beats MWPM' if winner_mean < winner_mwpm_mean else 'MWPM still wins'}).",
        "",
        "This is the honest outcome of the search, stated plainly — a search is run to find out "
        "the answer, not to manufacture a specific one.",
    ]
    return "\n".join(lines) + "\n"


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--p", type=float, default=0.002)
    parser.add_argument("--num-candidates", type=int, default=40)
    parser.add_argument("--cheap-train-shots", type=int, default=5000)
    parser.add_argument("--cheap-test-shots", type=int, default=3000)
    parser.add_argument("--cheap-epochs", type=int, default=10)
    parser.add_argument("--confirm-top", type=int, default=6)
    parser.add_argument("--confirm-train-shots", type=int, default=20000)
    parser.add_argument("--confirm-test-shots", type=int, default=10000)
    parser.add_argument("--confirm-epochs", type=int, default=25)
    parser.add_argument("--confirm-seeds", type=int, default=3)
    parser.add_argument("--results-path", default="results/hparam_search.jsonl")
    parser.add_argument("--summary-path", default="results/HPARAM_SEARCH.md")
    parser.add_argument("--device", default=None)
    parser.add_argument("--search-seed", type=int, default=0)
    args = parser.parse_args()

    winner, all_results = run_search(
        distance=args.distance, p=args.p, num_candidates=args.num_candidates,
        cheap_train_shots=args.cheap_train_shots, cheap_test_shots=args.cheap_test_shots,
        cheap_epochs=args.cheap_epochs, confirm_top=args.confirm_top,
        confirm_train_shots=args.confirm_train_shots, confirm_test_shots=args.confirm_test_shots,
        confirm_epochs=args.confirm_epochs, confirm_seeds=args.confirm_seeds,
        results_path=args.results_path, device=args.device, search_seed=args.search_seed,
    )
    summary = render_summary(all_results, winner, args.p)
    with open(args.summary_path, "w") as f:
        f.write(summary)
    print(f"\nwrote {args.summary_path}")
    print(f"winner: {winner}")


if __name__ == "__main__":
    _cli()
