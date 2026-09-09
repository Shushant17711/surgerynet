"""Task 7.1 — aggregates every experiment's JSON-lines result file
(written by experiments/common.py's `append_result`) into
`results/main_table.md`.

Rows sharing everything except `seed` are grouped into one table row
reporting mean +/- std over those seeds, with an explicit seed count —
"Single-seed numbers are not evidence." A group with fewer than
`min_seeds` (default 3) is still shown, but flagged
`(n=k, INSUFFICIENT SEEDS)` rather than silently presented as if it were
a solid estimate. `status="infeasible"` rows (the MLP/CNN-on-surgery-data
case) render as `N/A` regardless of seed count — there's no rate to
average when the decoder can't run at all.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from experiments.common import ExperimentResult, read_results

RESULT_FILENAMES: tuple[str, ...] = (
    "e2_memory_baseline.jsonl",
    "e3_zeroshot.jsonl",
    "e5_surgery_trained.jsonl",
    "e6_config_randomization.jsonl",
    "e7_timelike.jsonl",
    "e8_latency.jsonl",
    "e9_ablations.jsonl",
)

_GROUP_KEY_FIELDS = ("experiment", "decoder", "observable_kind", "status", "distance", "k", "p")
_COLUMNS = (*_GROUP_KEY_FIELDS, "logical_error_rate", "latency_seconds", "n_seeds")

DEFAULT_MIN_SEEDS = 3


def aggregate(results_dir: str | Path = "results") -> list[ExperimentResult]:
    results_dir = Path(results_dir)
    rows: list[ExperimentResult] = []
    for filename in RESULT_FILENAMES:
        rows.extend(read_results(results_dir / filename))
    return rows


def _group_key(row: ExperimentResult) -> tuple:
    return tuple(getattr(row, f) for f in _GROUP_KEY_FIELDS)


def group_by_config(rows: list[ExperimentResult]) -> dict[tuple, list[ExperimentResult]]:
    """Groups rows that differ only by `seed` — everything else in
    `_GROUP_KEY_FIELDS` must match."""
    groups: dict[tuple, list[ExperimentResult]] = defaultdict(list)
    for r in rows:
        groups[_group_key(r)].append(r)
    return groups


def _mean_std_cell(values: list[float], min_seeds: int) -> str:
    n = len(values)
    if n == 0:
        return ""
    if n == 1:
        return f"{values[0]:.4f} (n=1, INSUFFICIENT SEEDS)"
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    suffix = f"(n={n})" if n >= min_seeds else f"(n={n}, INSUFFICIENT SEEDS)"
    return f"{mean:.4f} ± {std:.4f} {suffix}"


def render_main_table(rows: list[ExperimentResult], min_seeds: int = DEFAULT_MIN_SEEDS) -> str:
    groups = group_by_config(rows)
    header = "| " + " | ".join(_COLUMNS) + " |"
    separator = "|" + "|".join(["---"] * len(_COLUMNS)) + "|"

    body_rows = []
    for key in sorted(groups, key=lambda k: tuple("" if v is None else str(v) for v in k)):
        group = groups[key]
        experiment, decoder, observable_kind, status, distance, k, p = key

        if status == "infeasible":
            rate_cell = "N/A"
            latency_cell = "N/A"
            n_seeds = len(group)
        else:
            rates = [r.logical_error_rate for r in group if r.logical_error_rate is not None]
            latencies = [r.latency_seconds for r in group if r.latency_seconds is not None]
            rate_cell = _mean_std_cell(rates, min_seeds)
            latency_cell = _mean_std_cell(latencies, min_seeds)
            n_seeds = max(len(rates), len(latencies))

        cells = [
            str(v) if v is not None else ""
            for v in (experiment, decoder, observable_kind, status, distance, k, p)
        ]
        cells += [rate_cell, latency_cell, str(n_seeds)]
        body_rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header, separator, *body_rows])


def write_main_table(results_dir: str | Path = "results", out_path: str | Path = "results/main_table.md") -> Path:
    rows = aggregate(results_dir)
    table = render_main_table(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table + "\n")
    return out_path


if __name__ == "__main__":
    path = write_main_table()
    print(f"wrote {path}")
