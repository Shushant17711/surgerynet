"""Shared result-record schema for E1-E9, so Task 7.1's aggregation
doesn't have to reconcile N ad-hoc formats. Every experiment script
appends `ExperimentResult` rows (JSON Lines) to its own file under
`results/`; `experiments/report.py` (Task 7.1) reads all of them.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from models.gnn_decoder import GNNDecoder
from models.train import train_step

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TQEC_PYTHON = PROJECT_ROOT / ".venv-tqec" / "bin" / "python"


def default_device() -> str:
    """Single source of truth for "what device should a script use when
    the caller didn't pin one" — CUDA if available (an RTX 4060 was idle
    on this machine while every experiment ran on CPU; see
    HPARAM_SEARCH.md), CPU otherwise, so tests on a CPU-only machine still
    pass unmodified."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def generate_surgery_circuit(k: int, p: float, kind: str, out_path: str | Path) -> Path:
    """Shells out to circuits/tqec_backend.py under .venv-tqec — every E5+
    experiment that needs a real surgery circuit at a given (k, p, kind)
    goes through this one function rather than re-deriving the subprocess
    call each time."""
    if not TQEC_PYTHON.exists():
        raise RuntimeError(
            f"{TQEC_PYTHON} not found — set up the TQEC backend first "
            "(see circuits/tqec_backend.py's module docstring)"
        )
    out_path = Path(out_path)
    subprocess.run(
        [
            str(TQEC_PYTHON),
            str(PROJECT_ROOT / "circuits" / "tqec_backend.py"),
            "--distance-k", str(k),
            "--p", str(p),
            "--kind", kind,
            "--out", str(out_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return out_path


def evaluate_gnn(model: GNNDecoder, graphs: list[Data], eval_batch_size: int = 256) -> float:
    """The one evaluation function E3 and E5 both call (Task 6.4's own
    requirement) so their reported metrics are computed identically —
    same decision rule (logit > 0), same batching, no per-experiment
    drift in how "logical error rate" is defined.

    Batches onto whatever device `model`'s own parameters already live
    on (no separate `device` argument to keep this call-compatible
    everywhere) — `train_gnn` below is what actually places a model on
    a device.

    Chunks `graphs` into `eval_batch_size`-sized batches instead of
    collating all of them into one giant `Batch` — this used to build one
    single batch from the *entire* test set, which was fine for small
    memory-circuit graphs but CUDA-OOM'd on large k=2 surgery-circuit
    test sets (thousands of graphs, each with far more nodes/edges than a
    memory graph, all forwarded through a 6-layer TransformerConv at
    once) — confirmed by actually hitting it mid-run, not by inspection."""
    model.eval()
    device = next(model.parameters()).device
    wrong = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(graphs), eval_batch_size):
            chunk = graphs[start : start + eval_batch_size]
            batch = Batch.from_data_list(chunk).to(device)
            preds = (model.forward_selected(batch) > 0).float()
            targets = batch.y.squeeze(-1)
            wrong += int((preds != targets).sum().item())
            total += len(chunk)
    return wrong / total


def train_gnn(
    train_graphs: list[Data],
    *,
    hidden_dim: int = 64,
    num_layers: int = 4,
    conv_type: str = "gat",
    heads: int = 4,
    use_norm: bool = True,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    epochs: int = 3,
    batch_size: int = 128,
    seed: int = 0,
    device: str | None = None,
    wandb_run: Any | None = None,
) -> GNNDecoder:
    """The one training loop E2/E5/E6/E7/E9 all build a model and run
    through (previously each hand-rolled its own copy of this loop with
    hardcoded hidden_dim=64/num_layers=4/lr=1e-3 — the exact knobs
    `experiments/hparam_search.py` needs to vary, so those are now real
    parameters instead of duplicated literals). `device` defaults to
    `default_device()` (CUDA when available) so every caller gets GPU
    training for free unless it explicitly pins `device="cpu"`.

    Clears the CUDA cache before building the new model — a long-running
    driver (`run_all.py`, `hparam_search.py`, `e7_threshold_sweep.py`)
    calls this repeatedly with graphs of very different sizes (k=1 vs
    k=2 surgery circuits in particular), and PyTorch's caching allocator
    doesn't automatically release memory reserved for a since-discarded
    model/batch — enough repeated size changes without this actually
    CUDA-OOM'd an 8GB card mid-sweep (see e7_threshold_sweep.py's
    `gnn_sweep` docstring).

    Calls `torch.manual_seed(seed)` right before constructing the model —
    `seed` previously only drove the numpy RNG used for minibatch shuffling,
    not `GNNDecoder`'s weight initialization (which uses PyTorch's global,
    un-seeded RNG). That meant two calls with the same `seed` were NOT
    bit-reproducible: the actual initialization depended on how much of
    PyTorch's global random state earlier calls in the same process had
    already consumed. This didn't invalidate any multi-seed statistics
    already produced (each "seed" was still a genuinely independent
    trial — the mean/std over 8 such trials is valid either way), it just
    meant `--seed N` wasn't a literal, rerunnable recipe. Fixed here rather
    than left as a footnote, since `results/`'s own numbers depend on
    exactly this reproducibility claim being true going forward."""
    device = device or default_device()
    if device == "cuda":
        torch.cuda.empty_cache()
    torch.manual_seed(seed)
    model = GNNDecoder(
        hidden_dim=hidden_dim, num_layers=num_layers, conv_type=conv_type, heads=heads, use_norm=use_norm
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(train_graphs))
        for start in range(0, len(train_graphs), batch_size):
            idx = order[start : start + batch_size]
            mini_batch = Batch.from_data_list([train_graphs[i] for i in idx]).to(device)
            train_step(model, optimizer, mini_batch, wandb_run=wandb_run, step=step)
            step += 1
    return model


@dataclass(slots=True)
class ExperimentResult:
    experiment: str  # "E2", "E3", ...
    decoder: str  # "gnn", "mwpm", "union_find", "mlp", "cnn"
    observable_kind: str  # "spacelike", "timelike", or "n/a" for e.g. latency rows
    status: str = "ok"  # "ok" | "infeasible" | "error"
    distance: int | None = None
    k: int | None = None
    p: float | None = None
    seed: int | None = None  # which of the >=3 seeds this row is from — see experiments/report.py's aggregation
    logical_error_rate: float | None = None  # E2/E3/E5/E6/E7/E9's own metric
    latency_seconds: float | None = None  # E8's own metric
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ("ok", "infeasible", "error"):
            raise ValueError(f"status must be ok/infeasible/error, got {self.status!r}")
        if self.status == "ok" and self.logical_error_rate is None and self.latency_seconds is None:
            raise ValueError("status='ok' rows must carry a logical_error_rate or a latency_seconds")


def append_result(path: str | Path, result: ExperimentResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(result)) + "\n")


def read_results(path: str | Path) -> list[ExperimentResult]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(ExperimentResult(**json.loads(line)))
    return out
