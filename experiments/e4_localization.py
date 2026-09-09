"""E4 — localization (design doc §6 E4, H2): per-round, per-region logical
error rate — the paper's key figure.

For each (round, region) bucket, computes P(decoder wrong | this shot has
at least one fired detector in that bucket) — a proxy for "how much does
activity here correlate with decoder failure," which is what H2 actually
asks: does accuracy loss concentrate in the merge window / near the
routing-region boundary, rather than spreading evenly.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from data.generate import ShotBatch
from data.to_graph import GraphBuildContext
from models.gnn_decoder import GNNDecoder
from schema import REGION_LABELS


def compute_localization_heatmap(
    model: GNNDecoder, ctx: GraphBuildContext, batch: ShotBatch, graphs: list[Data]
) -> tuple[np.ndarray, list[float]]:
    """Returns (heatmap, round_values). heatmap[r, c] is NaN where no shot
    ever had activity in that bucket (nothing to estimate an error rate
    from), otherwise a probability in [0, 1]."""
    all_t = sorted({int(lbl.t) for lbl in ctx.labels.values()})
    t_index = {t: i for i, t in enumerate(all_t)}
    num_rounds = len(all_t)
    num_regions = len(REGION_LABELS)

    model.eval()
    device = next(model.parameters()).device
    eval_batch_size = 256
    wrong_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(graphs), eval_batch_size):
            chunk = graphs[start : start + eval_batch_size]
            eval_batch = Batch.from_data_list(chunk).to(device)
            preds = (model.forward_selected(eval_batch) > 0).float()
            targets = eval_batch.y.squeeze(-1)
            wrong_chunks.append((preds != targets).cpu().numpy().astype(bool))
    wrong = np.concatenate(wrong_chunks) if wrong_chunks else np.zeros(0, dtype=bool)

    activity_count = np.zeros((num_rounds, num_regions), dtype=np.int64)
    wrong_count = np.zeros((num_rounds, num_regions), dtype=np.int64)

    for shot_idx in range(batch.detections.shape[0]):
        fired = np.nonzero(batch.detections[shot_idx])[0]
        buckets_hit: set[tuple[int, int]] = set()
        for d in fired:
            lbl = ctx.labels[int(d)]
            r = t_index[int(lbl.t)]
            c = REGION_LABELS.index(lbl.region)
            buckets_hit.add((r, c))
        for r, c in buckets_hit:
            activity_count[r, c] += 1
            if wrong[shot_idx]:
                wrong_count[r, c] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        heatmap = np.where(activity_count > 0, wrong_count / np.maximum(activity_count, 1), np.nan)
    return heatmap, [float(t) for t in all_t]


def render_heatmap(heatmap: np.ndarray, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, max(3, heatmap.shape[0] * 0.3)))
    im = ax.imshow(heatmap, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(REGION_LABELS)))
    ax.set_xticklabels(REGION_LABELS, rotation=30, ha="right")
    ax.set_ylabel("round")
    ax.set_title("Per-round, per-region decoder error rate")
    fig.colorbar(im, ax=ax, label="P(wrong | activity)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
