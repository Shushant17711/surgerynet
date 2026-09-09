"""Runs a real training session (E2 memory baseline, matching
experiments/e2_memory_baseline.py) logged to Weights & Biases.

NOTE: `anonymous="allow"` below is *not currently functional* — verified
empirically (see LIMITATIONS.md): the installed wandb SDK (0.29.0) prints
"The anonymous setting has no effect and will be removed in a future
version" and then hard-requires a real API key via `wandb login` or the
`WANDB_API_KEY` env var regardless. This script has not been run against
a live account. Do that first (`wandb login`) before running this.
"""

from __future__ import annotations

import argparse

import numpy as np
import stim
import torch
import wandb
from torch_geometric.data import Batch

from baselines.mwpm import MWPMBaseline
from data.generate import sample_shots
from data.to_graph import GraphBuildContext, batch_to_graphs
from models.gnn_decoder import GNNDecoder
from models.train import train_step


def main() -> str:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--p", type=float, default=0.008)
    parser.add_argument("--train-shots", type=int, default=3000)
    parser.add_argument("--test-shots", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run = wandb.init(
        project="surgerynet",
        anonymous="allow",
        config=vars(args),
        job_type="e2_memory_baseline",
    )

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=args.distance, rounds=args.distance,
        after_clifford_depolarization=args.p,
    )
    ctx = GraphBuildContext.build(circuit, patch_dimension=float(args.distance))

    rng = np.random.default_rng(args.seed)
    train_batch = sample_shots(circuit, shots=args.train_shots, seed=args.seed)
    train_graphs = batch_to_graphs(train_batch, ctx, observable_kind="spacelike", mask_surgery_features=True)

    model = GNNDecoder(num_layers=4, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    global_step = 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(train_graphs))
        for start in range(0, len(train_graphs), args.batch_size):
            idx = order[start : start + args.batch_size]
            mini_batch = Batch.from_data_list([train_graphs[i] for i in idx])
            train_step(model, optimizer, mini_batch, wandb_run=run, step=global_step)
            global_step += 1

    test_batch = sample_shots(circuit, shots=args.test_shots, seed=args.seed + 1)
    test_graphs = batch_to_graphs(test_batch, ctx, observable_kind="spacelike", mask_surgery_features=True)
    model.eval()
    with torch.no_grad():
        eval_batch = Batch.from_data_list(test_graphs)
        preds = (model.forward_selected(eval_batch) > 0).float()
        targets = eval_batch.y.squeeze(-1)
        gnn_rate = float((preds != targets).float().mean().item())

    mwpm = MWPMBaseline(circuit)
    mwpm_preds = mwpm.decode(test_batch.detections)
    mwpm_rate = float((mwpm_preds != test_batch.observables[:, 0]).mean())

    run.log({"eval/gnn_logical_error_rate": gnn_rate, "eval/mwpm_logical_error_rate": mwpm_rate})
    run.summary["gnn_logical_error_rate"] = gnn_rate
    run.summary["mwpm_logical_error_rate"] = mwpm_rate
    run.summary["gnn_beats_mwpm"] = gnn_rate < mwpm_rate

    url = run.url
    print(f"GNN logical error rate:  {gnn_rate:.4f}")
    print(f"MWPM logical error rate: {mwpm_rate:.4f}")
    print(f"W&B run URL: {url}")
    run.finish()
    return url


if __name__ == "__main__":
    main()
