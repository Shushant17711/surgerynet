"""Training loop for GNNDecoder (design doc §5.3, Task 4.3).

Loss is BCE-with-logits on whichever head each graph's `head_id` selects
(data/to_graph.py's per-shot single-label design: a batch generally mixes
spacelike- and timelike-labeled graphs, each routed to its own head via
`GNNDecoder.forward_selected`) — this is the "combined loss over both
heads" the design doc asks for, just computed per-example rather than as
two separately-batched sub-losses, since a shot never has both labels.

`wandb_run` is optional everywhere it appears: pass `None` (the default)
and none of this code touches the `wandb` package at all, so existing
callers/tests are unaffected. Pass an active run (from `wandb.init(...)`)
to get per-step loss logged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from models.gnn_decoder import GNNDecoder


def train_step(
    model: GNNDecoder,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    loss_fn: nn.Module | None = None,
    wandb_run: Any | None = None,
    step: int | None = None,
) -> float:
    loss_fn = loss_fn or nn.BCEWithLogitsLoss()
    model.train()
    optimizer.zero_grad()
    logits = model.forward_selected(batch)
    targets = batch.y.squeeze(-1)
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()
    loss_value = float(loss.item())
    if wandb_run is not None:
        wandb_run.log({"train/loss": loss_value}, step=step)
    return loss_value


def train_epoch(
    model: GNNDecoder,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[Batch],
    loss_fn: nn.Module | None = None,
    wandb_run: Any | None = None,
    epoch: int | None = None,
) -> float:
    total = 0.0
    n = 0
    for batch in batches:
        total += train_step(model, optimizer, batch, loss_fn, wandb_run=wandb_run)
        n += 1
    mean_loss = total / max(1, n)
    if wandb_run is not None:
        wandb_run.log({"train/epoch_mean_loss": mean_loss}, step=epoch)
    return mean_loss


def save_checkpoint(
    model: GNNDecoder, optimizer: torch.optim.Optimizer, path: str | Path, epoch: int
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(
    model: GNNDecoder, optimizer: torch.optim.Optimizer | None, path: str | Path
) -> int:
    """Returns the saved epoch number. `optimizer=None` loads model weights
    only (e.g. for evaluation-only scripts that never resume training)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"]
