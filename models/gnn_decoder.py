"""The GNN decoder (design doc §5.3): message-passing over detection-event
graphs, two output heads (spacelike, timelike).

A shot with zero fired detectors is common and valid (especially at low
physical error rates) and must not silently vanish from a batch: PyG's
global pooling only emits a row for batch indices actually present in
`batch.batch`, so an empty graph's row would otherwise be dropped and
misalign predictions against `y`. Passing `size=batch.num_graphs`
(verified empirically, see the pooling call below) fixes this.

**Edge features (post-hoc addition).** `edge_dim`, when set, wires each
conv layer up to accept `batch.edge_attr` — in particular the DEM-derived
log-odds weight `data/to_graph.py` now attaches to every edge, which is
literally the number MWPM's own matching weight comes from. Both `GATConv`
and `TransformerConv` support `edge_dim` natively. Left `None` by default
so old call sites/tests building edge-attr-free synthetic graphs are
unaffected — real experiment scripts pass `schema.NUM_EDGE_FEATURES`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, TransformerConv, global_max_pool, global_mean_pool

from schema import NUM_NODE_FEATURES, OUTPUT_HEADS

_CONV_TYPES = {"gat": GATConv, "transformer": TransformerConv}


class GNNDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = NUM_NODE_FEATURES,
        hidden_dim: int = 64,
        num_layers: int = 4,
        conv_type: str = "gat",
        heads: int = 4,
        use_norm: bool = True,
        edge_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not 4 <= num_layers <= 6:
            raise ValueError(f"design doc §5.3 calls for 4-6 message-passing layers, got {num_layers}")
        if conv_type not in _CONV_TYPES:
            raise ValueError(f"conv_type must be one of {list(_CONV_TYPES)}, got {conv_type!r}")
        ConvCls = _CONV_TYPES[conv_type]

        self.use_norm = use_norm
        self.edge_dim = edge_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.convs = nn.ModuleList(
            ConvCls(hidden_dim, hidden_dim // heads, heads=heads, concat=True, edge_dim=edge_dim)
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            (nn.LayerNorm(hidden_dim) if use_norm else nn.Identity()) for _ in range(num_layers)
        )
        self.head_spacelike = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.head_timelike = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        x = self.input_proj(batch.x)
        edge_attr = batch.edge_attr if self.edge_dim is not None else None
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, batch.edge_index, edge_attr=edge_attr) + x).relu()

        mean_pool = global_mean_pool(x, batch.batch, size=batch.num_graphs)
        max_pool = global_max_pool(x, batch.batch, size=batch.num_graphs)
        pooled = torch.cat([mean_pool, max_pool], dim=1)

        return {
            "spacelike": self.head_spacelike(pooled).squeeze(-1),
            "timelike": self.head_timelike(pooled).squeeze(-1),
        }

    def forward_selected(self, batch: Batch) -> torch.Tensor:
        """Logits for whichever head each graph in the batch actually has a
        label for (batch.head_id, one entry per graph — see
        data/to_graph.py's `head_id`, since spacelike/timelike shots
        never carry both labels at once)."""
        outputs = self.forward(batch)
        stacked = torch.stack([outputs[h] for h in OUTPUT_HEADS], dim=1)  # (N, 2)
        return stacked.gather(1, batch.head_id.view(-1, 1)).squeeze(1)
