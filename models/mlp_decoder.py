"""Fixed-input MLP decoder over a flattened syndrome vector (design doc
§5.4). Exists to demonstrate H1: trained at one patch size, it cannot even
be *evaluated* on a merged (bigger) patch — see ShapeInfeasibleError."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.errors import ShapeInfeasibleError


class MLPDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, detections: torch.Tensor) -> torch.Tensor:
        if detections.shape[-1] != self.input_dim:
            raise ShapeInfeasibleError(
                f"MLPDecoder was built for input_dim={self.input_dim}, "
                f"got a tensor with last dimension {detections.shape[-1]}"
            )
        return self.net(detections.float()).squeeze(-1)
