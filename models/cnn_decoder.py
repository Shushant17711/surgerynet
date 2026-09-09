"""Fixed-input CNN decoder over a syndrome grid (design doc §5.4). Exists
to demonstrate H1: the grid shape changes at merge time (a merged patch is
spatially bigger than either standalone patch), so a CNN sized for one
patch cannot even be *evaluated* on the merge window — see
ShapeInfeasibleError."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.errors import ShapeInfeasibleError


class CNNDecoder(nn.Module):
    def __init__(self, grid_shape: tuple[int, int, int], hidden_channels: int = 32) -> None:
        """`grid_shape` is (rounds, height, width) of the syndrome volume
        this instance is built for."""
        super().__init__()
        self.grid_shape = tuple(grid_shape)
        self.conv = nn.Sequential(
            nn.Conv3d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        if tuple(grid.shape[1:]) != self.grid_shape:
            raise ShapeInfeasibleError(
                f"CNNDecoder was built for grid_shape={self.grid_shape}, "
                f"got a tensor with per-sample shape {tuple(grid.shape[1:])}"
            )
        x = self.conv(grid.float().unsqueeze(1))
        x = self.pool(x).flatten(1)
        return self.head(x).squeeze(-1)
