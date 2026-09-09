import pytest
import torch

from models.cnn_decoder import CNNDecoder
from models.errors import ShapeInfeasibleError
from models.mlp_decoder import MLPDecoder


def test_mlp_forward_pass_at_training_shape():
    model = MLPDecoder(input_dim=24)
    out = model(torch.randn(8, 24))
    assert out.shape == (8,)


def test_mlp_raises_shape_infeasible_on_merged_patch_size():
    # trained at a memory-experiment patch size, then handed a bigger
    # merged-patch-sized input -- exactly design doc §5.4's H1 scenario.
    model = MLPDecoder(input_dim=24)
    merged_size_input = torch.randn(8, 60)
    with pytest.raises(ShapeInfeasibleError):
        model(merged_size_input)


def test_cnn_forward_pass_at_training_shape():
    model = CNNDecoder(grid_shape=(3, 4, 4))
    out = model(torch.randn(5, 3, 4, 4))
    assert out.shape == (5,)


def test_cnn_raises_shape_infeasible_on_merged_patch_grid():
    model = CNNDecoder(grid_shape=(3, 4, 4))
    merged_grid = torch.randn(5, 3, 4, 9)  # grid widened by the merge
    with pytest.raises(ShapeInfeasibleError):
        model(merged_grid)
