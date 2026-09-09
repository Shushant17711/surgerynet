import torch
from torch_geometric.data import Batch, Data

from models.gnn_decoder import GNNDecoder
from schema import NUM_NODE_FEATURES, OUTPUT_HEADS


def _synthetic_graph(num_nodes: int, head_id: int, seed: int) -> Data:
    torch.manual_seed(seed)
    x = torch.randn(num_nodes, NUM_NODE_FEATURES)
    if num_nodes > 1:
        src = torch.randint(0, num_nodes, (num_nodes * 2,))
        dst = torch.randint(0, num_nodes, (num_nodes * 2,))
        edge_index = torch.stack([src, dst], dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([[0.0]]),
        head_id=head_id,
        num_fired=num_nodes,
    )


def test_forward_pass_over_variable_sized_graphs_including_empty():
    graphs = [
        _synthetic_graph(0, head_id=0, seed=0),  # no fired detectors -- must not vanish
        _synthetic_graph(1, head_id=1, seed=1),
        _synthetic_graph(5, head_id=0, seed=2),
        _synthetic_graph(12, head_id=1, seed=3),
    ]
    batch = Batch.from_data_list(graphs)
    model = GNNDecoder(num_layers=4, hidden_dim=64, heads=4)
    outputs = model(batch)

    assert set(outputs.keys()) == set(OUTPUT_HEADS)
    for head in OUTPUT_HEADS:
        assert outputs[head].shape == (len(graphs),)
        assert torch.isfinite(outputs[head]).all()


def test_forward_selected_routes_each_graph_to_its_own_head():
    graphs = [
        _synthetic_graph(3, head_id=0, seed=0),
        _synthetic_graph(4, head_id=1, seed=1),
    ]
    batch = Batch.from_data_list(graphs)
    model = GNNDecoder(num_layers=4, hidden_dim=32, heads=4)

    selected = model.forward_selected(batch)
    full = model(batch)
    assert selected.shape == (2,)
    assert torch.allclose(selected[0], full["spacelike"][0])
    assert torch.allclose(selected[1], full["timelike"][1])


def test_rejects_layer_count_outside_4_to_6():
    import pytest

    with pytest.raises(ValueError):
        GNNDecoder(num_layers=2)
    with pytest.raises(ValueError):
        GNNDecoder(num_layers=8)
