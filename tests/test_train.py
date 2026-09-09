import torch
from torch_geometric.data import Batch, Data

from models.gnn_decoder import GNNDecoder
from models.train import load_checkpoint, save_checkpoint, train_epoch, train_step
from schema import NUM_NODE_FEATURES


def _synthetic_graph(num_nodes: int, head_id: int, label: float, seed: int) -> Data:
    torch.manual_seed(seed)
    x = torch.randn(num_nodes, NUM_NODE_FEATURES)
    src = torch.randint(0, num_nodes, (num_nodes * 2,))
    dst = torch.randint(0, num_nodes, (num_nodes * 2,))
    edge_index = torch.stack([src, dst], dim=0)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([[label]]), head_id=head_id, num_fired=num_nodes)


def _tiny_overfit_batch() -> Batch:
    graphs = [
        _synthetic_graph(4, head_id=0, label=1.0, seed=0),
        _synthetic_graph(5, head_id=0, label=0.0, seed=1),
        _synthetic_graph(3, head_id=1, label=1.0, seed=2),
        _synthetic_graph(6, head_id=1, label=0.0, seed=3),
    ]
    return Batch.from_data_list(graphs)


def test_one_training_step_strictly_decreases_loss_on_tiny_overfit_set():
    torch.manual_seed(0)
    model = GNNDecoder(num_layers=4, hidden_dim=32, heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    batch = _tiny_overfit_batch()

    losses = [train_step(model, optimizer, batch) for _ in range(20)]
    assert losses[-1] < losses[0]


def test_train_epoch_over_multiple_batches_returns_mean_loss():
    torch.manual_seed(0)
    model = GNNDecoder(num_layers=4, hidden_dim=32, heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    batches = [_tiny_overfit_batch(), _tiny_overfit_batch()]

    mean_loss = train_epoch(model, optimizer, batches)
    assert mean_loss > 0


def test_checkpoint_round_trips_model_and_optimizer_state(tmp_path):
    torch.manual_seed(0)
    model = GNNDecoder(num_layers=4, hidden_dim=32, heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    batch = _tiny_overfit_batch()
    for _ in range(3):
        train_step(model, optimizer, batch)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, path, epoch=3)

    fresh_model = GNNDecoder(num_layers=4, hidden_dim=32, heads=4)
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-2)
    epoch = load_checkpoint(fresh_model, fresh_optimizer, path)

    assert epoch == 3
    for p1, p2 in zip(model.parameters(), fresh_model.parameters()):
        assert torch.allclose(p1, p2)
