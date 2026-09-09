"""E8 — latency (design doc §6 E8): merged-patch decoding time vs patch
size, for the GNN (CPU and GPU) against MWPM. "GNN inference parallelizes;
matching is more sequential."
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import torch
from torch_geometric.data import Batch

from baselines.mwpm import MWPMBaseline
from circuits.configurations import k_to_distance
from circuits.lattice_surgery import load_generated_circuit
from data.generate import sample_shots
from data.to_graph import GraphBuildContext, batch_to_graphs
from experiments.common import ExperimentResult, generate_surgery_circuit
from models.gnn_decoder import GNNDecoder


def measure_gnn_latency(model: GNNDecoder, graphs: list, device: str, repeats: int = 3) -> float:
    """Mean wall-clock seconds per full-batch forward pass. `graphs`
    should already be sized for one target patch/merge size — the sweep
    varies `graphs`, not batch composition, per shot count."""
    model = model.to(device)
    model.eval()
    batch = Batch.from_data_list(graphs).to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(repeats):
            model(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed / repeats


def measure_mwpm_latency(baseline: MWPMBaseline, detections, repeats: int = 3) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        baseline.decode(detections)
    elapsed = time.perf_counter() - start
    return elapsed / repeats


def run_latency_sweep(
    ks: tuple[int, ...] = (1, 2),
    p: float = 0.01,
    shots: int = 500,
    devices: tuple[str, ...] = ("cpu", "cuda"),
    seed: int = 0,
) -> list[ExperimentResult]:
    devices = tuple(d for d in devices if d != "cuda" or torch.cuda.is_available())
    results: list[ExperimentResult] = []

    with tempfile.TemporaryDirectory() as tmp:
        for k in ks:
            out_path = Path(tmp) / f"k{k}_spacelike.stim"
            generate_surgery_circuit(k, p, "spacelike", out_path)
            circuit = load_generated_circuit(out_path)
            ctx = GraphBuildContext.build(circuit, patch_dimension=float(k_to_distance(k)))

            batch = sample_shots(circuit, shots=shots, seed=seed)
            graphs = batch_to_graphs(batch, ctx, observable_kind="spacelike")

            model = GNNDecoder(num_layers=4, hidden_dim=64)
            for device in devices:
                latency = measure_gnn_latency(model, graphs, device)
                results.append(
                    ExperimentResult(
                        experiment="E8", decoder="gnn", observable_kind="spacelike", k=k, p=p, seed=seed,
                        latency_seconds=latency, extra={"device": device, "shots": shots},
                    )
                )

            mwpm = MWPMBaseline(circuit)
            mwpm_latency = measure_mwpm_latency(mwpm, batch.detections)
            results.append(
                ExperimentResult(
                    experiment="E8", decoder="mwpm", observable_kind="spacelike", k=k, p=p, seed=seed,
                    latency_seconds=mwpm_latency, extra={"device": "cpu", "shots": shots},
                )
            )

    return results
