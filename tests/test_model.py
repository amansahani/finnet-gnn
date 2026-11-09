from __future__ import annotations

import torch

from finnet.config import ModelConfig
from finnet.data.common import TransactionEvent
from finnet.features import INPUT_DIM
from finnet.graph import DynamicTransactionGraph
from finnet.model import RiskGNN
from finnet.telemetry import FlowTelemetry
from finnet.training import GraphSample, collate_graphs


def _snapshot():
    graph = DynamicTransactionGraph()
    telemetry = FlowTelemetry()
    snapshots = []
    labels = []
    for index in range(3):
        event = TransactionEvent(
            event_id=f"t{index}",
            timestamp=10.0 + index,
            src_ip="192.0.2.10",
            subnet="192.0.2.0/24",
            device_id=f"T{index:07d}",
            card_token=f"card_{index}",
            merchant_id="M00000000000001",
            amount=50.0,
            label=index % 2,
        )
        telemetry_snapshot = telemetry.update(
            event.flow_key, event.raw_payload or b"x", at=event.timestamp
        )
        added = graph.add_transaction(event, telemetry_snapshot, now=event.timestamp)
        snapshots.append(
            GraphSample(
                snapshot=graph.snapshot(added.node_id),
                label=event.label,
                attack_type="normal",
                event_id=event.event_id,
            )
        )
        labels.append(event.label)
    return snapshots, labels


def test_collate_and_forward() -> None:
    samples, labels = _snapshot()
    batch = collate_graphs(samples)
    assert batch["x"].shape[1] == INPUT_DIM
    assert batch["target_index"].shape == (3,)
    assert batch["labels"].tolist() == labels

    model = RiskGNN(ModelConfig(input_dim=INPUT_DIM, hidden_dim=32, num_layers=2))
    logits = model(
        batch["x"],
        batch["node_types"],
        batch["edge_index"],
        batch["edge_type"],
        batch["edge_weight"],
        batch["target_index"],
    )
    assert logits.shape == (3,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, batch["labels"]
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_empty_edge_graph_forward() -> None:
    model = RiskGNN(ModelConfig(hidden_dim=16))
    x = torch.zeros((1, INPUT_DIM))
    logits = model(
        x,
        torch.tensor([4]),
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0),
        torch.tensor([0]),
    )
    assert torch.isfinite(logits).all()
