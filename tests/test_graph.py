from __future__ import annotations

import numpy as np

from finnet.config import GraphConfig
from finnet.data.common import TransactionEvent
from finnet.data.synthetic import generate_synthetic_events
from finnet.features import INPUT_DIM, NodeType, Relation
from finnet.graph import DynamicTransactionGraph
from finnet.telemetry import FlowTelemetry


def event(index: int, card: str, *, device: str = "T0000001", subnet: str = "192.0.2.0/24"):
    return TransactionEvent(
        event_id=f"e{index}",
        timestamp=100.0 + index,
        src_ip="192.0.2.10",
        subnet=subnet,
        device_id=device,
        card_token=card,
        merchant_id="M00000000000001",
        amount=10.0 + index,
        label=0,
        raw_payload=b"payload" * 10,
    )


def test_snapshot_is_typed_and_connected() -> None:
    graph = DynamicTransactionGraph(GraphConfig())
    telemetry = FlowTelemetry()
    ev = event(0, "card_a")
    snapshot = telemetry.update(ev.flow_key, ev.raw_payload, at=ev.timestamp)
    added = graph.add_transaction(ev, snapshot, now=ev.timestamp)
    exported = graph.snapshot(added.node_id)
    assert exported.x.shape == (5, INPUT_DIM)
    assert exported.target_index == 0
    assert exported.node_types[0] == NodeType.TRANSACTION
    assert exported.edge_index.shape[1] == 14  # 7 undirected edges, both directions
    assert set(exported.edge_type.tolist()) == {
        int(Relation.TX_SUBNET),
        int(Relation.TX_DEVICE),
        int(Relation.TX_CARD),
        int(Relation.TX_MERCHANT),
        int(Relation.SUBNET_DEVICE),
        int(Relation.DEVICE_CARD),
        int(Relation.CARD_MERCHANT),
    }
    assert np.isfinite(exported.x).all()


def test_new_entity_flags_and_card_fanout() -> None:
    graph = DynamicTransactionGraph(GraphConfig(max_snapshot_nodes=64))
    telemetry = FlowTelemetry()
    fanouts = []
    for index in range(7):
        ev = event(index, f"card_{index}")
        snapshot = telemetry.update(ev.flow_key, ev.raw_payload, at=ev.timestamp)
        added = graph.add_transaction(ev, snapshot, now=ev.timestamp)
        if index == 0:
            assert added.meta.new_subnet and added.meta.new_device
            assert added.meta.new_card and added.meta.new_merchant
        fanouts.append(added.meta.device_card_fanout)
    assert fanouts == list(range(1, 8))
    exported = graph.snapshot(added.node_id)
    assert exported.x[exported.target_index, 22] > 1.0


def test_snapshot_hops_and_risk_propagation() -> None:
    events = generate_synthetic_events(100)
    graph = DynamicTransactionGraph(GraphConfig(max_snapshot_nodes=32))
    telemetry = FlowTelemetry()
    last = None
    for ev in events:
        snap = telemetry.update(ev.flow_key, ev.raw_payload, at=ev.timestamp)
        last = graph.add_transaction(ev, snap, now=ev.timestamp)
    assert last is not None
    graph.update_risk(last.node_id, 0.9)
    exported = graph.snapshot(last.node_id)
    assert exported.num_nodes <= 32
    assert exported.x[exported.target_index, 4] > 0.0
    assert graph.stats()["transaction"] >= 1


def test_blocked_subnet_feature() -> None:
    graph = DynamicTransactionGraph(GraphConfig())
    ev = event(0, "card_a")
    telemetry = FlowTelemetry()
    snap = telemetry.update(ev.flow_key, ev.raw_payload, at=ev.timestamp)
    added = graph.add_transaction(ev, snap, now=ev.timestamp)
    assert graph.mark_subnet_blocked(ev.subnet, until=1000.0)
    exported = graph.snapshot(added.node_id, now=200.0)
    subnet_index = int(np.where(exported.node_types == NodeType.SUBNET)[0][0])
    assert exported.x[subnet_index, 6] == 1.0
