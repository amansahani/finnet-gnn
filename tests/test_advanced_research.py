from __future__ import annotations

from dataclasses import replace

from finnet.attacks import apply_attack
from finnet.config import GraphConfig
from finnet.data.synthetic import generate_synthetic_events
from finnet.graph import DynamicTransactionGraph
from finnet.history import HISTORY_FEATURE_DIM, HISTORY_LENGTH
from finnet.rejection import simulate_rejection
from finnet.telemetry import FlowTelemetry
from finnet.training import (
    split_events_card_holdout,
    split_events_device_holdout,
    split_events_merchant_holdout,
)


def _events(count: int = 2_000, seed: int = 7):
    return generate_synthetic_events(count, seed=seed)


def test_entity_holdout_splits_are_disjoint() -> None:
    events = _events(2_000)
    for entity, splitter in (
        ("card", split_events_card_holdout),
        ("device", split_events_device_holdout),
        ("merchant", split_events_merchant_holdout),
    ):
        train, test = splitter(events, 0.8)
        attr = {
            "card": "card_token",
            "device": "device_id",
            "merchant": "merchant_id",
        }[entity]
        train_entities = {getattr(event, attr) for event in train}
        test_entities = {getattr(event, attr) for event in test}
        assert train_entities
        assert test_entities
        assert train_entities.isdisjoint(test_entities)
        assert any(event.label for event in train)
        assert any(event.label for event in test)


def test_rejection_simulation_is_seeded_and_weighted() -> None:
    events = _events(900, seed=3)
    first = simulate_rejection(events, seed=88)
    second = simulate_rejection(events, seed=88)
    assert first.approved_events == second.approved_events
    assert first.rejected_count > 0
    assert first.approved_weights
    assert abs(sum(first.approved_weights) / len(first.approved_weights) - 1.0) < 1e-6
    assert first.report["approved"] == len(first.approved_events)
    assert first.report["rejected"] == first.rejected_count


def test_attack_transforms_preserve_fraud_labels() -> None:
    events = _events(900, seed=5)
    for attack in ("amount_noise", "merchant_rotation", "device_rotation", "subnet_hopping", "amount_split"):
        transformed = apply_attack(events, attack, seed=99)
        assert len(transformed) >= len(events)
        assert sum(event.label for event in transformed) >= sum(event.label for event in events)
        assert len({event.event_id for event in transformed}) == len(transformed)


def test_adversarial_time_scatter_reorders_deterministically() -> None:
    transformed = apply_attack(_events(100, seed=9), "time_scatter", seed=4)
    assert transformed == sorted(transformed, key=lambda event: (event.timestamp, event.event_id))


def test_approximate_fanout_tracks_distinct_cards() -> None:
    graph = DynamicTransactionGraph(GraphConfig(approx_fanout=True, approx_fanout_precision=6))
    telemetry = FlowTelemetry()
    fanouts = []
    for index in range(8):
        events = _events(100, seed=11)
        template = events[0]

        event = replace(
            template,
            event_id=f"approx_{index}",
            timestamp=100.0 + index,
            card_token=f"card_{index}",
        )
        snapshot = telemetry.update(event.flow_key, event.raw_payload, at=event.timestamp)
        added = graph.add_transaction(event, snapshot, now=event.timestamp)
        fanouts.append(added.meta.device_card_fanout)
    assert fanouts[-1] >= 6
    assert fanouts[-1] <= 12


def test_snapshot_history_is_padded_and_visible() -> None:

    graph = DynamicTransactionGraph(GraphConfig())
    telemetry = FlowTelemetry()
    template = _events(100, seed=11)[0]
    for index in range(3):
        event = replace(
            template,
            event_id=f"history_{index}",
            timestamp=100.0 + index,
            card_token="card_history_test",
            device_id="T_HISTORY",
            merchant_id="M_HISTORY",
            terminal_id="T_HISTORY",
        )
        snapshot = telemetry.update(event.flow_key, event.raw_payload, at=event.timestamp)
        added = graph.add_transaction(event, snapshot, now=event.timestamp)
    assert added.meta.history is not None
    assert added.meta.history.shape == (HISTORY_LENGTH, HISTORY_FEATURE_DIM)
    assert added.meta.history_mask is not None
    assert added.meta.history_mask.sum() >= 1.0
