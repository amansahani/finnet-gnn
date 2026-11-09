from __future__ import annotations

from collections import Counter

from finnet.data.synthetic import generate_synthetic_events
from finnet.training import split_events_time


def test_synthetic_stream_has_all_attack_types() -> None:
    events = generate_synthetic_events(3000, seed=3)
    counts = Counter(event.attack_type for event in events)
    assert counts["normal"] > 0
    assert counts["card_testing"] > 0
    assert counts["botnet"] > 0
    assert counts["account_takeover"] > 0
    assert sum(event.label for event in events) == len(events) - counts["normal"]
    assert all(event.subnet.endswith("/24") for event in events)
    assert all(len(event.device_id) <= 8 for event in events)
    assert len({event.event_id for event in events}) == len(events)


def test_temporal_split_contains_fraud_on_both_sides() -> None:
    events = generate_synthetic_events(2000, seed=4)
    train, test = split_events_time(events, 0.8)
    assert len(train) > 0 and len(test) > 0
    assert any(event.label for event in train)
    assert any(event.label for event in test)
    assert max(event.timestamp for event in train) <= min(
        event.timestamp for event in test
    )
