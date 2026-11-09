from __future__ import annotations

import pytest

from finnet.config import TelemetryConfig
from finnet.telemetry import FlowTelemetry, shannon_entropy


def test_entropy_bounds() -> None:
    assert shannon_entropy(b"") == 0.0
    assert shannon_entropy(b"aaaaaaaa") == 0.0
    assert shannon_entropy(bytes(range(256))) == 8.0


def test_flow_velocity_and_jitter() -> None:
    tracker = FlowTelemetry(TelemetryConfig(max_flows=10))
    key = ("192.0.2.0/24", "T0000001")
    snapshots = [
        tracker.update(key, b"a" * 100, at=10.0),
        tracker.update(key, b"a" * 100, at=10.2),
        tracker.update(key, b"a" * 100, at=10.4),
    ]
    assert snapshots[-1].flow_count == 3
    assert snapshots[-1].burst_velocity == 3
    assert snapshots[-1].inter_arrival_ms == pytest.approx(200.0)
    assert snapshots[-1].jitter_ms > 0
    assert snapshots[-1].payload_entropy == 0.0
    assert snapshots[-1].flow_age_s == pytest.approx(0.4)


def test_rtt_ewma_and_bounded_flows() -> None:
    tracker = FlowTelemetry(TelemetryConfig(max_flows=2, rtt_alpha=0.5))
    for index in range(5):
        tracker.update(("subnet", f"device-{index}"), b"x", at=float(index), rtt_ms=20.0)
    assert len(tracker) <= 2
    snapshot = tracker.update(("subnet", "device-x"), b"x", at=10.0, rtt_ms=40.0)
    assert snapshot.rtt_ms == 20.0
