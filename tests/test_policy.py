from __future__ import annotations

import time

from finnet.config import PolicyConfig
from finnet.policy import Action, PolicyEngine, TokenBucket
from finnet.telemetry import TelemetrySnapshot


def make_policy(tmp_path, **overrides) -> PolicyEngine:
    defaults = {
        "allow_threshold": 0.2,
        "throttle_threshold": 0.4,
        "quarantine_threshold": 0.7,
        "terminate_threshold": 0.9,
        "token_capacity": 1,
        "token_refill_per_s": 0.0,
        "throttle_penalty_ms": 1.0,
        "violations_to_quarantine": 100,
        "firewall_audit_path": str(tmp_path / "firewall.jsonl"),
    }
    defaults.update(overrides)
    return PolicyEngine(PolicyConfig(**defaults))


def telemetry() -> TelemetrySnapshot:
    return TelemetrySnapshot(burst_velocity=3.0, payload_entropy=7.0)


def test_allow_and_throttle(tmp_path) -> None:
    policy = make_policy(tmp_path)
    now = time.monotonic()
    allow = policy.decide(subnet="192.0.2.0/24", score=0.1, telemetry=telemetry(), now=now)
    assert allow.action == Action.ALLOW
    throttle = policy.decide(
        subnet="192.0.2.0/24", score=0.5, telemetry=telemetry(), now=now
    )
    assert throttle.action == Action.THROTTLE
    assert throttle.delay_ms >= 1.0


def test_token_exhaustion_terminates_and_quarantines(tmp_path) -> None:
    policy = make_policy(tmp_path)
    now = time.monotonic()
    first = policy.decide(subnet="192.0.2.0/24", score=0.5, telemetry=telemetry(), now=now)
    assert first.action == Action.THROTTLE
    second = policy.decide(subnet="192.0.2.0/24", score=0.5, telemetry=telemetry(), now=now)
    assert second.action == Action.TERMINATE
    assert second.close_socket and not second.send_response
    assert policy.precheck("192.0.2.0/24", now=now) is not None


def test_quarantine_threshold_and_ttl(tmp_path) -> None:
    policy = make_policy(tmp_path, quarantine_ttl_s=2.0)
    now = time.monotonic()
    decision = policy.decide(
        subnet="198.51.100.0/24", score=0.8, telemetry=telemetry(), now=now
    )
    assert decision.action == Action.QUARANTINE
    assert policy.precheck("198.51.100.0/24", now=now) is not None
    assert policy.precheck("198.51.100.0/24", now=now + 3.0) is None
    audit = (tmp_path / "firewall.jsonl").read_text()
    assert "198.51.100.0/24" in audit
    assert '"operation": "block"' in audit


def test_repeat_offender_promoted(tmp_path) -> None:
    policy = make_policy(tmp_path, violations_to_quarantine=2, token_capacity=10)
    now = time.monotonic()
    for _ in range(2):
        decision = policy.decide(
            subnet="203.0.113.0/24", score=0.5, telemetry=telemetry(), now=now
        )
    assert decision.action == Action.QUARANTINE


def test_token_bucket_refills() -> None:
    bucket = TokenBucket(capacity=2, refill_per_s=1, now=0.0)
    assert bucket.consume(now=0.0)
    assert bucket.consume(now=0.0)
    assert not bucket.consume(now=0.0)
    assert bucket.consume(now=1.0)
