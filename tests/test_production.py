from __future__ import annotations

import time
from dataclasses import replace

from finnet.config import load_config
from finnet.engine import GatewayEngine
from finnet.iso8583 import ISO8583Codec, ISO8583Message
from finnet.policy import Action, PolicyEngine
from finnet.security import (
    ReplayGuard,
    attach_mac,
    message_fingerprint,
    verify_mac,
)
from finnet.telemetry import TelemetrySnapshot


def test_mac_roundtrip_and_mismatch() -> None:
    secret = b"test-secret"
    payload = b"0200" + b"a" * 40
    signed = attach_mac(payload, secret)
    assert verify_mac(signed, secret).valid
    assert not verify_mac(signed + b"x", secret).valid
    assert not verify_mac(signed[:-1] + b"x", secret).valid


def test_replay_guard_remembers_and_expires() -> None:
    guard = ReplayGuard(ttl_s=1.0, max_entries=10)
    guard.remember("abc", "payload", now=0.0)
    assert guard.check("abc", now=0.5) == "payload"
    assert guard.check("abc", now=2.0) is None


def test_message_fingerprint_ignores_pan() -> None:
    fields = {2: "4111111111111111", 11: "000001", 37: "RRN000001", 4: "000000010000"}
    first = message_fingerprint(fields)
    second = message_fingerprint({**fields, 2: "4000000000000002"})
    assert first == second


async def test_engine_replay_returns_cached_decision(tmp_path) -> None:
    config = load_config("configs/default.yaml")
    config = replace(
        config,
        model=replace(config.model, checkpoint=str(tmp_path / "missing.pt")),
        policy=replace(
            config.policy,
            allow_threshold=.99,
            step_up_threshold=.991,
            review_threshold=.992,
            throttle_threshold=.993,
            soft_decline_threshold=.994,
            quarantine_threshold=.995,
            terminate_threshold=.996,
            firewall_audit_path=str(tmp_path / "fw.jsonl"),
        ),
        log_path=str(tmp_path / "decisions.jsonl"),
    )
    engine = GatewayEngine(config)
    message = ISO8583Message(
        "0200",
        {
            2: "4111111111111111",
            3: "000000",
            4: "000000001000",
            11: "123456",
            37: "RRN000000001",
            41: "T0000001",
            42: "M00000000000001",
            49: "356",
        },
    )
    codec = ISO8583Codec()
    raw = codec.encode(message)
    first = await engine.process(message, peer_ip="127.0.0.1", raw_payload=raw)
    second = await engine.process(message, peer_ip="127.0.0.1", raw_payload=raw)
    assert first.idempotent_replay is False
    assert second.idempotent_replay is True
    assert second.score_source == "idempotent_replay"
    assert engine.metrics.counter_value("finnet_idempotent_replays_total") == 1
    await engine.aclose()


def _decision(policy: PolicyEngine, score: float):
    return policy.decide(
        subnet="192.0.2.0/24",
        score=score,
        telemetry=TelemetrySnapshot(),
        now=time.monotonic(),
    )


def test_decision_ladder_actions(tmp_path) -> None:
    policy = PolicyEngine(
        load_config("configs/default.yaml").policy
    )
    assert _decision(policy, 0.1).action == Action.ALLOW
    assert _decision(policy, 0.46).action == Action.STEP_UP
    assert _decision(policy, 0.57).action == Action.REVIEW
    assert _decision(policy, 0.7).action == Action.THROTTLE
    assert _decision(policy, 0.8).action == Action.SOFT_DECLINE
    assert _decision(policy, 0.9).action == Action.QUARANTINE
    assert _decision(policy, 0.99).action == Action.TERMINATE
