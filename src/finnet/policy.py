"""Closed-loop mitigation policy: allow, throttle, quarantine, terminate."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

from .config import PolicyConfig
from .firewall import FirewallAdapter
from .telemetry import TelemetrySnapshot


class Action(str, Enum):
    ALLOW = "ALLOW"
    THROTTLE = "THROTTLE"
    QUARANTINE = "QUARANTINE"
    TERMINATE = "TERMINATE"


@dataclass(frozen=True)
class PolicyDecision:
    action: Action
    reason: str
    score: float
    response_code: str
    delay_ms: float = 0.0
    close_socket: bool = False
    send_response: bool = True
    quarantine_cidr: str | None = None


@dataclass
class QuarantineEntry:
    cidr: str
    reason: str
    score: float
    created_at: float
    expires_at: float


class TokenBucket:
    def __init__(self, capacity: float, refill_per_s: float, *, now: float | None = None) -> None:
        self.capacity = max(1.0, capacity)
        self.refill_per_s = max(0.0, refill_per_s)
        self.tokens = self.capacity
        self.updated_at = time.monotonic() if now is None else now
        self._lock = threading.Lock()

    def consume(self, amount: float = 1.0, *, now: float | None = None) -> bool:
        at = time.monotonic() if now is None else now
        with self._lock:
            elapsed = max(0.0, at - self.updated_at)
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_s)
            self.updated_at = at
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False


class TokenBucketRegistry:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> TokenBucket:
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(self.config.token_capacity, self.config.token_refill_per_s)
                self._buckets[key] = bucket
            return bucket

    def consume(self, key: str, *, now: float | None = None) -> bool:
        return self.get(key).consume(now=now)


class QuarantineStore:
    def __init__(self) -> None:
        self._entries: dict[str, QuarantineEntry] = {}
        self._lock = threading.RLock()

    def add(
        self,
        cidr: str,
        *,
        reason: str,
        score: float,
        ttl_s: float,
        now: float | None = None,
    ) -> QuarantineEntry:
        at = time.monotonic() if now is None else now
        entry = QuarantineEntry(
            cidr=cidr,
            reason=reason,
            score=score,
            created_at=at,
            expires_at=at + ttl_s,
        )
        with self._lock:
            self._entries[cidr] = entry
        return entry

    def is_blocked(self, cidr: str, *, now: float | None = None) -> QuarantineEntry | None:
        at = time.monotonic() if now is None else now
        with self._lock:
            self._expire_locked(at)
            return self._entries.get(cidr)

    def remove(self, cidr: str) -> bool:
        with self._lock:
            return self._entries.pop(cidr, None) is not None

    def snapshot(self, *, now: float | None = None) -> list[QuarantineEntry]:
        at = time.monotonic() if now is None else now
        with self._lock:
            self._expire_locked(at)
            return list(self._entries.values())

    def _expire_locked(self, now: float) -> None:
        expired = [cidr for cidr, entry in self._entries.items() if entry.expires_at <= now]
        for cidr in expired:
            self._entries.pop(cidr, None)


class PolicyEngine:
    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()
        self.firewall = FirewallAdapter(self.config)
        self.buckets = TokenBucketRegistry(self.config)
        self.quarantine = QuarantineStore()
        self._violations: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.RLock()

    def precheck(self, subnet: str, *, now: float | None = None) -> PolicyDecision | None:
        entry = self.quarantine.is_blocked(subnet, now=now)
        if entry is None:
            return None
        return PolicyDecision(
            action=Action.TERMINATE,
            reason=f"subnet already quarantined: {entry.reason}",
            score=entry.score,
            response_code="05",
            close_socket=True,
            send_response=False,
            quarantine_cidr=subnet,
        )

    def decide(
        self,
        *,
        subnet: str,
        score: float,
        telemetry: TelemetrySnapshot,
        now: float | None = None,
    ) -> PolicyDecision:
        at = time.monotonic() if now is None else now
        score = min(1.0, max(0.0, float(score)))

        if score >= self.config.terminate_threshold:
            self._quarantine(subnet, score, "risk >= terminate threshold", at)
            return PolicyDecision(
                action=Action.TERMINATE,
                reason="high-confidence fraud; socket teardown and quarantine",
                score=score,
                response_code="05",
                close_socket=True,
                send_response=False,
                quarantine_cidr=subnet,
            )

        if score >= self.config.quarantine_threshold:
            self._quarantine(subnet, score, "risk >= quarantine threshold", at)
            return PolicyDecision(
                action=Action.QUARANTINE,
                reason="quarantined offending subnet",
                score=score,
                response_code="05",
                close_socket=True,
                send_response=True,
                quarantine_cidr=subnet,
            )

        if score >= self.config.throttle_threshold:
            violations = self._record_violation(subnet, at)
            if violations >= self.config.violations_to_quarantine:
                self._quarantine(subnet, score, "repeated suspicious behaviour", at)
                return PolicyDecision(
                    action=Action.QUARANTINE,
                    reason="repeat offender promoted to quarantine",
                    score=score,
                    response_code="05",
                    close_socket=True,
                    send_response=True,
                    quarantine_cidr=subnet,
                )
            if not self.buckets.consume(subnet, now=at):
                self._quarantine(subnet, score, "token bucket exhausted", at)
                return PolicyDecision(
                    action=Action.TERMINATE,
                    reason="rate limit exhausted; attacking socket terminated",
                    score=score,
                    response_code="05",
                    close_socket=True,
                    send_response=False,
                    quarantine_cidr=subnet,
                )
            # A short delay is a network-visible throttle, while the response
            # still reaches the client for observability.
            delay = self.config.throttle_penalty_ms
            if telemetry.burst_velocity > 0:
                delay += min(30.0, telemetry.burst_velocity * 1.5)
            return PolicyDecision(
                action=Action.THROTTLE,
                reason="suspicious subnet rate-limited by token bucket",
                score=score,
                response_code="96",
                delay_ms=delay,
                close_socket=False,
                send_response=True,
            )

        return PolicyDecision(
            action=Action.ALLOW,
            reason="risk below mitigation threshold",
            score=score,
            response_code="00",
        )

    def _record_violation(self, subnet: str, now: float) -> int:
        with self._lock:
            window = self._violations[subnet]
            cutoff = now - self.config.violation_window_s
            while window and window[0] < cutoff:
                window.popleft()
            window.append(now)
            return len(window)

    def _quarantine(self, subnet: str, score: float, reason: str, now: float) -> None:
        self.quarantine.add(
            subnet,
            reason=reason,
            score=score,
            ttl_s=self.config.quarantine_ttl_s,
            now=now,
        )
        self.firewall.block(
            subnet,
            reason=reason,
            score=score,
            ttl_s=self.config.quarantine_ttl_s,
        )

    def status(self) -> dict[str, object]:
        entries = self.quarantine.snapshot()
        return {
            "quarantined_subnets": len(entries),
            "firewall_backend": self.config.firewall_backend,
            "os_firewall_enabled": self.config.enable_os_firewall,
            "quarantine": [
                {
                    "cidr": entry.cidr,
                    "reason": entry.reason,
                    "score": entry.score,
                    "remaining_s": max(0.0, entry.expires_at - time.monotonic()),
                }
                for entry in entries
            ],
        }
