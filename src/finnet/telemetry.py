"""Online network-flow telemetry extracted at the gateway.

A flow is keyed by (client subnet, device fingerprint). Card and merchant are
transaction attributes, not flow identifiers: a carding bot that rotates cards
must still accumulate velocity on its device and subnet.

RTT is an application-layer proxy when a trusted upstream is unavailable. The
ISO private field 62 timestamp is treated as untrusted and clamped by the codec.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass, field, replace

from .config import TelemetryConfig


@dataclass(frozen=True)
class TelemetrySnapshot:
    inter_arrival_ms: float = 0.0
    jitter_ms: float = 0.0
    burst_velocity: float = 0.0
    arrival_rate: float = 0.0
    payload_entropy: float = 0.0
    rtt_ms: float = 0.0
    flow_age_s: float = 0.0
    flow_count: int = 0
    bytes_mean: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "inter_arrival_ms": self.inter_arrival_ms,
            "jitter_ms": self.jitter_ms,
            "burst_velocity": self.burst_velocity,
            "arrival_rate": self.arrival_rate,
            "payload_entropy": self.payload_entropy,
            "rtt_ms": self.rtt_ms,
            "flow_age_s": self.flow_age_s,
            "flow_count": float(self.flow_count),
            "bytes_mean": self.bytes_mean,
        }


@dataclass
class _FlowState:
    first_seen: float
    last_seen: float
    count: int = 0
    last_interval_s: float = 0.0
    jitter_ms: float = 0.0
    rtt_ms: float = 0.0
    bytes_mean: float = 0.0
    arrival_times: deque[float] = field(default_factory=lambda: deque(maxlen=512))

    def observe(
        self,
        at: float,
        payload_len: int,
        rtt_ms: float | None,
        cfg: TelemetryConfig,
    ) -> TelemetrySnapshot:
        self.count += 1
        interval_s = max(0.0, at - self.last_seen) if self.count > 1 else 0.0
        if self.count > 2:
            delta_ms = abs(interval_s - self.last_interval_s) * 1000.0
            self.jitter_ms = (
                cfg.jitter_alpha * delta_ms + (1.0 - cfg.jitter_alpha) * self.jitter_ms
            )
        if rtt_ms is not None:
            self.rtt_ms = cfg.rtt_alpha * rtt_ms + (1.0 - cfg.rtt_alpha) * self.rtt_ms
        self.bytes_mean += (payload_len - self.bytes_mean) / self.count
        self.last_interval_s = interval_s
        self.last_seen = at
        self.arrival_times.append(at)

        while self.arrival_times and at - self.arrival_times[0] > cfg.rate_window_s:
            self.arrival_times.popleft()
        burst = sum(1 for ts in self.arrival_times if at - ts <= cfg.velocity_window_s)
        rate = len(self.arrival_times) / cfg.rate_window_s if self.arrival_times else 0.0
        return TelemetrySnapshot(
            inter_arrival_ms=interval_s * 1000.0,
            jitter_ms=self.jitter_ms,
            burst_velocity=float(burst),
            arrival_rate=rate,
            flow_age_s=max(0.0, at - self.first_seen),
            flow_count=self.count,
            bytes_mean=self.bytes_mean,
            rtt_ms=self.rtt_ms,
        )


class FlowTelemetry:
    """Thread-safe bounded flow table."""

    def __init__(self, config: TelemetryConfig | None = None, *, clock=time.monotonic) -> None:
        self.config = config or TelemetryConfig()
        self.clock = clock
        self._flows: dict[Hashable, _FlowState] = {}
        self._lock = threading.RLock()

    def update(
        self,
        key: Hashable,
        payload: bytes,
        *,
        at: float | None = None,
        rtt_ms: float | None = None,
    ) -> TelemetrySnapshot:
        now = self.clock() if at is None else at
        entropy = shannon_entropy(payload)
        with self._lock:
            state = self._flows.get(key)
            if state is None:
                state = _FlowState(first_seen=now, last_seen=now)
                self._flows[key] = state
                if len(self._flows) > self.config.max_flows:
                    self._evict_locked()
            snapshot = state.observe(now, len(payload), rtt_ms, self.config)
            # Entropy is a property of this message, not accumulated flow state.
            return replace(snapshot, payload_entropy=entropy)

    def _evict_locked(self) -> None:
        target = max(1, len(self._flows) - self.config.max_flows // 10)
        oldest = sorted(self._flows.items(), key=lambda item: item[1].last_seen)[:target]
        for key, _ in oldest:
            self._flows.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._flows)


def shannon_entropy(payload: bytes) -> float:
    """Byte-level Shannon entropy in bits/byte (0.0 to 8.0)."""

    if not payload:
        return 0.0
    counts = [0] * 256
    for byte in payload:
        counts[byte] += 1
    total = len(payload)
    entropy = 0.0
    for count in counts:
        if count:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy
