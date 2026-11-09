"""Async attack-traffic simulator for the real TCP gateway.

The simulator binds distinct loopback source addresses so the gateway observes
separate /24 subnets (127.0.0.x, 127.0.1.x, 127.0.2.x). This exercises the actual
L4 path, ISO codec, GNN scorer, token bucket and quarantine logic.
"""

from __future__ import annotations

import asyncio
import random
import struct
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from .iso8583 import ISO8583Codec, ISO8583Message


@dataclass(frozen=True)
class SimulatorConfig:
    host: str = "127.0.0.1"
    port: int = 8583
    events: int = 600
    fraud_share: float = 0.55
    legit_workers: int = 18
    carding_workers: int = 5
    botnet_workers: int = 18
    seed: int = 11
    timeout_s: float = 2.0


@dataclass
class WorkerStats:
    profile: str
    sent: int = 0
    responses: int = 0
    closed: int = 0
    errors: int = 0
    actions: Counter = field(default_factory=Counter)
    response_codes: Counter = field(default_factory=Counter)
    latencies_ms: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class SimulationReport:
    sent: int
    responses: int
    closed: int
    errors: int
    duration_s: float
    throughput_per_s: float
    action_counts: dict[str, int]
    response_counts: dict[str, int]
    profile: dict[str, dict[str, int]]
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float

    def as_dict(self) -> dict[str, object]:
        return {
            "sent": self.sent,
            "responses": self.responses,
            "closed": self.closed,
            "errors": self.errors,
            "duration_s": self.duration_s,
            "throughput_per_s": self.throughput_per_s,
            "action_counts": self.action_counts,
            "response_counts": self.response_counts,
            "profile": self.profile,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
        }


@dataclass(frozen=True)
class _Workload:
    profile: str
    source_ip: str
    subnet: str
    terminal_id: str
    merchant_id: str
    cards: tuple[str, ...]
    events: int
    interval_s: float


async def run_simulation(config: SimulatorConfig) -> SimulationReport:
    rng = random.Random(config.seed)
    workloads = _build_workloads(config)
    started = time.perf_counter()
    results = await asyncio.gather(
        *[_run_worker(workload, config, rng) for workload in workloads],
        return_exceptions=True,
    )
    duration = time.perf_counter() - started
    stats = [item for item in results if isinstance(item, WorkerStats)]
    for item in results:
        if isinstance(item, Exception):
            # Surface network setup failures in the aggregate.
            stats.append(WorkerStats(profile="error", errors=1))
    latencies = sorted(latency for item in stats for latency in item.latencies_ms)
    actions = Counter()
    responses = Counter()
    profile: dict[str, dict[str, int]] = {}
    for item in stats:
        actions.update(item.actions)
        responses.update(item.response_codes)
        bucket = profile.setdefault(
            item.profile,
            {"sent": 0, "responses": 0, "closed": 0, "errors": 0},
        )
        bucket["sent"] += item.sent
        bucket["responses"] += item.responses
        bucket["closed"] += item.closed
        bucket["errors"] += item.errors
    total_sent = sum(item.sent for item in stats)
    total_responses = sum(item.responses for item in stats)
    return SimulationReport(
        sent=total_sent,
        responses=total_responses,
        closed=sum(item.closed for item in stats),
        errors=sum(item.errors for item in stats),
        duration_s=duration,
        throughput_per_s=total_sent / duration if duration else 0.0,
        action_counts=dict(actions),
        response_counts=dict(responses),
        profile=profile,
        latency_p50_ms=_quantile(latencies, 0.50),
        latency_p95_ms=_quantile(latencies, 0.95),
        latency_p99_ms=_quantile(latencies, 0.99),
    )


def _build_workloads(config: SimulatorConfig) -> list[_Workload]:
    total = max(1, config.events)
    fraud = int(total * config.fraud_share)
    carding_events = int(fraud * 0.42)
    botnet_events = fraud - carding_events
    legit_events = total - fraud
    rng = random.Random(config.seed)
    workloads: list[_Workload] = []

    for index in range(config.legit_workers):
        count = _distribute(legit_events, config.legit_workers, index)
        if count <= 0:
            continue
        card = f"4{rng.randrange(10**14, 10**15):015d}"
        workloads.append(
            _Workload(
                profile="legitimate",
                source_ip=f"127.{10 + index}.0.{10 + index}",
                subnet=f"127.{10 + index}.0.0/24",
                terminal_id=f"T{index:07d}",
                merchant_id=f"M{1000 + index:014d}",
                cards=(card,),
                events=count,
                interval_s=0.32,
            )
        )

    for index in range(config.carding_workers):
        count = _distribute(carding_events, config.carding_workers, index)
        if count <= 0:
            continue
        cards = tuple(
            f"4{(index * 100_000 + card):015d}" for card in range(1, 800)
        )
        workloads.append(
            _Workload(
                profile="card_testing",
                source_ip=f"127.100.1.{10 + index}",
                subnet="127.100.1.0/24",
                terminal_id="T9999999",
                merchant_id="M99999999999999",
                cards=cards,
                events=count,
                interval_s=0.0015,
            )
        )

    for index in range(config.botnet_workers):
        count = _distribute(botnet_events, config.botnet_workers, index)
        if count <= 0:
            continue
        cards = tuple(
            f"4{(index * 100_000 + card):015d}" for card in range(1, 50)
        )
        workloads.append(
            _Workload(
                profile="botnet",
                source_ip=f"127.200.{index % 8}.{10 + index}",
                subnet=f"127.200.{index % 8}.0/24",
                terminal_id=f"T{800 + index:07d}",
                merchant_id=f"M{2000 + index:014d}",
                cards=cards,
                events=count,
                interval_s=0.004,
            )
        )
    return workloads


def _distribute(total: int, workers: int, index: int) -> int:
    base, remainder = divmod(max(0, total), max(1, workers))
    return base + (1 if index < remainder else 0)


async def _run_worker(
    workload: _Workload, config: SimulatorConfig, rng: random.Random
) -> WorkerStats:
    stats = WorkerStats(profile=workload.profile)
    codec = ISO8583Codec()
    try:
        # Loopback aliases are not available on macOS, so the simulator sends
        # its source network in ISO private field 61. The server must be started
        # with `trust_client_subnet` for this test-only path.
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(config.host, config.port),
            timeout=config.timeout_s,
        )
    except Exception:  # noqa: BLE001 - one unreachable worker must not abort the run
        stats.errors += 1
        return stats

    try:
        for index in range(workload.events):
            card = workload.cards[index % len(workload.cards)]
            message = _make_request(
                codec,
                workload,
                card=card,
                stan=(index + 1) % 1_000_000,
                amount=round(rng.uniform(1.0, 120.0), 2),
            )
            frame = codec.encode_frame(message)
            started = time.perf_counter()
            try:
                writer.write(frame)
                await asyncio.wait_for(writer.drain(), timeout=config.timeout_s)
                stats.sent += 1
                header = await asyncio.wait_for(
                    reader.readexactly(2), timeout=config.timeout_s
                )
                (length,) = struct.unpack(">H", header)
                body = await asyncio.wait_for(
                    reader.readexactly(length), timeout=config.timeout_s
                )
                response = codec.decode(body)
                elapsed = (time.perf_counter() - started) * 1000.0
                stats.latencies_ms.append(elapsed)
                stats.responses += 1
                stats.response_codes[response.response_code] += 1
                action = _response_action(response)
                stats.actions[action] += 1
            except (asyncio.IncompleteReadError, ConnectionError, asyncio.TimeoutError):
                stats.closed += 1
                break
            except Exception:  # noqa: BLE001 - count and stop this worker only
                stats.errors += 1
                break
            if workload.interval_s > 0:
                await asyncio.sleep(workload.interval_s * rng.uniform(0.5, 1.5))
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except ConnectionError:
            pass
    return stats


def _make_request(
    codec: ISO8583Codec,
    workload: _Workload,
    *,
    card: str,
    stan: int,
    amount: float,
) -> ISO8583Message:
    now = datetime.now()
    fields: dict[int, str] = {
        2: card,
        3: "000000",
        4: str(int(amount * 100)).zfill(12),
        7: now.strftime("%m%d%H%M%S"),
        11: str(stan).zfill(6),
        22: "051",
        41: workload.terminal_id[:8].ljust(8),
        42: workload.merchant_id[:15].ljust(15),
        49: "356",
        61: f"subnet={workload.subnet};src_ip={workload.source_ip}",
        62: f"ts_ms={time.time() * 1000:.3f}",
    }
    return ISO8583Message(mti="0200", fields=fields)


def _response_action(response: ISO8583Message) -> str:
    raw = response.fields.get(60, "")
    text = raw.decode("ascii", "ignore") if isinstance(raw, bytes) else str(raw)
    for token in text.split(";"):
        key, _, value = token.partition("=")
        if key == "action":
            return value
    if response.response_code == "00":
        return "ALLOW"
    return "UNKNOWN"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(len(values) * q) - 1))
    return values[index]
