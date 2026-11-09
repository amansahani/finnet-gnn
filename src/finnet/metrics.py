"""Dependency-free Prometheus-style metrics and a tiny scrape endpoint."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

_LATENCY_BUCKETS = (1, 2, 5, 10, 15, 20, 30, 50, 100, 250, 500, 1000, 5000)


def _labels_to_text(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    rendered = ",".join(f'{key}="{_escape(value)}"' for key, value in sorted(labels.items()))
    return "{" + rendered + "}"


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


@dataclass
class _Histogram:
    buckets: tuple[float, ...] = _LATENCY_BUCKETS
    counts: list[int] = field(default_factory=lambda: [0] * (len(_LATENCY_BUCKETS) + 1))
    total: float = 0.0
    count: int = 0
    values: deque[float] = field(default_factory=lambda: deque(maxlen=20_000))

    def observe(self, value: float) -> None:
        value = max(0.0, float(value))
        self.total += value
        self.count += 1
        self.values.append(value)
        for index, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[index] += 1
                break
        else:
            self.counts[-1] += 1

    def quantile(self, q: float) -> float:
        if not self.values:
            return 0.0
        ordered = sorted(self.values)
        index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
        return ordered[index]


class Metrics:
    def __init__(self) -> None:
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], _Histogram] = {}
        self._lock = threading.Lock()
        self.started_at = time.time()

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = (name, tuple(sorted((k, str(v)) for k, v in labels.items())))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def gauge(self, name: str, value: float, **labels: str) -> None:
        key = (name, tuple(sorted((k, str(v)) for k, v in labels.items())))
        with self._lock:
            self._gauges[key] = float(value)

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = (name, tuple(sorted((k, str(v)) for k, v in labels.items())))
        with self._lock:
            histogram = self._histograms.get(key)
            if histogram is None:
                histogram = _Histogram()
                self._histograms[key] = histogram
            histogram.observe(value)

    def quantile(self, name: str, q: float) -> float:
        with self._lock:
            values = [h.quantile(q) for (metric, _), h in self._histograms.items() if metric == name]
        return max(values, default=0.0)

    def counter_value(self, name: str, **labels: str) -> float:
        key = (name, tuple(sorted((k, str(v)) for k, v in labels.items())))
        with self._lock:
            return self._counters.get(key, 0.0)

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            emitted: set[str] = set()
            for (name, labels), value in sorted(self._counters.items()):
                if name not in emitted:
                    lines.append(f"# TYPE {name} counter")
                    emitted.add(name)
                lines.append(f"{name}{_labels_to_text(dict(labels))} {value:g}")
            emitted.clear()
            for (name, labels), value in sorted(self._gauges.items()):
                if name not in emitted:
                    lines.append(f"# TYPE {name} gauge")
                    emitted.add(name)
                lines.append(f"{name}{_labels_to_text(dict(labels))} {value:g}")
            emitted.clear()
            for (name, labels), hist in sorted(self._histograms.items()):
                if name not in emitted:
                    lines.append(f"# TYPE {name} histogram")
                    emitted.add(name)
                label_dict = dict(labels)
                cumulative = 0
                for index, bound in enumerate(hist.buckets):
                    cumulative += hist.counts[index]
                    bucket_labels = {**label_dict, "le": str(bound)}
                    lines.append(
                        f"{name}_bucket{_labels_to_text(bucket_labels)} {cumulative}"
                    )
                cumulative += hist.counts[-1]
                bucket_labels = {**label_dict, "le": "+Inf"}
                lines.append(f"{name}_bucket{_labels_to_text(bucket_labels)} {cumulative}")
                lines.append(f"{name}_sum{_labels_to_text(label_dict)} {hist.total:g}")
                lines.append(f"{name}_count{_labels_to_text(label_dict)} {hist.count}")
            lines.append("# TYPE finnet_uptime_seconds gauge")
            lines.append(f"finnet_uptime_seconds {time.time() - self.started_at:g}")
        return "\n".join(lines) + "\n"

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "counters": {
                    name: {**dict(labels), "value": value}
                    for (name, labels), value in self._counters.items()
                },
                "gauges": {
                    name: {**dict(labels), "value": value}
                    for (name, labels), value in self._gauges.items()
                },
                "latency": {
                    name: {
                        "count": hist.count,
                        "p50_ms": hist.quantile(0.50),
                        "p95_ms": hist.quantile(0.95),
                        "p99_ms": hist.quantile(0.99),
                    }
                    for (name, _), hist in self._histograms.items()
                },
                "uptime_s": time.time() - self.started_at,
            }


async def start_metrics_server(
    metrics: Metrics,
    host: str,
    port: int,
    *,
    extra_status: Callable[[], dict[str, object]] | None = None,
) -> asyncio.AbstractServer:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if line in {b"\r\n", b"\n", b""}:
                    break
            path = request_line.decode("latin-1").split(" ")[1] if " " in request_line.decode("latin-1") else "/"
            if path == "/metrics":
                body = metrics.render_prometheus().encode()
                ctype = "text/plain; version=0.0.4"
                status = "200 OK"
            elif path in {"/healthz", "/health"}:
                payload = {"status": "ok", "uptime_s": time.time() - metrics.started_at}
                if extra_status:
                    payload["gateway"] = extra_status()
                body = json.dumps(payload).encode()
                ctype = "application/json"
                status = "200 OK"
            else:
                body = b'{"error":"not found"}'
                ctype = "application/json"
                status = "404 Not Found"
            writer.write(
                f"HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {len(body)}\r\n"
                "Connection: close\r\n\r\n".encode()
                + body
            )
            await writer.drain()
        except (asyncio.TimeoutError, ConnectionError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass

    return await asyncio.start_server(handle, host, port)


def merge_metric_labels(items: Iterable[tuple[str, str]]) -> dict[str, str]:
    return dict(items)
