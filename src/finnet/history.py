"""Shared schema for the optional raw-transaction history GRU."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .data.common import hour_features
from .telemetry import shannon_entropy

HISTORY_FEATURE_NAMES = (
    "amount_log",
    "hour_sin",
    "hour_cos",
    "interarrival_ms_log",
    "payload_entropy",
    "history_position",
)
HISTORY_FEATURE_DIM = len(HISTORY_FEATURE_NAMES)
HISTORY_LENGTH = 8


@dataclass(frozen=True)
class HistoryEntry:
    event_id: str
    timestamp: float
    features: tuple[float, ...]


def history_feature_tuple(
    *,
    amount: float,
    timestamp: float,
    previous_timestamp: float | None,
    payload: bytes,
    position: int,
) -> tuple[float, ...]:
    hour_sin, hour_cos = hour_features(timestamp)
    interarrival_ms = (
        max(0.0, (timestamp - previous_timestamp) * 1000.0)
        if previous_timestamp is not None
        else 0.0
    )
    entropy = shannon_entropy(payload or b"x") / 8.0
    return (
        math.log1p(max(0.0, amount)),
        hour_sin,
        hour_cos,
        math.log1p(interarrival_ms),
        float(min(1.0, max(0.0, entropy))),
        float(position),
    )


def normalized_histories(
    entries: Iterable[HistoryEntry], length: int = HISTORY_LENGTH
) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted(entries, key=lambda entry: (entry.timestamp, entry.event_id))[-length:]
    x = np.zeros((length, HISTORY_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(length, dtype=np.float32)
    offset = length - len(selected)
    for index, entry in enumerate(selected):
        x[offset + index] = np.asarray(entry.features, dtype=np.float32)
        mask[offset + index] = 1.0
    return x, mask
