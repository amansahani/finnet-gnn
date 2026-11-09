"""Feature schema shared by the graph, model, trainer and explainability tools."""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class NodeType(IntEnum):
    SUBNET = 0
    DEVICE = 1
    CARD = 2
    MERCHANT = 3
    TRANSACTION = 4


class Relation(IntEnum):
    SUBNET_DEVICE = 0
    DEVICE_CARD = 1
    CARD_MERCHANT = 2
    TX_SUBNET = 3
    TX_DEVICE = 4
    TX_CARD = 5
    TX_MERCHANT = 6


NODE_TYPE_NAMES = ("subnet", "device", "card", "merchant", "transaction")
RELATION_NAMES = (
    "subnet--device",
    "device--card",
    "card--merchant",
    "transaction--subnet",
    "transaction--device",
    "transaction--card",
    "transaction--merchant",
)

NODE_FEATURE_NAMES = (
    # Entity statistics (0-7)
    "degree_log",
    "tx_count_log",
    "recency_s_log",
    "avg_amount_log",
    "risk_ewma",
    "distinct_counterparts_log",
    "blocked",
    "tenure_s_log",
    # Transaction / network telemetry (8-15)
    "amount_log",
    "hour_sin",
    "hour_cos",
    "interarrival_ms_log",
    "jitter_ms_log",
    "burst_velocity_log",
    "arrival_rate_log",
    "payload_entropy",
    # Transaction context and topology (16-23)
    "rtt_ms_log",
    "new_subnet",
    "new_device",
    "new_card",
    "new_merchant",
    "subnet_card_fanout_log",
    "device_card_fanout_log",
    "card_merchant_fanout_log",
)

INPUT_DIM = len(NODE_FEATURE_NAMES)


def feature_index(name: str) -> int:
    try:
        return NODE_FEATURE_NAMES.index(name)
    except ValueError as exc:
        raise KeyError(f"Unknown feature {name!r}") from exc


def empty_features() -> np.ndarray:
    return np.zeros(INPUT_DIM, dtype=np.float32)


def log1p_safe(value: float, cap: float = 1e9) -> float:
    if not np.isfinite(value) or value <= 0:
        return 0.0
    return float(np.log1p(min(value, cap)))
