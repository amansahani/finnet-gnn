"""Canonical transaction event used by offline training and online serving."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime

from ..iso8583 import ISO8583Message, subnet_for_ip, tokenize_pan


@dataclass(frozen=True)
class TransactionEvent:
    event_id: str
    timestamp: float
    src_ip: str
    subnet: str
    device_id: str
    card_token: str
    merchant_id: str
    amount: float
    currency: str = "356"
    terminal_id: str = ""
    stan: str = ""
    label: int = 0
    attack_type: str = "normal"
    rtt_ms: float | None = None
    payload_size: int = 0
    raw_payload: bytes = field(default=b"", compare=False)

    @property
    def flow_key(self) -> tuple[str, str]:
        return (self.subnet, self.device_id)

    @property
    def hour_fraction(self) -> float:
        hour = (self.timestamp / 3600.0) % 24.0
        return hour / 24.0


def event_from_iso(
    message: ISO8583Message,
    *,
    peer_ip: str,
    pan_salt: str,
    timestamp: float,
    rtt_ms: float | None,
    sequence: int,
    payload_size: int,
) -> TransactionEvent:
    pan = message.pan
    card_token = tokenize_pan(pan, pan_salt) if pan else "card_unknown"
    device_id = message.terminal_id or f"terminal_{_short_hash(pan + message.merchant_id)}"
    merchant_id = message.merchant_id or "merchant_unknown"
    event_id = f"tx_{int(timestamp * 1000)}_{sequence}_{_short_hash(message.stan + pan + str(timestamp))}"
    return TransactionEvent(
        event_id=event_id,
        timestamp=timestamp,
        src_ip=peer_ip,
        subnet=subnet_for_ip(peer_ip),
        device_id=device_id,
        card_token=card_token,
        merchant_id=merchant_id,
        amount=float(message.amount),
        currency=message.currency,
        terminal_id=device_id,
        stan=message.stan,
        rtt_ms=rtt_ms,
        payload_size=payload_size,
        raw_payload=message.raw,
    )


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "ignore")).hexdigest()[:10]


def hour_features(timestamp: float) -> tuple[float, float]:
    """Cyclical time-of-day features robust to arbitrary dataset epochs."""

    hour = (timestamp / 3600.0) % 24.0
    angle = 2.0 * math.pi * hour / 24.0
    return math.sin(angle), math.cos(angle)


def timestamp_to_datetime(timestamp: float) -> datetime:
    try:
        return datetime.fromtimestamp(timestamp)
    except (OverflowError, OSError, ValueError):
        return datetime(2024, 1, 1)
