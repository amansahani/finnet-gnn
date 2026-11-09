"""Controlled reject-inference tools for censored payment feedback.

Real card systems do not observe fraud outcomes on approved traffic in the same
way as issued-authorised traffic; historically declined transactions receive no
chargeback label. This module supports a reproducible simulation in which a
legacy rule accepts each event with a known propensity, training uses only
approved outcomes with importance weights, and test evaluation keeps the full
labelled stream.

It is a research harness. It is not a substitute for logged production
propensities from the actual legacy decision system.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

from .data.common import TransactionEvent

_MODEL_CUTOFF = 155.0
_BIAS = 1.75


@dataclass(frozen=True)
class RejectionResult:
    approved_events: list[TransactionEvent]
    approved_weights: list[float]
    approved_count: int
    rejected_count: int
    report: dict[str, object]


def legacy_approval_propensity(event: TransactionEvent) -> float:
    """Simple, label-free synthetic legacy model.

    Larger amounts are rarer for approval; hour-of-day introduces deterministic
    heterogeneity. No outcome label is used.
    """

    amount_term = -min(8.0, max(0.0, event.amount) / _MODEL_CUTOFF)
    hour = (event.timestamp / 3600.0) % 24.0
    hour_term = 0.35 * math.sin(2.0 * math.pi * hour / 24.0)
    z = _BIAS + amount_term + hour_term
    return float(1.0 / (1.0 + math.exp(-z)))


def simulate_rejection(
    events: list[TransactionEvent],
    *,
    seed: int = 101,
    propensity_floor: float = 0.05,
    weight_cap: float = 20.0,
) -> RejectionResult:
    """Simulate legacy decisions and return importance-weighted approved data."""

    propensities = [legacy_approval_propensity(event) for event in events]
    by_attack: dict[str, dict[str, int]] = {}
    rng = random.Random(seed)
    approvals: list[tuple[TransactionEvent, float, float]] = []
    raw_weights = [min(weight_cap, 1.0 / max(propensity_floor, p)) for p in propensities]
    for event, propensity, raw_weight in zip(events, propensities, raw_weights, strict=True):
        approved = rng.random() < propensity
        stats = by_attack.setdefault(
            event.attack_type, {"approved": 0, "rejected": 0, "fraud": event.label}
        )
        if approved:
            approvals.append((event, propensity, raw_weight))
            stats["approved"] += 1
        else:
            stats["rejected"] += 1
    mean_weight = sum(weight for _, _, weight in approvals) / max(1, len(approvals))
    approved_events = [event for event, _, _ in approvals]
    approved_weights = [float(raw_weight / mean_weight) for _, _, raw_weight in approvals]
    report: dict[str, object] = {
        "approved": len(approved_events),
        "rejected": len(events) - len(approved_events),
        "approval_rate": len(approved_events) / max(1, len(events)),
        "mean_weight": sum(approved_weights) / max(1, len(approved_weights)),
        "by_attack_type": by_attack,
        "note": "synthetic legacy-choice model; replace propensities with logged production scores for deployment",
    }
    return RejectionResult(
        approved_events=approved_events,
        approved_weights=approved_weights,
        approved_count=len(approved_events),
        rejected_count=len(events) - len(approved_events),
        report=report,
    )


def event_digest(event: TransactionEvent) -> str:
    material = f"{event.event_id}:{event.timestamp}:{event.amount}:{event.card_token}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
