"""Deterministic synthetic transaction stream with realistic attack topologies.

Three fraud archetypes are generated:
* card-testing bot: one terminal/subnet cycling through many cards quickly
* distributed botnet: many terminals/subnets sharing a small card pool
* account takeover: a normally stable customer suddenly uses a new device and
  a high-value merchant

The generator is intentionally honest: it is a reproducible development and
benchmark harness, not a substitute for the IEEE-CIS evaluation.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from .common import TransactionEvent

_LEGIT_SUBNETS = [f"198.18.{i // 256}.{i % 256}.0/24" for i in range(1_024)]
_ATTACK_SUBNETS = [f"203.0.{113 + i}.0/24" for i in range(24)]
_MERCHANTS = [f"M{i:014d}" for i in range(1, 161)]
_DEVICES = [f"T{i:07d}" for i in range(1, 641)]
_CARDS = [f"card_synth_{i:06d}" for i in range(1, 8_001)]


@dataclass(frozen=True)
class SyntheticProfile:
    customers: int = 900
    normal_merchants: int = 120
    attack_merchants: int = 24
    seed: int = 7


class SyntheticGenerator:
    def __init__(self, profile: SyntheticProfile | None = None) -> None:
        self.profile = profile or SyntheticProfile()
        self.rng = random.Random(self.profile.seed)
        self._identity = self._build_identities()

    def _build_identities(self) -> list[dict[str, str]]:
        identities = []
        for index in range(self.profile.customers):
            identities.append(
                {
                    "subnet": _LEGIT_SUBNETS[index % len(_LEGIT_SUBNETS)],
                    "device": _DEVICES[index % len(_DEVICES)],
                    "card": _CARDS[index % len(_CARDS)],
                    "merchant": self.rng.choice(_MERCHANTS[: self.profile.normal_merchants]),
                }
            )
        return identities

    def generate(
        self,
        count: int,
        *,
        fraud_rate: float = 0.08,
        start_time: float | None = None,
    ) -> list[TransactionEvent]:
        if start_time is None:
            start_time = time.time() - max(3_600.0, count * 0.7)
        fraud_count = max(1, int(count * fraud_rate))
        normal_count = max(1, count - fraud_count)
        events: list[TransactionEvent] = []
        normal_events = self._normal_events(normal_count, start_time)
        horizon = max(60.0, max(event.timestamp for event in normal_events) - start_time + 1.0)
        events.extend(normal_events)
        events.extend(self._fraud_events(fraud_count, start_time, horizon))
        events.sort(key=lambda event: (event.timestamp, event.event_id))
        return events

    # ------------------------------------------------------------------ normal
    def _normal_events(self, count: int, start: float) -> list[TransactionEvent]:
        """Normal traffic includes short cardholder sessions.

        Real customers often make several purchases in quick succession. Without
        these bursts a model learns the shortcut "fast == fraud", which would
        quarantine ordinary high-throughput merchants. Fraud remains separable
        through card/device fan-out, entropy and amount patterns instead.
        """

        events: list[TransactionEvent] = []
        at = start + self.rng.random()
        index = 0
        while index < count:
            identity = self.rng.choice(self._identity)
            burst_length = 1 if self.rng.random() < 0.68 else self.rng.randint(2, 7)
            if self.rng.random() < 0.025:
                # New-device false-positive pressure: ordinary customers also
                # occasionally switch devices.
                device = self.rng.choice(_DEVICES)
            else:
                device = identity["device"]
            for position in range(burst_length):
                if index >= count:
                    break
                amount = min(5_000.0, max(1.0, self.rng.lognormvariate(3.6, 1.0)))
                if position == 0:
                    at += max(0.008, self.rng.expovariate(1 / 1.8))
                else:
                    at += self.rng.uniform(0.12, 1.4)
                merchant = (
                    identity["merchant"]
                    if self.rng.random() < 0.8
                    else self.rng.choice(_MERCHANTS[: self.profile.normal_merchants])
                )
                events.append(
                    self._make_event(
                        index=f"n{index}",
                        timestamp=at,
                        subnet=identity["subnet"],
                        device=device,
                        card=identity["card"],
                        merchant=merchant,
                        amount=amount,
                        label=0,
                        attack_type="normal",
                    )
                )
                index += 1
        return events

    # ------------------------------------------------------------------- fraud
    def _fraud_events(
        self, count: int, start: float, horizon: float
    ) -> list[TransactionEvent]:
        carding_n = int(count * 0.42)
        botnet_n = int(count * 0.36)
        takeover_n = count - carding_n - botnet_n
        events: list[TransactionEvent] = []
        events.extend(self._card_testing(carding_n, start, horizon))
        events.extend(self._botnet(botnet_n, start, horizon))
        events.extend(self._takeover(takeover_n, start, horizon))
        return events

    def _card_testing(
        self, count: int, start: float, horizon: float
    ) -> list[TransactionEvent]:
        events: list[TransactionEvent] = []
        bots = 6
        waves = 4
        wave_offsets = (0.04, 0.28, 0.56, 0.9)
        per_wave = max(1, count // waves)
        produced = 0
        for wave in range(waves):
            at = [start + horizon * wave_offsets[wave] + i * 1.7 for i in range(bots)]
            for _ in range(per_wave):
                if produced >= count:
                    break
                index = produced
                bot = index % bots
                subnet = _ATTACK_SUBNETS[bot]
                device = _DEVICES[500 + bot]
                # Rotate cards aggressively: the classic carding signature.
                card = _CARDS[(index * 37 + bot * 11) % len(_CARDS)]
                amount = round(self.rng.uniform(0.5, 8.0), 2)
                at[bot] += self.rng.uniform(0.018, 0.09)
                events.append(
                    self._make_event(
                        index=f"c{index}",
                        timestamp=at[bot],
                        subnet=subnet,
                        device=device,
                        card=card,
                        merchant=_MERCHANTS[120 + (index % self.profile.attack_merchants)],
                        amount=amount,
                        label=1,
                        attack_type="card_testing",
                    )
                )
                produced += 1
        return events

    def _botnet(
        self, count: int, start: float, horizon: float
    ) -> list[TransactionEvent]:
        events: list[TransactionEvent] = []
        bots = 30
        waves = 3
        wave_offsets = (0.1, 0.45, 0.86)
        per_wave = max(1, count // waves)
        cards = _CARDS[6_500:6_800]
        subnets = _ATTACK_SUBNETS[8:] + _LEGIT_SUBNETS[:8]
        produced = 0
        for wave in range(waves):
            at = [start + horizon * wave_offsets[wave] + i * 0.31 for i in range(bots)]
            for _ in range(per_wave):
                if produced >= count:
                    break
                index = produced
                bot = index % bots
                subnet = subnets[bot % len(subnets)]
                device = _DEVICES[200 + bot]
                card = cards[(index // bots + bot) % len(cards)]
                amount = round(self.rng.uniform(5.0, 180.0), 2)
                at[bot] += self.rng.uniform(0.4, 1.5)
                events.append(
                    self._make_event(
                        index=f"b{index}",
                        timestamp=at[bot],
                        subnet=subnet,
                        device=device,
                        card=card,
                        merchant=_MERCHANTS[120 + (index % 20)],
                        amount=amount,
                        label=1,
                        attack_type="botnet",
                    )
                )
                produced += 1
        return events

    def _takeover(
        self, count: int, start: float, horizon: float
    ) -> list[TransactionEvent]:
        events: list[TransactionEvent] = []
        waves = 5
        wave_offsets = (0.18, 0.38, 0.58, 0.78, 0.94)
        per_wave = max(1, count // waves)
        produced = 0
        for wave in range(waves):
            at = start + horizon * wave_offsets[wave]
            for _ in range(per_wave):
                if produced >= count:
                    break
                index = produced
                identity = self.rng.choice(self._identity)
                at += self.rng.uniform(1.0, 5.0)
                events.append(
                    self._make_event(
                        index=f"a{index}",
                        timestamp=at,
                        subnet=self.rng.choice(_ATTACK_SUBNETS),
                        device=self.rng.choice(_DEVICES[600:]),
                        card=identity["card"],
                        merchant=self.rng.choice(_MERCHANTS[120:]),
                        amount=round(self.rng.uniform(600.0, 4_800.0), 2),
                        label=1,
                        attack_type="account_takeover",
                    )
                )
                produced += 1
        return events

    # ------------------------------------------------------------------ helper
    def _make_event(
        self,
        *,
        index: str,
        timestamp: float,
        subnet: str,
        device: str,
        card: str,
        merchant: str,
        amount: float,
        label: int,
        attack_type: str,
    ) -> TransactionEvent:
        ip = subnet.rsplit(".", 1)[0] + "." + str(10 + self.rng.randrange(1, 200))
        stan = f"{self.rng.randrange(1_000_000):06d}"
        entropy_pool = (
            f"iso:{device}:{merchant}:{amount:.2f}:{stan}:{timestamp:.3f}"
            if label == 1 and attack_type == "card_testing"
            else f"iso:{device}:{card}:{merchant}:{amount:.2f}:{stan}:{ip}:{timestamp:.3f}"
        )
        raw = entropy_pool.encode("ascii")
        return TransactionEvent(
            event_id=f"synth_{attack_type}_{index}",
            timestamp=timestamp,
            src_ip=ip,
            subnet=subnet,
            device_id=device,
            card_token=card,
            merchant_id=merchant,
            amount=amount,
            currency="356",
            terminal_id=device,
            stan=stan,
            label=label,
            attack_type=attack_type,
            payload_size=len(raw),
            raw_payload=raw,
        )


def generate_synthetic_events(
    count: int = 12_000,
    *,
    fraud_rate: float = 0.08,
    seed: int = 7,
    start_time: float | None = None,
) -> list[TransactionEvent]:
    return SyntheticGenerator(SyntheticProfile(seed=seed)).generate(
        count, fraud_rate=fraud_rate, start_time=start_time
    )


def event_to_row(event: TransactionEvent) -> dict[str, object]:
    """CSV-friendly representation for inspection and report appendices."""

    return {
        "event_id": event.event_id,
        "timestamp": event.timestamp,
        "src_ip": event.src_ip,
        "subnet": event.subnet,
        "device_id": event.device_id,
        "card_token": event.card_token,
        "merchant_id": event.merchant_id,
        "amount": event.amount,
        "currency": event.currency,
        "stan": event.stan,
        "label": event.label,
        "attack_type": event.attack_type,
        "payload_entropy_hint": _entropy_hint(event.raw_payload),
    }


def _entropy_hint(payload: bytes) -> float:
    if not payload:
        return 0.0
    counts: dict[int, int] = {}
    for byte in payload:
        counts[byte] = counts.get(byte, 0) + 1
    total = len(payload)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())
