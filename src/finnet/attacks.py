"""Protocol-sensitive adversarial evaluation for graph fraud models."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .config import AppConfig
from .data.common import TransactionEvent
from .evaluation import evaluate_scores
from .graph import DynamicTransactionGraph
from .model import load_model_checkpoint
from .telemetry import FlowTelemetry
from .training import (
    collect_phase,
    evaluate_model,
    split_events_time,
)

AttackTransform = Callable[[list[TransactionEvent]], list[TransactionEvent]]


@dataclass(frozen=True)
class AttackResult:
    name: str
    test_events: int
    metrics: dict[str, object]


def available_attacks() -> dict[str, str]:
    return {
        "amount_noise": "jitter fraud amounts without changing total volume",
        "merchant_rotation": "move fraud to ordinary-looking merchants",
        "device_rotation": "rotate fraud devices to unrelated terminals",
        "subnet_hopping": "move fraud into many source subnets",
        "amount_split": "split high-value fraud into two lower-value attempts",
        "time_scatter": "jitter fraud timestamps to weaken velocity signals",
    }


def apply_attack(events: list[TransactionEvent], attack: str, *, seed: int = 97) -> list[TransactionEvent]:
    try:
        transform = _ATTACKS[attack]
    except KeyError as exc:
        raise ValueError(f"unknown attack: {attack}") from exc
    return transform(list(events), random.Random(seed))



def _sample_amounts(samples):
    from .features import feature_index

    amount_index = feature_index("amount_log")
    return np.asarray(
        [
            max(0.0, math.expm1(float(sample.snapshot.x[sample.snapshot.target_index, amount_index])))
            for sample in samples
        ],
        dtype=float,
    )

def evaluate_adversarial(
    events: list[TransactionEvent],
    config: AppConfig,
    *,
    model_path: str,
    transformations: Iterable[str] | None = None,
    max_events: int | None = None,
) -> dict[str, object]:
    ordered = sorted(events, key=lambda event: (event.timestamp, event.event_id))
    if max_events is not None and len(ordered) > max_events:
        ordered = ordered[-max_events:]
    train_events, test_events = split_events_time(ordered, config.data.train_fraction)
    attacks = list(transformations or available_attacks())
    model, _ = load_model_checkpoint(model_path, config.model.device)

    report: dict[str, object] = {
        "model": model_path,
        "events": len(ordered),
        "train_events": len(train_events),
        "test_events": len(test_events),
        "attacks": {},
    }
    baseline_test, _, _, _ = _prepare_phase(test_events, train_events, config)
    baseline_amounts = _sample_amounts(baseline_test.samples)
    baseline_probabilities, baseline_labels = evaluate_model(
        model, baseline_test, batch_size=256, device=config.model.device
    )
    report["baseline"] = evaluate_scores(
        baseline_labels,
        baseline_probabilities,
        amounts=baseline_amounts,
        false_positive_cost=config.policy.cost_false_positive,
        false_negative_cost=config.policy.cost_false_negative,
    )
    base_ap = report["baseline"]["average_precision"]
    for attack in attacks:
        mutated = apply_attack(test_events, attack, seed=config.data.rejection_seed)
        phase, _, _, _ = _prepare_phase(mutated, train_events, config)
        attack_amounts = _sample_amounts(phase.samples)
        probabilities, labels = evaluate_model(
            model, phase, batch_size=256, device=config.model.device
        )
        metrics = evaluate_scores(
            labels,
            probabilities,
            amounts=attack_amounts,
            false_positive_cost=config.policy.cost_false_positive,
            false_negative_cost=config.policy.cost_false_negative,
        )
        metrics["ap_degradation"] = float(base_ap) - float(metrics["average_precision"])
        report["attacks"][attack] = metrics
    return report


def _prepare_phase(
    test_events: list[TransactionEvent],
    train_events: list[TransactionEvent],
    config: AppConfig,
):
    graph = DynamicTransactionGraph(config.graph)
    telemetry = FlowTelemetry(config.telemetry)
    # Replay training transactions to warm graph state without retaining output;
    # this models a model that has already observed the same prior stream.
    collect_phase(
        train_events,
        graph=graph,
        telemetry=telemetry,
        phase_name="attack_train",
        max_samples=1,
        negative_ratio=0.1,
        seed=config.model.seed,
        progress=None,
    )
    phase = collect_phase(
        test_events,
        graph=graph,
        telemetry=telemetry,
        phase_name="attack_test",
        max_samples=config.data.max_test_samples,
        negative_ratio=config.data.negative_ratio,
        seed=config.model.seed + 2,
        progress=None,
    )
    return phase, graph, telemetry, None


def _replace(event: TransactionEvent, **changes) -> TransactionEvent:
    from dataclasses import replace

    return replace(event, **changes)


def _amount_noise(events: list[TransactionEvent], rng: random.Random) -> list[TransactionEvent]:
    result = []
    for event in events:
        if event.label:
            amount = max(0.5, event.amount * rng.uniform(0.7, 1.3))
            result.append(_replace(event, amount=amount))
        else:
            result.append(event)
    return result


def _merchant_rotation(events: list[TransactionEvent], rng: random.Random) -> list[TransactionEvent]:
    result = []
    for event in events:
        if event.label:
            merchant = f"M{rng.randrange(1, 120):014d}"
            result.append(_replace(event, merchant_id=merchant))
        else:
            result.append(event)
    return result


def _device_rotation(events: list[TransactionEvent], rng: random.Random) -> list[TransactionEvent]:
    result = []
    for event in events:
        if event.label:
            device = f"X{rng.randrange(1, 600):07d}"[:8]
            result.append(_replace(event, device_id=device, terminal_id=device))
        else:
            result.append(event)
    return result


def _subnet_hopping(events: list[TransactionEvent], rng: random.Random) -> list[TransactionEvent]:
    result = []
    for event in events:
        if event.label:
            octet = rng.randrange(1, 255)
            subnet = f"192.0.2.{octet}.0/24"
            result.append(_replace(event, subnet=subnet, src_ip=f"192.0.2.{octet}.10"))
        else:
            result.append(event)
    return result


def _amount_split(events: list[TransactionEvent], rng: random.Random) -> list[TransactionEvent]:
    result: list[TransactionEvent] = []
    for index, event in enumerate(events):
        if event.label and event.amount >= 25.0:
            half = event.amount / 2.0
            for part in (0, 1):
                result.append(
                    _replace(
                        event,
                        event_id=f"{event.event_id}_split{part}",
                        amount=half * rng.uniform(0.95, 1.05),
                        timestamp=event.timestamp + part * 0.5,
                    )
                )
        else:
            result.append(event)
    return result


def _time_scatter(events: list[TransactionEvent], rng: random.Random) -> list[TransactionEvent]:
    result = []
    for event in events:
        if event.label:
            result.append(_replace(event, timestamp=event.timestamp + rng.uniform(-120.0, 120.0)))
        else:
            result.append(event)
    return sorted(result, key=lambda event: (event.timestamp, event.event_id))


_ATTACKS: dict[str, AttackTransform] = {
    "amount_noise": lambda events, rng: _amount_noise(events, rng),
    "merchant_rotation": lambda events, rng: _merchant_rotation(events, rng),
    "device_rotation": lambda events, rng: _device_rotation(events, rng),
    "subnet_hopping": lambda events, rng: _subnet_hopping(events, rng),
    "amount_split": lambda events, rng: _amount_split(events, rng),
    "time_scatter": lambda events, rng: _time_scatter(events, rng),
}
