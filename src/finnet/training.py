"""Temporal graph construction, mini-batch training and evaluation."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import AppConfig
from .data.common import TransactionEvent
from .features import INPUT_DIM, NODE_FEATURE_NAMES, RELATION_NAMES
from .graph import DynamicTransactionGraph, GraphSnapshot
from .model import RiskGNN
from .scorer import heuristic_score
from .telemetry import FlowTelemetry


@dataclass(frozen=True)
class GraphSample:
    snapshot: GraphSnapshot
    label: int
    attack_type: str
    event_id: str


class SampleSet:
    def __init__(self, samples: list[GraphSample], name: str) -> None:
        self.samples = samples
        self.name = name

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([sample.label for sample in self.samples], dtype=np.int64)

    @property
    def attack_types(self) -> list[str]:
        return [sample.attack_type for sample in self.samples]

    @property
    def event_ids(self) -> list[str]:
        return [sample.event_id for sample in self.samples]

    def summary(self) -> dict[str, object]:
        labels = self.labels
        attacks: dict[str, int] = {}
        for attack_type in self.attack_types:
            attacks[attack_type] = attacks.get(attack_type, 0) + 1
        return {
            "name": self.name,
            "samples": len(self.samples),
            "fraud": int(labels.sum()),
            "normal": int(len(labels) - labels.sum()),
            "attack_types": attacks,
        }


class GraphDataset(Dataset[GraphSample]):
    def __init__(self, samples: SampleSet) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples.samples)

    def __getitem__(self, index: int) -> GraphSample:
        return self.samples.samples[index]


def collate_graphs(items: Sequence[GraphSample]) -> dict[str, torch.Tensor]:
    x_parts: list[np.ndarray] = []
    type_parts: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    etype_parts: list[np.ndarray] = []
    eweight_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    target_parts: list[int] = []
    labels: list[int] = []
    node_offset = 0
    for item in items:
        snapshot = item.snapshot
        x_parts.append(snapshot.x)
        type_parts.append(snapshot.node_types)
        edge_parts.append(snapshot.edge_index + node_offset)
        etype_parts.append(snapshot.edge_type)
        eweight_parts.append(snapshot.edge_weight)
        time_parts.append(
            np.full(snapshot.num_nodes, snapshot.time_phase, dtype=np.float32)
        )
        target_parts.append(snapshot.target_index + node_offset)
        labels.append(item.label)
        node_offset += snapshot.num_nodes
    return {
        "x": torch.from_numpy(np.concatenate(x_parts, axis=0)).float(),
        "node_types": torch.from_numpy(np.concatenate(type_parts, axis=0)).long(),
        "edge_index": torch.from_numpy(np.concatenate(edge_parts, axis=1)).long(),
        "edge_type": torch.from_numpy(np.concatenate(etype_parts, axis=0)).long(),
        "edge_weight": torch.from_numpy(np.concatenate(eweight_parts, axis=0)).float(),
        "time_phase": torch.from_numpy(np.concatenate(time_parts, axis=0)).float(),
        "target_index": torch.tensor(target_parts, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float32),
    }


def collect_phase(
    events: Iterable[TransactionEvent],
    *,
    graph: DynamicTransactionGraph,
    telemetry: FlowTelemetry,
    phase_name: str,
    max_samples: int,
    negative_ratio: float,
    seed: int,
    progress: Callable[[str], None] | None = None,
) -> SampleSet:
    """Replay events once, capturing a stratified temporal sample.

    Every event updates the graph and telemetry table. Only selected snapshots
    are retained. Risk feedback uses the label-free heuristic prior, so no
    future label leaks into a transaction's features.
    """

    event_list = list(events)
    fraud_count = sum(1 for event in event_list if event.label == 1)
    normal_count = len(event_list) - fraud_count
    if fraud_count == 0:
        raise ValueError(f"{phase_name}: no fraud events available for training")
    fraud_target = min(fraud_count, max_samples)
    normal_target = min(normal_count, max(1, int(fraud_target * max(0.1, negative_ratio))))
    if fraud_target + normal_target > max_samples:
        scale = max_samples / (fraud_target + normal_target)
        fraud_target = max(1, int(fraud_target * scale))
        normal_target = max(1, int(normal_target * scale))
    fraud_probability = min(1.0, fraud_target / max(1, fraud_count))
    normal_probability = min(1.0, normal_target / max(1, normal_count))
    rng = random.Random(seed)
    retained: list[GraphSample] = []

    for index, event in enumerate(event_list):
        payload = event.raw_payload or f"{event.event_id}:{event.amount}".encode()
        snapshot = telemetry.update(
            event.flow_key,
            payload,
            at=event.timestamp,
            rtt_ms=event.rtt_ms,
        )
        added = graph.add_transaction(event, snapshot, now=event.timestamp)
        graph_snapshot = graph.snapshot(added.node_id, now=event.timestamp)
        probability = fraud_probability if event.label == 1 else normal_probability
        if rng.random() <= probability:
            retained.append(
                GraphSample(
                    snapshot=graph_snapshot,
                    label=event.label,
                    attack_type=event.attack_type,
                    event_id=event.event_id,
                )
            )
        graph.update_risk(added.node_id, heuristic_score(graph_snapshot), now=event.timestamp)
        if progress and (index + 1) % 5_000 == 0:
            progress(
                f"{phase_name}: replayed {index + 1}/{len(event_list)} "
                f"events, retained {len(retained)} snapshots"
            )
    if not retained:
        raise ValueError(f"{phase_name}: no snapshots retained; increase max_samples")
    return SampleSet(retained, phase_name)


def split_events_time(
    events: Sequence[TransactionEvent], train_fraction: float
) -> tuple[list[TransactionEvent], list[TransactionEvent]]:
    ordered = sorted(events, key=lambda event: (event.timestamp, event.event_id))
    if len(ordered) < 10:
        raise ValueError("Need at least 10 events")
    split = int(len(ordered) * min(0.95, max(0.5, train_fraction)))
    # A pure timestamp split can accidentally leave the held-out tail without a
    # single fraud. Move the boundary (never backwards in time) to the nearest
    # fraud so both sides are evaluable, while preserving temporal ordering.
    if not any(event.label == 1 for event in ordered[split:]):
        fraud_indices = [index for index, event in enumerate(ordered) if event.label == 1]
        if fraud_indices:
            split = min(split, fraud_indices[-1])
    if not any(event.label == 1 for event in ordered[:split]):
        fraud_indices = [index for index, event in enumerate(ordered) if event.label == 1]
        if fraud_indices:
            split = max(split, fraud_indices[0] + 1)
    split = min(len(ordered) - 1, max(1, split))
    return ordered[:split], ordered[split:]


def make_datasets(
    events: Sequence[TransactionEvent],
    config: AppConfig,
    *,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[SampleSet, SampleSet, DynamicTransactionGraph, FlowTelemetry]:
    train_events, test_events = split_events_time(events, config.data.train_fraction)
    graph = DynamicTransactionGraph(config.graph)
    telemetry = FlowTelemetry(config.telemetry)
    train_samples = collect_phase(
        train_events,
        graph=graph,
        telemetry=telemetry,
        phase_name="train",
        max_samples=max_train_samples or config.data.max_train_samples,
        negative_ratio=config.data.negative_ratio,
        seed=config.model.seed,
        progress=progress,
    )
    test_samples = collect_phase(
        test_events,
        graph=graph,
        telemetry=telemetry,
        phase_name="test",
        max_samples=max_test_samples or config.data.max_test_samples,
        negative_ratio=config.data.negative_ratio,
        seed=config.model.seed + 1,
        progress=progress,
    )
    return train_samples, test_samples, graph, telemetry


def evaluate_model(
    model: RiskGNN,
    dataset: SampleSet,
    *,
    batch_size: int = 128,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        GraphDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(
                batch["x"],
                batch["node_types"],
                batch["edge_index"],
                batch["edge_type"],
                batch["edge_weight"],
                batch["target_index"],
                time_phase=batch["time_phase"],
            )
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(labels).astype(np.int64)


def train_from_events(
    events: Sequence[TransactionEvent],
    config: AppConfig,
    *,
    output_path: str | Path | None = None,
    source_name: str = "synthetic",
    epochs: int | None = None,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    started = time.perf_counter()
    _seed_everything(config.model.seed)
    output = Path(output_path or config.model.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)

    train_samples, test_samples, graph, _ = make_datasets(events, config, progress=progress)
    if progress:
        progress(
            f"train samples={len(train_samples)} ({train_samples.summary()['fraud']} fraud), "
            f"test samples={len(test_samples)} ({test_samples.summary()['fraud']} fraud)"
        )

    model_config = replace(config.model, epochs=epochs or config.model.epochs)
    device = torch.device(model_config.device)
    model = RiskGNN(model_config).to(device)
    positive = int(train_samples.labels.sum())
    negative = len(train_samples) - positive
    pos_weight = torch.tensor([min(20.0, negative / max(1, positive))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.learning_rate,
        weight_decay=model_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_config.epochs)
    loader = DataLoader(
        GraphDataset(train_samples),
        batch_size=model_config.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, model_config.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["x"],
                batch["node_types"],
                batch["edge_index"],
                batch["edge_type"],
                batch["edge_weight"],
                batch["target_index"],
                time_phase=batch["time_phase"],
            )
            loss = criterion(logits, batch["labels"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        scheduler.step()
        mean_loss = total_loss / max(1, batches)
        history.append({"epoch": epoch, "loss": mean_loss})
        if progress:
            progress(f"epoch {epoch:02d}/{model_config.epochs} loss={mean_loss:.4f}")

    probabilities, labels = evaluate_model(
        model, test_samples, batch_size=256, device=device
    )
    metrics = compute_metrics(
        probabilities,
        labels,
        config,
        attack_types=test_samples.attack_types,
    )
    elapsed = time.perf_counter() - started
    metrics["train_seconds"] = elapsed
    metrics["history"] = history
    metrics["train_samples"] = train_samples.summary()
    metrics["test_samples"] = test_samples.summary()
    metrics["model_parameters"] = model.parameter_count()
    metrics["source"] = source_name
    metrics["feature_names"] = list(NODE_FEATURE_NAMES)
    metrics["relation_names"] = list(RELATION_NAMES)
    metrics["graph_stats"] = graph.stats()

    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "input_dim": INPUT_DIM,
        "feature_names": list(NODE_FEATURE_NAMES),
        "relation_names": list(RELATION_NAMES),
        "metrics": _json_safe(metrics),
        "source": source_name,
        "trained_at": time.time(),
    }
    torch.save(checkpoint, output)
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")

    # Persist test scores for report plots / threshold analysis.
    np.savez_compressed(
        output.with_suffix(".test_scores.npz"),
        probabilities=probabilities,
        labels=labels,
        event_ids=np.asarray(test_samples.event_ids, dtype=object),
        attack_types=np.asarray(test_samples.attack_types, dtype=object),
    )
    if progress:
        progress(
            f"saved checkpoint={output} report={report_path} "
            f"auc_roc={metrics['auc_roc']:.4f} auc_pr={metrics['auc_pr']:.4f} "
            f"in {elapsed:.1f}s"
        )
    return _json_safe(metrics)


def compute_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    config: AppConfig,
    *,
    attack_types: Sequence[str] | None = None,
) -> dict[str, object]:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    predictions = (probabilities >= 0.5).astype(int)
    result: dict[str, object] = {
        "samples": len(labels),
        "fraud": int(labels.sum()),
        "auc_roc": _safe(roc_auc_score, labels, probabilities),
        "auc_pr": _safe(average_precision_score, labels, probabilities),
        "accuracy_at_0.5": float(accuracy_score(labels, predictions)),
        "precision_at_0.5": float(precision_score(labels, predictions, zero_division=0)),
        "recall_at_0.5": float(recall_score(labels, predictions, zero_division=0)),
        "f1_at_0.5": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_at_0.5": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "score_mean_normal": float(probabilities[labels == 0].mean())
        if np.any(labels == 0)
        else 0.0,
        "score_mean_fraud": float(probabilities[labels == 1].mean())
        if np.any(labels == 1)
        else 0.0,
        "operating_points": {},
        "per_attack_type": {},
        "recommended_threshold": _recommended_threshold(probabilities, labels),
    }
    for name, threshold in (
        ("allow", config.policy.allow_threshold),
        ("throttle", config.policy.throttle_threshold),
        ("quarantine", config.policy.quarantine_threshold),
        ("terminate", config.policy.terminate_threshold),
    ):
        predicted = (probabilities >= threshold).astype(int)
        result["operating_points"][name] = {
            "threshold": threshold,
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "flagged": int(predicted.sum()),
        }
    if attack_types is not None and len(attack_types) == len(labels):
        types = np.asarray(attack_types, dtype=object)
        per_type: dict[str, dict[str, float | int]] = {}
        for attack_type in sorted(set(attack_types)):
            mask = types == attack_type
            if not np.any(mask):
                continue
            per_type[str(attack_type)] = {
                "samples": int(mask.sum()),
                "fraud": int(labels[mask].sum()),
                "mean_score": float(probabilities[mask].mean()),
                "recall_at_quarantine": float(
                    (
                        (probabilities[mask] >= config.policy.quarantine_threshold)
                        & (labels[mask] == 1)
                    ).sum()
                    / max(1, int((labels[mask] == 1).sum()))
                ),
            }
        result["per_attack_type"] = per_type
    return result


def _recommended_threshold(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Best recall at or below a 1% false-positive rate."""

    if len(np.unique(labels)) < 2:
        return {"threshold": 0.5, "recall": 0.0, "fpr": 0.0}
    candidates = np.unique(np.round(probabilities, 4))
    best = {"threshold": 1.0, "recall": 0.0, "fpr": 0.0}
    negatives = max(1, int((labels == 0).sum()))
    positives = max(1, int((labels == 1).sum()))
    for threshold in candidates:
        predicted = probabilities >= threshold
        fpr = float((predicted & (labels == 0)).sum() / negatives)
        recall = float((predicted & (labels == 1)).sum() / positives)
        if fpr <= 0.01 and recall > best["recall"]:
            best = {"threshold": float(threshold), "recall": recall, "fpr": fpr}
    return best


def _safe(metric: Callable, labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(metric(labels, probabilities))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_events_for_training(config: AppConfig) -> tuple[list[TransactionEvent], str]:
    if config.data.source == "synthetic":
        from .data.synthetic import generate_synthetic_events

        return (
            generate_synthetic_events(
                config.data.synthetic_events,
                fraud_rate=config.data.synthetic_fraud_rate,
                seed=config.data.synthetic_seed,
            ),
            "synthetic",
        )
    if config.data.source == "ieee":
        from .data.ieee_cis import load_ieee_cis

        return (
            load_ieee_cis(
                config.data.transactions_csv,
                config.data.identity_csv,
                max_events=config.data.max_events,
                pan_salt=config.data.pan_salt,
            ),
            "ieee_cis",
        )
    raise ValueError(f"Unknown data source: {config.data.source}")
