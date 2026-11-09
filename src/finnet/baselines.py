"""Reproducible classical-vs-GNN benchmark on shared temporal snapshots."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np

from .calibration import Calibrator
from .config import AppConfig
from .data.common import TransactionEvent
from .evaluation import (
    BaselineName,
    available_baselines,
    evaluate_scores,
    fit_baseline,
    flatten_target_features,
    predict_proba,
)
from .model import load_model_checkpoint
from .training import evaluate_model, make_datasets, train_from_events

CalibrationMethod = Literal["none", "isotonic", "platt", "temperature"]


def _stack_features(samples) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack(
        [flatten_target_features(sample.snapshot.x, sample.snapshot.target_index) for sample in samples],
        axis=0,
    )
    y = np.asarray([sample.label for sample in samples], dtype=int)
    return x, y


def _score_sets(x_train: np.ndarray, x_test: np.ndarray, model) -> tuple[np.ndarray, np.ndarray]:
    return predict_proba(model, x_train), predict_proba(model, x_test)


def run_baseline_benchmark(
    events: list[TransactionEvent],
    config: AppConfig,
    *,
    source_name: str,
    baselines: Iterable[BaselineName] | None = None,
    calibration_method: CalibrationMethod = "isotonic",
    model_path: str | Path | None = None,
    train_epochs: int | None = None,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 10.0,
    seed: int = 42,
) -> dict[str, object]:
    started = time.perf_counter()
    train_samples, test_samples, _, _ = make_datasets(events, config)
    if len(train_samples) == 0 or len(test_samples) == 0:
        raise ValueError("benchmark requires non-empty train/test snapshot sets")
    x_train, y_train = _stack_features(train_samples.samples)
    x_test, y_test = _stack_features(test_samples.samples)
    requested = tuple(baselines or available_baselines().keys())
    availability = available_baselines()
    results: dict[str, object] = {}

    for name in requested:
        if not availability.get(str(name), False):
            results[str(name)] = {"available": False, "skipped": True}
            continue
        baseline = fit_baseline(name, x_train, y_train, seed=seed)  # type: ignore[arg-type]
        baseline.fit(x_train, y_train)
        train_scores, test_scores = _score_sets(x_train, x_test, baseline)
        raw = evaluate_scores(
            y_test,
            test_scores,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        entry: dict[str, object] = {"available": True, "raw": raw}
        if calibration_method != "none":
            calibrator = Calibrator.fit(train_scores, y_train, calibration_method)
            calibrated_train = calibrator.predict(train_scores)
            calibrated_test = calibrator.predict(test_scores)
            entry["calibrator"] = {
                "method": calibration_method,
                "ece_train_before": raw["calibration"]["ece"],  # type: ignore[index]
                "ece_train_after": float(
                    evaluate_scores(y_train, calibrated_train)["calibration"]["ece"]  # type: ignore[index]
                ),
            }
            entry["calibrated"] = evaluate_scores(
                y_test,
                calibrated_test,
                false_positive_cost=false_positive_cost,
                false_negative_cost=false_negative_cost,
            )
        results[str(name)] = entry

    checkpoint = Path(model_path or config.model.checkpoint)
    if not checkpoint.exists() and train_epochs is not None:
        training_report = train_from_events(
            events,
            config,
            output_path=checkpoint,
            source_name=source_name,
            epochs=train_epochs,
            progress=None,
        )
        results["gnn_training"] = {
            "checkpoint": str(checkpoint),
            "metrics_from_round": training_report,
        }
    if checkpoint.exists():
        model, _ = load_model_checkpoint(str(checkpoint), config.model.device)
        gnn_probabilities, gnn_labels = evaluate_model(
            model,
            test_samples,
            batch_size=256,
            device=config.model.device,
        )
        gnn_raw = evaluate_scores(
            gnn_labels,
            gnn_probabilities,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        results["gnn"] = {"available": True, "checkpoint": str(checkpoint), "raw": gnn_raw}
        if calibration_method != "none":
            # Calibration for the GNN uses the concatenated train split scores.
            from torch.utils.data import DataLoader

            from .training import GraphDataset, collate_graphs
            loader = DataLoader(
                GraphDataset(train_samples), batch_size=256, shuffle=False, collate_fn=collate_graphs
            )
            import torch
            train_scores: list[np.ndarray] = []
            train_labels: list[np.ndarray] = []
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    batch = {k: v.to(torch.device(config.model.device)) for k, v in batch.items()}
                    logits = model(
                        batch["x"],
                        batch["node_types"],
                        batch["edge_index"],
                        batch["edge_type"],
                        batch["edge_weight"],
                        batch["target_index"],
                        time_phase=batch.get("time_phase"),
                    )
                    train_scores.append(torch.sigmoid(logits).cpu().numpy())
                    train_labels.append(batch["labels"].cpu().numpy().astype(np.int64))
            concatenated_scores = np.concatenate(train_scores)
            concatenated_labels = np.concatenate(train_labels)
            calibrator = Calibrator.fit(concatenated_scores, concatenated_labels, calibration_method)
            calibrated_gnn = calibrator.predict(gnn_probabilities)
            results["gnn"]["calibrated"] = evaluate_scores(
                gnn_labels,
                calibrated_gnn,
                false_positive_cost=false_positive_cost,
                false_negative_cost=false_negative_cost,
            )
    else:
        results["gnn"] = {
            "available": False,
            "reason": f"checkpoint not found: {checkpoint}; pass --train-epochs",
        }

    return {
        "source": source_name,
        "train_samples": train_samples.summary(),
        "test_samples": test_samples.summary(),
        "baselines": results,
        "availability": availability,
        "calibration_method": calibration_method,
        "cost_model": {
            "false_positive": false_positive_cost,
            "false_negative": false_negative_cost,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "feature_count": int(x_train.shape[1]),
    }
