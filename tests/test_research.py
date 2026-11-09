from __future__ import annotations

from dataclasses import replace

import numpy as np

from finnet.baselines import run_baseline_benchmark
from finnet.calibration import Calibrator, calibration_metrics
from finnet.config import load_config
from finnet.data.synthetic import generate_synthetic_events
from finnet.evaluation import evaluate_scores


def test_evaluate_scores_reports_cost_and_budget() -> None:
    labels = np.array([0, 0, 1, 1], dtype=int)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    metrics = evaluate_scores(labels, scores, false_negative_cost=100.0)
    assert metrics["auc_roc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["expected_cost_at_0.5"] == 0.0
    assert metrics["review_budget"]["top_1pct"]["recall"] > 0.0


def test_calibrator_round_trip(tmp_path) -> None:
    labels = np.array([0] * 50 + [1] * 50, dtype=int)
    scores = np.r_[np.linspace(0.05, 0.4, 50), np.linspace(0.6, 0.95, 50)]
    calibrator = Calibrator.fit(scores, labels, "isotonic")
    path = calibrator.save(tmp_path / "cal.json")
    loaded = Calibrator.load(path)
    calibrated = loaded.predict(scores)
    assert calibrated.shape == scores.shape
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))
    assert calibration_metrics(labels, calibrated)["brier_score"] >= 0.0


def test_baseline_smoke_with_risk_gate_and_time_encoding(tmp_path) -> None:
    config = load_config("configs/default.yaml")
    config = replace(
        config,
        data=replace(
            config.data,
            synthetic_events=900,
            max_train_samples=120,
            max_test_samples=80,
        ),
        model=replace(
            config.model,
            epochs=1,
            hidden_dim=16,
            risk_gate=True,
            temporal_encoding=True,
            time_dim=8,
            checkpoint=str(tmp_path / "model.pt"),
        ),
        graph=replace(config.graph, max_snapshot_nodes=32),
        log_path=str(tmp_path / "decisions.jsonl"),
    )
    events = generate_synthetic_events(900)
    report = run_baseline_benchmark(
        events,
        config,
        source_name="synthetic",
        baselines=("logistic",),
        calibration_method="platt",
        model_path=tmp_path / "missing.pt",
        train_epochs=1,
    )
    assert report["feature_count"] == 51
    assert report["baselines"]["logistic"]["available"] is True
    assert report["baselines"]["gnn"]["available"] is True
