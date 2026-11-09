from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from finnet.config import load_config
from finnet.data.synthetic import generate_synthetic_events
from finnet.training import train_from_events


def test_training_smoke(tmp_path: Path) -> None:
    config = load_config("configs/default.yaml")
    config = replace(
        config,
        data=replace(
            config.data,
            synthetic_events=900,
            max_train_samples=220,
            max_test_samples=110,
            train_fraction=0.75,
        ),
        model=replace(
            config.model,
            epochs=2,
            hidden_dim=16,
            batch_size=32,
            checkpoint=str(tmp_path / "model.pt"),
        ),
        graph=replace(config.graph, max_snapshot_nodes=48),
        log_path=str(tmp_path / "decisions.jsonl"),
        metrics_path=str(tmp_path / "metrics.json"),
    )
    events = generate_synthetic_events(900)
    report = train_from_events(
        events,
        config,
        output_path=tmp_path / "model.pt",
        epochs=2,
        progress=None,
    )
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "model.report.json").exists()
    assert 0.0 <= report["auc_roc"] <= 1.0
    assert report["train_samples"]["samples"] > 0
    assert report["graph_stats"]["nodes"] > 0
