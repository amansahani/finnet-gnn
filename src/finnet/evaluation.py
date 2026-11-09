"""Evaluation metrics and classical tabular baselines for graph snapshots."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import calibration_metrics

BaselineName = Literal[
    "logistic",
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
    "catboost",
]

DEFAULT_BASELINES: tuple[BaselineName, ...] = (
    "logistic",
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
    "catboost",
)


def flatten_target_features(x: np.ndarray, target_index: int) -> np.ndarray:
    return x[target_index].astype(float)


def available_baselines() -> dict[str, bool]:
    availability = {
        "logistic": True,
        "hist_gradient_boosting": True,
    }
    for module_name in ("lightgbm", "xgboost", "catboost"):
        try:
            __import__(module_name)
            availability[module_name] = True
        except ImportError:
            availability[module_name] = False
    return availability


def fit_baseline(
    name: BaselineName,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int = 42,
):
    y_train = np.asarray(y_train, dtype=int)
    positive = int(y_train.sum())
    negative = len(y_train) - positive
    scale_pos_weight = min(50.0, negative / max(1, positive))

    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                random_state=seed,
            ),
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            random_state=seed,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=4,
            verbose=-1,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            eval_metric="aucpr",
            n_jobs=4,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"Unknown baseline: {name}")


def predict_proba(model, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    return np.asarray(probabilities[:, 1], dtype=float)


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 10.0,
    review_fractions: Iterable[float] = (0.01, 0.05, 0.10),
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= 0.5).astype(int)
    result: dict[str, object] = {
        "samples": len(labels),
        "fraud": int(labels.sum()),
        "auc_roc": _safe_score(roc_auc_score, labels, scores),
        "average_precision": _safe_score(average_precision_score, labels, scores),
        "brier_score": float(brier_score_loss(labels, np.clip(scores, 0, 1))),
        "log_loss": float(log_loss(labels, np.clip(scores, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "precision_at_0.5": float(precision_score(labels, predictions, zero_division=0)),
        "recall_at_0.5": float(recall_score(labels, predictions, zero_division=0)),
        "expected_cost_at_0.5": float(
            false_positive_cost * np.sum((predictions == 1) & (labels == 0))
            + false_negative_cost * np.sum((predictions == 0) & (labels == 1))
        ),
        "review_budget": {},
        "cost_optimal_threshold": _cost_optimal_threshold(
            labels, scores, false_positive_cost, false_negative_cost
        ),
        "calibration": calibration_metrics(labels, scores),
    }
    for fraction in review_fractions:
        key = f"top_{int(fraction * 100)}pct"
        result["review_budget"][key] = recall_at_budget(labels, scores, fraction)
    return result


def recall_at_budget(labels: np.ndarray, scores: np.ndarray, fraction: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    count = len(labels)
    if count == 0:
        return {"fraction": float(fraction), "reviewed": 0, "recall": 0.0, "precision": 0.0}
    budget = max(1, round(count * float(fraction)))
    order = np.argsort(-scores)[:budget]
    selected = labels[order]
    fraud_total = max(1, int(labels.sum()))
    return {
        "fraction": float(fraction),
        "reviewed": int(budget),
        "recall": float(selected.sum() / fraud_total),
        "precision": float(selected.mean()),
    }


def _cost_optimal_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    fp_cost: float,
    fn_cost: float,
) -> dict[str, float]:
    candidates = np.unique(np.round(scores, 4))
    best = {
        "threshold": 0.5,
        "cost": float("inf"),
        "precision": 0.0,
        "recall": 0.0,
    }
    if len(candidates) == 0:
        return best
    positives = int((labels == 1).sum())
    for threshold in candidates:
        predicted = scores >= threshold
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum(~predicted & (labels == 1)))
        cost = fp_cost * fp + fn_cost * fn
        if cost < best["cost"]:
            best = {
                "threshold": float(threshold),
                "cost": float(cost),
                "precision": float((predicted & (labels == 1)).sum() / max(1, predicted.sum())),
                "recall": float((positives - fn) / max(1, positives)),
            }
    # Clean up the legacy branch artifact: precision over selected volume.
    predicted = scores >= best["threshold"]
    selected = predicted.sum()
    best["precision"] = float(((predicted) & (labels == 1)).sum() / max(1, selected))
    return best


def _safe_score(metric, labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(metric(labels, scores))
