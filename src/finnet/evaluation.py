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


def flatten_snapshot_features(snapshot) -> np.ndarray:
    """Extract target features plus structural meta-path counts.

    Structural counts let GBDT baselines use the same typed graph information
    without a GNN: 24 target features + 5 node-type counts + 7 relation counts
    + 15 undirected node-type-pair counts = 51 dimensions.
    """

    x = flatten_target_features(snapshot.x, snapshot.target_index)
    node_counts = np.bincount(snapshot.node_types, minlength=5).astype(float)
    edge_counts = np.bincount(snapshot.edge_type, minlength=7).astype(float)
    pair_index = {tuple(sorted((i, j))): index for index, (i, j) in enumerate(_NODE_TYPE_PAIRS)}
    pair_counts = np.zeros(len(_NODE_TYPE_PAIRS), dtype=float)
    for source, target in snapshot.edge_index.T:
        pair = tuple(sorted((int(snapshot.node_types[source]), int(snapshot.node_types[target]))))
        pair_counts[pair_index[pair]] += 1.0
    return np.concatenate(
        [
            x,
            np.log1p(node_counts),
            np.log1p(edge_counts),
            np.log1p(pair_counts),
            _ring_features(snapshot),
        ]
    ).astype(float)


def _ring_features(snapshot) -> np.ndarray:
    ids = snapshot.community_ids
    if ids is None:
        return np.zeros(6, dtype=float)
    entity_mask = snapshot.node_types != 4
    entity_ids = ids[entity_mask]
    counts: dict[int, int] = {}
    for value in entity_ids:
        if value > 0:
            counts[int(value)] = counts.get(int(value), 0) + 1
    if not counts:
        return np.zeros(6, dtype=float)
    total = sum(counts.values())
    distinct = len(counts)
    largest = max(counts.values())

    target_community = 0
    for source, target in snapshot.edge_index.T:
        source, target = int(source), int(target)
        if source == snapshot.target_index and entity_mask[target]:
            target_community = int(ids[target])
            break
        if target == snapshot.target_index and entity_mask[source]:
            target_community = int(ids[source])
            break
    if target_community <= 0:
        return np.asarray(
            [
                np.log1p(distinct),
                largest / max(1, total),
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
    members = np.where(ids == target_community)[0]
    target_size = len(members)
    max_risk = float(snapshot.x[members, 4].max(initial=0.0))
    risk_sum = float(snapshot.x[members, 4].sum())
    blocked = float((snapshot.x[members, 6] > 0).sum())
    return np.asarray(
        [
            np.log1p(distinct),
            largest / max(1, total),
            np.log1p(target_size),
            np.log1p(risk_sum),
            np.log1p(blocked),
            np.clip(max_risk, 0.0, 1.0),
        ],
        dtype=float,
    )


_NODE_TYPE_PAIRS = tuple(
    (first, second)
    for first in range(5)
    for second in range(first, 5)
)


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
    amounts: np.ndarray | None = None,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 10.0,
    review_fractions: Iterable[float] = (0.01, 0.05, 0.10),
    include_uncertainty: bool = False,
    bootstrap_iterations: int = 500,
    seed: int = 42,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float) if amounts is not None else np.ones(len(labels), dtype=float)
    fraud_dollar_total = float(amounts[labels == 1].sum())
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
        "dollar_recall_at_0.5": float(
            amounts[(labels == 1) & (predictions == 1)].sum() / fraud_dollar_total
            if fraud_dollar_total > 0
            else 0.0
        ),
        "fraud_amount_caught_at_0.5": float(amounts[(labels == 1) & (predictions == 1)].sum()),
        "fraud_amount_missed_at_0.5": float(amounts[(labels == 1) & (predictions == 0)].sum()),
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
    if include_uncertainty:
        result["confidence_intervals"] = bootstrap_uncertainty(
            labels,
            scores,
            iterations=bootstrap_iterations,
            seed=seed,
        )
    for fraction in review_fractions:
        key = f"top_{int(fraction * 100)}pct"
        result["review_budget"][key] = recall_at_budget(
            labels, scores, fraction, amounts=amounts
        )
    return result


def recall_at_budget(
    labels: np.ndarray,
    scores: np.ndarray,
    fraction: float,
    *,
    amounts: np.ndarray | None = None,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float) if amounts is not None else np.ones(len(labels), dtype=float)
    count = len(labels)
    if count == 0:
        return {
            "fraction": float(fraction),
            "reviewed": 0,
            "recall": 0.0,
            "dollar_recall": 0.0,
            "precision": 0.0,
        }
    budget = max(1, round(count * float(fraction)))
    order = np.argsort(-scores)[:budget]
    selected = labels[order]
    selected_amounts = amounts[order]
    fraud_total = max(1, int(labels.sum()))
    fraud_amount = float(amounts[labels == 1].sum())
    return {
        "fraction": float(fraction),
        "reviewed": int(budget),
        "recall": float(selected.sum() / fraud_total),
        "dollar_recall": float(
            selected_amounts[selected == 1].sum() / fraud_amount if fraud_amount > 0 else 0.0
        ),
        "precision": float(selected.mean()),
    }


def bootstrap_uncertainty(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    iterations: int = 500,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap intervals for AP and AUC."""

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(labels) < 20 or len(np.unique(labels)) < 2:
        return {
            "average_precision": {"low": 0.0, "high": 0.0},
            "auc_roc": {"low": 0.0, "high": 0.0},
        }
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"average_precision": [], "auc_roc": []}
    for _ in range(max(1, int(iterations))):
        indices = rng.integers(0, len(labels), len(labels))
        sample_labels = labels[indices]
        if len(np.unique(sample_labels)) < 2:
            continue
        sample_scores = scores[indices]
        values["average_precision"].append(
            average_precision_score(sample_labels, sample_scores)
        )
        values["auc_roc"].append(roc_auc_score(sample_labels, sample_scores))

    def interval(samples: list[float]) -> dict[str, float]:
        if not samples:
            return {"low": 0.0, "high": 0.0}
        return {
            "low": float(np.quantile(samples, 0.025)),
            "high": float(np.quantile(samples, 0.975)),
        }

    return {name: interval(samples) for name, samples in values.items()}


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
