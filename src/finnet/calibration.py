"""Safe post-hoc calibration for fraud probabilities.

Calibrators are serialized to JSON only: no pickle or arbitrary code loading.
Supported methods are isotonic regression, Platt scaling and temperature scaling.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

CalibrationMethod = Literal["isotonic", "platt", "temperature"]


@dataclass
class Calibrator:
    method: CalibrationMethod
    x_thresholds: list[float]
    y_values: list[float]

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if self.method == "isotonic":
            return np.interp(
                scores,
                np.asarray(self.x_thresholds, dtype=float),
                np.asarray(self.y_values, dtype=float),
                left=self.y_values[0] if self.y_values else 0.0,
                right=self.y_values[-1] if self.y_values else 1.0,
            ).astype(float)
        if self.method == "platt":
            slope, intercept = self.y_values
            return 1.0 / (1.0 + np.exp(-(slope * scores + intercept)))
        if self.method == "temperature":
            (temperature,) = self.y_values
            logits = np.log(np.clip(scores, 1e-6, 1.0 - 1e-6)) - np.log(
                np.clip(1.0 - scores, 1e-6, 1.0)
            )
            return 1.0 / (1.0 + np.exp(-logits / temperature))
        raise ValueError(f"Unknown calibration method: {self.method}")

    @classmethod
    def fit(
        cls, scores: np.ndarray, labels: np.ndarray, method: CalibrationMethod = "isotonic"
    ) -> Calibrator:
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if len(np.unique(labels)) < 2:
            raise ValueError("calibration requires both classes")
        if method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(scores, labels)
            return cls(
                method="isotonic",
                x_thresholds=[float(v) for v in model.X_thresholds_],
                y_values=[float(v) for v in model.y_thresholds_],
            )
        if method == "platt":
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            model.fit(scores.reshape(-1, 1), labels)
            return cls(
                method="platt",
                x_thresholds=[],
                y_values=[float(model.coef_[0, 0]), float(model.intercept_[0])],
            )
        if method == "temperature":
            logits = np.log(np.clip(scores, 1e-6, 1.0 - 1e-6)) - np.log(
                np.clip(1.0 - scores, 1e-6, 1.0)
            )

            def loss(temperature: float) -> float:
                probabilities = 1.0 / (1.0 + np.exp(-logits / temperature))
                return float(log_loss(labels, probabilities, labels=[0, 1]))

            result = minimize_scalar(loss, bounds=(0.05, 20.0), method="bounded")
            return cls(method="temperature", x_thresholds=[0.0], y_values=[float(result.x)])
        raise ValueError(f"Unknown calibration method: {method}")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "method": self.method,
            "x_thresholds": self.x_thresholds,
            "y_values": self.y_values,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Calibrator:
        if payload.get("version") != 1:
            raise ValueError("unsupported calibrator version")
        return cls(
            method=str(payload["method"]),  # type: ignore[arg-type]
            x_thresholds=[float(v) for v in payload.get("x_thresholds", [])],
            y_values=[float(v) for v in payload.get("y_values", [])],
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> Calibrator:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("calibrator file must contain a JSON object")
        return cls.from_dict(payload)


def calibration_metrics(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    mce = 0.0
    total = max(1, len(labels))
    for lower, upper in itertools.pairwise(edges):
        # Include the upper edge in the final bin, matching sklearn conventions.
        mask = (probabilities > lower) & (probabilities <= upper)
        if not np.any(mask):
            continue
        confidence = float(probabilities[mask].mean())
        accuracy = float(labels[mask].mean())
        gap = abs(confidence - accuracy)
        ece += float(mask.sum() / total) * gap
        mce = max(mce, gap)
    return {
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "ece": float(ece),
        "mce": float(mce),
    }
