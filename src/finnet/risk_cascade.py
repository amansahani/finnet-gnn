"""Two-tier online risk cascade.

A cheap model answers clear cases immediately. An optional, larger model is
consulted only when the primary score is ambiguous enough to change an
enforcement decision. Both tiers remain local, deterministic PyTorch modules.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import ModelConfig
from .graph import GraphSnapshot
from .scorer import ExplanationItem, RiskScorer, ScoreResult


class RiskCascade:
    """Duck-typed replacement for ``RiskScorer`` when ``cascade_checkpoint`` is set."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.primary = RiskScorer(config, config.checkpoint)
        cascade_config = (
            replace(config, checkpoint=config.cascade_checkpoint)
            if config.cascade_checkpoint
            else None
        )
        self.secondary: RiskScorer | None = (
            RiskScorer(cascade_config, config.cascade_checkpoint)
            if cascade_config is not None and Path(config.cascade_checkpoint).exists()
            else None
        )
        self._cascade_path = "primary"
        self._load_error = self.primary.load_error
        if config.cascade_checkpoint and self.secondary is None:
            self._load_error = (
                f"{self._load_error}; cascade checkpoint unavailable: "
                f"{config.cascade_checkpoint}"
            )

    @property
    def model_loaded(self) -> bool:
        return self.primary.model_loaded or bool(self.secondary and self.secondary.model_loaded)

    @property
    def load_error(self) -> str:
        return self._load_error

    @property
    def model(self):
        return self.primary.model

    @property
    def checkpoint_meta(self):
        return self.primary.checkpoint_meta

    @property
    def cascade_path(self) -> str:
        return self._cascade_path

    def predict(self, snapshot: GraphSnapshot) -> ScoreResult:
        primary = self.primary.predict(snapshot)
        if self.secondary is None:
            self._cascade_path = "primary"
            return primary
        uncertainty = abs(primary.score - 0.5)
        if uncertainty > 0.20:
            self._cascade_path = "primary"
            return primary

        secondary = self.secondary.predict(snapshot)
        if secondary.source.startswith("heuristic"):
            self._cascade_path = "primary_fallback"
            return primary
        secondary_weight = 0.60
        primary_weight = 1.0 - secondary_weight
        fused = secondary_weight * secondary.score + primary_weight * primary.score
        latency = primary.latency_ms + secondary.latency_ms
        within_budget = (
            primary.within_budget
            and secondary.within_budget
            and latency <= self.config.inference_budget_ms
        )
        self._cascade_path = "cascade"
        if not within_budget:
            self._cascade_path = "cascade_timeout"
        return ScoreResult(
            score=float(min(1.0, max(0.0, fused))),
            source=self._cascade_path,
            latency_ms=latency,
            within_budget=within_budget,
            error=secondary.error,
        )

    def explain(self, snapshot: GraphSnapshot, top_k: int = 6) -> list[ExplanationItem]:
        # Explain the primary decision channel; the fused path is otherwise
        # ambiguous and could hide which secondary features changed it.
        return self.primary.explain(snapshot, top_k=top_k)
