"""Risk scoring service: trained GNN with a deterministic fallback and budget guard."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .features import NODE_FEATURE_NAMES, feature_index
from .graph import GraphSnapshot
from .model import RiskGNN, load_model_checkpoint


@dataclass(frozen=True)
class ScoreResult:
    score: float
    source: str
    latency_ms: float
    within_budget: bool
    error: str = ""

    @property
    def is_model(self) -> bool:
        return self.source.startswith("gnn")


@dataclass(frozen=True)
class ExplanationItem:
    feature: str
    value: float
    contribution: float


class RiskScorer:
    def __init__(self, config: ModelConfig | None = None, checkpoint: str | None = None) -> None:
        self.config = config or ModelConfig()
        self.device = torch.device(self.config.device)
        self.model: RiskGNN | None = None
        self.checkpoint_meta: dict = {}
        self._load_error = ""
        torch.set_num_threads(max(1, self.config.torch_threads))
        path = checkpoint or self.config.checkpoint
        if Path(path).exists():
            try:
                self.model, self.checkpoint_meta = load_model_checkpoint(path, self.config.device)
                expected = int(self.checkpoint_meta.get("input_dim", -1))
                if expected not in {-1, self.config.input_dim}:
                    raise ValueError(
                        f"checkpoint feature dim {expected} != runtime {self.config.input_dim}"
                    )
            except Exception as exc:  # noqa: BLE001 - a bad checkpoint must not take the gateway down
                self.model = None
                self._load_error = f"{type(exc).__name__}: {exc}"
        else:
            self._load_error = f"checkpoint not found: {path}"
        if self.model is not None:
            self._warmup()

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    @property
    def load_error(self) -> str:
        return self._load_error

    def predict(self, snapshot: GraphSnapshot) -> ScoreResult:
        started = time.perf_counter()
        if self.model is None:
            score = self._heuristic(snapshot)
            elapsed = (time.perf_counter() - started) * 1000.0
            return ScoreResult(
                score=score,
                source="heuristic",
                latency_ms=elapsed,
                within_budget=elapsed <= self.config.inference_budget_ms,
            )

        try:
            score = self._gnn_score(snapshot)
            elapsed = (time.perf_counter() - started) * 1000.0
            if elapsed > self.config.inference_budget_ms:
                fallback = self._heuristic(snapshot)
                return ScoreResult(
                    score=fallback,
                    source="heuristic_timeout",
                    latency_ms=elapsed,
                    within_budget=False,
                    error=f"model latency {elapsed:.2f} ms exceeded budget",
                )
            return ScoreResult(
                score=score,
                source="gnn",
                latency_ms=elapsed,
                within_budget=True,
            )
        except Exception as exc:  # noqa: BLE001 - fall back rather than fail the request
            score = self._heuristic(snapshot)
            elapsed = (time.perf_counter() - started) * 1000.0
            return ScoreResult(
                score=score,
                source="heuristic_error",
                latency_ms=elapsed,
                within_budget=elapsed <= self.config.inference_budget_ms,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _gnn_score(self, snapshot: GraphSnapshot) -> float:
        assert self.model is not None
        x, types, edge_index, edge_type, edge_weight, target, time_phase, history, history_mask = _snapshot_tensors(
            snapshot, self.device
        )
        with torch.no_grad():
            logits = self.model(
                x,
                types,
                edge_index,
                edge_type,
                edge_weight,
                target,
                time_phase=time_phase,
                history=history,
                history_mask=history_mask,
            )
            return float(torch.sigmoid(logits).reshape(-1)[0].item())

    def _heuristic(self, snapshot: GraphSnapshot) -> float:
        return heuristic_score(snapshot)

    def explain(self, snapshot: GraphSnapshot, top_k: int = 6) -> list[ExplanationItem]:
        if self.model is None:
            return []
        x, types, edge_index, edge_type, edge_weight, target, time_phase, history, history_mask = _snapshot_tensors(
            snapshot, self.device
        )
        x = x.detach().clone().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logit = self.model(
            x,
            types,
            edge_index,
            edge_type,
            edge_weight,
            target,
            time_phase=time_phase,
            history=history,
            history_mask=history_mask,
        ).reshape(-1)[0]
        logit.backward()
        grads = x.grad[target]
        values = x.detach()[target]
        contributions = (grads * values).detach().cpu().numpy()
        order = np.argsort(np.abs(contributions))[::-1][:top_k]
        return [
            ExplanationItem(
                feature=NODE_FEATURE_NAMES[int(index)],
                value=float(values[int(index)].item()),
                contribution=float(contributions[int(index)]),
            )
            for index in order
        ]

    def _warmup(self) -> None:
        dummy = GraphSnapshot(
            x=np.zeros((5, self.config.input_dim), dtype=np.float32),
            node_types=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
            edge_index=np.asarray([[0, 1, 2, 3], [4, 4, 4, 4]], dtype=np.int64),
            edge_type=np.asarray([3, 4, 5, 6], dtype=np.int64),
            edge_weight=np.ones(4, dtype=np.float32),
            target_index=4,
            node_ids=np.arange(5, dtype=np.int64),
            timestamp=0.0,
        )
        for _ in range(3):
            self._gnn_score(dummy)


def heuristic_score(snapshot: GraphSnapshot) -> float:
    """Topology/telemetry prior used as the cold-start scorer and risk feedback.

    The GNN remains the decision model. This prior also maintains the online
    ``risk_ewma`` node feature during offline graph construction, keeping the
    feature distribution consistent between training and serving without using
    future labels.
    """

    x = snapshot.x[snapshot.target_index]
    burst = x[feature_index("burst_velocity_log")]
    arrival = x[feature_index("arrival_rate_log")]
    new_device = x[feature_index("new_device")]
    new_card = x[feature_index("new_card")]
    subnet_fanout = x[feature_index("subnet_card_fanout_log")]
    device_fanout = x[feature_index("device_card_fanout_log")]
    amount = x[feature_index("amount_log")]
    entropy = x[feature_index("payload_entropy")]
    value = -2.15
    value += 0.72 * min(burst / 3.0, 1.4)
    value += 0.23 * min(arrival / 2.0, 1.3)
    value += 0.52 * new_device
    value += 0.28 * new_card
    value += 0.42 * min(subnet_fanout / 4.0, 1.4)
    value += 0.50 * min(device_fanout / 3.0, 1.5)
    value += 0.18 * min(amount / 7.0, 1.0)
    value -= 0.45 * max(0.0, 0.72 - entropy)
    return float(1.0 / (1.0 + math.exp(-value)))


def _snapshot_tensors(
    snapshot: GraphSnapshot, device: torch.device
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    time_phase = np.full(snapshot.num_nodes, snapshot.time_phase, dtype=np.float32)
    if snapshot.history is None:
        from .history import HISTORY_FEATURE_DIM, HISTORY_LENGTH

        history = np.zeros((HISTORY_LENGTH, HISTORY_FEATURE_DIM), dtype=np.float32)
        history_mask = np.zeros(HISTORY_LENGTH, dtype=np.float32)
    else:
        history = snapshot.history.astype(np.float32, copy=False)
        history_mask = (
            snapshot.history_mask.astype(np.float32, copy=False)
            if snapshot.history_mask is not None
            else np.ones(history.shape[0], dtype=np.float32)
        )
    return (
        torch.from_numpy(snapshot.x).to(device),
        torch.from_numpy(snapshot.node_types).to(device),
        torch.from_numpy(snapshot.edge_index).to(device),
        torch.from_numpy(snapshot.edge_type).to(device),
        torch.from_numpy(snapshot.edge_weight).to(device),
        torch.tensor([snapshot.target_index], dtype=torch.long, device=device),
        torch.from_numpy(time_phase).to(device),
        torch.from_numpy(history[np.newaxis, :, :]).to(device),
        torch.from_numpy(history_mask[np.newaxis, :]).to(device),
    )
