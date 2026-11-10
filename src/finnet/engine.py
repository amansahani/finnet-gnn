"""Gateway orchestration: parse -> telemetry -> dynamic graph -> GNN -> policy."""

from __future__ import annotations

import asyncio
import itertools
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import Calibrator
from .config import AppConfig
from .data.common import TransactionEvent, event_from_iso
from .graph import DynamicTransactionGraph, GraphSnapshot
from .iso8583 import (
    ISO8583Message,
    client_network_claim,
    likely_rtt_ms,
    subnet_for_ip,
)
from .metrics import Metrics
from .policy import Action, PolicyDecision, PolicyEngine
from .risk_cascade import RiskCascade
from .scorer import RiskScorer, heuristic_score
from .security import ReplayGuard, message_fingerprint
from .telemetry import FlowTelemetry


@dataclass
class EngineResult:
    action: Action
    score: float
    reason: str
    response_code: str
    delay_ms: float
    close_socket: bool
    send_response: bool
    latency_ms: float
    event: TransactionEvent | None = None
    score_source: str = "unknown"
    inference_latency_ms: float = 0.0
    within_budget: bool = True
    raw_score: float | None = None
    calibrated_score: float | None = None
    shadow_action: str = ""
    idempotent_replay: bool = False
    features: dict[str, float] | None = None
    tx_node_id: int | None = None
    snapshot: GraphSnapshot | None = None
    error: str = ""

    def to_log(self) -> dict[str, Any]:
        event = self.event
        return {
            "ts": time.time(),
            "event_id": event.event_id if event else None,
            "subnet": event.subnet if event else None,
            "device_id": event.device_id if event else None,
            "card_token": event.card_token if event else None,
            "merchant_id": event.merchant_id if event else None,
            "amount": event.amount if event else None,
            "score": round(self.score, 6),
            "raw_score": round(self.raw_score, 6) if self.raw_score is not None else round(self.score, 6),
            "calibrated_score": round(self.calibrated_score, 6) if self.calibrated_score is not None else None,
            "shadow_action": self.shadow_action,
            "idempotent_replay": self.idempotent_replay,
            "score_source": self.score_source,
            "action": self.action.value,
            "reason": self.reason,
            "response_code": self.response_code,
            "latency_ms": round(self.latency_ms, 3),
            "inference_latency_ms": round(self.inference_latency_ms, 3),
            "within_budget": self.within_budget,
            "error": self.error,
        }


class GatewayEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.graph = DynamicTransactionGraph(config.graph)
        self.telemetry = FlowTelemetry(config.telemetry)
        self.scorer = (
            RiskCascade(config.model)
            if config.model.cascade_checkpoint
            else RiskScorer(config.model, config.model.checkpoint)
        )
        self.policy = PolicyEngine(config.policy)
        self.metrics = Metrics()
        self.calibrator: Calibrator | None = None
        self.calibration_error = ""
        calibration_path = Path(config.policy.calibration_path) if config.policy.calibration_path else None
        if calibration_path is not None:
            try:
                self.calibrator = Calibrator.load(calibration_path)
            except Exception as exc:  # noqa: BLE001 - bad calibration must not take the gateway down
                self.calibration_error = f"{type(exc).__name__}: {exc}"
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, config.server.worker_threads),
            thread_name_prefix="finnet-infer",
        )
        self.replay_guard = ReplayGuard(config.server.idempotency_ttl_s)
        self._sequence = itertools.count(1)
        self._log_lock = threading.Lock()
        self._log_path = Path(config.log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self._log_path.open("a", encoding="utf-8", buffering=1)
        self._record_count = 0
        self._closed = False
        self.started_at = time.time()

    # ------------------------------------------------------------------- public
    async def process(
        self,
        message: ISO8583Message,
        *,
        peer_ip: str,
        raw_payload: bytes,
    ) -> EngineResult:
        effective_ip, effective_subnet = self._effective_network(message, peer_ip)
        subnet = effective_subnet or subnet_for_ip(effective_ip)
        precheck = self.policy.precheck(subnet)
        if precheck is not None:
            return self._precheck_result(precheck, subnet)

        started = time.perf_counter()
        try:
            if self.config.server.inline_inference:
                result = self._process_sync(message, peer_ip, raw_payload, started)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    self._process_sync,
                    message,
                    peer_ip,
                    raw_payload,
                    started,
                )
        except Exception as exc:  # noqa: BLE001 - the gateway must remain available
            self.metrics.inc("finnet_errors_total", stage="engine")
            result = EngineResult(
                action=Action.ALLOW if self.config.policy.fail_open_on_timeout else Action.TERMINATE,
                score=0.0,
                reason=f"engine failure: {type(exc).__name__}",
                response_code="00" if self.config.policy.fail_open_on_timeout else "96",
                delay_ms=0.0,
                close_socket=not self.config.policy.fail_open_on_timeout,
                send_response=self.config.policy.fail_open_on_timeout,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        self._record(result)
        return result

    def precheck_connection(self, peer_ip: str) -> EngineResult | None:
        subnet = subnet_for_ip(peer_ip)
        decision = self.policy.precheck(subnet)
        if decision is None:
            return None
        result = self._precheck_result(decision, subnet)
        self._record(result)
        return result

    def explain(self, result: EngineResult, top_k: int = 6) -> list[dict[str, float | str]]:
        if result.snapshot is None:
            return []
        return [
            {
                "feature": item.feature,
                "value": item.value,
                "contribution": item.contribution,
            }
            for item in self.scorer.explain(result.snapshot, top_k=top_k)
        ]

    def status(self) -> dict[str, Any]:
        return {
            "uptime_s": time.time() - self.started_at,
            "model_loaded": self.scorer.model_loaded,
            "model_source": "cascade" if getattr(self.scorer, "cascade_path", "") == "cascade" else ("gnn" if getattr(self.scorer, "model_loaded", False) else "heuristic"),
            "cascade_configured": bool(self.config.model.cascade_checkpoint),
            "cascade_path": getattr(self.scorer, "cascade_path", ""),
            "model_load_error": self.scorer.load_error,
            "calibrated": self.calibrator is not None,
            "calibration_error": self.calibration_error,
            "shadow_mode": self.config.policy.shadow_mode,
            "graph": self.graph.stats(),
            "flows": len(self.telemetry),
            "policy": self.policy.status(),
            "metrics": {
                "p50_ms": self.metrics.quantile("finnet_e2e_latency_ms", 0.50),
                "p95_ms": self.metrics.quantile("finnet_e2e_latency_ms", 0.95),
                "p99_ms": self.metrics.quantile("finnet_e2e_latency_ms", 0.99),
            },
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        with self._log_lock:
            self._log_handle.close()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
        with self._log_lock:
            self._log_handle.close()

    # ------------------------------------------------------------------ internal
    def _process_sync(
        self,
        message: ISO8583Message,
        peer_ip: str,
        raw_payload: bytes,
        started: float,
    ) -> EngineResult:
        replay_key = None
        if self.config.server.replay_protection:
            replay_key = message_fingerprint(message.fields)
            cached = self.replay_guard.check(replay_key)
            if cached is not None:
                self.metrics.inc("finnet_idempotent_replays_total")
                return self._replay_result(cached, started=started)
        now = time.time()
        rtt_ms = likely_rtt_ms(message.client_timestamp_ms, now * 1000.0)
        effective_ip, _ = self._effective_network(message, peer_ip)
        event = event_from_iso(
            message,
            peer_ip=effective_ip,
            pan_salt=self.config.data.pan_salt,
            timestamp=now,
            rtt_ms=rtt_ms,
            sequence=next(self._sequence),
            payload_size=len(raw_payload),
        )
        telemetry = self.telemetry.update(
            event.flow_key,
            raw_payload,
            at=now,
            rtt_ms=rtt_ms,
        )
        added = self.graph.add_transaction(event, telemetry, now=now)
        snapshot = self.graph.snapshot(added.node_id, now=now)
        score_result = self.scorer.predict(snapshot)
        raw_score = score_result.score
        calibrated_score = raw_score
        if self.calibrator is not None:
            try:
                calibrated_score = float(self.calibrator.predict(np.asarray([raw_score]))[0])
            except Exception as exc:  # noqa: BLE001 - keep serving on calibrator fault
                calibrated_score = raw_score
                self.metrics.inc("finnet_calibration_errors_total", stage="predict")
                self.calibration_error = f"{type(exc).__name__}: {exc}"
        decision = self.policy.decide(
            subnet=event.subnet,
            score=calibrated_score,
            telemetry=telemetry,
            now=time.monotonic(),
        )
        shadow_action = ""
        if self.config.policy.shadow_mode and decision.action != Action.ALLOW:
            shadow_action = decision.action.value
            decision = PolicyDecision(
                action=Action.ALLOW,
                reason=f"shadow mode would take {shadow_action}: {decision.reason}",
                score=decision.score,
                response_code="00",
                delay_ms=0.0,
                close_socket=False,
                send_response=True,
            )
        # The label-free prior maintains graph risk features for future events.
        # The policy itself is driven only by the GNN score.
        self.graph.update_risk(added.node_id, heuristic_score(snapshot), now=now)
        if decision.quarantine_cidr:
            self.graph.mark_subnet_blocked(
                decision.quarantine_cidr,
                until=now + self.config.policy.quarantine_ttl_s,
            )
        latency = (time.perf_counter() - started) * 1000.0
        features = self.graph.feature_vector(snapshot)
        result = EngineResult(
            action=decision.action,
            score=calibrated_score,
            reason=decision.reason,
            response_code=decision.response_code,
            delay_ms=decision.delay_ms,
            close_socket=decision.close_socket,
            send_response=decision.send_response,
            latency_ms=latency,
            event=event,
            score_source=score_result.source,
            inference_latency_ms=score_result.latency_ms,
            within_budget=score_result.within_budget,
            raw_score=raw_score,
            calibrated_score=calibrated_score if self.calibrator is not None else None,
            shadow_action=shadow_action,
            features=features,
            tx_node_id=added.node_id,
            snapshot=snapshot,
            error=score_result.error,
        )
        if replay_key is not None:
            self.replay_guard.remember(replay_key, self._serialize_replay(result))
        return result

    def _effective_network(
        self, message: ISO8583Message, peer_ip: str
    ) -> tuple[str, str | None]:
        if self.config.server.trust_client_subnet:
            src_ip, subnet = client_network_claim(message)
            if src_ip:
                return src_ip, subnet
        return peer_ip, subnet_for_ip(peer_ip)

    def _serialize_replay(self, result: EngineResult) -> str:
        return json.dumps(
            {
                "action": result.action.value,
                "score": result.score,
                "reason": result.reason,
                "response_code": result.response_code,
                "delay_ms": result.delay_ms,
                "close_socket": result.close_socket,
                "send_response": result.send_response,
            },
            sort_keys=True,
        )

    def _replay_result(self, cached: str, *, started: float) -> EngineResult:
        try:
            payload = json.loads(cached)
            action = Action(payload["action"])
        except Exception:  # noqa: BLE001 - stale cache must degrade to a safe allow
            action = Action.ALLOW
            payload = {}
        return EngineResult(
            action=action,
            score=float(payload.get("score", 0.0)),
            reason=str(payload.get("reason", "idempotent replay")),
            response_code=str(payload.get("response_code", "00")),
            delay_ms=0.0,
            close_socket=bool(payload.get("close_socket", False)),
            send_response=bool(payload.get("send_response", True)),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            score_source="idempotent_replay",
            within_budget=True,
            idempotent_replay=True,
            features={},
            error="",
        )

    def _precheck_result(self, decision: PolicyDecision, subnet: str) -> EngineResult:
        return EngineResult(
            action=decision.action,
            score=decision.score,
            reason=decision.reason,
            response_code=decision.response_code,
            delay_ms=0.0,
            close_socket=True,
            send_response=False,
            latency_ms=0.0,
            score_source="quarantine",
            within_budget=True,
            features={},
            error="",
        )

    def _record(self, result: EngineResult) -> None:
        self.metrics.inc("finnet_messages_total")
        self.metrics.inc("finnet_actions_total", action=result.action.value)
        self.metrics.observe("finnet_e2e_latency_ms", result.latency_ms)
        if result.inference_latency_ms > 0:
            self.metrics.observe("finnet_inference_latency_ms", result.inference_latency_ms)
        if result.score_source.startswith("heuristic"):
            self.metrics.inc(
                "finnet_inference_fallbacks_total",
                source=result.score_source,
            )
        self.metrics.observe("finnet_risk_score", result.score * 100.0)
        self._record_count += 1
        # O(graph) telemetry must not run on the hot path of every transaction.
        if self._record_count % 128 == 0:
            stats = self.graph.stats()
            self.metrics.gauge("finnet_graph_nodes", stats["nodes"])
            self.metrics.gauge("finnet_graph_edges", stats["edges"])
            self.metrics.gauge(
                "finnet_quarantined_subnets", self.policy.status()["quarantined_subnets"]
            )
        if self.config.log_path:
            line = json.dumps(result.to_log(), sort_keys=True) + "\n"
            with self._log_lock:
                self._log_handle.write(line)
