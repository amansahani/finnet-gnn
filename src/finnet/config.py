"""Typed configuration for the gateway, model, graph and policy layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8583
    metrics_host: str = "127.0.0.1"
    metrics_port: int = 9101
    max_frame_bytes: int = 64 * 1024
    idle_timeout_s: float = 30.0
    max_connections: int = 2_000
    max_messages_per_connection: int = 100_000
    worker_threads: int = 4
    # Inline inference avoids a thread-pool queue and is the safest way to meet
    # a hard 20 ms scoring deadline for this small CPU model. Switch off when
    # using a larger model or an accelerator.
    inline_inference: bool = True
    request_timeout_s: float = 1.0
    use_uvloop: bool = True
    # Optional TLS/mTLS termination. Disabled by default so loopback demos remain
    # simple; must be enabled for production socket exposure.
    tls_enabled: bool = False
    tls_cert_file: str = ""
    tls_key_file: str = ""
    tls_ca_file: str = ""
    require_client_certificate: bool = False
    mac_required: bool = False
    mac_secret_file: str = ""
    mac_secret: str = ""
    replay_protection: bool = True
    idempotency_ttl_s: float = 300.0
    # Test-harness only: allow ISO private field 61 to claim a source subnet
    # when loopback aliases are unavailable. Never enable in production.
    trust_client_subnet: bool = False
    # Optional real upstream ISO-8583 acquirer. Empty means the gateway performs
    # local stand-in authorisation so the demo is self-contained.
    upstream_host: str = ""
    upstream_port: int = 0
    upstream_timeout_s: float = 2.0


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 24
    type_count: int = 5
    hidden_dim: int = 64
    num_layers: int = 2
    relation_count: int = 7
    dropout: float = 0.15
    risk_gate: bool = False
    temporal_encoding: bool = False
    time_dim: int = 16
    temporal_memory: bool = False
    history_length: int = 8
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 12
    seed: int = 42
    torch_threads: int = 1
    inference_budget_ms: float = 20.0
    checkpoint: str = "artifacts/risk_gnn.pt"
    cascade_checkpoint: str = ""
    device: str = "cpu"


@dataclass(frozen=True)
class GraphConfig:
    max_entities: int = 200_000
    max_tx_nodes: int = 50_000
    tx_ttl_s: float = 900.0
    snapshot_hops: int = 2
    max_snapshot_nodes: int = 128
    fanout_cap: int = 4_096
    approx_fanout: bool = False
    approx_fanout_precision: int = 8
    risk_ewma_alpha: float = 0.25


@dataclass(frozen=True)
class TelemetryConfig:
    jitter_alpha: float = 0.2
    rtt_alpha: float = 0.2
    velocity_window_s: float = 1.0
    rate_window_s: float = 5.0
    max_flows: int = 100_000


@dataclass(frozen=True)
class PolicyConfig:
    allow_threshold: float = 0.40
    step_up_threshold: float = 0.45
    review_threshold: float = 0.55
    throttle_threshold: float = 0.60
    soft_decline_threshold: float = 0.78
    quarantine_threshold: float = 0.86
    terminate_threshold: float = 0.96
    token_capacity: float = 20.0
    token_refill_per_s: float = 6.0
    throttle_penalty_ms: float = 40.0
    step_up_penalty_ms: float = 5.0
    review_penalty_ms: float = 5.0
    quarantine_ttl_s: float = 300.0
    violation_window_s: float = 60.0
    violations_to_quarantine: int = 4
    fail_open_on_timeout: bool = True
    enable_os_firewall: bool = False
    firewall_backend: str = "audit"  # audit | pf | iptables
    firewall_command_timeout_s: float = 3.0
    firewall_audit_path: str = "artifacts/firewall.jsonl"
    calibration_path: str = ""
    shadow_mode: bool = False
    shadow_action_label: str = "shadow"
    cost_false_positive: float = 1.0
    cost_false_negative: float = 10.0


@dataclass(frozen=True)
class DataConfig:
    source: str = "synthetic"
    synthetic_events: int = 12_000
    synthetic_fraud_rate: float = 0.08
    synthetic_seed: int = 7
    transactions_csv: str = "data/train_transaction.csv"
    identity_csv: str = "data/train_identity.csv"
    max_events: int = 150_000
    train_fraction: float = 0.8
    max_train_samples: int = 20_000
    max_test_samples: int = 5_000
    negative_ratio: float = 3.0
    pan_salt: str = "finnet-local-salt-change-me"
    # Simulation-only controlled benchmark for reject inference. In production,
    # propensities should come from logged legacy approval/decline probabilities.
    simulate_rejection: bool = False
    rejection_cutoff: float = 0.55
    rejection_seed: int = 101
    rejection_weight_cap: float = 20.0
    rejection_propensity_floor: float = 0.05
    attack_eval_max_events: int = 5_000


@dataclass(frozen=True)
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    log_path: str = "artifacts/decisions.jsonl"
    metrics_path: str = "artifacts/metrics.json"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "server": ServerConfig,
    "model": ModelConfig,
    "graph": GraphConfig,
    "telemetry": TelemetryConfig,
    "policy": PolicyConfig,
    "data": DataConfig,
}


def _build_section(cls: type, raw: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load defaults, then a YAML file, then dotted-key overrides.

    Example override: ``{"server": {"port": 9000}, "model": {"epochs": 3}}``.
    Unknown keys are rejected so configuration typos cannot silently weaken the
    risk policy.
    """

    merged: dict[str, Any] = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("Configuration root must be a mapping")
        merged.update(raw)
    if overrides:
        _deep_merge(merged, overrides)

    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        section_raw = merged.pop(name, {}) or {}
        sections[name] = _build_section(cls, dict(section_raw))

    top_level = {k: v for k, v in merged.items() if k in {"log_path", "metrics_path"}}
    unknown = set(merged) - set(top_level)
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {sorted(unknown)}")
    return AppConfig(**sections, **top_level)


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
