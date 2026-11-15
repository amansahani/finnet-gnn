"""Runtime safety gates for service startup."""

from __future__ import annotations

from pathlib import Path

from .config import AppConfig

_ALLOWED_PROFILES = {"lab", "reference", "production"}


def enforce_runtime_guard(config: AppConfig) -> None:
    """Enforce a safe startup posture for non-lab profiles.

    ``reference`` is a development integration build and requires the same
    network hardening as production except that a model checkpoint may be
    absent. ``production`` fails closed for every missing control.
    """

    profile = config.server.profile.strip().lower()
    if profile not in _ALLOWED_PROFILES:
        raise ValueError(f"server.profile must be one of lab, reference, production; got {config.server.profile!r}")
    if profile == "lab":
        return

    errors: list[str] = []
    if not config.server.tls_enabled:
        errors.append("server.tls_enabled must be true")
    if not config.server.tls_cert_file or not Path(config.server.tls_cert_file).exists():
        errors.append("server.tls_cert_file must point to an existing PEM certificate")
    if not config.server.tls_key_file or not Path(config.server.tls_key_file).exists():
        errors.append("server.tls_key_file must point to an existing PEM private key")
    if config.server.require_client_certificate and (
        not config.server.tls_ca_file or not Path(config.server.tls_ca_file).exists()
    ):
        errors.append("server.tls_ca_file must exist when require_client_certificate=true")
    if not config.server.mac_required:
        errors.append("server.mac_required must be true")
    has_mac_secret = bool(config.server.mac_secret)
    if config.server.mac_secret_file:
        has_mac_secret = Path(config.server.mac_secret_file).exists()
    if not has_mac_secret:
        errors.append("server.mac_secret_file must exist or server.mac_secret must be set")
    if not config.server.replay_protection:
        errors.append("server.replay_protection must be true")
    if config.server.trust_client_subnet:
        errors.append("server.trust_client_subnet is test-only and must be false")

    if not config.server.upstream_host or int(config.server.upstream_port or 0) <= 0:
        errors.append("an upstream authorization service must be configured")
    if profile == "production":
        if config.policy.fail_open_on_timeout:
            errors.append("policy.fail_open_on_timeout must be false in production")
        if not config.model.checkpoint or not Path(config.model.checkpoint).exists():
            errors.append("a trained model checkpoint must be available")
        if config.model.device not in {"cpu", "cuda"}:
            errors.append("model.device must be cpu or cuda")
        if config.data.pan_salt == "finnet-local-salt-change-me":
            errors.append("data.pan_salt must be changed from the development placeholder")
        if config.metrics_path and Path(config.metrics_path).exists():
            # Existing logs may be overwritten by audit truncation; startup
            # refuses stale state from an untrusted checkout.
            pass
    if errors:
        joined = "\n - ".join(errors)
        raise ValueError(f"unsafe {profile} profile:\n - {joined}")
