from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from finnet.config import load_config
from finnet.deployment import enforce_runtime_guard


def test_lab_profile_accepts_defaults() -> None:
    config = load_config("configs/default.yaml")
    enforce_runtime_guard(config)


def test_reference_profile_requires_tls_and_upstream() -> None:
    config = load_config("configs/default.yaml")
    config = replace(
        config,
        server=replace(config.server, profile="reference", trust_client_subnet=False),
    )
    with pytest.raises(ValueError, match="unsafe reference profile") as error:
        enforce_runtime_guard(config)
    message = str(error.value)
    assert "server.tls_enabled must be true" in message
    assert "an upstream authorization service must be configured" in message


def test_reference_profile_rejects_placeholder_salt_and_missing_model() -> None:
    config = load_config("configs/default.yaml")
    with pytest.raises(ValueError, match="unsafe"):
        enforce_runtime_guard(
            replace(
                config,
                server=replace(
                    config.server,
                    profile="production",
                    tls_enabled=True,
                    tls_cert_file="/tmp/finnet-test-cert.pem",
                    tls_key_file="/tmp/finnet-test-key.pem",
                    tls_ca_file="/tmp/finnet-test-ca.pem",
                    require_client_certificate=True,
                    mac_required=True,
                    mac_secret_file="/tmp/finnet-test-mac",
                    upstream_host="upstream.example.test",
                    upstream_port=9100,
                    replay_protection=True,
                    trust_client_subnet=False,
                ),
            )
        )


def test_reference_profile_passes_with_files(tmp_path: Path) -> None:
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    ca = tmp_path / "ca.pem"
    mac = tmp_path / "mac.bin"
    model = tmp_path / "risk_gnn.pt"
    for path in (cert, key, ca, mac, model):
        path.write_bytes(b"test")
    config = load_config("configs/default.yaml")
    config = replace(
        config,
        server=replace(
            config.server,
            profile="production",
            tls_enabled=True,
            tls_cert_file=str(cert),
            tls_key_file=str(key),
            tls_ca_file=str(ca),
            require_client_certificate=True,
            mac_required=True,
            mac_secret_file=str(mac),
            upstream_host="upstream.example.test",
            upstream_port=9100,
            replay_protection=True,
            trust_client_subnet=False,
        ),
        policy=replace(config.policy, fail_open_on_timeout=False),
        model=replace(config.model, checkpoint=str(model)),
        data=replace(config.data, pan_salt="unique-test-salt"),
    )
    enforce_runtime_guard(config)
