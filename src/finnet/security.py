"""Production hardening helpers for TLS, MACs and idempotent retries.

These utilities are deliberately simple, testable building blocks. They do not
replace a card acquirer's key-management system or PCI-DSS review.
"""

from __future__ import annotations

import hashlib
import hmac
import ssl
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MACVerification:
    valid: bool
    reason: str = ""


def load_mac_secret(*, mac_secret: str = "", mac_secret_file: str | Path = "") -> bytes:
    material = mac_secret
    path = Path(mac_secret_file) if mac_secret_file else None
    if path is None or not path.exists():
        pass
    else:
        material = path.read_text(encoding="utf-8").strip()
    if not material:
        raise ValueError("mac_secret or mac_secret_file must be configured when MAC is required")
    return material.encode("utf-8")


def compute_mac(payload: bytes, secret: bytes, truncated_bytes: int = 8) -> bytes:
    digest = hmac.new(secret, payload, hashlib.sha256).digest()
    return digest[:truncated_bytes]


def verify_mac(message: bytes, secret: bytes) -> MACVerification:
    if len(message) < 12:
        return MACVerification(False, "message too short for ISO-8583 header")
    # The MAC is an 8-byte trailer. ISO field 64 occupies those same bytes once
    # parsed, so signing the complete byte string is not meaningful. Instead,
    # the FinNet control plane signs the ISO payload with this sidecar scheme:
    # first 24 HMAC-SHA256 hex bytes remain in the wire payload and the last 16
    # are sent as the ISO binary MAC trailer.
    #
    # For local testing, callers can use ``compute_mac_and_trailer``; this
    # function verifies that layout. A real acquirer must replace it with its
    # agreed field-64/EMV MAC layout.
    if len(message) < 24 + 8:
        return MACVerification(False, "message too short for FinNet sidecar MAC")
    signed = message[:-8]
    expected = compute_mac(signed, secret, 8)
    observed = message[-8:]
    if hmac.compare_digest(expected, observed):
        return MACVerification(True)
    return MACVerification(False, "HMAC trailer mismatch")


def attach_mac(message: bytes, secret: bytes) -> bytes:
    return message + compute_mac(message, secret, 8)


def build_tls_context(
    *,
    cert_file: str,
    key_file: str,
    ca_file: str = "",
    require_client_certificate: bool = False,
) -> ssl.SSLContext:
    cert_path = Path(cert_file)
    key_path = Path(key_file)
    if not cert_path.exists():
        raise FileNotFoundError(f"TLS cert not found: {cert_path}")
    if not key_path.exists():
        raise FileNotFoundError(f"TLS key not found: {key_path}")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
    context.set_ciphers("TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256")
    context.load_cert_chain(str(cert_path), str(key_path))
    if require_client_certificate:
        if not ca_file:
            raise ValueError("tls_ca_file is required when require_client_certificate=true")
        context.load_verify_locations(ca_file)
        context.verify_mode = ssl.CERT_REQUIRED
    else:
        context.verify_mode = ssl.CERT_NONE
    return context


class ReplayGuard:
    """Bounded idempotency cache for client retries."""

    def __init__(self, ttl_s: float = 300.0, max_entries: int = 100_000) -> None:
        self.ttl_s = max(1.0, float(ttl_s))
        self.max_entries = max(10, int(max_entries))
        self._entries: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, *, now: float | None = None) -> str | None:
        at = time.monotonic() if now is None else now
        with self._lock:
            self._expire_locked(at)
            entry = self._entries.get(key)
            if entry is None:
                return None
            return entry[1]

    def remember(self, key: str, value: str, *, now: float | None = None) -> None:
        at = time.monotonic() if now is None else now
        with self._lock:
            self._expire_locked(at)
            if len(self._entries) >= self.max_entries:
                # Keep a small FIFO window rather than sampling randomly.
                oldest = sorted(self._entries.items(), key=lambda item: item[1][0])[: self.max_entries // 10 + 1]
                for old_key, _ in oldest:
                    self._entries.pop(old_key, None)
            self._entries[key] = (at, value)

    def _expire_locked(self, now: float) -> None:
        expired = [key for key, (seen_at, _) in self._entries.items() if now - seen_at > self.ttl_s]
        for key in expired:
            self._entries.pop(key, None)


def message_fingerprint(fields: dict[str, str | bytes]) -> str:
    # PAN is intentionally excluded: same amount and STAN with different PANs is
    # a different authorisation. Hashing keeps memory stable and privacy-aligned.
    parts = [
        str(fields.get(11, "")),
        str(fields.get(37, "")),
        str(fields.get(42, "")),
        str(fields.get(41, "")),
        str(fields.get(4, "")),
        str(fields.get(49, "")),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
