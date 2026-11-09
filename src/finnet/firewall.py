"""Firewall quarantine adapter.

The in-process quarantine table is always authoritative. Operating-system
firewall mutation is opt-in and audited because a /24 block on a developer
machine can disconnect the host. By default FinNet writes intent records only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .config import PolicyConfig

SAFE_FIREWALL_BACKENDS = {"audit", "pf", "iptables"}


class FirewallAdapter:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config
        if config.firewall_backend not in SAFE_FIREWALL_BACKENDS:
            raise ValueError(f"Unsupported firewall backend: {config.firewall_backend}")
        self._lock = threading.Lock()
        self.audit_path = Path(config.firewall_audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._blocked: set[str] = set()

    @property
    def active(self) -> bool:
        return self.config.enable_os_firewall

    def block(self, cidr: str, *, reason: str, score: float, ttl_s: float) -> bool:
        with self._lock:
            self._blocked.add(cidr)
            event = {
                "ts": time.time(),
                "operation": "block",
                "cidr": cidr,
                "reason": reason,
                "score": round(score, 6),
                "ttl_s": ttl_s,
                "backend": self.config.firewall_backend if self.active else "audit",
                "ok": True,
            }
            if self.active:
                ok, detail = self._execute("block", cidr)
                event["ok"] = ok
                event["detail"] = detail
            self._append_audit(event)
            return bool(event["ok"])

    def unblock(self, cidr: str) -> bool:
        with self._lock:
            self._blocked.discard(cidr)
            event: dict[str, Any] = {
                "ts": time.time(),
                "operation": "unblock",
                "cidr": cidr,
                "backend": self.config.firewall_backend if self.active else "audit",
                "ok": True,
            }
            if self.active:
                ok, detail = self._execute("unblock", cidr)
                event["ok"] = ok
                event["detail"] = detail
            self._append_audit(event)
            return bool(event["ok"])

    def blocked_count(self) -> int:
        with self._lock:
            return len(self._blocked)

    def _execute(self, operation: str, cidr: str) -> tuple[bool, str]:
        backend = self.config.firewall_backend
        if backend == "pf":
            action = "-T add" if operation == "block" else "-T delete"
            command = ["pfctl", "-t", "finnet_quarantine", *action.split(), cidr]
        elif backend == "iptables":
            action = "-I" if operation == "block" else "-D"
            command = ["iptables", action, "INPUT", "-s", cidr, "-j", "DROP"]
        else:
            return True, "audit-only"
        if not shutil.which(command[0]):
            return False, f"{command[0]} not found"
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.config.firewall_command_timeout_s,
                check=False,
            )
            return result.returncode == 0, (result.stderr or result.stdout).strip()[:500]
        except Exception as exc:  # noqa: BLE001 - adapter must never crash the gateway
            return False, f"{type(exc).__name__}: {exc}"

    def _append_audit(self, event: dict[str, Any]) -> None:
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
