"""Binary ISO-8583 codec with bitmap decoding and 2-byte TCP framing.

The implementation deliberately avoids third-party ISO libraries: the project's
network-engineering objective is to show how the wire format works. It supports
fixed fields, LLVAR/LLLVAR ASCII fields, binary fields, primary and secondary
bitmaps, and common card-authorisation fields.
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Literal

FieldKind = Literal["fixed", "llvar", "lllvar"]


class ISO8583Error(ValueError):
    """Raised when a frame is malformed or contains an unsupported field."""


@dataclass(frozen=True)
class FieldSpec:
    number: int
    name: str
    kind: FieldKind
    length: int
    binary: bool = False
    description: str = ""


# A practical subset of the ISO-8583:1987/2003 data elements used by card
# authorisation traffic. Unknown bitmap fields are rejected rather than skipped,
# because skipping without a schema silently corrupts the remaining message.
FIELD_SPECS: dict[int, FieldSpec] = {
    2: FieldSpec(2, "PAN", "llvar", 19, description="Primary account number"),
    3: FieldSpec(3, "processing_code", "fixed", 6),
    4: FieldSpec(4, "amount", "fixed", 12, description="Minor currency units"),
    5: FieldSpec(5, "settlement_amount", "fixed", 12),
    6: FieldSpec(6, "billing_amount", "fixed", 12),
    7: FieldSpec(7, "transmission_datetime", "fixed", 10, description="MMDDhhmmss"),
    9: FieldSpec(9, "conversion_rate", "fixed", 8),
    10: FieldSpec(10, "conversion_rate_settlement", "fixed", 8),
    11: FieldSpec(11, "stan", "fixed", 6),
    12: FieldSpec(12, "local_time", "fixed", 6, description="hhmmss"),
    13: FieldSpec(13, "local_date", "fixed", 4, description="MMDD"),
    14: FieldSpec(14, "expiry_date", "fixed", 4, description="YYMM"),
    18: FieldSpec(18, "mcc", "fixed", 4),
    19: FieldSpec(19, "acquiring_country", "fixed", 3),
    22: FieldSpec(22, "pos_entry_mode", "fixed", 3),
    23: FieldSpec(23, "card_sequence", "fixed", 3),
    25: FieldSpec(25, "pos_condition", "fixed", 2),
    32: FieldSpec(32, "acquirer_id", "llvar", 11),
    35: FieldSpec(35, "track2", "llvar", 37),
    37: FieldSpec(37, "rrn", "fixed", 12),
    38: FieldSpec(38, "approval_code", "fixed", 6),
    39: FieldSpec(39, "response_code", "fixed", 2),
    41: FieldSpec(41, "terminal_id", "fixed", 8, description="Device fingerprint"),
    42: FieldSpec(42, "merchant_id", "fixed", 15),
    43: FieldSpec(43, "merchant_name", "fixed", 40),
    44: FieldSpec(44, "additional_response_data", "llvar", 25),
    45: FieldSpec(45, "track1", "llvar", 76),
    48: FieldSpec(48, "additional_data", "lllvar", 999),
    49: FieldSpec(49, "currency", "fixed", 3),
    50: FieldSpec(50, "settlement_currency", "fixed", 3),
    51: FieldSpec(51, "billing_currency", "fixed", 3),
    52: FieldSpec(52, "pin_block", "fixed", 8, binary=True),
    54: FieldSpec(54, "additional_amounts", "lllvar", 120),
    55: FieldSpec(55, "emv_data", "lllvar", 999),
    60: FieldSpec(60, "private_60", "lllvar", 999, description="Gateway decision metadata"),
    61: FieldSpec(61, "private_61", "lllvar", 999),
    62: FieldSpec(62, "private_62", "lllvar", 999, description="Client send timestamp / flow metadata"),
    63: FieldSpec(63, "private_63", "lllvar", 999),
    64: FieldSpec(64, "mac", "fixed", 8, binary=True),
    70: FieldSpec(70, "network_management_code", "fixed", 3),
    90: FieldSpec(90, "original_data", "fixed", 42),
    95: FieldSpec(95, "replacement_amounts", "fixed", 42),
    100: FieldSpec(100, "receiving_institution", "llvar", 11),
    102: FieldSpec(102, "account_id_1", "llvar", 28),
    103: FieldSpec(103, "account_id_2", "llvar", 28),
    128: FieldSpec(128, "mac_extended", "fixed", 8, binary=True),
}

_RESPONSE_MTI = {
    "0100": "0110",
    "0200": "0210",
    "0220": "0230",
    "0400": "0410",
    "0800": "0810",
}

_COPY_TO_RESPONSE = (2, 3, 4, 7, 11, 12, 37, 41, 42, 49)


@dataclass
class ISO8583Message:
    mti: str
    fields: dict[int, str | bytes] = field(default_factory=dict)
    raw: bytes = b""
    received_at: float = 0.0

    @property
    def amount_minor(self) -> int:
        value = self.fields.get(4, "0")
        try:
            return int(str(value))
        except ValueError:
            return 0

    @property
    def amount(self) -> Decimal:
        try:
            return Decimal(str(self.fields.get(4, "0"))) / Decimal(100)
        except InvalidOperation:
            return Decimal(0)

    @property
    def pan(self) -> str:
        value = self.fields.get(2, "")
        return str(value)

    @property
    def masked_pan(self) -> str:
        pan = self.pan
        if len(pan) < 10:
            return "*" * len(pan)
        return f"{pan[:6]}{'*' * (len(pan) - 10)}{pan[-4:]}"

    @property
    def terminal_id(self) -> str:
        return str(self.fields.get(41, "")).strip()

    @property
    def merchant_id(self) -> str:
        return str(self.fields.get(42, "")).strip()

    @property
    def stan(self) -> str:
        return str(self.fields.get(11, "")).strip()

    @property
    def rrn(self) -> str:
        return str(self.fields.get(37, "")).strip()

    @property
    def currency(self) -> str:
        return str(self.fields.get(49, "356")).strip()

    @property
    def response_code(self) -> str:
        return str(self.fields.get(39, "")).strip()

    @property
    def client_timestamp_ms(self) -> float | None:
        """Optional monotonic-ish client send timestamp used for RTT telemetry.

        Private field 62 is untrusted input; the gateway clamps implausible
        values and never uses it as a security decision by itself.
        """

        raw = self.fields.get(62)
        if raw is None:
            return None
        text = raw.decode("ascii", "ignore") if isinstance(raw, bytes) else str(raw)
        for token in text.replace(";", "&").split("&"):
            key, _, value = token.partition("=")
            if key.strip().lower() in {"ts_ms", "client_ts_ms"}:
                try:
                    return float(value)
                except ValueError:
                    return None
        try:
            return float(text)
        except ValueError:
            return None

    @property
    def transmission_datetime(self) -> datetime | None:
        value = str(self.fields.get(7, ""))
        if len(value) != 10:
            return None
        try:
            now = datetime.now()
            return datetime.strptime(f"{now.year}{value}", "%Y%m%d%H%M%S")
        except ValueError:
            return None


class ISO8583Codec:
    """Encode and decode unframed ISO-8583 payloads."""

    def __init__(self, specs: dict[int, FieldSpec] | None = None) -> None:
        self.specs = specs or FIELD_SPECS

    # ------------------------------------------------------------------ decode
    def decode(self, payload: bytes) -> ISO8583Message:
        if len(payload) < 12:
            raise ISO8583Error(f"payload too short: {len(payload)} bytes")
        try:
            mti = payload[:4].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ISO8583Error("MTI is not ASCII") from exc
        if not mti.isdigit():
            raise ISO8583Error(f"invalid MTI: {mti!r}")

        primary = payload[4:12]
        has_secondary = self._bit(primary, 1)
        bitmap_len = 16 if has_secondary else 8
        if len(payload) < 4 + bitmap_len:
            raise ISO8583Error("truncated bitmap")
        bitmap = payload[4 : 4 + bitmap_len]
        offset = 4 + bitmap_len
        fields: dict[int, str | bytes] = {}

        for number in range(2, 65 if not has_secondary else 129):
            if not self._bit(bitmap, number):
                continue
            spec = self.specs.get(number)
            if spec is None:
                raise ISO8583Error(f"bitmap references unsupported field {number}")
            value, offset = self._read_field(payload, offset, spec)
            fields[number] = value

        if offset != len(payload):
            raise ISO8583Error(
                f"trailing bytes after decode: {len(payload) - offset} (offset {offset})"
            )
        return ISO8583Message(mti=mti, fields=fields, raw=payload)

    def decode_frame(self, frame: bytes) -> ISO8583Message:
        if len(frame) < 2:
            raise ISO8583Error("frame shorter than length prefix")
        (length,) = struct.unpack(">H", frame[:2])
        if length != len(frame) - 2:
            raise ISO8583Error(f"frame length mismatch: prefix={length}, body={len(frame) - 2}")
        return self.decode(frame[2:])

    # ------------------------------------------------------------------ encode
    def encode(self, message: ISO8583Message | str, fields: dict[int, str | bytes] | None = None) -> bytes:
        if isinstance(message, ISO8583Message):
            mti, values = message.mti, message.fields
        else:
            mti, values = message, fields or {}
        if len(mti) != 4 or not mti.isdigit():
            raise ISO8583Error(f"invalid MTI: {mti!r}")
        if 1 in values:
            raise ISO8583Error("field 1 is the secondary-bitmap indicator and must not be supplied")
        for number in values:
            if number not in self.specs:
                raise ISO8583Error(f"unsupported field {number}")

        has_secondary = any(number > 64 for number in values)
        bitmap = bytearray(16 if has_secondary else 8)
        if has_secondary:
            self._set_bit(bitmap, 1)
        for number in values:
            self._set_bit(bitmap, number)

        out = bytearray(mti.encode("ascii"))
        out.extend(bitmap)
        for number in sorted(values):
            spec = self.specs[number]
            out.extend(self._write_field(values[number], spec))
        return bytes(out)

    def encode_frame(self, message: ISO8583Message | str, fields: dict[int, str | bytes] | None = None) -> bytes:
        payload = self.encode(message, fields)
        if len(payload) > 65_535:
            raise ISO8583Error(f"payload exceeds 2-byte frame limit: {len(payload)}")
        return struct.pack(">H", len(payload)) + payload

    # ------------------------------------------------------------------ helpers
    def make_response(
        self,
        request: ISO8583Message,
        response_code: str,
        *,
        action: str,
        score: float,
        reason: str,
        extra_fields: dict[int, str | bytes] | None = None,
    ) -> ISO8583Message:
        response_mti = _RESPONSE_MTI.get(request.mti, request.mti[:2] + "10")
        fields: dict[int, str | bytes] = {}
        for number in _COPY_TO_RESPONSE:
            if number in request.fields:
                fields[number] = request.fields[number]
        fields[39] = response_code[:2].rjust(2, "0")
        fields[60] = f"action={action};score={score:.4f};reason={reason}"[:999]
        if extra_fields:
            fields.update(extra_fields)
        return ISO8583Message(mti=response_mti, fields=fields)

    @staticmethod
    def _bit(bitmap: bytes, number: int) -> bool:
        if number < 1 or number > len(bitmap) * 8:
            return False
        index = number - 1
        return bool(bitmap[index // 8] & (0x80 >> (index % 8)))

    @staticmethod
    def _set_bit(bitmap: bytearray, number: int) -> None:
        index = number - 1
        bitmap[index // 8] |= 0x80 >> (index % 8)

    def _read_field(self, payload: bytes, offset: int, spec: FieldSpec) -> tuple[str | bytes, int]:
        if spec.kind == "fixed":
            length = spec.length
        else:
            prefix_len = 2 if spec.kind == "llvar" else 3
            if offset + prefix_len > len(payload):
                raise ISO8583Error(f"truncated length prefix for field {spec.number}")
            try:
                length = int(payload[offset : offset + prefix_len].decode("ascii"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise ISO8583Error(f"invalid length prefix for field {spec.number}") from exc
            offset += prefix_len
            if length > spec.length:
                raise ISO8583Error(
                    f"field {spec.number} length {length} exceeds schema max {spec.length}"
                )
        end = offset + length
        if end > len(payload):
            raise ISO8583Error(f"truncated field {spec.number}: need {length} bytes")
        raw = payload[offset:end]
        if spec.binary:
            return raw, end
        try:
            return raw.decode("ascii"), end
        except UnicodeDecodeError as exc:
            raise ISO8583Error(f"field {spec.number} is not ASCII") from exc

    def _write_field(self, value: str | bytes, spec: FieldSpec) -> bytes:
        if isinstance(value, bytes):
            if not spec.binary:
                raise ISO8583Error(f"field {spec.number} expects text, got bytes")
            body = value
        else:
            if spec.binary:
                raise ISO8583Error(f"field {spec.number} expects bytes, got text")
            try:
                body = value.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ISO8583Error(f"field {spec.number} must be ASCII") from exc
        if len(body) > spec.length:
            raise ISO8583Error(
                f"field {spec.number} length {len(body)} exceeds schema max {spec.length}"
            )
        if spec.kind == "fixed":
            if len(body) != spec.length:
                raise ISO8583Error(
                    f"fixed field {spec.number} requires {spec.length} bytes, got {len(body)}"
                )
            return body
        prefix_len = 2 if spec.kind == "llvar" else 3
        return f"{len(body):0{prefix_len}d}".encode("ascii") + body


def tokenize_pan(pan: str, salt: str) -> str:
    """Return a stable keyed pseudonym; raw PANs never enter the graph."""

    digest = hmac.new(salt.encode("utf-8"), pan.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"card_{digest[:20]}"


def subnet_for_ip(ip: str) -> str:
    """Map a peer address to a policy-friendly subnet (IPv4 /24, IPv6 /64)."""

    import ipaddress

    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return "0.0.0.0/24"
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped:
            address = address.ipv4_mapped
        else:
            return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(ipaddress.ip_network(f"{address}/24", strict=False))


def subnet_sample_ip(subnet: str) -> str:
    network = __import__("ipaddress").ip_network(subnet, strict=False)
    if network.version == 4:
        return str(network.network_address + 10)
    return str(network.network_address + 10)


def normalize_fields(fields: Iterable[tuple[int, str | bytes]]) -> dict[int, str | bytes]:
    return {number: value for number, value in fields}


def client_network_claim(message: ISO8583Message) -> tuple[str | None, str | None]:
    """Read test-harness source network metadata from ISO private field 61.

    Returns ``(src_ip, subnet)``. Values are validated as IP/CIDR but remain
    untrusted claims; callers must gate their use behind an explicit test-only
    configuration flag.
    """

    import ipaddress

    raw = message.fields.get(61)
    if raw is None:
        return None, None
    text = raw.decode("ascii", "ignore") if isinstance(raw, bytes) else str(raw)
    values: dict[str, str] = {}
    for token in text.replace(";", "&").split("&"):
        key, _, value = token.partition("=")
        values[key.strip().lower()] = value.strip()
    src_ip = values.get("src_ip")
    subnet = values.get("subnet")
    if subnet:
        try:
            subnet = str(ipaddress.ip_network(subnet, strict=False))
        except ValueError:
            subnet = None
    if src_ip:
        try:
            address = ipaddress.ip_address(src_ip)
            src_ip = str(address)
            if subnet is None:
                subnet = subnet_for_ip(src_ip)
        except ValueError:
            src_ip = None
    if src_ip is None and subnet is not None:
        src_ip = subnet_sample_ip(subnet)
    return src_ip, subnet


def likely_rtt_ms(client_ts_ms: float | None, now_ms: float) -> float | None:
    """Validate an untrusted client timestamp before using it as RTT telemetry."""

    if client_ts_ms is None:
        return None
    delta = now_ms - client_ts_ms
    if 0.0 <= delta <= 60_000.0:
        return delta
    return None
