"""IEEE-CIS Fraud Detection dataset adapter.

Important dataset note
----------------------
IEEE-CIS is tabular and anonymised: it contains no IP packets, TCP RTT, device
fingerprints or merchant terminal IDs. We therefore deterministically derive
pseudonymous graph entities from the available columns and augment missing
network telemetry. This is documented in the report and must not be presented
as captured network data. The gateway's online telemetry is measured from real
sockets; the offline adapter exists to benchmark the graph model on a public,
large-scale fraud distribution.

Expected files (Kaggle competition "ieee-fraud-detection"):
    data/train_transaction.csv
    data/train_identity.csv
"""

from __future__ import annotations

import hashlib
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..iso8583 import tokenize_pan
from .common import TransactionEvent

_TX_CANDIDATES = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
]
_ID_CANDIDATES = [
    "TransactionID",
    "DeviceType",
    "DeviceInfo",
    "id_17",
    "id_19",
    "id_20",
    "id_30",
    "id_31",
    "id_33",
]


class IEEECISError(RuntimeError):
    pass


def load_ieee_cis(
    transactions_csv: str | Path,
    identity_csv: str | Path | None = None,
    *,
    max_events: int | None = None,
    pan_salt: str = "finnet-local-salt-change-me",
) -> list[TransactionEvent]:
    tx_path = Path(transactions_csv)
    if not tx_path.exists():
        raise IEEECISError(
            f"{tx_path} not found. Download the competition data and place "
            "train_transaction.csv / train_identity.csv in data/."
        )
    tx_columns = _intersect_columns(tx_path, _TX_CANDIDATES)
    if not {"TransactionID", "isFraud", "TransactionDT", "TransactionAmt"}.issubset(tx_columns):
        raise IEEECISError(f"{tx_path} is missing required IEEE-CIS columns")

    frame = pd.read_csv(tx_path, usecols=tx_columns)
    id_path = Path(identity_csv) if identity_csv else None
    if id_path and id_path.exists():
        id_columns = _intersect_columns(id_path, _ID_CANDIDATES)
        if "TransactionID" in id_columns and len(id_columns) > 1:
            identity = pd.read_csv(id_path, usecols=id_columns)
            frame = frame.merge(identity, on="TransactionID", how="left")
    frame = frame.sort_values("TransactionDT", kind="stable")
    if max_events:
        frame = frame.iloc[:max_events]
    return _frame_to_events(frame, pan_salt=pan_salt)


def _intersect_columns(path: Path, candidates: list[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0)
    return [column for column in candidates if column in header.columns]


def _frame_to_events(frame: pd.DataFrame, *, pan_salt: str) -> list[TransactionEvent]:
    events: list[TransactionEvent] = []
    for row in frame.itertuples(index=False):
        data = row._asdict()
        transaction_id = _clean(data.get("TransactionID"))
        timestamp = _number(data.get("TransactionDT")) or 0.0
        amount = _number(data.get("TransactionAmt")) or 0.0
        card_material = "|".join(
            [
                _clean(data.get("card1")),
                _clean(data.get("card2")),
                _clean(data.get("card3")),
                _clean(data.get("card4")),
                _clean(data.get("card6")),
            ]
        )
        card_token = tokenize_pan(card_material, pan_salt)
        device_material = "|".join(
            [
                _clean(data.get("DeviceInfo")),
                _clean(data.get("DeviceType")),
                _clean(data.get("id_30")),
                _clean(data.get("id_31")),
                _clean(data.get("id_33")),
            ]
        )
        if not device_material.strip("|"):
            device_material = card_material + "|" + _clean(data.get("addr1"))
        device_hash = _digest(device_material)
        device_id = f"T{device_hash[:7].upper()}"

        addr1 = _number(data.get("addr1"))
        addr2 = _number(data.get("addr2"))
        if addr1 is None:
            addr1 = float(int(card_token[-8:], 16) % 250) + 1
        if addr2 is None:
            addr2 = float(int(device_hash[-8:], 16) % 250) + 1
        subnet = f"10.{int(addr1) % 250}.{int(addr2) % 250}.0/24"
        src_ip = subnet.rsplit(".", 1)[0] + "." + str(10 + int(device_hash[:4], 16) % 200)

        merchant_material = "|".join(
            [
                _clean(data.get("ProductCD")),
                _clean(data.get("P_emaildomain")),
                _clean(data.get("M4")),
                _clean(data.get("M5")),
            ]
        )
        merchant_id = f"M{_digest(merchant_material)[:12].upper()}"

        label = int(_number(data.get("isFraud")) or 0)
        payload = (
            f"ieee:{transaction_id}:{amount:.2f}:{card_token}:{device_id}:"
            f"{merchant_id}:{timestamp:.0f}"
        ).encode()
        # Deterministic pseudo-RTT augmentation. Clearly flagged in the report:
        # IEEE-CIS does not carry network measurements.
        pseudo_rtt = 4.0 + (int(device_hash[:6], 16) % 4100) / 100.0
        events.append(
            TransactionEvent(
                event_id=f"ieee_{transaction_id}",
                timestamp=float(timestamp),
                src_ip=src_ip,
                subnet=subnet,
                device_id=device_id,
                card_token=card_token,
                merchant_id=merchant_id,
                amount=float(amount),
                currency="USD",
                terminal_id=device_id,
                stan=str(transaction_id)[-6:],
                label=label,
                attack_type="fraud" if label else "normal",
                rtt_ms=pseudo_rtt,
                payload_size=len(payload),
                raw_payload=payload,
            )
        )
    return events


def _clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "ignore")).hexdigest()


def download_ieee_cis(data_dir: str | Path = "data") -> Path:
    """Download via the Kaggle CLI, if credentials are configured."""

    destination = Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "ieee-fraud-detection.zip"
    try:
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                "ieee-fraud-detection",
                "-p",
                str(destination),
            ],
            check=True,
        )
    except FileNotFoundError as exc:
        raise IEEECISError(
            "Kaggle CLI not installed. Run `pip install kaggle`, configure "
            "~/.kaggle/kaggle.json, then retry."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise IEEECISError(
            "Kaggle download failed. Accept the competition rules on Kaggle and "
            "verify ~/.kaggle/kaggle.json."
        ) from exc
    if archive.exists():
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(destination)
    return destination
