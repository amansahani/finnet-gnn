from __future__ import annotations

import pandas as pd

from finnet.data.ieee_cis import IEEECISError, load_ieee_cis


def test_ieee_loader_maps_tabular_rows(tmp_path) -> None:
    transactions = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "isFraud": [0, 1],
            "TransactionDT": [100, 200],
            "TransactionAmt": [10.0, 500.0],
            "ProductCD": ["W", "C"],
            "card1": [1000, 1000],
            "card2": [200, 200],
            "card3": [150, 150],
            "card4": ["visa", "visa"],
            "card6": ["credit", "credit"],
            "addr1": [123, 123],
            "addr2": [45, 45],
            "P_emaildomain": ["gmail.com", "gmail.com"],
            "M4": ["M0", "M0"],
            "M5": ["M1", "M1"],
            "DeviceInfo": ["Windows", "Windows"],
            "id_31": ["chrome", "chrome"],
        }
    )
    path = tmp_path / "train_transaction.csv"
    transactions.to_csv(path, index=False)
    events = load_ieee_cis(path, max_events=10)
    assert len(events) == 2
    assert events[1].label == 1
    assert events[0].subnet.endswith("/24")
    assert events[0].device_id.startswith("T")
    assert events[0].merchant_id.startswith("M")
    assert events[0].card_token.startswith("card_")
    assert events[0].rtt_ms is not None


def test_missing_ieee_dataset_is_actionable(tmp_path) -> None:
    try:
        load_ieee_cis(tmp_path / "missing.csv")
    except IEEECISError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("expected IEEECISError")
