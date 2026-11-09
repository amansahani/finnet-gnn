# Datasets and Evaluation Protocol

FinNet-GNN supports two data sources with a shared event contract:

- **IEEE-CIS Fraud Detection** (Kaggle) — the primary public benchmark.
- **Synthetic stream** — a reproducible development and demonstration harness.

Both produce `data.common.TransactionEvent`, so the graph, telemetry, training
and policy code are identical. What differs is provenance and honesty about
network fields.

> **Read this first:** IEEE-CIS is anonymised, tabular transaction data. It has
> no IP addresses, packets, TCP round-trip times, device fingerprints or merchant
> terminal IDs. The IEEE adapter **derives** pseudonymous entities from available
> columns and **augments** a deterministic pseudo-RTT. Those values must be
> described as derived, never as captured network telemetry. The live gateway’s
> telemetry is measured from real sockets; the offline adapter exists to
> benchmark the graph model at scale.

---

## 1. Event contract

```python
@dataclass(frozen=True)
class TransactionEvent:
    event_id: str
    timestamp: float
    src_ip: str
    subnet: str
    device_id: str
    card_token: str
    merchant_id: str
    amount: float
    currency: str = "356"
    terminal_id: str = ""
    stan: str = ""
    label: int = 0
    attack_type: str = "normal"
    rtt_ms: float | None = None
    payload_size: int = 0
    raw_payload: bytes = b""
```

Invariants used by downstream code:

- `timestamp` is a float in seconds, monotonically sortable.
- `subnet` is a canonical CIDR string (`ip_network(..., strict=False)`).
- `card_token` is a keyed pseudonym, never a raw PAN.
- `device_id` is at most 8 characters when it must fit ISO field 41.
- `label` is 1 for fraud, 0 for normal.
- `attack_type` is a human-readable category used only for analysis.
- `raw_payload` feeds payload-entropy telemetry; it is excluded from equality.

---

## 2. IEEE-CIS Fraud Detection

### 2.1 What the dataset contains

The competition ships two CSVs:

| File | Rows (approx.) | Contents |
|---|---:|---|
| `train_transaction.csv` | 590,540 | TransactionID, isFraud, TransactionDT, TransactionAmt, card1–card6, addr1/addr2, ProductCD, email domains, C1–C14, D1–D15, M1–M9, V1–V339 |
| `train_identity.csv` | 144,233 | TransactionID, DeviceType, DeviceInfo, id_01–id_38 (device/browser/OS signals for a subset) |

The identity file covers only ~24% of transactions, so joins are left-outer and
many rows lack device signals.

### 2.2 Download

1. Create a Kaggle account.
2. Open <https://www.kaggle.com/c/ieee-fraud-detection/rules> and accept the
   competition rules.
3. Create an API token at <https://www.kaggle.com/settings> →
   **Create New Token**, which downloads `kaggle.json`.
4. Install and configure the CLI:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
pip install kaggle
```

5. Download through the project:

```bash
cd iiit-mtech/IOT/finnet-gnn
pip install -e ".[kaggle]"
finnet download-data --dest data
```

Expected result:

```text
data/
├── ieee-fraud-detection.zip
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

The loader only needs the two `train_*` files.

### 2.3 Schema mapping

The adapter reads only the columns it needs, so memory stays bounded.

| Graph entity | IEEE-CIS source | Derivation |
|---|---|---|
| **Card token** | `card1`, `card2`, `card3`, `card4`, `card6` | concatenated, then keyed HMAC-SHA256 (`tokenize_pan`) → `card_<20 hex>` |
| **Device fingerprint** | `DeviceInfo`, `DeviceType`, `id_30`, `id_31`, `id_33` | concatenated + SHA-256, first 7 hex chars → `T<7 hex>` |
| **Device fallback** | card material + `addr1` | used when the identity columns are all empty |
| **Subnet** | `addr1`, `addr2` | `10.<addr1 mod 250>.<addr2 mod 250>.0/24`; missing values are filled deterministically from card/device hashes |
| **Source IP** | derived from subnet + device hash | `<subnet prefix>.<10 + hash mod 200>` |
| **Merchant group** | `ProductCD`, `P_emaildomain`, `M4`, `M5` | concatenated + SHA-256, first 12 hex chars → `M<12 hex>` |
| **Amount** | `TransactionAmt` | used directly |
| **Timestamp** | `TransactionDT` | seconds offset from an arbitrary epoch; used for ordering and time-of-day features |
| **Label** | `isFraud` | 1 = fraud |
| **RTT proxy** | none | deterministic `4.0 + (device_hash mod 4100)/100` ms — **augmentation** |
| **Payload** | none | short synthetic byte string built from the row; used only for entropy |

All derived identifiers are stable for a given row and salt, so the graph can
connect transactions that share a card, device, address, or merchant group.

### 2.4 What is *not* available

- No source IP, port, ASN, geo-IP, or packet capture.
- No TCP RTT, jitter, retransmission or throughput.
- No real device fingerprint; `DeviceInfo` is a coarse browser/OS string for a
  subset of rows.
- No merchant terminal ID; `M*` columns are anonymised match flags.
- No STAN/RRN/ISO-8583 message structure.
- Timestamps are relative offsets, not wall-clock dates.

Consequences for the report:

- Do **not** claim network-flow detection performance on IEEE-CIS.
- Do **not** compare IEEE-CIS RTT/jitter numbers to a real gateway.
- Report the adapter as “graph learning over derived entities,” and keep the
  network-telemetry claims tied to the live/synthetic socket experiments.

### 2.5 Training command

```bash
finnet train --source ieee \
  --transactions data/train_transaction.csv \
  --identity data/train_identity.csv \
  --max-events 150000 \
  --epochs 12 \
  --output artifacts/risk_gnn_ieee.pt
```

Useful limits in `configs/default.yaml`:

```yaml
data:
  source: ieee
  transactions_csv: data/train_transaction.csv
  identity_csv: data/train_identity.csv
  max_events: 150000        # cap for CPU runtime
  train_fraction: 0.8
  max_train_samples: 20000
  max_test_samples: 5000
  negative_ratio: 3.0
```

`--max-events` truncates after sorting by `TransactionDT`, so you always take the
earliest fraction of the stream. Increase it if you have time and memory; the
full dataset is ~590k rows and hundreds of sparse columns (the loader reads only
the needed subset).

### 2.6 Evaluation protocol

1. Sort by `TransactionDT`.
2. Split at 80%; the splitter moves the boundary to the nearest fraud if one side
   would have none, without reversing time.
3. Replay training events through the online graph and telemetry.
4. Retain fraud snapshots up to the cap and normals at `negative_ratio`× fraud.
5. Train the same GNN used online.
6. Evaluate on the held-out tail:
   - ROC-AUC and PR-AUC;
   - precision/recall/F1 at 0.5;
   - precision/recall/F1 at each policy operating point;
   - confusion matrix;
   - per-attack-type mean score and recall (IEEE rows use `attack_type="fraud"`);
   - best threshold at ≤1% false-positive rate.

Report the tail split explicitly. A random split leaks future information through
the graph and inflates scores.

### 2.7 Reproduction checklist

- [ ] Competition rules accepted; `~/.kaggle/kaggle.json` permissions `600`.
- [ ] `data/train_transaction.csv` and `data/train_identity.csv` present.
- [ ] Salt changed from the default (`data.pan_salt`); the default is a
      development placeholder.
- [ ] `max_events` and sample caps recorded in the report.
- [ ] CPU/GPU, Torch version, and thread count recorded.
- [ ] The report states that network fields are derived.
- [ ] Checkpoint + `*.report.json` + `*.test_scores.npz` archived.

---

## 3. Synthetic stream

### 3.1 Purpose

The synthetic generator exists so the whole system is runnable and testable
without Kaggle credentials or 1+ GB of data. It is **not** a substitute for
public benchmark evaluation. Synthetic scores are expected to be near-perfect
and must not be presented as evidence of real-world performance.

### 3.2 Traffic composition

`data/synthetic.py` builds identities and three attack archetypes.

| Profile | Share of events | Behaviour |
|---|---:|---|
| Normal | 92% (default) | 900 customers, stable subnets/devices/cards, occasional new device (2.5%), short bursts of 2–7 purchases |
| Card testing | ~3.4% | 6 rotating cards per bot, very short intervals, tiny amounts, consistent device/subnet |
| Botnet | ~2.9% | 30 bots across attack and some legitimate subnets, shared small card pool, moderate amounts |
| Account takeover | ~1.7% | stolen card used from attack subnet/new device at high-value merchants |

Normal traffic deliberately includes bursts so the model cannot learn the
shortcut “fast ⇒ fraud.” Fraud remains separable through card/device fan-out,
entropy, amount and new-entity patterns.

Fraud is distributed in waves across the time horizon, which guarantees both
sides of a temporal split contain fraud.

### 3.3 Determinism

- `SyntheticProfile(seed=...)` seeds identity construction.
- Event generation uses the same `random.Random`.
- Events are sorted by `(timestamp, event_id)`.
- The simulator uses a separate seed for workload construction and jitter.

Given the same seed and count, the event stream is reproducible.

### 3.4 Commands

```bash
# inspect the stream as CSV
finnet gen-data --events 12000 --out data/synthetic_transactions.csv

# train
finnet train --source synthetic --events 12000 --epochs 12

# drive the live gateway
finnet simulate --port 8583 --events 900 --fraud-share 0.6

# one-command train + serve + attack
finnet demo
```

### 3.5 Why the simulator uses ISO field 61

macOS does not allow binding arbitrary `127.x.x.x` loopback aliases. To exercise
real per-subnet policy paths, the simulator connects from `127.0.0.1` and sends
its intended source network in ISO private field 61:

```
subnet=127.100.1.0/24;src_ip=127.100.1.10
```

The server uses that claim only when started with `--trust-client-subnet`
(`server.trust_client_subnet: true`). This is test-harness behaviour and must be
disabled in production, where the real TCP peer address is used.

---

## 4. Comparing sources honestly

| Dimension | Live gateway | Synthetic | IEEE-CIS adapter |
|---|---|---|---|
| Source IP / subnet | Real TCP peer | Simulated claim | Derived from `addr1/2` |
| Device identity | ISO field 41 | Generator | Derived hash |
| Card identity | HMAC token | Generator | HMAC token |
| Merchant | ISO field 42 | Generator | Derived hash |
| RTT | Client timestamp proxy | None | Deterministic pseudo-RTT |
| Jitter / burst | Measured | Flow replay | Flow replay over derived timestamps |
| Payload entropy | Measured | Synthetic bytes | Synthetic bytes |
| Labels | None (live) | Generator | `isFraud` |
| Suitable for | Demo / latency | Development | Model benchmark |

When writing up results, label each table with its source and state which
columns are derived. A reviewer should be able to tell immediately whether a
number came from a socket or from a CSV transformation.

---

## 5. Suggested experiments

1. **Graph ablation:** entity features only vs. telemetry only vs. full 24-D.
2. **Relation ablation:** remove transaction hub edges, or remove fan-out
   counters, and measure PR-AUC.
3. **Snapshot size:** 32 / 64 / 128 / 256 nodes vs. latency and recall.
4. **Temporal vs. random split:** quantify leakage.
5. **Threshold sensitivity:** precision/recall and quarantine volume across the
   policy bands.
6. **Latency:** p50/p95/p99 for graph + GNN at 1k/10k/100k retained transactions.
7. **Concept drift:** train on early IEEE-CIS weeks, test on later weeks.
8. **Attack-type breakdown:** card testing vs. botnet vs. takeover recall.

Record the exact command, seed, event cap, and checkpoint hash for each run.

---

## 6. Licensing and ethics

- IEEE-CIS is distributed under the Kaggle competition rules. Review them before
  redistribution. Do not commit the CSVs to this repository.
- `data/*.csv`, `data/*.zip` and `data/*.parquet` are gitignored.
- The default `pan_salt` is a placeholder. Change it before any real use and keep
  it out of version control.
- Never run the gateway against real cardholder data without a security and
  compliance review.
- Derived identifiers are pseudonymous, not anonymous. Treat them as sensitive.
