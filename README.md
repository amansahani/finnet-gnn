# FinNet-GNN

**Autonomous Financial Network Gateway with Real-Time Dynamic GNN Fraud Mitigation**

An end-to-end, pure-Python research prototype built for the IIIT Sri City M.Tech (IoT & Autonomous Systems) programme. It terminates raw ISO-8583 card traffic over TCP, builds a live transaction graph of subnets, terminals, cards and merchants, scores risk with an inductive PyTorch GNN, and closes the loop by throttling, quarantining, or tearing down attacking sockets.

> **What is real vs. dataset-derived:** the TCP server, ISO-8583 parser, flow telemetry, graph, GNN, policy engine, token bucket, quarantine and tests are real implementations. IEEE-CIS is anonymised tabular data with no IPs, packets, RTT or device IDs. Its adapter deterministically derives pseudonymous entities and pseudo-RTT to benchmark the graph model; those values are **not** captured network telemetry. Synthetic traffic exists so the gateway is runnable and reproducible without downloading 1+ GB of Kaggle data. See [Honest scope](#honest-scope).

---

## Architecture

```
                       ISO-8583 over TCP (binary, 2-byte length prefix)
                                        |
                     +------------------v------------------+
                     |  asyncio / uvloop L4 server         |
                     |  2-byte framing, max-frame guard    |
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |  ISO8583Codec                       |
                     |  MTI + primary/secondary bitmap     |
                     |  fixed / LLVAR / LLLVAR / binary    |
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |  FlowTelemetry                      |
                     |  inter-arrival, jitter, burst rate, |
                     |  payload entropy, RTT proxy         |
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |  DynamicTransactionGraph            |
                     |  subnet -- device -- card -- merchant|
                     |  + transient transaction hub nodes |
                     |  bounded TTL / fan-out counters     |
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |  Inductive Relational GNN (PyTorch) |
                     |  2-hop snapshot, 7 edge relations  |
                     |  < 20 ms budget, heuristic fallback|
                     +------------------+------------------+
                                        |
                     +------------------v------------------+
                     |  Closed-loop PolicyEngine           |
                     |  ALLOW | THROTTLE | QUARANTINE |    |
                     |  TERMINATE + firewall audit/OS hook |
                     +------------------+------------------+
                                        |
          ISO response 0210            |            TCP teardown / quarantine CIDR
          response code 00/96/05       |            token bucket per subnet
```

### Three project tasks

| Task | Implementation | Key files |
|---|---|---|
| **1. Network engineering** | Async TCP gateway, uvloop support, binary framing, ISO-8583 bitmap codec, live flow telemetry | `server.py`, `iso8583.py`, `telemetry.py` |
| **2. Graph machine learning** | Dynamic multipartite graph, 24-feature schema, 7-relation GraphSAGE, temporal training/eval, explainability | `graph.py`, `features.py`, `model.py`, `training.py`, `scorer.py` |
| **3. Autonomous mitigation** | Threshold policy, per-subnet token buckets, socket teardown, TTL quarantine, audited firewall adapter | `policy.py`, `firewall.py`, `engine.py` |

---

## Quick start

From a clone of this repository:

```bash
git clone https://github.com/amansahani/finnet-gnn.git
cd finnet-gnn

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# Optional, required for `finnet baseline-run` boosting baselines:
pip install -e ".[benchmark]"
```

If you prefer an existing virtual environment, activate it before running the commands:

```bash
source /path/to/workspace/bin/activate
cd path/to/finnet-gnn
pip install -e ".[dev]"
```

### One-command demo

```bash
finnet demo
```

This trains a demo checkpoint on synthetic traffic if none exists, starts the gateway on ephemeral ports, launches legitimate + carding + botnet clients, and prints the mitigation report.

Observed demo behaviour (reproducible synthetic run):

```text
legitimate:   120 sent / 120 responses / 0 closed
card_testing:   8 sent /   3 responses / 5 closed
botnet:        43 sent /  25 responses / 18 closed
quarantined subnets: 9 attack ranges
engine p50: 1.14 ms, p95: 1.80 ms, p99: 2.72 ms
```

### Manual workflow

```bash
# 1. Train on synthetic data
finnet train --source synthetic --events 12000 --epochs 12

# 2. Start the gateway (test mode allows simulator source subnets in ISO field 61)
finnet serve --port 8583 --metrics-port 9101 --trust-client-subnet

# 3. In a second terminal, drive legitimate and attack traffic
finnet simulate --port 8583 --events 900 --fraud-share 0.6

# 4. Scrape live metrics
curl -s http://127.0.0.1:9101/metrics | grep finnet

# 5. Benchmark graph export + GNN scoring without networking
finnet benchmark --iterations 2000
```

> `--trust-client-subnet` is a **test-harness only** option. It trusts an ISO private-field-61 source-subnet claim so the simulator can represent many subnets on macOS where loopback aliases are unavailable. Never enable it in production; production uses the real TCP peer address.

### Dashboard / observability

- **Prometheus text endpoint:** `GET /metrics`
- **Health + engine status:** `GET /healthz`
- **Decision audit log:** `artifacts/decisions.jsonl`
- **Firewall intent log:** `artifacts/firewall.jsonl`
- **Training report:** `artifacts/risk_gnn.report.json`
- **Test scores:** `artifacts/risk_gnn.test_scores.npz`

---

## IEEE-CIS dataset

Primary benchmark: [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection/data).

```bash
# Accept the competition rules on Kaggle first, then:
pip install kaggle
# Configure ~/.kaggle/kaggle.json
finnet download-data --dest data

finnet train --source ieee \
  --transactions data/train_transaction.csv \
  --identity data/train_identity.csv \
  --max-events 150000 --epochs 12 \
  --output artifacts/risk_gnn_ieee.pt

finnet serve --model artifacts/risk_gnn_ieee.pt
```

### Dataset adapter details

IEEE-CIS has `card1..card6`, `addr1/2`, `DeviceInfo`, email domains and merchant/product fields, but no network layer. The adapter:

- tokenises card identity with keyed HMAC-SHA256 (raw PAN/card material never enters the graph);
- derives stable device fingerprints from `DeviceInfo`/`id_30/31/33` (falling back to card+address);
- derives subnet IDs from `addr1/addr2` and top-level transaction metadata;
- derives merchant groups from `ProductCD`, email domain and `M4/M5`;
- computes inter-arrival, velocity, jitter and payload entropy through the same online telemetry code;
- adds a **deterministic pseudo-RTT** because none exists in the dataset.

The adapter is for model benchmarking only. It cannot be used to claim real network measurement on IEEE-CIS. This distinction is essential for a defensible report.

---

## Model

### Graph schema

Nodes: **subnet**, **device/terminal**, **card token**, **merchant**, **transaction**.

```
subnet ---- device ---- card ---- merchant
   \          |          |          /
    +---------+----------+---------+
              transaction hub
```

Seven relation types:

1. subnet ↔ device
2. device ↔ card
3. card ↔ merchant
4. transaction ↔ subnet
5. transaction ↔ device
6. transaction ↔ card
7. transaction ↔ merchant

Edges are undirected for message passing, carry a repeat-count weight, and are bounded by transaction TTL and entity caps.

### 24 input features

Entity features: degree, transaction count, recency, average amount, risk EWMA, distinct counterpart count, blocked flag, tenure.

Flow/telemetry features: amount, hour sin/cos, inter-arrival, jitter, burst velocity, arrival rate, payload entropy, RTT proxy.

Topology features: **new subnet**, **new device**, **new card**, **new merchant**, subnet card fan-out, device card fan-out, card merchant fan-out.

### Inductive GNN

`RelationalSAGEConv` performs relation-aware mean aggregation:

```
m_ij = W_neigh · h_j + e_relation
a_i  = mean_j(edge_weight · risk_gate_j · m_ij)
h_i' = ReLU(LayerNorm(W_self · h_i + a_i))
```

- 2 layers, hidden width 64 by default. Baseline default remains ~24k parameters; `--risk-gate` and `--temporal-encoding` add small projection layers.
- Optional **risk gate** (`--risk-gate`) scales messages by the source node's propagated `risk_ewma`, so high-risk neighbors are not diluted by uniform aggregation.
- Optional **Time2Vec** (`--temporal-encoding --time-dim 16`) projects transaction time phase and broadcasts it across the local snapshot before message passing.
- Node type embeddings make subnet/device/card/merchant/transaction separable.
- Snapshots are local 2-hop subgraphs (max 128 nodes), so **unseen entities are scored immediately** with no retraining and no entity-ID lookup table.
- Training uses time-ordered replay: no future transaction leaks into a snapshot.
- Risk feedback uses a label-free topology prior, not future labels, keeping offline graph state consistent with online serving.
- Class weighting handles the fraud imbalance; `positive_weight = min(20, normal/fraud)`.

### Training protocol

1. Sort events by timestamp.
2. Time split (default 80/20) with a safeguard so both sides contain fraud.
3. Replay training events, updating graph and telemetry.
4. Retain a stratified sample of snapshots (all-fraud cap + normal sampling at `negative_ratio`).
5. Mini-batch concatenated snapshots through the GNN.
6. Evaluate held-out tail with ROC-AUC, PR-AUC, precision/recall/F1, confusion matrix, policy operating points, per-attack-type recall, and threshold-at-1%-FPR.

### Research-informed evaluation harness

Recent graph-fraud research is strongest on **average precision**, temporal splits and strong GBDT baselines. This repository includes a reproducible benchmark harness rather than relying on synthetic AUC alone.

```bash
# Train the upgraded model (adds risk gating and Time2Vec)
finnet train --source synthetic --events 12000 --epochs 12 \
  --risk-gate --temporal-encoding --time-dim 16 \
  --output artifacts/risk_gnn_v2.pt

# Compare logistic regression, histogram boosting, LightGBM, XGBoost,
# CatBoost and the current GNN on the same temporal snapshots
finnet baseline-run --source synthetic --events 12000 \
  --model artifacts/risk_gnn_v2.pt \
  --json-out artifacts/baseline_report.json

# Fit an isotonic calibrator for the checkpoint
finnet calibrate --source synthetic --events 12000 \
  --model artifacts/risk_gnn_v2.pt \
  --method isotonic --output artifacts/calibrator.json

# Shadow deployment records decisions without enforcing them
finnet serve --port 8583 --metrics-port 9101 \
  --trust-client-subnet --calibrator artifacts/calibrator.json --shadow-mode
```

On the synthetic run, all classical baselines and the GNN saturate (AP ≈ 1.0). That is expected and should not be interpreted as proof of IEEE-CIS performance; it validates the harness and feature pipeline. Run `baseline-run --source ieee` with Kaggle data for a meaningful public comparison.

`artifacts/baseline_report.json` reports Average Precision, ROC-AUC, Brier score, log loss, expected-cost-at-0.5, top-1%/5%/10% review-budget recall, calibrated ECE/MCE, and calibrated AP for each model.

Synthetic benchmark from `artifacts/risk_gnn.report.json`:

| Metric | Value |
|---|---:|
| ROC-AUC | 1.000 |
| PR-AUC | 1.000 |
| Accuracy @ 0.5 | 0.998 |
| Precision @ 0.5 | 0.992 |
| Recall @ 0.5 | 1.000 |
| Train samples / test samples | 2,783 / 1,039 |
| Parameters | 23,937 |
| Train time (CPU) | 17.7 s |

The synthetic dataset is intentionally separable and is **not** evidence of IEEE-CIS performance. Replace this table with your IEEE-CIS run. The report includes confusion matrices and operating points.

### Latency

`finnet benchmark --iterations 2000` on the workspace CPU:

| Measurement | p50 | p95 | p99 |
|---|---:|---:|---:|
| Graph update + snapshot + GNN + policy-ready score | 0.906 ms | 1.339 ms | 1.721 ms |
| GNN forward only | 0.331 ms | 0.397 ms | 0.603 ms |

- 100% of iterations within the 20 ms budget; 0 fallbacks.
- Inline inference is enabled by default (`server.inline_inference: true`) because the model is small; this removes thread-pool queueing from the deadline.
- If inference exceeds `model.inference_budget_ms`, the scorer records `within_budget=false` and returns the deterministic heuristic fallback rather than violating the deadline.
- End-to-end socket latency under deliberate attack bursts is dominated by the 40+ ms token-bucket throttle penalty and by attack queueing; **model scoring** is the 20 ms SLA and remains sub-millisecond to low-millisecond.

---

## Closed-loop mitigation

| GNN score | Action | Network effect | ISO response |
|---|---|---|---|
| `< 0.40` | `ALLOW` | Forward/authorise | `00` |
| `0.40–0.60` | `ALLOW` with observation | normal response | `00` |
| `0.60–0.86` | `THROTTLE` | token bucket per subnet + delay | `96` |
| `0.86–0.96` | `QUARANTINE` | add CIDR to TTL quarantine; socket close | `05` |
| `>= 0.96` | `TERMINATE` | immediate socket teardown; firewall intent | none |

- **Token bucket:** capacity 20, refill 6 tokens/s per subnet. Exhaustion escalates to termination/quarantine.
- **Repeat-offender promotion:** repeated suspicious decisions inside a 60 s window can quarantine a subnet.
- **Quarantine TTL:** 300 s; the in-process table is authoritative.
- **Firewall:** default backend is `audit`, which writes JSONL intent and never touches the host firewall. Real `pf`/`iptables` mutation is opt-in. See [docs/FIREWALL.md](docs/FIREWALL.md).
- **Socket teardown:** TERMINATE closes the writer without an ISO response; QUARANTINE sends a decline then closes.
- **Fail policy:** engine errors fail open by default to preserve availability; configure `policy.fail_open_on_timeout: false` for fail-closed deployments.

---

## Project layout

```text
finnet-gnn/
├── configs/default.yaml
├── src/finnet/
│   ├── iso8583.py       # binary codec + framing + PAN tokenisation
│   ├── telemetry.py     # RTT/jitter/velocity/entropy flow table
│   ├── graph.py         # dynamic multipartite graph + snapshots
│   ├── features.py      # shared 24-feature schema
│   ├── model.py         # relational GraphSAGE
│   ├── scorer.py        # GNN inference, budget guard, explainability
│   ├── training.py      # temporal sampling, training, metrics
│   ├── policy.py        # allow/throttle/quarantine/terminate
│   ├── firewall.py      # audited OS firewall adapter
│   ├── engine.py        # parse → telemetry → graph → GNN → policy
│   ├── server.py        # asyncio/uvloop L4 gateway
│   ├── upstream.py      # optional ISO-8583 relay
│   ├── simulator.py     # legitimate + carding + botnet traffic
│   ├── metrics.py       # Prometheus exposition + health endpoint
│   ├── cli.py           # train/serve/demo/simulate/benchmark
│   └── data/
│       ├── synthetic.py
│       ├── ieee_cis.py
│       └── common.py
├── tests/               # 33 unit/integration tests
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PROJECT_REPORT.md
│   ├── FIREWALL.md
│   └── DATASET.md
├── Makefile
└── pyproject.toml
```

---

## Testing

```bash
pytest -q
```

Current suite covers:

- ISO-8583 primary + secondary bitmap roundtrip, framing, binary fields, LLVAR/LLLVAR, malformed frames, PAN tokenisation, source-subnet claims;
- telemetry entropy, jitter, burst velocity, bounded flow table, RTT EWMA;
- graph topology, relation types, fan-out, snapshots, blocked features, risk propagation;
- model batching, forward/backward, empty-edge graphs;
- policy bands, token exhaustion, repeat-offender promotion, quarantine TTL, firewall audit;
- engine + real socket integration through the async server;
- temporal split, synthetic attack composition, IEEE adapter, full training smoke test;
- metrics rendering and quantiles;
- calibration JSON round-trip, average precision/review-budget/cost metrics, and the classical-baseline benchmark harness with the upgraded risk-gated/Time2Vec model.

---

## Configuration

All settings are typed in `src/finnet/config.py` and validated by `load_config`; unknown YAML keys raise an error so a typo cannot silently weaken policy. Override with `configs/default.yaml` or CLI flags.

Important knobs:

```yaml
server:
  inline_inference: true      # no thread-pool queue for the 20 ms SLA
  trust_client_subnet: false  # never true in production
model:
  inference_budget_ms: 20
policy:
  allow_threshold: 0.40
  throttle_threshold: 0.60
  quarantine_threshold: 0.86
  terminate_threshold: 0.96
  token_capacity: 20
  token_refill_per_s: 6
  quarantine_ttl_s: 300
  enable_os_firewall: false
graph:
  snapshot_hops: 2
  max_snapshot_nodes: 128
  max_tx_nodes: 50000
```

---

## Honest scope

### What is genuinely implemented

- Binary ISO-8583 encoding/decoding and real TCP framing.
- Asyncio/uvloop L4 server with backpressure, idle timeout, max-frame validation, connection limits.
- Real online flow measurements: inter-arrival, jitter, burst velocity, byte entropy, and an RTT proxy.
- Dynamic multipartite graph with bounded memory and inductive subgraph export.
- PyTorch GNN training, evaluation, checkpoints, inference-budget enforcement, gradient explanation.
- Closed-loop policy with real socket closure and token-bucket rate limiting.
- Optional upstream relay and audited OS firewall adapter.

### What is simulated / derived

- **Synthetic dataset:** all transaction identities, attacks and timing. Useful for reproducible demos and unit tests, not a substitute for public benchmark evaluation.
- **IEEE-CIS network fields:** no IPs, RTT, packets, subnets or devices exist in the dataset. The adapter derives pseudonymous entities and pseudo-RTT. State this explicitly in your dissertation.
- **OS firewall:** default audit mode only. In-process quarantine is enforced by the gateway; host packet filtering is opt-in and requires root.
- **RTT:** application/client timestamp proxy when no upstream measurement is available. It is treated as untrusted and clamped.
- **Attack traffic:** the simulator generates card-testing, botnet and account-takeover patterns. It is not a real botnet.

### Not claimed

- Production-grade PCI-DSS authorisation, HSM integration, TLS, mTLS, key rotation, or regulatory compliance.
- Real-world fraud-rate superiority over commercial systems.
- True network RTT from PCAP without an upstream or kernel TCP_INFO integration.
- A complete ISO-8583 specification (only common card-authorisation fields are implemented).
- Security hardening of the host firewall.

---

## References

1. IEEE-CIS Fraud Detection, Kaggle.
2. Hamilton, Ying, Leskovec. *Inductive Representation Learning on Large Graphs* (GraphSAGE), NeurIPS 2017.
3. Schlichtkrull et al. *Modeling Relational Data with Graph Convolutional Networks* (R-GCN), ESWC 2018.
4. ISO 8583:1987 / ISO 8583:2003 financial transaction card originated messages.
5. PyTorch documentation, `torch.nn`, `index_add_`, `DataLoader`.
6. uvloop documentation.

---

## License / academic use

Prototype for academic coursework and research. Review dataset licences and institutional ethics requirements before use. Do not deploy against real cardholder data without a security review.
