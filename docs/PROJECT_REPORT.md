# FinNet-GNN: Autonomous ISO-8583 Financial Gateway with Real-Time Dynamic GNN Fraud Mitigation

**Project report scaffold — IIIT Sri City, M.Tech (IoT & Autonomous Systems)**

> **How to use this file.** This is a ready-to-edit dissertation/report skeleton.
> Sections marked **[FILL IN]** need your own text, IEEE-CIS numbers, or
> institutional details. Numbers under *Synthetic benchmark* are real outputs
> from this repository and are reproducible with the commands shown. Every claim
> is scoped as stated: synthetic results are not evidence of IEEE-CIS
> performance, and IEEE-CIS has no real network telemetry.

| Field | Value |
|---|---|
| Author | Aman Sahani |
| Programme | M.Tech, IoT & Autonomous Systems, IIIT Sri City |
| Course | **[FILL IN]** |
| Supervisor | **[FILL IN]** |
| Submission date | **[FILL IN]** |
| Repository | `iiit-mtech/IOT/finnet-gnn/` |
| Model checkpoint | `artifacts/risk_gnn.pt` (synthetic, reproducible) |

---

## Abstract

Card-not-present fraud is a real-time adversarial problem: attackers rotate
cards, devices and source networks faster than periodic batch models can adapt.
Traditional rule engines are interpretable but brittle, while tabular ML models
ignore the relational structure that exposes coordinated fraud. This project
presents **FinNet-GNN**, an autonomous financial gateway that terminates raw
ISO-8583 card traffic over TCP, extracts live network-flow telemetry, maintains
a dynamic multipartite graph of subnets, terminals, cards and merchants, scores
each transaction with an inductive relational GraphSAGE model, and closes the
loop by throttling, quarantining or terminating the offending traffic.

The system is implemented in Python with `asyncio`/`uvloop`, a binary ISO-8583
codec with primary/secondary bitmap decoding, a bounded in-memory graph, a
24-feature schema, and a PyTorch GNN with a strict 20 ms inference budget and a
deterministic fallback. On a reproducible synthetic benchmark the model reaches
ROC-AUC 1.000 and PR-AUC 1.000, and graph-export plus GNN scoring runs at
0.906 ms p50 / 1.339 ms p95 / 1.721 ms p99, with 100% of 2,000 iterations inside
the 20 ms budget. On a live socket demo, 120/120 legitimate transactions were
served without closure, while card-testing and botnet connections were closed
and 9 attack subnets were quarantined. An IEEE-CIS adapter is included for
large-scale benchmarking; because that dataset carries no network layer, derived
entities and a pseudo-RTT are used and reported as such.

**Keywords:** ISO-8583, fraud detection, graph neural networks, GraphSAGE,
inductive learning, asynchronous networking, autonomous mitigation, IEEE-CIS.

---

## 1. Introduction

### 1.1 Motivation

Payment authorisation is a latency-sensitive, adversarial control problem. A
gateway must return an approve/decline decision in milliseconds while an attacker
probes thousands of cards per minute. Fraud signals are rarely local to a single
transaction: they emerge from relationships — one terminal touching hundreds of
cards, one card appearing across unrelated merchants, one subnet emitting
machine-regular bursts.

### 1.2 Problem statement

Design and evaluate a gateway that:

1. speaks binary ISO-8583 over TCP and extracts real flow telemetry;
2. represents transactions relationally and scores them inductively within a
   20 ms window;
3. autonomously mitigates suspicious traffic without operator intervention.

### 1.3 Contributions

- A binary ISO-8583 codec with strict bitmap/field validation and 2-byte framing.
- A live flow-telemetry layer (inter-arrival, jitter, burst velocity, arrival
  rate, payload entropy, RTT proxy).
- A bounded dynamic multipartite graph with fan-out counters and inductive
  2-hop snapshot export.
- A relational GraphSAGE model with relation embeddings, edge weights, optional
  risk-gated message passing and transaction-time Time2Vec encoding.
- A reproducible classical-vs-GNN evaluation harness with LightGBM, XGBoost,
  CatBoost, average precision, review-budget recall, expected cost and ECE.
- A closed-loop policy engine with token buckets, TTL quarantine, socket
  teardown, optional isotonic/Platt/temperature calibration, shadow mode and an
  audited firewall adapter.
- A reproducible synthetic benchmark, a temporal training protocol, and an
  IEEE-CIS adapter with an explicit honesty boundary.

### 1.4 Report organisation

Section 2 reviews related work. Section 3 describes the architecture. Section 4
details implementation. Section 5 defines the experimental protocol. Section 6
reports results. Section 7 discusses findings. Section 8 lists limitations.
Section 9 concludes and outlines future work.

---

## 2. Background and related work

### 2.1 ISO-8583

ISO-8583 defines financial card-originated messages: a 4-digit message type
indicator (MTI), a bitmap marking present data elements, and field values encoded
as fixed, LLVAR or LLLVAR. Common card-authorisation traffic uses MTI `0100` or
`0200` and responses `0110`/`0210`. The bitmap is the contract: skipping an
unknown set bit desynchronises the entire remainder of the message, so this
implementation rejects unknown fields rather than guessing.

### 2.2 Graph neural networks

GraphSAGE [1] learns inductive node representations by aggregating sampled
neighbourhoods, allowing inference on unseen nodes. R-GCN [2] extends message
passing to multiple relation types. FinNet-GNN combines both ideas: a
relation-aware mean aggregator with learned per-relation embeddings, applied to a
local subgraph exported at transaction time. No per-entity embedding table is
used, which is essential when new cards and terminals appear continuously.

### 2.3 Fraud detection datasets

IEEE-CIS Fraud Detection [3] is a large public benchmark with anonymised card and
device fields. It is tabular; it contains no packets, IPs, RTT or true terminal
identifiers. Research that reports “network” features on IEEE-CIS must therefore
be explicit about derivation. This project keeps the public benchmark for model
evaluation and uses live/synthetic sockets for network claims.

### 2.4 Online mitigation

Rate limiting and quarantine are classical defence mechanisms. This project
couples them to model risk: the score selects an action band, a per-subnet token
bucket bounds sustained load, and a TTL quarantine plus socket teardown removes
confirmed attackers. Fail-open defaults protect availability; fail-closed is a
configuration choice.

---

## 3. System architecture

### 3.1 Overview

```
ISO-8583/TCP → codec → telemetry → dynamic graph → GNN → policy → response/teardown
```

The gateway is a single asyncio server. Each connection reads length-prefixed
frames, decodes ISO-8583, and calls the engine. The engine updates telemetry,
adds the transaction to the graph, exports a local snapshot, scores it, decides a
policy action, and returns an ISO response or closes the socket. Figure 1 of the
final document should be replaced with a rendered version of the diagram in
`docs/ARCHITECTURE.md`.

### 3.2 Protocol layer

- 2-byte big-endian length prefix.
- Maximum frame size 64 KiB; idle timeout 30 s; max 2,000 concurrent connections.
- Strict length/bitmap/field validation; malformed frames close the connection
  and increment `finnet_protocol_errors_total`.

### 3.3 Telemetry layer

Flows are keyed by `(subnet, device)` so card rotation cannot reset velocity.
Signals: inter-arrival time, jitter (EWMA), burst velocity (1 s), arrival rate
(5 s), payload entropy, RTT proxy (EWMA), flow age, count and mean bytes. The
flow table is bounded at 100,000 entries with oldest-10% eviction.

### 3.4 Graph layer

Five node types (subnet, device, card, merchant, transaction) and seven relation
types. Persistent entity edges form `subnet—device—card—merchant`; each
transaction is a hub connected to its four entities. Fan-out sets track distinct
counterparts per entity type — the direct signature of card testing and account
takeover. Transaction nodes expire after 900 s or 50,000 nodes; entities are
capped at 200,000.

### 3.5 Model layer

Two relational GraphSAGE layers, 64 hidden units, node-type embeddings, learned
relation embeddings, weighted mean aggregation and a residual connection. The
classifier reads only the target transaction node. Total: 23,937 parameters.

### 3.6 Policy layer

| Score | Action | Effect |
|---|---|---|
| `< 0.40` | ALLOW | authorise |
| `0.60–0.86` | THROTTLE | token bucket + 40 ms delay, code `96` |
| `0.86–0.96` | QUARANTINE | TTL 300 s, decline `05`, close |
| `>= 0.96` | TERMINATE | immediate close, quarantine |

Token bucket: capacity 20, refill 6/s per subnet. Four throttles in 60 s promote
to quarantine. Quarantine is in-process authoritative; the firewall adapter is
audit-only by default.

---

## 4. Implementation

### 4.1 Repository layout

See `README.md` for the full tree. Core modules: `iso8583.py`, `telemetry.py`,
`graph.py`, `features.py`, `model.py`, `scorer.py`, `training.py`, `policy.py`,
`firewall.py`, `engine.py`, `server.py`, `simulator.py`, `metrics.py`, `cli.py`.

### 4.2 Feature schema

24 features in three groups:

- entity (degree, tx count, recency, average amount, risk EWMA, distinct
  counterparts, blocked flag, tenure);
- flow/transaction (amount, hour sin/cos, inter-arrival, jitter, burst velocity,
  arrival rate, entropy, RTT);
- topology (new subnet/device/card/merchant, subnet-card fan-out,
  device-card fan-out, card-merchant fan-out).

### 4.3 Training

Time-ordered replay through the same graph and telemetry as serving. Stratified
retention of snapshots (fraud cap, normals at 3× fraud). `BCEWithLogitsLoss`
with positive weighting, AdamW, cosine schedule, gradient clipping. Offline risk
feedback uses a label-free heuristic prior so no future label leaks.

### 4.4 Latency engineering

- 128-node local subgraph, no full-graph forward pass.
- Inline inference by default; no thread-pool queue for a sub-millisecond model.
- Persistent line-buffered decision log; O(graph) gauges refreshed every 128
  messages.
- Hard budget: if inference exceeds 20 ms, return the heuristic and mark
  `within_budget=false`.

### 4.5 Safety and privacy

- PANs are HMAC-SHA256 tokenised before entering the graph.
- Untrusted client timestamps are clamped; private-field subnet claims are
  test-only.
- Firewall mutation is opt-in, audited and non-fatal.
- Engine failures default to fail-open; configurable.

---

## 5. Experimental protocol

### 5.1 Hardware and software

| Item | Value |
|---|---|
| Platform | macOS (darwin), CPU-only |
| Python | 3.12.2 |
| PyTorch | 2.9.0 |
| NumPy / pandas / scikit-learn | 2.5.0 / 2.3.3 / 1.7.0 |
| LightGBM / XGBoost / CatBoost | 4.6.0 / 3.3.0 / 1.2.10 |
| uvloop | 0.23.0 |

### 5.2 Synthetic benchmark

```bash
finnet train --source synthetic --events 12000 --epochs 12 \
  --risk-gate --temporal-encoding --time-dim 16 \
  --output artifacts/risk_gnn_v2.pt
finnet baseline-run --source synthetic --events 12000 \
  --model artifacts/risk_gnn_v2.pt --json-out artifacts/baseline_report.json
finnet calibrate --source synthetic --events 12000 \
  --model artifacts/risk_gnn_v2.pt --method isotonic --output artifacts/calibrator.json
finnet benchmark --model artifacts/risk_gnn_v2.pt --iterations 2000 --events 8000
finnet demo --no-train --events 300
```

### 5.3 IEEE-CIS benchmark **[FILL IN]**

```bash
finnet download-data --dest data
finnet train --source ieee \
  --transactions data/train_transaction.csv \
  --identity data/train_identity.csv \
  --max-events 150000 --epochs 12 \
  --output artifacts/risk_gnn_ieee.pt
```

Record: row cap, split date/index, sample counts, runtime, CPU/GPU, seed, and the
fact that network fields are derived.

### 5.4 Metrics

Primary: Average Precision/PR-AUC, Recall@1%/5%/10% review budgets, expected cost
under FP/FN costs, Brier score and expected calibration error. Secondary:
ROC-AUC, precision/recall/F1 at 0.5 and at policy bands, confusion matrix,
per-attack-type recall, policy operating points, and
latency/budget breaches. Reported results must include temporal and
entity-held-out splits once IEEE-CIS is available.

---

## 6. Results

### 6.1 Synthetic model quality

Reproduced from `artifacts/risk_gnn.report.json`:

| Metric | Value |
|---|---:|
| ROC-AUC | 1.000 |
| PR-AUC | 1.000 |
| Accuracy @ 0.5 | 0.9981 |
| Precision @ 0.5 | 0.9923 |
| Recall @ 0.5 | 1.000 |
| Train snapshots | 2,783 (698 fraud) |
| Test snapshots | 1,039 (257 fraud) |
| Parameters | 23,937 |
| Train time (CPU) | 17.7 s |

Per-attack-type mean score (test tail):

| Attack type | Samples | Mean score | Recall at quarantine |
|---|---:|---:|---:|
| Normal | 782 | 0.0032 | — |
| Card testing | 100 | 0.99995 | 1.000 |
| Botnet | 115 | 0.99987 | 1.000 |
| Account takeover | 42 | 0.99980 | 1.000 |

**Interpretation:** the synthetic generator is intentionally separable; near-perfect
scores confirm the pipeline learns the intended topology, not that it would
generalise to real traffic. This table must be labelled synthetic in the final
report.

### 6.1a Upgraded synthetic model: risk gate + Time2Vec

Reproduced from `artifacts/risk_gnn_v2.report.json`:

| Metric | Value |
|---|---:|
| ROC-AUC | 1.000 |
| PR-AUC / AP | 1.000 |
| Accuracy @ 0.5 | 0.9990 |
| Precision @ 0.5 | 0.9961 |
| Recall @ 0.5 | 1.000 |
| Train / test snapshots | 2,783 / 1,039 |
| Parameters | 25,313 |
| Train time (CPU) | 22.5 s |
| Graph + snapshot + GNN p50 / p95 / p99 | 1.014 / 1.493 / 2.572 ms |
| GNN forward only p50 / p95 / p99 | 0.439 / 0.542 / 1.254 ms |

The classical-vs-GNN benchmark on the same temporal snapshots also saturates
(AP ≈ 1.0 for histogram boosting, LightGBM, XGBoost, CatBoost and GNN), because
the synthetic generator is separable. This result establishes the evaluation
harness and should not be used as evidence of IEEE-CIS performance. See
`artifacts/baseline_report.json`.

### 6.2 Latency

`finnet benchmark --iterations 2000 --events 8000`:

| Stage | p50 | p95 | p99 |
|---|---:|---:|---:|
| Graph update + snapshot + GNN + score | 0.906 ms | 1.339 ms | 1.721 ms |
| GNN forward only | 0.331 ms | 0.397 ms | 0.603 ms |

- 100% of iterations within 20 ms; 0 budget fallbacks.
- RSS ≈ 329 MB after the benchmark.
- Graph at end: 3,048 nodes, 6,433 edges, 812 retained transactions.

### 6.3 Live socket demo

`finnet demo --no-train --events 300`:

| Profile | Sent | Responses | Closed |
|---|---:|---:|---:|
| Legitimate | 120 | 120 | 0 |
| Card testing | 8 | 3 | 5 |
| Botnet | 43 | 25 | 18 |

- Actions: 140 ALLOW, 4 THROTTLE, 4 QUARANTINE.
- Quarantined subnets: 9.
- Engine p50 1.14 ms, p95 1.80 ms, p99 2.72 ms.
- Simulator socket p95 is higher (tens of ms) because THROTTLE deliberately
  sleeps 40+ ms; the 20 ms SLA is a model-scoring budget.

### 6.4 IEEE-CIS results **[FILL IN]**

| Metric | Value |
|---|---:|
| Events used | **[FILL IN]** |
| Train / test snapshots | **[FILL IN]** |
| ROC-AUC | **[FILL IN]** |
| PR-AUC | **[FILL IN]** |
| Precision @ 0.5 | **[FILL IN]** |
| Recall @ 0.5 | **[FILL IN]** |
| Threshold @ ≤1% FPR | **[FILL IN]** |
| Train time | **[FILL IN]** |

### 6.5 Ablations **[FILL IN]**

Suggested: features only / no telemetry / no topology; relation ablation; snapshot
size 32/64/128/256; temporal vs random split.

---

## 7. Discussion

- **Inductive design is the key enabler.** Removing entity-ID embeddings lets new
  cards and terminals be scored on first sight, which matches how fraud appears.
- **Flow identity matters.** Keying flows by subnet/device rather than card
  preserves velocity evidence under card rotation.
- **Fan-out counters are high-precision.** Card testing and botnet reuse are
  visible in distinct-counterpart counts even when per-transaction amounts look
  benign.
- **Latency comes from scoping, not micro-optimisation.** A 128-node subgraph and
  a 24k-parameter model are fast because the problem is made small, not because
  of exotic kernels.
- **Autonomy needs a safety envelope.** Audit-by-default firewall, fail-open
  errors, TTL quarantine and explicit test-only flags keep the system from
  becoming the outage.
- **Synthetic scores are a sanity check.** They validate plumbing and learning
  capacity; they do not estimate real-world precision.

---

## 8. Limitations and threats to validity

1. Synthetic data is separable and overstates real performance.
2. IEEE-CIS lacks network telemetry; entities and RTT are derived.
3. Only common ISO-8583 fields are implemented; no TLS/HSM/PCI-DSS compliance.
4. RTT is an application proxy, not kernel `TCP_INFO`.
5. OS firewall integration is minimal and single-host; no shared state across
   replicas.
6. Thresholds are fixed, not cost-optimised per merchant or currency.
7. The demo authorises locally unless an upstream is configured.
8. CPU-only evaluation; GPU latency and batching are untested.
9. Initial constrained attack evaluation exists (`finnet attack-eval`), but it
   does not cover seeded graph poisoning, adaptive attackers, or multi-organisation
   collaboration; full robustness claims remain out-of-scope.
10. Simulator source subnets are claims via ISO field 61, not real network
    origins.

---

## 9. Conclusion and future work

FinNet-GNN demonstrates an end-to-end, reproducible path from raw ISO-8583 bytes
to an autonomous mitigation decision. The combination of strict protocol
parsing, live flow telemetry, an inductive relational graph model, and a
closed-loop policy meets a 20 ms scoring budget on commodity CPU while blocking
simulated carding and botnet traffic.

Future work:

- Complete the IEEE-CIS evaluation and report the derived-feature boundary.
- Add temporal drift monitoring and scheduled retraining.
- Replace fixed thresholds with cost-sensitive or conformal decision rules.
- Add shared quarantine state (Redis/control plane) for horizontal scaling.
- Integrate kernel RTT (`TCP_INFO`) and optional TLS/mTLS.
- Add nftables and cloud security-group backends.
- Evaluate adaptive attackers and evasion of graph features.
- Add analyst-facing explanation review and feedback capture.

---

## References

1. W. L. Hamilton, R. Ying, J. Leskovec. *Inductive Representation Learning on
   Large Graphs.* NeurIPS 2017.
2. M. Schlichtkrull et al. *Modeling Relational Data with Graph Convolutional
   Networks.* ESWC 2018.
3. IEEE-CIS Fraud Detection, Kaggle.
   <https://www.kaggle.com/c/ieee-fraud-detection>
4. ISO 8583:1987 / ISO 8583:2003, *Financial transaction card originated
   messages — Interchange message specifications.*
5. A. Paszke et al. *PyTorch: An Imperative Style, High-Performance Deep
   Learning Library.* NeurIPS 2019.
6. uvloop documentation. <https://uvloop.readthedocs.io/>

---

## Appendix A — Reproducibility checklist

- [ ] Python/Torch/NumPy versions recorded.
- [ ] Seeds recorded (`model.seed`, `data.synthetic_seed`).
- [ ] Exact CLI commands and config file archived.
- [ ] Checkpoint, report JSON and test-score NPZ archived.
- [ ] Synthetic vs IEEE-CIS tables clearly separated.
- [ ] Derived network fields labelled in every IEEE-CIS table.
- [ ] Latency measured on the same hardware reported.
- [ ] All figures regenerate from saved artefacts.

## Appendix B — Artefacts

| File | Contents |
|---|---|
| `artifacts/risk_gnn.pt` | Trained checkpoint + config + metrics |
| `artifacts/risk_gnn.report.json` | Full training/evaluation report |
| `artifacts/risk_gnn.test_scores.npz` | Held-out probabilities, labels, types |
| `artifacts/benchmark.json` | Latency benchmark output |
| `artifacts/sim_report.json` | Live simulator report |
| `artifacts/decisions.jsonl` | Per-transaction decision audit |
| `artifacts/firewall.jsonl` | Firewall intent audit |

## Appendix C — Glossary

| Term | Meaning |
|---|---|
| MTI | Message Type Indicator (4 digits) |
| PAN | Primary Account Number |
| STAN | System Trace Audit Number |
| RRN | Retrieval Reference Number |
| CIDR | Classless Inter-Domain Routing prefix |
| EWMA | Exponentially Weighted Moving Average |
| TTL | Time To Live |
| ROC-AUC | Area under receiver-operating-characteristic curve |
| PR-AUC | Area under precision-recall curve |
| FPR | False-positive rate |
