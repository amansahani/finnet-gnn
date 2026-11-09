# FinNet-GNN Architecture

This document describes the runtime design of the gateway: the wire protocol, the
online telemetry pipeline, the dynamic graph, the inductive GNN, the mitigation
loop, and the concurrency/latency decisions that let scoring fit inside a 20 ms
budget.

All file references are relative to `src/finnet/`.

---

## 1. Design goals and constraints

| Goal | Constraint | Design response |
|---|---|---|
| Parse real card traffic | ISO-8583 is binary and bitmap-driven | Explicit codec with primary/secondary bitmap decoding, no lenient skipping |
| Score fraud in real time | Hard 20 ms inference window | 128-node local subgraph, 2-layer GNN, inline inference, budget guard |
| Detect unseen entities | New cards/terminals appear constantly | Inductive subgraph export; no entity-ID embedding table |
| Resist card rotation | Fraud rotates cards to evade per-card rules | Flow keyed by `(subnet, device)`, fan-out counters across cards |
| Bound memory | Traffic is unbounded | TTL and capacity limits on flows, transaction nodes, entities, fan-out sets |
| Stay available | A model or firewall error must not kill the gateway | Fail-open engine policy, heuristic fallback scorer, audited firewall adapter |
| Preserve privacy | Raw PANs must never reach the graph | Keyed HMAC-SHA256 PAN tokens |

Non-goals are listed in the top-level `README.md` (production PCI-DSS
authorisation, HSM/TLS integration, full ISO-8583 coverage).

---

## 2. Component map

```
                    ┌──────────────────────────────┐
   TCP bytes ──────►│ server.GatewayServer         │
                    │  asyncio / uvloop            │
                    │  2-byte frame length         │
                    └──────────────┬───────────────┘
                                   │ ISO8583Message
                    ┌──────────────▼───────────────┐
                    │ iso8583.ISO8583Codec         │
                    │  MTI, bitmap, field schema   │
                    │  PAN tokenisation            │
                    └──────────────┬───────────────┘
                                   │ decoded message
                    ┌──────────────▼───────────────┐
                    │ engine.GatewayEngine         │
                    │  orchestration + metrics     │
                    └───┬───────────┬──────────┬───┘
                        │           │          │
          ┌─────────────▼──┐  ┌─────▼──────┐  ┌▼─────────────┐
          │ telemetry      │  │ graph      │  │ scorer       │
          │ FlowTelemetry  │  │ Dynamic    │  │ RiskScorer   │
          │ RTT/jitter/    │  │ Transaction│  │ GNN + budget │
          │ burst/entropy  │  │ Graph      │  │ + fallback   │
          └────────────────┘  └─────┬──────┘  └──────┬───────┘
                                    │ GraphSnapshot  │ score
                                    └────────┬───────┘
                                   ┌─────────▼─────────┐
                                   │ policy.PolicyEngine│
                                   │ allow/throttle/    │
                                   │ quarantine/terminate│
                                   └─────────┬─────────┘
                                             │ intent
                                   ┌─────────▼─────────┐
                                   │ firewall adapter   │
                                   │ audit | pf | iptables│
                                   └───────────────────┘
```

Supporting modules:

- `config.py` — frozen dataclasses, strict YAML loading, unknown-key rejection.
- `features.py` — the single 24-feature schema shared by graph, model, trainer and explainer.
- `metrics.py` — dependency-free Prometheus exposition and `/healthz`.
- `upstream.py` — optional ISO-8583 relay for allowed traffic.
- `data/` — synthetic generator and IEEE-CIS adapter producing the same `TransactionEvent`.

---

## 3. Wire protocol and ISO-8583 codec

### 3.1 Framing

Every message on the TCP stream is prefixed with a 2-byte big-endian unsigned
length (`struct >H`). The server enforces:

- `length > 0`
- `length <= server.max_frame_bytes` (default 64 KiB)
- the body must arrive within `server.idle_timeout_s`

The codec also supports `encode_frame` / `decode_frame` so tests and the
simulator exercise the exact framing used on the wire.

### 3.2 Message model

`ISO8583Message` stores:

- `mti` — 4 ASCII digits
- `fields` — `dict[int, str | bytes]`
- `raw` — original payload
- `received_at`

Convenience properties expose `amount`, `amount_minor`, `masked_pan`,
`terminal_id` (field 41), `merchant_id` (field 42), `stan` (field 11),
`response_code` (field 39), `currency` (field 49), and the untrusted client
timestamp from private field 62.

### 3.3 Decoding

1. Require at least 12 bytes (4-byte MTI + 8-byte primary bitmap).
2. Validate the MTI is four digits.
3. Read the primary bitmap. Bit 1 is the secondary-bitmap indicator.
4. When present, read the secondary bitmap (fields 65–128).
5. Iterate fields 2..64 (or 2..128), and for each set bit:
   - look up `FIELD_SPECS`; an unknown field raises `ISO8583Error` rather than
     silently desynchronising the rest of the message;
   - read fixed, LLVAR (2-digit length) or LLLVAR (3-digit length) data;
   - binary fields (e.g. 52 PIN block, 64 MAC) are returned as `bytes`.
6. Reject trailing bytes: `offset != len(payload)` is an error. This turns a
   malformed length prefix into a loud failure instead of data corruption.

Supported fields cover common card-authorisation traffic (PAN, processing code,
amount, transmission time, STAN, MCC, POS entry mode, acquirer ID, track 2, RRN,
response code, terminal ID, merchant ID/name, currency, EMV data, private fields
60–63, MACs, and more). See `FIELD_SPECS` in `iso8583.py`.

### 3.4 Encoding

`encode` validates:

- MTI is four digits;
- field 1 is never supplied (it is the secondary-bitmap indicator);
- every field exists in the schema;
- fixed fields have exactly the schema length;
- variable fields do not exceed the schema maximum;
- binary fields receive `bytes`, text fields receive `str` (ASCII only).

The secondary bitmap is set automatically when any field > 64 is present.

### 3.5 Responses

`make_response` maps request MTI to response MTI (`0200 → 0210`, `0100 → 0110`,
etc.), copies a safe subset of request fields (`_COPY_TO_RESPONSE`), sets field
39 to the response code, and writes the decision into field 60 as:

```
action=<ALLOW|THROTTLE|QUARANTINE>;score=<0..1>;reason=<text>
```

TERMINATE responses are not sent; the socket is closed.

### 3.6 Privacy

`tokenize_pan(pan, salt)` returns `card_<20 hex chars>` from HMAC-SHA256. The raw
PAN exists only inside the request object for the duration of processing. Graph
keys, logs and checkpoints contain tokens only. The response still echoes the
PAN because an acquirer protocol requires it; this is an ISO-8583 protocol
property, not a graph feature.

### 3.7 Untrusted metadata

Two private fields are treated as untrusted claims:

- Field 62 may carry `ts_ms=<client milliseconds>` for an RTT proxy.
  `likely_rtt_ms` accepts it only if `0 <= now - ts <= 60 s`.
- Field 61 may carry `subnet=<CIDR>;src_ip=<IP>` for the simulator.
  `client_network_claim` validates both values, but the engine uses them **only**
  when `server.trust_client_subnet` is true. Production uses the real TCP peer
  address from `writer.get_extra_info("peername")`.

---

## 4. Online flow telemetry

`FlowTelemetry` keeps a bounded table keyed by `event.flow_key =
(subnet, device_id)`. Card and merchant are transaction attributes, not flow
identifiers: a carding bot that rotates cards still accumulates velocity on its
device and subnet.

Per observation the flow state updates:

| Signal | Definition |
|---|---|
| `inter_arrival_ms` | time since the previous message on this flow |
| `jitter_ms` | EWMA of absolute change in inter-arrival time |
| `burst_velocity` | messages in the last `velocity_window_s` (default 1 s) |
| `arrival_rate` | messages in `rate_window_s` divided by that window (default 5 s) |
| `payload_entropy` | Shannon entropy of this message in bits/byte (0–8) |
| `rtt_ms` | EWMA of a valid application RTT proxy |
| `flow_age_s` | time since the flow was first seen |
| `flow_count` | messages observed on this flow |
| `bytes_mean` | running mean payload size |

Jitter and RTT use independent EWMA coefficients (`jitter_alpha`, `rtt_alpha`,
default 0.2). Arrival timestamps are held in a bounded deque (`maxlen=512`).
When the flow table exceeds `max_flows` (100k), the oldest 10% of flows are
evicted.

Entropy is computed per message from the raw payload byte histogram. It is a
property of the current message, not accumulated state, and is attached with
`dataclasses.replace`.

The telemetry table is guarded by an `RLock` because the gateway may run
inference inline on the event loop or in a worker thread.

---

## 5. Dynamic multipartite graph

### 5.1 Node and relation types

```
NodeType        Relation
0 SUBNET        0 SUBNET_DEVICE
1 DEVICE        1 DEVICE_CARD
2 CARD          2 CARD_MERCHANT
3 MERCHANT      3 TX_SUBNET
4 TRANSACTION   4 TX_DEVICE
                5 TX_CARD
                6 TX_MERCHANT
```

A transaction creates or touches four persistent entity nodes (subnet, device,
card, merchant) and one transient transaction node. Entity links are layered
(`subnet—device—card—merchant`) and the transaction node is a hub connected to
all four. Edges are stored in both directions for message passing and carry a
repeat count. `EdgeState.weight = min(1, 0.35 + 0.2·ln(1 + count))` dampens
one-off links while preserving repeated relationships.

### 5.2 Fan-out counters

Each entity keeps a `counterparts: dict[NodeType, set[int]]`. This is the key
fraud signal that simple rate limits miss:

- one subnet/device touching many cards → card testing;
- one card touching many merchants/devices → account takeover or botnet reuse.

Counters are capped at `fanout_cap` (4096) per `(entity, type)` pair, which also
bounds memory for a deliberate fan-out flood.

### 5.3 Memory bounding

- Transaction nodes are queued in insertion order and pruned when older than
  `tx_ttl_s` (900 s) or when the queue exceeds `max_tx_nodes` (50k).
- Entity growth is checked every 256 additions; if entities exceed
  `max_entities` (200k), the least-recently-seen 5% (at least the excess) are
  removed.
- Removal drops the node's index entry and adjacency links; counterpart sets
  are cleaned for entity eviction.

### 5.4 Snapshot export (inductive)

For a target transaction node, `snapshot()`:

1. Runs a breadth-first search up to `snapshot_hops` (2), visiting neighbours in
   descending `last_seen` order.
2. Stops at `max_snapshot_nodes` (128).
3. Re-indexes the selected nodes 0..N-1 and builds:
   - `x` — `(N, 24)` float32 feature matrix;
   - `node_types` — `(N,)` int64;
   - `edge_index` — `(2, E)` int64;
   - `edge_type` — `(E,)` int64;
   - `edge_weight` — `(E,)` float32;
   - `target_index` — local index of the transaction;
   - `node_ids` — original graph IDs (diagnostics only);
   - `timestamp`.

The model never receives original node IDs, so a never-before-seen entity is
scored inductively from its features and local topology. Snapshots are detached
NumPy arrays, safe to hand to a worker thread or batch collator.

### 5.5 Risk feedback

After a score is produced, `update_risk(tx_node_id, score)` applies an EWMA
(`risk_ewma_alpha = 0.25`) to the transaction and its direct entity neighbours.
This becomes a feature on **future** snapshots. The value fed back online is a
label-free topology/telemetry prior (`heuristic_score`), never the online GNN
output and never a label, so the online graph state matches the offline training
replay. The policy itself is driven only by the GNN score.

---

## 6. Feature schema (24 dimensions)

All features are shared through `features.NODE_FEATURE_NAMES`.

### Entity features (0–7)

| Index | Name | Meaning |
|---:|---|---|
| 0 | `degree_log` | `log1p` of current graph degree |
| 1 | `tx_count_log` | `log1p` of lifetime transactions on the entity |
| 2 | `recency_s_log` | `log1p` seconds since the entity was last seen |
| 3 | `avg_amount_log` | `log1p` of mean transaction amount |
| 4 | `risk_ewma` | propagated risk in [0, 1] |
| 5 | `distinct_counterparts_log` | `log1p` of distinct counterpart entities |
| 6 | `blocked` | 1 if the entity is inside an active block window |
| 7 | `tenure_s_log` | `log1p` seconds since first seen |

### Flow and transaction features (8–15)

| Index | Name | Meaning |
|---:|---|---|
| 8 | `amount_log` | `log1p` transaction amount |
| 9 | `hour_sin` | cyclical hour-of-day sine |
| 10 | `hour_cos` | cyclical hour-of-day cosine |
| 11 | `interarrival_ms_log` | `log1p` milliseconds since previous flow message |
| 12 | `jitter_ms_log` | `log1p` inter-arrival jitter |
| 13 | `burst_velocity_log` | `log1p` messages in the velocity window |
| 14 | `arrival_rate_log` | `log1p` messages per second |
| 15 | `payload_entropy` | byte entropy / 8, in [0, 1] |

### Context and topology features (16–23)

| Index | Name | Meaning |
|---:|---|---|
| 16 | `rtt_ms_log` | `log1p` RTT proxy |
| 17 | `new_subnet` | first time this subnet is seen |
| 18 | `new_device` | first time this device is seen |
| 19 | `new_card` | first time this card token is seen |
| 20 | `new_merchant` | first time this merchant is seen |
| 21 | `subnet_card_fanout_log` | `log1p` distinct cards on the subnet |
| 22 | `device_card_fanout_log` | `log1p` distinct cards on the device |
| 23 | `card_merchant_fanout_log` | `log1p` distinct merchants on the card |

Transaction nodes use the transaction variant of the feature builder; entity
nodes use the entity variant. Numeric inputs are passed through `log1p_safe`,
which maps non-finite or non-positive values to 0.0 and caps very large values.

---

## 7. Model

`model.RiskGNN` is a two-layer inductive relational GraphSAGE network written in
plain PyTorch (no PyTorch Geometric dependency).

### 7.1 Input projection

```
h_i^(0) = Dropout(ReLU(LayerNorm(W_in · x_i + t_{type(i)})))
```

- `W_in: 24 → 64`
- `t` is an embedding for the five node types
- LayerNorm stabilises heterogeneous feature scales
- Dropout defaults to 0.15

### 7.2 Relational message passing

For each layer ℓ and node *i*:

```
m_{j→i} = [W_neigh · h_j^(ℓ) + e_{rel(i,j)}] · w_ij · σ(W_risk · risk_j)
a_i     = (1 / |N(i)|) · Σ_{j∈N(i)} m_{j→i}
h_i^(ℓ+1) = Dropout(ReLU(LayerNorm(W_self · h_i^(ℓ) + a_i + b)))
```

- `e_r` is a learned embedding per relation (7 relations)
- `w_ij` is the repeat-count edge weight, clamped to `[0.05, 1.0]`
- `risk_j` is the source node scalar `risk_ewma`; the risk gate is learnable and disabled by default for backward compatibility
- if `temporal_encoding` is enabled, a learned Time2Vec projection of the transaction time phase is added after input projection
- aggregation is a weighted mean via `index_add_`, implemented for both
  directions of each undirected edge
- a residual connection (`h + layer(h)`) improves optimisation of the 2-layer stack
- an empty edge set degrades gracefully to zeros

### 7.3 Classification head

```
z = Linear(ReLU(Linear(h_target)))    # 64 → 64 → 1
p = sigmoid(z)
```

The head reads only the target transaction node's embedding, so one batched
forward pass can score many snapshots with disjoint node index offsets.

### 7.4 Why inductive

The model has no per-entity embedding table. Identity is resolved to features and
local relations at snapshot time. This matters because fraud detection sees new
cards and terminals continuously; a transductive model would fail until
retraining.

### 7.5 Parameter count

The default configuration has **23,937** trainable parameters; enabling the
optional risk gate and Time2Vec in `--risk-gate --temporal-encoding` yields
25,313 parameters. The model remains small enough for sub-millisecond to
low-millisecond CPU inference.

---

## 8. Scoring and the 20 ms budget

`scorer.RiskScorer` wraps model loading, inference and failure handling.

### 8.1 Loading

- If the checkpoint path exists, load it with `torch.load`.
- Validate the checkpoint feature dimension against the runtime schema.
- Warm up with three dummy forwards so lazy allocation does not hit the first
  live request.
- Any load error leaves `model = None` and records `load_error`; the gateway
  still starts.

### 8.2 Inference

1. Start a `perf_counter` timer.
2. Convert the snapshot arrays to tensors on the configured device.
3. Run the GNN under `torch.no_grad()`.
4. If elapsed time exceeds `model.inference_budget_ms` (20 ms), return the
   heuristic score with `source="heuristic_timeout"` and `within_budget=false`.
5. If inference raises, return the heuristic with `source="heuristic_error"`.
6. Otherwise return `source="gnn"` and `within_budget=true`.

The heuristic is a transparent topology/telemetry prior using burst velocity,
arrival rate, new-entity flags, fan-out, amount and entropy. It is never labelled
as a GNN result.

### 8.3 Explanation

`explain()` differentiates the target logit with respect to the input feature
matrix and reports `gradient × value` for the top-k features. This is a local
linear attribution, useful for analyst review and dissertation examples. It is
computed off the decision path.

### 8.4 Calibration and shadow mode

`finnet calibrate` fits an isotonic, Platt or temperature calibrator on a train
split and serializes it as JSON. A gateway started with
`--calibrator artifacts/calibrator.json` applies that mapping before policy
scoring, leaving raw and calibrated probabilities in the decision log. A
temperature is computed from logit scaling; isotonic and Platt are loaded from a
JSON parameter document rather than executable pickle data.

`policy.shadow_mode` records the action the GNN would have chosen but forces an
allow/serve response. The original action is preserved in `shadow_action` and the
audit log so operators can quantify projected enforcement before enabling it.
Shadow mode does not quarantine or mutate the graph block window.

---

## 9. Policy and closed-loop mitigation

`policy.PolicyEngine` evaluates the GNN score in descending severity.

| Score band | Action | Socket | ISO response | Side effects |
|---|---|---|---|---|
| `< 0.40` | `ALLOW` | keep open | `00` | none |
| `0.60 – 0.86` | `THROTTLE` | keep open | `96` | consume a token; add 40 ms + burst-based delay |
| `0.86 – 0.96` | `QUARANTINE` | send then close | `05` | add CIDR to TTL quarantine; firewall intent |
| `>= 0.96` | `TERMINATE` | close immediately | none | quarantine + firewall intent |

Additional rules:

- **Precheck:** every new connection and every message checks the subnet against
  the quarantine table. A quarantined subnet is terminated before parsing.
- **Token bucket:** one bucket per subnet, capacity 20, refill 6 tokens/s.
  Capacity/refill are independent of the attack score; exhaustion escalates a
  throttled subnet to termination + quarantine.
- **Repeat offender:** repeated throttles inside `violation_window_s` (60 s);
  reaching `violations_to_quarantine` (4) promotes the subnet to quarantine.
- **Quarantine TTL:** 300 s. The in-process store is authoritative; expiry is
  lazy on read.
- **Graph feedback:** a blocked subnet is marked in the graph for the TTL so the
  `blocked` node feature reflects it.

### 9.1 Firewall adapter

`firewall.FirewallAdapter` supports three backends:

- `audit` (default): append intent JSON to `artifacts/firewall.jsonl`, never touch
  the host.
- `pf` (macOS): `pfctl -t finnet_quarantine -T add <cidr>`.
- `iptables` (Linux): `iptables -I INPUT -s <cidr> -j DROP`.

OS mutation only happens when `policy.enable_os_firewall` is true. The adapter
times out commands (`firewall_command_timeout_s`, 3 s), checks that the binary
exists, records success/failure, and never raises into the gateway. See
[FIREWALL.md](FIREWALL.md) for operational setup and safety.

### 9.2 Failure policy

Engine-level exceptions produce a fail-open ALLOW when
`policy.fail_open_on_timeout` is true (default), preserving availability; set it
to false for fail-closed behaviour. The server also enforces a per-message
`request_timeout_s` (1 s) and counts deadlines as protocol/engine errors.

---

## 10. Concurrency and latency design

### 10.1 Event loop

`GatewayServer` is a single `asyncio` server (optionally on `uvloop`). Per
connection:

1. Acquire a `Semaphore` slot (`max_connections`, 2000).
2. Check the peer subnet against quarantine.
3. Loop: read 2-byte header, read body, decode, process, write response.

The connection loop is sequential per socket, which is correct for ISO-8583
request/response and keeps ordering deterministic.

### 10.2 Inline inference

`server.inline_inference` defaults to `true`. The GNN forward pass for a 128-node
subgraph is hundreds of microseconds on CPU, so running it inline avoids:

- thread-pool queueing latency,
- GIL hand-off jitter,
- unbounded executor backlog under attack,
- per-request future allocation.

`worker_threads` and the `ThreadPoolExecutor` remain available for larger models
or accelerators; setting `inline_inference: false` restores worker dispatch.

### 10.3 Why latency stays low

- **Bounded subgraph:** at most 128 nodes and a few hundred edges.
- **Small model:** 23.9k parameters, 24-dim input, 64-dim hidden state.
- **No ID lookup:** snapshots are pure tensors; no sparse embedding gather.
- **Warm tensors:** the model is warmed at startup.
- **Cheap features:** `log1p`, EWMA and set counters; no external calls.
- **Hot-path logging:** decision logs use a persistent line-buffered handle;
  O(graph) metric gauges are refreshed every 128 messages, not per message.
- **Separate concerns:** explanation and report generation are off the hot path.

Measured on the workspace CPU (`finnet benchmark --iterations 2000`):

| Stage | p50 | p95 | p99 |
|---|---:|---:|---:|
| Graph update + snapshot + GNN + score | 0.856 ms | 1.250 ms | 1.416 ms |
| GNN forward only | 0.292 ms | 0.343 ms | 0.390 ms |

All 2000 iterations completed inside 20 ms with zero fallbacks.

Note that end-to-end **socket** latency in the simulator can exceed 20 ms because
the THROTTLE policy intentionally sleeps 40+ ms and attack workers queue behind
each other. The 20 ms SLA applies to model scoring, not to a deliberately
delayed throttled response.

### 10.4 Ordered request/response flow

```
client                      server                         engine
  |-- len + 0200 ------------>|                              |
  |                           |-- decode ------------------>|
  |                           |-- precheck subnet --------->|
  |                           |-- process ----------------->|
  |                           |     telemetry.update        |
  |                           |     graph.add_transaction   |
  |                           |     graph.snapshot          |
  |                           |     scorer.predict (GNN)    |
  |                           |     policy.decide           |
  |                           |     graph.update_risk       |
  |                           |<-- EngineResult ------------|
  |<-- len + 0210 ------------|                              |
  |    (or close)             |                              |
```

---

## 11. Observability

`metrics.Metrics` is thread-safe and supports counters, gauges, and histograms
with labels. The metrics server exposes:

- `GET /metrics` — Prometheus text format, including
  `finnet_messages_total`, `finnet_actions_total{action}`,
  `finnet_errors_total{stage}`, `finnet_inference_fallbacks_total{source}`,
  `finnet_e2e_latency_ms`, `finnet_inference_latency_ms`, `finnet_risk_score`,
  graph gauges, active connections and uptime;
- `GET /healthz` (or `/health`) — JSON status with engine summary.

`artifacts/decisions.jsonl` is an append-only decision log with event IDs,
tokenised entities, score, source, action, latency, budget flag and error.
`artifacts/firewall.jsonl` records firewall intent.

---

## 12. Training pipeline

`training.py`:

1. Load events (synthetic or IEEE-CIS) and sort by timestamp.
2. Split by time (`train_fraction`, default 0.8); if one side has no fraud, move
   the boundary to the nearest fraud without reversing time.
3. Replay training events through the same graph and telemetry code used online.
4. Retain a stratified sample: fraud up to the cap, normals at
   `negative_ratio` (default 3.0) times the retained fraud count.
5. Train with `BCEWithLogitsLoss` and positive class weighting
   (`min(20, negative/positive)`), AdamW, cosine schedule, gradient clipping.
6. Evaluate on the held-out tail with ROC-AUC, PR-AUC, precision/recall/F1 at
   0.5, policy operating points, per-attack-type recall, and the best threshold
   at ≤ 1% false-positive rate.
7. Save checkpoint, JSON report, and compressed test scores.

Offline risk feedback uses `scorer.heuristic_score`, which contains no labels and
no model output. This keeps the graph feature distribution consistent between
training and serving.

---

## 13. Determinism and reproducibility

- Synthetic generation uses a seeded `random.Random`.
- Training seeds Python, NumPy, and PyTorch.
- Event IDs and sorted timestamps make replay order deterministic.
- Checkpoints include model config, feature names, relation names, source and
  metrics.
- The simulator seeds workload construction and per-worker jitter.

GPU kernels can be non-deterministic; the reference configuration is CPU
(`model.device: cpu`) and one Torch thread for predictable latency.

---

## 14. Known limitations

- Only common card-authorisation ISO-8583 fields are implemented.
- The demo uses an in-process stand-in authorisation unless an upstream is
  configured; this is not a switch/acquirer connection.
- IEEE-CIS has no network layer; derived entities and pseudo-RTT are synthetic.
- RTT is an application-layer proxy, not kernel `TCP_INFO`.
- The OS firewall adapter is minimal and not a hardened packet-filter manager.
- Token buckets and quarantine are per-process; horizontal scaling would need a
  shared store (e.g. Redis) and a control plane.
- Attack simulation is local and synthetic, not a real botnet.
