# Cascade architecture, TLS and idempotency

FinNet-GNN provides three implementation layers for online decisions.

## 1. Cascade routing

`src/finnet/risk_cascade.py` wraps two `RiskScorer`s behind the interface used by
`GatewayEngine`:

1. **Fast tier** — the checkpoint configured through `model.checkpoint`.
2. **Nearline tier** — the optional checkpoint configured through
   `model.cascade_checkpoint` or `finnet serve --cascade-model PATH`.

Routing is deliberately asymmetric:

- If the fast score is confident (`abs(score - 0.5) > 0.20`), it decides.
- If it is uncertain, the nearline model is called and their probabilities are
  fused `60% nearline / 40% fast`.
- If the nearline model cannot satisfy the budget or is unavailable, the fast
  decision is used and the degradation is logged through `score_source`.

This is not a distributed service mesh. Both tiers remain PyTorch modules in the
same process so the 20 ms budget and deployment footprint remain simple.

## 2. TLS/mTLS termination

`server.tls_enabled` now starts the ISO listener through `asyncio.start_server`
with a TLS context. Production deployment must enable it. Optional
`require_client_certificate=true` requires a client certificate whose issuer
chain is validated against `tls_ca_file`.

The HTTP metrics listener is intentionally not wrapped in the same TLS context.
Bind it to loopback or a private operations network and protect it with a proxy
or network policy before exposing it outside the process host.

Operational notes:

- TLS is not a substitute for ISO message MAC validation.
- Certificate and private-key files should be readable only by the service user.
- Certificate rotation requires reloading/restarting the gateway in the current
  implementation.

## 3. MAC hook

`server.mac_required=true` validates an 8-byte FinNet sidecar HMAC trailer over
the complete wire payload. `server.mac_secret` or `server.mac_secret_file`
configures the signing material.

This is an implementation hook, not an Acquirer/EMV-compliant MAC layout:

- real EMV/AES-CMAC/KSN/KSU negotiation is out of scope;
- the trailer layout belongs to the deployment integration contract;
- a card network may therefore reject messages that a happy-path FinNet test
  accepts.

The MAC runs before ISO decoding so malicious framing fails `ISO8583Error`-class
validation without reaching the GNN.

## 4. Replay/idempotency protection

`server.replay_protection` fingerprints `(STAN, RRN, merchant ID, terminal ID,
amount, currency)` with SHA-256 and remembers the resulting decision for
`server.idempotency_ttl_s` seconds. A repeated message returns the cached
response instead of entering the graph a second time.

The fingerprint intentionally excludes PAN, so hashed cache entries cannot
enumerate card numbers. It is an in-process cache; for multi-replica deployments
it must move to a shared store before the setting can be considered end-to-end.

`finnet_idempotent_replays_total` increments whenever this path is used.

## 5. Fraud-ring communities

`DynamicTransactionGraph` maintains an incremental union-find over active entity
nodes. Community IDs are copied to every snapshot and used by benchmark baselines
to build community-scale features. GNN online features remain compatible at 24
dimensions; community IDs are not fed directly to the GNN.

The component maintenance is approximate under deletion: it decrements the node
from an active-size counter rather than recomputing a split. The graph would
require a periodic full pass to become exact.
