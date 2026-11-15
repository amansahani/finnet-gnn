# FinNet-GNN Product Scope and Release Levels

FinNet-GNN is an open-source ISO-8583 risk-enforcement reference gateway.
It is a middleware component, not a bank, switch, card scheme, or PCI-certified
product. It terminates an ISO-8583 message, scores fraud risk, and applies a
configurable response policy.

## What it is

- A high-throughput async TCP reference implementation.
- A fraud-risk instrument with an audited action ladder:
  `ALLOW`, `STEP_UP`, `REVIEW`, `THROTTLE`, `SOFT_DECLINE`, `QUARANTINE`,
  `TERMINATE`.
- A benchmark harness for temporal fraud models.
- A dependency-light, single-binary deployment suitable for experiments and
  certification-style lab integrations.

## What it is not

- A replacement for a card-network switch or licensed payment processor.
- A PCI-DSS, EMV or HSM-certified product.
- A production authorization service with real card-network connectivity out of
  the box.
- A lawful reason to process PAN, CVV or track data without legal review.
- A guarantee of fraud prevention, uptime or regulatory compliance.

## Deployment profiles

| Profile | Intent | Startup guard |
|---|---|---|
| `lab` | local demo and unit tests | permissive, no TLS/upstream requirement |
| `reference` | integration lab/upstream testing | TLS, mTLS/CA, MAC, replay, upstream |
| `production` | operator-managed deployment | `reference` requirements plus trained model and fail-closed policy |

`reference` and `production` configurations intentionally fail during startup
until every mandatory control is present. This is a feature, not a bug.

## Release levels

### v0.1-safe lab
- Simulator, reference ISO codec, baseline GNN, metrics and smoke demo.

### v0.2 production reference
- Docker packaging, mTLS/MAC hooks, upstream adapter, idempotency, runbooks and
  benchmark harness.

### v0.3 operational benchmark
- Reproducible IEEE-CIS benchmark, model artifact registry, signing, Helm chart
  and expanded fuzzing.

### v1.0 production candidate
- Independent security review, stable APIs and config, formal incident and
  rollback procedures, external deployment reports, long-running soak tests and
  documented compliance evidence.

Only the owner of the deployment may claim it is “production ready.” The software
must never be used to imply card-scheme certification or regulatory clearance.
