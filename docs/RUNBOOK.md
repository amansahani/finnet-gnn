# Production and Operations Runbook

## 1. Startup

```bash
docker build -t finnet-gnn .
docker run --rm finnet-gnn validate --config configs/default.yaml
docker run --rm -p 8583:8583 -p 9101:9101 finnet-gnn
```

Default profile is `lab`; it is intentionally not a production configuration.

## 2. Health and metrics

- Health: `http://127.0.0.1:9101/healthz`
- Prometheus metrics: `http://127.0.0.1:9101/metrics`
- Decision audit: `artifacts/decisions.jsonl`
- Firewall intent audit: `artifacts/firewall.jsonl`

## 3. What to monitor

| Signal | Expected condition | Triage |
|---|---|---|
| `finnet_active_connections` | bounded by config | check max-connections and source networks |
| `finnet_protocol_errors_total` | stable baseline | investigate malformed upstream clients |
| `finnet_errors_total{stage="engine"}` | near zero | stop rollout; inspect model/features |
| `finnet_inference_fallbacks_total` | zero | model/checkpoint/path issue |
| e2e latency p95/p99 | below 20 ms scoring budget, gateway separately | inspect CPU saturation |
| `finnet_idempotent_replays_total` | low normal retry volume | distinguish retries from replay attacks |
| quarantine count | controlled by cohort rules | inspect false positives |

## 4. Safe rollout

1. Run `finnet validate --config <path>` outside runtime.
2. Start with `server.profile=reference`.
3. Prove `/healthz`, `/metrics`, and one transaction round trip.
4. Enable `policy.shadow_mode=true`.
5. Verify projected actions for several business days.
6. Disable shadow mode only after operator review.
7. Roll out one noncritical cohort with automatic rollback.
8. Keep previous model and calibrator as a versioned rollback pair.

## 5. Incident playbooks

### Elevated fraud scores or false positives
1. Switch to shadow mode or allow-only policy.
2. Capture `decisions.jsonl`, model version and calibrator version.
3. Roll back to previous checkpoint.
4. Do not patch features while incident evidence is being collected.

### Parser errors
1. Preserve raw samples in a protected evidence bucket.
2. Identify MTI/bitmap/field failure.
3. If data may contain PANs, do not paste into chat/issue.
4. Patch and add a regression frame test.

### Upstream outage
1. Confirm circuit breaker behavior and timeout alarm.
2. Switch to documented fail-closed/fail-open policy.
3. Do not activate emergency override unless authorized.
4. Reconcile all idempotency keys after recovery.

### Suspected secret leak
1. Remove secret version from running pods.
2. Rotate MAC/TLS/upstream credentials.
3. Invalidate artifacts.
4. Purge search caches and local files.
5. File a security incident report.

## 6. Rollback

- Keep prior Docker image immutable by digest.
- Keep prior checkpoint and calibrator JSON.
- The metrics endpoint exposes model load errors; verify after rollback.
- Rollback is complete only when a live synthetic transaction scores with the
  intended old model and all connections return to normal.
