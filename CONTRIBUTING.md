# Contributing

FinNet-GNN is an ISO-8583 risk-enforcement reference gateway. Changes should
preserve its core safety promise: deterministic protocol handling, explicit risk
decisions and no cardholder secrets in telemetry.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,benchmark]"
```

## Required checks before a pull request

```bash
pytest -q
ruff check src tests
finnet validate --config configs/default.yaml
docker build -t finnet-gnn .
```

## Guidelines

- Never log PANs, track data, CVVs or PIN blocks.
- Do not make defaults less safe for convenience.
- Keep lab defaults permissive; gate reference/production hardening explicitly.
- Add protocol tests for any ISO field change.
- Add benchmark results when changing model features or policy thresholds.
- Document operational consequences in `docs/RUNBOOK.md` and `docs/PRODUCT.md`.
- Maintain backward-compatible scoring only behind a documented model artifact
  migration path.

## Code review expectations

Reviewers check correctness before cleverness. Model improvements should be
measured, not asserted. Features that can only be demonstrated on synthetic
data must say so.
