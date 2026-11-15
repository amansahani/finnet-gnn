## Summary

<replace>

## Type of change

- [ ] Protocol / ISO parsing
- [ ] Gateway/networking
- [ ] GNN / benchmark
- [ ] Security / compliance
- [ ] Deployment / operations
- [ ] Documentation

## Verification

- [ ] `pytest -q`
- [ ] `ruff check src tests`
- [ ] `finnet validate --config configs/default.yaml`
- [ ] `docker build -t finnet-gnn .`

## Compliance

- [ ] No PAN, CVV, track data or PIN block added to logs, tests or docs.
- [ ] Production-safety defaults remain fail-safe.
- [ ] Operational consequences documented.
