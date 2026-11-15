# Security Policy

FinNet-GNN is reference software, not a certified PCI/EMV security product.
Do not report a general “it cannot by itself make my payment system secure”
statement as a vulnerability; such concerns are explicitly documented here.

## In scope

- Parser desynchronisation or memory exhaustion bugs.
- Credential leakage in logs or errors.
- Collision or unsafe handling in the idempotency cache.
- Cryptographic misuse in `security.py`.
- Command injection via firewall adapter configuration.
- Denial of service through valid-frame resource consumption.
- Incorrect authorization bypass in the configured upstream adapter.

## Out of scope

- PCI-DSS, EMV, COTS or acquirer certification status.
- Fraud losses from an incorrectly configured decision threshold.
- Synthetic data or IEEE-CIS dataset licences.
- Security posture of an unspecified deployment environment.

## Reporting

If you believe you found a security vulnerability, report privately to the
maintainers through the repository security advisory workflow or email the
owner. Do not open a public issue for an exploitable vulnerability before a fix
is available.

Please include:

- affected version and configuration,
- frame/request samples with PAN data redacted,
- impact and preconditions,
- reproduction steps,
- suggested remediation if you can provide it.

## Response expectations

- Maintainers will acknowledge within five business days when contacted.
- Critical reports will be prioritised over feature work.
- Coordinated disclosure will be used by default.
