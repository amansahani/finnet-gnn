# Firewall Quarantine Operations

FinNet-GNN has two layers of blocking:

1. **In-process quarantine** (always on) — the gateway refuses and terminates
   traffic from a quarantined CIDR before parsing. This is authoritative for
   connections that reach this process.
2. **OS firewall intent** (opt-in) — a `pf` or `iptables` rule that also drops
   packets before they reach the gateway process.

The default is **audit-only**. Nothing in this project modifies the host
firewall unless you explicitly enable it. This is a deliberate safety decision:
an over-broad `/24` drop on a developer laptop can disconnect your own session,
your SSH access, or a shared CI runner.

---

## 1. How the adapter behaves

`src/finnet/firewall.py` implements `FirewallAdapter`.

| Setting | Effect |
|---|---|
| `policy.firewall_backend: audit` | Append intent JSON to `artifacts/firewall.jsonl`; never touch the host |
| `policy.enable_os_firewall: false` | Force audit behaviour even if a real backend is selected |
| `policy.enable_os_firewall: true` + `firewall_backend: pf` | Call `pfctl` |
| `policy.enable_os_firewall: true` + `firewall_backend: iptables` | Call `iptables` |

Every block/unblock writes one JSON line:

```json
{"backend": "audit", "cidr": "203.0.113.0/24", "ok": true, "operation": "block", "reason": "risk >= quarantine threshold", "score": 0.9123, "ts": 1760000000.0, "ttl_s": 300.0}
```

The adapter:

- validates the backend name at construction;
- checks `shutil.which` before invoking a binary;
- enforces `firewall_command_timeout_s` (default 3 s);
- records stderr/stdout on failure;
- never raises into the request path;
- tracks the set of CIDRs it has blocked.

The in-process quarantine store owns the TTL (default 300 s). The OS adapter
currently adds and removes rules on demand; a production deployment should add a
reaper loop that calls `unblock` when the TTL expires, because a killed process
will not remove its own rules.

---

## 2. Safe rollout procedure

Do this in a lab or disposable VM first.

### Step 1 — validate detection in audit mode

```bash
cd iiit-mtech/IOT/finnet-gnn

finnet serve \
  --port 8583 --metrics-port 9101 \
  --trust-client-subnet \
  --firewall-backend audit

# second terminal
finnet simulate --port 8583 --events 900 --fraud-share 0.6

# confirm intent records
jq -c 'select(.operation=="block")' artifacts/firewall.jsonl | tail
```

You should see attack subnets such as `203.0.113.x/24` in the simulator or
`127.100.1.0/24` / `127.200.x.0/24` in the live demo. Verify the CIDRs are what
you expect **before** enabling enforcement.

### Step 2 — dry-run with the OS command

Run the exact command manually and observe the effect:

```bash
# macOS
sudo pfctl -t finnet_quarantine -T add 203.0.113.0/24
sudo pfctl -t finnet_quarantine -T show

# Linux
sudo iptables -I INPUT -s 203.0.113.0/24 -j DROP
sudo iptables -L INPUT -n --line-numbers
```

Have a second terminal open with a way to undo the change:

```bash
# macOS
sudo pfctl -t finnet_quarantine -T delete 203.0.113.0/24

# Linux
sudo iptables -D INPUT -s 203.0.113.0/24 -j DROP
```

### Step 3 — enable enforcement

```bash
finnet serve \
  --port 8583 --metrics-port 9101 \
  --trust-client-subnet \
  --enable-os-firewall \
  --firewall-backend pf        # or iptables
```

The gateway must run with sufficient privilege for `pfctl`/`iptables`. Run it
under a service manager with a narrow sudoers rule rather than an interactive
root shell.

### Step 4 — verify and monitor

```bash
curl -s http://127.0.0.1:9101/healthz | jq '.gateway.policy'
curl -s http://127.0.0.1:9101/metrics | grep finnet_quarantined_subnets
tail -f artifacts/firewall.jsonl
```

Check `ok: false` records immediately. A failed firewall call is surfaced in the
audit log and counted, but the in-process quarantine still applies.

---

## 3. macOS `pf` setup

macOS ships `pf` but does not enable its anchor/table configuration by default.

### 3.1 Create a quarantine table

Add a persistent anchor to `/etc/pf.conf` (back it up first):

```pf
# /etc/pf.conf — append, do not replace the existing file
table <finnet_quarantine> persist
block drop in quick from <finnet_quarantine> to any
```

Then reload:

```bash
sudo pfctl -f /etc/pf.conf
sudo pfctl -E
```

The `persist` keyword keeps the table alive when it becomes empty. Without it,
`pfctl -T add` can fail or the table can disappear.

### 3.2 Table commands

```bash
sudo pfctl -t finnet_quarantine -T add 203.0.113.0/24
sudo pfctl -t finnet_quarantine -T show
sudo pfctl -t finnet_quarantine -T delete 203.0.113.0/24
sudo pfctl -t finnet_quarantine -T flush
```

### 3.3 Check status

```bash
sudo pfctl -s info
sudo pfctl -s rules | grep finnet
sudo pfctl -t finnet_quarantine -T show
```

### 3.4 macOS safety notes

- **Do not block `127.0.0.0/8` or a subnet containing your SSH/screen-sharing
  address.** The simulator deliberately uses loopback-derived CIDRs so this
  risk is real in local demos.
- If you lock yourself out, boot into Recovery Mode or use another account to
  edit `/etc/pf.conf`, then `sudo pfctl -d` to disable pf temporarily.
- `pf` rules apply system-wide, not just to this process.

---

## 4. Linux `iptables` setup

The adapter issues:

```bash
iptables -I INPUT -s <cidr> -j DROP
iptables -D INPUT -s <cidr> -j DROP
```

### 4.1 Manual verification

```bash
sudo iptables -I INPUT -s 203.0.113.0/24 -j DROP
sudo iptables -S INPUT
sudo iptables -D INPUT -s 203.0.113.0/24 -j DROP
```

### 4.2 Persistence

`iptables` rules are not persistent across reboot unless saved by your
distribution:

```bash
# Debian/Ubuntu
sudo netfilter-persistent save

# RHEL/Fedora
sudo service iptables save
```

However, you should generally **not** persist dynamic quarantine rules. Persisting
them means a crash can leave stale blocks after a restart. Prefer a startup flush
of the FinNet chain/table and let the process rebuild state.

### 4.3 Use a dedicated chain (recommended for production)

Instead of inserting directly into `INPUT`, create a chain:

```bash
sudo iptables -N FINNET_QUARANTINE
sudo iptables -I INPUT -j FINNET_QUARANTINE
sudo iptables -I FINNET_QUARANTINE -s 203.0.113.0/24 -j DROP
```

That keeps FinNet rules isolated and makes cleanup auditable:

```bash
sudo iptables -F FINNET_QUARANTINE
```

The current adapter targets `INPUT` directly; switch it to a dedicated chain
before production use.

### 4.4 nftables

Modern distributions may front `iptables` with `nftables`. The `iptables`
command often still works via `iptables-nft`, but for a native deployment use:

```bash
sudo nft add table inet finnet
sudo nft add chain inet finnet input '{ type filter hook input priority 0; }'
sudo nft add rule inet finnet input ip saddr 203.0.113.0/24 drop
```

That is not implemented in the adapter yet; it is a straightforward extension of
`_execute`.

---

## 5. Privilege and service hardening

The gateway itself does not need root; only the firewall command does. Options:

### Option A — narrow sudoers rule

```sudoers
finnet ALL=(root) NOPASSWD: /sbin/pfctl -t finnet_quarantine -T add *, \
                             /sbin/pfctl -t finnet_quarantine -T delete *
```

Then have the adapter call `sudo -n pfctl ...` (requires a small adapter change).
Never grant blanket `NOPASSWD: ALL`.

### Option B — privileged helper

Run a tiny local helper (systemd unit or launchd daemon) that accepts validated
CIDRs over a Unix socket and performs the firewall mutation. The gateway stays
unprivileged. This is the recommended production shape.

### Option C — network-layer enforcement elsewhere

Send quarantine intent to a router, load balancer, or cloud security group via
an API. This scales beyond a single host and avoids local privilege entirely.

---

## 6. Failure and rollback

| Symptom | Cause | Response |
|---|---|---|
| `ok: false`, `pfctl not found` | macOS `pfctl` missing from PATH | Run the service with a PATH that includes `/sbin` |
| `ok: false`, permission denied | Gateway lacks root/sudo for the backend | Use Option A/B above |
| Table resets after reload | Missing `persist` in `pf.conf` | Add `table <finnet_quarantine> persist` |
| Legitimate customer blocked | Subnet too coarse or threshold too low | Raise `quarantine_threshold`, shorten TTL, allowlist the CIDR |
| Stale rules after crash | No reaper / no startup flush | Flush the table/chain on startup |
| Whole office disconnected | Shared NAT subnet quarantined | Use per-tenant identity or manual allowlist; revisit `/24` granularity |

### Emergency rollback

```bash
# macOS
sudo pfctl -t finnet_quarantine -T flush
sudo pfctl -d            # disable pf entirely, if necessary

# Linux
sudo iptables -F FINNET_QUARANTINE
# or, if rules went directly into INPUT:
sudo iptables -D INPUT -s 203.0.113.0/24 -j DROP
```

The in-process quarantine can be cleared only by restarting the gateway today
(no admin endpoint). That is intentional: there is no remote API that can be
used to unblock an attacker without authentication.

---

## 7. Production checklist

- [ ] Detection validated in `audit` mode against known-good and attack traffic.
- [ ] CIDR granularity reviewed (`/24` can be too coarse behind carrier NAT).
- [ ] Firewall backend tested manually before enabling.
- [ ] Privilege limited to the exact commands (sudoers or helper daemon).
- [ ] Startup flush of FinNet table/chain to avoid stale rules.
- [ ] TTL reaper removes expired rules even if the gateway is restarted.
- [ ] Allowlist for management/SSH ranges exists outside FinNet's control.
- [ ] `ok: false` alerts wired into monitoring.
- [ ] Rollback procedure documented and tested.
- [ ] `trust_client_subnet` is **off** in production.
- [ ] Fail-open/fail-closed choice (`policy.fail_open_on_timeout`) is explicit.
