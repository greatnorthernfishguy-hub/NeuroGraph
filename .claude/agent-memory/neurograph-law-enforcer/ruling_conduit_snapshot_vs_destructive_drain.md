---
name: ruling-conduit-snapshot-vs-destructive-drain
description: CC Callosum Leg1 — a 2nd reader that snapshots a single-drainer-truncates tract INDEPENDENTLY of the drain's own read loses bytes in the read-gap (not delayed — lost)
metadata:
  type: project
---

> ⚠️ **SUPERSEDED IN PART — READ [`docs/CC-CALLOSUM-TRUTH.md`](/home/josh/docs/CC-CALLOSUM-TRUTH.md) FIRST.**
> The consolidated, measurement-verified state of the callosum, the wholeness ring,
> hyperedge binding and orphan collection lives there (2026-07-31). The wholeness ring
> **already exists** (Leg 2); the real open defect is the merge-journal poison-pill.
> Do not re-derive any of it from this file.


CC Corpus Callosum Leg 1 (#70) conduit: the laptop daemon snapshots `cc_gateway_tract_path()` bytes at pulse line 1406, THEN calls the destructive `drain_ingest_tract()` (line 1409, reads its own `data` + truncates it), THEN `trickle_gateway_conduit(snapshot)` (line 1410).

**The divergence bug (HIGH, review 2026-07-27):** snapshot and drain do two SEPARATE reads-from-0. Bytes miniTID (Rust, no handshake) appends in the window (snapshot-read, drain-read) are in drain's `data` → drained into the laptop forest AND truncated away by drain's `remainder = current[len(data):]` logic → but NOT in the snapshot → never trickled → never in any FUTURE snapshot (drain already removed them). Result: those turns reach the laptop hemisphere and are LOST to the VPS hemisphere permanently. NOT "delayed one pulse" as the changelog claims. Low-rate (sub-ms window) but silent + unbounded + undetectable — exactly the hemisphere-divergence Leg 1 exists to eliminate (it's retiring the lossy top-N JSONL sync).

**Reusable ruling:** any second consumer of a single-drainer-truncates tract MUST read the exact same bytes the drainer consumes, or bytes in the read-gap are lost. drain's own `remainder` logic protects drain's OWN consistency; it does nothing for a second reader's divergence.

**Remedy (substrate-native, LAW-4 fix-at-source; drain_ingest_tract is CC-owned in cc_ng_organism.py, NOT vendored):** make drain return the exact bytes it consumed (additive optional kwarg, e.g. `return_consumed=True` → `(absorbed, consumed)`), trickle THOSE. One read, byte-for-byte parity, zero window. Leg 2 will hit the same shape — see [[ruling-cc-river-merge-intra-mind]].

Also flagged same review: corrupt/unparseable batch file is retried-forever + never quarantined (drain returns early before truncate → getsize!=0 → never deleted), and only logs at logger.debug (a format-skew pile-up in the git-synced ~/docs/ng_topology would be silent) — MEDIUM, fix before gate-on. Gate-off is clean EXCEPT the laptop snapshot read itself is ungated (one extra file read/pulse even when off) — LOW vs the byte-identical gate-off house style ([[ruling_pith_stage4_predictive_promotion]]).
