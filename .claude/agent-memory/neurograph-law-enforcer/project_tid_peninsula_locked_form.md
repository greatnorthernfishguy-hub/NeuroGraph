---
name: project-tid-peninsula-locked-form
description: TID substrate peninsula (#97) — the transmit/deposit-replay form is DEAD; only the share form (Q2 shared-mem attach) or per-decision synchronous bucket are Law-clean
metadata:
  type: project
---

TID Substrate Peninsula (Task #97) — the locked compliance ruling, so resurrections get caught fast.

**Fact:** The peninsula CONCEPT (function-split; pure-compute body in its own process; substrate-half in-process on the Commons; pull-not-push) is AFFIRMED clean. The MECHANISM is constrained: only the **share** form is Law-clean — ONE NGLite instance mmap'd into both address spaces (Syl's Q2 shared-memory attach), OR a per-decision synchronous bucket with NO body-side store. The **two-sub-peninsula / cross-boundary deposit-replay / msgpack-transmit** form is DEAD (Josh + compliance + Syl, 2026-06-21, in `~/docs/prd/substrate-peninsula-design.md`).

**The durable discriminator:** across the process boundary, are bytes TRANSMITTED (serialize→send→deserialize into a *second* store kept in agreement by replay/merge → two media synced → LAW-1 VIOLATION) or SHARED (one NGLite store mmap'd into both, no second instance, no merge → CLEAN)? A TID-side cached topology store that TID buckets from = NGLite instance #2 = transmit = violation. `commons.py` is a process-singleton *specifically* to prevent the dual-instance split.

**Tells to watch for in any re-proposal:** (1) a "local current-state / cached enhanced topology" on the TID-body side; (2) msgpack/socket/asyncio carrying "topology splashes" between halves; (3) "one direction at a time, never simultaneously bidirectional" — that lost-update avoidance is the *confession* that two states exist; (4) a "raw, not classified" defense — that clears LAW 7, NOT LAW 1; don't let LAW-7 reasoning answer a LAW-1 question.

**Why:** This transmit-form has been proposed at least twice under different clothes (deposit-replay 2026-06-21; msgpack-over-asyncio 2026-06-27). It keeps coming back because it looks clean and is easy to build. It is not clean.

**How to apply:** When reviewing any peninsula build, verify it is the share form (or no-body-store per-decision bucket) before GO. Confirm the two gating prerequisites too: TID stability (firebreak), and TID-body sheds its local NG-Lite (LAW 3 — no parallel Tier-1 instance). See [[The Substrate Axiom]] and commons.py singleton.
