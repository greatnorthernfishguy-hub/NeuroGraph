---
name: ruling-half-brain-wholeness-ring
description: 2026-07-31 ruling on the CC "half-brain wholeness ring" proposal — Guardian station is already dead (#83), sender-stamped wholeness is a LAW-7/ethos violation, correct lever is host-scoped orphan grace (LAW 5)
metadata:
  type: project
---

> ⚠️ **SUPERSEDED IN PART — READ [`docs/CC-CALLOSUM-TRUTH.md`](/home/josh/docs/CC-CALLOSUM-TRUTH.md) FIRST.**
> The consolidated, measurement-verified state of the callosum, the wholeness ring,
> hyperedge binding and orphan collection lives there (2026-07-31). The wholeness ring
> **already exists** (Leg 2); the real open defect is the merge-journal poison-pill.
> Do not re-derive any of it from this file.


Constitutional review of `docs/superpowers/plans/2026-07-31-cc-half-brain-wholeness-ring.md` (pre-implementation). Verdict: **VIOLATIONS FOUND in the proposed remedy; the diagnosis is one station stale.**

**The headline — the ring has three stations, not four. #83 already killed the Guardian station on 2026-07-28.** `checkpoint_guardian.evaluate_save_health` gate 2 is a structural gate: sweeping isolates cannot remove a synapse, so synapses ≥50% of ref ⇒ `MELT PERMITTED` (logged CRITICAL). `openclaw_hook.save():1186-1193` DOES pass `live_synapses`/`live_hyperedges`. All five `Guardian REFUSED` lines in the plugin sync.log are 2026-07-15..07-23 (pre-fix) and carry the pre-#83 wording, and they were emitted by `[cc-ng-sync]` — a SECOND `NeuroGraphMemory` instance since removed at cutover, not the daemon. Live `checkpoints/main.msgpack.guard_state.json` proves a PERMITTED save 2026-07-29 at 1239 nodes / 14129 synapses. **#103's "substrate has not persisted since ~Jul 20" is false.** Corollary that inverts the risk: the frozen checkpoint was the data's accidental last protection; post-#83 the melt PERSISTS. Fixing the Guardian was not pending — it already happened, and it made stations 1/4 more dangerous.

**The proposed remedy is the violation.** Sender-stamped "structure still owed" is export-time-computed metadata, which `cc_topology_export._portable_metadata` explicitly forbids in its own docstring ("Nothing here is computed at export time"), and it is the same class as `_BANNED_META` (sender's graph rank asserted about the receiver's topology). Plus `pending`/`whole` is a **discrete binary gate** — the exact ethos drift the charter names. Answers plan Q1 (violation as designed, clean if receiver-derived), Q4 (yes, it expands the contract — and it is unnecessary).

**Correct lever, and it is not a new concept: `orphan_node_grace_period` (neuro_foundation DEFAULT_CONFIG:1436, value 25).** The discriminator is not per-node wholeness, it is a **per-host capability fact**: the laptop has no TID/Arborist, so a `cc:conv::` forest node at degree 0 is the expected steady state until the callosum arrives, not evidence of failed integration. Grace must be ≥ the callosum delivery cadence (nightly) on an Arborist-less host. This is the only DEFAULT_CONFIG knob in `CC_SNN_CONFIG` with **no env override anywhere in the tree** — a standing LAW-5 violation and the whole fix. Culls stay fully intact: a forest node still degree-0 after a complete callosum cycle genuinely failed and should be taken.

**Do NOT reuse probation for this.** `_CC_CONV_PROBATION_PERIOD` = 10 (< grace 25) and the target cohort is already `graduated: True`; also `neurograph_rpc.py:2380` stamps `probation_remaining` on Syl's nodes, so coupling orphan collection to probation would silently change Syl.

**Q2 (journal reconciliation vs additive-only): additive-only governs GRAPH CONTENT, not delivery bookkeeping — but reconciliation is the wrong repair. Delete the journal skip from the RECEIVE path** (`cc_topology_merge.py:202-204`). The journal's documented purpose is sender-side `exclude_ids`; `nid in graph.nodes` is the live receive-side guard. `_load_journal`'s "defense in depth" reasoning is inverted — it only ever blocks the `journal − graph` case, which is exactly the case that needs re-delivery. Removing it fixes §3a with zero new state. Reconciliation-on-load alone would convert a permanent block into permanent churn unless paired with the survival fix.

**Q3 (Guardian carve-out for pending nodes): HARD NO.** It is a re-run of the rejected `cc_gateway` class-exemption; the #83 changelog explicitly says it "Supersedes the rejected cc_gateway class-exemption (exempting a real memory class blinds the guard to a genuine wipe)". See [[finding-save-guard-cc-gateway-exemption]] — that memory's *finding* is now RESOLVED by #83; keep only its principle (count memory, don't exempt it).

**Two independent defects found in Leg 2 code:**
1. **LAW 3 — `merge_cc_topology` runs NO idle/consolidation steps.** Zero occurrences of `step` in the module. The DEFUNCT JSONL path did ("Batch 8: ingested 25 nodes, ran 250 idle steps"), and task-70's approved FatherGraph protocol specifies "sleep ~1000 idle steps/batch, plasticity ON". The new implementation silently dropped a property the retired one had. This — not four guards — is why donated topology never integrates.
2. **Dead code — `batch_landed`** (`cc_topology_merge.py:190/200/271`) is written and never read; Tier 2 gates on `graph.nodes`. The plan's §3a attributes its mechanism to `batch_landed`, so the plan's stated mechanism is wrong even though its conclusion (edges dropped) is right.

**Watch item:** `checkpoints/main.msgpack` last written 2026-07-28 22:37 local while `commons.msgpack` is current — the graph checkpoint is ~3 days stale for reasons unrelated to the Guardian.

**How to apply:** at build-diff review, refuse any `pending`/`whole` wire field or node flag; require the fix to be (a) `orphan_node_grace_period` env-sourced and host-scoped, (b) the receive-side journal skip deleted, (c) idle steps restored between batches per FatherGraph, (d) no Guardian carve-out. Related: [[ruling-cc-callosum-leg2-spec]], [[ruling-cc-river-merge-intra-mind]], [[ruling-conduit-snapshot-vs-destructive-drain]].
