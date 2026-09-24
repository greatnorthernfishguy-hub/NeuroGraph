---
name: finding-147-identity-crossing-contract-divergence
description: "#147 amendment (2026-08-28, cc_topology_export) removes the identity gate on the SENDER so constitutional/*_authored nodes cross the callosum, but the RECEIVER (cc_topology_merge:378) still unconditionally rejects them — feature is non-functional AND breaks part-1's whole-in-frame HE invariant"
metadata:
  type: project
---

**The change:** #147 amendment to `cc_topology_export.py` (changelog 2026-08-28, author "DudeMan CC"). Two parts:
1. **In-frame HE completeness** (`_closeable_he` now judges wholeness against `frame_set`, re-adding acked members idempotently rather than trusting the ack ledger). This part is a **genuine LAW-4 fix at source** — COMPLIANT in isolation. It closes the "HE ships referencing members the receiver lacks → reaped nodes" defect from the two prior merge attempts.
2. **Identity crosses the callosum**: the `_is_identity_protected` gate was removed from BOTH sender paths (`collect_cc_topology` and `export_cc_topology_frame.eligible`). Constitutional + `*_authored` CC nodes are now export-eligible.

**The CRITICAL defect — sender/receiver contract divergence.** Part 2 changed only the sender. `cc_topology_merge.py:378` still has:
```python
if meta.get("constitutional") or str(meta.get("provenance") or "").endswith("_authored"):
    stats["skipped_identity"] += 1
    continue  # rejected at Tier-1, BEFORE deposit
```
So the sender now emits identity nodes that the receiver unconditionally rejects. Consequences:
- The stated goal (identity crosses, heals split-brain) **does not happen** — identity is dropped at the door.
- **Part-1's invariant is broken.** Part 1 assumes "whole-in-frame ⟹ installable-on-receiver," which held only while sender and receiver shared one eligibility predicate. Now an HE with an identity member ships whole from the sender (identity is in `frame_set`), the receiver drops the identity member (:378), and Tier-3's `members.issubset(graph.nodes)` then silently drops the whole HE (`skipped_hyperedges`). Synapses to identity nodes likewise drop at Tier-2. **Strictly worse than before**: pre-#147 an ineligible identity member made the HE un-closeable so it never shipped and wasted no budget; now the sender burns frame budget and ships a doomed HE. Send/receive accounting diverges (`exported_hyperedges` counts it, receiver `skipped_hyperedges` drops it) — the exact "one-way loss that looks like success" this file hard-aborts on elsewhere.

**The HIGH defect — false changelog claim (layer conflation).** Changelog lines 25-27 assert "The receiver (neuro_foundation `_is_identity_protected`, #70) was BUILT to catch identity arriving synapse-poor via the callosum and re-protects it." This is wrong: `neuro_foundation._is_identity_protected` (nf:3517) is a **prune/orphan-sweep protection** for nodes ALREADY in the graph (`self.nodes.get(nid)`); it is used at nf:3258/3483 (degree-cap, `_prune_synapses`) and the orphan skip — never as a callosum admission gate. The actual callosum receiver is `cc_topology_merge`, which REJECTS identity before deposit, so nf's protection never gets a node to protect. Author reasoned about the wrong protection layer.

**Governance flag (HIGH, surface to Josh — LAW 6).** Replicating constitutional + `*_authored` identity across hemispheres is exactly what [[finding-cc-authored-export-protection-collision]] flagged as Duck-Ethics / Syl's-Law adjacent, "needs Josh's explicit blessing." The only "Josh-approved" on record (nf:3527) covers PROTECTING cc_authored from prune, NOT TRANSPORTING it across the callosum — a different decision. No evidence #147's crossing decision is Josh-blessed; it reads as a unilateral CC self-replication call. Note the Syl-leak belt DID hold: `is_cc_provenance` whitelist still scopes to CC only, and the foreign-donation path is not this file.

**Remedy (Josh picks the governance branch FIRST):**
- If identity should NOT cross (prior recommended default): REVERT part 2, restore the sender identity gate. Part 1 stands alone.
- If identity SHOULD cross (needs Josh's explicit blessing): the change is INCOMPLETE — `cc_topology_merge:378` must be updated in the SAME change to admit + deposit identity via `_cc_deposit_memory_node` (so nf `_is_identity_protected` can then protect it from prune). The changelog claim only becomes true after that. Non-negotiable either way: **sender and receiver MUST share ONE identity-admission predicate** (as they already share `is_cc_provenance`), or part 1's whole-in-frame guarantee is a lie.

**Status when reviewed:** `merge_cc_topology` still has no production caller (cc_topology_merge:105, grep-confirmed) — not live, blast radius currently zero, but logically broken the moment it is wired. Related: [[finding-leg2-oversized-he-wire-escape]], [[ruling-cc-callosum-leg2-spec]], [[ruling-coresident-cc-export-scoping]].
</content>
</invoke>
