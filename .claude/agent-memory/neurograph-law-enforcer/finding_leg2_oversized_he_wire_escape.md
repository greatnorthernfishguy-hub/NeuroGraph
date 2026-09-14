---
name: finding-leg2-oversized-he-wire-escape
description: "RESOLVED 2026-08-19 by emit-step wholeness cap. #88 §10.4-A export_cc_topology_frame: oversized-HE hard cap was enforced only in node-placement greedy fill, NOT at collect_incident_structure emit — oversized HE crossed silently once members trickled over via other anchors"
metadata:
  type: project
---

> **RESOLVED 2026-08-19** (uncommitted change, reviewed COMPLIANT). `export_cc_topology_frame()` now re-applies `len(members) > hard_cap` to the collected `hes` right after the frame-incidence filter (`cc_topology_export.py:860-882`), dropping + alarming any blob that slips through piecemeal. Implements the exact remedy prescribed below: guard at the sender's HE-emit step, NOT in shared `collect_incident_structure` (Leg-1/Leg-3 untouched). Dedup keyspace is coherent — branch (e) at :800 keys `_oversized_seen` by `he_id` (from `node_hes`), emit guard at :871 keys by `h.get("id")` which `collect_incident_structure` sets to the same `he_id` (:398), so shared `oversized_he_at_source` counter never double-counts. Caught on exactly the frame the last member lands (whole-containment filter). Member nodes + incident synapses still cross via their own anchors; no dangling ref (payloads reference node ids, not HE ids). LAW 2/3/4/7 clean. Regression: `test_cc_topology_frame.py::test_oversized_hyperedge_cannot_reassemble_whole_via_piecemeal_members`. Doc synced in `~/docs/CC-CALLOSUM-TRUTH.md` (two-point enforcement). Leg S (VPS dream split) still owes the actual source-side blob repair; this only refuses transport + surfaces the unfinished repair loudly. Below is the original finding, kept for context.

---

`export_cc_topology_frame()` (cc_topology_export.py, #88 §10.4-A, commit bbdc3de) claims the invariant: "a single HE past the hard cap is skipped, never crossed — proof Leg S hasn't run."

**The guard leaks.** The `hard_cap` (frame_size*overflow_factor) only bounds how many NEW nodes the greedy fill places per frame (branches b/c and `_closeable_he`). The hyperedge RECORD is actually put on the wire by `collect_incident_structure(graph, frame_set | exclude_ids)`, which ships EVERY non-archived HE whose members ⊆ member_set with **no size check**. So an oversized HE whose members each have some OTHER anchor (synapse or smaller HE) gets its members shipped individually across many frames; on the frame where the last member lands, `members.issubset(member_set)` is True and the oversized HE crosses in one conduit frame — uncapped.

Worse: when all members place via branches (a)/(b)/(d), branch (e)'s oversized-detection is never reached, so **`oversized_he_at_source` never fires** — the blob crosses SILENTLY. Defeats the co-tenant RAM bound the whole paced design exists for.

**Why:** node-count bound holds; HE-record-size bound does not — they're enforced in two different places.
**How to apply:** remedy is a size guard at the frame sender's HE-emit step (after `hes = [...]`, drop+alarm any HE with member count > hard_cap), NOT in shared `collect_incident_structure` (that's also Leg-1/Leg-3, changing it there alters their behavior). Dropped nodes still ship via their other anchors; the HE waits for Leg S — exactly the intended semantics. Re-check this on any future Leg-2/Leg-S review. Related: [[ruling_cc_callosum_leg2_spec]].
