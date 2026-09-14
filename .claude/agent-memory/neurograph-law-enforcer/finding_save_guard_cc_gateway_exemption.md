---
name: finding-save-guard-cc-gateway-exemption
description: CC save-guard wire-deposit exemption (2026-07-21) exempts conversational-memory nodes, not transit — inverts the guard's founding purpose
metadata:
  type: project
---

CC daemon save-guard (`docs/scripts/cc-ng-daemon.py`, `_substantive_node_count`) was changed 2026-07-21 to EXEMPT `metadata['source']=='cc_gateway'` nodes from the collapse-count, calling them "ephemeral ingest-tract transit, not substrate."

**The premise is false at the graph-node level.** In the GRAPH, `source=='cc_gateway'` is stamped ONLY by `run_conversational_dual_pass` (`cc_ng_organism.py:1271`) and those nodes ALSO carry `creation_mode:"conversational"`. They are the CC's primary conversational-memory nodes (dedup'd per turn, id `cc:conv::<sha1>`), surfacing-eligible. The TRACT FILE on disk is the transit; the graph node the dual-pass crystallizes from it is memory. The change conflates the two.

**Empirical:** live incident 2026-07-21 melt 1236→483 total nodes; the ~750-1036 shed were the exempted cc_gateway/conversational class. The guard's own founding checkpoint (~1800 nodes, 2026-07-09 incident) was ~84% cc_gateway — so the change declares ~84% of what the guard was built to protect "not substrate."

**Regression:** if a bad restore brings the daemon up with the constitutional/identity/concept core intact but conversational memory wiped, substantive≈ref → guard PERMITS overwriting the good checkpoint. Re-opens the exact incident class (post-crash empty/degraded restore clobbers healthy checkpoint) for the most valuable subset.

**Not a LAW 1-7 mechanical violation.** Guard-time read of pre-existing metadata is extraction-time classification (see [[ruling-structural-dedup-vs-classification]]), LAW 7 clean. LAW 2 clean (CC-local file, openclaw_hook untouched). LAW 5 clean (env-overridable). The severity is safety/continuity: it erodes protection of the mind's conversational memory (Duck-Ethics-adjacent).

**False analogy flagged:** changelog cites "Mirrors Syl's transient-region exclusion (commons_enhance.py)." That transient region is a per-cycle SANDBOX deleted every cycle (`assert leaked == 0`) — genuinely ephemeral compute scratch, NOT a persistent node class. Not analogous.

**Substrate-native remedy:** the real bug was a STALE high-water healthy-ref (1236) vs an intended melt crossing the 0.5 ratio. Fix by modeling the healthy-ref as an EMA / slow-decaying reference (a melt drags ref down with it; a collapse is a fast drop relative to EMA) or a rate-of-change gate — the substrate's own vocabulary (firing_rate_ema/novelty EMA), ethos-aligned continuous model over static class-exemption. Count conversational memory, don't exempt it.

**Transition hazard:** ref only updates on a non-refused save, so a redeploy/rollback that leaves a raw-basis ref (e.g. 1236) on disk while the graph is post-melt gets STUCK refusing forever. This deploy evidently cleared `.healthy_node_count` manually (reseeded to 484 at 23:00:33). Undocumented required step.

**Tests lock in the wrong invariant:** `test_save_guard_wire_exempt.py` blesses the inversion and never tests the dangerous case (conversational wipe, core intact → guard should refuse but now permits).
