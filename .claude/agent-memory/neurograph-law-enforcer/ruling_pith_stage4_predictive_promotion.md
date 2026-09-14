---
name: ruling-pith-stage4-predictive-promotion
description: Pith Stage 4 phase 5a — promoting already-resident primed nodes at recall is extraction-side/LAW-7 clean; gate-off byte-identical contract pattern
metadata:
  type: project
---

Pith Stage 4 phase 5a (`cc_pattern_completion_recall`, cc_ng_organism.py ~1500-1600): live `primed_nodes` NOT found by the query harvest are injected as recall candidates (strength=`_CC_ANTICIPATE_BONUS`), then `cc_gsg_rescore` + rank/budget decide survival. Far-from-query promotions (GSG dist > `_CC_PITH_PREFETCH_LOD_DIST`) staged as `pith_stage2_keyframe` summaries.

**Ruling: LAW-7 CLEAN — extraction-side.** Promotion pulls already-deposited graph nodes; `primed_nodes` come from `cc_anticipate` deposit-time Hebbian pre-activation, not ingest classification. LOD summarization is extraction-time compression. Same family as [[ruling-pith-extraction-side-reranking]]. LAW 1 clean (own substrate only; `from ng_embed import embed` is the established `cc_gsg_rescore` pattern, not inter-module).

**Reusable gate-off contract pattern (verify on every gated CC feature):** init the sentinel (`promoted_ids=set()`) OUTSIDE the enable branch so all downstream `if promoted_ids:` blocks are inert when off → byte-identical. Confirmed here; `test_prefetch_gate_defaults_off` + `test_promotion_noop_when_gated_off` lock it.

**Recurring drift seen this review:**
- Changelog header omission — Stage 4 diff added no header entry to cc_ng_organism.py (repo CLAUDE.md: "Not optional"). Watch Pith/CC diffs for this.
- `_cc_node_query_distance` COPIES cc_gsg_rescore's spherical/hyperbolic manifold branch instead of factoring it out (docstring claims "factors out" but doesn't). LAW-4-spirit single-source drift risk — flag on any second copy of the GSG distance branch.

**Why:** #55 Pith arc; a prior "Stage 4 done" claim was FALSE (only the signal existed). This build genuinely promotes harvest-missed nodes (proven by real-harvest fixture, 18/18 green). Gated OFF by default (`CC_PITH_PREFETCH_ENABLED`), measure-first per spec 5a.
**How to apply:** Stage 4 promotion is sanctioned extraction-side. When reviewing 5b (warm buffer `_PITH_PREFETCH` / idle-pulse), re-check: local module state only, gate-off inert, no write-back to substrate.
