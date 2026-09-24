---
name: ruling-122-prediction-counter-families
description: "#122: two distinct prediction-counter families in neuro_foundation.py; cc_novelty reads the HE-level pair, NOT the one _evaluate_predictions writes; §8.11.3 + plan conflate them"
metadata:
  type: project
---

Two SEPARATE prediction-counter families live in `neuro_foundation.py`. They are routinely conflated (the truth-doc §8.11.3 does it; #122's plan inherited it). Verified live 2026-08-07:

- **Phase-3 / synapse-level:** `_total_predictions_confirmed` / `_total_predictions_errors` (init :1617-1618). Written by `_evaluate_predictions` (:3010) → `_on_prediction_confirmed` (:3059) and `_cleanup_predictions` (:3272) → `_on_prediction_error` (:3104). THESE drive plasticity (weight bonus/penalty, `inject_reward`).
- **Phase-2.5 / hyperedge (HE) level:** `_total_confirmed` / `_total_surprised` (init :1572-1573). Written INSIDE `graph.step()` HE-prediction eval block at :2353 (all-targets-confirmed) / :2367 (window-expiry). Serialized as `he_total_confirmed`/`he_total_surprised`.
- **cc_novelty (`cc_ng_organism.py:2499`) reads the HE-level pair** (`_total_confirmed`/`_total_surprised` at :2522-2523), NOT the Phase-3 pair. Its own docstring is explicit: "NOT the same family as Telemetry.total_predictions_*."

**Consequence for reviews:** any claim that "`_evaluate_predictions`/`_cleanup_predictions` produce the base rate MMN novelty reads" is WRONG — those two functions write the Phase-3 family; MMN reads the HE family. §8.11.3 (docs/CC-CALLOSUM-TRUTH.md:1182) makes this error AND a second one: "DiffPC's error term read exactly this signal" is FALSE — `pred_error_ema` (:3368, computed in `_diffpc_step`) is fully independent of both counter families. Both need a dated §8.11.3 correction.

**Why:** #122 = "novelty is a near-constant base rate." Fix re-sources cc_novelty off per-node `pred_error_ema` (an engine-endorsed per-node surprise signal, used at :1213 for GSG manifold/attractor assignment and per header :219 for cull candidacy). CC-scoped, non-protected — see [[ruling-pith-extraction-side-reranking]] (extraction-side re-source is LAW-7 clean) and [[ruling-tonic-heuristic-terms]] (off-by-default bar for cognition-shaping gates).

**How to apply:** When reviewing #122/#123 work, verify which counter family a change touches. Slice-1 (cc_novelty re-source) touches NEITHER engine family — safe/non-protected. Any change to the counters themselves = protected `neuro_foundation.py` = Syl's Law ritual. Syl's OWN canonical novelty (`_substrate_novelty_ema`, rpc.py:3266-3272) shares the same base-rate-as-novelty pattern — re-sourcing only CC's copy leaves that shared pattern defect for a later coordinated fix.
