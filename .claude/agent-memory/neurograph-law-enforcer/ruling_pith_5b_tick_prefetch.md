---
name: ruling-pith-5b-tick-prefetch
description: Pith Stage 4 phase 5b (tick-riding Markov prefetch) — LAW 1/2/3/4/6/8 clean, LAW 5 fail; write-mode deviation from signed-off design + brakes/cap bypass are the merge-blockers
metadata:
  type: project
---

Phase 5b take 2 (2026-09-05): `TonicEngine.set_prefetch_seed(fn)` + `_merge_prefetch_seeds()` fold host-supplied live `primed_nodes` into each Tonic tick's prime. Take 1 (`_PITH_PREFETCH` dict cache + third standing pulse) was fully reverted — no shrapnel left (LAW 3 clean). Seed source injected by the host, so `tonic_engine.py` knows nothing of `conv_state` (LAW 1 clean, good dependency inversion). No vendored file and no `neuro_foundation.py` touched — the spec §7 back-vendor sign-off was NOT consumed (LAW 2 clean).

**Design-doc deviation that needs Josh's re-signoff.** `2026-09-05-pith-5b-take2-design.md` prescribes `prime_and_propagate(..., write_mode=False)` and justifies it as keeping prefetch "clear of LAW 7 and of plasticity side effects". The build instead rides the Tonic's **write_mode=True** tick. The implementation is physically right (verified `neuro_foundation.py` restores voltages/refractory at the end of every read-mode call, so a standalone read-mode prime warms nothing) — the doc is stale. But write mode means predicted nodes now get STDP, `_sprout_synapses` (#163) and age-on-write. The doc's plasticity rationale is void and nothing replaced it. Changelog does not disclose this.

**`_merge_prefetch_seeds` bypasses the engine's own governors — the real merge-blocker.** It appends AFTER `_heuristic_inference` has already run `_apply_brakes(seen)` and capped at `max_activation_nodes` (10). So prefetch seeds get (a) no #62 divisive brake damping and (b) no budget. Default `_CC_PITH_PREFETCH_MAX`=15 > the whole engine budget of 10. Worse: `cc_anticipate` scores by **accumulated outgoing synapse weight**, which is structurally biased toward exactly the high-degree hub nodes the #62 brakes exist to damp (#59's ~310-node self-feeding blob). Remedy is to merge BEFORE brakes+cap, not to add a second cap. See [[ruling-tonic-heuristic-terms]].

**Score-scaling is largely inert.** `cc_anticipate` scores are unbounded weight SUMS, not [0,1] confidence; `min(1.0, ...)` clamps most hub seeds to a flat full current. `_CC_PITH_PREFETCH_MAX` (15) == `_CC_ANTICIPATE_TOP_K` (15) by coincidence, so the cap never binds and the sort is moot.

**Repetition amplification (nowhere in any doc).** `_CC_ANTICIPATE_TTL_S`=120s vs Tonic interval 2-10s → the SAME ≤15 predicted nodes are re-primed in write mode **12-60 times per turn**, each with STDP + sprouting. Not "warm the neighbourhood once."

**LAW 7 ruling: NOT a violation.** Seeds are substrate-derived (Hebbian neighbours of fired nodes), not external classification, and the Tonic already write-primes `graph.active_predictions` (heuristic term 3). Same kind, not a new kind. **But** `active_predictions` has confirm/surprise resolution (`_total_confirmed`/`_total_surprised`, `_evaluate_predictions`); `primed_nodes` has **none** — nothing ever penalises a wrong prediction, so 5b is an open-loop autocatalytic circuit (predict → fire → STDP strengthens the predicting synapse → predict harder). Ethos finding, not a Law violation.

**Measurement confound.** `prefetch_seeded` lives on `TonicEngine.status`; `prefetch_surfaced` lives in `PithMetrics` (the daemon's `pith_metrics` RPC reads ONLY `_PITH_METRICS`). An operator taking the §6 def-of-done hit-rate cannot distinguish "prediction is bad" from "prefetch never ran." Surface both on one readout before measuring.

**Verification gotcha (cost me a bogus finding):** `tests/test_tonic_prefetch.py` appeared to fail 9/9 with `AttributeError: no attribute '_CC_PITH_PREFETCH_WARM_ENABLED'`. That was a **concurrent write** to `tonic_engine.py` mid-review, not a defect. Re-run after `md5sum` stabilises: 63/63 green across pith_stage4 + tonic_prefetch + pith_stage5 + cc_recall_unification + tonic_write_mode, both orders. Always md5-stabilise the file before trusting a test failure on a live-edited tree.

**Thread-safety verified clean:** every `primed_nodes` writer REBINDS (`state["primed_nodes"] = {...}`), never mutates in place, so the Tonic thread iterating a stale reference is safe. Do not re-flag this.

Related: [[ruling-pith-stage4-predictive-promotion]], [[pith-gate-tests-need-monkeypatch]].
