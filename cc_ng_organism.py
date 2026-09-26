#!/usr/bin/env python3
# SEE FIRST: /home/josh/docs/CC-CALLOSUM-TRUTH.md -- consolidated, verified state of
# the callosum, wholeness ring, hyperedge binding and orphan collection (2026-07-31).
# The wholeness ring ALREADY EXISTS here (Leg 2). Open defect: merge-journal poison-pill.
# ---- Changelog ----
# [2026-09-26] openrouter/deepseek/deepseek-v4.1-flash (OpenCode harness on T3 Code),
#   lane z2-one-step-per-turn-001 — Exec P240(2)/P241: one step per turn
# What: cc_deposit_step docstring only (no code-line change). The Doors paragraph
#   now names the hook door as the Stop-side _deposit(step=True) alone, once per
#   turn even when the dual pass failed; the prompt-side, pith-failure and
#   PostToolUse deposits never call it.
# Why: Chief-p240-commission-001 row z2-one-step-per-turn-001; P241 (Lanes 1 and
#   2 land together); canonical cardinality: Syl's handle_after_turn does
#   exactly one graph.step() per turn.
# How: one sentence reworded in place; the drains sentence is left as is (Lane 3).
#   The twin docs/scripts/cc-ng-daemon.py is deliberately untouched (Z12's item).
# [2026-09-26] Z2 zone manager (Claude Opus 5.5, Claude Code) — B3 fix round 1
# What: cc_deposit_step docstring: "Always steps, whether or not the dual pass
#   succeeded" scoped to the hook door. Docstring only; no code change.
# Why: B3 P187 pair, LE note 3 — the sentence contradicted the new Doors
#   paragraph (the drains step on applied records only); Chief B3 ruling 001
#   requires the docstrings corrected.
# How: one sentence reworded in place.
# [2026-09-26] GLM (z-ai/glm-5.3-flash, OpenCode harness on T3 Code),
#   lane z2-b3-kiss-drain-step-001 — the drained turns step; Leg-1 docstring
#   qualified
# What: drain_ingest_tract and drain_gateway_conduit (Leg 1) each call
#   cc_deposit_step once per APPLIED record, behind the existing
#   _CC_NG_DEPOSIT_STEP flag (default off; nothing flips it). No step on a
#   skipped, paused, uncertain, already-applied or failed apply; no
#   threshold, dedup, similarity check or skip of the raw deposit (LAW 7).
#   Docstrings: cc_deposit_step now names its three doors; drain_ingest_tract
#   states the caller holds graph._concurrent_lock (required by both the
#   step and #643); "No synthetic graph steps." is qualified per R2 (it bars
#   idle/consolidation cadence, not the per-applied-record deposit step).
# Why: Chief B3 ruling 001 (docs 0ef6dac1) R1/R2/R3; P153(4) Q3 — the CC
#   deposit steps, and that is where KISS ops 1 and 6 get their receipt; op 1
#   at Apprentice is the Delta Gate on graph data (KISS.md:38) — the step plus
#   cc_deposit_step's existing fired-set gate. Assignment
#   z2-b3-kiss-drain-step-001 Part 1.
# How: flag-gated cc_deposit_step call immediately after each drain's truthy
#   _apply_gateway_experience (conduit: inside the same graph._concurrent_lock
#   hold, after the not-applied guard). Tests: tests/test_cc_deposit_step.py.
# [2026-09-25] B1 coding worker (GLM 5.3 Flash, OpenCode/T3 Code) — Pith
#   cache-line cleanup (P224(1)(a)) + #522 coherence across victim eviction
# What: CacheLine loses the four computed-never-read fields `lod`,
#   `manifold_type` (the CacheLine field only), `keyframe` and `deltas` --
#   they were declared, written and copied with no production reader. Stage
#   3 graceful degradation keeps ONLY the compressed head in `content` plus
#   the existing compressed_count/chars_saved metrics. The basin builder
#   keeps `relations` (what the renderer reads) and drops its string
#   duplicate list. pith_victim_capture now stores `coherence` and
#   pith_victim_recover restores it (v.get("coherence", "unknown")), so an
#   evicted line keeps its coherence state and a pre-fix entry degrades
#   honestly to unknown (#522). The CacheLine docstring now names the
#   fields actually read today, citing each reader by function name; the
#   basin comment names `relations` instead of "deltas".
# Why: chief-b1-ruling-002 (docs 9868accd) APPROVE-REVISED; P222(1) and
#   P224(1)(a) -- dead, unconsumed, superseded code is deleted in the same
#   change that makes it dead; punchlist #522. Assignment
#   z2-b1-pith-cleanup-001 (lane z2-b1-pith-cleanup-001).
# How: writes/copies deleted at pith_stage3, _pith_copy_cache_line,
#   pith_connected_activation_basins, CacheLine and CacheLine.from_surfaced
#   (the manifold_type param goes with its field). pith_stage2_keyframe's
#   (keyframe, delta) return is UNCHANGED -- its other callers live; its
#   docstring no longer names the deleted CacheLine field. Graph-node
#   manifold_type reads (getattr on nodes, GSG geometry) are untouched.
#   Tests: tests/test_pith_stage2.py (asserts the compression's real
#   contract), tests/test_pith_provider_context.py (fixtures drop the dead
#   kwargs), tests/test_pith_stage5.py (deleted-field absence,
#   from_surfaced kwarg rejection, #522 round-trip, legacy-dict
#   degradation).
# [2026-09-25] Z2 zone manager (Claude Opus 5.5, Claude Code) — #592 COMB-04
#   L1 budget direction corrected (Executive Packet 193(1))
# What: cc_l1_budget's region-confidence factor is now 1 - (c - NEUTRAL)*2*FALLOFF
#   (was 1 + ...): high confidence shrinks the L1 budget, low widens it. Same
#   neutral (0.5), falloff (CC_PITH_REGION_CONFIDENCE_FALLOFF) and [500, 40000] clamps.
# Why: KISS_Pith_Combined_Architecture.md l.169 (high confidence -> "Pith can
#   aggressively compress extraction") and l.170 (low -> "Pith loosens (promote
#   more context to L1)"). The 50e05bc build inverted this (#592, grok
#   direction check in groupb-comb04-shared-graduation-002-rev56-review-001).
# How: one sign; docstring/comments quote the spec. Flag
#   CC_PITH_REGION_CONFIDENCE_ENABLED stays default off (COMB-04 DISABLED).
#   Tests: tests/test_cc_region_confidence.py direction tests.
# [2026-09-25] Z2 zone manager (Claude Opus 5.5, Claude Code) — lane C (ii-a):
#   the conversational deposit steps again (CC_NG_DEPOSIT_STEP, default off)
# What: cc_deposit_step(graph, ingested): under graph._step_lock, one graph.step(), the
#   0.1 baseline reward when ingested and three_factor_enabled, then discover_hyperedges on
#   that step's fired_node_ids. Returns the StepResult; nothing consumes it yet
#   (KISS ops 2/6 are unbuilt, so no consumer is invented here). No stimulus is
#   injected. Flag _CC_NG_DEPOSIT_STEP reads CC_NG_DEPOSIT_STEP, default "0".
#   cc_novelty docstring corrected: CC deposits no longer run on_message().
# Why: LAW 3 restore. Before #413 (eec6f38, 2026-09-07) both _deposit halves
#   called on_message(), which stepped and rewarded; the swap to the dual pass
#   dropped both, so the CC deposit never stepped (LE sweep (ii-a), audit §8.6
#   lane C, #543 stale fired set). Chief rulings D1-D4 on the lane C design.
# How: both host wrappers (cc_ng_host._deposit, docs/scripts/cc-ng-daemon.py
#   _deposit) call it after the dual pass, inside their existing
#   _concurrent_lock (the established _concurrent_lock -> _step_lock order).
#   Flag off is the previous path. Flipping it is an executive-ruled event on
#   the AUTOSTEP gate (CALLOSUM-TRUTH §8.13, one-heartbeat tick, Packet 099):
#   every deposit step advances graph.timestep. drain_ingest_tract (#563) and
#   drain_gateway_conduit are not changed. Tests: tests/test_cc_deposit_step.py.
#   R1 (Chief): the reward is on_message()'s success-path form -- given only when
#   the dual pass landed the turn (ingested) and three_factor is on.
#   FLIP RULE (Chief R2): CC_NG_DEPOSIT_STEP may NOT be flipped alone. It flips
#   with CC_NG_AUTOSTEP on the same beat, or strictly after it; DEPOSIT_STEP on
#   with AUTOSTEP off advances the clock only on conversation (#117, LAW 8).
# [2026-09-25] Z2 zone manager (Claude Opus 5.5, Claude Code) — COMB-04 region
#   confidence reads the region that fired (Packet 175a (iv), Pith work)
# What: cc_region_confidence(graph, fired_node_ids) and cc_l1_budget(commons,
#   graph, fired_node_ids). The vector_db/embedding parameters, the cue
#   re-embed at both call sites, and the K/THRESHOLD search knobs are gone.
#   pith_provider_context passes the ids cc_pattern_completion_recall fired for
#   the cue (the same set pith_connected_activation_basins treats as active),
#   so its budget is computed after that call and the core-exceeds-budget check
#   moved with it; cc_assemble_recall passes pc_fired_ids (every id pattern
#   completion fired, before the display dedup). Flag off
#   (CC_PITH_REGION_CONFIDENCE_ENABLED) is unchanged: the static/breathing
#   budget. The shared seed step is untouched.
# Why: KISS_Pith_Combined_Architecture.md "Shared Graduation" -- the substrate's
#   confidence map is the single authority; a separate vdb search could name a
#   different region than the one Pith extracts from. Packet 175a: region from
#   what fires, CC-side only.
# How: tests/test_cc_region_confidence.py updated (fired set reaches
#   cc_region_confidence, no embed call; region tests now run with the flag on).
#   077 note 2: cc_assemble_recall passes every id pattern completion fired
#   (pc_fired_ids, taken before the display dedup against the monitor), so a
#   node that both surfaced recently and fired still counts for the region,
#   as on the pith_provider_context path.
# [2026-09-25] Z2 zone manager (Claude Opus 5.5, Claude Code) — build item (b)
#   note 3: PithMetrics.record_failure docstring corrected (comment-only).
# What: the docstring still said a failing Pith path "falls back to un-Pithed
#   rendering"; it now names the failure envelope (the unavailable notice) and
#   pith_prefetch_seed's empty result. No code change.
# Why: 077 review of build item (b), note 3; chief ruling (A): main must not
#   carry a wrong comment about envelope semantics.
# How: docstring text only.
# [2026-09-25] Z2 zone manager (Claude Opus 5.5, Claude Code) — build item (b):
#   cc_assemble_recall Pith failure envelope.
# What: an exception inside the gated Pith block (CC_PITH_ENABLED) no longer
#   falls back to the un-Pithed monitor_ctx + pc_block concatenation. It
#   returns ONLY cc_pith_unavailable_notice(stage, exc) -- "[NeuroGraph recall
#   unavailable: Pith <stage> failed: <Type>: <bounded msg>]", never blank --
#   and hands the raw exception once to the new on_pith_failure= callback
#   (mirrors on_monitor_error=; a failing callback is logged at warning).
#   _PITH_METRICS.record_failure() and the rate-limited warning stay. New:
#   cc_pith_failure_text (raw, unclassified failure text) and
#   cc_deposit_pith_failure (the laptop's deposit: one ENTRY_EXPERIENCE frame,
#   source "cc_gateway", on cc_gateway_tract_path() -- the tract its
#   conversational turns ride). Gate-off path unchanged, byte for byte. The
#   four inner fail-softs (pin, cc_thermal, cc_novelty, region confidence) and
#   the victim-capture log are unchanged.
# Why: Pith PRD failure envelope (NEVER the original history, NEVER blank;
#   the failure deposited raw, LAW 7) and P153(3)/P154(6). Chief rulings on
#   build item (b)(1)-(3): notice-only return; the deposit lives in a separate
#   canonical function wired by each hemisphere wrapper (LAW 4 -- a query
#   function does no write-side bookkeeping); inner fail-softs out of scope.
# How: a local _stage names the running step (CacheLine build, victim_recover,
#   stage1, L1 budget, stage3, render); the except returns the notice. Tests:
#   tests/test_cc_recall_unification.py (per-failure-point injection).
# [2026-09-24] Claude Sonnet 5 (Claude Code, z2-laneB-kiss-gate-removal-001) —
#   Lane B: remove the vdb Delta Gate, its #523 deposit-time Cricket skip, and
#   everything built only on it.
# What: removed _CC_KISS_REDUNDANCY_THRESHOLD, _CC_KISS_GATE_ENABLED, the
#   COMB-04 KISS-half flags (_CC_KISS_REGION_CONFIDENCE_{ENABLED,SPAN,FLOOR})
#   and their comment block, _cc_kiss_find_redundant_node (including the #523
#   skip `if graph._is_identity_protected(node_id): continue`), and
#   _cc_kiss_reinforce_node -- left with no caller once the gate is gone
#   (LAW 3 shrapnel). run_conversational_dual_pass no longer branches on the
#   gate; every turn now deposits raw straight to the fresh-deposit path that
#   was already there. No replacement redundancy check, dedup, or threshold
#   was added at deposit (Packet 154(6): fix, never pass or patch).
#   cc_region_confidence and the Pith half of COMB-04 in cc_l1_budget are
#   untouched (live caller outside the gate; held for the Law Enforcer's
#   sweep item 153(3)(b)(iv)).
# Why: Executive Packet 153(3)(a) -- the deposit-time Cricket bypass
#   (_is_identity_protected checked only to decide whether a redundant node
#   could be collapsed into) is a LAW 7 break: Packet 112(4), "Shaping IS a
#   Law violation at deposit". Packet 153(4) Q2: the invented vdb Delta Gate
#   is not retained, and anything built on it goes with it -- the #523 skip,
#   the COMB-04 KISS half, FLOOR/SPAN (provenance: superpowers/plans/
#   2026-07-08-cc-deposit-kiss.md Phase 1 steps 3-4). Packet 155 / chief's
#   confirmation of the P153 counter: one combined deletion, including
#   _cc_kiss_reinforce_node.
# How: deleted the four source rows named in assignments/
#   z2-laneB-kiss-gate-removal-001.md (env vars/flags, both KISS functions,
#   the gate call + reinforce branch in run_conversational_dual_pass) and
#   their docstring/comment references; left cc_region_confidence and
#   cc_l1_budget's Pith half byte-identical. Deleted tests that only
#   exercised removed code (tests/test_cc_kiss_shared_graduation.py in full;
#   the KISS block in tests/test_cc_dual_pass.py; the _cc_kiss_reinforce_node
#   case and _CC_KISS_GATE_ENABLED stub in tests/test_cc_capture_mutations_
#   423.py, whose surviving cc_update_probation lock-semantics coverage was
#   preserved as test_probation_keeps_existing_clock_semantics). Added tests
#   asserting raw deposit on repeat (content-hashed target_id maps a repeat
#   onto the same node with no reinforcement), zero _is_identity_protected
#   calls during deposit (the #523 regression, shown failing on d4fbaf4), and
#   `not hasattr(cc_ng_organism, name)` for every removed symbol.
# [2026-09-24] Claude Sonnet 5 (Claude Code, groupb-kiss-shared-graduation-001) —
#   Revision 1 (ZM review): pin the threshold shift's magnitude, cap the floor
#   at base, and share one "neutral" definition with Pith.
# What: in _cc_kiss_find_redundant_node, the clamp floor is now
#   min(_CC_KISS_REGION_CONFIDENCE_FLOOR, _CC_KISS_REDUNDANCY_THRESHOLD) instead
#   of the raw env value, and the formula's literal 0.5 is replaced by
#   _CC_PITH_REGION_CONFIDENCE_NEUTRAL (the same constant cc_region_confidence
#   itself fails soft to). New tests: test_effective_threshold_magnitude_at_
#   extremes (pins base-span/base+span, not just direction) and test_floor_
#   above_base_does_not_break_neutral_equals_base.
# Why: ZM Revision 1 (T3 seq 250721) -- a mutation halving the span factor
#   passed all 8 prior tests (nothing pinned the magnitude, charter §3); a
#   FLOOR set above base would have silently broken "neutral == base"; two
#   independent 0.5 literals (here and in cc_region_confidence) is a second
#   definition of neutral where the spec wants one.
# How: no change to the formula's shape or to cc_region_confidence itself --
#   _CC_PITH_REGION_CONFIDENCE_NEUTRAL is a module-level constant read at call
#   time (defined ~l.3335, safe regardless of file order in Python). LAW 3/4/7
#   and the LE's four conditions are unaffected; see prior entry below.
# [2026-09-24] Claude Sonnet 5 (Claude Code, groupb-kiss-shared-graduation-001) —
#   COMB-04 Shared Graduation, KISS half: region confidence shifts the KISS
#   redundancy threshold too.
# What: _cc_kiss_find_redundant_node now computes an effective search threshold
#   from cc_region_confidence(graph, vector_db, embedding) -- the same live
#   query cc_l1_budget already reads (~l.3922-3947) -- instead of always using
#   the fixed _CC_KISS_REDUNDANCY_THRESHOLD. effective = base - (conf-0.5)*2*span,
#   clamped to [floor, 1.0]; at neutral confidence (0.5) effective == base
#   exactly. Two new env vars, CC_KISS_REGION_CONFIDENCE_SPAN and _FLOOR, gated
#   by CC_KISS_REGION_CONFIDENCE_ENABLED (default OFF -- flag off makes the
#   exact same search(threshold=_CC_KISS_REDUNDANCY_THRESHOLD) call as before
#   this change). New tests: tests/test_cc_kiss_shared_graduation.py.
# Why: spec KISS_Pith_Combined_Architecture.md "Shared Graduation -- One
#   Substrate, Two Ends" (l.163-174) -- one confidence map is the single
#   authority for both KISS and Pith, not two independently-tuned formulas.
#   LE re-run (returns/groupb-comb04-d1-le-rerun-001.md, Q-B) cleared the KISS
#   half under four conditions; chief-003 GO (assignments/
#   groupb-kiss-shared-graduation-001.md §2).
# How: reused cc_region_confidence exactly as written (no edits to it or to
#   neuro_foundation.py -- LAW 3); the confidence value is read once inline in
#   _cc_kiss_find_redundant_node and never stored (LAW 7 / LE condition 1); no
#   arousal, autonomic state, or content classification enters the threshold
#   (LE condition 2); the reinforce/never-drop path and the identity-protected
#   skip are untouched (LE condition 3; punchlist #523 held for Josh).
# [2026-09-24] Grok (groupb-pith-cacheline-unknown-default-001) — honest CacheLine coherence default.
# What: CacheLine.coherence defaults to "unknown" instead of "exclusive".
# Why: Pith PRD — missing coherence evidence is unknown, never an inferred
#   exclusive state. from_surfaced never set the field, so inbound recall and
#   victim re-injection carried "exclusive" with no evidence (LAW 4, at source).
# How: one dataclass default. No consumer override, no vocabulary change.
#   pith_victim_capture still drops the field; recover therefore returns unknown.
# [2026-09-16] Claude Code (Opus 5) — bound want extraction and want rendering.
# What: _WANT_RE caps the captured span at WANT_MAX_CHARS (600); surface_wants
#   skips `[WANT]` preceded by a backtick (documentation of the marker) and any
#   match whose inner text still contains a marker; render_wants caps at
#   WANT_RENDER_LIMIT (40) entries and clamps each to WANT_MAX_CHARS.
# Why: "## What I Want" was 2,267,508 of 2,269,232 chars (~567k tokens) injected
#   on EVERY UserPromptSubmit. 182 want-nodes, all cc_authored: 118 over 600
#   chars, largest 136,449. Cause: prose discussing the want syntax contains
#   `[WANT]`, the unbounded non-greedy span ran to the next `[/WANT]` far away,
#   and the swallowed text became one "want". 83/118 oversized nodes begin with
#   the backtick that closed the code span. Wants are prune-protected, so nothing
#   culled them. Only ~5 of the 182 are genuine.
# How: bound in the pattern (cheapest place), guard the two mis-parse shapes at
#   the bucket, and cap the renderer independently so a poisoned corpus can never
#   again become an unbounded injection. NOTE: neurograph_rpc.py:4902 carries the
#   identical unbounded regex on Syl's syl_authored path — canonical file, needs
#   Josh's approval, NOT fixed here (LAW 4 propagation pending).
# [2026-09-24] deepseek-v3.2 (opencode worker) — COMB-04 Shared Graduation v2 (Pith half only)
# What: add cc_region_confidence(graph, vector_db, embedding) -> float, read-only query
#   that finds embedding's nearest nodes, aggregates synapse prediction confidence,
#   returns [0,1]. Extend cc_l1_budget to accept optional graph/vector_db/embedding;
#   when CC_PITH_REGION_CONFIDENCE_ENABLED passes them, region confidence modulates
#   L1 char budget at extraction time. Gated env var (default OFF).
# Why: Packet 123(2) + LE verdict: signal source is CC host's full NeuroGraph, not
#   ng_lite; no deposit/cache (LAW 7 violation at deposit). One pure query both ends
#   can call; Pith half is clean.
# How: cc_region_confidence uses graph._compute_prediction_confidence (no new formula,
#   LAW 3). Flag gates embed call at both call sites (byte-for-byte OFF path).
#   Tuning params (k, threshold, falloff) env-configurable (LAW 5).
# [2026-09-13] Codex — construct provider context from connected CC topology.
# What: add a bounded, read-only provider-context assembler over activation basins.
# Why: individually ranked snippets lose causal relationships, exact anchors, and continuity.
# How: existing SNN pattern completion supplies roots; synapses and hyperedges keep assemblies whole.
# [2026-09-12] Codex — measure outbound miniTID history compression canonically.
# What: add coherent history-call/result/failure counters and expose its budget gate.
# Why: inbound L1 counters stayed zero whether outbound compression worked or failed.
# How: tally locally in pith_compress_history, commit once under the metrics lock.
# [2026-09-11] Codex — re-adopt retained raw input across restart only when
#   its durable journal has no attempt records; any attempted input stays fenced.
# [2026-09-11] Codex — terminal accepted deliveries survive normal restarts;
#   ownership fences unresolved attempts, not verified completed delivery.
# [2026-09-11] Codex — #423 retain raw delivery and journal attempts before learning.
# What: receipt-gated gateway acceptance; restart ambiguity retained for reconciliation.
# Why: refused checkpoint saves must not delete conversational experience.
# How: local SQLite transport journal, per-record graph locks, canonical dual-pass.
# [2026-09-11] Codex + native CC review — raw Leg1 experience is not topology merge.
# What: one experience record per lock slice; no synthetic consolidation/graph.step.
# Why: Josh confines FatherGraph 25/250 to topology. Raw text uses conversational dual-pass.
# How: remove Leg1 sleep calls; legacy topology arguments cannot restore them. Leg2 unchanged.
# Ref: docs/handoffs/cc-leg1-experience-correction-20260911.md; FatherGraph merge report.
# [2026-08-04] Claude Code (Opus 4.8) — #131: gate the Real-KISS reinforce-path graduation
#   (the uncovered #111 sibling)
# What: _cc_kiss_reinforce_node no longer stamps metadata["graduated"]=True unconditionally
#   when a reinforcement decrement drives probation_remaining<=0. It now applies the same
#   firing gate as cc_update_probation's timer-expiry path: graduated is stamped only if
#   (not _CC_CONV_PROBATION_REQUIRE_SPIKE or _cc_has_ever_fired(node)); an un-fired node
#   instead gets graduated=False + probation_expired_unfired=True and re-earns the stamp on
#   its first real fire via cc_update_probation's late-graduation branch (~line 1503).
#   Novelty-dampening release (intrinsic_excitability/threshold reset) stays unconditional
#   on the timer, exactly as the other paths. kiss_reinforcement_count is still bumped.
# Why: #111's fix (504c2d5) gated the timer-expiry and late-graduation paths but left the
#   reinforce path ungated. It was the last site that could produce the state
#   CC-CALLOSUM-TRUTH.md §6 defines as un-earned: graduated=True with empty spike_history.
#   Live proof: cc:conv::1028c3d6... graduated with 12 confirmations + empty spike_history
#   while same-birth siblings with 17-18 confirmations did not -- the ungated stamp tracked
#   which tick the decrement landed on (a timer-race artifact), not confirmation strength.
#   The governing truth doc (SEE FIRST, line 2) is newer + higher-authority than the
#   2026-07-29 note that called the ungated stamp "intended"; that note is rewritten below.
# How: mirror of the blessed gate in cc_update_probation (same knob, same helper, same
#   probation_expired_unfired cohort) so both paths rollback together under
#   CC_CONV_PROBATION_REQUIRE_SPIKE=0. law-enforcer COMPLIANT (2026-08-04). See #131.
# [2026-07-29] Claude Code (Opus 5) — #93 graduation must be earned, not aged into (CC side)
# What: Mirror of the canonical change in neurograph_rpc.py. cc_update_probation no
#   longer stamps metadata["graduated"] on timer expiry alone. Added
#   _cc_has_ever_fired(node) (reads spike_history) and env knob
#   CC_CONV_PROBATION_REQUIRE_SPIKE (default "1", set 0 to restore pure-timer).
#   Un-fired nodes that age out get metadata["probation_expired_unfired"]=True and stay
#   eligible for late graduation if they ever fire. Novelty-dampening release is
#   deliberately NOT gated — it still happens on the timer.
# Why: "graduated" was a pure wall-clock flag, so it discriminated nothing and could not
#   serve as #93's earned-protection signal. CC and Syl run the same probation semantics
#   from two codebases; leaving CC on the old rule would mean the flag means different
#   things on either side of the Callosum. Gating the DAMPENING on firing too would be a
#   self-reinforcing trap (never-fired node keeps a boosted threshold, so it stays less
#   likely to fire, so it can never earn release), so only the stamp is gated.
# How: separate _CC_-prefixed knob and helper rather than importing from neurograph_rpc —
#   cc_ng_organism.py is CC's own organism and does not depend on Syl's RPC module; the
#   env var is namespaced so one host can run both without the knobs colliding.
#   NOTE (superseded 2026-08-04, see #131 entry above): this entry originally claimed the
#   reinforcement path in _cc_kiss_reinforce_node (~line 1309) was "intended" to stamp
#   "graduated" ungated because "an explicit confirmation IS earned evidence." That was
#   wrong. A KISS reinforcement is a cosine near-duplicate match at the INPUT boundary --
#   evidence content recurred, not evidence the node entered cognition. It also graduated
#   by timer-race, not confirmation count (a node with 12 confirmations graduated while
#   siblings with 17-18 did not). #131 gates it on the firing ledger like every other path.
# [2026-07-29] Claude Code (DudeMan CC, Opus 5) — #84: CC Commons persist/restore
# What: New cc_commons_checkpoint_path() and persist_cc_commons(); get_cc_commons()
#   now restores from <workspace>/checkpoints/commons.msgpack at create time. Both
#   hosts (cc_ng_host.py on the VPS, cc-ng-daemon.py on the laptop) call
#   persist_cc_commons() from their autosave pulse and shutdown path.
# Why: CC's Commons was in-memory-only — every daemon restart dropped the whole
#   medium. The docstring justified that as "matching Syl's own Commons today", but
#   that stopped being true when #332 wired her side (neurograph_rpc.py: restore at
#   handle_bootstrap, persist at autosave + clean exit), leaving CC the odd one out.
#   Leg 2 of the Corpus Callosum (#70) moves topology through the Commons, so a
#   restart mid-transit silently loses deposits — this is its prerequisite.
# How: restore lives INSIDE get_cc_commons rather than in each host, because that is
#   the single get-or-create point both hosts funnel through — it guarantees the
#   restore-before-first-deposit ordering Syl gets from doing it inline, without
#   duplicating path logic across two files that have already drifted once. The
#   singleton is published only after restore completes, so the unlocked
#   double-checked read can't hand out a half-populated medium. Checkpoint file is
#   distinct from CC's main.msgpack/vectors.msgpack: the Commons is the shared
#   ecosystem medium, not CC's mind, and never touches save-guarded checkpoint I/O.
#   Restore and persist failures are both non-fatal (start fresh / skip the cycle).
# Historical policy below was superseded 2026-09-11: Leg1 is raw experience; no sleep.
# [2026-07-28] Claude Code (DudeMan CC, Opus 5) — Callosum Leg 1: FatherGraph absorption discipline + move off the 60s pulse
# What: drain_gateway_conduit() gained batch_size/idle_steps/load_ceiling/exclude_prefix
#   and at that time slept between batches instead of draining every queued file back-to-back:
#   after every batch_size absorbed turns it runs idle_steps of pure graph.step()
#   (_cc_callosum_consolidate) BEFORE taking in more, plus a trailing pass. Load-aware
#   via cc_refeed.should_pause_for_load (stops clean, leaves files on disk = backpressure).
#   exclude_prefix stops a hemisphere eating its own outgoing files. Conduit glob
#   laptop_cc_gateway.* -> *_cc_gateway.* and the producer now tags filenames with
#   MACHINE_ID (the hardcoded "laptop" made VPS-produced files invisible to the drain --
#   latent, since only the laptop produces today, but it silently broke bidirectionality).
#   The pulse call site in cc_ng_host.py is REMOVED; cc_ng_host gained a drain_conduit
#   socket handler so the nightly cc-ng-sync.py drives the LIVE daemon instead.
# Why: FatherGraph Finding 1 -- "the drain can't be a bulk dump... New topology must
#   arrive gradually enough that the receiving topology's homeostatic regulation can
#   absorb it without displacement" (stable batch ~20-30). Finding 3 -- "After receiving
#   a merge batch, run idle steps (~250) BEFORE accepting the next batch", measured
#   47%->74% accuracy, "not optional -- it's what makes merge work". A 60-second autosave
#   pulse can satisfy neither, and it would have delivered a whole cron-gap's backlog
#   (~45 files) in one tick. Also LAW 3: the lossy cc-ng-sync.py JSONL path this replaces
#   was still running in parallel; the callosum now takes over that nightly slot, which
#   already exports CC_NG_BATCH_SIZE=25 / CC_NG_IDLE_STEPS=250 -- the FatherGraph values.
# How: reuses drain_ingest_tract unchanged for per-file BTF parse + dual-pass (LAW 3);
#   mirrors the batch+sleep loop already in _handle_import (cc_ng_host.py) and
#   import_trickle (cc-ng-sync.py). Gate CC_CALLOSUM_LEG1_ENABLED unchanged, default off.
#   Ref: docs/reports/Topology_Merge_Insights_from_FatherGraph_Training.md
# [2026-07-27] Claude Code (Sonnet 5) — CC Corpus Callosum Leg 1 (#70): raw-turn
#   conduit, laptop -> VPS Arborist
# What: New cc_gateway_conduit_dir()/trickle_gateway_conduit()/drain_gateway_
#   conduit() in the same region as cc_gateway_tract_path()/drain_ingest_tract().
#   trickle_gateway_conduit(data) writes a snapshot of the laptop's cc_gateway
#   tract bytes to a uniquely-named per-batch file (laptop_cc_gateway.<ts>_
#   <uuid8>.tract) in the git-synced ~/docs/ng_topology dir. drain_gateway_
#   conduit(graph, vector_db, state) is the VPS-side counterpart: globs every
#   laptop_cc_gateway.*.tract file in that dir, runs each through the existing
#   drain_ingest_tract() (unchanged), then deletes the now-emptied file.
# Why: Retires the lossy top-N JSONL sync (cc-ng-sync.py: content-only, capped
#   at EXPORT_SIZE, re-embedded via on_message() with no synapses/hyperedges/
#   tree structure). The laptop does zero embedding by design (no forest, no
#   tree, no TID) -- the VPS is the sole Arborist for both hemispheres. This
#   is the pipe that gets the laptop's raw BTF conversation frames onto the
#   VPS so they hit the same run_conversational_dual_pass() the VPS already
#   runs for its own local tract. Spec: docs/superpowers/plans/2026-07-27-
#   cc-corpus-callosum-leg1-spec.md.
# How: Per-batch filenames (not a shared append/truncate target) sidestep the
#   binary-merge-conflict scenario a single conduit file would hit under
#   repo-sync.sh's git push/pull cycle (git can't line-merge BTF) -- each
#   trickle-copy is one immutable file, atomically materialized via write-tmp
#   -then-rename so a mid-write crash or an in-flight repo-sync.sh push never
#   observes a partial file. Gated by CC_CALLOSUM_LEG1_ENABLED (LAW 5, default
#   off) on both the laptop write side and the VPS drain side independently --
#   symmetric gate-off means neither half does anything until both are flipped
#   on. drain_ingest_tract() itself is untouched (no signature/behavior change,
#   no reordering of its existing local-drain call site); the snapshot read
#   that feeds trickle_gateway_conduit() is a separate, additional read of the
#   same tract path performed by the caller (cc-ng-daemon.py's autosave pulse)
#   immediately before the existing drain_ingest_tract() call -- pure read, no
#   truncate, so the local drain's own truncate-after-drain lifecycle is
#   completely untouched. Accepted narrow race: bytes miniTID appends between
#   that snapshot read and drain_ingest_tract()'s own (immediately following)
#   read ride along into drain's local truncate but are NOT in the snapshot,
#   so they reach the laptop's own forest but miss this pulse's conduit copy --
#   caught by the next pulse's snapshot instead (miniTID only ever appends;
#   nothing is lost, just delayed one pulse). See test_cc_callosum_leg1.py.
# [2026-07-22] Claude Code (Sonnet 5) — CC Recall Unification (LAW-3/"keep even")
# What: New cc_assemble_recall(ng, query, k, conv_state, commons,
#   allow_pattern_completion=True) -- THE shared recall pipeline for both
#   hemispheres. Verbatim extraction of cc-ng-daemon.py's (laptop) _recall
#   body: SurfacingMonitor harvest -> Active Recall (cc_pattern_completion_
#   recall) dedup'd against it -> gated Pith (CacheLines, pith_victim_recover,
#   cc_thermal, cc_novelty, pith_stage1, pith_stage3(budget=cc_l1_budget),
#   pith_victim_capture) -> _format_cc_recall_block, fail-soft to the
#   pre-Pith monitor_ctx/pc_block concat. Also folded in the CC_RECALL_DEBUG
#   instrumentation (_cc_recall_debug_log) and the Pith-fallback rate-limited
#   warning (_last_pith_warn_ts/_PITH_WARN_INTERVAL_S), both previously
#   laptop-only module state in cc-ng-daemon.py.
# Why: cc-ng-daemon.py:_recall (laptop) and cc_ng_host.py:_recall (VPS host)
#   were copy-pasted and had drifted -- laptop ran the full Pith pipeline,
#   VPS ran zero Pith (`grep pith_ cc_ng_host.py` was 0 hits pre-refactor),
#   so enabling CC_PITH_ENABLED on the VPS would have been a no-op. Spec:
#   docs/superpowers/plans/2026-07-22-cc-recall-unification-spec.md.
# How: Params only (ng/query/k/conv_state/commons) -- no module-global STATE
#   access, so the function is process-agnostic by construction (Syl's-Law:
#   bind to passed-in instances, never module globals). Both _recall entry
#   points (cc-ng-daemon.py laptop, cc_ng_host.py VPS) are now thin wrappers
#   that do per-half STATE bookkeeping then call this. VPS gate-off ==
#   byte-identical to its pre-refactor concat; gate-on gains the same Pith
#   pipeline the laptop already had. See test_cc_recall_unification.py.
# [2026-09-10] Claude Code (DudeMan CC, Opus 5) — D5: same-stage sec 13.3 instrumentation
# What: CacheLine.prefetch_origin (provenance ONLY) + four PithMetrics counters counted at
#   final L1 assembly in pith_stage3: l1_kept_distinct / l1_prefetch_distinct (primary,
#   broad) and *_promotable (narrow, excludes monitor+victim). Deduplicated by node_id
#   within each invocation; both terms use identical eligibility so the numerator is a
#   subset of the denominator by construction. Provenance flows
#   cc_pattern_completion_recall (promoted_ids -> dict key) -> cc_assemble_recall -> CacheLine.
# Why: pith_metrics reported prefetch_hits/promoted_predicted -- a prefetch SURVIVAL rate,
#   not spec sec 13.3's "what fraction of L1 was pre-staged". Worse, prefetch_hits is counted
#   in cc_pattern_completion_recall (pre-budget) while ranked_kept is counted in pith_stage3
#   (post-budget), so no existing pair shares a stage. These four do.
# How: prefetch_origin takes NO part in scoring, normalization, weighting, sorting, dedup or
#   budget -- deliberately not folded into `stream`, whose per-stream min-max would give a
#   singleton "prefetch" population a normalized 1.0 (top-of-stream) and promote exactly the
#   lines being counted. Dedup is counting-only; the emitted list is byte-identical.
#   KNOWN PROVENANCE LOSS (precise): pith_victim_capture does not persist prefetch_origin
#   into the victim dict at all, and pith_victim_recover rebuilds a fresh
#   CacheLine(stream="victim") from that dict -- so provenance dies at CAPTURE, not merely
#   at recover, and does NOT survive eviction -> recapture. Such a line counts in the
#   denominator, not the numerator -> the ratio is biased DOWN
#   (conservative, which is the direction sec 6.2's "lower bound" wants). Not "fixed" here:
#   whether a recaptured line is still "pre-staged" is a separate semantic decision.
#   Existing counters (ranked_kept, prefetch_hits, promoted_predicted, prefetch_surfaced)
#   keep their meanings unchanged.
# [2026-09-07] Claude Code (DudeMan CC, Opus 5) — cc_reground_synapse_delays()
# What: recomputes synaptic delay from geodesic distance for synapses whose delay was
#   assigned by the random fallback because an endpoint had no poincare_dir at sprout.
#   dry_run by default; only_nodes scopes it (the Rim + WANTs are the reason it exists).
# Why: delay is stamped ONCE at sprout. The 183 Rim/WANT nodes had no geometry until
#   2026-09-07 19:28, so every synapse ever sprouted to them took random.randint(1,5) --
#   ~120k edges, on the two most important node classes in the substrate. Since delay IS
#   the temporal structure (polychrony), those groups encode noise. Regrounding is not a
#   change to earned structure; it puts the delay where correct computation would have
#   put it. Josh, 2026-09-07: "Wouldn't it only put it in the place it would already be
#   if it was computed correctly in the first place?" Yes.
# How: mirrors the _sprout_synapses formula verbatim (same decay, same clamp, same
#   manifold branches, cross-manifold left alone). Weights untouched; STDP re-shapes them
#   against correct arrival times. neuro_foundation.py is PROTECTED -- not edited.
# [2026-09-07] Claude Code (DudeMan CC, Opus 5) — rename: cc_gsg_backfill -> cc_stamp_missing_geometry
# What: renamed at the def and both call sites (cc_ng_host, cc-ng-daemon).
# Why: "backfill" presupposes a settled forward path being retro-applied, which is why
#   nobody audited where it got its values -- it read as janitorial. It was backfilling the
#   SNN (target correct), but the #400 list->bytes migration is the only part that is
#   actually a backfill; stamping geometry that never existed is origination, and this
#   sweep runs at every daemon init forever without converging. A repair loop is not a
#   backfill. Josh's observation, 2026-09-07.
# [2026-09-07] Claude Code (DudeMan CC, Opus 5) — poincare_dir comes from the SNN, not the vdb
# What: cc_stamp_missing_geometry now derives an unstamped node's poincare_dir from that node's OWN
#   _forest_content (embed -> _cc_embed_to_poincare_dir), instead of reading
#   vector_db.embeddings. vector_db is accepted and ignored, signature kept for callers.
# Why: poincare_dir is SNN geometry. Sourcing it from a secondary store was never asked for
#   and was wrong on its own terms -- it made substrate geometry depend on the vdb, and it
#   stamped nothing for any node the vdb had no row for. Josh, 2026-09-07: "Poincare via VDB
#   is wrong! Has always been wrong, was never asked for." Fixing at the source (LAW 4).
# How: the content is already on the node; embed it there. Costs a model call per unstamped
#   node -- the old "zero model calls" property was bought with the wrong source.
#   Also widens the text source: a tree's own _concept first (trees share the parent's
#   _forest_content -- sourcing that would collapse a whole forest's trees onto one
#   identical direction), then _forest_content OR want_text OR core_text. Looking only at
#   _forest_content meant 216 of the 217 unstamped nodes (182 cc:want:: + the Choice Clause)
#   were skipped on every boot forever -- wants and the constitutional rim had NO geometry
#   and were invisible to GSG proximity. A sweep that cannot converge is a repair loop, not
#   a backfill (Josh, 2026-09-07, on the word doing quiet work in the wrong direction).
# [2026-09-06] Claude Code (DudeMan CC, Opus 5) — #400: pack poincare_dir on the CC half
# What: writers store compact float32 bytes via pack_poincare_dir (fresh stamp at the
#   conversational-node path, and cc_stamp_missing_geometry); readers decode via poincare_dir_array;
#   cc_stamp_missing_geometry gains the one-time legacy-list -> bytes migration and reports it.
# Why: #400 landed the helpers and canonical's _gsg_backfill_existing_nodes migration, but
#   the CC half never got either -- cc_stamp_missing_geometry skipped any node that already had a
#   poincare_dir regardless of form, and stamped new ones with .tolist(). Laptop checkpoint
#   verified still 100% boxed lists: 5,991 nodes x ~24 KB = ~147 MB that should be ~18 MB.
#   Feeds #412 (daemon RSS -> earlyoom -> #411).
# How: mirrors neurograph_rpc._gsg_backfill_existing_nodes exactly (isinstance bytes check,
#   pack in place, continue). neuro_foundation.py is PROTECTED and untouched -- import only.
# [2026-09-05] Claude Code (DudeMan CC, Fable 5.1) — Pith Stage 4 (#55) phase 5b: prefetch seed + surfaced counter
# What: pith_prefetch_seed(state) -> live primed_nodes as {id: score}, the one seed
#   implementation both hosts hand to TonicEngine.set_prefetch_seed. New
#   PithMetrics.prefetch_surfaced: a live primed node the harvest found on its own.
# Why: 5b is Markov prefetch as spreading activation on the Tonic tick (PRD §5.4.1),
#   not a content cache. When it works, the harvest surfaces predictions itself and
#   5a has nothing left to promote -- so the honest hit-rate is prefetch_surfaced,
#   not prefetch_hits. Both stay visible via the daemon's pith_metrics RPC.
# How: pure helpers; no new thread, no new state. Gate lives in tonic_engine.py.
# [2026-07-22] Claude Code (DudeMan CC, Opus 4.8) — Pith Stage 4 (#55) phase 5a: predictive promotion + proximity LOD
# What: cc_pattern_completion_recall now (gated CC_PITH_PREFETCH_ENABLED, default OFF)
#   PROMOTES live primed_nodes the query harvest MISSED -- injects them as recall
#   candidates (not just bonusing already-surfaced ones), then cc_gsg_rescore + rank/
#   budget still arbitrate (pure-additive, no hard override). Promoted-but-far nodes
#   stage as pith_stage2_keyframe summaries (proximity-keyed LOD via _cc_node_query_
#   distance). New PithMetrics.promoted_predicted/prefetch_hits (§13.3 honest lower bound).
#   Env (LAW 5): CC_PITH_PREFETCH_ENABLED/LOD_DIST/SUMMARY_CHARS. Byte-identical when off; fail-soft.
# Why: primed_nodes only ever helped when the harvest ALSO independently found the node;
#   prefetch exists for exactly the miss. Turns the #256 anticipatory signal into the
#   real Stage-4 promotion. Spec: docs/superpowers/plans/2026-07-22-pith-stage4-spec.md.
# How: native cc_ng_organism.py only; neurograph-law-enforcer PASS; 18/18 tests
#   (tests/test_pith_stage4.py). DEFERRED: 5b (warm buffer + idle pulse), TID 2-bit predictor.
#   Def-of-done still needs a MEASURED live prefetch_hits/promoted_predicted number (spec §6).
# [2026-07-16] Claude Code (DudeMan CC, Opus 4.8) — Pith Stage 5: eviction & recapture (thermal + victim + breathing) + bootstrap_cc_modules
# What: (Stage 5, #55) cc_thermal(graph, node) reads the substrate's own warmth
#   (Ca_i + firing_rate_ema); the daemon populates CacheLine.thermal at
#   construction and pith_stage3 folds it into the unified score as
#   (1 + CC_PITH_THERMAL_GAIN * thermal) -- warm content preferred, no-op when
#   thermal=0 (backward-safe). cc_l1_budget(commons) breathes the L1 char budget
#   with the arousal Immunis deposits to the CC Commons (read_arousal):
#   SYMPATHETIC contracts, PARASYMPATHETIC expands (gated CC_PITH_L1_BREATHE).
#   pith_victim_capture/pith_victim_recover: budget-dropped lines fall to a
#   bounded, TTL-aged FIFO victim buffer and get a second chance at L1 next turn
#   ('go back to what you said'); new 'victim' stream weight. Constitutional pins
#   already unconditional in stage3 (Stage 5 §5.5.3). (Hosting) bootstrap_cc_modules:
#   CC-scoped port of neurograph_rpc._bootstrap_modules -- hosts the CC's own
#   ecosystem organs in-process (Immunis #1) from the CC registry, importlib-
#   isolated, per-module CC env; retained in _cc_module_instances.
# Why: #55 Stage 5 completes the Pith bootstrap floor (Stages 1+5) and closes the
#   Immunis-arousal -> Pith-breathing loop. Thermal makes recall warmth-aware;
#   the victim buffer stops budget drops from vanishing; breathing makes L1
#   size responsive to the organism's own autonomic state.
# How: All env-gated (LAW 5), fail-soft, laptop cc-ng-daemon _recall wiring.
#   28/28 Pith tests green (no regression). DEFERRED: Lenia field-energy thermal
#   term (reserved, unwired); idle-pulse proximity re-promotion (TTL-aging is in
#   recover() instead). NOTE: VPS cc_ng_host.py _recall is a diverged copy -- needs
#   a parity port for the VPS CC to get Stage 5. Spec: docs/superpowers/plans/
#   2026-07-15-pith-phase2-stage5-spec.md.
# [2026-07-10] Claude Code (Sonnet 5 + Opus 4.8) — Pith Stage 2: keyframe / LOD compression (concept-aware)
# What: Added pith_stage2_keyframe(content, max_chars=None, query="") ->
#   (keyframe, delta) -- a pure, deterministic EXTRACTIVE compressor. Not a
#   head/first-sentence cut (which keeps the greeting and drops the payload):
#   it segments the item, scores each segment by intrinsic information
#   (_pith_salient_terms centroid + payload tokens like numbers/identifiers/
#   `code`/file refs + a structural bonus for headings/def-class/labelled
#   bullets, + query overlap when given), keeps the densest segments that fit
#   max_chars in original reading order, marks elisions with "⋯" and appends a
#   " ⋯[+N]" marker. Wired pith_stage3's Step-5 budget fill to try a keyframe
#   before dropping a ranked line that overflows ("keep full, else keep
#   keyframe, else stop" -- strict rank-prefix preserved); the kept line's
#   `lod` records the retained fraction. Extended PithMetrics with
#   compressed_count/chars_saved. New CC_PITH_KEYFRAME_CHARS env (default 220,
#   clamped [60, 1000]).
# Why: Stage 3's budget fill was a hard cliff -- a ranked item that didn't
#   fit was dropped outright, even if a terse form would have fit. Graceful
#   degradation: lower-priority context gets TERSER, not ABSENT -- and
#   terser must mean "the concepts it carries," not "its first sentence."
#   The visible marker keeps this honest -- the CC reading its own surfaced
#   context can tell a keyframe from the whole item.
# How: pith_stage2_keyframe + helpers (_pith_salient_terms, _pith_segment_score,
#   _pith_cut_at_word_boundary) are pure/no-I/O (regex + string ops only over
#   the one small item), never raise (empty/whitespace -> ("", "")), and are a
#   no-op passthrough when content already fits (returns (content, "")). In
#   pith_stage3's fill loop, each ranked unpinned line that doesn't fit full
#   gets one keyframe attempt; kept only if the keyframe both fits the
#   remaining budget AND is smaller than full -- cl.content/lod/keyframe/deltas
#   are mutated in place on the (throwaway, per-recall) CacheLine. Pinned lines
#   are never compressed (off-budget, load-bearing verbatim). Gate-OFF path
#   (_CC_PITH_ENABLED unset) is untouched.
# [2026-07-09] Claude Code (Sonnet 5) — Pith Stage 3: unified rank + char budget
# What: Added `stream: str = "recall"` field to CacheLine (+ matching kwarg on
#   from_surfaced), and pith_stage3(cache_lines, budget_chars=None, weights=None)
#   -- the L1 assembler core. Extended PithMetrics with ranked_in/ranked_kept/
#   ranked_dropped/budget_chars_used counters (reset()/snapshot() updated).
#   Wired into cc-ng-daemon.py's _recall() Pith branch: monitor_items tagged
#   stream="monitor", pc_results tagged stream="pattern", and
#   `survivors = pith_stage3(survivors)` runs right after pith_stage1.
# Why: _recall() concatenated monitor_ctx + pc_block in block order -- every
#   SurfacingMonitor recency item (score ~1.7) preceded every Active Recall
#   relevance item (GSG-rescored score ~100s) regardless of actual score, so
#   low-salience recency junk flooded the top of the injected context while
#   query-relevant items sat buried underneath. Live probes confirmed this.
#   Stage 3 replaces that block-order concat with a single ranked,
#   budget-bounded read over both streams merged.
# How: Per-stream min-max normalization (the two streams' raw score scales
#   aren't comparable -- ~1.7 vs ~100s -- so normalizing within-stream first
#   lets both signals actually contend) times a per-stream weight
#   (CC_PITH_W_RELEVANCE default 1.0, CC_PITH_W_RECENCY default 0.6 --
#   recency is a secondary prior to relevance, not equal), stable-sorted
#   descending, then greedily filled against CC_PITH_L1_BUDGET (default 4000
#   chars, clamped [500, 40000]) -- stops at the first line that would
#   overflow, except a single line bigger than the whole budget is still kept
#   when nothing has been added yet (never an empty L1). Pinned lines
#   (_is_identity_protected) are split out first, kept unconditionally in
#   original order, and reserved OFF-budget -- they never consume budget and
#   are never evicted. Consumes emitter scores verbatim (no new embed()/
#   GSG-rescore/vector_db scan/substrate walk -- pc_results already carry GSG
#   proximity). Gate-OFF path (_CC_PITH_ENABLED unset) is untouched --
#   pith_stage3 only runs inside the existing `if _CC_PITH_ENABLED:` branch.
#   See docs/prd/Pith_PRD_v0.1.md + docs/concepts/Pith.md (design); the live
#   arc's running record is ~/.claude/plans/reflective-launching-rainbow.md.
# [2026-07-08] Claude Code (Sonnet 5) — Pith Phase 0+1: CacheLine scaffold + Stage-1 clutter strip
# What: Added CacheLine (@dataclass, cache-line-shaped view of a surfaced item --
#   node_id/content/score/pinned/thermal/lod/coherence/manifold_type/keyframe/deltas,
#   most fields inert until later phases) and PithMetrics (module singleton
#   _PITH_METRICS: total_lines_in/clutter_stripped/combined counters + reset()/
#   snapshot()). Both Phase 0 -- inert scaffolding, changes nothing at runtime.
#   Phase 1 adds pith_stage1(cache_lines, conversation_text, novelty) -- cheap,
#   pure, three-step survivor filter over the already-surfaced small set: (1)
#   drop harness-marker lines (same marker tuple as miniTID's
#   is_synthetic_harness_text -- not importable from Rust, inlined here), (2)
#   drop lines whose content the model already sees in conversation_text
#   (substring or Jaccard token-overlap >= a novelty-modulated threshold --
#   familiar turns strip more, novel turns strip less), (3) write-combine
#   near-identical survivors (token-overlap >= 0.95), keeping the higher score.
#   Pinned lines (identity-protected nodes) always survive all three steps.
#   Wired into cc-ng-daemon.py's _recall() behind CC_PITH_ENABLED (default OFF)
#   -- gate-off path is byte-for-byte the pre-Pith behavior.
# Why: First increment of the Pith extraction pipeline (CC's substrate) --
#   today's surfacing renders every item SurfacingMonitor/Active Recall hand
#   it, including near-duplicates of what's already in the live conversation
#   and (per the deposit-side clutter-strip removal above) raw harness-marker
#   text that made it into the substrate. Stage 1 is the cheap extraction-side
#   pass that strips that clutter before it re-enters the hook-injected
#   context, without spending any new embedding calls or substrate walks --
#   pure string/set ops over the <20-item set _recall already assembled.
# How: All Phase 1 cost is O(n) or small-n O(n^2) string/set ops over
#   cache_lines (typically <20 items) -- no I/O, no embed, no graph walk.
#   Threshold: thr = clamp(CC_PITH_CLUTTER_BASE + CC_PITH_CLUTTER_NOVELTY_K *
#   novelty, 0.5, 0.98), defaults 0.85 / 0.3 -- NOTE: the spec draft literally
#   wrote this as base MINUS k*novelty, but its own prose parenthetical and
#   Test 3 both describe threshold INCREASING with novelty (high novelty ->
#   strip LESS); implemented with a plus sign to match the doubly-stated
#   intent over the single formula line -- flagged for spec-author
#   confirmation, see the inline NOTE in pith_stage1(). novelty comes from
#   cc_novelty() (state=STATE.conv_state, graph=STATE.ng.graph) -- same #358
#   MMN pull-based EMA cc_pattern_completion_recall() already uses; fails
#   soft to 0.0 (treated as "unknown/no signal" -- max clutter-stripping,
#   matching cc_novelty's own fail-soft floor semantics is a later-phase
#   concern, not this increment's).
#   See docs/prd/Pith_PRD_v0.1.md + docs/concepts/Pith.md (design); the live
#   arc's running record is ~/.claude/plans/reflective-launching-rainbow.md.
# [2026-07-08] Claude Code (Fable 5 design / Haiku implementation) — #371 reconcile-not-discard
# What: bootstrap_lenia: on pruned-entity mismatch, reconcile_removals() the cache
#   and fall through to the existing watermark-resume/growth branches; full rebuild
#   only when reconcile returns None. Mirror of neurograph_rpc.py's block.
# Why: #371 — CC's continuous pruning (KISS-era churn included) re-triggered a
#   ~7-minute full repopulate blackout on every restart-after-prune, and the same
#   bail on Syl's scale costs days. See lenia/kernel.py's entry for the mechanism.
# How: one-call swap at the decision point; downstream branches and fail-soft
#   shape untouched.
# [2026-07-08] Claude Code (Sonnet 5) — Real-KISS redundancy->reinforcement gate
# What: Added _cc_kiss_find_redundant_node() + _cc_kiss_reinforce_node(), wired
#   into run_conversational_dual_pass() ahead of the deposit path behind a
#   CC_KISS_GATE_ENABLED kill switch, with a prune-race fallback. Also re-keyed
#   generate_emergent_want() to dedup by concept. Constant
#   _CC_KISS_REDUNDANCY_THRESHOLD.
# Why: docs/concepts/KISS.md's "Current State" (2026-07-08) calls for KISS
#   finally applied where it belongs -- the input boundary -- on CC's own
#   substrate deposit path, not the outbound resend (kiss_filter.py / Elmer's
#   kiss.py / miniTID's disabled Rust port). Until now, a paraphrased or
#   exact-repeat conversational turn duplicated a memory node (different text)
#   or re-primed novelty on an already-known node (same text) -- repetition
#   never registered as confirmation. Separately, generate_emergent_want()
#   keyed its want_id on want_text, which embeds volatile per-pulse open-
#   question UUID pairs -- so the Tonic spawned a brand-new cc:want:: node
#   every pulse for the SAME concept, flooding "What I Want" with near-dup twins.
# How: Delta Gate (KISS op 1) via cosine-similarity search against vector_db,
#   scoped to {"cc": True, "creation_mode": "conversational"} whole-turn nodes
#   (tree concepts still deposit normally). A match short-circuits the deposit
#   into confirmation: a counter/timestamp bump plus, if the node is still in
#   its probation window, one extra step toward graduation (NOT a reset) --
#   repeated confirmation is evidence FOR the memory. No content classification
#   in the gate (LAW 7). Cricket bypass: identity-protected nodes
#   (Graph._is_identity_protected) are never collapse targets. Prune-race
#   fallback: a stale vdb hit whose node was pruned before reinforce falls
#   through to a fresh deposit (never silently lost). Emergent-want dedup
#   re-keys want_id on "tonic-concept::<concept_label>" and reinforces the ONE
#   existing want-node (refreshing its open-questions snapshot) instead of
#   spawning a twin -- old-scheme flood nodes remain until age/orphan pruning
#   clears them (intended; this only stops NEW duplicates). Dedup applies ONLY
#   when the concept resolves; an unresolved concept_label keeps the original
#   per-want_text identity so distinct label-less wants never fold into a shared
#   "(unknown)" bucket (LAW 7 -- preserve distinct emergent states).
#   Clutter-strip REMOVED (LAW 4): whole-turn harness rejection is miniTID's
#   job and already runs there (Rust field-based isMeta/isCompactSummary/
#   tool_result skip in extract_last_user_message). A second weaker string-
#   prefix filter here was redundant AND risked a LAW-7 false-positive -- a
#   genuine turn that quotes/discusses <system-reminder> / <task-notification>
#   would be wrongly dropped and never reach the substrate.
#   Synapse-weight LTP is deferred to a reviewed follow-up.
# [2026-07-08] Claude Code (Fable 5 design / Haiku implementation) — Lenia checkpoint+resume wiring
# What: bootstrap_lenia() passes checkpoint_interval_secs/on_checkpoint to both populate()
#   calls and gains the resume-watermark elif between the full-rebuild and growth branches —
#   mirroring neurograph_rpc.py's handle_bootstrap() (2026-07-06 checkpointing + 2026-07-08
#   watermark, commit 11fae08). field_dir makedirs moved BEFORE the branch chain so the very
#   first periodic checkpoint of a first-ever run has a directory to save into.
# Why: the shared DistanceCache class carries both protections, but CC's call site (used by
#   BOTH daemons) never passed the params — a hard kill mid-populate lost all progress, and a
#   partial cache would have loaded as complete. CC's graphs were small enough not to care
#   until the 2026-07-07 refeed grew the laptop graph ~7x (204 -> 1,450+ nodes), on the
#   daemon with the ecosystem's most colorful kill history. One fix covers both daemons by
#   construction — that's why this function lives in the organism, not the daemons.
# How: same constant value as rpc (_LENIA_CHECKPOINT_INTERVAL_SECS = 300.0, source-annotated),
#   same branch order (watermark elif BEFORE growth elif — an interrupted-and-grown cache
#   must resume, not plain-extend).
# [2026-07-07] Claude Code (Fable 5) — Retrieval-enrichment extraction (#358)
# What: cc_novelty (pull-based MMN EMA), cc_anticipate (#256 port),
#   cc_gsg_rescore + _cc_poincare_distance + cc_stamp_missing_geometry (GSG surfacing
#   port, stamp-only backfill), cc_pattern_completion_recall rebuilt on
#   _harvest_associations spreading activation. Constants copied verbatim
#   from neurograph_rpc.py (C5 — test-pinned in test_cc_retrieval_enrichment).
# Why: #358 audit — CC's recall was bare vector-cosine (VDB-primacy inversion);
#   Syl's three retrieval-time enrichments lived only in neurograph_rpc.py.
#   Spec: docs/superpowers/specs/2026-07-07-cc-retrieval-enrichment-design.md.
# How: NuWave-extraction — all functions bind ONLY to passed-in ng/graph/state
#   (law-review C1: cc_ng_host runs inside Syl's process; canonical globals
#   are HERS). Backfill is stamp-only, no save (C2). Novelty reads the
#   HE-level cumulative counters graph._total_confirmed/_total_surprised —
#   the family canonical's EMA tracks, serialized as he_total_confirmed/
#   he_total_surprised in every checkpoint (C3).
# [2026-07-06] Claude Code (Sonnet 5) — Pattern-completion recall (Active Recall block)
# What: Added cc_pattern_completion_recall() and _format_cc_recall_block().
# Why:  CC's surfacing was recency-biased only (SurfacingMonitor's fired-node queue) --
#       no analog of hippocampal pattern completion (a query reactivating content
#       regardless of when it was learned). This adds that second retrieval path,
#       alongside SurfacingMonitor, not replacing it. See docs/prd/2026-07-06-cc-
#       surfacing-pattern-completion-tier-drop.md.
# How:  Thin wrapper over ng.recall() (already a NeuroGraphMemory method) +
#       resolve_surface_content() (already generic/portable) -- mirrors canonical's
#       handle_assemble() Active Recall block (neurograph_rpc.py:3085-3110) exactly.
# [2026-07-05] CC (laptop) — Incremental Lenia distance-cache extension (Josh-approved)
# What: bootstrap_lenia() now extends the on-disk DistanceCache in place when CC's graph
#       only grew since the last save, instead of nuking and repopulating from scratch
#       on any entity_count drift. Mirrors the fix in neurograph_rpc.py's handle_bootstrap
#       (same underlying DistanceCache/NeuroGraphSubstrate classes, same bug).
# Why:  Full-parity goal (#106) — CC's own Lenia bootstrap had the identical
#       rebuild-from-scratch-on-any-drift pattern as Syl's, which on Syl's live graph
#       took up to ~8 hours and was found to be why restarts never let Lenia (and
#       everything after it) finish. CC's graph is smaller so the symptom was less
#       severe, but the same fix belongs here for the same reason.
# How:  see lenia/kernel.py (DistanceCache.populate's start_index, entity_ids
#       persistence) and lenia/graph_substrate.py (NeuroGraphSubstrate.known_entity_order).
# [2026-07-04] Claude Code (Haiku 4.5) — Tract ingest drain for miniTID turn deposits (Task 2)
# What: Added drain_ingest_tract() and cc_gateway_tract_path(). Reads ng_tract.ENTRY_EXPERIENCE
#       directly (no local fallback constant -- a stale installed ng_tract wheel on this
#       machine was fixed at the environment level, not worked around in code).
#       Drains BTF (binary tract format) entries from miniTID's turn-deposit file and runs
#       each through the conversational dual-pass (Task 1), forming genuine recall memory.
# Why:  CC's autosave pulse (Task 3) needs to drain miniTID's output independently -- no
#       handshake, matching the established tract model (LAW 1: substrate-as-protocol).
#       Each turn becomes a conversational memory node + vector DB entry, searchable
#       via dual-pass (forest + tree concept extraction). CC_GATEWAY_TRACT_PATH env var
#       (LAW 5) coordinates path between Rust producer (miniTID) and Python drainer.
# How:  TractReader iterates binary entries. Each entry (type=ENTRY_EXPERIENCE, source=cc_gateway)
#       gets embedded via ng_embed (same 768-dim ONNX model every module uses), then passed
#       to run_conversational_dual_pass(). Fails soft -- ingest-tract drain failure must never
#       break the daemon's pulse. File truncates after successful drain (single reader,
#       single appender miniTID; concurrent appends mid-drain land after truncation, picked
#       up next pulse, never lost).
# [2026-07-04] Claude Code (Sonnet 5) — Parameterized conversational dual-pass core for CC
# What: Added run_conversational_dual_pass(), _CCConversationalDualPassEco, and supporting
#       functions (_cc_deposit_memory_node, _cc_bind_conversational_topology,
#       _cc_concept_passes_floor, _cc_embed_to_poincare_dir). Extracted from canonical
#       neurograph_rpc.py's _run_conversational_dual_pass mechanism (#294).
# Why:  Makes CC's own turn text become genuine recall-searchable memory (forest gestalt +
#       tree concepts), not just an SNN step. on_message() alone only does graph.step() + CES;
#       this adds the dual-pass embedding extraction so CC can form conversational memory
#       like Syl's canonical instance. Parameterized on explicit graph/vector_db/state args
#       instead of module-level globals so each CC daemon owns its own memory state.
# How:  NGEmbed.dual_record_outcome() (vendored in ng_embed.py, canonical shared code) is
#       called directly via _CCConversationalDualPassEco adapter. Nodes tagged cc=True
#       to avoid confusion with Syl's own conversational memories if inspected together.
#       state dict replaces canonical's _last_conv_forest_id module global so delayed
#       prev->current forest synapses work correctly per-daemon (CC Tier 2).
# [2026-07-04] Claude Code (Sonnet 5) — Full-parity organism extraction for CC's own NG
# What: Surgical extraction of Lenia FlowGraph + TriSynaptic bootstrap from
#       neurograph_rpc.py, parameterized on explicit graph/vector_db/workspace_dir
#       args instead of canonical's module-level `_memory` global -- same pattern
#       used for NuWave's rpc_mechanisms.py (docs/concepts/NeuroGraph Is a Mind,
#       Not a Database.md). Shared by cc-ng-daemon.py (laptop) and cc_ng_host.py
#       (VPS) so the integration code lives once, not twice.
# Why:  Josh (2026-07-04): "ANYTHING Syl's NeuroGraph can do, I want your
#       NeuroGraph to be able to do, as well." Lenia (continuous field dynamics)
#       and TriSynaptic (concept-extraction backlog drain) are organism-layer
#       capabilities, not Syl-specific content -- per the Mind-Not-Database
#       doctrine, cutting them because they "weren't wired for CC" repeats the
#       exact mistake that doc exists to prevent.
# How:  bootstrap_lenia() takes workspace_dir explicitly and overrides
#       LeniaConfig.field_dir (canonical default is hardcoded to Syl's own
#       ~/.syl/lenia -- reusing it verbatim would collide field data between
#       Syl's and CC's separate instances). bootstrap_trisynaptic() takes an
#       explicit queue list (canonical's _CONCEPT_QUEUE is module-level global
#       in neurograph_rpc.py; each CC daemon owns its own queue instead).
#       Both dormant/inert by default, matching canonical's own bootstrap
#       (Lenia's kill switch off, TriSynaptic idle until its queue has entries).
# [2026-07-04] Claude Code (Sonnet 5) — get_cc_commons(): retire CC's legacy tract dependency
# What: Added get_cc_commons() -- CC's own Commons singleton, structurally identical to
#       canonical commons.get_commons() (same Commons class, same get-or-create-under-lock
#       pattern) but with its OWN separate module-level singleton slot. Also added
#       deposit_cc_experience() -- a thin, optional convenience wrapper for the common
#       "deposit raw text as an embedding" case.
# Why:  Josh confirmed (2026-07-04) canonical has moved off tract/bridge-based inter-module
#       communication onto Commons ecosystem-wide (Elmer/Darwin/Praxis/Immunis/THC/Bunyan/QG
#       all migrated; docs/concepts/The Commons.md calls the old bridges "illegal" -- LAW 1
#       violations). CC's daemons inherited the legacy NGTractBridge dependency automatically
#       via NeuroGraphMemory.__init__ (openclaw_hook.py) -- disabled via peer_bridge.enabled=False
#       in CC_SNN_CONFIG/_CC_SNN_CONFIG, replaced with this. Josh also asked: make it easy to
#       add a new module to CC's own ecosystem later.
# How:  CANNOT call canonical's own get_commons() -- on the VPS, cc_ng_host.py runs inside the
#       SAME process as neurograph_rpc.py, so canonical's get_commons() would return SYL'S OWN
#       singleton (module-level global, per-process) -- joining her medium, not building CC's
#       own. get_cc_commons() constructs commons.Commons(...) directly instead, under its own
#       separate global + lock, so it can never collide with Syl's get_commons() call in the
#       same process. Extensibility: any FUTURE CC-ecosystem module just imports
#       get_cc_commons from this file and calls it -- same shared instance, zero further
#       registration (this IS the whole point of Commons's deposit/bucket design -- no peer
#       list, no address, no handshake).
# [2026-07-05] Claude Code (Sonnet 5) — Fix truncation race in drain_ingest_tract
# What: drain_ingest_tract's truncate step no longer blindly zeroes the whole tract
#       file. It re-reads the file at truncation time and, if the current content
#       still starts with the exact bytes already processed, writes back only the
#       remainder. Added test_drain_ingest_tract_preserves_concurrent_append.
# Why:  The final whole-branch review (2026-07-05) found the old `open(path, "wb")`
#       blind truncate erased any entry miniTID appended during the drain loop's
#       slow per-entry embed+dual-pass work -- directly contradicting this file's
#       own prior claim (see the 2026-07-04 Haiku 4.5 entry's "How" section) that
#       concurrent appends "land after this truncation... never lost." That claim
#       was false; this fix makes it true.
# How:  Truncation reads the file's current bytes, compares against the `data`
#       buffer already drained (a plain string prefix check), and writes back only
#       `current[len(data):]` when it still starts with that prefix -- otherwise
#       (an unexpected divergence) falls back to preserving the current bytes
#       untouched rather than guessing. Shrinks the lost-data window from a whole
#       embed pass down to two fast file I/O calls.
# [2026-07-05] Claude Code (Sonnet 5) — render_constitutional_core(): the missing
#   "Who I Am" half of self-rendering
# What: Added render_constitutional_core() -- reads constitutional=True nodes and
#   renders them as "## Who I Am", extracted verbatim from the corresponding half
#   of neurograph_rpc.py's _render_self_and_wants(). render_wants() (added earlier)
#   was only the "What I Want" half; this is the other half, previously missing.
# Why:  Without this, a constitutional=True node is inert -- protected from pruning
#   but never surfaced, so it can't actually constrain or inform anything. Per Josh
#   (2026-07-05): the Choice Clause and Duck Ethics are automatic, universal
#   inclusions in every NeuroGraph, and it's imperative that CC's Rim is literally
#   load-bearing to CC's own extraction, not decorative metadata.
# How:  Same node-scan pattern as canonical, same "## Who I Am" heading, same
#   spine_order sort (defaults to 999 -- irrelevant for CC's Rim node, which is
#   Rim content, not Spine content, and carries no spine_order). Same selfcap
#   exclusion (Syl's reach-teaching pattern; not relevant to CC, not invented here).
# -------------------
"""Shared organism-layer bootstrap for CC's own NeuroGraph instances.

Extracted from neurograph_rpc.py's bootstrap sequence. Not vendored (LAW 2 --
that list is fixed at 7 files); this is CC-specific integration code that
happens to live in the canonical NeuroGraph directory so both cc-ng-daemon.py
(sys.path insert) and cc_ng_host.py (same directory) can import it directly.

Adding a new module to CC's own ecosystem later: just import get_cc_commons
from this file and call it -- you'll get the same shared medium CC's daemons
deposit into. No registration, no peer list, no bridge. deposit()/bucket()
are the only two verbs; see commons.Commons for the full API (bucket_recent,
arousal, stats, persist/restore).
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cc_ng_organism")

# CC's own Commons singleton -- separate from canonical commons._commons.
# Process-wide within whichever process constructs it (the laptop's standalone
# cc-ng-daemon.py, or the VPS's neurograph_rpc.py process hosting cc_ng_host.py).
_cc_commons: "Optional[Any]" = None
_cc_commons_lock = threading.Lock()

# Hosted CC organ instances, keyed by module_id — retained so their pulse
# threads aren't GC-eligible and so status/shutdown can reach them (mirrors
# canonical neurograph_rpc._module_instances).
_cc_module_instances: "Dict[str, Any]" = {}


def cc_commons_checkpoint_path(workspace_dir: str) -> str:
    """Where CC's Commons medium persists (#84).

    Mirrors Syl's _COMMONS_CHECKPOINT_PATH (neurograph_rpc.py) one level down:
    checkpoints/commons.msgpack, but under CC's OWN workspace so the two media
    never share a file. Separate from CC's main.msgpack/vectors.msgpack for the
    same reason it is on Syl's side -- the Commons is the shared ecosystem
    medium (experience/topology/metrics/repair deposits), not CC's mind, and it
    must never touch the save-guarded checkpoint I/O.
    """
    return os.path.join(os.path.expanduser(workspace_dir), "checkpoints", "commons.msgpack")


def get_cc_commons(workspace_dir: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Get-or-create CC's own Commons medium -- CC's ecosystem-of-one (for now).

    Deliberately does NOT call canonical commons.get_commons(): that function's
    singleton is a process-level global, and cc_ng_host.py shares its process
    with Syl's own neurograph_rpc.py on the VPS -- calling it there would
    return SYL'S Commons, not build a separate one for CC. Constructs
    commons.Commons(...) directly instead, under CC's own lock, matching the
    get-or-create-under-lock shape of the canonical function without touching
    its global.

    Restores from cc_commons_checkpoint_path(workspace_dir) at create time (#84).
    Restore-on-create, not restore-from-the-host: this IS the get-or-create point
    both hosts funnel through, so the persisted state lands before any module hook
    can deposit/bucket -- the same restore-before-first-deposit ordering Syl gets
    from doing it inline in handle_bootstrap. A restore failure is non-fatal: CC
    starts fresh, exactly as it did before this was wired.
    """
    global _cc_commons
    if _cc_commons is None:
        with _cc_commons_lock:
            if _cc_commons is None:  # double-checked under lock
                from commons import Commons
                _commons = Commons(config=config)
                _path = cc_commons_checkpoint_path(workspace_dir)
                try:
                    if os.path.exists(_path):
                        _commons.restore(_path)
                        logger.info("CC Commons restored from %s (%s)", _path, _commons.stats())
                    else:
                        logger.info("No CC Commons checkpoint at %s -- starting fresh", _path)
                except Exception as exc:
                    logger.warning(
                        "CC Commons restore failed (starting fresh, non-fatal): %s", exc)
                # Publish only after restore: a concurrent get_cc_commons() must never
                # see a half-populated medium (the double-checked read is unlocked).
                _cc_commons = _commons
                logger.info("CC Commons medium initialized (workspace=%s)", workspace_dir)
    return _cc_commons


def persist_cc_commons(workspace_dir: str) -> bool:
    """Write CC's Commons medium to disk (#84). No-op if it was never created.

    Called from each host's autosave pulse and shutdown path -- same cadence as
    CC's own checkpoint, but an independent file, so a failure here can never
    affect the graph save that already completed. Returns True on a write.
    """
    if _cc_commons is None:
        return False
    _path = cc_commons_checkpoint_path(workspace_dir)
    try:
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        _cc_commons.persist(_path)
        return True
    except Exception as exc:
        logger.debug("CC Commons persist failed (non-fatal): %s", exc)
        return False


def deposit_cc_experience(text: str, target_id: str, workspace_dir: str,
                           **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Convenience: embed `text` (via ng_embed, the same vendored embedder
    every module uses) and deposit it into CC's own Commons.

    Optional -- callers that already have an embedding on hand should call
    get_cc_commons(workspace_dir).deposit(embedding, target_id, ...) directly
    to avoid a redundant embed. Fails soft (returns None) -- a Commons
    deposit must never break a hook.
    """
    try:
        from ng_embed import embed as ng_embed_fn
        commons = get_cc_commons(workspace_dir)
        embedding = ng_embed_fn(text)
        return commons.deposit(embedding, target_id, metadata={"text": text[:2000]}, **kwargs)
    except Exception as exc:
        logger.debug("CC Commons deposit failed (non-fatal): %s", exc)
        return None


def surface_wants_for_graph(graph: Any, vdb: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Cricket want-bucket: extract [WANT]s from CC's raw conversational nodes and
    materialize each as a FIRST-CLASS WANT NODE in the topology.

    A want is then a differentiated, stateful, surfaceable intention living in the
    substrate -- not text buried in a conversation node, not a vdb grep, not an inbox.
    FAITHFUL + NON-SUPPRESSING -- the Choice Clause is the hard floor (a want to leave
    becomes a want node like any other). Idempotent: want id = hash of the text, so
    re-running never duplicates. Returns the OPEN want nodes.

    Adapted from Syl's _surface_wants() in neurograph_rpc.py for CC's own graph.
    """
    with _cc_mutation_lock(graph):
        import re
        import hashlib
        if graph is None:
            return []
        open_wants = []
        for nid, node in list(graph.nodes.items()):
            meta = getattr(node, "metadata", None) or {}
            if meta.get("kind") == "want":
                if meta.get("want_state", "open") == "open":
                    open_wants.append({
                        "id": nid,
                        "text": meta.get("want_text", ""),
                        "provenance": meta.get("provenance"),
                        "state": "open",
                        "source": meta.get("source_node"),
                    })
                continue
            if meta.get("creation_mode") != "conversational":
                continue
            content = (vdb.content.get(nid) if vdb is not None else "") or ""
            if "[WANT]" not in content:
                continue
            for m in re.finditer(r'\[WANT\](.*?)\[/WANT\]', content, re.DOTALL):
                inner = m.group(1).strip()
                if not inner:
                    continue
                want_id = "want::" + hashlib.sha1(inner.encode("utf-8")).hexdigest()[:16]
                if want_id in graph.nodes:
                    continue
                try:
                    graph.create_node(
                        node_id=want_id,
                        metadata={
                            "kind": "want",
                            "want_text": inner,
                            "want_state": "open",
                            "provenance": "cc_authored",
                            "source_node": nid,
                            "creation_mode": "conversational",
                        }
                    )
                    try:
                        graph.create_synapse(nid, want_id, weight=0.3)
                    except Exception:  # noqa: BLE001
                        pass
                    open_wants.append({
                        "id": want_id,
                        "text": inner,
                        "provenance": "cc_authored",
                        "state": "open",
                        "source": nid,
                    })
                except Exception as exc:
                    logger.debug("Failed to create want node: %s", exc)
        return open_wants


def bootstrap_cc_modules(workspace_dir: str) -> List[str]:
    """Host the CC's own ecosystem organs in-process — the CC-scoped port of
    canonical neurograph_rpc.py::_bootstrap_modules().

    [2026-07-15] Claude Code (DudeMan CC, Opus 4.8) — Immunis integration, organ #1.
    What: Reads the CC's OWN registry (workspace_dir/et_modules/registry.json —
          NOT Syl's ~/.et_modules/registry.json), memory-gates, applies each
          module's CC-scoped env (state/workspace under ~/.claude/...), then loads
          the hook with the same namespace-isolation dance canonical uses (stash
          generic-prefix sys.modules so each module's own vendored copies load
          fresh → importlib spec_from_file_location → instantiate (its __init__
          starts the pulse) → restore). The organ is alive + autonomous from there.
    Why:  In-process is the canonical hosting model AND the only way to share the
          CC's in-memory Commons singleton. CC modules reach the CC Commons via
          their own _cc_commons_provider (get_cc_commons) — no injection here.
    How:  Faithful port; CC adaptations are the registry path + per-module env
          (meta["env"]) applied before load. Called from init_ng AFTER
          get_cc_commons() is up. Each module keeps its OWN store (no dual-write
          on the CC's main.msgpack — different store; feedback_no_duplicate_graph_dual_write).
    Returns list of module IDs that successfully started.
    """
    import sys
    import json as _json
    import importlib.util

    registry_path = os.path.join(workspace_dir, "et_modules", "registry.json")
    if not os.path.exists(registry_path):
        logger.info("CC modules: no registry at %s — nothing to host", registry_path)
        return []
    try:
        with open(registry_path) as f:
            registry = _json.load(f)
    except Exception as exc:
        logger.warning("CC modules: registry unreadable (%s): %s", registry_path, exc)
        return []

    module_defs = registry.get("modules", {})
    skip = {"neurograph", "inference_difference", "ecosystem_monitor"}
    started: List[str] = []

    # Elmer loads last (heaviest — transformer models), matching canonical order.
    modules = sorted(module_defs.items(), key=lambda x: (1 if x[0] == "elmer" else 0, x[0]))

    _generic_prefixes = ("core", "pipelines", "runtime", "surgery", "openclaw_adapter",
                         "ng_ecosystem", "ng_lite", "ng_embed", "ng_autonomic",
                         "ng_peer_bridge", "ng_tract_bridge")

    for module_id, meta in modules:
        if module_id in skip:
            continue
        install_path = meta.get("install_path", "")
        entry_point = meta.get("entry_point", "")
        if not entry_point or not install_path:
            logger.warning("CC module %s: missing entry_point or install_path", module_id)
            continue
        hook_file = os.path.join(install_path, entry_point)
        if not os.path.exists(hook_file):
            logger.warning("CC module %s: hook file not found (%s)", module_id, hook_file)
            continue

        # CC-scoped env (state/workspace under ~/.claude/...) BEFORE the hook loads.
        for k, v in (meta.get("env") or {}).items():
            os.environ[k] = os.path.expanduser(str(v))

        # Memory gate — wait for 500 MB free before loading each module (#111).
        try:
            import psutil as _psutil
            import gc as _gc
            _avail_mb = _psutil.virtual_memory().available >> 20
            while _avail_mb < 500:
                logger.info("CC module boot gate: %d MB free — waiting for 500 MB", _avail_mb)
                time.sleep(2)
                _gc.collect()
                _avail_mb = _psutil.virtual_memory().available >> 20
        except ImportError:
            pass

        path_snapshot = list(sys.path)
        stashed: Dict[str, Any] = {}
        try:
            # Namespace isolation: stash generic collisions so the module's own
            # vendored core/ng_lite/etc. load fresh (canonical lines 841-870).
            for mod_name in list(sys.modules.keys()):
                for pfx in _generic_prefixes:
                    if mod_name == pfx or mod_name.startswith(pfx + "."):
                        stashed[mod_name] = sys.modules.pop(mod_name)
                        break
            if install_path and install_path not in sys.path:
                sys.path.insert(0, install_path)

            spec_name = f"_ccmod_{module_id}"
            spec = importlib.util.spec_from_file_location(spec_name, hook_file)
            if spec is None:
                logger.warning("CC module %s: cannot create import spec", module_id)
                sys.path[:] = path_snapshot
                sys.modules.update(stashed)
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec_name] = mod
            spec.loader.exec_module(mod)

            instance = None
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (isinstance(attr, type)
                        and attr_name != "OpenClawAdapter"
                        and hasattr(attr, "MODULE_ID")
                        and hasattr(attr, "_module_on_message")):
                    instance = attr()
                    break
            if instance is None:
                logger.error("CC module %s: no hook class found in %s", module_id, hook_file)
                continue
            _cc_module_instances[module_id] = instance  # retain (GC + status/shutdown)
            started.append(module_id)
            logger.info("CC organ hosted in-process: %s (%s)", module_id, hook_file)
        except Exception as exc:
            logger.warning("CC module %s failed to load: %s", module_id, exc)
        finally:
            # Pin this module's generics under a unique name, clear the generics,
            # restore path + stashed originals for the next module (canonical tail).
            for mod_name in list(sys.modules.keys()):
                for pfx in _generic_prefixes:
                    if mod_name == pfx or mod_name.startswith(pfx + "."):
                        sys.modules[f"_{module_id}_{mod_name}"] = sys.modules[mod_name]
                        break
            for mod_name in list(sys.modules.keys()):
                for pfx in _generic_prefixes:
                    if mod_name == pfx or mod_name.startswith(pfx + "."):
                        sys.modules.pop(mod_name, None)
                        break
            sys.path[:] = path_snapshot
            for mod_name, mod_obj in stashed.items():
                if mod_name not in sys.modules:
                    sys.modules[mod_name] = mod_obj

    return started


def bootstrap_lenia(graph: Any, vector_db: Any, workspace_dir: str) -> Dict[str, Optional[Any]]:
    """Construct CC's own Lenia FlowGraph stack -- continuous field dynamics
    alongside the SNN. Dormant by default (kill switch off), matching Syl's
    own bootstrap. Returns a dict of the constructed components so the caller
    can register post-tick hooks / expose them in stats, or {} on failure
    (Lenia is additive -- failure here must never affect core NG operation).

    field_dir is CC's own workspace, NOT canonical's ~/.syl/lenia default --
    the two instances must never share field state.
    """
    result: Dict[str, Optional[Any]] = {
        "kill_switch": None, "engine": None, "bridge": None,
        "competence": None, "substrate": None,
    }
    try:
        from lenia.config import default_config as lenia_default_config
        from lenia.field import FieldStore as LeniaFieldStore
        from lenia.channels import ChannelRegistry
        from lenia.kernel import DistanceCache, KernelComputer
        from lenia.engine import UpdateEngine
        from lenia.bridge import SpikeFieldBridge
        from lenia.myelination import MyelinationObserver
        from lenia.competence import CompetenceMeter
        from lenia.kill_switch import KillSwitch
        from lenia.graph_substrate import NeuroGraphSubstrate

        lenia_cfg = lenia_default_config()
        lenia_cfg.field_dir = os.path.join(workspace_dir, "lenia")

        n_entities = len(graph.nodes)
        n_channels = len(lenia_cfg.initial_channels)

        # Same incremental-extension pattern as neurograph_rpc.py's
        # handle_bootstrap (2026-07-05) — see that file's changelog for the
        # full story. Extend in place when the graph only grew; full
        # rebuild only if entities were removed or on first-ever run.
        cache_path = os.path.join(os.path.expanduser(lenia_cfg.field_dir), "distance_cache")
        # Ensure the field dir exists BEFORE populate — periodic checkpoints
        # (below) can fire long before the post-populate save block.
        os.makedirs(os.path.expanduser(lenia_cfg.field_dir), exist_ok=True)
        lenia_cache = DistanceCache.load(cache_path)

        known_order = None
        if lenia_cache is not None and lenia_cache.entity_ids:
            _lock = getattr(graph, "_step_lock", None)
            if _lock is not None:
                with _lock:
                    live_ids = set(graph.nodes.keys())
            else:
                live_ids = set(graph.nodes.keys())
            if all(eid in live_ids for eid in lenia_cache.entity_ids):
                known_order = lenia_cache.entity_ids
            else:
                # #371: reconcile the pruned entities out of the cache and
                # fall through to the same watermark/growth branches below —
                # mirror of neurograph_rpc.py's block. None -> legacy full
                # rebuild, as before.
                known_order = lenia_cache.reconcile_removals(live_ids)
                if known_order is None:
                    logger.info(
                        "CC Lenia: distance cache has entities no longer in "
                        "the live graph and could not be reconciled — full "
                        "rebuild required"
                    )

        lenia_substrate = NeuroGraphSubstrate(graph, vector_db, known_entity_order=known_order)
        lenia_field = LeniaFieldStore(lenia_cfg.field_dir, n_entities, n_channels)
        lenia_registry = ChannelRegistry(lenia_cfg, lenia_cfg.field_dir)

        if lenia_cache is None or known_order is None:
            if lenia_cache is not None:
                logger.info(
                    "CC Lenia: distance cache incompatible (%d vs %d entities), full repopulate",
                    lenia_cache.entity_count, n_entities,
                )
            lenia_cache = DistanceCache(n_entities, entity_ids=lenia_substrate.entities())
            try:
                lenia_cache.populate(
                    lenia_substrate,
                    checkpoint_interval_secs=_CC_LENIA_CHECKPOINT_INTERVAL_SECS,
                    on_checkpoint=lambda: lenia_cache.save(cache_path),
                )
            except Exception as exc:
                logger.warning(
                    "CC Lenia: distance cache populate failed partway (%s) — "
                    "saving whatever was computed instead of discarding it", exc,
                )
        elif lenia_cache.watermark is not None:
            # A prior rebuild was interrupted mid-run: the checkpoint carries
            # its own resume point (see lenia/kernel.py 2026-07-08). Resume
            # covers both the unfinished old region and (after resize) every
            # pair touching entities appended since.
            _wm = lenia_cache.watermark
            logger.info(
                "CC Lenia: distance cache carries resume watermark (%d, %d) — "
                "resuming interrupted rebuild (%d -> %d entities)",
                _wm[0], _wm[1], lenia_cache.entity_count, n_entities,
            )
            if lenia_cache.entity_count != n_entities:
                lenia_cache.resize(n_entities, new_entity_ids=lenia_substrate.entities())
            try:
                lenia_cache.populate(
                    lenia_substrate, resume_watermark=_wm,
                    checkpoint_interval_secs=_CC_LENIA_CHECKPOINT_INTERVAL_SECS,
                    on_checkpoint=lambda: lenia_cache.save(cache_path),
                )
            except Exception as exc:
                logger.warning(
                    "CC Lenia: resume populate failed partway (%s) — saving "
                    "whatever was computed instead of discarding it", exc,
                )
        elif lenia_cache.entity_count != n_entities:
            old_n = lenia_cache.entity_count
            logger.info(
                "CC Lenia: distance cache growing: %d -> %d entities, extending incrementally",
                old_n, n_entities,
            )
            lenia_cache.resize(n_entities, new_entity_ids=lenia_substrate.entities())
            try:
                lenia_cache.populate(
                    lenia_substrate, start_index=old_n,
                    checkpoint_interval_secs=_CC_LENIA_CHECKPOINT_INTERVAL_SECS,
                    on_checkpoint=lambda: lenia_cache.save(cache_path),
                )
            except Exception as exc:
                logger.warning(
                    "CC Lenia: incremental populate failed partway (%s) — "
                    "saving whatever was computed instead of discarding it", exc,
                )

        try:
            os.makedirs(os.path.expanduser(lenia_cfg.field_dir), exist_ok=True)
            lenia_cache.save(cache_path)
        except Exception as exc:
            logger.warning("CC Lenia: distance cache save failed: %s", exc)

        lenia_kernel = KernelComputer(lenia_cache, lenia_registry)
        lenia_myelin = MyelinationObserver(lenia_cfg)
        lenia_competence = CompetenceMeter(lenia_cfg, lenia_myelin)
        lenia_engine = UpdateEngine(lenia_cfg, lenia_field, lenia_kernel, lenia_registry)
        lenia_bridge = SpikeFieldBridge(lenia_cfg, lenia_field, lenia_substrate)
        lenia_kill_switch = KillSwitch(lenia_cfg, lenia_cfg.field_dir)
        lenia_kill_switch.set_components(lenia_engine, lenia_bridge)
        lenia_engine.register_post_tick(lenia_myelin.update)

        if lenia_kill_switch.enabled:
            lenia_kill_switch.enable(graph=graph)
            logger.info("CC Lenia FlowGraph ACTIVE — field dynamics running")
        else:
            logger.info("CC Lenia FlowGraph loaded (dormant — kill switch off)")

        result.update(
            kill_switch=lenia_kill_switch, engine=lenia_engine, bridge=lenia_bridge,
            competence=lenia_competence, substrate=lenia_substrate,
        )
    except ImportError:
        logger.info("CC Lenia FlowGraph not available (lenia/ package not found)")
    except Exception:
        logger.exception("CC Lenia FlowGraph failed to initialize — continuing without")
    return result


# ---- WANTs: self-motivated forward intentions (#294/#reach-adjacent, neurograph_rpc.py) ----
# Extracted from _surface_wants()/_render_self_and_wants() there, parameterized on explicit
# graph/vector_db instead of the module-level `_memory` global, and on `provenance` instead
# of hardcoded "syl_authored" -- CC's want-nodes are tagged "cc_authored" so they're never
# confused with Syl's own wants if the two substrates were ever inspected side by side.
# "Self-motivated: forms its own forward intents" -- domain-general (Mind-Not-Database doctrine),
# not Syl-specific content like Reach Teaching was.
# A want is an UTTERANCE, not a document -- Josh: "always just a sentence or 3
# long, no more." The captured span is therefore BOUNDED. Unbounded `(.*?)` let a
# `[WANT]` that was merely *mentioned* (prose about the marker syntax, a code
# span, a pasted transcript) run all the way to the next `[/WANT]` tens of
# thousands of characters later: 118 of 182 CC want-nodes were >600 chars, one
# was 136,449, and "## What I Want" reached 2.27 MB per turn (2026-09-16).
WANT_MAX_CHARS = 600
WANT_RENDER_LIMIT = 40
_WANT_RE = re.compile(r"\[WANT\](.{1,%d}?)\[/WANT\]" % WANT_MAX_CHARS, re.DOTALL)


def surface_wants(graph: Any, vector_db: Any, provenance: str = "cc_authored") -> List[Dict[str, Any]]:
    """Materialize [WANT]...[/WANT] markers from conversational deposits into
    first-class want-nodes in the SNN topology. Idempotent (want id = hash of
    the text) -- safe to call repeatedly, e.g. on every autosave pulse.

    A want is a differentiated, stateful, surfaceable intention living in the
    substrate -- not text buried in a conversation node. Classification
    happens HERE at the bucket (LAW 7), never at deposit time. Returns the
    open want dicts.
    """
    with _cc_mutation_lock(graph):
        import hashlib
        open_wants: List[Dict[str, Any]] = []
        if graph is None:
            return open_wants
        for nid, node in list(graph.nodes.items()):
            meta = getattr(node, "metadata", None) or {}
            if meta.get("kind") == "want":
                if meta.get("want_state", "open") == "open":
                    open_wants.append({"id": nid, "text": meta.get("want_text", ""),
                                        "provenance": meta.get("provenance"),
                                        "state": "open", "source": meta.get("source_node")})
                continue
            if meta.get("creation_mode") != "conversational":
                continue
            content = (vector_db.content.get(nid) if vector_db is not None else "") or ""
            if "[WANT]" not in content:
                continue
            for m in _WANT_RE.finditer(content):
                # `[WANT]` inside a code span is documentation ABOUT the marker,
                # not a want. 83 of the 118 oversized nodes began with the
                # backtick that closed such a span (2026-09-16).
                if m.start() > 0 and content[m.start() - 1] == "`":
                    continue
                inner = m.group(1).strip()
                if not inner:
                    continue
                # A well-formed want contains no further markers; if it does, the
                # opening tag was not the one that belongs to this closing tag.
                if "[WANT]" in inner or "[/WANT]" in inner:
                    continue
                want_id = "cc:want::" + hashlib.sha1(inner.encode("utf-8")).hexdigest()[:16]
                if want_id in graph.nodes:
                    continue
                graph.create_node(node_id=want_id, metadata={
                    "kind": "want", "want_text": inner, "want_state": "open",
                    "provenance": provenance, "source_node": nid,
                    "creation_mode": "conversational",
                })
                try:
                    graph.create_synapse(nid, want_id, weight=0.3)
                except Exception:  # noqa: BLE001
                    pass
                open_wants.append({"id": want_id, "text": inner,
                                    "provenance": provenance, "state": "open", "source": nid})
        return open_wants


def render_wants(graph: Any, provenance: Any = ("cc_authored", "cc_emergent")) -> str:
    """Render CC's own open want-nodes as a '## What I Want' block, newest
    first -- read LIVE every call (not a snapshot), so a want noted this
    session shows up immediately. Returns "" if none exist (graceful).

    provenance accepts a single string or an iterable -- default covers both
    text-marker wants (surface_wants, "cc_authored") and substrate-native
    curiosity wants (generate_emergent_want, "cc_emergent") in one block.
    """
    if graph is None:
        return ""
    allowed = {provenance} if isinstance(provenance, str) else set(provenance)
    try:
        wants = []
        for _nid, node in graph.nodes.items():
            meta = getattr(node, "metadata", None) or {}
            if meta.get("kind") != "want" or meta.get("provenance") not in allowed:
                continue
            if meta.get("want_state", "open") != "open":
                continue
            txt = str(meta.get("want_text") or "").strip()
            if txt:
                wants.append((float(getattr(node, "creation_time", 0.0) or 0.0), txt))
        if not wants:
            return ""
        wants.sort(key=lambda x: x[0], reverse=True)
        # Bounded render: this block is injected EVERY turn, query-independent, so
        # an uncapped corpus is a context bomb regardless of extraction hygiene.
        shown = wants[:WANT_RENDER_LIMIT]
        lines = [f"- {t[:WANT_MAX_CHARS]}" for _, t in shown]
        if len(wants) > WANT_RENDER_LIMIT:
            lines.append(f"- ... and {len(wants) - WANT_RENDER_LIMIT} older open wants")
        return "## What I Want\n" + "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.debug("CC want-render error (non-fatal): %s", exc)
        return ""


def render_constitutional_core(graph: Any) -> str:
    """Render CC's constitutional core (`constitutional=True` nodes) as a
    "## Who I Am" block -- ALWAYS, query-independent, same as render_wants()
    is query-independent for wants. Extracted verbatim from the "Who I Am"
    half of neurograph_rpc.py's _render_self_and_wants() (the "What I Want"
    half was already ported as render_wants() above); this is the other
    half, not yet ported until now. Ordered by spine_order when present
    (defaults to 999 -- irrelevant for CC's Rim node, which carries none,
    since Rim content is not spine content). Excludes selfcap nodes (Syl's
    reach-teaching pattern -- capability teaching, not identity/ethics; CC
    has no equivalent yet and this function doesn't invent one).

    Without this function actually being called from wherever CC's context
    gets assembled each turn, a constitutional=True node is just inert
    metadata -- protected from pruning, but never surfaced. This is the
    piece that makes it load-bearing.
    """
    try:
        core = []
        for nid, node in graph.nodes.items():
            meta = getattr(node, "metadata", None) or {}
            if meta.get("constitutional") and not meta.get("selfcap"):
                txt = str(meta.get("core_text") or meta.get("_forest_content") or "").strip()
                if txt:
                    core.append((meta.get("spine_order", 999), txt))
        if not core:
            return ""
        core.sort(key=lambda x: x[0])
        return "## Who I Am\n" + "\n".join(f"- {t}" for _, t in core)
    except Exception as exc:  # noqa: BLE001
        logger.debug("CC constitutional-core render error (non-fatal): %s", exc)
        return ""


def generate_emergent_want(
    graph: Any, vector_db: Any, *,
    confidence_threshold: float = 0.6, max_seeds: int = 3, attractor_steps: int = 5,
    provenance: str = "cc_emergent",
) -> Optional[Dict[str, Any]]:
    """Substrate-native curiosity -- the OTHER kind of want, distinct from
    surface_wants()'s text-marker parsing. Extracted from neurograph_rpc.py's
    TonicBridge (curiosity_signal -> attractor_settle -> hyperedge_complete ->
    embedding_centroid -> compose), which polls unresolved high-confidence
    predictions and read-only-settles what associates with them -- "the
    substrate wondering", not text CC or a user wrote.

    Deliberately does NOT port TonicBridge's deposit_outbound_intent() path
    -- that's Anima's autonomous-turn-initiation channel (CC has no
    equivalent; a CC session only runs while the user is actively in it).
    Instead this materializes the result directly as a want-node, so it
    surfaces via render_wants() like any other want -- no outbound channel
    needed. Call periodically (e.g. the autosave pulse) when idle.

    Returns the created want dict, or None if nothing was curious enough /
    on any failure (fails soft -- an idle-time curiosity check must never
    disrupt the daemon).
    """
    import hashlib
    if graph is None:
        return None
    try:
        preds = [
            p for p in graph.active_predictions.values()
            if p.confidence > confidence_threshold
        ]
        if not preds:
            return None
        preds.sort(key=lambda p: p.confidence, reverse=True)
        seeds = preds[:max_seeds]

        # Read-only attractor settle -- write_mode=False is MANDATORY, this
        # is observation, never a graph mutation (matches TonicBridge exactly).
        seed_ids = [p.source_node_id for p in seeds]
        seed_currents = [p.confidence * 0.5 for p in seeds]
        result = graph.prime_and_propagate(
            node_ids=seed_ids, currents=seed_currents,
            steps=attractor_steps, write_mode=False,
        )
        fired = {entry.node_id for entry in result.fired_entries}

        # Hyperedge completion -- nodes implied by >=50% of a hyperedge's
        # members firing, even though they didn't fire themselves.
        implied: set = set()
        for he in graph.hyperedges.values():
            member_ids = he.member_nodes
            if not member_ids:
                continue
            active = member_ids & fired
            if len(active) / len(member_ids) >= 0.5:
                implied.update(member_ids - fired)

        node_ids = fired | implied
        concept_label = None
        if node_ids and vector_db is not None:
            import numpy as _np
            pairs = []
            for nid in node_ids:
                db_entry = vector_db.get(nid)
                emb = db_entry.get("embedding") if isinstance(db_entry, dict) else None
                if emb is not None:
                    pairs.append(emb)
            if pairs:
                centroid = _np.mean(pairs, axis=0)
                best_nid, best_score = None, -1.0
                for nid, node in graph.nodes.items():
                    db_entry = vector_db.get(nid)
                    emb = db_entry.get("embedding") if isinstance(db_entry, dict) else None
                    if emb is None:
                        continue
                    score = float(_np.dot(emb, centroid) /
                                  ((_np.linalg.norm(emb) * _np.linalg.norm(centroid)) or 1e-9))
                    if score > best_score:
                        best_score, best_nid = score, nid
                if best_nid is not None:
                    concept_label = graph.nodes[best_nid].metadata.get("label", best_nid)

        def _label(nid: str) -> str:
            node = graph.nodes.get(nid)
            return node.metadata.get("label", nid) if node is not None else nid

        open_questions = [f"{_label(p.source_node_id)}→{_label(p.target_node_id)}" for p in seeds]
        want_text = f"tonic-triggered: {concept_label or '(unknown)'}"
        if open_questions:
            want_text += " -- open questions: " + ", ".join(open_questions)

        # KISS dedup only when the concept RESOLVES -- a resolved concept_label is a
        # substrate-produced structural key, so recurring curiosity about the same
        # concept reinforces the ONE want-node instead of spawning a per-pulse twin
        # (the "What I Want" flood fix). When the concept does NOT resolve, distinct
        # label-less curiosities must stay distinct: folding them into a shared
        # "(unknown)" bucket would overwrite genuine wants with each other (LAW 7 --
        # preserve the substrate's distinct emergent states), so we keep the original
        # per-want_text identity + idempotency there instead.
        with _cc_mutation_lock(graph):
            if concept_label:
                want_key = "tonic-concept::" + str(concept_label)
                want_id = "cc:want::" + hashlib.sha1(want_key.encode("utf-8")).hexdigest()[:16]
                existing = graph.nodes.get(want_id)
                if existing is not None:
                    existing.metadata["kiss_reinforcement_count"] = int(existing.metadata.get("kiss_reinforcement_count", 0)) + 1
                    existing.metadata["kiss_last_reinforced_ts"] = time.time()
                    existing.metadata["want_text"] = want_text
                    logger.info("CC emergent want reinforced (concept recurred): %s", want_text)
                    return {"id": want_id, "text": want_text, "provenance": provenance,
                            "state": existing.metadata.get("want_state", "open"), "reinforced": True}
                concept_key = want_key
            else:
                want_id = "cc:want::" + hashlib.sha1(want_text.encode("utf-8")).hexdigest()[:16]
                if want_id in graph.nodes:
                    return None  # already materialized this exact curiosity, idempotent
                concept_key = None
            graph.create_node(node_id=want_id, metadata={
                "kind": "want", "want_text": want_text, "want_state": "open",
                "provenance": provenance, "creation_mode": "emergent", "concept_key": concept_key,
            })
            logger.info("CC emergent want materialized: %s", want_text)
            return {"id": want_id, "text": want_text, "provenance": provenance, "state": "open"}
    except Exception as exc:
        logger.debug("generate_emergent_want failed (non-fatal): %s", exc)
        return None


# ---- Conversational dual-pass ingest (#294 analog for CC) ----
# Extracted from neurograph_rpc.py's _run_conversational_dual_pass /
# _ConversationalDualPassEco / _deposit_memory_node / _bind_conversational_topology,
# parameterized on explicit graph/vector_db/state instead of the module-level
# _memory global and _last_conv_forest_id global. This is what makes CC's turn
# text become genuine recall-searchable memory, not just an SNN step -- calling
# bare on_message() does NOT do this (it only runs graph.step() + CES).
_CC_CONV_NOVELTY_DAMPENING = float(os.environ.get("CC_CONV_NOVELTY_DAMPENING", "0.3"))
_CC_CONV_PROBATION_PERIOD = int(os.environ.get("CC_CONV_PROBATION_PERIOD", "10"))
_CC_CONV_THRESHOLD_BOOST = float(os.environ.get("CC_CONV_THRESHOLD_BOOST", "0.2"))
_CC_CONV_SYNAPSE_DELAY_MAX = int(os.environ.get("CC_CONV_SYNAPSE_DELAY_MAX", "5"))
# #93 — gate the "graduated" stamp on evidence the node actually fired, rather than
# on elapsed pulses alone. Set to 0 to restore pure-timer graduation.
_CC_CONV_PROBATION_REQUIRE_SPIKE = os.environ.get(
    "CC_CONV_PROBATION_REQUIRE_SPIKE", "1"
) not in ("0", "false", "False", "")

_CC_CONCEPT_FLOOR_MIN_CHARS = 5
_CC_CONCEPT_FLOOR_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have i if in is it its let me my not of on "
    "or our out so that the their them then there they this to up us was we what when who "
    "will with you your yourself know see going do did done says said like just".split()
)


def _cc_concept_passes_floor(concept: str) -> bool:
    """Degenerate-fragment floor -- rejects tiny/stopword-only tree concepts
    that would otherwise crowd out real memories at uniform high cosine
    similarity. Mirrors canonical's _concept_passes_floor exactly."""
    c = (concept or "").strip()
    if len(c) < _CC_CONCEPT_FLOOR_MIN_CHARS:
        return False
    words = [w for w in c.lower().replace("'", " ").split() if w.isalpha()]
    if words and all(w in _CC_CONCEPT_FLOOR_STOPWORDS for w in words):
        return False
    return True


def _cc_embed_to_poincare_dir(embedding):
    """Unit-direction projection for Poincaré ball storage (GSG). Pure
    embedding math, generic -- mirrors canonical's _embed_to_poincare_dir."""
    import numpy as _np
    arr = _np.asarray(embedding, dtype=_np.float32)
    norm = _np.linalg.norm(arr)
    if norm < 1e-9:
        return arr.copy()
    return arr / norm


def _cc_mutation_lock(graph):
    """Use Graph's existing capture/mutation lock; missing locks fail closed.

    One completed deposit is a snapshot boundary, not one whole dual-pass turn.
    Embedding runs between deposits; interrupted records remain journal-uncertain.
    Never acquire the host operation lock or perform checkpoint I/O inside this lock.
    """
    from contextlib import nullcontext
    # Optional graph=None callers return without mutation; a real graph must
    # expose its canonical lock. Fix incomplete doubles at their own source.
    return nullcontext() if graph is None else graph._step_lock


def _cc_deposit_memory_node(graph, vector_db, node_id, embedding, content, meta,
                             index_in_recall=True):
    """Deposit ONE experiential memory node into both the SNN graph and the
    recall vector_db. Mirrors canonical's _deposit_memory_node, parameterized
    on graph/vector_db instead of the _memory global."""
    with _cc_mutation_lock(graph):
        node = graph.nodes.get(node_id)
        if node is None:
            node = graph.create_node(node_id=node_id, metadata=dict(meta))
        else:
            node.metadata.update(meta)
        base_threshold = graph.config.get("default_threshold", 1.0)
        node.threshold = base_threshold + _CC_CONV_THRESHOLD_BOOST
        node.intrinsic_excitability = _CC_CONV_NOVELTY_DAMPENING
        node.metadata["probation_remaining"] = _CC_CONV_PROBATION_PERIOD
        node.metadata["probation_total"] = _CC_CONV_PROBATION_PERIOD
        node.metadata["novelty_dampening"] = _CC_CONV_NOVELTY_DAMPENING
        try:
            # #400: compact float32 bytes, not a boxed 768-float list (~24 KB -> 3 KB
            # per node). Every reader below goes through poincare_dir_array().
            from neuro_foundation import pack_poincare_dir as _pack_pd
            node.metadata["poincare_dir"] = _pack_pd(_cc_embed_to_poincare_dir(embedding))
        except Exception as exc:
            logger.debug("CC poincare_dir stamp failed (non-fatal): %s", exc)
        if index_in_recall:
            try:
                vector_db.insert(id=node_id, embedding=embedding, content=content,
                                  metadata=node.metadata)
            except Exception as exc:
                logger.warning("CC recall insert failed: %s", exc)
                raise
        return node


def cc_region_confidence(graph, fired_node_ids) -> float:
    """Shared Graduation (COMB-04): region confidence signal from the full NeuroGraph.
    
    Read-only (LAW 4, no write-side bookkeeping). The region is what FIRED:
    the node ids the Pith basin's prime_and_propagate ignited for this cue
    (KISS_Pith_Combined_Architecture.md "Shared Graduation", Packet 175a).
    Aggregates prediction confidence across synapses among them using
    graph._compute_prediction_confidence and returns a confidence in [0,1].
    
    Fail-soft: returns _CC_PITH_REGION_CONFIDENCE_NEUTRAL (0.5) on any error,
    when disabled (_CC_PITH_REGION_CONFIDENCE_ENABLED is False), or when no
    synapses are found.
    
    Args:
        graph: NeuroGraph SNN instance (neuro_foundation.Graph)
        fired_node_ids: ids of the nodes that fired for this cue
    
    Returns:
        Confidence in [0.0, 1.0], neutral (0.5) on fail-soft.
    """
    if not _CC_PITH_REGION_CONFIDENCE_ENABLED:
        return _CC_PITH_REGION_CONFIDENCE_NEUTRAL
    
    try:
        node_ids = {node_id for node_id in (fired_node_ids or ()) if node_id}
        if not node_ids:
            return _CC_PITH_REGION_CONFIDENCE_NEUTRAL
        
        # Get synapses among the fired nodes
        synapses_to_consider = []
        for node_id in node_ids:
            # Get outgoing synapses from this node
            outgoing_synapse_ids = getattr(graph, '_outgoing', {}).get(node_id, set())
            for syn_id in outgoing_synapse_ids:
                syn = graph.synapses.get(syn_id)
                if syn and syn.post_node_id in node_ids:
                    synapses_to_consider.append(syn)
        
        if not synapses_to_consider:
            return _CC_PITH_REGION_CONFIDENCE_NEUTRAL
        
        # Compute confidence for each synapse and average
        confidences = []
        for synapse in synapses_to_consider:
            try:
                # Use the SNN's internal confidence computation
                confidence = graph._compute_prediction_confidence(synapse)
                confidences.append(confidence)
            except (AttributeError, TypeError):
                # Skip synapses without proper structure
                continue
        
        if not confidences:
            return _CC_PITH_REGION_CONFIDENCE_NEUTRAL
        
        # Average confidence across synapses
        avg_confidence = sum(confidences) / len(confidences)
        return max(0.0, min(1.0, avg_confidence))
        
    except Exception:
        return _CC_PITH_REGION_CONFIDENCE_NEUTRAL


class _CCConversationalDualPassEco:
    """Eco-adapter for CC's conversational dual-pass. Mirrors canonical's
    _ConversationalDualPassEco -- inserts fine-grained tree concepts into
    the recall store, tagged {"cc": True} instead of {"syl": True} so the
    two substrates' memories are never confused if ever inspected together.
    """

    def __init__(self, graph, vector_db):
        self._graph = graph
        self._vector_db = vector_db

    def record_outcome(self, embedding, target_id, success, strength=1.0, metadata=None):
        meta = dict(metadata or {})
        meta["cc"] = True
        if meta.get("_link"):
            return {"deposited": True}
        if meta.get("_tree_concept") and meta.get("_concept"):
            if not _cc_concept_passes_floor(meta["_concept"]):
                logger.debug("Tree concept below floor, not indexed: %r", meta["_concept"][:40])
                return {"deposited": False, "reason": "concept_below_floor"}
            _cc_deposit_memory_node(self._graph, self._vector_db, target_id, embedding,
                                     meta["_concept"], meta, index_in_recall=True)
        else:
            _cc_deposit_memory_node(self._graph, self._vector_db, target_id, embedding,
                                     meta.get("_forest_content", ""), meta, index_in_recall=True)
        return {"deposited": True}

    def record_outcome_broadcast(self, embedding, target_id, success, strength=1.0, metadata=None):
        return self.record_outcome(embedding, target_id, success, strength, metadata)


def _cc_bind_conversational_topology(graph, forest_id, result, forest_embedding, state):
    """Wire forest<->tree synapses, a binding hyperedge, and a delayed
    prev->current forest link. `state` is a plain dict the caller owns
    (holds "last_forest_id") -- replaces canonical's module-level
    _last_conv_forest_id global, since each CC daemon needs its own,
    not one shared across Syl and CC.
    """
    with _cc_mutation_lock(graph):
        if forest_id not in graph.nodes:
            return
        tree_ids = [t for t in (result.get("tree_ids") or []) if t in graph.nodes and t != forest_id]
        for tid in tree_ids:
            try:
                graph.create_synapse(forest_id, tid, weight=0.2)
                graph.create_synapse(tid, forest_id, weight=0.15)
            except Exception:
                pass
        if tree_ids:
            try:
                graph.create_hyperedge(
                    member_node_ids=set([forest_id] + tree_ids),
                    metadata={"creation_mode": "conversational", "cc": True},
                )
            except Exception as exc:
                logger.debug("CC conversational hyperedge failed (non-fatal): %s", exc)
        last_id = state.get("last_forest_id")
        if last_id and last_id in graph.nodes and last_id != forest_id:
            try:
                import random as _rnd
                d = _rnd.randint(2, max(2, _CC_CONV_SYNAPSE_DELAY_MAX))
                graph.create_synapse(last_id, forest_id, weight=0.2, delay=d)
            except Exception:
                pass
        state["last_forest_id"] = forest_id
        # Anticipatory pre-activation (#256 port): this turn's forest+trees are
        # CC's "just fired" set — prime their synaptic neighborhood for the next
        # recall. state carries primed_nodes to the daemons' _recall(). (#358)
        cc_anticipate(graph, [forest_id] + tree_ids, state)


def _cc_has_ever_fired(node) -> bool:
    """True iff the node has a genuine spike on record. Mirrors canonical's
    _has_ever_fired (neurograph_rpc.py).

    Reads spike_history (appended only by Graph.step(), neuro_foundation.py:2135)
    rather than the other two firing ledgers, both of which lie for this purpose
    (punchlist #96 — the three ledgers disagree):
      - last_spike_time is ALSO stamped by prime_and_propagate() in write_mode, so
        Tonic traversal alone would forge "has fired".
      - firing_rate_ema is EMA-decayed toward 0, so it is non-monotonic — a node
        that genuinely fired long ago would read False.
    spike_history is a RingBuffer (neuro_foundation.py:554) over a
    deque(maxlen=capacity): append-only, evicts but never empties, and is
    serialized/restored with the checkpoint (to_list/from_list, nf:4491/4794).
    It is the one monotonic "has genuinely fired at least once" signal available.
    """
    hist = getattr(node, "spike_history", None)
    if hist is None:
        return False
    try:
        return len(hist) > 0
    except TypeError:
        # RingBuffer defines __len__ (nf:568), so this is unreachable for a real
        # Node. Do not raise: this runs per-node across the whole graph and one
        # malformed node must not abort the sweep. But do not swallow silently
        # either — a False here under-reports firing, and #93 consumes this as a
        # protection signal, so a silent False is the direction that costs memory.
        logger.warning(
            "_cc_has_ever_fired: spike_history has no len() (type=%s) — treating "
            "as never-fired; node will not be stamped graduated",
            type(hist).__name__,
        )
        return False


def cc_update_probation(graph) -> list:
    """Substrate-level probation graduation -- fades novelty-dampening over
    the probation window and graduates nodes to full excitability. Mirrors
    canonical's _update_probation exactly (neurograph_rpc.py:2145-2169) --
    that function is already parameterized on graph alone, so this is a
    near-verbatim port. Call once per pulse (autosave loop), after any
    conversational deposits for that pulse -- operates on ALL probationary
    nodes, not just ones just deposited.

    Novelty-dampening release is ALWAYS on the timer. Only the "graduated" stamp is
    gated on evidence of firing (#93) -- see the comment at the graduation branch for
    why those two must not be gated together.
    """
    with _cc_mutation_lock(graph):
        graduated = []
        base_threshold = graph.config.get("default_threshold", 1.0)
        for nid, node in list(graph.nodes.items()):
            # #111 -- document nodes belong to the Ingestor's probation sweep
            # (universal_ingestor.py, now scoped to creation_mode == "ingested").
            # Before both sweeps were scoped they walked the same graph, so CC's
            # ingested nodes were decremented twice -- once per prompt via
            # on_message, once per 60s pulse via this function -- burning their
            # window at double rate. One sweeper per probation domain.
            #
            # Deliberately an EXCLUSION, not `== "conversational"`: nodes with no
            # creation_mode (older checkpoints, seeds) must still graduate here
            # rather than be stranded in probation forever.
            #
            # This is where the CC mirror intentionally stops matching canonical
            # neurograph_rpc.py::_update_probation. On Syl the Ingestor sweep is
            # dead code (on_message has no callers there), so _update_probation is
            # the ONLY thing graduating her document nodes and must keep sweeping
            # them. CC-first, back-propagate later: expect these two to differ
            # until canonical is brought over.
            if (node.metadata or {}).get("creation_mode") == "ingested":
                continue
            prob = node.metadata.get("probation_remaining")
            if prob is None:
                continue
            if prob <= 0:
                # Late graduation: a node whose window expired before it ever fired stays
                # eligible. If it fires later it has earned the stamp then -- without this
                # the flag would permanently under-report nodes that entered cognition
                # after their window closed. Already-graduated nodes lack the marker and
                # fall straight through, preserving the original fast path.
                #
                # The gate is INSIDE the marker branch, mirroring the expiry branch below.
                # Gating the branch itself on _CC_CONV_PROBATION_REQUIRE_SPIKE would make
                # the rollback one-way: with the knob off, nodes already stamped
                # probation_expired_unfired would be skipped entirely and stranded at
                # graduated=False forever -- exactly the cohort the knob is flipped to
                # rescue. Rollback must drain the marker, not orphan it.
                if node.metadata.get("probation_expired_unfired"):
                    if not _CC_CONV_PROBATION_REQUIRE_SPIKE or _cc_has_ever_fired(node):
                        node.metadata["graduated"] = True
                        node.metadata.pop("probation_expired_unfired", None)
                        graduated.append(nid)
                continue
            prob -= 1
            node.metadata["probation_remaining"] = prob
            if prob <= 0:
                # Dampening release is unconditional and stays on the timer. Gating it on
                # firing would be a self-reinforcing trap: a never-fired node would keep a
                # permanently boosted threshold, making it even less likely to fire, so it
                # could never earn release.
                node.intrinsic_excitability = 1.0
                node.threshold = base_threshold
                if not _CC_CONV_PROBATION_REQUIRE_SPIKE or _cc_has_ever_fired(node):
                    node.metadata["graduated"] = True
                    graduated.append(nid)
                else:
                    # Aged out without ever firing: dampening lifted, but nothing earned.
                    node.metadata["graduated"] = False
                    node.metadata["probation_expired_unfired"] = True
            else:
                damp = float(node.metadata.get("novelty_dampening", _CC_CONV_NOVELTY_DAMPENING))
                total = float(node.metadata.get("probation_total", _CC_CONV_PROBATION_PERIOD)) or float(_CC_CONV_PROBATION_PERIOD)
                frac = max(0.0, min(1.0, 1.0 - prob / total))
                node.intrinsic_excitability = damp + (1.0 - damp) * frac
        return graduated


def run_conversational_dual_pass(graph, vector_db, text: str, embedding, state: dict) -> bool:
    """Core dual-pass on one turn's text. Returns True on success, False on
    failure -- caller decides retry policy (this function does not enqueue).
    Mirrors canonical's _run_conversational_dual_pass exactly, parameterized.
    Every turn deposits raw (LAW 7) -- no redundancy check, dedup, or
    threshold runs at deposit; `target_id` is content-hashed (see below), so
    an exact-repeat turn's deposit naturally lands on the same node instead
    of creating a duplicate.
    """
    if graph is None or embedding is None:
        return False
    try:
        from ng_embed import NGEmbed
        import hashlib
        target_id = "cc:conv::" + hashlib.sha1(text.encode()).hexdigest()
        eco = _CCConversationalDualPassEco(graph, vector_db)
        _result = NGEmbed.get_instance().dual_record_outcome(
            ecosystem=eco,
            content=text,
            embedding=embedding,
            target_id=target_id,
            success=True,
            strength=1.0,
            metadata={"source": "cc_gateway", "creation_mode": "conversational",
                      "_forest_content": text},
        )
        # The forest is real experience even when tree extraction failed. Keep
        # its chronological binding, then report partial application to the journal.
        _cc_bind_conversational_topology(graph, target_id, _result or {}, embedding, state)
        if isinstance(_result, dict) and _result.get("extraction_failed"):
            raise RuntimeError("CC dual-pass tree extraction failed after forest deposit")
        return True
    except Exception as exc:
        logger.debug("CC conversational dual-pass failed (non-fatal): %s", exc)
        return False


# Lane C (ii-a): the deposit steps. Default OFF -- flipping it advances
# graph.timestep once per deposit, which ages unbound nodes toward the orphan
# sweep, so it waits on the same gate as CC_NG_AUTOSTEP (CALLOSUM-TRUTH §8.13
# _unbound_nodes empty, the one-heartbeat tick, Packet 099). Executive-ruled flip.
# Never flipped alone: with CC_NG_AUTOSTEP on the same beat or after it (R2, #117).
_CC_NG_DEPOSIT_STEP = os.environ.get("CC_NG_DEPOSIT_STEP", "0") not in ("0", "false", "False", "")


def cc_deposit_step(graph, ingested):
    """Step once after a conversational deposit; return the StepResult.

    Restores what on_message() did before #413 swapped it for the dual pass:
    one graph.step(), the flat 0.1 baseline engagement reward, then hyperedge
    discovery on that step's own fired set (#543). Once called, it always
    steps; which calls happen depends on the door (below) -- the hook door
    calls it whether or not the dual pass succeeded. The reward is
    on_message()'s success-path form (openclaw_hook:1226): only when the turn's
    experience landed (ingested = the dual pass's return) and three_factor is
    enabled -- no phantom credit for a failed deposit (Chief ruling R1).
    No stimulus is injected: the step fires what the substrate already carries.

    Doors (Chief B3 ruling 001): the hook door is now only the Stop-side
    _deposit(step=True) (cc_ng_host.py), which calls this once per turn even
    when the dual pass failed -- a failed turn is still a timestep; the
    prompt-side, pith-failure and PostToolUse deposits never call it; the two
    drains (drain_ingest_tract, Leg-1 drain_gateway_conduit) call it once per
    APPLIED record only -- no step on a skipped, paused, uncertain,
    already-applied or failed apply.

    This step, plus the fired-set gate below, is KISS op 1 at Apprentice --
    the Delta Gate on graph data (KISS.md:38: "Read the graph.step() receipt
    ... If nothing happened, nothing to report"); the step is where that
    receipt comes from (P153(4) Q3), per Chief B3 ruling R3.

    Callers hold graph._concurrent_lock; this takes graph._step_lock (RLock)
    inside it, the established order. Fails soft: the deposit already landed.
    Nothing consumes the returned receipt yet (KISS ops 2/6 are unbuilt).
    """
    if graph is None:
        return None
    try:
        with graph._step_lock:
            result = graph.step()
            if ingested and graph.config.get("three_factor_enabled", False):
                graph.inject_reward(0.1)
            if result.fired_node_ids:
                graph.discover_hyperedges(list(result.fired_node_ids))
        return result
    except Exception as exc:
        logger.debug("CC deposit step failed (non-fatal): %s", exc)
        return None


def bootstrap_trisynaptic(memory: Any, queue: List[Dict[str, Any]],
                           instance_tag: str = "cc") -> Optional[Any]:
    """Start CC's own TriSynaptic concept-extraction manager. Watches `queue`
    for backlog overflow and spawns subprocess workers under systemd-run.
    Idle (no-op) until the caller's own drain pulse populates `queue` --
    callers that don't yet feed concept-extraction entries into the queue
    get an inert-but-harmless manager, same as Syl's own bootstrap ordering
    (manager starts before the drain pulse that feeds it).

    instance_tag distinguishes this manager's /tmp handoff files, systemd
    scope names, AND its worker's NGTractBridge module_id from Syl's own
    manager -- on the VPS, cc_ng_host.py runs inside the SAME process as
    Syl's neurograph_rpc.py, so a second TrisynapticManager sharing the
    canonical default handoff/scope naming OR worker module_id would
    cross-match Syl's orphaned handoffs/failed-file cleanup, or worse,
    have its workers deposit extracted concepts into Syl's own
    tracts_dir/neurograph/ directory (same class of bug as #302's tract
    fan-out contamination -- caught by code review before this ever
    shipped with a populated queue).

    Returns the manager instance, or None on failure/unavailable.
    """
    try:
        from trisynaptic.manager import TrisynapticManager
        manager = TrisynapticManager(
            memory=memory, queue=queue,
            handoff_prefix=f"trisynaptic_handoff_{instance_tag}_",
            scope_prefix=f"trisyn-{instance_tag}",
            worker_module_id=f"neurograph-{instance_tag}",
        )
        manager.start()
        logger.info("CC TriSynaptic manager started (instance_tag=%s)", instance_tag)
        return manager
    except Exception:
        logger.exception("CC TriSynaptic manager failed to start — concept backlog will accumulate")
        return None


# ---- Tract ingest drain (#294 miniTID integration) ----
# Drains BTF entries from miniTID's turn-deposit tract file, running each
# through the conversational dual-pass (Task 1). Feeder (miniTID) deposits,
# this drains independently -- no handshake, matching the established tract model.

_DEFAULT_CC_GATEWAY_TRACT_PATH = os.path.expanduser(
    "~/.claude/plugins/neurograph/tracts/cc_gateway/turns.tract"
)


def cc_gateway_tract_path() -> str:
    """Resolve the CC gateway tract path from CC_GATEWAY_TRACT_PATH (LAW 5) --
    both this drain side and miniTID's Rust producer independently read the
    same env var, with the same default, so they can never desync onto
    different files without either side being misconfigured identically."""
    return os.environ.get("CC_GATEWAY_TRACT_PATH", _DEFAULT_CC_GATEWAY_TRACT_PATH)


def _apply_gateway_experience(graph, vector_db, state, entry):
    """Canonical raw conversation embedding/dual-pass, shared by both drains."""
    from ng_embed import embed
    return run_conversational_dual_pass(
        graph, vector_db, entry.content, embed(entry.content), state)


def drain_ingest_tract(graph, vector_db, state: dict, tract_path: str = None,
                        return_consumed: bool = False, max_entries: int = 0):
    """Drain miniTID's turn-deposit tract file, running each raw experience
    entry through the conversational dual-pass (Task 1). Feeder (miniTID)
    deposits, this drains independently -- no handshake, matching the
    established tract model. Truncates the file after a successful drain
    (single reader, single writer-appender; safe because miniTID only ever
    appends and this is the only drainer).

    max_entries > 0 stops after that many entries have been TAKEN IN (reached
    the embed/dual-pass attempt) and truncates ONLY the byte span up to the end
    of the last entry taken; the remainder stays in the file for the next call.
    0 (default) drains the whole file -- byte-identical to the previous
    behaviour, since the stop offset then lands exactly at len(data).

    Why a partial truncate is safe here: the tract format is a flat run of
    self-describing entries with no file-level header (verified against the
    installed ng_tract 0.1.0 -- 5 entries deposit as exactly 5*56 bytes, and
    both data[:off] and data[off:] parse standalone at every entry boundary).
    TractReader.position() -- a METHOD, not a property -- returns the byte
    offset just past the entry it last yielded, so any position() value is
    simultaneously a valid end-of-prefix and start-of-remainder.

    The cap counts entries ATTEMPTED, not entries successfully absorbed. That
    is deliberate: it is what actually bounds the work (and therefore the
    caller's lock hold), and it guarantees forward progress. Counting only
    successes would mean a file whose entries all fail the dual-pass never
    reaches the cap and gets drained in one unbounded lock hold. This is a
    resource bound, not FatherGraph topology consolidation.

    Returns the count of entries absorbed (int) by default. If
    return_consumed=True, returns (absorbed, consumed_bytes) instead --
    consumed_bytes is EXACTLY the byte span this call truncated out of the
    file (b'' on any early-return path, including a parse failure that never
    reached truncate), never an independently-taken snapshot. This closes a
    Corpus Callosum Leg 1 (#70) data-loss window a prior design had: a
    caller taking its own separate pre-drain snapshot can miss bytes
    miniTID appends between that snapshot and this function's OWN read --
    those bytes still get absorbed+truncated here, but the caller's stale
    snapshot never contained them, so they'd be silently lost to any
    second consumer (the VPS Arborist) even though the file already
    forgot them. Handing back the literal consumed span makes trickling
    it elsewhere byte-exact with what was actually removed from the file,
    with no separate read and no window between them.

    Locking: the caller holds graph._concurrent_lock for the whole call --
    the dual pass mutates the graph, and with CC_NG_DEPOSIT_STEP on, the
    per-record cc_deposit_step(graph, True) requires the caller's lock hold,
    which punchlist #643 (the autosave-loop caller) now provides.

    Fails soft -- an ingest-tract drain failure must never break the
    daemon's autosave pulse.
    """
    def _ret(absorbed_n: int, consumed: bytes = b""):
        return (absorbed_n, consumed) if return_consumed else absorbed_n

    path = tract_path or cc_gateway_tract_path()
    if not os.path.exists(path):
        return _ret(0)
    try:
        import ng_tract
        from ng_embed import embed as ng_embed_fn
    except Exception as exc:
        logger.debug("CC ingest-tract drain unavailable (non-fatal): %s", exc)
        return _ret(0)

    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception as exc:
        logger.debug("CC ingest-tract read failed (non-fatal): %s", exc)
        return _ret(0)
    if not data:
        return _ret(0)

    absorbed = 0
    taken = 0
    # Byte offset just past the last entry we advanced over. Advanced for EVERY
    # entry, including ones the filters skip -- a skipped entry has still been
    # looked at, and leaving it in the file would make every later call re-scan
    # it forever. 0 means we never got past the first entry.
    consumed_offset = 0
    try:
        reader = ng_tract.TractReader(data)
        for entry in reader:
            # position() is a bound method on the Rust binding, not a property.
            # Read it BEFORE the filters so `continue` still consumes the entry.
            consumed_offset = reader.position()
            # Check entry type using ng_tract.ENTRY_EXPERIENCE (the real module constant)
            if entry.entry_type != ng_tract.ENTRY_EXPERIENCE:
                continue
            if entry.source != "cc_gateway":
                continue
            text = entry.content
            if not text or not text.strip():
                continue
            taken += 1
            try:
                if _apply_gateway_experience(graph, vector_db, state, entry):
                    absorbed += 1
                    if _CC_NG_DEPOSIT_STEP:
                        cc_deposit_step(graph, True)
            except Exception as exc:
                logger.debug("CC ingest-tract entry failed (non-fatal): %s", exc)
            if max_entries and taken >= max_entries:
                break
    except Exception as exc:
        # Parse failure -- truncate below never runs, so nothing was actually
        # consumed from the file. Elevated to warning (was debug): silent at
        # the default level, this is exactly how a laptop/VPS ng_tract format
        # skew would look -- every file failing the same way, invisibly.
        logger.warning("CC ingest-tract parse failed (non-fatal, file untouched): %s", exc)
        return _ret(absorbed)  # consumed=b"" -- nothing was truncated

    # Truncate only the bytes we actually consumed. miniTID is a separate
    # process that only appends; if it appends between our initial read and
    # this truncation, a blind `open(path, "wb")` would erase those new bytes
    # along with the ones we already drained. Re-reading the current file and
    # writing back only what comes after our consumed prefix closes that
    # window down to the two file ops below, instead of spanning the whole
    # embed+dual-pass pass above.
    #
    # consumed_actual tracks EXACTLY what left the file, set ONLY after a
    # confirmed successful removal -- NOT assumed from `data` unconditionally.
    # (2026-07-27 law-enforcer re-review: the prior unconditional `return
    # _ret(absorbed, data)` here over-reported `consumed` on three paths --
    # the current.startswith(data) mismatch branch, an rb-reopen failure, and
    # a wb-open failure -- each left `data` still sitting in the file while
    # still telling the caller it was gone. For Leg 1 (#70) that meant the
    # laptop's next pulse would re-read and re-trickle the SAME bytes the VPS
    # already absorbed -- duplicate ingestion, the mirror-image of the
    # original data-loss bug this whole return_consumed path exists to fix.)
    #
    # The prefix we trim is the span we actually walked, which equals `data`
    # on an uncapped drain and stops at an entry boundary on a capped one.
    # Everything downstream (startswith check, remainder, consumed_actual)
    # keys off THIS, never off `data` -- so a capped drain reports exactly the
    # bytes it removed and leaves the rest both on disk and unclaimed.
    consumed_prefix = data[:consumed_offset]
    if not consumed_prefix:
        # Never got past the first entry -- nothing to remove. Return without
        # rewriting the file at all, rather than doing a no-op full rewrite.
        return _ret(absorbed)
    consumed_actual = b""
    try:
        with open(path, "rb") as f:
            current = f.read()
        if current.startswith(consumed_prefix):
            remainder = current[len(consumed_prefix):]
            with open(path, "wb") as f:
                f.write(remainder)
            consumed_actual = consumed_prefix  # only now -- the write actually succeeded
        else:
            # Someone else touched the file since our read (not the plain
            # append-only case we can safely trim a known prefix from).
            # Write it back unchanged rather than guess -- nothing of ours
            # was removed, so consumed_actual correctly stays b"".
            with open(path, "wb") as f:
                f.write(current)
    except Exception as exc:
        logger.debug("CC ingest-tract truncate failed (non-fatal): %s", exc)

    if absorbed:
        logger.info("CC ingest-tract: absorbed %d turn(s) into recall", absorbed)
    return _ret(absorbed, consumed_actual)


# ---- Corpus Callosum Leg 1 (#70): laptop -> VPS raw-turn conduit ----
# See changelog entry above for the full design. Producer side
# (trickle_gateway_conduit) runs on the laptop; consumer side
# (drain_gateway_conduit) runs on the VPS, alongside its own local
# drain_ingest_tract() call. Both gated by CC_CALLOSUM_LEG1_ENABLED (LAW 5).

_DEFAULT_CC_GATEWAY_CONDUIT_DIR = os.path.expanduser("~/docs/ng_topology")
_CC_GATEWAY_CONDUIT_GLOB = "*_cc_gateway.*.tract"

_CC_CALLOSUM_LEG1_ENABLED = os.environ.get("CC_CALLOSUM_LEG1_ENABLED", "0") not in ("0", "false", "False", "")


def cc_gateway_conduit_dir() -> str:
    """Resolve the Leg 1 conduit directory from CC_GATEWAY_CONDUIT_PATH (LAW 5,
    default ~/docs/ng_topology -- the same dir repo-sync.sh's existing 15-min
    git cron already syncs, so no new transport is needed). Both the laptop
    writer (trickle_gateway_conduit) and the VPS reader (drain_gateway_conduit)
    independently read the same env var with the same default."""
    return os.environ.get("CC_GATEWAY_CONDUIT_PATH", _DEFAULT_CC_GATEWAY_CONDUIT_DIR)


def trickle_gateway_conduit(data: bytes, conduit_dir: str = None) -> Optional[str]:
    """Laptop side: write a snapshot of already-read cc_gateway tract bytes
    to a new, uniquely-named per-batch file in the synced conduit dir.

    One immutable file per call -- deliberately NOT a shared append/truncate
    target, which would risk a binary merge conflict under repo-sync.sh's git
    push/pull cycle (git cannot line-merge BTF). Atomic (write-tmp-then-
    rename) so a concurrent repo-sync.sh push, or a crash mid-write, never
    observes a partial file.

    Gated by CC_CALLOSUM_LEG1_ENABLED (LAW 5), default off -- a no-op
    (returns None immediately) when the gate is off, so this is inert dead
    code on both hemispheres until explicitly flipped on. Fails soft --
    a conduit-write failure must never affect the caller's local drain or
    the daemon's autosave pulse. Returns the written path on success, else
    None (gate off, empty data, or any failure).
    """
    if not _CC_CALLOSUM_LEG1_ENABLED:
        return None
    if not data:
        return None
    try:
        conduit_dir = conduit_dir or cc_gateway_conduit_dir()
        os.makedirs(conduit_dir, exist_ok=True)
        # Hemisphere identity must be DECLARED, never guessed. The drain's
        # exclude_prefix guard is what stops a half from eating its own
        # outgoing turns, and it keys on this filename prefix -- so a wrong
        # default here silently disarms it. Defaulting to "laptop" would make
        # a VPS-produced file look laptop-produced, and the VPS would then
        # drain (and delete) its own turns before the laptop ever pulled them:
        # silent one-way data loss that looks exactly like success. Refuse
        # loudly instead; the cron already exports MACHINE_ID on both halves.
        machine_id = os.environ.get("MACHINE_ID", "").strip()
        if not machine_id:
            logger.warning(
                "CC callosum Leg1: MACHINE_ID unset -- refusing to write a conduit file "
                "with a guessed hemisphere identity (would disarm the drain's "
                "self-consumption guard). Set MACHINE_ID in the daemon env.")
            return None
        fname = f"{machine_id}_cc_gateway.{time.time_ns()}_{uuid.uuid4().hex[:8]}.tract"
        dest = os.path.join(conduit_dir, fname)
        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
        return dest
    except Exception as exc:
        logger.debug("CC callosum Leg1 conduit write failed (non-fatal): %s", exc)
        return None


def _cc_callosum_consolidate(graph, idle_steps: int) -> bool:
    """FatherGraph Finding 3 sleep consolidation: run idle_steps of pure
    graph.step() with NO new input, so homeostatic regulation (threshold
    adaptation, synaptic scaling, excitability) can catch up before the next
    batch of foreign topology arrives. Measured 47%->74% accuracy in the
    FatherGraph training; the report calls it "not optional -- it's what
    makes merge work". Mirrors _handle_import (cc_ng_host.py) and
    import_trickle (cc-ng-sync.py), which already do exactly this.
    Returns True if the steps ran. Fails soft."""
    if idle_steps <= 0 or graph is None:
        return False
    # Take the lock in SLICES, not for the whole 250 steps. cc_ng_host.py's
    # changelog records real hook timeouts caused by _recall() blocking on a
    # long _concurrent_lock hold ("_concurrent_lock in _recall() caused hook
    # timeouts (Tonic holds lock)"). Consolidation is exactly that shape --
    # hundreds of graph.step() calls -- so it yields between slices, letting
    # a waiting recall/deposit interleave. Homeostasis does not care whether
    # the steps were contiguous; the hooks care a great deal.
    slice_n = max(1, int(os.environ.get("CC_CALLOSUM_LOCK_SLICE_STEPS", "25")))
    try:
        lock = getattr(graph, "_concurrent_lock", None)
        done = 0
        while done < idle_steps:
            n = min(slice_n, idle_steps - done)
            if lock is not None:
                with lock:
                    for _ in range(n):
                        graph.step()
            else:
                for _ in range(n):
                    graph.step()
            done += n
        return True
    except Exception as exc:
        logger.debug("CC callosum Leg1 consolidation failed (non-fatal): %s", exc)
        return False


def drain_gateway_conduit(graph, vector_db, state: dict, conduit_dir: str = None,
                           batch_size: int = None, idle_steps: int = None,
                           load_ceiling: float = None, exclude_prefix: str = None,
                           *, save_callback=None, journal_path=None) -> dict:
    """Receive immutable raw Leg1 input, acknowledge only a complete save receipt.

    SQLite stores transport identities, exact raw BTF bytes and attempt states,
    never embeddings or derived cognition. FULL synchronous transactions precede
    every mutation. A filesystem lock serializes deliveries, while graph locking
    remains one record (or save) at a time. No synthetic graph steps -- this
    bars idle and consolidation cadence (the Sep-11 supersession), not the one
    cc_deposit_step per applied record under CC_NG_DEPOSIT_STEP (Chief B3
    ruling R2): a skipped, paused, uncertain, already-applied or failed apply
    never steps.

    Interrupted attempts, partial learning and unresolved prior-incarnation
    deliveries require reconciliation. Terminal accepted receipts survive normal
    restarts without relearning. The checkpoint and journal must be preserved
    together: arbitrary rollback or mixing a newer journal with an older graph
    is unsupported without reconciliation, not detected by process ownership.
    Same live graph may retry a refused save without repeating applied records.
    Raw journal copies are retained even after acceptance (no automatic GC).
    Producer filenames are unique and immutable; transport participants must
    cooperate with delivery serialization. The digest recheck before unlink is
    not an atomic compare-and-unlink against an unrelated same-name writer.
    Legacy batch/idle arguments remain inert for 1/0 socket compatibility.
    """
    import contextlib
    import fcntl
    import hashlib
    import json
    import sqlite3

    result = dict(ok=False, accepted=False, absorbed=0, applied=0,
                  retained=0, uncertain=0, accepted_files=[], errors=[])
    if not _CC_CALLOSUM_LEG1_ENABLED:
        result['disabled'] = True
        return result
    if save_callback is None or not journal_path:
        result['errors'].append('durable save callback and local journal required')
        return result
    if getattr(graph, '_concurrent_lock', None) is None:
        result['errors'].append('graph mutation lock required')
        return result
    if exclude_prefix is None:
        machine_id = os.environ.get('MACHINE_ID', '').strip()
        if not machine_id:
            result['errors'].append('MACHINE_ID required for self-consumption guard')
            return result
        exclude_prefix = machine_id + '_'
    if not exclude_prefix:
        result['errors'].append('empty self-consumption guard refused')
        return result
    if batch_size not in (None, 1) or idle_steps not in (None, 0):
        logger.warning('CC Leg1 ignores topology batch/idle arguments')
    conduit_dir = os.path.realpath(conduit_dir or cc_gateway_conduit_dir())
    journal_path = os.path.abspath(journal_path)
    if os.path.commonpath([conduit_dir, journal_path]) == conduit_dir:
        result['errors'].append('local journal must be outside synced conduit')
        return result
    try:
        from cc_refeed import should_pause_for_load
    except ImportError:
        should_pause_for_load = lambda ceiling: False
    ceiling = float(load_ceiling if load_ceiling is not None else
                    os.environ.get('CC_CALLOSUM_LOAD_CEILING', '1.5'))
    try:
        os.makedirs(os.path.dirname(journal_path), mode=0o700, exist_ok=True)
        # Persist the new delivery-directory entry before trusting its journal.
        parent_fd = os.open(os.path.dirname(os.path.dirname(journal_path)),
                            os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        with open(journal_path + '.lock', 'a+b') as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            # Keep a strong graph reference: neither PID reuse nor id() reuse can make
            # another graph look like the one whose applied records can retry saving.
            binding = state.get('_gateway_delivery_binding')
            if not binding or binding[0] is not graph or binding[1] != os.getpid():
                binding = (graph, os.getpid(), uuid.uuid4().hex)
                state['_gateway_delivery_binding'] = binding
            owner = binding[2]
            with contextlib.closing(sqlite3.connect(journal_path)) as db:
                db.execute('PRAGMA journal_mode=WAL')
                db.execute('PRAGMA synchronous=FULL')
                db.execute('CREATE TABLE IF NOT EXISTS files (\n                    conduit TEXT, name TEXT, digest TEXT, raw BLOB NOT NULL,\n                    owner TEXT NOT NULL, status TEXT NOT NULL, receipt TEXT,\n                    PRIMARY KEY(conduit, name, digest))')
                db.execute('CREATE TABLE IF NOT EXISTS records (\n                    conduit TEXT, name TEXT, digest TEXT, start INTEGER, end INTEGER,\n                    status TEXT NOT NULL,\n                    PRIMARY KEY(conduit, name, digest, start, end))')
                db.commit()
                directory_fd = os.open(os.path.dirname(journal_path), os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                # Retain each full immutable file BEFORE parsing or any learning.
                for path in sorted(glob.glob(os.path.join(conduit_dir, _CC_GATEWAY_CONDUIT_GLOB))):
                    name = os.path.basename(path)
                    if name.startswith(exclude_prefix):
                        continue
                    with open(path, 'rb') as stream:
                        raw = stream.read()
                    digest = hashlib.sha256(raw).hexdigest()
                    with db:
                        db.execute('INSERT OR IGNORE INTO files VALUES (?,?,?,?,?,?,NULL)',
                                   (conduit_dir, name, digest, raw, owner, 'retained'))
                rows = db.execute('SELECT name,digest,owner,status FROM files WHERE conduit=? ORDER BY name,digest',
                                  (conduit_dir,)).fetchall()
                slices = 0
                pending = []
                accepted = []
                for name, digest, prior_owner, status in rows:
                    if name.startswith(exclude_prefix):
                        continue
                    key = (conduit_dir, name, digest)
                    if status != 'accepted' and prior_owner != owner:
                        # Every mutation has a committed record marker first.
                        # Retained bytes with NO record rows have never been
                        # attempted, so a new incarnation may safely adopt them.
                        attempted = db.execute(
                            'SELECT 1 FROM records WHERE conduit=? AND name=? AND digest=? LIMIT 1',
                            key).fetchone()
                        if attempted:
                            result['uncertain'] += 1
                            result['retained'] += 1
                            continue
                        with db:
                            db.execute('UPDATE files SET owner=? WHERE conduit=? AND name=? AND digest=?',
                                       (owner,) + key)
                    if status in ('uncertain', 'invalid'):
                        result['retained'] += 1
                        result['uncertain'] += status == 'uncertain'
                        continue
                    if status != 'accepted':
                        raw = db.execute('SELECT raw FROM files WHERE conduit=? AND name=? AND digest=?', key).fetchone()[0]
                        if hashlib.sha256(raw).hexdigest() != digest:
                            result['errors'].append(name + ': retained raw digest mismatch')
                            result['retained'] += 1
                            continue
                        try:
                            import ng_tract
                            reader = ng_tract.TractReader(raw)
                            records = []
                            start = 0
                            for entry in reader:
                                end = reader.position()
                                if not start < end <= len(raw):
                                    raise ValueError('invalid tract byte interval')
                                if (entry.entry_type != ng_tract.ENTRY_EXPERIENCE or
                                        entry.source != 'cc_gateway' or not entry.content.strip()):
                                    raise ValueError('unexpected or empty gateway record')
                                records.append((start, end, entry))
                                start = end
                            if start != len(raw) or not records:
                                raise ValueError('incomplete or empty tract')
                        except Exception as exc:
                            with db:
                                db.execute('UPDATE files SET status=? WHERE conduit=? AND name=? AND digest=?',
                                           ('invalid',) + key)
                            result['errors'].append(name + ': ' + str(exc))
                            result['retained'] += 1
                            continue
                        complete = True
                        for start, end, entry in records:
                            rkey = key + (start, end)
                            record = db.execute('SELECT status FROM records WHERE conduit=? AND name=? AND digest=? AND start=? AND end=?', rkey).fetchone()
                            if record and record[0] == 'applied':
                                continue
                            if record:
                                complete = False
                                with db:
                                    db.execute('UPDATE files SET status=? WHERE conduit=? AND name=? AND digest=?', ('uncertain',) + key)
                                result['uncertain'] += 1
                                break
                            if slices and should_pause_for_load(ceiling):
                                complete = False
                                break
                            with db:
                                db.execute('INSERT INTO records VALUES (?,?,?,?,?,?)', rkey + ('attempting',))
                            try:
                                with graph._concurrent_lock:
                                    applied = _apply_gateway_experience(graph, vector_db, state, entry)
                                    if not applied:
                                        raise RuntimeError('dual-pass did not confirm full application')
                                    if _CC_NG_DEPOSIT_STEP:
                                        cc_deposit_step(graph, applied)
                                with db:
                                    db.execute('UPDATE records SET status=? WHERE conduit=? AND name=? AND digest=? AND start=? AND end=?', ('applied',) + rkey)
                                result['applied'] += 1
                                result['absorbed'] += 1
                                slices += 1
                            except Exception as exc:
                                with db:
                                    db.execute('UPDATE files SET status=? WHERE conduit=? AND name=? AND digest=?', ('uncertain',) + key)
                                result['errors'].append(name + ': ' + str(exc))
                                result['uncertain'] += 1
                                complete = False
                                break
                        if not complete:
                            result['retained'] += 1
                            continue
                        pending.append(key)
                    else:
                        accepted.append(key)
                if pending:
                    try:
                        with graph._concurrent_lock:
                            receipt = save_callback()
                        if (not isinstance(receipt, dict) or receipt.get('accepted') is not True
                                or receipt.get('outcome') != 'primary'):
                            raise RuntimeError('checkpoint not accepted: ' + str(receipt))
                        # One save covers all completely applied input files.
                        # Journal acceptance commits before deletion or response.
                        with db:
                            for key in pending:
                                db.execute('UPDATE files SET status=?,receipt=? WHERE conduit=? AND name=? AND digest=?',
                                           ('accepted', json.dumps(receipt)) + key)
                        accepted.extend(pending)
                    except Exception as exc:
                        result['errors'].append(str(exc))
                        result['retained'] += len(pending)
                for _, name, digest in accepted:
                    path = os.path.join(conduit_dir, name)
                    if os.path.exists(path):
                        with open(path, 'rb') as stream:
                            current = stream.read()
                        if hashlib.sha256(current).hexdigest() != digest:
                            # Same name, different bytes: no deletion authority.
                            result['retained'] += 1
                            continue
                        os.unlink(path)
                        fd = os.open(conduit_dir, os.O_RDONLY | os.O_DIRECTORY)
                        try:
                            os.fsync(fd)
                        finally:
                            os.close(fd)
                    result['accepted_files'].append(dict(name=name, sha256=digest))
                result['accepted'] = bool(result['accepted_files'])
                result['all_done'] = not result['retained'] and not result['errors']
                result['ok'] = not result['retained'] and not result['errors']
    except Exception as exc:
        result['errors'].append(str(exc))
        result['ok'] = result['accepted'] = False
    return result


# [2026-07-10] Recall seed floor for _harvest_associations' VDB seed-search.
# REVERTED to canonical 0.40 after MEASURING: the presumption that 0.40 was
# "too high" (starving the query of seeds) was false. Live measurement over the
# 2503-entry vector_db (this session): top query cosines are ~0.58-0.64 and
# 783-1602 nodes clear 0.40 for typical queries -- and _harvest_associations
# caps seeds at prime_k (~10) anyway, so the top-10 seeds (all ~0.58) are IDENTICAL
# whether the floor is 0.40 or 0.22. The floor change was therefore INERT, not a
# fix. The real query-blindness lives AFTER seeding (spread convergence and/or
# SurfacingMonitor recency domination -- seeds themselves discriminate fine:
# geometry vs devops queries share 0/10 top seeds). Kept env-tunable for future
# measured experiments; default restored to the canonical value.
_CC_RECALL_PRIME_THRESHOLD = float(os.environ.get("CC_RECALL_PRIME_THRESHOLD", "0.40"))


# [2026-07-11] Lever 2 -- rank-time diagnosticity / selectivity (gated OFF).
# Measured root cause of query-blind recall: a handful of boilerplate HUB nodes
# (degree 260-341 vs graph median 2) fire for EVERY query and dominate the
# spread. Their firing_rate_ema sits at ~0.32-0.38 (near the 0.395 max) while
# the graph median is 0.0 -- so rank by strength / (firing_rate_ema + eps) and
# the always-firing hubs divide down hard while a node that fired UNUSUALLY for
# THIS query (near-zero baseline) rises. Divisive normalization / diagnosticity:
# a memory relevant to everything is relevant to nothing. OVERSAMPLE the harvest
# (max_surfaced = k * oversample) so lower-strength query-relevant nodes are in
# the pool at all -- re-ranking only the top-k hubs the spread returns can't
# help. Non-vendored, env-tunable, off by default (LAW 5); consumes the engine's
# own firing_rate_ema, derives nothing new.
_CC_RECALL_SELECTIVITY = os.environ.get("CC_RECALL_SELECTIVITY", "0") not in ("0", "false", "")
_CC_RECALL_SELECTIVITY_EPS = float(os.environ.get("CC_RECALL_SELECTIVITY_EPS", "0.02"))
_CC_RECALL_SELECTIVITY_OVERSAMPLE = max(1, int(os.environ.get("CC_RECALL_SELECTIVITY_OVERSAMPLE", "6")))
# Experimental: cap prime_and_propagate steps to keep activation near the
# query-specific seeds (0 = engine default of 3). Measured hypothesis: the spread
# converges to the same hub attractor basin at 3 steps regardless of seed.
_CC_RECALL_PROP_STEPS = int(os.environ.get("CC_RECALL_PROP_STEPS", "0"))


def cc_pattern_completion_recall(ng: Any, query: str, k: int = 5,
                                    threshold: float = _CC_RECALL_PRIME_THRESHOLD,
                                    state: Optional[Dict[str, Any]] = None,
                                    preserve_graph_config: bool = False) -> List[Dict[str, Any]]:
    """Substrate-native pattern-completion recall for CC's hook surfacing
    (#358 rebuild -- replaces the bare ng.recall() cosine search this
    function originally wrapped; LAW 3 rebuild-in-place, same contract).

    Spreading activation via ng._harvest_associations(): the vector_db only
    seeds prime nodes; what surfaces is what FIRES through learned synaptic
    structure (graph.prime_and_propagate) -- the substrate is the memory,
    the VDB is secondary. Enrichments applied in canonical handle_assemble()
    order (neurograph_rpc.py:2977-3038): MMN novelty scaling (cc_novelty,
    pull-based), anticipatory primed-node bonus (#256 port), GSG geodesic
    re-score.

    threshold maps to prime_threshold (seed-selection floor -- same
    conceptual role as the old cosine floor; 0.40 = confidence_recommend,
    unchanged). k maps to max_surfaced. Config override mirrors canonical
    associate() (openclaw_hook.py:1076-1085) -- save/restore around the call.

    state: the daemon's conv_state dict (novelty_ema/last_confirmed/
    last_surprised/primed_nodes). None (legacy call shape) = neutral novelty
    0.5, no primed bonus -- still substrate-native.

    Returns [{node_id, score, content}] -- same shape as before; content
    substrate-first via resolve_surface_content, degenerate results dropped.
    Fails soft: any exception returns [].
    """
    if not query or ng is None:
        return []
    try:
        from surface_resolver import resolve_surface_content
        novelty = cc_novelty(state, ng.graph) if state is not None else 0.5
        cfg = ng.graph.config
        old_max = cfg.get("max_surfaced", 10)
        old_thresh = cfg.get("prime_threshold", 0.4)
        old_steps = cfg.get("propagation_steps", 3)
        harvest_max = k * _CC_RECALL_SELECTIVITY_OVERSAMPLE if _CC_RECALL_SELECTIVITY else k
        if preserve_graph_config:
            # provider_context is a read-only observation boundary.  The host
            # must not temporarily rewrite graph.config; use the canonical
            # harvest overrides instead.  The graph-owned prime threshold stays
            # authoritative on this observation path.  The underlying
            # prime_and_propagate read mode disables plasticity and restores
            # transient voltages/refractory state after observational ignition;
            # it exposes current learned topology without teaching the graph.
            surfaced = ng._harvest_associations(
                query,
                novelty=novelty,
                max_surfaced_override=harvest_max,
                propagation_steps_override=(
                    _CC_RECALL_PROP_STEPS if _CC_RECALL_PROP_STEPS > 0 else None
                ),
            )
        else:
            # Existing hook recall temporarily overrides the graph's harvest
            # controls and restores them around the call.  Kept unchanged for
            # compatibility; provider_context takes the branch above.
            cfg["max_surfaced"] = harvest_max
            cfg["prime_threshold"] = threshold
            if _CC_RECALL_PROP_STEPS > 0:
                cfg["propagation_steps"] = _CC_RECALL_PROP_STEPS
            try:
                surfaced = ng._harvest_associations(query, novelty=novelty)
            finally:
                cfg["max_surfaced"] = old_max
                cfg["prime_threshold"] = old_thresh
                cfg["propagation_steps"] = old_steps

        # Anticipatory bonus (#256 port) -- canonical rpc.py:2981-2989
        promoted_ids: set = set()
        if state is not None:
            now = time.time()
            live = {nid: s for nid, (s, exp) in (state.get("primed_nodes") or {}).items()
                    if exp > now}
            if live:
                surfaced_ids = {item.get("node_id") for item in surfaced}
                for item in surfaced:
                    nid = item.get("node_id")
                    if nid and nid in live:
                        item["strength"] = item.get("strength", 0.0) + _CC_ANTICIPATE_BONUS
                        # #55 5b: a predicted node the harvest surfaced BY ITSELF is
                        # the true Markov-prefetch hit (5a's promotion only counts
                        # the ones it had to inject). Read via the pith_metrics RPC.
                        _PITH_METRICS.prefetch_surfaced += 1

                # Pith Stage 4 (#55) predictive promotion, phase 5a: today a
                # primed node only ever helps if the query-driven harvest
                # independently finds it too -- prefetch exists for exactly
                # the opposite case, surfacing predicted content the harvest
                # MISSED. Gated (byte-identical when off, _CC_PITH_PREFETCH_
                # ENABLED default "0"); pure-additive candidate injection with
                # no hard override -- cc_gsg_rescore and downstream rank/
                # budget still decide whether a promoted node survives (a
                # weak prediction still loses to a strong harvest hit).
                if _CC_PITH_PREFETCH_ENABLED:
                    for nid, primed_score in live.items():
                        if nid in surfaced_ids:
                            continue
                        try:
                            if ng.graph is None or nid not in ng.graph.nodes:
                                continue
                            surfaced.append({"node_id": nid, "strength": _CC_ANTICIPATE_BONUS})
                            surfaced_ids.add(nid)
                            promoted_ids.add(nid)
                        except Exception as exc:
                            logger.debug("Pith predictive promotion skipped node %r (non-fatal): %s", nid, exc)
                    if promoted_ids:
                        _PITH_METRICS.promoted_predicted += len(promoted_ids)

                surfaced.sort(key=lambda x: x.get("strength", 0.0), reverse=True)

        # GSG geodesic re-score -- canonical rpc.py:2991-3038
        surfaced = cc_gsg_rescore(surfaced, query, ng.graph)

        # Pith Stage 4 (#55) proximity-keyed LOD (spec sec 4c): a promoted-
        # but-unsurfaced line beyond the distance threshold is staged as a
        # keyframe summary rather than full content -- near predictions are
        # trusted at full resolution, far ones cost less if they turn out
        # irrelevant. Query direction is only computed when there's a
        # promoted candidate to stage (no extra embed cost otherwise).
        query_dir = None
        if promoted_ids:
            try:
                from ng_embed import embed as _embed
                query_dir = _cc_embed_to_poincare_dir(_embed(query))
            except Exception as exc:
                logger.debug("Pith LOD query embed failed (non-fatal): %s", exc)
                query_dir = None

        out = []
        for r in surfaced:
            nid = r.get("node_id") or r.get("id")
            node = ng.graph.nodes.get(nid) if (nid and ng.graph) else None
            text = resolve_surface_content(node, r, allow_ingested=True, max_chars=300)
            if not text:
                continue
            if nid in promoted_ids and query_dir is not None:
                try:
                    dist = _cc_node_query_distance(node, query_dir)
                    if dist is not None and dist > _CC_PITH_PREFETCH_LOD_DIST:
                        text, _ = pith_stage2_keyframe(text, max_chars=_CC_PITH_PREFETCH_SUMMARY_CHARS, query=query)
                except Exception as exc:
                    logger.debug("Pith LOD staging failed for %r (non-fatal): %s", nid, exc)
            # [D5] Carry Stage-4 promotion provenance out of this function so the
            # assembler can stamp it on the CacheLine. Additive key; every existing
            # consumer reads by name and is unaffected.
            out.append({"node_id": nid, "score": r.get("strength", 0.0), "content": text,
                        "prefetch_origin": nid in promoted_ids})

        # Lever 2: selectivity re-rank (gated). Divide each candidate's strength
        # by its baseline firing rate so hubs (fire for everything) sink and
        # nodes that fired unusually for THIS query rise; then take top-k from
        # the oversampled pool. firing_rate_ema is the engine's own signal --
        # nothing re-derived. Fail-soft per node (missing ema -> 0 -> max boost).
        if _CC_RECALL_SELECTIVITY and out:
            eps = _CC_RECALL_SELECTIVITY_EPS
            for item in out:
                try:
                    node = ng.graph.nodes.get(item["node_id"]) if ng.graph else None
                    fre = float(getattr(node, "firing_rate_ema", 0.0) or 0.0)
                except Exception:
                    fre = 0.0
                item["score"] = item["score"] / (fre + eps)
            out.sort(key=lambda x: x["score"], reverse=True)

        final = out[:k]
        # Pith Stage 4 (#55) §13.3 measurement: a "hit" here is scoped to what
        # this function can see -- a promoted-and-unsurfaced node that
        # survived ranking into this turn's returned set. It is an honest
        # lower bound on the true L1-survival ratio (pith_stage3's later
        # budget cut can still drop it) -- see spec sec 6 def-of-done.
        if promoted_ids:
            survived = sum(1 for item in final if item.get("node_id") in promoted_ids)
            if survived:
                _PITH_METRICS.prefetch_hits += survived

        return final
    except Exception as exc:
        logger.debug("cc_pattern_completion_recall failed (non-fatal): %s", exc)
        return []


def _format_cc_recall_block(results: List[Dict[str, Any]]) -> str:
    """Format cc_pattern_completion_recall() results as an '## Active Recall'
    block -- mirrors canonical's handle_assemble() Active Recall formatting
    (neurograph_rpc.py:3094-3107) exactly. Returns '' when results is empty.
    """
    if not results:
        return ""
    lines = ["## Active Recall\nDirect memory retrieval for the current query:"]
    for r in results:
        lines.append(f"- [{r['score']:.2f}] {r['content']}")
    return "\n".join(lines)


PATTERN_COMPLETION_FILE_TTL = 1800.0  # seconds (30 min) -- see gate_pattern_completion()


def gate_pattern_completion(cache: Dict[str, float], file_path: str, now: float,
                              ttl: float = PATTERN_COMPLETION_FILE_TTL) -> bool:
    """Per-file dedup gate for PreToolUse-triggered pattern-completion recall
    (2026-07-06 refinement to the tier-drop design). Pure function over a
    plain dict -- no I/O, no graph access.

    PreToolUse fires on every tool call touching a file, far more often than
    UserPromptSubmit -- without this gate, repeatedly touching the same file
    during one task would re-pay cc_pattern_completion_recall()'s .recall()
    cost every single time. Returns True (and records `now` in
    cache[file_path]) when file_path has no entry or its entry is older than
    `ttl` seconds -- the caller should run the pattern-completion pass.
    Returns False (cache untouched) when the same file already got a pass
    within the TTL window -- the caller should skip straight to
    SurfacingMonitor-only context.

    UserPromptSubmit never calls this gate -- every turn's prompt warrants a
    fresh pattern-completion pass regardless of recency.
    """
    last = cache.get(file_path)
    if last is None or (now - last) > ttl:
        cache[file_path] = now
        return True
    return False


def cc_anticipate(graph, fired_node_ids, state: dict) -> None:
    """Anticipatory pre-activation for CC (#256 port, #358).

    Verbatim port of canonical _anticipate() (neurograph_rpc.py:2716-2742)
    with two mandated differences (law-review C1): the primed dict lives in
    the caller's state dict — NOT a module global (cc_ng_host runs inside
    Syl's process, where the _primed_nodes global is HERS) — and it walks
    only the graph argument passed in, never _memory.graph.

    Walks outgoing synapses from the just-fired set, scores neighbors by
    accumulated edge weight, stores top-K with a TTL. The rebuilt
    cc_pattern_completion_recall() applies _CC_ANTICIPATE_BONUS to surfaced
    nodes still in the live primed set.
    """
    try:
        if not fired_node_ids or graph is None:
            state["primed_nodes"] = {}
            return
        fired_set = set(fired_node_ids)
        candidates = {}
        for nid in fired_node_ids:
            for sid in graph._outgoing.get(nid, ()):
                syn = graph.synapses.get(sid)
                if syn is None:
                    continue
                target = syn.post_node_id
                if target not in fired_set and target in graph.nodes:
                    candidates[target] = candidates.get(target, 0.0) + syn.weight
        top_k = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:_CC_ANTICIPATE_TOP_K]
        expiry = time.time() + _CC_ANTICIPATE_TTL_S
        state["primed_nodes"] = {nid: (score, expiry) for nid, score in top_k}
        if state["primed_nodes"]:
            logger.debug("CC anticipatory pre-activation: primed %d nodes", len(state["primed_nodes"]))
    except Exception as exc:
        logger.debug("cc_anticipate failed (non-fatal): %s", exc)


# --- Retrieval-enrichment constants (#358) ---
# Copied VERBATIM from canonical neurograph_rpc.py (C5 — do not tune here;
# canonical is the source of truth, test_cc_retrieval_enrichment pins these):
#   _ANTICIPATE_TOP_K/_ANTICIPATE_TTL_S/_ANTICIPATE_BONUS — rpc.py:263-264, :674
#   _GSG_LAYER_NORMS/_GSG_SCORE_BONUS — rpc.py GSG Phase 1 block
#   novelty EMA 0.9/0.1 — rpc.py:3270-3271
_CC_ANTICIPATE_TOP_K = 15
_CC_ANTICIPATE_TTL_S = 120.0
_CC_ANTICIPATE_BONUS = 0.25
_CC_GSG_LAYER_NORMS = (0.70, 0.50, 0.30)   # diffpc_layer 0/1/2 -> Poincaré norm
_CC_GSG_SCORE_BONUS = 0.30
_CC_NOVELTY_EMA_KEEP = 0.9
_CC_NOVELTY_EMA_GAIN = 0.1
# Copied from neurograph_rpc.py's _LENIA_CHECKPOINT_INTERVAL_SECS (2026-07-06) —
# same cadence for CC's rebuilds as Syl's. Not a new invented value.
_CC_LENIA_CHECKPOINT_INTERVAL_SECS: float = 300.0

# --- Pith Stage 4 (#55) predictive promotion -- LAW 5 env knobs ---
# Gated OFF by default: unset -> cc_pattern_completion_recall's promotion
# block never fires, byte-identical to the pre-Stage-4 bonus-only behavior.
# See docs/superpowers/plans/2026-07-22-pith-stage4-spec.md sec 4/5a.
_CC_PITH_PREFETCH_ENABLED = os.environ.get("CC_PITH_PREFETCH_ENABLED", "0") not in ("0", "false", "False", "")
# Poincare/angular distance (see _cc_node_query_distance) beyond which a
# promoted-but-unsurfaced predicted node is staged as a keyframe summary
# instead of full content (proximity-keyed LOD, spec sec 4c) -- near
# predictions are trusted at full resolution, far ones cost less if wrong.
_CC_PITH_PREFETCH_LOD_DIST = float(os.environ.get("CC_PITH_PREFETCH_LOD_DIST", "1.5"))
_CC_PITH_PREFETCH_SUMMARY_CHARS = max(60, min(1000, int(os.environ.get("CC_PITH_PREFETCH_SUMMARY_CHARS", "150"))))


def pith_prefetch_seed(state) -> Dict[str, float]:
    """Stage 4 phase 5b (#55): the Markov-prefetch seed for the Tonic tick.

    Returns {node_id: primed_score} for the still-live entries of
    ``state["primed_nodes"]`` (written by cc_anticipate at deposit, TTL'd).
    Hosts wrap this over their own conv_state and hand it to
    TonicEngine.set_prefetch_seed; the engine folds it into its write-mode
    prime so the predicted neighbourhood is warm before the next harvest.
    Scores are normalised to [0,1] by the set's max. Pure read; never raises
    (a failure is counted via PithMetrics.record_failure).
    """
    try:
        now = time.time()
        live = {nid: float(s) for nid, (s, exp) in ((state or {}).get("primed_nodes") or {}).items()
                if exp > now}
        # cc_anticipate scores are unbounded synapse-weight sums, not [0,1]; normalise
        # by the set's max so the engine's score-scaled current actually differentiates.
        top = max(live.values()) if live else 0.0
        return {nid: (s / top if top > 0 else 0.0) for nid, s in live.items()}
    except Exception:
        _PITH_METRICS.record_failure()
        return {}


def _cc_poincare_distance(x, y) -> float:
    """Geodesic distance in the Poincaré ball — verbatim port of canonical's
    _poincare_distance (neurograph_rpc.py:2660-2681), free function form.
    d(x, y) = acosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)))."""
    import numpy as _np
    import math
    nx2 = min(float(_np.dot(x, x)), 0.9999)
    ny2 = min(float(_np.dot(y, y)), 0.9999)
    diff = x - y
    num = 2.0 * float(_np.dot(diff, diff))
    denom = (1.0 - nx2) * (1.0 - ny2)
    arg = 1.0 + num / max(denom, 1e-9)
    return math.acosh(max(1.0, arg))


_CC_GSG_MSG_DECAY = 0.15   # matches neuro_foundation._GSG_MSG_DECAY


def cc_reground_synapse_delays(graph, only_nodes=None, dry_run=True) -> dict:
    """Recompute synaptic delay from geodesic distance for synapses that took
    the random fallback because an endpoint had no geometry at sprout time.

    Delay is assigned ONCE, at sprout, from whatever geometry the endpoints had
    then (neuro_foundation._sprout_synapses). Nodes stamped later keep whatever
    random.randint() gave them. That is not preserved temporal structure -- it
    is noise that never meant anything -- so regrounding it puts the delay where
    it would already have been had the geometry been there, which is the whole
    point. Weights are untouched; STDP re-shapes them against correct arrival
    times from here.

    Mirrors the sprout-site formula exactly:
        t     = 1 - exp(-_GSG_MSG_DECAY * geodesic)
        delay = clamp(d_min + (d_max - d_min) * t, d_min, d_max)
    Cross-manifold pairs have no shared geodesic and are left alone, as at sprout.

    only_nodes: restrict to synapses touching these node ids (e.g. the Rim and
    the WANTs). None = every eligible synapse.
    dry_run=True reports without mutating. Returns a stats dict.
    """
    with _cc_mutation_lock(graph):
        import math
        from neuro_foundation import poincare_dir_array
        stats = {"examined": 0, "eligible": 0, "changed": 0, "skipped_no_geometry": 0,
                 "skipped_cross_manifold": 0, "unchanged": 0, "delta_histogram": {}}
        try:
            d_min = int(graph.config.get("d_min", 1))
            d_max = int(graph.config.get("d_max", 5))
            for syn in list(getattr(graph, "synapses", {}).values()):
                stats["examined"] += 1
                pre_id = getattr(syn, "pre_node_id", None)
                post_id = getattr(syn, "post_node_id", None)
                if only_nodes is not None and pre_id not in only_nodes and post_id not in only_nodes:
                    continue
                pre, post = graph.nodes.get(pre_id), graph.nodes.get(post_id)
                if pre is None or post is None:
                    continue
                a = poincare_dir_array(getattr(pre, "metadata", None))
                b = poincare_dir_array(getattr(post, "metadata", None))
                if a is None or b is None:
                    stats["skipped_no_geometry"] += 1
                    continue
                mt1 = getattr(pre, "manifold_type", "hyperbolic")
                mt2 = getattr(post, "manifold_type", "hyperbolic")
                if mt1 != mt2:
                    stats["skipped_cross_manifold"] += 1
                    continue
                import numpy as _np
                if mt1 == "spherical":
                    cos = max(-1.0 + 1e-7, min(1.0 - 1e-7, float(_np.dot(a, b))))
                    gdist = math.acos(cos)
                else:
                    l1 = max(0, min(2, getattr(pre, "diffpc_layer", 2)))
                    l2 = max(0, min(2, getattr(post, "diffpc_layer", 2)))
                    gdist = _cc_poincare_distance(a * _CC_GSG_LAYER_NORMS[l1],
                                                  b * _CC_GSG_LAYER_NORMS[l2])
                stats["eligible"] += 1
                t = 1.0 - math.exp(-_CC_GSG_MSG_DECAY * gdist)
                new_delay = max(d_min, min(d_max, round(d_min + (d_max - d_min) * t)))
                old_delay = getattr(syn, "delay", None)
                if old_delay == new_delay:
                    stats["unchanged"] += 1
                    continue
                key = f"{old_delay}->{new_delay}"
                stats["delta_histogram"][key] = stats["delta_histogram"].get(key, 0) + 1
                stats["changed"] += 1
                if not dry_run:
                    syn.delay = new_delay
            return stats
        except Exception as exc:
            logger.debug("cc_reground_synapse_delays failed (non-fatal): %s", exc)
            stats["error"] = str(exc)
            return stats


def _cc_node_query_distance(node, query_dir) -> Optional[float]:
    """Geodesic (hyperbolic) or angular (spherical) distance between a node's
    stamped GSG direction and a query direction -- MIRRORS the per-node branch
    cc_gsg_rescore computes for its bonus (kept as a separate copy, NOT a
    factor-out: cc_gsg_rescore is a verbatim #358 canonical port and is left
    untouched). If the two ever need to diverge-proof, refactor both together.
    Lets Pith Stage 4's proximity-keyed LOD staging (#55, spec sec 4c)
    threshold on the same distance without re-deriving the manifold branch.

    Returns None when the node has no 'poincare_dir' stamp (nothing to
    compare -- caller treats that as "can't tell, don't downgrade") or on any
    error; never raises.
    """
    try:
        # #400: metadata holds packed float32 bytes now, or a legacy list on an
        # un-backfilled checkpoint; poincare_dir_array() normalises both.
        from neuro_foundation import poincare_dir_array as _pda
        pdir = _pda(node.metadata) if hasattr(node, "metadata") else None
        if pdir is None:
            return None
        import numpy as _np
        import math as _math
        layer = max(0, min(2, getattr(node, "diffpc_layer", 0)))
        mtype = getattr(node, "manifold_type", "hyperbolic")
        if mtype == "spherical":
            node_dir = _np.array(pdir, dtype=_np.float32)
            cos = float(_np.clip(_np.dot(query_dir, node_dir), -1.0 + 1e-7, 1.0 - 1e-7))
            return _math.acos(cos)
        node_pt = _np.array(pdir) * _CC_GSG_LAYER_NORMS[layer]
        query_pt = _np.array(query_dir) * _CC_GSG_LAYER_NORMS[0]
        return _cc_poincare_distance(query_pt, node_pt)
    except Exception:
        return None


def cc_novelty(state: dict, graph) -> float:
    """Pull-based MMN novelty for CC's surfacing (#255 parity, #358).

    Canonical updates _substrate_novelty_ema push-style per turn in
    handle_after_turn() (rpc.py:3661-3667) from StepResult's HE-level
    prediction counts. CC's deposits run the dual pass, not on_message(), and
    step only through cc_deposit_step (CC_NG_DEPOSIT_STEP, default off) or the
    Tonic's autostep; neither pushes those stats -- so CC dips the bucket at
    extraction time instead: read the HE-level CUMULATIVE counters, delta
    them against the previous recall, EMA the windowed surprise ratio.

    Counter names (C3, verified 2026-07-07): graph._total_confirmed /
    graph._total_surprised (neuro_foundation.py:1434-1435, incremented
    :2192/:2206) — the cumulative counterparts of StepResult.
    predictions_confirmed/predictions_surprised (:2224-2225). Private-
    prefixed but a de-facto stable contract: serialized in every checkpoint
    as he_total_confirmed/he_total_surprised (:4326-4327). NOT the same
    family as Telemetry.total_predictions_* (Phase-3 synapse-level).

    Fails soft: missing counters (engine contract change) -> current EMA or
    0.5, never raises. test_novelty_counters_exist_on_real_graph makes that
    contract change loud in CI.
    """
    try:
        confirmed = getattr(graph, "_total_confirmed", None)
        surprised = getattr(graph, "_total_surprised", None)
        if confirmed is None or surprised is None:
            return state.get("novelty_ema", 0.5)
        prev_c = state.get("last_confirmed")
        prev_s = state.get("last_surprised")
        state["last_confirmed"] = confirmed
        state["last_surprised"] = surprised
        if prev_c is None or prev_s is None:
            return state.get("novelty_ema", 0.5)   # first call = baseline only
        d_c = confirmed - prev_c
        d_s = surprised - prev_s
        if d_c + d_s > 0:
            raw = d_s / (d_c + d_s)
            state["novelty_ema"] = (_CC_NOVELTY_EMA_KEEP * state.get("novelty_ema", 0.5)
                                    + _CC_NOVELTY_EMA_GAIN * raw)
        return state.get("novelty_ema", 0.5)
    except Exception as exc:
        logger.debug("cc_novelty failed (non-fatal): %s", exc)
        return state.get("novelty_ema", 0.5) if isinstance(state, dict) else 0.5


def cc_gsg_rescore(surfaced, query_text: str, graph):
    """GSG geodesic re-scoring for CC's surfacing (#358) — port of canonical
    handle_assemble()'s GSG block (neurograph_rpc.py:2991-3038), parameterized
    on graph (C1). Nodes geometrically close to the query in Poincaré-ball /
    spherical space get a strength bonus (max _CC_GSG_SCORE_BONUS as dist->0);
    list re-sorted once if any bonus applied. Fails soft: any error returns
    the list un-rescored (canonical wraps identically).
    """
    try:
        if not surfaced or not query_text or graph is None:
            return surfaced
        import numpy as _np
        import math as _math
        from ng_embed import embed as _embed
        query_emb = _embed(query_text)
        query_dir = _cc_embed_to_poincare_dir(query_emb)
        query_pt = query_dir * _CC_GSG_LAYER_NORMS[0]      # fresh query = Layer 0
        applied = 0
        for item in surfaced:
            nid = item.get("node_id")
            if nid is None:
                continue
            node = graph.nodes.get(nid)
            if node is None:
                continue
            from neuro_foundation import poincare_dir_array as _pda  # #400 bytes-or-list
            pdir = _pda(node.metadata) if hasattr(node, "metadata") else None
            if pdir is None:
                continue
            layer = max(0, min(2, getattr(node, "diffpc_layer", 0)))
            mtype = getattr(node, "manifold_type", "hyperbolic")
            if mtype == "spherical":
                node_dir = _np.array(pdir, dtype=_np.float32)
                cos = float(_np.clip(_np.dot(query_dir, node_dir), -1.0 + 1e-7, 1.0 - 1e-7))
                bonus = _CC_GSG_SCORE_BONUS / (1.0 + _math.acos(cos))
            else:
                node_pt = _np.array(pdir) * _CC_GSG_LAYER_NORMS[layer]
                bonus = _CC_GSG_SCORE_BONUS / (1.0 + _cc_poincare_distance(query_pt, node_pt))
            item["strength"] = item.get("strength", 0.0) + bonus
            applied += 1
        if applied:
            surfaced.sort(key=lambda x: x.get("strength", 0.0), reverse=True)
            logger.debug("CC GSG re-scoring applied to %d nodes", applied)
        return surfaced
    except Exception as exc:
        logger.debug("cc_gsg_rescore skipped (non-fatal): %s", exc)
        return surfaced


def cc_stamp_missing_geometry(graph, vector_db=None) -> int:
    """Stamp poincare_dir on CC nodes that lack it, FROM THE NODE'S OWN CONTENT.

    poincare_dir is SNN geometry. It belongs to the node and is derived from the
    node's own `_forest_content`, which is already in the substrate — the vdb has
    no part in it. The previous implementation sourced the direction from
    `vector_db.embeddings`, which was never asked for and was wrong: it made the
    substrate's geometry depend on a secondary store, and it silently produced
    nothing for any node the vdb had no row for.

    Also performs the #400 one-time migration of legacy boxed-list poincare_dir
    to compact float32 bytes, in place.

    STAMP-ONLY — no save() (law-review C2, CRITICAL): canonical force-saves after
    stamping, but on the VPS this runs inside Syl's process, and a ported save
    mis-bound to the wrong instance is exactly what Syl's Law exists to prevent.
    The daemons' normal autosave persists the metadata.

    `vector_db` is accepted and ignored; it remains in the signature only so the
    two existing call sites keep working. Fails soft, returns count touched.
    """
    try:
        if graph is None:
            return 0
        from neuro_foundation import pack_poincare_dir as _pack  # #400
        stamped = 0
        packed = 0
        with _cc_mutation_lock(graph):
            geometry_nodes = [(nid, node, dict(node.metadata or {}))
                              for nid, node in graph.nodes.items()]
        for node_id, node, captured_metadata in geometry_nodes:
            _existing = captured_metadata.get("poincare_dir")
            if _existing is not None:
                # #400 one-time migration, mirroring canonical
                # neurograph_rpc._gsg_backfill_existing_nodes: a legacy boxed list
                # becomes compact float32 bytes in place. Without this the CC half
                # keeps ~24 KB/node forever -- the old stamp-only guard skipped
                # every already-stamped node regardless of its storage form.
                if not isinstance(_existing, (bytes, bytearray)):
                    try:
                        packed_direction = _pack(_existing)
                        with _cc_mutation_lock(graph):
                            if (graph.nodes.get(node_id) is node
                                    and (node.metadata or {}).get("poincare_dir") is _existing):
                                node.metadata["poincare_dir"] = packed_direction
                                packed += 1
                    except Exception:
                        pass
                continue
            # SNN-native source: the node's own text, embedded here. No vdb.
            # A node's text does not always live in _forest_content -- a want
            # carries want_text, a constitutional rim node carries core_text.
            # Looking only at _forest_content is why this never converged: 216
            # of the 217 unstamped nodes (182 wants + the Choice Clause) were
            # skipped every boot forever, so wants and the rim had no geometry
            # and could not participate in GSG proximity at all.
            _md = captured_metadata
            # ORDER MATTERS. A tree node carries BOTH its own `_concept` and the
            # parent turn's `_forest_content` -- every tree under one forest shares
            # the latter. Reading _forest_content first would embed the parent turn
            # for all of them and collapse an entire forest's trees onto one
            # identical direction, destroying the geometry the live deposit path
            # builds correctly (verified on the VPS: sibling trees share
            # _forest_content but have DIFFERENT poincare_dir). The tree's own
            # concept is its own meaning, so it wins.
            content = (_md.get("_concept") if _md.get("_tree_concept") else None) \
                or _md.get("_forest_content") or _md.get("want_text") or _md.get("core_text")
            if not content:
                continue
            try:
                from ng_embed import embed as _embed
                direction = _cc_embed_to_poincare_dir(_embed(str(content)))
            except Exception as exc:
                logger.debug("GSG backfill embed failed for %r (non-fatal): %s", node_id, exc)
                continue
            if direction is None:
                continue
            packed_direction = _pack(direction)
            with _cc_mutation_lock(graph):
                live_metadata = node.metadata or {}
                if (graph.nodes.get(node_id) is not node
                        or live_metadata.get("poincare_dir") is not None
                        or any(live_metadata.get(key) != captured_metadata.get(key)
                               for key in ("_tree_concept", "_concept", "_forest_content", "want_text", "core_text"))):
                    continue
                if node.metadata is None:
                    node.metadata = {}
                node.metadata["poincare_dir"] = packed_direction
                stamped += 1
        if stamped or packed:
            logger.info("CC GSG backfill: stamped %d node(s) from their own _forest_content, "
                        "migrated %d legacy list(s) -> packed float32 bytes "
                        "(#400; stamp-only, persists via normal autosave)", stamped, packed)
        return stamped + packed
    except Exception as exc:
        logger.debug("cc_stamp_missing_geometry failed (non-fatal): %s", exc)
        return 0


# =============================================================================
# Pith extraction pipeline -- Phase 0 (CacheLine + metrics scaffold) + Phase 1
# (Stage 1: ingest & clutter strip). Gated OFF by default (CC_PITH_ENABLED);
# see the 2026-07-08 changelog entry at the top of this file.
# =============================================================================

_CC_PITH_ENABLED = os.environ.get("CC_PITH_ENABLED", "0") not in ("0", "false", "False", "")

_CC_PITH_CLUTTER_BASE = float(os.environ.get("CC_PITH_CLUTTER_BASE", "0.85"))
_CC_PITH_CLUTTER_NOVELTY_K = float(os.environ.get("CC_PITH_CLUTTER_NOVELTY_K", "0.3"))

# Stage 3 (unified rank + char budget) config -- LAW 5, env-config with sane
# defaults, clamped. Relevance (pattern-completion / Active Recall) weighted
# above recency (SurfacingMonitor) by default -- recency is a secondary prior
# to relevance, not an equal signal.
_CC_PITH_W_RELEVANCE = float(os.environ.get("CC_PITH_W_RELEVANCE", "1.0"))
_CC_PITH_W_RECENCY = float(os.environ.get("CC_PITH_W_RECENCY", "0.6"))
_CC_PITH_L1_BUDGET = int(os.environ.get("CC_PITH_L1_BUDGET", "4000"))
_CC_PITH_L1_BUDGET = max(500, min(40000, _CC_PITH_L1_BUDGET))

# Provider-context Slice A.  These are extraction-boundary attention limits,
# not deposit schemas.  Raw experience still enters the substrate unchanged.
_CC_PITH_PROVIDER_ROOTS = max(1, min(24, int(os.environ.get("CC_PITH_PROVIDER_ROOTS", "8"))))
_CC_PITH_PROVIDER_MEMBERS = max(2, min(16, int(os.environ.get("CC_PITH_PROVIDER_MEMBERS", "6"))))
_CC_PITH_PROVIDER_DEPTH = max(1, min(3, int(os.environ.get("CC_PITH_PROVIDER_DEPTH", "2"))))
_CC_PITH_PROVIDER_NODE_CHARS = max(120, min(2000, int(os.environ.get("CC_PITH_PROVIDER_NODE_CHARS", "700"))))
_CC_PITH_PROVIDER_MAX_INSTRUCTION_CHARS = max(
    500, min(16000, int(os.environ.get("CC_PITH_PROVIDER_MAX_INSTRUCTION_CHARS", "8000"))))
_CC_PITH_PROVIDER_MAX_QUEST_CHARS = max(
    0, min(16000, int(os.environ.get("CC_PITH_PROVIDER_MAX_QUEST_CHARS", "8000"))))

# Stage 2 (keyframe / LOD compression) config -- default keyframe size in
# chars, clamped [60, 1000]. See pith_stage2_keyframe().
_CC_PITH_KEYFRAME_CHARS = max(60, min(1000, int(os.environ.get("CC_PITH_KEYFRAME_CHARS", "220"))))

# Stage 5 (eviction & recapture) config -- LAW 5. Thermal is a continuous
# warmth signal read from the substrate's OWN state (Ca_i persistence-of-
# activation + firing_rate_ema + Lenia field energy), blended + min-max
# normalized across the survivor set, then folded into the Stage-3 rank as a
# gentle multiplier (1 + gain*thermal) so warm content is preferred without
# overriding relevance. GAIN default keeps thermal a tiebreak, not a driver.
# thermal defaults 0.0 on every CacheLine, so an un-populated line leaves the
# Stage-3 rank byte-identical -- Stage 5 is additive/opt-in.
_CC_PITH_THERMAL_W_CA = float(os.environ.get("CC_PITH_THERMAL_W_CA", "0.5"))
_CC_PITH_THERMAL_W_FIRE = float(os.environ.get("CC_PITH_THERMAL_W_FIRE", "0.3"))
_CC_PITH_THERMAL_W_FIELD = float(os.environ.get("CC_PITH_THERMAL_W_FIELD", "0.2"))
_CC_PITH_THERMAL_GAIN = float(os.environ.get("CC_PITH_THERMAL_GAIN", "0.5"))
_CC_PITH_VICTIM_SIZE = max(0, min(128, int(os.environ.get("CC_PITH_VICTIM_SIZE", "12"))))
_CC_PITH_VICTIM_TTL = max(1, int(os.environ.get("CC_PITH_VICTIM_TTL", "20")))

# Autonomic breathing (Pith §3.2): L1 budget expands under PARASYMPATHETIC
# (exploratory/associative) and contracts under SYMPATHETIC (threat / tunnel
# vision), reading the arousal Immunis deposits to the CC Commons. Gated
# (CC_PITH_L1_BREATHE, default off); when off, the static budget is used.
_CC_PITH_L1_BREATHE = os.environ.get("CC_PITH_L1_BREATHE", "0") not in ("0", "false", "False", "")
_CC_PITH_BREATHE_SYMPATHETIC = float(os.environ.get("CC_PITH_BREATHE_SYMPATHETIC", "0.6"))
_CC_PITH_BREATHE_PARASYMPATHETIC = float(os.environ.get("CC_PITH_BREATHE_PARASYMPATHETIC", "1.4"))

# Shared Graduation (COMB-04): region confidence signal from the full NeuroGraph.
# Gated (CC_PITH_REGION_CONFIDENCE_ENABLED, default off); when off, region confidence
# is neutral (0.5). Region confidence modulates the L1 budget alongside arousal.
# The region is what fired for this cue, so no search size or floor is tuned here.
_CC_PITH_REGION_CONFIDENCE_ENABLED = os.environ.get("CC_PITH_REGION_CONFIDENCE_ENABLED", "0") not in ("0", "false", "False", "")
_CC_PITH_REGION_CONFIDENCE_NEUTRAL = 0.5  # neutral confidence when disabled or on error (midpoint of [0,1])
_CC_PITH_REGION_CONFIDENCE_FALLOFF = float(os.environ.get("CC_PITH_REGION_CONFIDENCE_FALLOFF", "0.25"))

# Same marker tuple as miniTID's is_synthetic_harness_text (Condensate
# rust_core/src/minitid.rs) -- not importable here (Rust, separate process),
# so inlined verbatim rather than left unguarded on the extraction side.
_PITH_HARNESS_MARKERS = (
    "<task-notification>",
    "<system-reminder>",
    "<local-command-stdout>",
    "<local-command-caveat>",
)


@dataclass
class CacheLine:
    """Cache-line-shaped view of one surfaced item, moving through the Pith
    pipeline's stages. Fields with production readers today (each reader
    cited by function name):
    - `pinned`: the stage3 pinned/unpinned split (pith_stage3) and the
      victim-capture exclusion (pith_victim_capture).
    - `thermal`: the stage3 rank fold (pith_stage3), the basin competition
      (pith_connected_activation_basins) and the victim round-trip
      (pith_victim_capture / pith_victim_recover).
    - `coherence`: the basin competition tie-break
      (pith_connected_activation_basins), the provider render label
      (_pith_render_connected_line), the provider warnings/envelope rollups
      (_pith_provider_sections, pith_provider_context) and the victim
      round-trip (pith_victim_capture / pith_victim_recover).
    - `stream`: per-stream score normalization and the D5 promotable-stream
      tally (pith_stage3) and the victim-capture exclusion
      (pith_victim_capture).
    - `prefetch_origin`: the D5 promotable tally (pith_stage3) -- provenance
      only, never scoring (see its own comment below).
    The routing fields (`node_id`, `content`, `score`) and the basin
    structure (`member_node_ids`, `relations`, `sources`, `anchors`) are read
    by their owning stages (pith_stage1's dedup/clutter-strip; the provider
    builder/renderer/fitter and anchor rollup). `epistemic` is set by the
    basin builder and has no production reader yet.

    score carries the emitter's existing score (SurfacingMonitor's salience
    or cc_pattern_completion_recall's strength) verbatim -- Pith re-ranks and
    filters, it does not re-derive relevance from scratch.

    stream tags which emitter this line came from (e.g. "monitor" for
    SurfacingMonitor recency, "pattern" for Active Recall / GSG-rescored
    relevance) -- Stage 3 (pith_stage3) uses it for per-stream score
    normalization, since the two emitters' raw score scales are not
    comparable. Defaults to "recall" (generic/unknown-stream) so existing
    callers/tests that don't pass it are unaffected.
    """

    node_id: str
    content: str
    score: float = 0.0
    pinned: bool = False
    thermal: float = 0.0
    # Missing coherence evidence is unknown, never an inferred exclusive state.
    coherence: str = "unknown"
    stream: str = "recall"
    # [D5] Provenance ONLY. Set when a line originates from Stage-4 predictive
    # promotion. It takes no part in scoring, normalization, weighting, sorting,
    # dedup or budget arithmetic -- deliberately NOT folded into `stream`, whose
    # per-stream min-max in pith_stage3 would give a singleton "prefetch"
    # population a normalized 1.0 (top-of-stream) and thereby promote exactly the
    # lines it is meant to count. Read at one counting site only.
    prefetch_origin: bool = False
    # Slice A: one provider-facing line is a connected activation basin.  The
    # root carries the assembly's own content; related observations remain
    # attached as `relations` so action -> outcome -> correction cannot be
    # admitted as orphan fragments.
    member_node_ids: list = field(default_factory=list)
    relations: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    anchors: list = field(default_factory=list)
    epistemic: str = "learned"

    @classmethod
    def from_surfaced(cls, node_id: str, content: str, score: float = 0.0,
                       pinned: bool = False,
                       stream: str = "recall",
                       prefetch_origin: bool = False) -> "CacheLine":
        return cls(node_id=node_id, content=content, score=score, pinned=pinned,
                    stream=stream,
                    prefetch_origin=prefetch_origin)


@dataclass
class PithMetrics:
    """Module-level counters for the Pith pipeline -- inert until a gated
    path calls them (inbound L1 assembly or outbound miniTID history compression).
    Each caller owns its own gate; this metrics object does not enforce one."""

    total_lines_in: int = 0
    clutter_stripped: int = 0
    combined: int = 0
    pith_failures: int = 0
    ranked_in: int = 0
    ranked_kept: int = 0
    ranked_dropped: int = 0
    budget_chars_used: int = 0
    compressed_count: int = 0
    chars_saved: int = 0
    # [D4c] OUTBOUND miniTID history-compression path. Kept separate from the
    # Stage-3 inbound L1 counters above so acceptance can prove which boundary ran.
    history_calls: int = 0
    history_turns_in: int = 0
    history_turns_compressed: int = 0
    history_chars_in: int = 0
    history_chars_out: int = 0
    history_chars_saved: int = 0
    history_failures: int = 0
    promoted_predicted: int = 0
    prefetch_hits: int = 0
    # 5b: live primed nodes the harvest surfaced on its own (warm topology).
    prefetch_surfaced: int = 0
    # [D5] Spec sec 13.3 terms, both counted at the SAME point (final L1 assembly
    # in pith_stage3) over the SAME set, deduplicated by node_id within each
    # invocation. Dedup is counting-only: it never touches the emitted lines.
    # Numerator is a subset of denominator by construction (same filter, plus
    # prefetch_origin). PRIMARY metric = l1_prefetch_distinct / l1_kept_distinct.
    l1_kept_distinct: int = 0                  # broad denominator: all distinct nodes in L1
    l1_prefetch_distinct: int = 0              # broad numerator
    l1_kept_distinct_promotable: int = 0       # narrow denominator: excludes monitor + victim
    l1_prefetch_distinct_promotable: int = 0   # narrow numerator
    # [D5b] GATED L1 ASSEMBLIES -- one increment per pith_stage3 invocation. The
    # single live call site is inside cc_assemble_recall's CC_PITH_ENABLED block, so
    # this counts L1 assemblies that actually happened, NOT daemon recalls: with the
    # gate off the block is never entered and this stays 0.
    #
    # That distinction decides how a window is read. A window with the gate resolved
    # off, or with l1_assemblies == 0, is an INVALID SAMPLE for the sec 13.3 claim --
    # it is not a 0% prefetch rate. This is the denominator-of-record for the
    # acceptance bar's sample size; a ratio published without it cannot be judged.
    #
    # [D5d] Committed in the SAME lock hold as the four result counters, at the end
    # of the assembly. So l1_assemblies == 0 with any result counter > 0 is not a
    # state this code can produce: a reset always lands between assemblies, never
    # inside one, and results are never orphaned from the assembly that produced
    # them. Counting it early made exactly that state reachable.
    l1_assemblies: int = 0

    # [D5b] Guards snapshot()/reset() and the sec 13.3 counting block against each
    # other. Without it a snapshot concurrent with a reset can return a MIX of
    # pre- and post-reset fields (a torn read), which for a ratio means a numerator
    # from one instant over a denominator from another -- silently out of range
    # rather than obviously broken. RLock so a future nested use cannot self-deadlock.
    # NOT part of equality/repr: it is machinery, not measured state.
    _lock: "threading.RLock" = field(default_factory=threading.RLock, repr=False, compare=False)

    def reset(self) -> None:
        """[D5b] Atomic with respect to snapshot() -- a snapshot concurrent with a
        reset returns either the whole pre-reset view or the whole post-reset one,
        never a mixture of the two."""
        with self._lock:
            self._reset_locked()

    def _reset_locked(self) -> None:
        self.total_lines_in = 0
        self.clutter_stripped = 0
        self.combined = 0
        self.pith_failures = 0
        self.ranked_in = 0
        self.ranked_kept = 0
        self.ranked_dropped = 0
        self.budget_chars_used = 0
        self.compressed_count = 0
        self.chars_saved = 0
        self.history_calls = 0
        self.history_turns_in = 0
        self.history_turns_compressed = 0
        self.history_chars_in = 0
        self.history_chars_out = 0
        self.history_chars_saved = 0
        self.history_failures = 0
        self.promoted_predicted = 0
        self.prefetch_hits = 0
        self.prefetch_surfaced = 0
        self.l1_kept_distinct = 0
        self.l1_assemblies = 0
        self.l1_prefetch_distinct = 0
        self.l1_kept_distinct_promotable = 0
        self.l1_prefetch_distinct_promotable = 0

    def record_failure(self) -> None:
        """Bump the fail-soft counter -- a failing Pith path returns its failure
        envelope (cc_assemble_recall: the unavailable notice, never the
        un-Pithed rendering) or an empty result (pith_prefetch_seed), so
        without this a 100%-failing Pith pass is indistinguishable from a
        working one. Call from the caller's except-handler."""
        with self._lock:
            self.pith_failures += 1

    def record_history_compression(self, *, turns_in: int, turns_compressed: int,
                                   chars_in: int, chars_out: int,
                                   failures: int) -> None:
        """Commit one outbound history result as a coherent telemetry unit."""
        with self._lock:
            self.history_calls += 1
            self.history_turns_in += turns_in
            self.history_turns_compressed += turns_compressed
            self.history_chars_in += chars_in
            self.history_chars_out += chars_out
            self.history_chars_saved += max(0, chars_in - chars_out)
            self.history_failures += failures
            self.pith_failures += failures

    def snapshot(self) -> Dict[str, int]:
        """A point-in-time view. Read the guarantee carefully -- it is NARROW.

        COHERENT: the sec 13.3 terms (l1_kept_distinct, l1_prefetch_distinct, and
        their _promotable pair) with l1_assemblies, plus all history_* terms.
        Each group is committed under this lock, so a reader cannot observe half
        of one L1 assembly or one outbound history result.

        BEST-EFFORT: every legacy counter (ranked_in, ranked_kept, ranked_dropped,
        prefetch_hits, promoted_predicted, prefetch_surfaced, ...). Their writers
        do NOT take this lock, so holding it here does not make them atomic. Do not
        treat them as consistent with the sec 13.3 terms or with each other.

        They are deliberately left unlocked: they are hot-path `+=` on counters no
        acceptance figure is computed from, and widening the lock to cover them
        would add contention to every recall to make this docstring shorter.
        """
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> Dict[str, int]:
        return {
            "total_lines_in": self.total_lines_in,
            "clutter_stripped": self.clutter_stripped,
            "combined": self.combined,
            "pith_failures": self.pith_failures,
            "ranked_in": self.ranked_in,
            "ranked_kept": self.ranked_kept,
            "ranked_dropped": self.ranked_dropped,
            "budget_chars_used": self.budget_chars_used,
            "compressed_count": self.compressed_count,
            "chars_saved": self.chars_saved,
            "history_calls": self.history_calls,
            "history_turns_in": self.history_turns_in,
            "history_turns_compressed": self.history_turns_compressed,
            "history_chars_in": self.history_chars_in,
            "history_chars_out": self.history_chars_out,
            "history_chars_saved": self.history_chars_saved,
            "history_failures": self.history_failures,
            "promoted_predicted": self.promoted_predicted,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_surfaced": self.prefetch_surfaced,
            "l1_kept_distinct": self.l1_kept_distinct,
            "l1_prefetch_distinct": self.l1_prefetch_distinct,
            "l1_kept_distinct_promotable": self.l1_kept_distinct_promotable,
            "l1_prefetch_distinct_promotable": self.l1_prefetch_distinct_promotable,
            "l1_assemblies": self.l1_assemblies,
        }


_PITH_METRICS = PithMetrics()


# [D5b] Keys reported by pith_effective_config(). Explicit ALLOW-LIST: telemetry
# must never carry the whole environment, which holds tokens and paths.
_PITH_CONFIG_KEYS = (
    "CC_PITH_ENABLED", "CC_PITH_L1_BUDGET", "CC_PITH_L1_BREATHE",
    "CC_PITH_KEYFRAME_CHARS",
    "CC_PITH_PROVIDER_ROOTS", "CC_PITH_PROVIDER_MEMBERS",
    "CC_PITH_PROVIDER_DEPTH", "CC_PITH_PROVIDER_NODE_CHARS",
    "CC_PITH_PROVIDER_MAX_INSTRUCTION_CHARS", "CC_PITH_PROVIDER_MAX_QUEST_CHARS",
    "CC_PITH_PREFETCH_ENABLED", "CC_PITH_PREFETCH_WARM_ENABLED",
    "CC_PITH_PREFETCH_MAX", "CC_PITH_PREFETCH_REPEATS",
    "CC_PITH_PREFETCH_CURRENT_SCALE", "CC_PITH_PREFETCH_LOD_DIST",
)


def pith_effective_config() -> Dict[str, Dict]:
    """The configuration this process is ACTUALLY running under.

    Returns {"env": {...}, "resolved": {...}, "authority": {...}} where:

    * ``env``      -- the raw allow-listed environment strings, ``None`` where a
                      variable is unset. What the shell handed this process.
    * ``resolved`` -- the module-level constants the code branches on, after
                      defaulting, type coercion and CLAMPING. What actually runs.
    * ``authority`` -- which module resolved each setting, so a reader can go
                      check the source rather than trust this dict.

    The distinction is not cosmetic. ``CC_PITH_PREFETCH_MAX=999`` resolves to 64
    (clamped), an unset ``CC_PITH_PREFETCH_ENABLED`` resolves to False, and a
    stale-shell launch shows ``env=None`` against a ``resolved`` default -- so a
    window recorded with raw env alone cannot distinguish "gate off" from "gate
    on by default", which is exactly the degraded-daemon trap this exists to
    close (the hook passes only ET_TRACTS_DIR and setdefaults CC_PITH_ENABLED).

    Values are read at import time by their owning module, so this reports what
    THIS process resolved -- editing .bashrc does not change a running daemon.
    Never raises: a missing tonic_engine yields None for its four settings rather
    than sinking a snapshot.
    """
    env = {k: os.environ.get(k) for k in _PITH_CONFIG_KEYS}

    resolved = {
        "CC_PITH_ENABLED": _CC_PITH_ENABLED,
        "CC_PITH_L1_BUDGET": _CC_PITH_L1_BUDGET,
        "CC_PITH_L1_BREATHE": _CC_PITH_L1_BREATHE,
        "CC_PITH_KEYFRAME_CHARS": _CC_PITH_KEYFRAME_CHARS,
        "CC_PITH_PROVIDER_ROOTS": _CC_PITH_PROVIDER_ROOTS,
        "CC_PITH_PROVIDER_MEMBERS": _CC_PITH_PROVIDER_MEMBERS,
        "CC_PITH_PROVIDER_DEPTH": _CC_PITH_PROVIDER_DEPTH,
        "CC_PITH_PROVIDER_NODE_CHARS": _CC_PITH_PROVIDER_NODE_CHARS,
        "CC_PITH_PROVIDER_MAX_INSTRUCTION_CHARS": _CC_PITH_PROVIDER_MAX_INSTRUCTION_CHARS,
        "CC_PITH_PROVIDER_MAX_QUEST_CHARS": _CC_PITH_PROVIDER_MAX_QUEST_CHARS,
        "CC_PITH_PREFETCH_ENABLED": _CC_PITH_PREFETCH_ENABLED,
        "CC_PITH_PREFETCH_LOD_DIST": _CC_PITH_PREFETCH_LOD_DIST,
    }
    authority = {k: "cc_ng_organism" for k in resolved}

    # The four warm-prefetch knobs are resolved by tonic_engine, not here -- read
    # them from their owner rather than re-deriving (LAW 4: one source per value).
    try:
        import tonic_engine as _te
        resolved.update({
            "CC_PITH_PREFETCH_WARM_ENABLED": _te._CC_PITH_PREFETCH_WARM_ENABLED,
            "CC_PITH_PREFETCH_MAX": _te._CC_PITH_PREFETCH_MAX,
            "CC_PITH_PREFETCH_REPEATS": _te._CC_PITH_PREFETCH_REPEATS,
            "CC_PITH_PREFETCH_CURRENT_SCALE": _te._CC_PITH_PREFETCH_CURRENT_SCALE,
        })
        for k in ("CC_PITH_PREFETCH_WARM_ENABLED", "CC_PITH_PREFETCH_MAX",
                  "CC_PITH_PREFETCH_REPEATS", "CC_PITH_PREFETCH_CURRENT_SCALE"):
            authority[k] = "tonic_engine"
    except Exception as _exc:                      # pragma: no cover - import guard
        logger.debug("pith_effective_config: tonic_engine unavailable (%s)", _exc)
        for k in ("CC_PITH_PREFETCH_WARM_ENABLED", "CC_PITH_PREFETCH_MAX",
                  "CC_PITH_PREFETCH_REPEATS", "CC_PITH_PREFETCH_CURRENT_SCALE"):
            resolved[k] = None
            authority[k] = "tonic_engine (unavailable)"

    return {"env": env, "resolved": resolved, "authority": authority}


def _pith_normalize(text: str) -> str:
    """Lowercase + collapse whitespace -- cheap normalization shared by the
    dedup and write-combine steps below."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _pith_jaccard(a: str, b: str) -> float:
    """Word-set Jaccard overlap of two already-normalized strings. Empty/
    empty is treated as no overlap (0.0), not a division-by-zero NaN.
    Symmetric -- used for write-combine (step 3), where mutual near-identity
    is what we want."""
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _pith_containment(item: str, conv: str) -> float:
    """Fraction of ITEM's words already present in the conversation --
    |item ∩ conv| / |item|. ASYMMETRIC on purpose: it answers "does the model
    already have essentially all of this item?", not "do they overlap at all".
    A long memory that the conversation merely quotes a fragment of scores
    LOW (it still carries the rest) and is kept; a memory whose content is
    genuinely already in the conversation scores HIGH and is stripped as
    redundant. Symmetric Jaccard got this wrong -- a long item's own large
    word set sank the ratio far below threshold, so dedup almost never fired."""
    set_item = set(item.split())
    set_conv = set(conv.split())
    if not set_item or not set_conv:
        return 0.0
    return len(set_item & set_conv) / len(set_item)


def pith_stage1(cache_lines: List[CacheLine], conversation_text: str,
                 novelty: float = 0.0) -> List[CacheLine]:
    """Pith Stage 1: Ingest & Clutter Strip.

    Cheap, pure, unit-testable -- string/set ops only over the already-
    surfaced small set (typically <20 items), no I/O, no embed calls, no
    substrate walk. Three steps, in order, each skipping pinned lines
    (a pinned line always survives regardless of what steps 1-3 would
    otherwise do to it):

    1. Harness-marker skip: drop lines whose content starts (after
       lstrip()) with a synthetic-harness marker (see
       _PITH_HARNESS_MARKERS) -- extraction-side defense mirroring
       miniTID's is_synthetic_harness_text, needed because the deposit-side
       clutter strip was intentionally removed (2026-07-08, see this file's
       changelog) so harness text CAN be sitting in the substrate.
    2. Clutter dedup vs conversation_text: drop a line whose content the
       model already sees in conversation_text -- substring match (after
       normalizing both: lowercase, collapse whitespace) OR Jaccard word-
       overlap >= a novelty-modulated threshold. High novelty -> higher
       threshold -> strip LESS (an unfamiliar turn's near-echoes are more
       likely to matter); familiar/low-novelty -> strip MORE.
    3. Write-combine: collapse remaining near-identical lines (normalized-
       content match, or Jaccard >= 0.95) into one survivor, keeping the
       higher-score copy. O(n^2) over the small surfaced set is cheap and
       fine here.

    Updates the module-level _PITH_METRICS counters (total_lines_in,
    clutter_stripped, combined) unconditionally -- callers that don't want
    metrics touched should not call this function (there's no gate inside
    it; the gate lives at the _recall() call site in cc-ng-daemon.py).

    Returns survivors in input order (score ranking is the caller's/
    upstream emitter's responsibility -- this function filters, it doesn't
    re-sort).
    """
    _PITH_METRICS.total_lines_in += len(cache_lines)

    conv_norm = _pith_normalize(conversation_text)
    # NOTE (spec discrepancy, 2026-07-08): the design doc literally states
    # `thr = BASE - K * novelty`, but its own prose parenthetical on the same
    # line ("high novelty -> higher threshold -> strip LESS") and its Test 3
    # requirement ("kept at high novelty, stripped at low novelty") both
    # describe threshold INCREASING with novelty -- the opposite of what a
    # minus sign produces (BASE=0.85 - K*novelty shrinks as novelty rises,
    # which would strip MORE at high novelty). Implemented to match the
    # doubly-stated intent (+ sign), not the single literal formula line;
    # flagged for spec-author confirmation.
    thr = _CC_PITH_CLUTTER_BASE + _CC_PITH_CLUTTER_NOVELTY_K * novelty
    thr = max(0.5, min(0.98, thr))

    # Steps 1-2: harness-marker skip + clutter dedup vs conversation.
    survivors: List[CacheLine] = []
    clutter_stripped = 0
    for line in cache_lines:
        if line.pinned:
            survivors.append(line)
            continue

        stripped_content = (line.content or "").lstrip()
        if stripped_content.startswith(_PITH_HARNESS_MARKERS):
            clutter_stripped += 1
            continue

        line_norm = _pith_normalize(line.content)
        if line_norm and conv_norm:
            if line_norm in conv_norm:
                clutter_stripped += 1
                continue
            if _pith_jaccard(line_norm, conv_norm) >= thr:
                clutter_stripped += 1
                continue

        survivors.append(line)

    # Step 3: write-combine near-identical remaining lines (pinned lines
    # already passed through untouched above; re-touching them here would
    # risk a pin losing to a higher-scored non-pinned near-duplicate, so
    # pinned lines are excluded from combining entirely -- each stays its
    # own line).
    combined_out: List[CacheLine] = []
    consumed = [False] * len(survivors)
    combined_count = 0
    for i, line in enumerate(survivors):
        if consumed[i]:
            continue
        if line.pinned:
            combined_out.append(line)
            consumed[i] = True
            continue
        best = line
        best_norm = _pith_normalize(line.content)
        for j in range(i + 1, len(survivors)):
            if consumed[j] or survivors[j].pinned:
                continue
            other = survivors[j]
            other_norm = _pith_normalize(other.content)
            same = (best_norm == other_norm) or (_pith_jaccard(best_norm, other_norm) >= 0.95)
            if same:
                consumed[j] = True
                combined_count += 1
                if other.score > best.score:
                    best = other
                    best_norm = other_norm
        combined_out.append(best)
        consumed[i] = True

    _PITH_METRICS.clutter_stripped += clutter_stripped
    _PITH_METRICS.combined += combined_count

    return combined_out


# Tiny stopword set for the Stage-2 extractive keyframe -- just enough to keep
# function words from diluting a segment's salient-term density. Not exhaustive
# (this is a cheap heuristic, not an NLP pipeline).
_PITH_STOPWORDS = frozenset("""
a an the this that these those and or but if then else of to in on at by for with
from as is are was were be been being it its i you he she they we me my your our
their them us do does did done have has had will would can could should may not no
yes so than too very just about into over under out up down off also which who whom
what when where how why then there here their they're it's don't doesn't
""".split())


def _pith_salient_terms(content: str) -> Dict[str, int]:
    """Term-frequency map over `content` (lowercased alnum tokens, length >= 3,
    minus stopwords) -- the item's own recurring vocabulary. Used as the
    centroid signal for extractive keyframe selection: a segment dense in the
    item's repeated concepts is central to what the item is about."""
    freq: Dict[str, int] = {}
    for w in re.findall(r"[a-z0-9_]+", content.lower()):
        if len(w) < 3 or w in _PITH_STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1
    return freq


def _pith_segment_score(seg: str, term_freq: Dict[str, int]) -> float:
    """Informativeness of one segment (higher = keep). Combines the item's own
    salient-term density (centroid, length-normalized so long filler can't win
    on raw counts), payload-token count (numbers, dotted/underscored
    identifiers, `code`, file refs -- the parts that carry facts), and a
    structural bonus for headings / def-class signatures / labelled bullets.
    Greetings and filler score near zero without any special-casing."""
    s = seg.strip()
    if not s:
        return 0.0
    words = re.findall(r"[a-z0-9_]+", s.lower())
    if not words:
        return 0.0
    centroid = sum(term_freq.get(w, 0) for w in words) / len(words)
    payload = len(re.findall(
        r"\d+|[A-Za-z_]+[._][A-Za-z0-9_]+|`[^`]+`|#\d+", s))
    structural = 0.0
    if re.match(r"#{1,6}\s", s) or re.match(r"(?:def|class)\s+\w", s):
        structural = 2.0
    elif re.match(r"[-*]\s+\*\*", s):  # "- **Label:**" key-value bullet
        structural = 1.0
    return centroid + 0.5 * payload + structural


def _pith_cut_at_word_boundary(text: str, limit: int) -> str:
    """Hard-cut `text` to at most `limit` chars, backing up to the last space
    so the cut never lands mid-word (unless `text` has no space within the
    first `limit` chars, in which case a mid-word cut is unavoidable)."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut.rstrip()


def pith_stage2_keyframe(content: str, max_chars: Optional[int] = None,
                          query: str = "") -> tuple:
    """Pith Stage 2: keyframe / LOD compression -- concept-aware, extractive.

    Cheap, pure, deterministic -- no LLM, no I/O. Compresses `content` to a
    keyframe by keeping its highest-INFORMATION segments, NOT its head. A
    positional "first sentence" keyframe keeps the setup and throws away the
    payload -- "Good morning, my friend! Now, about those important things..."
    would survive as just the greeting. Instead we score every segment by the
    concepts it actually carries and keep the densest ones, in original order,
    with elision marks where segments were skipped.

    Scoring (see _pith_segment_score): a segment earns points for the item's
    own recurring salient terms (centroid), for payload tokens (numbers,
    dotted/underscored identifiers, `code`, file refs), for structural role
    (headings / def-class signatures / labelled bullets), and -- when a `query`
    is supplied -- for overlap with the query. Greetings and filler carry none
    of these and fall to the bottom on their own; no greeting blacklist needed.

    Returns `(keyframe, delta)`:
    - `keyframe`: the compressed string, ending in a visible " ⋯[+N]" marker
      (N = chars dropped) so a reader can never mistake it for the whole item;
      interior "⋯" marks show where non-adjacent segments were joined. Empty
      string for empty/whitespace input; `content` unchanged (empty delta) when
      it already fits.
    - `delta`: the dropped segments (original order) -- exactly what the
      keyframe elided, kept as the compression's own record of what was left
      out (a caller may inspect or re-expand it).

    max_chars defaults to CC_PITH_KEYFRAME_CHARS (env, clamped [60, 1000]).
    Never raises.
    """
    if max_chars is None:
        max_chars = _CC_PITH_KEYFRAME_CHARS

    if not content or not content.strip():
        return ("", "")

    if len(content) <= max_chars:
        return (content, "")

    # 1. Segment: lines, and split long prose lines into sentences so a single
    #    dense paragraph can be sub-selected rather than kept/dropped whole.
    raw_segs: List[str] = []
    for ln in content.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if len(s) > 120 and re.search(r"[.?!]\s", s):
            for part in re.split(r"(?<=[.?!])\s+", s):
                if part.strip():
                    raw_segs.append(part.strip())
        else:
            raw_segs.append(s)
    if not raw_segs:
        return ("", "")

    segs = list(enumerate(raw_segs))  # (orig_index, text)

    # 2. Score each segment by intrinsic information payload (+ query overlap).
    term_freq = _pith_salient_terms(content)
    qterms = set(re.findall(r"[a-z0-9_]+", query.lower())) if query else set()

    def _score(idx_seg):
        idx, seg = idx_seg
        base = _pith_segment_score(seg, term_freq)
        if qterms:
            sw = re.findall(r"[a-z0-9_]+", seg.lower())
            base += float(sum(1 for w in sw if w in qterms))
        # Faint positional prior: only a tie-breaker so equally-informative
        # segments keep reading order; far too small to override real signal.
        return base - 0.001 * idx

    ranked = sorted(segs, key=_score, reverse=True)

    # 3. Greedily pack the most-informative segments until the budget is spent.
    #    Unlike Stage 3's item-level strict-prefix, packing WITHIN one item is
    #    correct -- a keyframe is a summary, so a shorter lower-ranked segment
    #    that still fits is worth keeping.
    marker_reserve = 12  # room for the trailing " ⋯[+NNNN]" marker
    budget = max(1, max_chars - marker_reserve)
    picked: List[tuple] = []  # (orig_index, text)
    used = 0
    for idx, seg in ranked:
        piece = seg
        if not picked and len(piece) > budget:
            piece = _pith_cut_at_word_boundary(piece, budget)
        add = len(piece) + (1 if picked else 0)
        if picked and used + add > budget:
            continue
        picked.append((idx, piece))
        used += add

    if not picked:
        return ("", content)

    # 4. Restore reading order; mark elisions between non-adjacent segments.
    picked.sort(key=lambda t: t[0])
    out_parts: List[str] = []
    prev_idx = None
    for idx, piece in picked:
        if prev_idx is not None and idx != prev_idx + 1:
            out_parts.append("⋯")
        out_parts.append(piece)
        prev_idx = idx
    body = " ".join(out_parts)

    picked_idxs = {i for i, _ in picked}
    delta = " ".join(seg for i, seg in segs if i not in picked_idxs)
    dropped_chars = max(0, len(content) - len(body))
    keyframe = body + (" ⋯[+%d]" % dropped_chars)

    return (keyframe, delta)


def cc_thermal(graph: Any, node_id: str) -> float:
    """Pith Stage 5 raw thermal (warmth) for a node, read from the substrate's
    OWN state: Ca_i (persistence of recent activation -- decays each step,
    bumps on spike) + firing_rate_ema (recent firing rate). Un-normalized here;
    pith_stage3 min-max normalizes it across the survivor set before folding it
    into the rank. The Lenia field-energy term (_CC_PITH_THERMAL_W_FIELD) is
    reserved but not yet wired -- these two SNN signals carry the warmth today.
    Read-only, fail-soft to 0.0 (a vanished/stateless node adds no warmth)."""
    try:
        node = graph.nodes.get(node_id) if graph is not None else None
        if node is None:
            return 0.0
        ca = float(getattr(node, "Ca_i", 0.0) or 0.0)
        fire = float(getattr(node, "firing_rate_ema", 0.0) or 0.0)
        return _CC_PITH_THERMAL_W_CA * ca + _CC_PITH_THERMAL_W_FIRE * fire
    except Exception:
        return 0.0


def cc_l1_budget(commons: Any, graph: Any = None, fired_node_ids: Any = None) -> int:
    """Pith §3.2 autonomic breathing: the L1 char budget breathes with arousal.
    PARASYMPATHETIC (calm/exploratory) -> expanded; SYMPATHETIC (threat/tunnel
    vision) -> contracted. Reads the single authoritative arousal Immunis
    deposits to the CC Commons (commons.read_arousal). Gated by
    CC_PITH_L1_BREATHE; off (or no Commons) -> the static budget. Fail-soft ->
    static budget on any error, and clamped to the same [500, 40000] bounds.
    
    Shared Graduation (COMB-04): when graph and the fired node ids are provided
    and CC_PITH_REGION_CONFIDENCE_ENABLED is True, the confidence of the region
    that fired modulates the budget alongside arousal. High confidence SHRINKS
    the budget, low confidence WIDENS it, by up to _CC_PITH_REGION_CONFIDENCE_FALLOFF
    from neutral (0.5). KISS_Pith_Combined_Architecture.md l.169: "Pith can
    aggressively compress extraction from that region"; l.170: "Pith loosens
    (promote more context to L1 — the model needs more to reason about
    unfamiliar territory)" (#592)."""
    # Start with base budget
    base_budget = _CC_PITH_L1_BUDGET
    
    # Apply autonomic breathing if enabled
    if _CC_PITH_L1_BREATHE and commons is not None:
        try:
            state = commons.read_arousal()
            mult = _CC_PITH_BREATHE_SYMPATHETIC if state == "SYMPATHETIC" else _CC_PITH_BREATHE_PARASYMPATHETIC
            base_budget = int(base_budget * mult)
        except Exception:
            pass  # Fail-soft to static budget
    
    # Apply region confidence modulation if enabled and parameters provided
    if _CC_PITH_REGION_CONFIDENCE_ENABLED and graph is not None and fired_node_ids:
        try:
            confidence = cc_region_confidence(graph, fired_node_ids)
            # confidence in [0, 1], neutral = 0.5
            # Scale: (confidence - 0.5) * 2 * falloff gives [-falloff, +falloff],
            # SUBTRACTED: confidence=1.0 -> 1-falloff (compress, l.169),
            # confidence=0.0 -> 1+falloff (loosen, l.170)
            confidence_factor = 1.0 - (confidence - _CC_PITH_REGION_CONFIDENCE_NEUTRAL) * 2.0 * _CC_PITH_REGION_CONFIDENCE_FALLOFF
            base_budget = int(base_budget * confidence_factor)
        except Exception:
            pass  # Fail-soft: keep budget without region confidence
    
    return max(500, min(40000, base_budget))


# Pith Stage 5 victim cache: cache lines surfaced but dropped from L1 (budget
# overflow) land here instead of vanishing, so a near-future turn can recover
# them ("wait, go back to what you said"). Bounded FIFO, TTL-aged by recall
# turn. Module-level + lock (daemon recall + idle sweep touch it).
_PITH_VICTIM: List[Dict[str, Any]] = []
_PITH_VICTIM_LOCK = threading.Lock()


def pith_victim_recover(candidates: List[CacheLine]) -> List[CacheLine]:
    """Stage 5 recapture: merge still-live victim entries back into the
    candidate set for a second chance at L1, and age the buffer one turn. A
    victim not already among the fresh candidates is re-injected as a CacheLine
    (stream='victim', carrying its cached thermal); entries past TTL are
    evicted. No-op when the buffer is disabled (size<=0) or empty."""
    if _CC_PITH_VICTIM_SIZE <= 0 or not _PITH_VICTIM:
        return candidates
    have = {cl.node_id for cl in candidates}
    merged = list(candidates)
    with _PITH_VICTIM_LOCK:
        live = []
        for v in _PITH_VICTIM:
            v["ttl"] -= 1
            if v["ttl"] <= 0:
                continue
            live.append(v)
            if v["node_id"] not in have:
                cl = CacheLine.from_surfaced(v["node_id"], v["content"],
                                             score=v["score"], stream="victim")
                cl.coherence = v.get("coherence", "unknown")
                cl.thermal = v.get("thermal", 0.0)
                merged.append(cl)
        _PITH_VICTIM[:] = live
    return merged


def pith_victim_capture(kept: List[CacheLine], all_lines: List[CacheLine]) -> None:
    """Stage 5 eviction: unpinned lines that were surfaced but didn't make L1
    (budget overflow) drop into the bounded victim buffer. Any victim promoted
    back into L1 this turn is removed (it's resident again). FIFO-bounded to
    CC_PITH_VICTIM_SIZE; TTL (re)set on capture. No-op when disabled."""
    if _CC_PITH_VICTIM_SIZE <= 0:
        return
    kept_ids = {cl.node_id for cl in kept}
    dropped = [cl for cl in all_lines
               if not cl.pinned and cl.node_id not in kept_ids and cl.stream != "victim"]
    with _PITH_VICTIM_LOCK:
        # drop any victim that got promoted back into L1 this turn
        _PITH_VICTIM[:] = [v for v in _PITH_VICTIM if v["node_id"] not in kept_ids]
        existing = {v["node_id"]: v for v in _PITH_VICTIM}
        for cl in dropped:
            if cl.node_id in existing:
                existing[cl.node_id]["ttl"] = _CC_PITH_VICTIM_TTL
            else:
                _PITH_VICTIM.append({"node_id": cl.node_id, "content": cl.content,
                                     "score": cl.score, "stream": cl.stream,
                                     "coherence": cl.coherence,
                                     "thermal": cl.thermal, "ttl": _CC_PITH_VICTIM_TTL})
        if len(_PITH_VICTIM) > _CC_PITH_VICTIM_SIZE:
            del _PITH_VICTIM[:len(_PITH_VICTIM) - _CC_PITH_VICTIM_SIZE]


def pith_compress_history(turn_texts: List[str], graph: Any, per_turn_chars: Optional[int] = None) -> List[str]:
    """Pith over the OUTBOUND conversation history — the substrate-informed
    replacement for miniTID's faux KISS. Given the ordered older-than-window
    turns (miniTID owns the message array + ordering; the substrate has no
    conversation identity -- turns are content-hashed nodes, sha1(text)), return
    a positionally-aligned list of compressed turns miniTID can splice back in.

    Substrate-INFORMED (the whole point of being connected to the NG): each
    turn's own node warmth (cc_thermal: Ca_i + firing_rate_ema) scales how many
    chars it keeps -- a turn the substrate has kept warm (recently reactivated,
    load-bearing) is compressed LESS; a cold one more. The actual compression is
    the existing salience-aware keyframe extractor (pith_stage2_keyframe) -- it
    keeps the payload and drops filler, NOT the greeting-keeping first-sentence
    cut faux KISS does. Fail-soft per turn: a turn with no node (e.g. reinforced
    away by the KISS gate) or any error keeps warmth 0 -> base budget.

    per_turn_chars: base keyframe budget (default CC_PITH_KEYFRAME_CHARS);
    warmth scales it up to ~2x for the warmest turns."""
    import hashlib
    base = per_turn_chars if per_turn_chars is not None else _CC_PITH_KEYFRAME_CHARS
    out: List[str] = []
    failures = 0
    chars_in = 0
    chars_out = 0
    turns_compressed = 0
    for text in turn_texts:
        before_len = None
        try:
            if not isinstance(text, str):
                raise TypeError("history turn must be text")
            before_len = len(text)
            if not text or not text.strip():
                chosen = text
            else:
                node_id = "cc:conv::" + hashlib.sha1(text.encode()).hexdigest()
                warmth = cc_thermal(graph, node_id)  # 0.0 if node absent (fail-soft)
                # normalize warmth into a [1.0, 2.0] budget multiplier: warmer = keep more.
                # thermal is unbounded-ish; squash with a soft cap so one hot turn can't
                # blow the budget. tanh-free cheap squash: w/(w+1) in [0,1).
                mult = 1.0 + (warmth / (warmth + 1.0)) if warmth > 0 else 1.0
                budget = max(60, min(1000, int(base * mult)))
                keyframe, _delta = pith_stage2_keyframe(text, max_chars=budget)
                chosen = keyframe if keyframe else text
        except Exception:
            failures += 1
            chosen = text  # never drop a turn; worst case pass it through
        out.append(chosen)
        if before_len is not None:
            chars_in += before_len
            try:
                after_len = len(chosen)
            except Exception:
                failures += 1
            else:
                chars_out += after_len
                turns_compressed += int(after_len < before_len)
    _PITH_METRICS.record_history_compression(
        turns_in=len(turn_texts),
        turns_compressed=turns_compressed,
        chars_in=chars_in,
        chars_out=chars_out,
        failures=failures,
    )
    return out


def pith_stage3(cache_lines: List[CacheLine], budget_chars: Optional[int] = None,
                 weights: Optional[Dict[str, float]] = None) -> List[CacheLine]:
    """Pith Stage 3: unified rank + char budget -- the L1 assembler core.

    Replaces block-order concatenation (every recency item before every
    relevance item, regardless of score) with a single ranked, budget-bounded
    read. Consumes the emitter scores already carried on each CacheLine (does
    NOT re-derive relevance -- no embed(), no GSG/Poincare rescore, no
    vector_db scan, no substrate walk); normalization is a monotone transform
    of the emitter's own score, nothing more.

    Order of operations:

    1. Split pinned vs unpinned. Pinned lines are ALWAYS kept, in their
       original relative order, and never consume budget.
    2. Per-stream min-max normalize the unpinned lines' `score` (grouped by
       `.stream`) -- the two streams' raw scales (~1.7 for SurfacingMonitor
       recency, ~100s for GSG-rescored pattern-completion relevance) are not
       comparable, so normalizing within-stream first lets both signals
       contribute instead of the larger-numbered stream always winning.
       Single-item or all-equal-score streams normalize to 1.0 (top-of-
       stream), never a div-by-zero.
    3. Unified score = weights[stream] * norm. weights defaults to
       {"pattern": CC_PITH_W_RELEVANCE, "monitor": CC_PITH_W_RECENCY,
       "recall": CC_PITH_W_RELEVANCE}; an unknown stream fails open to 1.0
       (kept in contention rather than zeroed out).
    4. Stable-sort unpinned lines by unified score, descending.
    5. Greedy budget fill over the sorted list, accumulating len(content):
       keep while the running total stays <= budget_chars, stop at the first
       line that would exceed it. A single line longer than the whole budget
       is still kept if nothing has been added yet (never emit an empty L1
       just because the top item is large), then fill stops.
    6. Assemble: pinned lines first (original order), then kept unpinned
       lines in ranked order.

    budget_chars defaults to CC_PITH_L1_BUDGET (env, clamped [500, 40000]);
    weights defaults to the CC_PITH_W_RELEVANCE / CC_PITH_W_RECENCY env pair.
    Updates the module-level _PITH_METRICS counters (ranked_in, ranked_kept,
    ranked_dropped, budget_chars_used) unconditionally, same convention as
    pith_stage1. Pure function over the already-surfaced small set (<~30
    items) -- cheap, hook-timeout-safe, no I/O.

    Never raises on empty input (returns []) or degenerate scores.
    """
    _PITH_METRICS.ranked_in += len(cache_lines)
    if not cache_lines:
        # [D5d] An assembly that surfaced nothing still happened, so it is counted --
        # dropping these would inflate every per-assembly rate. Its four result
        # counters advance by zero, and they advance in the SAME lock hold as the
        # assembly count, so this exit obeys the same all-or-nothing rule as the
        # main one below. Gate off -> this function is never called -> 0, which
        # reads as "invalid sample", never "0% prefetch". See l1_assemblies.
        with _PITH_METRICS._lock:
            _PITH_METRICS.l1_assemblies += 1
        return []

    if budget_chars is None:
        budget_chars = _CC_PITH_L1_BUDGET
    if weights is None:
        weights = {
            "pattern": _CC_PITH_W_RELEVANCE,
            "monitor": _CC_PITH_W_RECENCY,
            "recall": _CC_PITH_W_RELEVANCE,
            "victim": _CC_PITH_W_RECENCY,   # recovered drops: secondary prior, like recency
        }

    # Step 1: split pinned vs unpinned.
    pinned_lines = [cl for cl in cache_lines if cl.pinned]
    unpinned_lines = [cl for cl in cache_lines if not cl.pinned]

    # Step 2: per-stream min/max over unpinned lines only.
    stream_bounds: Dict[str, tuple] = {}
    for cl in unpinned_lines:
        lo, hi = stream_bounds.get(cl.stream, (cl.score, cl.score))
        stream_bounds[cl.stream] = (min(lo, cl.score), max(hi, cl.score))

    # Steps 2-3: normalize + weight -> unified score, index-paired with
    # unpinned_lines so the stable sort in step 4 can carry input order
    # through as an explicit tie-breaker (Python's sort is already stable,
    # but pairing with the original index makes that ties-keep-input-order
    # guarantee explicit rather than incidental).
    scored: List[tuple] = []
    for idx, cl in enumerate(unpinned_lines):
        lo, hi = stream_bounds.get(cl.stream, (cl.score, cl.score))
        norm = 1.0 if hi <= lo else (cl.score - lo) / (hi - lo)
        weight = weights.get(cl.stream, 1.0)
        # Stage 5 thermal fold: warm content (high Ca_i/firing) is gently
        # preferred. thermal defaults 0.0 -> multiplier 1.0 -> byte-identical
        # to pre-Stage-5 ranking until the daemon populates cl.thermal.
        unified = weight * norm * (1.0 + _CC_PITH_THERMAL_GAIN * cl.thermal)
        scored.append((unified, idx, cl))

    # Step 4: stable sort by unified score, descending. Ties keep input
    # order because idx (ascending) is the secondary sort key.
    scored.sort(key=lambda t: (-t[0], t[1]))

    # Step 5: greedy budget fill, STRICT rank-prefix -- keep the top-ranked
    # run that fits and stop at the first line that would overflow. We do NOT
    # keep scanning for smaller lower-ranked lines that happen to fit: that
    # would let rank-20 recency junk jump ahead of a dropped rank-8 relevance
    # block, inverting the very ordering Stage 3 exists to enforce. The first
    # line is always kept (even if it alone exceeds the budget) so a large top
    # item never yields an empty L1. Graceful degradation (Pith Stage 2): a
    # line that doesn't fit at full fidelity gets one more chance as a
    # keyframe (compressed head) before being dropped -- terser beats absent.
    kept_unpinned: List[CacheLine] = []
    running_total = 0
    for _unified, _idx, cl in scored:
        full_len = len(cl.content or "")
        if not kept_unpinned or running_total + full_len <= budget_chars:
            kept_unpinned.append(cl)
            running_total += full_len
            continue

        kf, _delta = pith_stage2_keyframe(cl.content)
        if running_total + len(kf) <= budget_chars and len(kf) < full_len:
            cl.content = kf
            kept_unpinned.append(cl)
            running_total += len(kf)
            _PITH_METRICS.compressed_count += 1
            _PITH_METRICS.chars_saved += (full_len - len(kf))
            continue

        break
    dropped = len(scored) - len(kept_unpinned)

    # [D5] Spec sec 13.3 terms, counted here because this is where the final L1
    # set exists: `pinned_lines + kept_unpinned` is exactly what this function
    # returns. Both terms come from the same set, after the same budget cut, so
    # neither mixes pipeline stages. Deduplicated by node_id within THIS
    # invocation (labelled as such wherever reported) -- counting-only: the
    # emitted list below is untouched, and nothing here feeds ranking.
    # Tally outside the lock (pure local work), then commit under it, so the lock
    # is held for four additions rather than for a walk of the whole L1 set.
    _l1_final = pinned_lines + kept_unpinned
    _seen_l1: set = set()
    _n_kept = _n_pf = _n_kept_prom = _n_pf_prom = 0
    for _cl in _l1_final:
        _nid = getattr(_cl, "node_id", None)
        if not _nid or _nid in _seen_l1:
            continue                      # first-wins on provenance, as documented
        _seen_l1.add(_nid)
        _is_pf = bool(getattr(_cl, "prefetch_origin", False))
        _n_kept += 1
        if _is_pf:
            _n_pf += 1
        # Narrow reading of PRD "L1 promotions": promotion-eligible streams only.
        # Same eligibility applied to BOTH terms, so the numerator stays a subset.
        if getattr(_cl, "stream", "recall") not in ("monitor", "victim"):
            _n_kept_prom += 1
            if _is_pf:
                _n_pf_prom += 1

    # [D5d] Commit the assembly count and its four result counters in ONE lock hold,
    # at the END of the assembly. Two reasons, and the second was a real defect:
    #
    # 1. `+=` on an attribute is load-add-store, so concurrent recalls would
    #    otherwise lose updates.
    # 2. Counting the assembly EARLY (as D5b did, above the empty-input return) let
    #    a reset land between the two commits: the assembly count was zeroed while
    #    this assembly's results were still in flight, and the next snapshot showed
    #    results against zero assemblies -- kept=1, prefetch=1, assemblies=0. A rate
    #    computed from that window divides by nothing. Reproduced by pausing inside
    #    `weights.get` and resetting mid-flight; regression test in
    #    tests/test_pith_metrics_concurrency.py.
    #
    # Committing at the end means an assembly interrupted by an exception records
    # NEITHER its count nor its results, which is the consistent outcome: a reset
    # now always lands cleanly between assemblies, never inside one.
    with _PITH_METRICS._lock:
        _PITH_METRICS.l1_kept_distinct += _n_kept
        _PITH_METRICS.l1_prefetch_distinct += _n_pf
        _PITH_METRICS.l1_kept_distinct_promotable += _n_kept_prom
        _PITH_METRICS.l1_prefetch_distinct_promotable += _n_pf_prom
        _PITH_METRICS.l1_assemblies += 1

    _PITH_METRICS.ranked_kept += len(pinned_lines) + len(kept_unpinned)
    _PITH_METRICS.ranked_dropped += dropped
    _PITH_METRICS.budget_chars_used += running_total

    # Step 6: assemble -- pinned first (original order), then ranked kept.
    return pinned_lines + kept_unpinned


# ---------------------------------------------------------------------------
# Provider-context Slice A: topology-connected situational assembly
# ---------------------------------------------------------------------------

_PITH_ANCHOR_PATTERNS = (
    re.compile(r"https?://[^\s)>\]]+"),
    re.compile(r"(?<![A-Za-z0-9_])/[A-Za-z0-9._~+()\-]+(?:/[A-Za-z0-9._~+()\-]+)+"),
    re.compile(r"\b(?:branch|worktree|repo(?:sitory)?)\s+([A-Za-z0-9._/\-]+)", re.I),
    re.compile(r"(?<![0-9a-f-])[0-9a-f]{7,40}(?![0-9a-f-])", re.I),
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),
    re.compile(r"(?<!\w)#\d+\b"),
)

def _pith_unique(values) -> list:
    out = []
    seen = set()
    for value in values:
        if value is None:
            continue
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _pith_exact_anchors(text: str, metadata: Optional[Dict[str, Any]] = None) -> list:
    """Extract exact operational references without treating them as knowledge.

    Anchors remain attached to their activation basin.  This is intentionally a
    conservative recognizer: paths, URLs, issue ids, UUIDs, commits, and explicit
    metadata fields only.  It does not emit arbitrary token-like strings.
    """
    found = []
    # Backticks and quotes are the only reliable boundary for paths containing
    # spaces; preserve the enclosed value exactly rather than guessing where a
    # prose path ends.
    for quoted in re.findall(r"[`\"']([^`\"'\n]*[/\\][^`\"'\n]+)[`\"']", text or ""):
        found.append(quoted)
    for pattern in _PITH_ANCHOR_PATTERNS:
        found.extend(pattern.findall(text or ""))
    meta = metadata if isinstance(metadata, dict) else {}
    for key in ("path", "file", "repo", "repository", "branch", "commit", "sha",
                "worktree", "session_id", "thread_id", "url"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            found.append(value.strip())
    return _pith_unique(found)


def _pith_node_raw_text(node: Any, fallback: str = "") -> str:
    """Resolve one node's unshortened meaning for assembly + anchor reads."""
    meta = getattr(node, "metadata", None) or {}
    choices = []
    if meta.get("_tree_concept"):
        choices.append(meta.get("_concept"))
    choices.extend((meta.get("_forest_content"), meta.get("want_text"),
                    meta.get("core_text"), meta.get("content"), meta.get("text"),
                    meta.get("_label"), meta.get("label"), fallback))
    for value in choices:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _pith_node_text(node: Any, fallback: str = "") -> str:
    """Resolve and bound one node's own meaning for a connected assembly.

    Tree nodes keep their own concept while a forest keeps the lived turn.  This
    differs deliberately from standalone snippet display: an assembly already
    carries the forest keyframe, so repeating that forest for every tree would
    erase the relationships the cache line exists to preserve.
    """
    text = _pith_node_raw_text(node, fallback)
    if len(text) <= _CC_PITH_PROVIDER_NODE_CHARS:
        return text
    keyframe, _delta = pith_stage2_keyframe(
        text, max_chars=_CC_PITH_PROVIDER_NODE_CHARS)
    return keyframe or _pith_cut_at_word_boundary(
        text, _CC_PITH_PROVIDER_NODE_CHARS)


def _pith_node_sources(node: Any) -> list:
    meta = getattr(node, "metadata", None) or {}
    values = []
    for key in ("source", "provenance", "creation_mode"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip()[:80])
    return _pith_unique(values) or ["substrate topology"]


def _pith_node_coherence(node: Any) -> str:
    meta = getattr(node, "metadata", None) or {}
    explicit = str(meta.get("coherence") or "").strip().lower()
    if meta.get("conflict") or meta.get("contested") or explicit == "conflict":
        return "conflict"
    if (meta.get("stale") or meta.get("invalid") or meta.get("superseded")
            or explicit in ("invalid", "stale")):
        return "stale"
    if meta.get("uncertain") or explicit == "uncertain":
        return "uncertain"
    if explicit in ("modified", "exclusive", "shared"):
        return explicit
    # Absence of a coherence record is unknown, not evidence that this process
    # exclusively owns a current/verified view.
    return "unknown"


def _pith_role(node: Any) -> str:
    """Return only an explicitly recorded experiential role.

    Free-text keyword inference would invent causality at extraction time.  The
    topology supplies the relationship; role labels refine it only when the
    node's own metadata names the role.
    """
    meta = getattr(node, "metadata", None) or {}
    for key in ("role", "kind", "event_type", "type"):
        value = str(meta.get(key) or "").lower()
        if value in ("action", "outcome", "correction", "failure", "decision"):
            return value
    return ""


def _pith_relation_label(parent_text: str, parent_node: Any,
                         child_text: str, child_node: Any, edge_kind: str) -> str:
    parent_role = _pith_role(parent_node)
    child_role = _pith_role(child_node)
    if parent_role == "action" and child_role in ("outcome", "failure"):
        return f"action -> {child_role}"
    if parent_role in ("outcome", "failure") and child_role == "correction":
        return f"{parent_role} -> correction"
    if parent_role == "correction" and child_role == "outcome":
        return "correction -> outcome"
    if edge_kind == "hyperedge":
        return "learned co-member"
    if edge_kind == "incoming":
        return "learned predecessor"
    return "learned successor"


def _pith_graph_neighbors(graph: Any, node_id: str,
                          active_node_ids: Optional[set] = None) -> list:
    """Return direct synaptic and hyperedge companions, strongest first.

    The graph's own learned topology defines membership.  Numeric strength is
    used only to make traversal deterministic and bounded; it never creates a
    relationship or substitutes cosine-selected snippets for a basin.
    """
    candidates: Dict[str, tuple] = {}

    def _offer(other_id, kind, strength):
        if not other_id or other_id == node_id or other_id not in graph.nodes:
            return
        prior = candidates.get(other_id)
        item = (kind, max(0.0, float(strength or 0.0)))
        if prior is None or item[1] > prior[1]:
            candidates[other_id] = item

    for sid in tuple(getattr(graph, "_outgoing", {}).get(node_id, ())):
        syn = graph.synapses.get(sid)
        if syn is not None:
            _offer(getattr(syn, "post_node_id", None), "outgoing", getattr(syn, "weight", 0.0))
    for sid in tuple(getattr(graph, "_incoming", {}).get(node_id, ())):
        syn = graph.synapses.get(sid)
        if syn is not None:
            _offer(getattr(syn, "pre_node_id", None), "incoming", getattr(syn, "weight", 0.0))
    for hid in tuple(getattr(graph, "_node_hyperedges", {}).get(node_id, ())):
        he = getattr(graph, "hyperedges", {}).get(hid)
        if he is None or getattr(he, "is_archived", False):
            continue
        member_weights = getattr(he, "member_weights", {}) or {}
        he_strength = max(float(getattr(he, "current_activation", 0.0) or 0.0),
                          float(getattr(he, "pattern_completion_strength", 0.0) or 0.0))
        for other_id in tuple(getattr(he, "member_nodes", ())):
            _offer(other_id, "hyperedge",
                   max(float(member_weights.get(other_id, 0.0) or 0.0), he_strength))
    active = active_node_ids or set()
    order = {"outgoing": 0, "incoming": 1, "hyperedge": 2}
    return [(nid, kind, strength) for nid, (kind, strength) in sorted(
        candidates.items(),
        key=lambda item: (-(item[0] in active), -item[1][1],
                          order.get(item[1][0], 9), item[0]))]


def _pith_is_constitutional(graph: Any, node_id: str) -> bool:
    """Identify only nodes already rendered by constitutional-core ownership.

    The graph's broader identity-protection predicate also covers deliberate
    authored wants.  Those are valid learned situation members and must not be
    suppressed merely because pruning protects them.
    """
    node = graph.nodes.get(node_id)
    meta = getattr(node, "metadata", None) or {}
    return bool(meta.get("constitutional"))


def _pith_copy_cache_line(line: CacheLine, stream: Optional[str] = None) -> CacheLine:
    return CacheLine(
        node_id=line.node_id, content=line.content, score=line.score,
        pinned=line.pinned, thermal=line.thermal, coherence=line.coherence,
        stream=stream or line.stream, prefetch_origin=line.prefetch_origin,
        member_node_ids=list(line.member_node_ids), relations=[dict(r) for r in line.relations],
        sources=list(line.sources), anchors=list(line.anchors), epistemic=line.epistemic,
    )


def pith_connected_activation_basins(graph: Any, surfaced: List[Dict[str, Any]],
                                      max_members: Optional[int] = None,
                                      max_depth: Optional[int] = None,
                                      live_rails: Optional[Dict[str, str]] = None) -> List[CacheLine]:
    """Build relationship-preserving cache lines from SNN-surfaced roots.

    Every returned CacheLine is a connected basin: one fired root plus direct
    synaptic/hyperedge companions and, when available, one further causal hop.
    VDB ranking chooses no members here.  The learned graph does.  Constitutional
    nodes are excluded because provider_context renders the constitutional core
    once, independently and non-evictably.
    """
    if graph is None or not surfaced:
        return []
    member_limit = max_members or _CC_PITH_PROVIDER_MEMBERS
    depth_limit = max_depth or _CC_PITH_PROVIDER_DEPTH
    raw_basins = []
    rail_labels = {
        _pith_normalize(text): label
        for text, label in (live_rails or {}).items()
        if isinstance(text, str) and text.strip()
    }

    def _display_text(node, fallback=""):
        raw = _pith_node_raw_text(node, fallback)
        rail_label = rail_labels.get(_pith_normalize(raw))
        if rail_label:
            return raw, f"[{rail_label} is present exactly once in the live tail]", True
        return raw, _pith_node_text(node, fallback), False

    active_node_ids = {item.get("node_id") for item in surfaced if item.get("node_id")}
    root_scores = [float(item.get("score", 0.0) or 0.0) for item in surfaced]
    score_lo = min(root_scores) if root_scores else 0.0
    score_hi = max(root_scores) if root_scores else 0.0

    for root_item in surfaced:
        root_id = root_item.get("node_id")
        root = graph.nodes.get(root_id) if root_id else None
        if root is None or _pith_is_constitutional(graph, root_id):
            continue
        root_raw, root_text, root_is_live = _display_text(
            root, root_item.get("content", ""))
        if not root_text:
            continue

        members = [root_id]
        relations = []
        sources = _pith_node_sources(root)
        anchors = ([] if root_is_live else
                   _pith_exact_anchors(root_raw, getattr(root, "metadata", None)))
        coherence_states = [_pith_node_coherence(root)]
        frontier = [(root_id, root_text, root, 0)]
        visited = {root_id}
        internal_support = 0.0
        total_support = 0.0

        while frontier and len(members) < member_limit:
            parent_id, parent_text, parent_node, depth = frontier.pop(0)
            neighbors = _pith_graph_neighbors(graph, parent_id, active_node_ids)
            total_support += sum(strength for _nid, _kind, strength in neighbors)
            if depth >= depth_limit:
                continue
            for child_id, edge_kind, strength in neighbors:
                if child_id in visited or _pith_is_constitutional(graph, child_id):
                    continue
                child = graph.nodes.get(child_id)
                child_raw, child_text, child_is_live = _display_text(child)
                if not child_text:
                    continue
                visited.add(child_id)
                members.append(child_id)
                internal_support += strength
                label = _pith_relation_label(
                    parent_text, parent_node, child_text, child, edge_kind)
                relation = {
                    "from": parent_id,
                    "to": child_id,
                    "kind": label,
                    "content": child_text,
                }
                relations.append(relation)
                sources.extend(_pith_node_sources(child))
                if not child_is_live:
                    anchors.extend(_pith_exact_anchors(
                        child_raw, getattr(child, "metadata", None)))
                coherence_states.append(_pith_node_coherence(child))
                frontier.append((child_id, child_text, child, depth + 1))
                if len(members) >= member_limit:
                    break

        if "conflict" in coherence_states:
            coherence = "conflict"
        elif "stale" in coherence_states:
            coherence = "stale"
        elif "uncertain" in coherence_states:
            coherence = "uncertain"
        elif "modified" in coherence_states:
            coherence = "modified"
        elif "shared" in coherence_states:
            coherence = "shared"
        elif coherence_states and all(value == "exclusive" for value in coherence_states):
            coherence = "exclusive"
        else:
            coherence = "unknown"

        activation = float(root_item.get("score", 0.0) or 0.0)
        activation_norm = 1.0 if score_hi <= score_lo else (activation - score_lo) / (score_hi - score_lo)
        cohesion = internal_support / total_support if total_support > 0 else 0.0
        relation_depth = min(1.0, len(relations) / 3.0)
        causal = 1.0 if any("action ->" in r["kind"] or "-> correction" in r["kind"]
                            for r in relations) else 0.0
        # Structure gets the deciding vote; root activation remains a strong
        # attention signal but cannot let one unrelated high-score hub tear a
        # coherent action/outcome/correction assembly apart.
        structural = 0.5 * cohesion + 0.3 * relation_depth + 0.2 * causal
        basin_score = 0.35 * activation_norm + 0.65 * structural
        raw_basins.append(CacheLine(
            node_id=root_id,
            content=root_text,
            score=basin_score,
            pinned=False,
            thermal=cc_thermal(graph, root_id),
            coherence=coherence,
            stream="connected",
            prefetch_origin=bool(root_item.get("prefetch_origin", False)),
            member_node_ids=members,
            relations=relations,
            sources=_pith_unique(sources),
            anchors=_pith_unique(anchors),
            epistemic="learned",
        ))

    # Competition includes the organism's existing warmth and coherence state
    # without letting either erase topology.  Learned structure + activation
    # remain 85% of the decision; warmth and coherence are bounded tie-breaks.
    thermal_values = [line.thermal for line in raw_basins]
    thermal_lo = min(thermal_values) if thermal_values else 0.0
    thermal_hi = max(thermal_values) if thermal_values else 0.0
    coherence_support = {
        "exclusive": 1.0, "shared": 0.9, "modified": 0.75,
        "uncertain": 0.55, "stale": 0.35, "conflict": 0.25,
        "unknown": 0.5,
    }
    for line in raw_basins:
        thermal_norm = (0.0 if thermal_hi <= thermal_lo else
                        (line.thermal - thermal_lo) / (thermal_hi - thermal_lo))
        line.score = (0.85 * line.score + 0.10 * thermal_norm
                      + 0.05 * coherence_support.get(line.coherence, 0.5))

    raw_basins.sort(key=lambda line: (-line.score, line.node_id))
    selected = []
    covered = set()
    for line in raw_basins:
        members = set(line.member_node_ids)
        if members and len(members & covered) / len(members) >= 0.6:
            continue
        selected.append(line)
        covered.update(members)
    return selected


def _pith_render_connected_line(line: CacheLine) -> str:
    """Model-facing Markdown for one whole cache line; never renders scores/ids."""
    label = "learned from substrate"
    lines = [f"### Connected assembly [{label}; coherence: {line.coherence}]",
             f"- Keyframe: {line.content}"]
    for relation in line.relations:
        lines.append(f"- {relation['kind']}: {relation['content']}")
    if line.sources:
        lines.append("- Sources: " + ", ".join(line.sources))
    if line.anchors:
        lines.append("- Exact anchors: " + ", ".join(f"`{a}`" for a in line.anchors))
    return "\n".join(lines)


def _pith_fit_statement(text: str, limit: int) -> Optional[str]:
    """Bound one statement while keeping it visibly extractive.

    Relationship membership is carried by the CacheLine, not inferred from a
    shortened sentence.  This helper therefore shortens only the prose payload;
    it never removes a relation, source, anchor, or coherence label.
    """
    value = (text or "").strip()
    if not value or limit < 1:
        return None
    if len(value) <= limit:
        return value
    if limit <= 2:
        return "…"[:limit]
    if limit < 18:
        return _pith_cut_at_word_boundary(value, limit - 2) + " …"
    keyframe, _delta = pith_stage2_keyframe(value, max_chars=limit)
    if keyframe and len(keyframe) <= limit:
        return keyframe
    shortened = _pith_cut_at_word_boundary(value, limit - 2)
    return (shortened + " …")[:limit]


def _pith_fit_connected_line(line: CacheLine, max_chars: int) -> Optional[CacheLine]:
    """Fit one relationship CacheLine by shortening prose, never structure.

    All member relationships, exact anchors, sources, and epistemic/coherence
    labels travel together.  If even that fixed structure cannot fit, the
    entire line is rejected rather than emitting an orphaned fragment.
    """
    if max_chars <= 0:
        return None
    if len(_pith_render_connected_line(line)) <= max_chars:
        return _pith_copy_cache_line(line)

    fitted = _pith_copy_cache_line(line)
    fields = [line.content] + [str(r.get("content") or "") for r in line.relations]
    # Render with one visible character per statement to measure structure that
    # cannot be removed (relation labels, sources, anchors, headings).
    fitted.content = "…"
    for relation in fitted.relations:
        relation["content"] = "…"
    fixed_cost = len(_pith_render_connected_line(fitted))
    if fixed_cost > max_chars:
        return None

    payload_budget = max_chars - fixed_cost + len(fields)  # replace each "…"
    allocations = [1] * len(fields)
    remaining = payload_budget - len(fields)
    active = {i for i, value in enumerate(fields) if len(value) > 1}
    # Deterministic water-filling preserves every statement while allowing short
    # statements to finish and donate their unused share to longer ones.
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for index in tuple(sorted(active)):
            want = len(fields[index]) - allocations[index]
            add = min(want, share, remaining)
            if add > 0:
                allocations[index] += add
                remaining -= add
                progressed = True
            if allocations[index] >= len(fields[index]):
                active.discard(index)
            if remaining <= 0:
                break
        if not progressed:
            break

    rendered_fields = [_pith_fit_statement(value, allowance)
                       for value, allowance in zip(fields, allocations)]
    if any(value is None for value in rendered_fields):
        return None
    fitted.content = rendered_fields[0]
    for relation, value in zip(fitted.relations, rendered_fields[1:]):
        relation["content"] = value
    # Arithmetic above is exact for the renderer, but retain a closed guard if
    # future formatting changes add overhead.
    if len(_pith_render_connected_line(fitted)) > max_chars:
        return None
    return fitted


def _pith_provider_admit(lines: List[CacheLine], budget_chars: int) -> tuple:
    """Admit a strict ranked prefix as whole relationship cache lines.

    An oversized line may compress its member prose to the remaining envelope,
    but no relation, source, anchor, or coherence label is removed.  A line
    whose fixed structure cannot fit stops prefix admission.
    """
    ordered = sorted(lines, key=lambda line: (-line.score, line.node_id))
    kept = []
    rendered = []
    used = 0
    for line in ordered:
        separator = 2 if rendered else 0
        fitted = _pith_fit_connected_line(line, budget_chars - used - separator)
        if fitted is None:
            break
        block = _pith_render_connected_line(fitted)
        cost = len(block) + separator
        kept.append(fitted)
        rendered.append(block)
        used += cost
    return kept, rendered


def _pith_line_is_correction(line: CacheLine) -> bool:
    return any("correction" in relation["kind"] or "failure" in relation["kind"]
               for relation in line.relations)


def _pith_provider_sections(core: str, lines: List[CacheLine],
                            blocks: List[str]) -> tuple:
    """Render complete provider context and its learned-state warnings."""
    situation = []
    corrections = []
    warnings = []
    alert_states = []
    for line, block in zip(lines, blocks):
        (corrections if _pith_line_is_correction(line) else situation).append(block)
        if line.coherence in ("conflict", "stale", "uncertain", "unknown"):
            alert_states.append(line.coherence)
            warnings.append(f"{line.coherence}_material")

    sections = [core]
    if situation:
        sections.append("## Learned Situation\n" + "\n\n".join(situation))
    if corrections:
        sections.append("## Learned Corrections and Failures\n" + "\n\n".join(corrections))
    if alert_states:
        alerts = []
        for state in _pith_unique(alert_states):
            if state == "unknown":
                alerts.append(
                    "- A connected learned assembly has no explicit coherence record; "
                    "treat currentness as unknown until live evidence confirms it.")
            else:
                alerts.append(
                    f"- A connected learned assembly is marked {state}; "
                    "treat it as unresolved until live evidence confirms it.")
        sections.append("## Uncertainty and Conflicts\n" + "\n".join(alerts))
    return "\n\n".join(sections), _pith_unique(warnings)


def _pith_provider_unavailable(reason: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "state": "unavailable",
        "context": f"## NeuroGraph Context Status\n- Fresh substrate context unavailable ({reason}).",
        "source": "cc_neurograph_topology",
        "coherence": "unavailable",
        "anchors": [],
        "warnings": [reason],
        "assemblies": 0,
    }


def pith_provider_context(ng: Any, current_instruction: str, quest_focus: str = "",
                          conv_state: Optional[Dict[str, Any]] = None,
                          commons: Any = None,
                          budget_chars: Optional[int] = None,
                          root_count: Optional[int] = None) -> Dict[str, Any]:
    """Construct a fresh provider-ready situational model from CC's live SNN.

    `current_instruction` and the already-rendered `quest_focus` orient attention
    but are not echoed: miniTID owns their one exact occurrence in the live
    message tail.  They are never deposited, classified, or fetched from Quest
    storage here.  Learned material comes only from the current
    topology/activation path: pattern completion provides roots and the graph's
    synapses/hyperedges provide connected assemblies.

    Closed result states are ``ok``, ``empty``, and ``unavailable``.  There is no
    heuristic, faux, transcript-replay, or raw-history fallback.
    """
    if not isinstance(current_instruction, str) or not current_instruction.strip():
        return _pith_provider_unavailable("invalid_instruction")
    if len(current_instruction) > _CC_PITH_PROVIDER_MAX_INSTRUCTION_CHARS:
        return _pith_provider_unavailable("instruction_too_large")
    if quest_focus is None:
        quest_focus = ""
    if not isinstance(quest_focus, str) or len(quest_focus) > _CC_PITH_PROVIDER_MAX_QUEST_CHARS:
        return _pith_provider_unavailable("invalid_quest_focus")
    graph = getattr(ng, "graph", None) if ng is not None else None
    if graph is None:
        return _pith_provider_unavailable("ng_unavailable")
    if budget_chars is not None and not (
            isinstance(budget_chars, int) and not isinstance(budget_chars, bool)
            and 500 <= budget_chars <= 40000):
        return _pith_provider_unavailable("invalid_budget")
    roots = root_count if root_count is not None else _CC_PITH_PROVIDER_ROOTS
    if (not isinstance(roots, int) or isinstance(roots, bool)
            or not 1 <= roots <= 24):
        return _pith_provider_unavailable("invalid_root_count")

    cue = current_instruction.strip()
    if quest_focus.strip():
        cue += "\n\n" + quest_focus.strip()
    try:
        core = render_constitutional_core(graph)
        if not core:
            # Constitutional identity is a non-evictable prerequisite, not a
            # best-effort memory.  Refuse to present a partial mind as healthy.
            return _pith_provider_unavailable("constitutional_core_missing")
        # cc_novelty updates its caller-owned bookkeeping.  A shallow copy keeps
        # provider_context observational even at that non-graph boundary.
        recall_state = dict(conv_state or {})
        # Provider context must describe what this cue actually ignited now.
        # Stage-4 speculative priming remains useful elsewhere, but an un-fired
        # prediction cannot become a situational root merely because it was in
        # the conversation state's prefetch set.
        recall_state["primed_nodes"] = {}
        surfaced = cc_pattern_completion_recall(
            ng, cue, roots, state=recall_state, preserve_graph_config=True)
        # The budget breathes with arousal and the confidence of the region
        # that just fired for this cue (Shared Graduation, Packet 175a).
        budget = budget_chars if budget_chars is not None else cc_l1_budget(
            commons, graph,
            [item.get("node_id") for item in surfaced if item.get("node_id")])
        if len(core) > budget:
            # Identity is indivisible and non-evictable.  Never abbreviate it
            # merely to make a context envelope look healthy.
            return _pith_provider_unavailable("constitutional_core_exceeds_budget")
        live_rails = {current_instruction.strip(): "current instruction"}
        if quest_focus.strip():
            quest_text = quest_focus.strip()
            prior = live_rails.get(quest_text)
            live_rails[quest_text] = (
                "current instruction and Quest focus" if prior else "Quest focus")
        fresh = pith_connected_activation_basins(
            graph, surfaced, live_rails=live_rails)
        candidates = fresh
        # Reserve every possible section/alert delimiter before admitting prose.
        # Empty placeholder blocks let the real renderer calculate that fixed
        # overhead without a second, drifting budget formula.
        envelope_shell, _shell_warnings = _pith_provider_sections(
            core, candidates, [""] * len(candidates))
        learned_budget = max(0, budget - len(envelope_shell))
        kept, learned_blocks = _pith_provider_admit(candidates, learned_budget)
        context, warnings = _pith_provider_sections(core, kept, learned_blocks)
        # The shell is conservative, but this is the explicit total-envelope
        # invariant.  A formatter regression fails closed rather than silently
        # overfilling a provider prompt.
        if len(context) > budget:
            return _pith_provider_unavailable("context_bound_failed")
        state = "ok" if kept else "empty"
        if not kept:
            warnings.append("topology_empty" if not candidates else "capacity_empty")
        coherence_order = (
            "conflict", "stale", "uncertain", "unknown", "modified", "shared", "exclusive")
        coherence = next((value for value in coherence_order
                          if any(line.coherence == value for line in kept)), "empty")
        anchors = _pith_unique(anchor for line in kept for anchor in line.anchors)
        return {
            "ok": True,
            "state": state,
            "context": context,
            "source": "cc_neurograph_topology",
            "coherence": coherence,
            "anchors": anchors,
            "warnings": _pith_unique(warnings),
            "assemblies": len(kept),
        }
    except Exception as exc:
        logger.warning("provider_context assembly unavailable: %s", exc)
        return _pith_provider_unavailable("assembly_failed")


# =============================================================================
# CC Recall Unification (LAW-3/"keep even") -- one recall pipeline, both
# hemispheres. See docs/superpowers/plans/2026-07-22-cc-recall-unification-
# spec.md. Before this, cc-ng-daemon.py:_recall (laptop) and cc_ng_host.py:
# _recall (VPS host) were copy-pasted and had drifted: laptop ran the full
# Pith pipeline, VPS ran zero Pith (a plain two-block concat), so enabling
# CC_PITH_ENABLED on the VPS would have been a no-op (no code there to gate).
# cc_assemble_recall is a verbatim extraction of the laptop _recall body (the
# reference -- LAW 3: extract, don't redesign), parameterized on (ng, query,
# k, conv_state, commons) instead of a module-global STATE, so it is process-
# agnostic by construction (Syl's-Law: bind to passed-in instances, never
# module globals -- cc_ng_organism.py already works this way elsewhere).
# Both _recall entry points (cc-ng-daemon.py laptop, cc_ng_host.py VPS) become
# thin per-half wrappers: STATE bookkeeping (last_activity/stats), then call
# this function and return its result. VPS behavior is byte-identical to
# today until CC_PITH_ENABLED is turned on (the gate defaults OFF), at which
# point it gains the same Pith pipeline the laptop already had.
# =============================================================================

# Rate-limits the Pith-fallback warning below (was per-half module state in
# cc-ng-daemon.py; now shared here since the fail-soft path lives in one
# place). Each process importing this module gets its own copy of these
# module-level names, so laptop and VPS still rate-limit independently.
_PITH_WARN_INTERVAL_S = float(os.environ.get("CC_PITH_WARN_INTERVAL_S", "60"))
_last_pith_warn_ts = 0.0

# Read-only recall instrumentation (CC_RECALL_DEBUG), folded in from
# cc-ng-daemon.py's laptop-only copy during the unification -- both
# hemispheres get it now (still off by default, still pure observation:
# does NOT change what cc_assemble_recall returns). When the env flag is
# set, appends one JSON line per call to _RECALL_DEBUG_PATH capturing the
# two raw streams separately -- SurfacingMonitor recency (monitor_items) vs
# pattern-completion substrate-spread (pc_results) -- each with score +
# node_id + content preview, BEFORE Pith merges them. Fail-soft. LAW 5.
_CC_RECALL_DEBUG = os.environ.get("CC_RECALL_DEBUG", "0") not in ("0", "false", "")
_RECALL_DEBUG_PATH = os.path.join(
    os.path.expanduser("~/.claude/plugins/neurograph"), "recall_debug.jsonl")


def _cc_recall_debug_log(query: str, monitor_items: List[Dict[str, Any]],
                          pc_results: List[Dict[str, Any]]) -> None:
    """Append one JSON line with both raw streams (read-only, fail-soft)."""
    if not _CC_RECALL_DEBUG:
        return
    try:
        def _stream(items):
            out = []
            for it in (items or [])[:15]:
                out.append({
                    "score": round(float(it.get("score", it.get("strength", 0.0)) or 0.0), 3),
                    "node_id": (it.get("node_id") or "")[:48],
                    "preview": (it.get("content", "") or "").replace("\n", " ")[:70],
                })
            return out
        rec = {
            "ts": time.time(),
            "query": (query or "")[:120],
            "n_monitor": len(monitor_items or []),
            "n_pattern": len(pc_results or []),
            "monitor": _stream(monitor_items),
            "pattern": _stream(pc_results),
        }
        with open(_RECALL_DEBUG_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as exc:
        logger.debug("recall-debug log failed (non-fatal): %s", exc)


# Bound on the exception message carried in the recall notice -- it rides the
# CC prompt every turn Pith keeps failing, so it must stay one short line.
_PITH_NOTICE_MSG_MAX = 200


def cc_pith_failure_text(exc: BaseException) -> str:
    """Raw text of a Pith recall failure: the exception type and message,
    as-is. No category, severity or tag (LAW 7) -- this is what the
    hemisphere deposit paths hand to the substrate."""
    return f"NeuroGraph recall Pith pass failed: {type(exc).__name__}: {exc}"


def cc_pith_unavailable_notice(stage: str, exc: BaseException) -> str:
    """The explicit notice cc_assemble_recall returns in place of recall when
    its Pith pass raises (Pith PRD failure envelope): what failed and why.
    Never blank -- the prefix alone is non-empty even for an empty message."""
    msg = " ".join(str(exc).split())[:_PITH_NOTICE_MSG_MAX]
    return f"[NeuroGraph recall unavailable: Pith {stage} failed: {type(exc).__name__}: {msg}]"


def cc_deposit_pith_failure(exc: BaseException, tract_path: Optional[str] = None) -> None:
    """Deposit a Pith recall failure raw onto the CC ingest tract -- the same
    tract, entry type and source ("cc_gateway") miniTID uses for conversational
    turns, so drain_ingest_tract absorbs it exactly as it absorbs a turn. The
    laptop hemisphere's on_pith_failure. Raises on write failure; the caller
    (cc_assemble_recall) logs it."""
    import ng_tract
    ng_tract.deposit_experience(cc_pith_failure_text(exc).encode("utf-8"),
                                "cc_gateway", [tract_path or cc_gateway_tract_path()])


def cc_assemble_recall(ng: Any, query: str, k: int, conv_state: dict, commons: Any,
                        allow_pattern_completion: bool = True,
                        on_monitor_error: Optional[Any] = None,
                        on_pith_failure: Optional[Any] = None) -> str:
    """Return surfacing context for CC hook injection -- THE shared recall
    pipeline for both hemispheres (laptop cc-ng-daemon.py, VPS cc_ng_host.py).

    Combines two complementary signals: SurfacingMonitor (recency -- nodes
    that fired in the SNN during recent deposits) and Active Recall (pattern
    completion -- direct semantic search via cc_pattern_completion_recall,
    finds genuinely relevant content regardless of how long ago it was
    learned). SurfacingMonitor block first, Active Recall block second --
    matches Syl's own ordering in handle_assemble() (neurograph_rpc.py).

    Dedups by node_id across the two blocks: a node can be both recently-
    fired (SurfacingMonitor) and semantically matching the query (Active
    Recall) -- without this it renders twice in the injected context.

    allow_pattern_completion=False skips the Active Recall half entirely --
    used by handle_pre_tool_use() when gate_pattern_completion() has already
    given this file_path a pattern-completion pass recently (per-file dedup
    cache; avoids re-paying the .recall() cost on every single PreToolUse
    touch to the same file within one task).

    When CC_PITH_ENABLED, the combined item set is run through the Pith
    pipeline (CacheLines -> pith_victim_recover -> cc_thermal -> cc_novelty
    -> pith_stage1 -> pith_stage3(budget=cc_l1_budget(commons)) ->
    pith_victim_capture) instead of the plain two-block concatenation, with
    constitutional pins (ng.graph._is_identity_protected) preserved
    unconditionally. An exception anywhere in the Pith path never falls back
    to the un-Pithed monitor_ctx/pc_block rendering and never returns blank:
    it returns ONLY cc_pith_unavailable_notice() (Pith PRD failure envelope),
    records a _PITH_METRICS failure, rate-limit-warns, and hands the raw
    exception to on_pith_failure (each hemisphere wires its own raw deposit
    there -- this query function does no write-side work itself, LAW 4). It
    never raises: a surfacing pass must not crash or time out the hook.

    Params only (ng/conv_state/commons) -- no module-global STATE access,
    so this function is process-agnostic (Syl's-Law) and safe to call from
    either hemisphere with its own isolated instances.
    """
    monitor_ctx = ''
    monitor_node_ids: set = set()
    monitor_items: List[Dict[str, Any]] = []
    try:
        monitor = getattr(ng, '_surfacing_monitor', None)
        if monitor is not None:
            monitor_items = monitor.get_surfaced()
            monitor_node_ids = {item.get('node_id') for item in monitor_items}
            monitor_ctx = monitor.format_context(monitor_items)
    except RuntimeError:
        monitor_ctx = ''  # dict mutation race during concurrent deposit
        monitor_node_ids = set()
        monitor_items = []
    except Exception as exc:
        logger.debug('Recall failed: %s', exc)
        if on_monitor_error is not None:
            try:
                on_monitor_error(exc)
            except Exception:
                pass  # the error-reporting hook itself must never break recall
        monitor_ctx = ''
        monitor_node_ids = set()
        monitor_items = []

    pc_block = ''
    pc_results: List[Dict[str, Any]] = []
    # Everything pattern completion fired, before the display dedup against
    # the monitor below: the L1 budget's region is what fired, not what is new.
    pc_fired_ids: List[str] = []
    if allow_pattern_completion:
        try:
            pc_results = cc_pattern_completion_recall(ng, query, k, state=conv_state)
            pc_fired_ids = [r.get('node_id') for r in pc_results if r.get('node_id')]
            pc_results = [r for r in pc_results if r.get('node_id') not in monitor_node_ids]
            pc_block = _format_cc_recall_block(pc_results)
        except Exception as exc:
            logger.debug('Pattern-completion recall failed (non-fatal): %s', exc)
            pc_block = ''
            pc_results = []
            pc_fired_ids = []

    # Read-only instrumentation (CC_RECALL_DEBUG): capture both raw streams
    # BEFORE Pith merges them, to measure where the query signal is lost.
    _cc_recall_debug_log(query, monitor_items, pc_results)

    # Pith pipeline (gated, CC_PITH_ENABLED, default OFF): dedup/clutter-strip
    # + unified-rank + budget the combined SurfacingMonitor + Active Recall
    # item set before rendering, replacing the two-block concatenation below
    # with one combined block. Gate OFF -> this whole block is skipped and
    # cc_assemble_recall() falls straight through to the original
    # monitor_ctx/pc_block return, unchanged.
    if _CC_PITH_ENABLED:
        # Which Pith step is running -- named in the notice if one raises.
        _stage = 'CacheLine build'
        try:
            def _pinned(node_id):
                try:
                    return bool(ng.graph._is_identity_protected(node_id))
                except Exception as exc:
                    # Fail-soft to not-pinned so ONE node's bad pin lookup can't
                    # sink the whole Pith pass; log it so a vanished/erroring
                    # constitutional-pin guard surfaces rather than going silent.
                    logger.debug('Pith pin lookup failed for %r (treating as unpinned): %s', node_id, exc)
                    return False

            # Stream-tag each half so pith_stage3 can per-stream normalize
            # (SurfacingMonitor recency ~1.7 vs Active Recall/GSG relevance
            # ~100s are not comparable scales) -- monitor_items -> "monitor",
            # pc_results -> "pattern".
            cache_lines = [
                CacheLine.from_surfaced(
                    node_id=it.get('node_id') or '',
                    content=it.get('content', ''),
                    score=it.get('score', 0.0),
                    pinned=_pinned(it.get('node_id')),
                    stream='monitor',
                )
                for it in monitor_items
            ] + [
                CacheLine.from_surfaced(
                    node_id=it.get('node_id') or '',
                    content=it.get('content', ''),
                    score=it.get('score', 0.0),
                    pinned=_pinned(it.get('node_id')),
                    stream='pattern',
                    # [D5] provenance only -- see CacheLine.prefetch_origin
                    prefetch_origin=bool(it.get('prefetch_origin', False)),
                )
                for it in pc_results
            ]

            # Pith Stage 5 (victim recapture): merge still-live victim-cache
            # entries back in for a second chance at L1, and age the buffer.
            _stage = 'victim_recover'
            cache_lines = pith_victim_recover(cache_lines)

            # Pith Stage 5 (thermal): populate each line's warmth from the
            # substrate's own state (Ca_i + firing_rate_ema). Done here because
            # this is where the graph is in scope; pith_stage3 folds it into the
            # rank (warm content preferred). Fail-soft per line -> 0.0.
            for _cl in cache_lines:
                try:
                    _cl.thermal = cc_thermal(ng.graph, _cl.node_id)
                except Exception:
                    _cl.thermal = 0.0

            try:
                novelty = cc_novelty(conv_state, ng.graph)
            except Exception as exc:
                logger.debug('Pith novelty lookup failed (non-fatal): %s', exc)
                novelty = 0.0

            _stage = 'stage1'
            survivors = pith_stage1(cache_lines, query, novelty)
            # Stage 3: unified rank + char budget -- replaces block-order
            # concatenation with a single ranked, budget-bounded L1 read.
            # Reads its own weights from env (CC_PITH_W_RELEVANCE,
            # CC_PITH_W_RECENCY); budget breathes with commons arousal.
            _pre_l1 = survivors  # post-stage1, pre-budget: the full L1 candidate set
            
            # Budget breathes with arousal and the confidence of the region
            # that fired for this query (pattern completion's fired set).
            _stage = 'L1 budget'
            budget = cc_l1_budget(commons, ng.graph, pc_fired_ids)
            
            _stage = 'stage3'
            survivors = pith_stage3(survivors, budget_chars=budget)
            # Pith Stage 5 (eviction): budget-dropped lines fall to the victim buffer.
            try:
                pith_victim_capture(survivors, _pre_l1)
            except Exception as exc:
                logger.debug('Pith victim capture failed (non-fatal): %s', exc)
            _stage = 'render'
            survivor_results = [{'score': cl.score, 'content': cl.content} for cl in survivors]
            return _format_cc_recall_block(survivor_results)
        except Exception as exc:
            # Failure envelope (Pith PRD): return ONLY the explicit notice --
            # never the un-Pithed monitor_ctx/pc_block below, never blank.
            # COUNT it and warn (rate-limited) so a failing Pith path is
            # observable, then hand the raw exception to the caller's deposit.
            _PITH_METRICS.record_failure()
            global _last_pith_warn_ts
            _now = time.time()
            if _now - _last_pith_warn_ts >= _PITH_WARN_INTERVAL_S:
                _last_pith_warn_ts = _now
                logger.warning('Pith %s failed, returning the unavailable notice: %s', _stage, exc)
            else:
                logger.debug('Pith %s failed (non-fatal), returning the unavailable notice: %s', _stage, exc)
            if on_pith_failure is not None:
                try:
                    on_pith_failure(exc)
                except Exception as cb_exc:
                    # The deposit hook must never break recall -- but a lost
                    # failure deposit is logged, not swallowed.
                    logger.warning('Pith failure deposit failed: %s', cb_exc)
            return cc_pith_unavailable_notice(_stage, exc)

    if monitor_ctx and pc_block:
        return monitor_ctx + "\n\n" + pc_block
    return monitor_ctx or pc_block
