#!/usr/bin/env python3
# SEE FIRST: /home/josh/docs/CC-CALLOSUM-TRUTH.md -- consolidated, verified state of
# the callosum, wholeness ring, hyperedge binding and orphan collection (2026-07-31).
# The wholeness ring ALREADY EXISTS here (Leg 2). Open defect: merge-journal poison-pill.
# ---- Changelog ----
# [2026-10-04] Claude (lane 810-onto-s4) — rebased #810 WANT legitimacy onto trial s4; surface_wants keeps the trial's
#   per-want guarded create_node (#915) and loud synapse reporting (#904/P489) and takes parse_wants()/want.text from #810,
#   with _log_want_skips after the lock; no change to parse_wants or any helper (byte-identical to d8f99c3).
# [2026-10-03] Claude Opus 5.5 (Executive, MVP) — cc_assemble_recall gains on_surfaced(rendered, dropped): reports what one pass surfaced (and what the Pith budget cut) so a host can log it; unset = unchanged.
# [2026-10-03] Claude Opus 5.5 (Executive, MVP, overnight under Josh's tweak authority) — drain_ingest_tract gains defer_tract_path/defer_over_bytes: oversize turns move whole to a deferred tract.
# [2026-10-03] Claude Opus 5.5 (Executive, MVP, Josh) — drain_ingest_tract gains retry_tract_path: a failed turn moves whole to a retry tract; the pass continues.
# [2026-10-03] Claude Opus 5.5 (Executive, MVP) — drain_ingest_tract gains max_seconds (keyword): ends a pass by wall time; the rest stays in the tract.
# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane ng-trial-chain-s4) -- TRIAL-branch hand-resolution of cherry-pick ee94f7d (#905)
#   onto D24+#794..#897: _cc_callosum_consolidate's except block keeps D24's loud logger.error (it replaced the logger.debug line) AND
#   #905's _fill(done, failed=True) before `return False`; the guard/progress signature and loop merged without conflict; no new behaviour.
# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane ng-trial-chain-s4) -- TRIAL-branch hand-resolution of cherry-pick c625623 (#794)
#   onto D24: drain_ingest_tract keeps D24's batch_nodes/receipt AND #794's hold_on_failure (LAST); the loop keeps both rules
#   (the hold break, then D24's size check, then the entries cap); both changelog entries below kept; no new behaviour.
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane drain-pacing-d24, dispatches #12669/#12731/#12929) — D24 (re-scoped):
#   pace the ingest-tract drain by NODES; make _cc_callosum_consolidate's failure loud
#   [D24 FOLD, #12929, Exec Packet 482 / #896: NO code change in this file's behaviour -- the KNOWN LIMIT text below is
#   REPLACED (the guard now covers the WHOLE unbound population, see the daemon's FOLD entry) and the receipt docstring's
#   scope sentence is corrected. The real-function seam test is tests/test_cc_drain_pacing_seam.py.]
# What: (1) drain_ingest_tract gains two OPTIONAL default-None parameters, `batch_nodes` and `receipt`. With
#   `batch_nodes` set (> 0) the drain takes WHOLE turns until the NODES created in this call reach it (the turn that
#   crosses it is absorbed whole -- a turn's dual pass is ATOMIC -- then the batch ends; a single turn alone over the
#   size is ONE batch); the remainder stays in the tract file through the EXISTING partial-truncate (the same
#   mechanism max_entries uses; no second one). `receipt` is an OUT-PARAMETER dict the caller reads afterwards:
#   {nodes_created, ended_on_size, arrivals (set of node ids: graph.nodes ids after minus before, taken once per
#   call), turns_taken, reason}. `reason` is a hardcoded constant: size_reached | entries_cap_reached |
#   tract_exhausted | parse_failed | no_batch. Receipt writes are guarded (756b-reporter style: never raises, logs
#   the exception CLASS NAME only, never str(exc), never entry text). Nodes are counted as the len(graph.nodes)
#   delta around the atomic call, so the dual pass is untouched; an exact-repeat turn (content-hashed ids) creates 0.
#   (2) _cc_callosum_consolidate's failure was logged at DEBUG (return False, nothing else): the silent class
#   (plan-002 7A item 5c). It is now logged at ERROR with the exception CLASS NAME only. Return value, steps, slicing
#   and the success path are byte-identical.
# Why: Exec P471 (Josh): the first-drain cap is 25 NODES (FatherGraph 25/250: batches of 25 nodes, 250 idle steps
#   between), not 25 turns. Exec P472: the idle steps are the DRAIN PATH's own, through the Leg 2 merge's step-and-guard
#   code (reuse, LAW 3), no dependency on #117/CC_NG_AUTOSTEP. Exec P473/P476: the lock is never held across the idle
#   steps -- TWO-PHASE: this function stays CALLER-LOCKED and DEPOSIT-ONLY (its lock contract is unchanged, it
#   acquires and releases nothing, it never steps, it never consolidates); the DAEMON, after its autosave section's
#   `finally` releases graph._concurrent_lock, runs cc_topology_merge._unbound_nodes -- over the WHOLE graph since the D24
#   FOLD (#896); it was the receipt's arrivals under P476(d) -- and then _cc_callosum_consolidate (which slices the lock itself). P476(e): the DEBUG swallow is fixed where the drain's
#   phase 2 reuses the function (P370).
# How: with both new parameters unset the function is byte-identical to e4ebf982 (proved by a test that loads the
#   base module from `git show e4ebf982:cc_ng_organism.py`). Syl's process and the VPS host call site
#   (cc_ng_host.py:1525-1526) pass neither, so they are unchanged.
#   Two records exist for a failed consolidation, deliberately: _cc_callosum_consolidate logs the CAUSE (exception
#   class + steps run); the daemon's phase 2 logs the CONSEQUENCE (pacing re-armed, next batch held).
# KNOWN LIMIT, replacing P476(d)'s text. F1 (le-044) is RESOLVED by the D24 FOLD (Exec Packet 482 / #896): the guard the
#   daemon applies covers the WHOLE unbound population, including the laptop's pre-existing unbound cohort (~806,
#   source=cc_gateway, CC-CALLOSUM-TRUTH.md section 2 :197-201 and 10.4-H :2735-2749: only Leg 2 binds them). While ANY
#   unbound node exists the 250 idle steps do not run AND no batch is deposited (the daemon skips the drain, the tract
#   waits), logged loudly every cycle. The earlier justification for an arrival-scoped guard -- that a whole-graph guard would
#   "stall the drain forever" -- is withdrawn as a reason to ignore the cohort: that stall was the guard doing its job.
#   F2: an unbound arrival skipped at batch N was NOT protected from batch N+1's steps (the cross-batch case the merge's own
#   comment names, cc_topology_merge.py:584-591); the whole-graph guard now also blocks batch N+1's steps, and its deposit, for as
#   long as that arrival stays unbound. Still open: when Leg 2 resumes, merge and drain consolidation MUST be coordinated (one
#   consolidator at a time, or a combined arrival set), #876 and the #806 post-track lane.
#   F4: cc_deposit_step's docstring below ("the two drains ... never step at all", P240(3)) and cc_ng_host.py ~:1527 ("raw
#   experience has no synthetic consolidation cadence") are STALE against D24's cadence on the laptop CC (250 steps after each
#   batch, run by the daemon -- never inside drain_ingest_tract, which still never steps); the newer ruling wins and reconciling the
#   two texts is #895. drain_gateway_conduit's inert batch_size/idle_steps arguments coexist with this live path for the other
#   drain (LAW 3 shrapnel, #895).
# [2026-10-01] Z12 builder (Claude Sonnet 5.5), lane dual-pass-atomic-904, dispatch #13437 — #904 ROUND 2 (le-053 F1-1..F1-4)
# What: (F1-2) the rollback of a PRE-EXISTING node's re-stamp is NARROWED to exactly what the failed call wrote. It no longer clears and
#   refills the node's whole metadata dict and resets threshold / excitability unconditionally (that could erase a change the probation
#   sweep, which rewrites those fields under the same lock on the autosave pulse, made between the snapshot and the rollback). The eco
#   snapshots only the deposit's write-set (_STAMPED_ATTRS, _STAMPED_META_KEYS and every key of the call's own `meta`; a key absent
#   before is remembered as absent), records after the deposit what it wrote, and the rollback restores a field ONLY while it still holds
#   that value (a key absent before is deleted); a field another writer changed since, and every key the deposit never wrote, are left alone.
#   (F1-1) the failure record no longer says "rolled back, nothing of this turn remains" when vdb_kept_unprovable > 0: it says what was
#   KEPT ("kept: <n> pre-existing vdb entr(y/ies) that could not be proven"). (F1-4) the unrestorable count is printed
#   ("unrestorable=<n>;" after the existing vdb_kept_unprovable field; every earlier substring is unchanged).
# Why: le-053 corrections F1-1 / F1-2 / F1-4; #904's own principle "rollback exactly what the call wrote".
# How: _cc_deposit_memory_node is UNCHANGED (AST-pinned; tests/test_cc_capture_mutations_423.py passes unmodified): the narrowing lives on the
#   eco class. (F1-3) tests/test_cc_bind_atomic_904.py ties the restore's field list to the deposit's write-set by AST.
#   Known limit of a conditional restore: a field the sweep rewrote in between is left as the sweep left it (e.g. probation_remaining one
#   lower than the call's reset), so a failed attempt's reset can survive in that field; that is the price of never erasing another writer.
# [2026-10-01] Z12 builder (Claude Sonnet 5.5), lane dual-pass-atomic-904, dispatch #13259 — #904 FOLD-UP (Chief-003 + Exec Packet 497 (e))
# What: (C4/F1, a DEFECT) a failed attempt now leaves NO trace on a PRE-EXISTING node. _cc_deposit_memory_node re-stamps a pre-existing
#   node's threshold, intrinsic_excitability, probation/novelty metadata and poincare_dir and re-writes its vdb entry on EVERY deposit,
#   so a held exact-repeat poison turn re-stamped it every autosave cycle, forever. The eco now SNAPSHOTS such a node (threshold,
#   excitability, a copy of its metadata in key order, the references of its vdb entry) BEFORE delegating to the deposit, and rollback
#   RESTORES them last-in-first-out (metadata cleared and re-filled IN PLACE, because the vdb entry shares that dict; the original
#   embedding array put back exactly). A snapshot that cannot be taken makes the rollback report "a partial write REMAINS". (checker-038
#   correction 2) a vdb.get() that RAISES for a node that is NEW this call no longer orphans the entry the call inserted (a new node's
#   entry cannot belong to a node that existed, so it is deleted); for a PRE-EXISTING node whose probe raised the entry is kept and the
#   failure record counts it (vdb_kept_unprovable=<n>).
# Why: le-050 F1 / C4 (Exec Packet 497 (e): the re-stamp is a defect signal, a failed attempt must leave no trace); checker-038 C2.
# How: STAMP SNAPSHOT + RESTORE, not "move the stamp to commit time": the stamp is inside _cc_deposit_memory_node, which
#   tests/test_cc_capture_mutations_423.py extracts by AST and pins (it is UNCHANGED and that file passes UNMODIFIED), and it is called by
#   dual_record_outcome in the middle of extraction, so deferring it would mean editing or duplicating that function. The snapshot lives
#   on the eco class (same reason as the journal). The success path is unchanged (a successful exact repeat still re-stamps, as before).
#   Tests: tests/test_cc_bind_atomic_904.py (fold-up section).
# [2026-10-01] Z12 builder (Claude Sonnet 5.5), lane held-clock-visible-901, dispatch #13433 (Exec P501 / le-052 #4) — the want-create failure counter is READABLE
#   and its WARNING is rate-limited
# What: (a) new public accessor cc_want_create_failure_count() (the total of _CC_WANT_CREATE_FAILURES) so a caller (the docs daemon's autosave section)
#   can read the counter without reaching for a private name; the counter itself and the per-call WARNING text are unchanged. (b) the want-create
#   WARNING is rate-limited per function with the module's EXISTING interval, _PITH_WARN_INTERVAL_S (env CC_PITH_WARN_INTERVAL_S, default 60 s; no new
#   variable), keyed per fn_name in _cc_want_create_warn_last: the FIRST failure always warns, further failures inside the interval are COUNTED but not
#   re-logged. The counter is always bumped. Why: surface_wants_for_graph runs per deposit call (cc_ng_host.py ~:698), so an unthrottled WARNING could
#   fire per message while a want keeps failing (its id never lands in the graph, so every pulse retries it). (c) the daemon side (docs repo) reads this
#   counter's delta around its surface_wants call, because the per-want guard now absorbs the raise and the autosave section's except no longer sees it.
# How: logging / counting only; the success path (return value, graph, vdb, synapses) is byte-identical. Tests: tests/test_cc_swallows_915.py.
# [2026-10-01] Z12 builder (Claude Sonnet 5.5), lane held-clock-visible-901, dispatch #13220 — #915 (Exec Packet 495): three swallows made
#   loud, class-name only (the NG half; the daemon half is in the docs repo)
# What: (a) _cc_deposit_memory_node's recall-insert WARNING no longer prints str(exc) (a tree node's id/exception text can carry the user's
#   concept words): it logs a fixed reason code and the exception CLASS NAME. The function is otherwise UNCHANGED: it still re-raises and
#   still fabricates no rollback (pinned by tests/test_cc_capture_mutations_423.py, which passes unmodified). (b) surface_wants: the
#   graph.create_node(...) for a want was UNGUARDED, so one raise aborted the whole loop and the later wants were not materialized that
#   pulse. It is now guarded PER WANT: a failed create is counted, the loop CONTINUES with the next want, and ONE WARNING per call
#   reports it (_cc_report_want_create_failures: fixed reason code want_create_failed, the function name, counts, exception CLASS names
#   only). (c) surface_wants_for_graph: the outer `except Exception: logger.debug("Failed to create want node: %s", exc)` is the same
#   report (counter + ONE WARNING per call, class names only) instead of a DEBUG with str(exc); it was already non-fatal per want.
# Why: the autosave section swallows are what made a want/drain failure invisible; P370 (no silent swallow); the ONE RULE (Packet 494):
#   exception text that can name a node never goes into a log.
# How: comments / logging / guarding ONLY. Every success path is byte-identical (return value, graph, vdb, synapses). H-1: nothing about
#   the 182 cc_authored wants or the constitutional node is deleted, re-tagged, edited or de-flagged: a want whose create raised is not
#   created and is retried on the next pulse (its id is not in the graph); a partially-created node is NOT removed (no fabricated
#   rollback). Counters: the module-level _CC_WANT_CREATE_FAILURES (per function, in-process). Volume: a want that keeps failing is
#   reported once per call (per autosave pulse) while it keeps failing. generate_emergent_want is NOT touched (#905).
# [2026-10-01] Z12 builder (Claude Sonnet 5.5), lane dual-pass-atomic-904, dispatch #13058 — #904 (Exec Packet 487 via Chief-003):
#   the dual-pass write path is ATOMIC (or loud, with a truthful False), never silently half-written
# What: run_conversational_dual_pass keeps a write JOURNAL (on _CCConversationalDualPassEco: every node write, plus the
#   synapses and the hyperedge the bind creates) and, on ANY exception after the first write, ROLLS BACK exactly what this
#   call created (synapses, then the hyperedge, then the nodes it created, then the vdb entries it inserted fresh), restores
#   state["last_forest_id"] / state["primed_nodes"], logs at WARNING (ERROR if the rollback itself failed): hardcoded text,
#   stage, counts, exception CLASS NAME only (never str(exc), never node ids/text), and returns False. A failure BEFORE the
#   first write (nothing to roll back) is unchanged: debug line, False. _cc_bind_conversational_topology no longer swallows
#   its own failures (synapse loops, hyperedge, delay chain, previous-forest link) and no longer returns silently when the
#   forest is absent: they raise, so the dual pass can fail truthfully (it used to return True after a hyperedge failure).
#   It takes one optional keyword, journal=None; with None it is byte-identical on every success path.
# Why: #898 found the 147 conversational nodes were NEVER bound: forest/trees/windows were written in separate locks, the
#   bind was a later step, and any exception after the forest write skipped it at logger.debug and returned False (or, for a
#   failure inside the bind, returned True), while every caller ignored the result and the drain truncated the turn's bytes.
#   P370 (no silent swallow), LAW 4 (fix at the source), LAW 7 (raw experience is never destroyed), #805 (retry-safety).
# How: ATOMIC, not LOUD: a retry re-extracts a DIFFERENT concept set (LLM output) and binds only that attempt's tree_ids, so a
#   kept partial write leaves stragglers (#898 bucket 5) on every retry; rolling back makes a retry equal one clean pass.
#   Rollback never touches a node that existed before the call (only nodes this call created are removed; a pre-existing
#   node's idempotent re-stamp, which a successful repeat of the same turn performs anyway, is the one effect not undone) and
#   needs no edit to ng_embed.py (vendored) or any protected file: it uses the public Graph.remove_synapse /
#   remove_hyperedge / remove_node and SimpleVectorDB.delete. The journal lives on the eco class (not a new module-level
#   helper) so tests/test_cc_capture_mutations_423.py's AST extraction of those functions still works. _cc_deposit_memory_node
#   is UNCHANGED (its "no fabricated rollback" contract is pinned by that file). Tests: tests/test_cc_bind_atomic_904.py.
#   ADDENDUM (Exec Packet 489, class member (c)): surface_wants and surface_wants_for_graph wrote the want's seed synapse
#   (graph.create_synapse(source_node, want, weight=0.3)) inside `except Exception: pass`, which can leave a PROTECTED
#   (*_authored) want with no synapse and no hyperedge: a node that cannot bind, made silently. A failed seed synapse is now
#   counted per call (attempted / failed / distinct exception CLASS names) and reported by ONE WARNING per call
#   (_cc_report_want_synapse_failures: fixed reason code want_source_synapse_failed, the function name, the counts, class names
#   only: never str(exc), want text or ids). The EXISTING contract is unchanged: a want whose synapse failed is still created
#   and still returned in the list (that is what both functions did); weights (0.3), direction (source -> want) and sources are
#   untouched; authored wants are NEVER rolled back, edited or re-tagged (H-1). generate_emergent_want is NOT touched (#905).
#   Not changed, only noted: surface_wants_for_graph's outer `except Exception: logger.debug("Failed to create want node")`
#   (a failed create_node) and surface_wants' unguarded create_node remain as they were.
#   ADDENDUM 2 (Exec Packet 491, #257 write-path family; item (i), persisting last_forest_id, is in the docs daemon):
#   (ii) EVERY write site in _cc_bind_conversational_topology is loud and truthful: the five sites (tree_synapse, window_synapse,
#   hyperedge, window_chain, sequence_link) plus forest_absent are tagged as they are written (journal["site"], journal["attempts"]);
#   the ONE failure record now also carries site=<code> site_attempted=<n> site_failed=1; each failure is COUNTED in the caller-owned
#   state["bind_site_failures"][site] (in-process; it survives the rollback); the bind returns a small STATUS dict (trees, tree_pairs,
#   window_pairs, hyperedge, window_chain, sequence, has_trees, has_windows). TRUTH RULE (chosen and justified): a turn that HAS trees must
#   come out with EVERY forest<->tree synapse pair AND its hyperedge, else run_conversational_dual_pass raises inside its own try (site
#   bind_postcondition), rolls the turn back and returns False with its own loud record: all-or-nothing, because with the rollback a partial
#   bind is never kept (a retry equals one clean pass), and "any one tree pair" would let a bind that dropped most of the trees pass.
#   Fail-fast, not continue-and-count, for the same reason: after the first failed write the turn is rolled back anyway. (iii) a forest this
#   call CREATED, with NO predecessor (state pointer None / absent / itself) and NO trees and NO windows, logs ONE WARNING at birth (reason
#   forest_born_unbound, the three booleans, no id/text). Expected volume once (i) is in place: one per true first forest ever (or a pointer
#   to a vanished forest), not one per restart. Success-path return value and graph are byte-identical (golden); weights, directions,
#   delays, links and what a successful bind creates are unchanged; nothing existing is retro-wired.
# [2026-09-30] Z12 worker (Claude Sonnet 5.5), lane ingest-tract-swallow-781 — #794:
#   opt-in hold_on_failure on drain_ingest_tract (Chief-003 ruling / Exec P386)
# What: drain_ingest_tract gains ONE new LAST keyword, hold_on_failure=False. With
#   hold_on_failure=True the tract offset advances only past entries that were
#   absorbed (dual pass True) or legitimately filter-skipped (wrong type / wrong
#   source / empty text). At the FIRST entry whose absorb returns False or raises,
#   the drain stops, truncates ONLY the prefix before it, leaves that entry and
#   everything after it in the file, emits ONE logger.warning (fixed reason code +
#   exception CLASS NAME only, never str(exc)/entry text/path) and RETURNS NORMALLY.
#   With the default False every path, return value, side effect, log line and
#   exception is byte-identical to e4ebf982 (the new code is reachable only under
#   the flag). on_degraded, the daemon slice, per-step try, frame policy and the
#   drain cap stay parked.
# Why: #794. Today consumed_offset is set BEFORE the filters and BEFORE the absorb
#   attempt for every entry, so an entry whose absorb returned False or raised is
#   truncated out of the file and lost (LAW 7: raw experience destroyed, no retry,
#   no signal). Exec P386: the offset must advance only AFTER a successful absorb.
#   Default stays off because cc_ng_host.py:1526 (VPS half, Path B, Josh-gated) and
#   the Leg 1 return_consumed callers must not change behaviour.
# How: safe_offset tracks the offset just past the last absorbed-or-filter-skipped
#   entry; under the flag the truncation prefix is data[:safe_offset] instead of
#   data[:consumed_offset]. safe_offset == 0 takes the existing early return (file
#   not rewritten). The parse-failure handler, consumed_actual/return_consumed logic
#   and the truncate step are untouched. Tests: tests/test_cc_drain_hold_on_failure.py.
# [2026-09-30] Claude Code (Sonnet 5.5), Z12 worker seat, lane
#   daemon-recall-organism-756b (punchlist rows #756 / #779 / #780; slice B of the
#   #756 daemon recall-swallow chain; Chief-003 Decision 2 = YES with constraints,
#   decision B = pith_fallback is in this slice) — optional recall-failure REPORTERS
# What: four optional, default-None reporting kwargs. Nothing else changes.
#   cc_assemble_recall(..., on_degraded=None): on_degraded(code, exc), code one of
#   'monitor_race' (the `except RuntimeError` around the SurfacingMonitor harvest,
#   which had no log, no callback and no counter), 'pattern_completion_failed'
#   (the swallow INSIDE cc_pattern_completion_recall, reached through its on_error,
#   and the outer `except` around that call) and 'pith_fallback' (the un-Pithed
#   fallback branch, reported AFTER the existing on_pith_failure call). exc is the
#   exception object: a caller may log its class; its message can carry the prompt,
#   paths or secrets and must never go on a wire.
#   cc_pattern_completion_recall(..., on_error=None): on_error(exc) from its
#   `except Exception`; still returns [].
#   render_constitutional_core(graph, on_error=None) and render_wants(graph,
#   provenance=..., on_error=None): on_error(exc) from their own `except`; still
#   return "" -- so a caller can tell "raised" from the legitimate "nothing to
#   render" (both are ""). One private helper, _cc_report(callback, *args), calls a
#   reporter and swallows anything it raises.
# Why: Josh P361/P370 (LAW 4, fix at the source): the hook cannot tell a FAILED
#   recall from an EMPTY one because this origin swallows the failure.
#   plan-001.md rev 1 sections 1 A3/A5/A8, 3, 7, 9 I6/I7 and ruling #780. The
#   daemon (a later slice) counts, logs and puts the reason on the wire; this file
#   only REPORTS, exactly like on_monitor_error / on_pith_failure (no new logging,
#   no counters here).
# How: every reporter defaults to None and is appended LAST. With every new kwarg
#   unset the returns, side effects, log records and exceptions are byte-identical
#   to e4ebf982 (the call cc_assemble_recall makes to cc_pattern_completion_recall
#   carries no new kwarg unless on_degraded is set), so the VPS half (Path B,
#   cc_ng_host.py in Syl's process, which passes none) and pith_provider_context
#   (calls both with no reporter) are unchanged. Reporters are called after the
#   existing statements of each handler, so existing order is untouched; a raising
#   reporter never changes a return. NOT touched: pith_provider_context (sibling
#   silent failure B6 is listed for later), any vendored or protected file.
#   ROLLOUT: this must merge BEFORE the daemon wiring slice -- a daemon passing the
#   new kwargs to an older organism raises TypeError. Tests:
#   tests/test_cc_recall_reporting.py.
# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane in-transit-hold-905d, dispatch #14776) — #978: DOCSTRING ONLY (stale 'defined ONCE, in _unbound_nodes')
# What: the _cc_callosum_consolidate `guard` paragraph now says `held_unbound_nodes` defines the HOLD (#905-DELTA) over `_unbound_nodes`, which
#   defines the sweep-eligible unbound base / what "bound" means. No code change (proof in returns/build-002.md).
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane emergent-want-bound-905, dispatch #13491) — #905 ROUND 2: DOCSTRING ONLY
# What: the _cc_callosum_consolidate docstring's `guard` paragraph now says what "blocks" / "bound" means (defined once, in
#   cc_topology_merge._unbound_nodes: bound = NOT sweep-eligible, NOT "a complete turn"; turn completeness is not a gate condition).
# Why: Chief-003 ROUND 2 AMENDMENT item 2 (the P493 R1 sentence was absent from this file too). No code change; every non-comment,
#   non-docstring token is identical to ee94f7d2 (proof in returns/build-002.md).
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane emergent-want-bound-905, dispatch #13138) — #905 parts A and D
# What: (A) generate_emergent_want writes the new want BORN BOUND: in the SAME _step_lock block as create_node it
#   writes one synapse seed -> want (weight 0.3, the surface_wants pattern) per existing, de-duplicated seed. If NO
#   seed can be bound the new node is rolled back (graph.remove_node, same lock) and the function returns None with
#   ONE WARNING (fixed reason code, counts, exception CLASS name only); if SOME bind the want stays and the
#   shortfall is counted in ONE WARNING. The success return dict is byte-identical. (D) _cc_callosum_consolidate takes
#   keyword-only `guard=None, progress=None`: before EACH slice (incl. the first) a given guard is called WITHOUT
#   _concurrent_lock; a non-empty result stops the pass at the slice boundary (ONE logger.error, fixed reason code,
#   steps done/remaining, COUNT of blockers, NO ids, NO str(exc)), returns False with progress held=True; a raising
#   guard fails CLOSED (class name only, held and failed True). `progress` is filled on every return path. With
#   guard=None the pass is byte-identical to before.
# Why: Exec Packets 488 (C1) / 489 / 490. The emergent want was created with NO synapse: the whole-graph guards counted
#   it as blocking and the orphan sweep reaps '*_emergent' (not identity-protected) after grace 25. And the "is it
#   safe to advance the clock" check ran ONCE before 250 steps, so a node that became unbound between 25-step slices
#   (a hook deposit) aged against grace through the rest of the pass (le-047 C3, le-048 note 3a; row #906 / C10).
# How: the guard is built in ONE place, cc_topology_merge.whole_graph_guard, and both callers (the Leg 2 merge, the
#   daemon's drain) pass it; this function holds no predicate of its own (LAW 3 / LAW 4). Slice size, lock slicing,
#   success return value and what the steps do are unchanged.
# [2026-10-02] Claude Sonnet 5.5 (Z12 build worker, lane sweep-probation-p552, NG-3c / round 2) — the window logic is CANONICAL; this module keeps only the host's share
#   (Josh's ruling Exec P550 / P552; Exec P561 P1 + P2; Exec P563; Chief-003 Addenda 3-4; CC-CALLOSUM-TRUTH §8.13)
# FRAMING (Josh): the fair-chance window is SHARED MACHINERY being TESTED FIRST on the CC, not CC-specific code: the pioneer implementation of canonical §8.13 arrival
#   protection; rollout to other NeuroGraphs (Syl's) is Josh's call, LAW 8 gate per host.
# What: SUPERSEDED (removed, no alias, no leftover name): from NG-3 `3ccc749` the predicate `fair_chance_window_open`, the heartbeat state / `probation_heartbeat_arm` / the stamp /
#   the staleness check, `_probation_step_window_tick`, `_probation_clock`, `_probation_finite_number`; from NG-3b `7d34381` the window-size env reader and constant
#   (`CC_PROBATION_STEP_WINDOW`: the HOST (the laptop daemon) now reads the generic `NG_FAIR_CHANCE_WINDOW_STEPS` and hands it to the canonical registration). All of that logic is
#   now canonical and host-neutral in neuro_foundation.Graph (NG-4' `8a76bed`). KEPT here: `probation_population` (the advancer's own skip), now over the single constant
#   `PROBATION_UNADVANCED_CREATION_MODES = ("ingested",)` that the daemon also hands to the registration as the excluded modes (one definition); `_cc_deposit_memory_node` opens
#   the window through `graph.fair_chance_stamp`; `cc_update_probation` calls `graph.fair_chance_advance` once per node and `graph.fair_chance_heartbeat_stamp` at the end of a
#   NON-RAISING pass. All three are resolved with getattr, so on a graph that predates them (or an unregistered one) this module behaves EXACTLY as before: graduation,
#   the ramp, the release, `probation_remaining` / `probation_total` / `novelty_dampening` are BYTE-IDENTICAL (tests/test_cc_fair_chance_host.py runs the BASE function against this
#   one over seeded graphs, registered and unregistered).
# Why: Exec P563: the window is shared machinery, not CC code; LAW 4: no duplicated step / seed / decrement logic in the host.
# How: the host's switch is the daemon's call to Graph.enable_fair_chance_window, made once its advance runs on its own autonomic clock (LAW 8). Syl's host registers nothing.
# [2026-10-02] Claude Sonnet 5.5 (Z12 build worker, lane sweep-probation-p552, NG-3b / round 2) — the step window's size is a DEDICATED knob
#   (Josh's ruling Exec P550 / P552; Exec P561 P1; Chief-003 Addendum 2; CC-CALLOSUM-TRUTH §8.13)
# What: new `CC_PROBATION_STEP_WINDOW` (a positive integer, default `_CC_PROBATION_STEP_WINDOW_DEFAULT` = 10, today's effective window), read ONCE at
#   import into `_CC_PROBATION_STEP_WINDOW` by the pure `_read_probation_step_window_env`: absent => the default silently; set but invalid (empty /
#   non-integer / a bool-ish word / <= 0) => the default and ONE WARNING naming the variable and which rule failed. EVERY place the step window is
#   stamped, reset or seeded (`_cc_deposit_memory_node`, which an exact repeat re-runs, and `_probation_step_window_tick`'s seeding) now reads the new
#   constant. `CC_CONV_PROBATION_PERIOD` stays GRADUATION-ONLY and is untouched (probation_remaining, probation_total, the ramp, the release, #93).
# Why: P1 exists to DECOUPLE the fair-chance window (graph STEPS) from the graduation timer (pulses); sharing one variable would re-couple them at the
#   tuning level (raising the graduation period would silently lengthen the window). NG-3 (3ccc749) reused the graduation constant; this is its
#   correction commit (never amended).
# How: logging choice: IMPORT-TIME read (the same pattern as CC_CONV_PROBATION_PERIOD). In the laptop daemon main() configures logging before the
#   organism is first imported (lazily, from init_ng), so the WARNING lands in the daemon log; in any other host Python's last-resort handler prints
#   it to stderr, so it is not lost. Tests: tests/test_cc_fair_chance_window_p561.py pins the two knobs INDEPENDENT in both directions.
# [2026-10-02] Claude Sonnet 5.5 (Z12 build worker, lane sweep-probation-p552, NG-3 / round 2) — the fair-chance window is counted in
#   GRAPH STEPS and closes when the advancer stalls (Josh's ruling Exec P550 / P552; Exec P561 P1 + P2; Exec P562 Addendum 1;
#   CC-CALLOSUM-TRUTH §8.13)
# What: (1) the predicate the host registers is RENAMED to say what it returns: `probation_advances` -> `fair_chance_window_open(node)`
#   (no alias; one name everywhere). It is now the whole window logic, owned by the HOST layer: `probation_population(node)` (the old pure
#   test, `creation_mode != "ingested"`, defined ONCE, also `cc_update_probation`'s own skip) AND the completion HEARTBEAT is fresh (P2)
#   AND the step-keyed window is open (`probation_steps_remaining` a finite number, not bool, `0 < v < inf`; every other shape => False).
#   (2) P1: two NEW metadata fields, read ONLY by that predicate: `probation_steps_remaining` (the step-keyed window count) and
#   `probation_last_timestep` (the `graph.timestep` seen at this node's last decrement / stamp / seed). A decrement counts only if
#   `graph.timestep` advanced since THAT node's last one; many steps in one pulse count ONCE; a clock that goes BACKWARDS (restore from an
#   older checkpoint) resets `last` and never decrements. `_cc_deposit_memory_node` stamps both (an exact repeat RESETS both, as it resets
#   the old count); `cc_update_probation` seeds them fresh for any population node that lacks `probation_steps_remaining` (Z12 design call
#   A: this protects nodes that predate this build; a ONE-TIME bounded fresh window of at most CC_PROBATION_STEP_WINDOW stepped pulses).
#   (3) P2: `probation_heartbeat_arm(max_age_s)` (the host calls it once, BEFORE it registers the predicate) and a completion stamp written
#   at the END of a NON-RAISING `cc_update_probation`; armed and older than `max_age_s` => the predicate is False for EVERY node (today's
#   sweep) and logs ONE WARNING per stale episode; the next completion re-arms the latch. NOT armed (the VPS host, any host that does not
#   arm) => the heartbeat is not enforced. The window SIZE is its own knob `CC_PROBATION_STEP_WINDOW` (LAW 5; set in NG-3b, which superseded NG-3's reuse of CC_CONV_PROBATION_PERIOD).
# Why: §8.13 keys the fair chance to firing/wiring opportunities, which only happen when STEPS run (`Graph.timestep` is advanced only by
#   `step()`, neuro_foundation.py:2185; the age-on-write Tonic cycle explicitly does not advance it, :102-115), so ten autosave pulses with the
#   clock held used to expire a window with ZERO opportunities (le-062 E-1). And "spared only while ACTUALLY advanced" is only true if the
#   exemption closes when the pulse stalls (le-062 L1-L5). The GRADUATION FINDING: `probation_remaining` also drives the dampening ramp, the
#   unconditional release at 0, the `graduated` stamp (#93) and late graduation, and mirrors Syl's `_update_probation` on purpose, so it is NOT
#   step-keyed: the step count lives on the separate fields and graduation/ramp/release/`probation_total`/`novelty_dampening` are BYTE-
#   IDENTICAL to the base (tests/test_cc_fair_chance_window_p561.py runs the BASE function against the new one over 1000 seeded graphs).
# How: the sweep (neuro_foundation.py, protected) stays host-agnostic: NG-4 reduces its body to `getattr(self, "_fair_chance_window_open",
#   None)` and parses nothing. Corrected here, from the round-1 text: callosum arrivals WITH an embedding DO get a window (cc_topology_merge.py:450
#   -> `_cc_deposit_memory_node`); only the no-embedding structural install has no key. Who is covered is unchanged by this build.
# [2026-10-02] Claude Sonnet 5.5 (Z12 build worker, lane sweep-probation-p552, NG-1) —
#   probation_advances(node): the ONE definition of "this node's probation is actually
#   advanced by cc_update_probation" (Josh's ruling Exec P550 / P552, amended Exec P554,
#   placement fixed Exec P556; CC-CALLOSUM-TRUTH §8.13)
# What: new module-level predicate probation_advances(node). cc_update_probation's inline
#   `creation_mode == "ingested"` skip is REPLACED by `if not probation_advances(node):
#   continue`; behaviour is byte-identical to the base (proved by the seeded base-vs-new
#   comparison in tests/test_cc_probation_advances_p552.py).
# Why: the orphan sweep (neuro_foundation.Graph._collect_orphan_nodes, NG-2) spares an unbound
#   node while its probation window is open (§8.13's firing-keyed arrival exemption). A node is
#   only protected while its window is ACTUALLY being advanced, so the sweep must read the SAME
#   predicate as the thing that advances it (LAW 4: defined once, consulted, never duplicated).
#   Without it an `ingested` node (decremented only by the Ingestor's sweep, which on the laptop
#   runs only inside on_message <- handle_import) would be spared forever: an unbounded leak.
# How: the host registers this function on its graph (graph._probation_advances = ...; the
#   laptop daemon's init_ng does; Syl's host registers NOTHING, so her sweep is unchanged).
#   No cc_-named module is imported by canonical neuro_foundation.py (Exec P556).
#   CANONICAL SCOPE NOTE (recorded, NOT acted on): on Syl's process
#   neurograph_rpc._update_probation is conversation-gated (handle_after_turn) and decrements
#   EVERY node with the key INCLUDING `ingested` ones (this mirror skips them), so her window
#   does not advance on the autonomic clock (LAW 8). Whether and when she registers is Josh's
#   rollout decision, not this trial's.
# [2026-09-30] Z12 worker (Claude Sonnet 5.5), lane want-parser-legitimacy-810 (#810 turn 4,
#   le-019 F1/F2/F3/F4/F5) -- NEW FINAL FUNCTION (supersedes the turn-3 pin).
# What: (F2) a comma-continued JSON literal needs REAL JSON context: walking back over complete
#   JSON values (string, number, whole-token true/false/null, balanced [..]/{..}, `"key":`
#   members) separated by commas must reach an opening `{` or `[`
#   (_want_json_element_in_container, memoised per comma). A bare prose word, number or quoted
#   phrase + comma + quote (`it was 3, "[WANT]..`, `the answer is true, "..`, `In 2026, ".."`,
#   `She said "a", "b [WANT]x[/WANT]"`, `ok: true, ".."`) is NOT JSON and mints like base.
#   (F1) _want_url_continues_after_pair bisects a closer-offset list computed once per node
#   (was a `find` per opener: O(openers x node), 16.3 s at 40,000 openers; now linear, ~1.2 s).
#   (F3) an opener-only-masked mention opener that arrives while a real opener is pending pairs
#   with the next closer by the nearest-opener rule (mention_stack), so a mention pair inside a
#   real want stays inside it (whole want) and an unpaired mention yields nothing, as base does.
# Why: le-019: F2 dropped well-formed plain-prose wants (year/number/word + comma + quote) that
#   base minted; F1 was a quadratic scan under the mutation lock; F3 let a nested mention's closer
#   END the real want and mint a truncated marker-bearing text permanently.
# How: golden cases ASSERTED AGAINST BASE e4ebf982 in tests/test_cc_want_legitimacy_810.py; the
#   complete named-residual list (F4) is pinned there and in returns/build-004.md. The turn-2
#   "SAME result as base" claim remains qualified to the tested grammar (entry below).
# [2026-09-30] Z12 worker (Claude Sonnet 5.5), lane want-parser-legitimacy-810 (#810 turn 3,
#   le-016 MEDIUM #2/#3 + checker-018 notes 1-2) -- FINAL FUNCTION: lexical-guess false
#   negatives fixed ONCE, before the #801 counts.
# What: (1) CLOSERS: a marker inside a JSON literal / fence / inline code span is masked as a
#   mention only when it pairs with a MASKED OPENER IN THE SAME REGION (or nothing is pending);
#   a closer in a region with a REAL opener pending and no mention opener in that region is the
#   closer of the real want (`[WANT]rename "a", "b[/WANT]", next` mints again; so does a want
#   whose closer sits in a stray-backtick span). Quoted / escaped / code_adjacent / URL / link /
#   JSON-hug stay OPENER-only (audited: no closer is judged by a guess on its own text).
#   parse_wants pairs masked markers per region; _want_marker_mention_reason is replaced by
#   _want_marker_region (any marker) + _want_opener_mention_reason (openers). (2) in_url: the
#   blocklist _WANT_URL_TERMINATORS is gone; in_url needs URL-internal glue -- `/` always;
#   `? # = &` only when URL characters CONTINUE after the paired closing tag (marker INSIDE the
#   token); `: - em/en dash * _ ~ literal backslash-n .` etc. mint again. (3) true/false/null
#   in _want_json_opens_literal must be a whole token (`untrue,` `nonnull,` `intrue,`).
#   (4) in_link_target needs a real link: the `]` must close a balanced `[` whose left
#   neighbour is not an identifier / `)` / `]` (`arr[0](https://x/[WANT]..)` is not a link
#   target; it is still in_url because the opener is glued to a URL path). (5) this header's
#   turn-2 claim is qualified to the tested grammar.
# Why: le-016 (MEDIUM #2, #3) and checker-018 notes 1-2: each was a real want base minted that
#   the turn-2 build dropped by a lexical guess; P406's boundary is the OPENER's context. The
#   #801 counts and the repair plan's function pin are taken only on this final function.
# How: golden cases ASSERTED AGAINST BASE e4ebf982 in tests/test_cc_want_legitimacy_810.py.
#   Named residuals (still mint / still drop) are pinned there and listed in build-003.md.
#   The INFO skip log is unchanged and no content-aware exemption was added (correct).
# [2026-09-30] Z12 worker (Claude Sonnet 5.5), lane want-parser-legitimacy-810 (#810 turn 2,
#   #815 / Exec P414) -- three more structural skip reasons: in_json_string, in_url,
#   in_link_target.
# What: a want tag inside a JSON string literal, a URL or a markdown link destination is text
#   being CARRIED or TALKED ABOUT, not the author wanting something (P406: not legitimate).
#   in_json_string = the marker lies inside a structurally valid JSON string literal (opener
#   context `{` `[` `"key":` or a finished value + `,`; no raw newline; closer context `, } ] :`
#   or end) -- checked FIRST so a fence carried inside JSON is rejected as JSON, not by the
#   coincidence that its backtick runs pair up -- OR the opener is hugged by JSON-escaped quotes
#   (\"[WANT]\"). in_url / in_link_target = the OPENER is glued (no whitespace, no other WANT
#   marker between) to a run containing `scheme://` (a trailing `) ] } > " ' ` , ; ! .` ends URL
#   context -- SUPERSEDED BY TURN 3: the blocklist is gone, in_url is an allowlist of URL-internal
#   glue; see the turn-3 entry) or to the destination of a `](` (turn 3: a real link's `[` is required). Logged at INFO exactly like the existing skips.
# Why: Exec P414 (checker-016's examples). The reason applies to the context the OPENER sits in,
#   never to what the want text contains, and closers are never judged by the URL/link/hug
#   guesses (a closer guard would drop `[WANT]read https://x.com/a[/WANT]` -- the le-014 C1 class).
# How: _want_json_string_ranges / _want_json_opens_literal (constant work per quote: measured
#   0.5 s on a 200k-quote node), _want_opener_carried_reason; parity vs base stays 0 divergences
#   on a 3,726-shape corpus. NOT recognised (pinned in tests): single-quoted JSON-ish, link TEXT,
#   JSON with raw newlines, scheme-less reference definitions, bare prose mentions.
# [2026-09-30] Z12 worker (Claude Sonnet 5.5), lane want-parser-legitimacy-810 (#810 turn 2,
#   le-014 corrections C1-C4 + LAW 5) -- the parser again gives the SAME result as base for
#   every want shape in the TESTED grammar (the prefix x body x suffix corpus in
#   tests/test_cc_want_legitimacy_810.py), NOT for every string: le-016 found shapes outside
#   that grammar (a closer inside a quote fragment whose opener is plain prose, glued URL
#   punctuation, `untrue,`, `arr[0](...)`) that were fixed in turn 3 (entry above).
# What: the code_adjacent / escaped / quoted adjacency guesses now apply to OPENERS only
#   (_want_marker_mention_reason takes is_close); fences and inline code spans still mask any
#   marker. The three WANT_SKIP_* log bounds are env-sourced (CC_WANT_SKIP_SUMMARY_INTERVAL_S,
#   CC_WANT_SKIP_SEEN_MAX, CC_WANT_SKIP_DETAIL_PER_CALL_MAX; current values are the defaults).
#   The flood claim is stated exactly; the "never logs marker text" wording is corrected (the
#   detail line carries the literal marker token as its kind, no body or surrounding text).
# Why: le-014 C1 (HIGH): a real want that ENDS IN INLINE CODE (`[WANT]check `foo()`[/WANT]`)
#   was dropped, opener included, because the closer was treated as code_adjacent; base only
#   ever guarded the opener. C2: a real want ending inside a quote pair (`x "[WANT]I want
#   "x"[/WANT]" y`) was dropped the same way. The #801 id-equality guarantee requires the
#   parser to equal base on well-formed wants (LAW 4: fix at the source).
# How: is_close short-circuits after the two structural masks; golden cases and a
#   combinatorial parity corpus are asserted against BASE e4ebf982 in
#   tests/test_cc_want_legitimacy_810.py.
# [2026-09-29] Z12 worker (Claude Sonnet 5.5), lane want-parser-legitimacy-810 (#810,
#   Exec P406/P408) -- WANT extraction is structural LEGITIMACY, not a length limit.
# What: new pure parse_wants(content) -> WantParse (+ want_id_for_text) is the ONE
#   implementation of "is this [WANT]...[/WANT] a real want?". A want is the text between
#   a real [WANT] opener and its paired [/WANT] closer, NO length limit. A marker inside a
#   fenced block / inline code span / right after a backtick / backslash-escaped / wrapped
#   in a quote pair is a MENTION; the closer pairs with the NEAREST live opener; a stray
#   closer, an unclosed opener and an empty pair are skipped. surface_wants calls it and
#   logs skips at INFO in a flood-bounded form (per-call summary on change + hourly
#   heartbeat, per-marker detail once per (node, offset, reason), <=50 detail lines per
#   call, no want body or surrounding text in the log -- the literal marker token is the kind). _WANT_RE (the 600-char pattern) is removed. WANT_MAX_CHARS stays
#   DEFINED only because render_wants still clamps with it -- render_wants is NOT touched
#   here (Exec P408: the standing "## What I Want" block is retired in a separate turn).
# Why: Josh (Exec P406): "WANTs just need to have the WANT brackets on either side. A parser
#   just needs to make certain that it's legit, and not just us talking about WANTs." The
#   2026-09-16 600-char cap silently DROPPED a genuine long want (LAW 7: no truncation of raw
#   experience) and was a length heuristic standing in for legitimacy. The repaired #801
#   ids must equal a re-parse's ids, so both must call the same function (LAW 3/4).
# How: see the plan/return doc handoffs/z12-want-legitimacy-810/returns/build-001.md.
#   Id derivation, node metadata and synapse are UNCHANGED (cc:want::+sha1(inner)[:16]).
#   surface_wants_for_graph (host twin, #755), neurograph_rpc.py and cc_ng_host.py untouched.
# [2026-10-01] Claude Sonnet 5.5 (Z12 lane surfacing-whole-812, dispatch #12618) — #812 turn 1:
#   cc_pattern_completion_recall resolves WHOLE
# What: the resolve_surface_content call in cc_pattern_completion_recall drops its
#   max_chars=300 argument (the resolver's new default is NO bound). Nothing else here changes;
#   the _CC_PITH_* caps, pith_stage2_keyframe and the Pith budget code are untouched (#813).
# Why:  Exec P468 / Josh "fix stuff correctly, not monkey patch" + Chief-003 ruling (addendum 1):
#   same lossy-clipping class as the resolver's 240 default; fixed at the source.
# How:  One argument removed, on the e4ebf982 base. Expected conflict with #813 (84a0968a) at
#   this call site: see the #812 turn-1 return for the correct merged form.
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code), lane pith-clip-removal-813,
#   TURN 7 (dispatch #11293) -- le-027 N-2/N-4/N-5 (N-1 RECORD ONLY)
# What: (N-2, MEDIUM) a dead or missing identity guard still FAILS CLOSED (everything pinned, WARNING
#   once per id) and now ALSO emits ONE count-only WARNING per call on BOTH paths that use the probe
#   (Stage 3 / Pith-ON and the un-Pithed renderer): "K of N items pinned because the identity guard
#   failed (<type>); L1 X chars vs budget B" -- counts and exception type only, no text, no ids -- so a
#   persistent dead guard is not silent while it leaves the L1 far over budget. A working guard emits
#   nothing. WARNING because a dead identity guard is an operational fault (the drop/reference lines are
#   INFO because they report designed behaviour). The probe carries the per-call record
#   (_pinned.guard_failed); _cc_log_guard_pins turns it into the line. (N-4) the coherence vocabulary is
#   left as five code sites and a TEST asserts they all equal _PITH_COHERENCE_STATES (deriving them
#   would restructure the ladder / weight dict / order tuple = behaviour, for a LOW note). (N-5) tests only.
# Why: le-027: fail-closed is right (Duck Ethics) but silent while the L1 stays over budget.
# How: cc_ng_host.py, surfacing.py, surface_resolver.py untouched. C4 (no drop line for pins) stays the
#   Exec's open note; the count-only line is now its signal. N-1 (the optimistic-limit band) is recorded
#   in returns/build-007.md and NOT implemented.
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code), lane pith-clip-removal-813,
#   TURN 6 (dispatch #11238) -- le-025 C-1..C-5 (Chief ruling docs e19962de)
# What: (C-2, MEDIUM) _pith_provider_node_limit is now the OPTIMISTIC bound -- measured from the
#   SMALLEST overhead the renderer can have (one ordinary alert-free line, no sources/anchors/
#   relations) -- so it can never be smaller than what fits: every (core, budget, node) that BASE
#   or the turn-5 parent rendered whole renders whole (swept in a test); a node that truly cannot
#   fit is caught at admit as a loud NEVER-FIT drop (whole-or-absent). Turn 5 measured the worst
#   case and dropped/referenced small nodes in a tight budget. (C-5) _cc_pin_probe FAILS CLOSED: a
#   raising or missing identity guard => PINNED + a WARNING (id, exception type; first-seen), on
#   Stage 3 and the un-Pithed renderer; base failed soft to NOT pinned at DEBUG. (C-4) the alert
#   coherence set is defined once (_PITH_COHERENCE_STATES / _PITH_ALERT_COHERENCE) and drives both
#   the renderer and the limit. (C-3) the docstring no longer claims admit catches an over-estimated
#   band, and the INFO line says "above the reference limit L", not "over-budget". (C-1) a test
#   compares the pin probe with e4ebf982's closure: identical except the deliberate C-5 change.
# Why: le-025 (fresh law enforcer, turn 5): C-2 violated "never so small it drops what fit before";
#   the Chief ruled C-5 (identity fails toward keeping content, #92 / Duck Ethics).
# How: cc_ng_host.py, surfacing.py, surface_resolver.py untouched. The same fail-soft exists in
#   shared code -- recorded in returns/build-006.md (row #829), NOT edited.
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code), lane pith-clip-removal-813,
#   TURN 5 (dispatch #11114) -- le-022 N1/N2/N3 + LOWs (Chief ruling docs 3604cfb1)
# What: (N1) _cc_render_unpithed exempts identity-protected items from the budget EXACTLY as
#   Stage 3 does (one shared test, _cc_pin_probe): rendered whole, never ranked/dropped/swapped for
#   a reference, never lost to the strict-prefix stop; ordinary items behave as before (golden
#   unchanged). Pins are NOT folded into THE ONE rule (C4 stays the Exec's open note).
#   (N2) _pith_whole_node_reference no longer promises "so its concepts follow": nothing is said
#   for a 0-tree node; "K of N concept trees follow" when the L1/un-Pithed path knows; "where they
#   fit" on the provider path; a hedge that trees may cover only part of it. (LOW) the constants
#   800/200 are gone: _pith_provider_node_limit MEASURES the renderer's own worst-case shell and
#   minimum line overhead. (T1) pointer comments: _cc_monitor_items_whole / _format_cc_monitor_block
#   are an INTERIM fork that the #812 source fix deletes.
# Why: le-022 (law enforcer, final diff): N1 was an identity-continuity regression on the DEFAULT
#   (gate-off) path -- an identity-protected item was budget-droppable there.
# How: cc_ng_host.py, surfacing.py, surface_resolver.py untouched.
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code), lane pith-clip-removal-813,
#   TURN 4 (dispatch #11061) -- checker-022 C1/C2/C3 (Chief ruling docs 2a3b5fbf)
# What: (C1) pith_stage3's docstring step 5 now describes what the body does: an over-budget
#   line, even the first, is skipped whole-or-absent with the INFO drop line (D8 CONFIRMED: the
#   first-line overrun guard stays REMOVED). (C2) _cc_monitor_items_whole: an item whose
#   re-resolve RAISES is dropped (never the shared 240-char snippet), with a WARNING naming the
#   node id + exception type (no text; id first-time-seen); the pattern-stream dedupe set is
#   recomputed from the SURVIVING monitor items so the dropped node's whole twin is not lost too.
#   (C3) _pith_reference_text logs ONE INFO line when it leaves concept trees out (included /
#   total / left out, counts only). (C4) pinned Stage-3 lines sit off-budget: recorded, unchanged.
# Why: checker-022 PASS-WITH-NOTES on turns 2+3. No behaviour change on the happy path.
# How: cc_ng_host.py, surfacing.py, surface_resolver.py untouched.
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code), lane pith-clip-removal-813,
#   TURN 3 (dispatch #11011) -- #817 REVERTED, DEFERRED to the post-track VPS/daemon lane
# What: pith_compress_history and the PithMetrics.history_* group (7 fields, reset/snapshot
#   lines, record_history_compression) are RESTORED exactly as at turn 1; cc_ng_host.py is
#   byte-identical to base e4ebf982 again (its _handle_compress_history + dispatch entry).
# Why: Chief ruling (docs 084b4161): pith_compress_history has TWO LIVE Python callers --
#   cc_ng_host.py:974 (VPS host) and cc-ng-daemon.py:1617 (laptop daemon) -- via a call-time
#   import that fails soft. It is NOT dead (Condensate had no live RUST caller; that was true and
#   beside the point). Retiring it must remove the function TOGETHER with BOTH handlers (LAW 3),
#   which spans two other lanes, and this track is laptop-only (P329: no VPS-affecting change).
# How: `git revert` of afc9b3e; the only conflicts were changelog/test-file stacking. Every other
#   turn-2 change (#816, the ONE budget rule, #818, #819, F2/F6/F8) is kept. The keyframe
#   primitive's docstring again names pith_compress_history as its only remaining caller.
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code), lane pith-clip-removal-813,
#   TURN 2 (dispatch #10952) step (2c) -- #819: an over-budget node surfaces through its TREES
#   plus a one-line whole-node REFERENCE
# What: a node whose whole text cannot fit the usable envelope renders as ONE reference line (id,
#   size, date, tree count) and its concept trees -- whole, each small -- on the provider path
#   (pith_connected_activation_basins(node_limit=...): the trees are the basin's ordinary graph
#   neighbours), the Pith-ON L1 path (_pith_reference_lines) and the un-Pithed path
#   (_pith_reference_items); one INFO line (_pith_log_reference). Text-derived anchors are not
#   mined from the unshown whole (metadata anchors are kept).
# Why: Exec P417 (LAW 7: raw means complete; Josh P360: a long turn stays ONE node / one forest);
#   brief TURN 2 item 4; le-017 F8. NO split at ingest, NO new node type, the node is never
#   modified and still activates and learns in full: ONLY its RENDERING changes.
# How: DEPENDENCY -- for pre-PASS-2 forests the trees cover only the first 2,000 chars until
#   PASS 2 (the laptop TID) runs; full coverage arrives with PASS 2. Until then the reference is
#   honest that the whole exists and is not shown. The per-node limit is budget - core - 800
#   (a fixed shell allowance); a node between that and the true learned budget is still caught,
#   loudly, as a never-fit assembly at admit.
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code), lane pith-clip-removal-813,
#   TURN 2 (dispatch #10952) step (2b) -- #818: EVERY drop is loud
# What: _pith_log_drop (ONE INFO line per call and reason: count, total chars, reason; ids
#   named first-time-seen only, bounded -- the #810 skip-log shape) now reports the drops that
#   were silent: neighbours declined by CC_PITH_PROVIDER_MEMBERS (member_limit) and by
#   CC_PITH_PROVIDER_DEPTH (depth_limit) in pith_connected_activation_basins, basins skipped at
#   >=60% overlap, and recall results beyond the root/result count k (roots).
# Why: brief TURN 2 item 3 (Exec P416): silent member/overlap/count-limit drops. Counted: only
#   candidates the walk REACHED and declined, and only if they appear in no selected basin.
# How: no behaviour change to WHAT is selected. NOT covered (named in build-002): Stage-1
#   clutter/dedup drops (counted in _PITH_METRICS, not logged) and the graph engine's own
#   max_surfaced cap inside _harvest_associations (shared neuro_foundation code).
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code), lane pith-clip-removal-813,
#   TURN 2 (dispatch #10952) step (1b) -- #816 on BOTH Pith-ON streams AND the gate-off path,
#   plus THE ONE budget rule
# What: (1) cc_assemble_recall asks recall for whole_content=True (was the 300-char snippet on
#   the Pith-ON pattern stream AND the gate-off block). (2) The SurfacingMonitor stream -- cut
#   at 240 chars in SHARED surfacing.py/surface_resolver -- is re-resolved WHOLE by node_id
#   through a CC-ONLY route (_cc_monitor_items_whole); the shared modules are not edited.
#   (3) The gate-off / Pith-failure rendering is _cc_render_unpithed: a CC-side monitor block
#   (_format_cc_monitor_block, layout-identical, no 200-char cut) + the Active Recall block,
#   size controlled by HOW MANY items under the existing cc_l1_budget. (4) THE ONE BUDGET RULE
#   (_pith_admit_strict_prefix + _pith_log_budget_drop) now serves _pith_provider_admit,
#   pith_stage3 and _cc_render_unpithed: whole or absent; strict rank prefix on the remaining
#   envelope; a never-fit unit is skipped (not emitted over budget, does not end the prefix);
#   every drop is ONE INFO line naming count, total chars and never-fit node ids (bounded,
#   first-time-seen). pith_stage3's "keep the first line even if it exceeds the budget" guard is
#   REMOVED (silent overrun; le-017 F2 / checker-019 C3). (5) _pith_unified_rank factored out of
#   pith_stage3 (no behaviour change; the un-Pithed path ranks by the same rule).
#   _pith_fit_connected_line deleted (dead after the rewrite).
# Why: checker-019 C1 (HIGH), C3; le-017 F1 (HIGH), F2, F8; Exec P410(c)/P416; brief TURN 2
#   ADDENDUM 2. Shared surfacing.py/surface_resolver.py serve Syl's /assemble (P329, #812).
# How: see handoffs/z12-pith-clip-813/plan-002.md sections 2-3. Decision D8 for the reviewer:
#   the Stage 3 first-line guard is gone, so a recall whose every item is never-fit is empty
#   (loudly) until #819's reference form supplies content.
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code), lane pith-clip-removal-813
#   (dispatch #10841) — remove the Pith per-node clip; the budget is met by fewer
#   WHOLE items, loudly
# What: (1) _pith_node_text returns the node's text WHOLE; CC_PITH_PROVIDER_NODE_CHARS
#   (_CC_PITH_PROVIDER_NODE_CHARS) is deleted; pith_effective_config() reports the
#   retired name as resolved=None / authority="retired" and it stays in
#   _PITH_CONFIG_KEYS, because env == resolved == that tuple == the host allow-list
#   is asserted (tests) and the host is not edited in this change. (2) _pith_fit_statement and the water-filling in
#   _pith_fit_connected_line are deleted: a connected line is admitted whole or not at
#   all. (3) _pith_provider_admit keeps its strict rank-order prefix, additionally skips
#   an assembly that cannot fit even an EMPTY envelope (so one giant top-ranked
#   assembly cannot blank every other), and logs ONE INFO line -- count and total
#   rendered size -- whenever it drops anything. (4) pith_stage3 loses its keyframe
#   fallback for an over-budget line (drop whole instead) and logs the same INFO line
#   for what it drops; the never-empty-L1 guard is kept. (5) _pith_node_sources no
#   longer slices labels to 80 chars. (6) cc_pattern_completion_recall gains the
#   opt-in whole_content=False parameter (default byte-identical); pith_provider_context
#   passes True so the node-text fallback is not the 300-char snippet. (7) the
#   prefetch LOD keyframe staging (and the query embed only it used) is removed.
#   (8) pith_stage2_keyframe's docstring states it applies only WITH its delta.
# Why: Exec P411/P413 via Chief-003 (Josh: no truncation). A keyframe whose delta is
#   discarded is a cut with a marker; no budgeted output has room to carry the delta,
#   so a keyframe never satisfies a binding budget. Audit + reasoning:
#   handoffs/z12-pith-clip-813/plan-001-audit.md.
# How: repaired in place (LAW 3, same function names). Untouched on purpose: the
#   REJECT-LOUDLY guards MAX_INSTRUCTION_CHARS / MAX_QUEST_CHARS (a whole-request
#   refusal is not a cut), pith_stage2_keyframe itself (pure, still used by the
#   history-compression handler), the now-unused CC_PITH_PREFETCH_SUMMARY_CHARS /
#   CC_PITH_PREFETCH_LOD_DIST constants (dead-tunable removal touches host allow-lists).
# [2026-09-26] Z2 worker (openrouter/deepseek/deepseek-v4.1-flash, OpenCode/T3 Code),
#   lane z2-ng-recall-passthrough-restore-001 — restore the un-Pithed recall
#   fallback in cc_assemble_recall (LAW 3, pre-46f9cf8 behavior)
# What: an exception inside the gated Pith block (CC_PITH_ENABLED) again falls
#   through to the un-Pithed monitor_ctx/pc_block return -- monitor_ctx + "\n\n"
#   + pc_block, or whichever of the two is non-empty -- exactly as before
#   46f9cf8. The notice-only return is gone: cc_pith_unavailable_notice and
#   _PITH_NOTICE_MSG_MAX are deleted as dead code (LAW 3; nothing else
#   referenced them). The rate-limited log lines say "falling back to un-Pithed
#   rendering" and keep the _stage name; the docstring and the
#   _PITH_METRICS.record_failure docstring describe the fail-soft fallback
#   again. Everything frozen stays frozen: the on_pith_failure callback and its
#   call site, cc_pith_failure_text, cc_deposit_pith_failure, the raw failure
#   deposit and its wiring in cc_ng_host.py and cc-ng-daemon.py,
#   _PITH_METRICS.record_failure() and the rate limit, the Pith success path,
#   the four inner fail-softs, victim capture, and gate-off (byte-identical).
# Why: Josh's ruling 2026-09-26 ("When Pith fails, there HAS to be pass-through,
#   in order for any model to remain useful to fix anything else. No massive
#   history CAN build up, anyway, between KISS and Pith."); Exec P309(1)/P311(2);
#   LAW 3 restore of the pre-46f9cf8 fallback; the notice envelope is superseded.
# How: remove `return cc_pith_unavailable_notice(_stage, exc)` from the except in
#   cc_assemble_recall; delete cc_pith_unavailable_notice and _PITH_NOTICE_MSG_MAX;
#   reword the logs/docstrings. Tests: replace every notice assertion with a
#   fallback assertion at each injection point, and restore the concat-fallback
#   test 46f9cf8 replaced. Assignment z2-ng-recall-passthrough-restore-001.
# [2026-09-28] Z2 worker (openrouter/deepseek/deepseek-v4.1-flash, OpenCode/T3 Code),
#   lane z2-ng-recall-passthrough-restore-001, Exec P313(2c) — comment-only follow-up:
#   cc_assemble_recall's _stage comment said "named in the notice"; the notice
#   envelope is gone, so it now reads "named in the fallback log line".
# -------------------
# [2026-09-26] #640 coding worker (deepseek/deepseek-v3.2, OpenCode/T3 Code) — Pith connected-line label reads `line.epistemic`; `- Keyframe:` becomes `- Root:`
# What: `_pith_render_connected_line` now reads `CacheLine.epistemic` for the heading
#   label, showing `"learned from substrate"` only when `line.epistemic == "learned"`;
#   otherwise shows the `epistemic` value directly. The first bullet changes from
#   `"- Keyframe:"` to `"- Root:"` per chief-640-ruling-001 Ruling B. CacheLine
#   docstring updated to mention `_pith_render_connected_line` as reader.
# Why: chief-640-ruling-001 APPROVE; chief-b1-ruling-003 §2 (LAW 4, wire the field,
#   do not delete it); PRD §5.3.1 ("learned, never promoted to 'verified'");
#   punchlist #640.
# How: `label = "learned from substrate" if line.epistemic == "learned" else line.epistemic`;
#   `"- Keyframe:"` → `"- Root:"`. New tests:
#   `test_source_coherence_and_exact_anchors_remain_attached_to_basin` (:161) stays unchanged
#   (still passes), plus new test A (verified line) and new test B (root label) in
#   `tests/test_pith_provider_context.py`. Assignment z2-640-pith-epistemic-label-001.
# [2026-09-26] openrouter/deepseek/deepseek-v4.1-flash (OpenCode harness on T3 Code),
#   lane z2-remove-deposit-step-flag-001 r2 — drain_ingest_tract Locking wording
#   (P187 finding 3)
# What: the drain_ingest_tract docstring's Locking sentence is reworded
#   comment-only — the caller's graph._concurrent_lock (punchlist #643) is what
#   makes the dual pass's mutation of the graph safe. No code line moves.
# Why: Chief-003 ruling on the Lane 3 twin ripple; P187 LE finding 3 / Grok note 10.
# How: one docstring sentence reworded. The flag gate is removed per P240(3); R2's
#   AUTOSTEP pairing is an activation condition (CALLOSUM-TRUTH §8.13; the laptop
#   is an exec ruling), not met here. CC_NG_AUTOSTEP untouched.
# -------------------
# [2026-09-26] openrouter/deepseek/deepseek-v4.1-flash (OpenCode harness on T3 Code),
#   lane z2-remove-deposit-step-flag-001 — Exec P240(3)/P242: drop the deposit-step flag
# What: the _CC_NG_DEPOSIT_STEP flag and its comment are deleted; the two drain
#   step sites in drain_ingest_tract and drain_gateway_conduit are deleted, so
#   neither drain ever steps; cc_deposit_step stays (its only caller is the
#   Stop-side host door, via P240(3)'s unconditional `if step:`). The Doors
#   paragraph, the drain docstrings and the cc_novelty docstring no longer name
#   the flag. The #643 lock is intact and the twin cc-ng-daemon.py is untouched
#   (Z12's rebuild item); the drains stay for P240(4).
# Why: Exec P240(3) (chief-p240-commission-001), P242 (transcripts don't step
#   the graph) and Chief's approval of the Z2 Lane 3 proposal.
# How: flag and both `if _CC_NG_DEPOSIT_STEP:` blocks removed; comment/docstring
#   corrections only elsewhere. R2's AUTOSTEP pairing becomes an activation
#   condition (gate removed per P240(3)); do not flip CC_NG_AUTOSTEP.
# -------------------
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
# -------------------
# [2026-09-25] Claude Code (kimi-k2.7-code) — Packet 214 D-3: remove dead
#   extraction_failed branch in run_conversational_dual_pass.
# What: the check for _result.get("extraction_failed") and the comment
#   "forest is real experience even when tree extraction failed" are removed.
# Why:  ng_embed.dual_record_outcome now raises DualPassIncompleteError on
#   pass-2 failure (R3 atomicity); a returned dict means extraction succeeded
#   or legitimately returned no concepts. The old branch was unreachable.
# How:  replaced with a comment describing the raise-based contract and kept
#   the _cc_bind_conversational_topology call unchanged.
# -------------------
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
# [2026-09-22] Grok 4.6 — punchlist-001 B1/B9: intra-turn window chains + atomic dual-pass
# What: run_conversational_dual_pass calls embed_windows (non-fatal), deposits
#   {target_id}::window::{i} via _cc_deposit_memory_node(..., index_in_recall=False),
#   and passes window_ids into _cc_bind_conversational_topology for forest links,
#   hyperedge membership, and #257 delay-chains. Short turns (windows==()) add
#   no window nodes. Deleted the extraction_failed partial-success raise and
#   the "forest is real experience even when tree extraction failed" comment.
# Why: Mirror neurograph_rpc._run_conversational_dual_pass. Dual-pass is atomic
#   or there is no deposit; forest-only is not an outcome. Josh: polychrony
#   window order is substrate structure; windows stay out of vector_db.
# How: Same sampler as the prev-forest link (randint(2, max(2, _CC_CONV_SYNAPSE_DELAY_MAX))).
#   DualPassIncompleteError from NGEmbed already prevents a forest write.
# [2026-09-22] Claude Code — correct stale Leg1 rationale comment (source, LAW 4).
# What: Update the 2026-07-27 Leg1 changelog entry's "Why" (and its header) to
#   reflect the current laptop/VPS embedding and Tonic responsibilities.
# Why: Josh ruled on 2026-09-22 that the laptop's RAM upgrade removes the constraint
#   that kept it from running the Tonic. The old comment ("laptop does zero embedding
#   by design ... VPS is the sole Arborist for both hemispheres") had become false and
#   was misdirecting architecture decisions. LAW 4: fix the stale claim at its source
#   in this file, not in downstream consumers.
# How: Reword the Leg1 header and "Why" paragraph only; no code or behavior change.
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
#   conduit, laptop -> VPS
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
#   tree structure). Per Josh's 2026-09-22 ruling (EXECUTIVE-TODO Packet 052)
#   and Packet 073(C)/(D), the laptop's RAM upgrade lets it run its own Tonic
#   and embedding (protoUniBrain) instead of borrowing the VPS's for that --
#   once it embeds for itself, Leg 1 retires. Tree growth (TID) stays VPS-side
#   regardless (ng_embed.py's concept extraction needs TID; the laptop has
#   none), so Leg 2 still carries that structure back down. CC-CALLOSUM-TRUTH.md
#   §1.4/§1.4.1/§8.5.1 still describe the pre-ruling state; Packet 073(D)
#   assigns Z12 to update it with a dated entry. Spec:
#   docs/superpowers/plans/2026-07-27-cc-corpus-callosum-leg1-spec.md
#   (superseded in part by the above rulings, pending that CALLOSUM-TRUTH update).
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
import sys
import threading
import time
import uuid
from bisect import bisect_left, bisect_right
from collections import Counter, OrderedDict
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Any, Dict, List, Optional, Tuple

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


_CC_WANT_CREATE_FAILURES = {"surface_wants": 0, "surface_wants_for_graph": 0}   # #915: failed want creates, per function, in-process
_cc_want_create_warn_last: Dict[str, float] = {}    # fn_name -> monotonic time of its last want-create WARNING (rate limit, #P501)


def cc_want_create_failure_count() -> int:
    """The TOTAL number of want creates that failed in this process (both surface_wants and surface_wants_for_graph). Read-only:
    a caller that wants to notice a failure the per-want guard absorbed reads it before and after the call (its delta)."""
    return int(sum(_CC_WANT_CREATE_FAILURES.values()))


def _cc_report_want_create_failures(fn_name, attempted, failed, exc_types):
    """#915 / Exec P495: COUNT and report at most ONE WARNING per surface_wants* call (rate-limited per function by the module's
    existing _PITH_WARN_INTERVAL_S, Exec P501) in which a want's create_node raised.
    Hardcoded reason code + the function name + counts + exception CLASS NAMES only: never str(exc), a want's text, a
    node id or a path. The failed want is NOT created (nothing is half-rolled-back, H-1) and the loop went on to the
    next want; it is retried on the next pulse because its id is not in the graph."""
    if not failed:
        return
    _CC_WANT_CREATE_FAILURES[fn_name] = _CC_WANT_CREATE_FAILURES.get(fn_name, 0) + failed       # ALWAYS counted
    now = time.monotonic()
    last = _cc_want_create_warn_last.get(fn_name)
    if last is not None and (now - last) < _PITH_WARN_INTERVAL_S:
        return              # rate-limited (the module's existing interval): counted above, not re-logged
    _cc_want_create_warn_last[fn_name] = now
    logger.warning(
        "CC want create FAILED: reason=want_create_failed fn=%s attempted=%d failed=%d exc_types=%s -- the loop "
        "continued with the remaining wants; no authored want was deleted, re-tagged or edited",
        fn_name, attempted, failed, ",".join(exc_types[:3]))


def _cc_report_want_synapse_failures(fn_name, attempted, failed, exc_types):
    """#904 / Exec P489: ONE WARNING per surface_wants* call whose want seed-synapse(s) failed.
    Hardcoded reason code + the function name + counts + exception CLASS NAMES only: never
    str(exc), a want's text, a node id or a path. A want whose synapse failed IS created and IS
    reported (the existing contract, unchanged) but is UNBOUND; authored wants are never rolled
    back (H-1). Volume: a want that exists is skipped on later pulses, so each failure is
    reported once, not per autosave."""
    if not failed:
        return
    logger.warning(
        "CC want seed synapse FAILED: reason=want_source_synapse_failed fn=%s attempted=%d failed=%d "
        "exc_types=%s -- the want node(s) exist and are reported as before but are UNBOUND (no synapse "
        "from the source node); wants are not rolled back",
        fn_name, attempted, failed, ",".join(exc_types[:3]))


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
        syn_attempted = syn_failed = 0
        syn_exc_types = []
        create_attempted = create_failed = 0
        create_exc_types = []
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
                    create_attempted += 1
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
                        syn_attempted += 1
                        graph.create_synapse(nid, want_id, weight=0.3)
                    except Exception as syn_exc:  # noqa: BLE001
                        # #904 / P489: LOUD, not silent. The want stays (authored, H-1) and is still reported.
                        syn_failed += 1
                        if type(syn_exc).__name__ not in syn_exc_types:
                            syn_exc_types.append(type(syn_exc).__name__)
                    open_wants.append({
                        "id": want_id,
                        "text": inner,
                        "provenance": "cc_authored",
                        "state": "open",
                        "source": nid,
                    })
                except Exception as exc:  # noqa: BLE001
                    # #915: LOUD, not a DEBUG with str(exc). Non-fatal per want; counted; class name only.
                    create_failed += 1
                    if type(exc).__name__ not in create_exc_types:
                        create_exc_types.append(type(exc).__name__)
        _cc_report_want_create_failures("surface_wants_for_graph", create_attempted, create_failed, create_exc_types)
        _cc_report_want_synapse_failures("surface_wants_for_graph", syn_attempted, syn_failed, syn_exc_types)
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
# A want is the text between a REAL [WANT] opener and its paired [/WANT] closer --
# NO length limit (Josh, Exec P406: "WANTs just need to have the WANT brackets on either
# side. A parser just needs to make certain that it's legit, and not just us talking
# about WANTs."). Legitimacy is STRUCTURAL: parse_wants() below. History: on 2026-09-16
# prose that merely *mentioned* `[WANT]` let the old unbounded `(.*?)` run to a far
# `[/WANT]` (118 of 182 CC want-nodes >600 chars, one 136,449; "## What I Want" reached
# 2.27 MB per turn). The 600-char pattern cap that fixed it was a length heuristic that
# also silently dropped genuine long wants; it is gone (#810). The render half of that
# incident is handled by retiring the block (Exec P408), not by this parser.
#
# WANT_MAX_CHARS is RENDER-ONLY now: render_wants() still clamps each line with it. It is
# no longer used by any parser path, and is deleted with the renderer it serves.
WANT_MAX_CHARS = 600
WANT_RENDER_LIMIT = 40
WANT_OPEN = "[WANT]"
WANT_CLOSE = "[/WANT]"
_WANT_MARKER_RE = re.compile(r"\[(/?)WANT\]")

# Why a marker was skipped (a failing marker is DISCUSSION, never a want).
WANT_SKIP_REASONS = (
    "in_fence", "in_code_span", "code_adjacent", "escaped", "quoted",
    "in_json_string", "in_url", "in_link_target",
    "opener_unclosed", "closer_without_opener", "empty_pair",
)

# Flood bounds for the INFO skip log (surface_wants runs on every autosave pulse). Operational
# tunables, so env-with-default like the other CC_* knobs in this file (LAW 5); a junk value
# falls back to the default rather than breaking import.
# Volume claim, stated exactly: while the corpus holds <= WANT_SKIP_SEEN_MAX distinct
# (node, offset, reason) skips, a static corpus costs ~1 INFO line per heartbeat interval
# (1/hour by default). Beyond WANT_SKIP_SEEN_MAX distinct skips the FIFO evicts entries that
# then re-qualify as unseen, so the steady state degrades to at most
# WANT_SKIP_DETAIL_PER_CALL_MAX detail lines per pulse (never unbounded).
def _want_skip_env_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


WANT_SKIP_SUMMARY_INTERVAL_S = _want_skip_env_int("CC_WANT_SKIP_SUMMARY_INTERVAL_S", 3600, 1)
WANT_SKIP_SEEN_MAX = _want_skip_env_int("CC_WANT_SKIP_SEEN_MAX", 4096, 1)
WANT_SKIP_DETAIL_PER_CALL_MAX = _want_skip_env_int("CC_WANT_SKIP_DETAIL_PER_CALL_MAX", 50, 1)

_WANT_FENCE_LINE_RE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})([^\r\n]*)\r?$", re.MULTILINE)
_WANT_BLANK_LINE_RE = re.compile(r"\r?\n[ \t]*\r?\n")
_WANT_BACKTICK_RUN_RE = re.compile(r"`+")
_WANT_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’", "«": "»"}


@dataclass(frozen=True)
class WantSpan:
    """One legitimate want: `text` is exactly what surface_wants stores and hashes."""
    text: str
    want_id: str
    open_start: int     # char offset of the real [WANT] in the parsed content
    close_end: int      # char offset just past the paired [/WANT]


@dataclass(frozen=True)
class SkippedMarker:
    """One marker that is discussion, not a want (`reason` is in WANT_SKIP_REASONS)."""
    marker: str         # WANT_OPEN or WANT_CLOSE
    start: int          # char offset in the parsed content
    reason: str


@dataclass(frozen=True)
class WantParse:
    wants: Tuple[WantSpan, ...]
    skipped: Tuple[SkippedMarker, ...]


def want_id_for_text(text: str) -> str:
    """THE want-node id: "cc:want::" + sha1(text utf-8)[:16]. `text` is the stripped inner
    text. The #801 repair tool and surface_wants both call this, so repaired ids equal
    what a re-parse mints."""
    import hashlib
    return "cc:want::" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _want_fence_spans(content: str) -> List[Tuple[int, int]]:
    """Fenced code blocks (CommonMark): ```/~~~ line, closed by >= as long a fence of the
    same char with only whitespace after. An UNCLOSED fence runs to the end of content --
    skipping a real want is logged and recoverable; minting a bogus one is permanent."""
    spans: List[Tuple[int, int]] = []
    opened: Optional[Tuple[int, str, int]] = None
    for m in _WANT_FENCE_LINE_RE.finditer(content):
        run, info = m.group(1), m.group(2)
        ch, n = run[0], len(run)
        if opened is None:
            if ch == "`" and "`" in info:
                continue            # ```inline``` on one line is a code span, not a fence
            opened = (m.start(), ch, n)
        elif ch == opened[1] and n >= opened[2] and not info.strip():
            spans.append((opened[0], m.end()))
            opened = None
    if opened is not None:
        spans.append((opened[0], len(content)))
    return spans


def _want_code_span_ranges(content: str, fences: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Inline code spans outside fences, per paragraph (never across a blank line or a
    fence). A backtick run of length n opens a span closed by the next run of exactly n;
    a run with no closer in its paragraph is literal. Linear in the number of runs."""
    free: List[Tuple[int, int]] = []
    pos = 0
    for f_start, f_end in fences:
        if f_start > pos:
            free.append((pos, f_start))
        pos = max(pos, f_end)
    if pos < len(content):
        free.append((pos, len(content)))
    spans: List[Tuple[int, int]] = []
    for seg_start, seg_end in free:
        paragraphs: List[Tuple[int, int]] = []
        p = seg_start
        for sep in _WANT_BLANK_LINE_RE.finditer(content, seg_start, seg_end):
            paragraphs.append((p, sep.start()))
            p = sep.end()
        paragraphs.append((p, seg_end))
        for para_start, para_end in paragraphs:
            runs = [(m.start(), m.end()) for m in
                    _WANT_BACKTICK_RUN_RE.finditer(content, para_start, para_end)]
            if len(runs) < 2:
                continue
            next_same: List[Optional[int]] = [None] * len(runs)
            last_by_len: Dict[int, int] = {}
            for i in range(len(runs) - 1, -1, -1):
                length = runs[i][1] - runs[i][0]
                next_same[i] = last_by_len.get(length)
                last_by_len[length] = i
            i = 0
            while i < len(runs):
                j = next_same[i]
                if j is None:
                    i += 1          # unmatched run: literal backticks, masks nothing
                    continue
                spans.append((runs[i][0], runs[j][1]))
                i = j + 1
    return spans


# --- #815: a want tag inside JSON, a URL or a link target is text being CARRIED or TALKED ABOUT ---
# JSON string literal opener context: `{` / `[` (first element / key), `key":` (value), or a
# completed JSON value then `,` (next element). Closer context: `, } ] :` or end of content.
# Deliberately NOT "any quoted text": prose like `He said "go [WANT]x[/WANT]", then left` has
# no such context and stays a real want.
_WANT_JSON_AFTER_RE = re.compile(r'\s*(?:[}\],:]|$)')
_WANT_URL_SCHEME_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://")
_WANT_URL_LOOKBACK = 2048
# in_url fires ONLY on genuine URL-internal glue (turn 3; the turn-2 blocklist of "terminators"
# dropped real wants after `url:` / bold / italic / dash / tilde / literal backslash-n).
#   `/`            the opener sits in a URL path: `https://x.org/a/[WANT]...`   -> in_url, always.
#   `? # = &`      query / fragment glue: in_url ONLY when the marker is INSIDE the URL token, i.e.
#                  URL characters CONTINUE right after the paired closing tag
#                  (`https://x.org/a?[WANT]s[/WANT]&p=1`); a want typed AFTER the URL
#                  (`https://x.org/a?[WANT]follow up[/WANT]` + end/space/punctuation) stays real.
#   anything else  (`: - – * _ ~ . , ) ] } > " ' ! ;` a letter, a literal backslash-n ...) stays real.
_WANT_URL_PATH_GLUE = "/"
_WANT_URL_QUERY_GLUE = frozenset("?#=&")
_WANT_URL_CONTINUE_RE = re.compile(r"[A-Za-z0-9/_~%&=+#@\-]")
_WANT_LINK_TEXT_LOOKBACK = 1024
_WANT_JSON_WORDS = ("true", "false", "null")
_WANT_JSON_NUMBER_CHARS = frozenset("0123456789.+-eE")
# Windows for the backward JSON-structure walk (turn 4). Fail-open toward minting: a container
# whose elements are longer than these is simply not recognised.
_WANT_JSON_STRING_WINDOW = 512
_WANT_JSON_CONTAINER_WINDOW = 256


def _want_json_string_start(content: str, end_quote: int) -> int:
    """Index of the opening `"` of the JSON string whose closing `"` is at `end_quote` (same
    line, bounded window, backslash-escaped quotes skipped), else -1."""
    lo = max(0, end_quote - _WANT_JSON_STRING_WINDOW)
    i = end_quote - 1
    while i >= lo:
        c = content[i]
        if c == "\r" or c == "\n":
            return -1
        if c == '"':
            slashes = 0
            while i - 1 - slashes >= lo and content[i - 1 - slashes] == "\\":
                slashes += 1
            if slashes % 2 == 0:
                return i
        i -= 1
    return -1


def _want_json_container_start(content: str, close_idx: int) -> int:
    """Index of the `[` / `{` matching the `]` / `}` at `close_idx` (bracket counting, bounded
    window), else -1."""
    lo = max(0, close_idx - _WANT_JSON_CONTAINER_WINDOW)
    depth = 0
    i = close_idx
    while i >= lo:
        c = content[i]
        if c == "]" or c == "}":
            depth += 1
        elif c == "[" or c == "{":
            depth -= 1
            if depth == 0:
                return i
        i -= 1
    return -1


def _want_json_value_start(content: str, j: int) -> int:
    """Start index of the complete JSON VALUE ending at content[j] -- a string, a number, a
    WHOLE-token true/false/null, or a balanced [...] / {...} -- else -1."""
    c = content[j]
    if c == '"':
        return _want_json_string_start(content, j)
    if c == "]" or c == "}":
        return _want_json_container_start(content, j)
    if c.isdigit():
        i = j
        while i > 0 and content[i - 1] in _WANT_JSON_NUMBER_CHARS:
            i -= 1
        if i > 0 and (content[i - 1].isalnum() or content[i - 1] == "_"):
            return -1
        return i
    for word in _WANT_JSON_WORDS:
        if content.endswith(word, 0, j + 1):
            i = j + 1 - len(word)
            if i == 0 or not (content[i - 1].isalnum() or content[i - 1] == "_"):
                return i
    return -1


def _want_json_prev_nonspace(content: str, i: int) -> int:
    while i >= 0 and content[i].isspace():
        i -= 1
    return i


def _want_json_element_in_container(content: str, comma_idx: int, memo: Optional[Dict[int, bool]]) -> bool:
    """True when the comma at `comma_idx` continues a JSON ARRAY or OBJECT: walking back over
    complete JSON values (and `"key":` members) separated by commas reaches an opening `{` / `[`.
    A bare prose word, number or quoted phrase before the comma (`it was 3,` `the answer is true,`
    `She said "a",`) never does (turn 4, le-019 F2). Memoised per comma, so a whole node is linear."""
    visited: List[int] = []
    pos = comma_idx
    result = False
    while True:
        if memo is not None and pos in memo:
            result = memo[pos]
            break
        visited.append(pos)
        j = _want_json_prev_nonspace(content, pos - 1)
        if j < 0:
            break
        s = _want_json_value_start(content, j)
        if s < 0:
            break
        k = _want_json_prev_nonspace(content, s - 1)
        if k < 0:
            break
        p = content[k]
        if p == "[" or p == "{":
            result = True
            break
        if p == ",":
            pos = k
            continue
        if p == ":":
            k2 = _want_json_prev_nonspace(content, k - 1)
            if k2 < 0 or content[k2] != '"':
                break
            ks = _want_json_string_start(content, k2)
            if ks < 0:
                break
            k3 = _want_json_prev_nonspace(content, ks - 1)
            if k3 < 0:
                break
            if content[k3] == "{":
                result = True
                break
            if content[k3] == ",":
                pos = k3
                continue
        break
    if memo is not None:
        for v in visited:
            memo[v] = result
    return result


def _want_json_opens_literal(content: str, q: int, memo: Optional[Dict[int, bool]] = None) -> bool:
    """True when the `"` at content[q] is in REAL JSON opener context: right after `{` / `[`,
    right after `"key":`, or after a comma that continues a JSON array/object
    (_want_json_element_in_container). A bare prose word or number + comma + quote is NOT JSON
    context. Bounded work per quote."""
    j = _want_json_prev_nonspace(content, q - 1)
    if j < 0:
        return False
    c = content[j]
    if c == "{" or c == "[":
        return True
    if c == ":":
        j = _want_json_prev_nonspace(content, j - 1)
        return j >= 0 and content[j] == '"'
    if c == ",":
        return _want_json_element_in_container(content, j, memo)
    return False


def _want_json_string_ranges(content: str) -> List[Tuple[int, int]]:
    """(start, end) of every structurally valid JSON string literal (quotes included): an
    unescaped `"` in JSON opener context, closed by the next unescaped `"` on the SAME line
    (JSON strings carry no raw newline), followed by JSON closer context. Linear."""
    ranges: List[Tuple[int, int]] = []
    memo: Dict[int, bool] = {}
    n = len(content)
    pos = 0
    while True:
        q = content.find('"', pos)
        if q < 0:
            break
        slashes = 0
        while q - 1 - slashes >= 0 and content[q - 1 - slashes] == "\\":
            slashes += 1
        if slashes % 2 == 1 or not _want_json_opens_literal(content, q, memo):
            pos = q + 1
            continue
        i = q + 1
        closed = False
        while i < n:
            c = content[i]
            if c == "\\":
                i += 2
                continue
            if c == '"':
                closed = True
                break
            if c in "\r\n":
                break
            i += 1
        if not closed or not _WANT_JSON_AFTER_RE.match(content, i + 1):
            pos = q + 1
            continue
        ranges.append((q, i + 1))
        pos = i + 1
    return ranges


def _want_glued_run_before(content: str, start: int) -> str:
    """The non-whitespace characters immediately before `start`, never crossing another WANT
    marker -- the token an opener is glued to (empty when whitespace precedes it)."""
    lo = max(0, start - _WANT_URL_LOOKBACK)
    i = start
    while i > lo and not content[i - 1].isspace():
        i -= 1
    run = content[i:start]
    last = None
    for last in _WANT_MARKER_RE.finditer(run):
        pass
    return run[last.end():] if last is not None else run


def _want_link_text_opens_a_link(content: str, close_bracket: int) -> bool:
    """True when the `]` at content[close_bracket] closes the text of a real markdown link: a
    balanced `[` is found before it (same paragraph, bounded lookback, backslash-escaped
    brackets ignored) and that `[` is not glued to an identifier / `)` / `]` -- `arr[0](...)`,
    `f(x)[0](...)`, `a[b][c](...)` are index / call shapes, not links (turn 3)."""
    lo = max(0, close_bracket - _WANT_LINK_TEXT_LOOKBACK)
    floor = lo
    for blank in _WANT_BLANK_LINE_RE.finditer(content, lo, close_bracket):
        floor = blank.end()
    depth = 1
    i = close_bracket - 1
    while i >= floor:
        c = content[i]
        if c == "]" or c == "[":
            slashes = 0
            while i - 1 - slashes >= floor and content[i - 1 - slashes] == "\\":
                slashes += 1
            if slashes % 2 == 0:
                if c == "]":
                    depth += 1
                else:
                    depth -= 1
                    if depth == 0:
                        return i == 0 or not (content[i - 1].isalnum() or content[i - 1] in "_)]")
        i -= 1
    return False


def _want_url_continues_after_pair(content: str, opener_end: int, closer_starts: List[int]) -> bool:
    """True when the WANT pair that starts after `opener_end` is INSIDE a URL token: URL
    characters continue immediately after the nearest closing tag. `closer_starts` is the sorted
    offset list of every `[/WANT]` in the node, computed once by parse_wants, so this is a
    bisect, not a scan (turn 4, le-019 F1: the old per-opener `find` was O(openers x node))."""
    i = bisect_left(closer_starts, opener_end)
    if i >= len(closer_starts):
        return False
    after = closer_starts[i] + len(WANT_CLOSE)
    return after < len(content) and _WANT_URL_CONTINUE_RE.match(content, after) is not None


def _want_opener_carried_reason(content: str, start: int, end: int, closer_starts: List[int]) -> Optional[str]:
    """#815, OPENER context only (a closer guard would swallow `[WANT]read https://x/a[/WANT]`,
    the le-014 C1 class): is this opener glued to a link destination or a URL?
      in_link_target  the glued run holds `](` with no `)` after it AND that `]` closes the text
                      of a real markdown link (`[` present, not an index/call shape);
      in_url          the glued run holds `scheme://` and the character right before the opener
                      is URL-internal glue (see _WANT_URL_PATH_GLUE / _WANT_URL_QUERY_GLUE)."""
    run = _want_glued_run_before(content, start)
    if not run:
        return None
    idx = run.rfind("](")
    if idx >= 0 and ")" not in run[idx + 2:] and _want_link_text_opens_a_link(content, start - len(run) + idx):
        return "in_link_target"         # [text](dest/[WANT]...  -- the opener sits in the destination
    if _WANT_URL_SCHEME_RE.search(run):
        prev = run[-1]
        if prev == _WANT_URL_PATH_GLUE:
            return "in_url"             # scheme://host/path/[WANT]...  -- in a URL path
        if prev in _WANT_URL_QUERY_GLUE and _want_url_continues_after_pair(content, end, closer_starts):
            return "in_url"             # scheme://host/a?[WANT]s[/WANT]&p=1  -- inside the token
    return None


def _want_marker_region(start: int,
                        fences: List[Tuple[int, int]], fence_starts: List[int],
                        codes: List[Tuple[int, int]], code_starts: List[int],
                        jsons: List[Tuple[int, int]], json_starts: List[int]
                        ) -> Optional[Tuple[str, Tuple[str, int]]]:
    """The structural region (JSON string literal, fenced block, inline code span) the marker at
    `start` lies in, as (reason, (kind, index)), or None. JSON is checked first so a fence
    carried inside a JSON string is rejected as JSON, not by the coincidence that its backtick
    runs pair up. The (kind, index) key identifies the region so parse_wants can pair a masked
    closer with a masked opener of the SAME region."""
    k = bisect_right(json_starts, start) - 1
    if k >= 0 and start < jsons[k][1]:
        return "in_json_string", ("j", k)
    k = bisect_right(fence_starts, start) - 1
    if k >= 0 and start < fences[k][1]:
        return "in_fence", ("f", k)
    k = bisect_right(code_starts, start) - 1
    if k >= 0 and start < codes[k][1]:
        return "in_code_span", ("c", k)
    return None


def _want_opener_mention_reason(content: str, start: int, end: int, closer_starts: List[int]) -> Optional[str]:
    """Why an OPENER outside every structural region is still a mention, or None. These are
    guesses about the context the OPENER sits in and are NEVER applied to a closer (le-014
    C1/C2, le-016 #3: base never guarded closers, and a closer judged by a lexical guess on its
    own text turns `[WANT]check `foo()`[/WANT]` / `[WANT]rename "a", "b[/WANT]", next` into a
    dropped want). A mentioned opener's orphan closer is skipped as closer_without_opener."""
    carried = _want_opener_carried_reason(content, start, end, closer_starts)
    if carried is not None:
        return carried
    if start >= 2 and content[start - 2:start] == '\\"' and content[end:end + 2] == '\\"':
        return "in_json_string"         # JSON-escaped text: \"[WANT]\"
    if start > 0 and content[start - 1] == "`":
        return "code_adjacent"      # the pre-#810 guard (OPENER only, as in base), for runs that cannot be paired
    n_slash = 0
    while start - 1 - n_slash >= 0 and content[start - 1 - n_slash] == "\\":
        n_slash += 1
    if n_slash % 2 == 1:
        return "escaped"
    if start > 0 and end < len(content):
        close_quote = _WANT_QUOTE_PAIRS.get(content[start - 1])
        if close_quote is not None and content[end] == close_quote:
            return "quoted"
    return None


def parse_wants(content: str) -> WantParse:
    """THE legitimacy test (#810): which [WANT]...[/WANT] pairs in `content` are real wants.

    PURE function of the string: no I/O, no logging, no graph. A want is the text between
    a real opener and its paired closer, with NO length limit. An OPENER inside a JSON string
    literal, a fenced block or an inline code span, glued to a link destination or URL, hugged
    by JSON-escaped quotes, directly after a backtick, backslash-escaped, or wrapped in a
    matching quote pair is a mention. A CLOSER is judged only relative to its OPENER: it is a
    mention when it closes a masked opener of the same JSON literal / fence / code span, when it
    closes a mention opener nested inside the pending want (nearest-opener rule), or when no
    live opener is pending; otherwise it is that want's closer whatever surrounds it (le-014
    C1/C2, le-016 #3, le-019 F3). JSON context is REAL JSON only: a bare prose word/number +
    comma + quote is not (le-019 F2). Skipped; the closer pairs with the
    NEAREST live opener, so a returned want contains no live marker; a stray closer, an
    unclosed opener and an empty pair are skipped. Every skip carries its reason and offset
    (the caller logs them -- never silent). `wants[i].text` is `.strip()`ped inner text,
    un-normalised otherwise, exactly what surface_wants stores and hashes.
    """
    if not content or "WANT]" not in content:
        return WantParse((), ())
    fences = _want_fence_spans(content)
    codes = sorted(_want_code_span_ranges(content, fences))
    jsons = _want_json_string_ranges(content)
    fence_starts = [s for s, _ in fences]
    code_starts = [s for s, _ in codes]
    json_starts = [s for s, _ in jsons]
    wants: List[WantSpan] = []
    skipped: List[SkippedMarker] = []
    pending: Optional[Tuple[int, int]] = None       # (start, end) of the live opener awaiting a closer
    masked_openers: Dict[Tuple[str, int], int] = {}  # region -> masked openers not yet closed
    mention_stack: List[str] = []                    # opener-only mention openers nested in the pending want
    markers = list(_WANT_MARKER_RE.finditer(content))
    closer_starts = [mk.start() for mk in markers if mk.group(1)]
    for m in markers:
        is_close = bool(m.group(1))
        marker = WANT_CLOSE if is_close else WANT_OPEN
        region = _want_marker_region(m.start(), fences, fence_starts, codes, code_starts,
                                     jsons, json_starts)
        if region is not None:
            reason, key = region
            if not is_close:
                masked_openers[key] = masked_openers.get(key, 0) + 1
                skipped.append(SkippedMarker(marker, m.start(), reason))
                continue
            if masked_openers.get(key, 0) > 0:      # closes a mention opener in the SAME region
                masked_openers[key] -= 1
                skipped.append(SkippedMarker(marker, m.start(), reason))
                continue
            if pending is None:                     # nothing it could close: a stray mention closer
                skipped.append(SkippedMarker(marker, m.start(), reason))
                continue
            # else: a closer in a region with a REAL opener pending and no mention opener in that
            # region is the closer of the real want -- judged relative to ITS opener, not by the
            # text around it (le-016 #3). Fall through to pairing.
        elif not is_close:
            reason = _want_opener_mention_reason(content, m.start(), m.end(), closer_starts)
            if reason is not None:
                if pending is not None:
                    # a mention opener INSIDE a real want: by the nearest-opener rule it pairs with
                    # the next closer, so a mention pair stays inside the real want (turn 4, F3)
                    mention_stack.append(reason)
                skipped.append(SkippedMarker(marker, m.start(), reason))
                continue
        if not is_close:
            if pending is not None:     # a nearer opener arrived: the earlier one never closed
                skipped.append(SkippedMarker(WANT_OPEN, pending[0], "opener_unclosed"))
            pending = (m.start(), m.end())
            mention_stack.clear()
            continue
        if pending is None:
            skipped.append(SkippedMarker(WANT_CLOSE, m.start(), "closer_without_opener"))
            continue
        if mention_stack:               # closes the nearest (mention) opener, not the real one
            skipped.append(SkippedMarker(WANT_CLOSE, m.start(), mention_stack.pop()))
            continue
        inner = content[pending[1]:m.start()].strip()
        if inner:
            wants.append(WantSpan(inner, want_id_for_text(inner), pending[0], m.end()))
        else:
            skipped.append(SkippedMarker(WANT_OPEN, pending[0], "empty_pair"))
            skipped.append(SkippedMarker(WANT_CLOSE, m.start(), "empty_pair"))
        pending = None
    if pending is not None:
        skipped.append(SkippedMarker(WANT_OPEN, pending[0], "opener_unclosed"))
    skipped.sort(key=lambda s: s.start)
    return WantParse(tuple(wants), tuple(skipped))


_WANT_SKIP_LOCK = threading.Lock()
_WANT_SKIP_SEEN: "OrderedDict[Tuple[str, int, str], None]" = OrderedDict()
_WANT_SKIP_STATE: Dict[str, Any] = {"last_key": None, "last_emit": None}


def _reset_want_skip_log_state() -> None:
    """Forget what has been logged (process start / tests)."""
    with _WANT_SKIP_LOCK:
        _WANT_SKIP_SEEN.clear()
        _WANT_SKIP_STATE["last_key"] = None
        _WANT_SKIP_STATE["last_emit"] = None


def _log_want_skips(events: List[Tuple[str, SkippedMarker]]) -> None:
    """INFO-log skipped markers, flood-bounded and never silent. `events` is every skipped
    marker found by ONE surface_wants call as (node_id, SkippedMarker). Summary line when
    the per-reason counts changed or WANT_SKIP_SUMMARY_INTERVAL_S elapsed; per-marker detail
    once per (node, offset, reason) (bounded FIFO, <= WANT_SKIP_DETAIL_PER_CALL_MAX per
    call, the rest deferred). Logs NO want body and NO surrounding text: a detail line carries
    only the node id, the offset, the literal marker token ("[WANT]" / "[/WANT]") as its
    kind, and the reason -- so a pasted secret can never reach the log."""
    try:
        with _WANT_SKIP_LOCK:
            counts = Counter(sk.reason for _, sk in events)
            key = tuple(sorted(counts.items()))
            now = time.monotonic()
            details: List[Tuple[str, SkippedMarker]] = []
            deferred = 0
            for nid, sk in events:
                seen_key = (nid, sk.start, sk.reason)
                if seen_key in _WANT_SKIP_SEEN:
                    continue
                if len(details) >= WANT_SKIP_DETAIL_PER_CALL_MAX:
                    deferred += 1
                    continue
                _WANT_SKIP_SEEN[seen_key] = None
                if len(_WANT_SKIP_SEEN) > WANT_SKIP_SEEN_MAX:
                    _WANT_SKIP_SEEN.popitem(last=False)
                details.append((nid, sk))
            last_emit = _WANT_SKIP_STATE["last_emit"]
            emit_summary = bool(events) and (
                key != _WANT_SKIP_STATE["last_key"]
                or last_emit is None
                or now - last_emit >= WANT_SKIP_SUMMARY_INTERVAL_S)
            _WANT_SKIP_STATE["last_key"] = key
            if emit_summary:
                _WANT_SKIP_STATE["last_emit"] = now
        if emit_summary:
            logger.info(
                "surface_wants: skipped %d marker(s) in %d node(s) as mentions, not wants (%s)%s",
                len(events), len({nid for nid, _ in events}),
                ", ".join("%s=%d" % (r, c) for r, c in sorted(counts.items())),
                "; %d more detail line(s) deferred to later pulses" % deferred if deferred else "")
        for nid, sk in details:
            logger.info("surface_wants: skipped %s node=%s offset=%d reason=%s",
                        sk.marker, nid, sk.start, sk.reason)
    except Exception as exc:  # noqa: BLE001 - logging must never break surfacing
        logger.debug("want skip log failed (non-fatal): %s", exc)


def surface_wants(graph: Any, vector_db: Any, provenance: str = "cc_authored") -> List[Dict[str, Any]]:
    """Materialize [WANT]...[/WANT] markers from conversational deposits into
    first-class want-nodes in the SNN topology. Idempotent (want id = hash of
    the text) -- safe to call repeatedly, e.g. on every autosave pulse.

    A want is a differentiated, stateful, surfaceable intention living in the
    substrate -- not text buried in a conversation node. Classification
    happens HERE at the bucket (LAW 7), never at deposit time. Returns the
    open want dicts.

    Which markers are real wants is decided by parse_wants() (#810) -- structural
    legitimacy, no length limit. Markers it rejects as mentions are logged at INFO
    (flood-bounded, after the graph lock is released), never dropped silently.
    """
    skip_events: List[Tuple[str, SkippedMarker]] = []
    with _cc_mutation_lock(graph):
        open_wants: List[Dict[str, Any]] = []
        if graph is None:
            return open_wants
        syn_attempted = syn_failed = 0
        syn_exc_types: List[str] = []
        create_attempted = create_failed = 0
        create_exc_types: List[str] = []
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
            if "WANT]" not in content:
                continue
            parsed = parse_wants(content)
            skip_events.extend((nid, sk) for sk in parsed.skipped)
            for want in parsed.wants:
                want_id = want.want_id
                if want_id in graph.nodes:
                    continue
                try:
                    create_attempted += 1
                    graph.create_node(node_id=want_id, metadata={
                        "kind": "want", "want_text": want.text, "want_state": "open",
                        "provenance": provenance, "source_node": nid,
                        "creation_mode": "conversational",
                    })
                except Exception as exc:  # noqa: BLE001
                    # #915: guarded PER WANT -- one raise no longer aborts the loop. Counted, class name only, never
                    # str(exc); no rollback and no edit of any existing want (H-1); retried next pulse.
                    create_failed += 1
                    if type(exc).__name__ not in create_exc_types:
                        create_exc_types.append(type(exc).__name__)
                    continue
                try:
                    syn_attempted += 1
                    graph.create_synapse(nid, want_id, weight=0.3)
                except Exception as syn_exc:  # noqa: BLE001
                    # #904 / P489: LOUD, not silent. The want stays (authored, H-1) and is still reported.
                    syn_failed += 1
                    if type(syn_exc).__name__ not in syn_exc_types:
                        syn_exc_types.append(type(syn_exc).__name__)
                open_wants.append({"id": want_id, "text": want.text,
                                    "provenance": provenance, "state": "open", "source": nid})
        _cc_report_want_create_failures("surface_wants", create_attempted, create_failed, create_exc_types)
        _cc_report_want_synapse_failures("surface_wants", syn_attempted, syn_failed, syn_exc_types)
    _log_want_skips(skip_events)
    return open_wants


def _cc_report(callback: Optional[Any], *args: Any) -> None:
    """Call an optional reporting callback; never raises. The organism only
    REPORTS a swallowed failure (the caller counts/logs -- LAW 4), and a bug in
    a reporter must not change what the pipeline returns. Same guard
    cc_assemble_recall already applies to on_monitor_error."""
    if callback is None:
        return
    try:
        callback(*args)
    except Exception:  # noqa: BLE001
        pass  # the error-reporting hook itself must never break the pipeline


def _cc_surfaced_item(item: Any, stream: str = '') -> Dict[str, Any]:
    """One on_surfaced entry from a CacheLine or a raw surfaced/recall dict."""
    if isinstance(item, dict):
        return {'stream': stream, 'node_id': item.get('node_id') or '',
                'score': float(item.get('score', 0.0) or 0.0), 'content': item.get('content', '') or ''}
    return {'stream': item.stream, 'node_id': item.node_id,
            'score': float(item.score), 'content': item.content}


def render_wants(graph: Any, provenance: Any = ("cc_authored", "cc_emergent"),
                 on_error: Optional[Any] = None) -> str:
    """Render CC's own open want-nodes as a '## What I Want' block, newest
    first -- read LIVE every call (not a snapshot), so a want noted this
    session shows up immediately. Returns "" if none exist (graceful).

    provenance accepts a single string or an iterable -- default covers both
    text-marker wants (surface_wants, "cc_authored") and substrate-native
    curiosity wants (generate_emergent_want, "cc_emergent") in one block.

    on_error: optional reporter, on_error(exc), called when rendering RAISED
    (the swallowed exception, after the existing debug log). The return is
    still "" -- which is also the legitimate "no wants" value -- so this is the
    only way a caller can tell the two apart. Guarded: a raising reporter does
    not change the return. Unset (default) = behaviour unchanged.
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
        _cc_report(on_error, exc)
        return ""


def render_constitutional_core(graph: Any, on_error: Optional[Any] = None) -> str:
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

    on_error: optional reporter, on_error(exc), called when rendering RAISED
    (the swallowed exception, after the existing debug log). The return is
    still "" -- which is also the legitimate "no constitutional nodes" value --
    so this is the only way a caller can tell the two apart. Guarded: a
    raising reporter does not change the return. Unset (default) = behaviour
    unchanged.
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
        _cc_report(on_error, exc)
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
            # #905 part A (Exec P488 C1): the want is BORN BOUND, in this same
            # lock block -- one synapse seed -> want (weight 0.3, source -> want:
            # the surface_wants pattern) per EXISTING, de-duplicated seed. A node
            # with no synapse and no hyperedge is what the whole-graph guards
            # count as blocking and what the orphan sweep reaps ('*_emergent' is
            # not identity-protected, by design), so an unbindable want is never
            # minted: if no seed binds, the node is rolled back below.
            unique_seeds = list(dict.fromkeys(seed_ids))
            bound = missing = failed = 0
            fail_class = ""
            for seed_id in unique_seeds:
                if seed_id == want_id or seed_id not in graph.nodes:
                    missing += 1
                    continue
                try:
                    graph.create_synapse(seed_id, want_id, weight=0.3)
                    bound += 1
                except Exception as exc:  # noqa: BLE001 - counted and logged below, never silent (P370)
                    failed += 1
                    fail_class = type(exc).__name__
            if bound == 0:
                rolled_back = True
                try:
                    graph.remove_node(want_id)
                except Exception as exc:  # noqa: BLE001
                    rolled_back = False
                    fail_class = type(exc).__name__
                # ONE record: fixed reason code, counts, exception CLASS only
                # (never str(exc), never the want text or any id).
                logger.log(
                    logging.WARNING if rolled_back else logging.ERROR,
                    "CC emergent want NOT materialized: reason=%s seeds=%d bound=0 "
                    "missing=%d failed=%d exc_class=%s rolled_back=%s",
                    "no_bindable_seed" if rolled_back else "rollback_failed",
                    len(unique_seeds), missing, failed, fail_class or "-", rolled_back)
                return None
            if missing or failed:
                logger.warning(
                    "CC emergent want partially bound: reason=seed_shortfall seeds=%d "
                    "bound=%d missing=%d failed=%d exc_class=%s",
                    len(unique_seeds), bound, missing, failed, fail_class or "-")
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
# [P563] CC_CONV_PROBATION_PERIOD, the line above, is GRADUATION-ONLY (probation_remaining, probation_total, the dampening ramp, the release at 0, #93). The
# fair-chance window's SIZE is not read here: the HOST (the laptop daemon) reads its own environment (NG_FAIR_CHANCE_WINDOW_STEPS) and hands it to the canonical
# registration, neuro_foundation.Graph.enable_fair_chance_window, so the two quantities can never be re-coupled by one variable.
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
        # P561 / P563: open (or, for an exact repeat, RE-open) this node's FAIR-CHANCE WINDOW through the CANONICAL helper
        # neuro_foundation.Graph.fair_chance_stamp: a no-op until the host has registered the window on its graph, for a creation_mode the
        # host excluded, and on a graph (or a test double) that predates the helper. Graduation (probation_remaining / probation_total / the
        # ramp) stays on its own fields, stamped above.
        _stamp = getattr(graph, "fair_chance_stamp", None)
        if _stamp is not None:
            _stamp(node)
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
                # #915: fixed reason code + the exception CLASS name only -- str(exc) / the node id can carry user words.
                logger.warning("CC recall insert failed: reason=recall_insert_failed exc_type=%s", type(exc).__name__)
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

    # What _cc_deposit_memory_node stamps on a node, besides every key of the call's own `meta` (its metadata.update(meta)):
    # the rollback restores EXACTLY these fields and nothing else (#904 round 2, F1-2). A test (F1-3) ties these two tuples to
    # the deposit's write-set by AST, so a new stamped field cannot be added to the deposit without failing it.
    _STAMPED_ATTRS = ("threshold", "intrinsic_excitability")
    _STAMPED_META_KEYS = ("probation_remaining", "probation_total", "novelty_dampening", "poincare_dir")
    _ABSENT = object()

    def __init__(self, graph, vector_db):
        self._graph = graph
        self._vector_db = vector_db
        # #904 write journal. `writes`: one (node_id, kind, node_was_new, vdb_fresh) per node write
        # this call attempted, appended BEFORE the write so a write that raises midway is still
        # journalled. `topology`: the ids of the synapses / hyperedges the bind created.
        self.writes = []
        self.topology = {"synapses": [], "hyperedges": []}
        # #904 fold-up (C4/F1): what a PRE-EXISTING node looked like before this call's deposit re-stamped it
        # (_cc_deposit_memory_node resets its threshold, excitability and probation metadata and re-writes its vdb
        # entry on EVERY deposit). Restored by rollback so a failed attempt leaves NO trace on it. `unrestorable`
        # counts snapshots that could not be taken (the rollback then reports a partial write REMAINS);
        # `vdb_unprovable` counts vdb probes that raised for a pre-existing node (entry kept, never deleted).
        self.restores = []
        self.unrestorable = 0
        self.vdb_unprovable = 0

    def deposit(self, node_id, embedding, content, meta, index_in_recall=True, kind="forest"):
        """_cc_deposit_memory_node, journalled. Same arguments, same effects, same raise."""
        with _cc_mutation_lock(self._graph):
            node_was_new = node_id not in self._graph.nodes
            vdb_fresh = False
            entry = None
            if index_in_recall:
                try:
                    entry = self._vector_db.get(node_id)
                    vdb_fresh = entry is None
                except Exception:
                    if node_was_new:
                        # A NEW node's recall entry cannot belong to a node that existed: whatever this call inserts
                        # is its own, so the rollback deletes it even though the probe could not say (#904 fold-up).
                        vdb_fresh = True
                    else:
                        # A pre-existing node: cannot prove this call creates the entry, never delete it (counted).
                        self.vdb_unprovable += 1
            snap = None
            if not node_was_new:
                snap = self._snapshot_existing(node_id, entry, meta)
            self.writes.append((node_id, kind, node_was_new, vdb_fresh))
            try:
                return _cc_deposit_memory_node(self._graph, self._vector_db, node_id, embedding,
                                                content, meta, index_in_recall=index_in_recall)
            finally:
                if snap is not None:
                    self._record_written(snap)

    def _snapshot_existing(self, node_id, vdb_entry, meta):
        """Snapshot, BEFORE the deposit re-stamps a PRE-EXISTING node, only the fields the deposit writes: the two stamped
        attributes and the stamped metadata keys plus every key of this call's `meta` (a key ABSENT before is remembered as
        absent, so the rollback deletes it), and the references of the node's vdb entry. Called under the mutation lock.
        Returns the snapshot (or None when it could not be taken: counted `unrestorable`)."""
        try:
            node = self._graph.nodes[node_id]
            absent = self._ABSENT
            keys = list(dict.fromkeys(list(meta) + list(self._STAMPED_META_KEYS)))
            snap = {
                "id": node_id, "node": node, "keys": keys,
                "attrs_before": {a: getattr(node, a) for a in self._STAMPED_ATTRS},
                "meta_before": {k: (node.metadata[k] if k in node.metadata else absent) for k in keys},
                "attrs_wrote": None, "meta_wrote": None,
                "vdb": ((vdb_entry["embedding"], vdb_entry["content"], vdb_entry["metadata"])
                        if vdb_entry else None),
            }
            self.restores.append(snap)
            return snap
        except Exception:
            self.unrestorable += 1
            return None

    def _record_written(self, snap):
        """After the deposit (also when it raised): the value of each field NOW, i.e. what THIS call wrote. The rollback
        restores a field only while it still holds that value, so a change another writer (the probation sweep) made in
        between is never erased."""
        try:
            node, absent = snap["node"], self._ABSENT
            snap["attrs_wrote"] = {a: getattr(node, a) for a in self._STAMPED_ATTRS}
            snap["meta_wrote"] = {k: (node.metadata[k] if k in node.metadata else absent) for k in snap["keys"]}
        except Exception:
            self.unrestorable += 1

    def counts(self):
        """(forest, trees, windows, synapses, hyperedges) this call wrote so far."""
        n = {"forest": 0, "tree": 0, "window": 0}
        for _nid, kind, _new, _fresh in self.writes:
            n[kind] = n.get(kind, 0) + 1
        return (n["forest"], n["tree"], n["window"],
                len(self.topology["synapses"]), len(self.topology["hyperedges"]))

    def rollback(self):
        """Undo exactly what this call wrote (#904): synapses, then the hyperedge(s), then the
        nodes it CREATED, then the vdb entries it inserted fresh. A node that existed before the
        call is never removed. Each step is isolated so one failure cannot stop the rest; returns
        True only if every step succeeded (False means a partial write REMAINS). Also restores what the deposit
        re-stamped on a PRE-EXISTING node (#904 fold-up)."""
        graph, vdb = self._graph, self._vector_db
        ok = True
        with _cc_mutation_lock(graph):
            for sid in reversed(self.topology["synapses"]):
                if sid is None:
                    continue
                try:
                    if sid in graph.synapses:
                        graph.remove_synapse(sid)
                except Exception:
                    ok = False
            for hid in reversed(self.topology["hyperedges"]):
                if hid is None:
                    continue
                try:
                    if hid in graph.hyperedges:
                        graph.remove_hyperedge(hid)
                except Exception:
                    ok = False
            for node_id, _kind, node_was_new, vdb_fresh in reversed(self.writes):
                if node_was_new:
                    try:
                        if node_id in graph.nodes:
                            graph.remove_node(node_id)
                    except Exception:
                        ok = False
                if vdb_fresh:
                    try:
                        vdb.delete(node_id)
                    except Exception:
                        ok = False
            # A pre-existing node's re-stamp is undone LAST-IN-FIRST-OUT (a node deposited twice in one call ends at its
            # original state), and ONLY what this call wrote: the stamped attributes and the metadata keys the deposit wrote,
            # each put back (a key absent before is deleted) while it STILL holds the value this call wrote; a field another
            # writer changed since (the probation sweep) is left alone, as is every key the deposit never wrote. Then the
            # node's vdb entry gets its original embedding / content / metadata references back.
            def same(a, b):
                try:
                    return a is b or (type(a) is type(b) and bool(a == b))
                except Exception:
                    return False
            for snap in reversed(self.restores):
                try:
                    node, absent = snap["node"], self._ABSENT
                    if graph.nodes.get(snap["id"]) is node and snap["meta_wrote"] is not None:
                        for a, before in snap["attrs_before"].items():
                            if same(getattr(node, a), snap["attrs_wrote"][a]):
                                setattr(node, a, before)
                        for k in snap["keys"]:
                            cur = node.metadata[k] if k in node.metadata else absent
                            if same(cur, snap["meta_wrote"][k]):
                                if snap["meta_before"][k] is absent:
                                    node.metadata.pop(k, None)
                                else:
                                    node.metadata[k] = snap["meta_before"][k]
                    if snap["vdb"] is not None:
                        emb, content, meta = snap["vdb"]
                        vdb.insert(id=snap["id"], embedding=emb, content=content, metadata=meta)
                        store = getattr(vdb, "embeddings", None)
                        if isinstance(store, dict):   # insert re-normalises: put the ORIGINAL array back, exactly
                            store[snap["id"]] = emb
                except Exception:
                    ok = False
            if self.unrestorable:
                ok = False
        return ok

    def record_outcome(self, embedding, target_id, success, strength=1.0, metadata=None):
        meta = dict(metadata or {})
        meta["cc"] = True
        if meta.get("_link"):
            return {"deposited": True}
        if meta.get("_tree_concept") and meta.get("_concept"):
            if not _cc_concept_passes_floor(meta["_concept"]):
                logger.debug("Tree concept below floor, not indexed: %r", meta["_concept"][:40])
                return {"deposited": False, "reason": "concept_below_floor"}
            self.deposit(target_id, embedding, meta["_concept"], meta,
                         index_in_recall=True, kind="tree")
        else:
            self.deposit(target_id, embedding, meta.get("_forest_content", ""), meta,
                         index_in_recall=True, kind="forest")
        return {"deposited": True}

    def record_outcome_broadcast(self, embedding, target_id, success, strength=1.0, metadata=None):
        return self.record_outcome(embedding, target_id, success, strength, metadata)


def _cc_bind_conversational_topology(graph, forest_id, result, forest_embedding, state, window_ids=None,
                                      journal=None):
    """Wire forest<->tree synapses, intra-turn window delay-chains (#257
    polychrony), a binding hyperedge, and a delayed prev->current forest
    link. `state` is a plain dict the caller owns (holds "last_forest_id")
    -- replaces canonical's module-level _last_conv_forest_id global, since
    each CC daemon needs its own, not one shared across Syl and CC.

    #904: nothing in here is swallowed any more. A failed synapse, hyperedge,
    delay-chain link or previous-forest link RAISES (and a forest that is absent
    when the bind runs raises too, instead of returning silently), so
    run_conversational_dual_pass can fail truthfully and roll the turn back. The
    real graph only raises for a missing node / a self-connection, both excluded
    below, so no success path changes. `journal` (default None, optional) is the
    dual pass's write journal: every synapse / hyperedge id this call creates is
    appended (to undo it), and the SITE being written is recorded in
    journal["site"] / journal["attempts"] (so the failure record can name which
    of the five sites failed and how many writes of it were attempted).

    Returns a small STATUS dict (what was written, per kind), so the caller can
    be truthful about a bind that wrote less than a turn with trees needs:
    trees (how many it was handed), tree_pairs / window_pairs (forest<->member synapse PAIRS
    written), hyperedge (bool), window_chain (links), sequence ("written" | "no_predecessor"), plus
    has_trees / has_windows. Existing nodes and synapses are never touched: this
    only ever ADDS this turn's topology.
    """
    status = {"trees": 0, "tree_pairs": 0, "window_pairs": 0, "hyperedge": False, "window_chain": 0,
              "sequence": "no_predecessor", "has_trees": False, "has_windows": False}

    def _site(name):
        if journal is not None:
            journal["site"] = name
            attempts = journal.setdefault("attempts", {})
            attempts[name] = attempts.get(name, 0) + 1

    def _syn(site, pre, post, weight, delay=None):
        _site(site)
        if delay is None:
            syn = graph.create_synapse(pre, post, weight=weight)
        else:
            syn = graph.create_synapse(pre, post, weight=weight, delay=delay)
        if journal is not None:
            journal["synapses"].append(getattr(syn, "synapse_id", None))

    with _cc_mutation_lock(graph):
        if forest_id not in graph.nodes:
            _site("forest_absent")
            raise RuntimeError("forest node absent at bind")
        tree_ids = [t for t in (result.get("tree_ids") or []) if t in graph.nodes and t != forest_id]
        window_ids = [w for w in (window_ids or []) if w in graph.nodes and w != forest_id]
        status["trees"] = len(tree_ids)
        status["has_trees"] = bool(tree_ids)
        status["has_windows"] = bool(window_ids)
        for tid in tree_ids:
            _syn("tree_synapse", forest_id, tid, 0.2)
            _syn("tree_synapse", tid, forest_id, 0.15)
            status["tree_pairs"] += 1
        for wid in window_ids:
            _syn("window_synapse", forest_id, wid, 0.2)
            _syn("window_synapse", wid, forest_id, 0.15)
            status["window_pairs"] += 1
        if tree_ids or window_ids:
            _site("hyperedge")
            he = graph.create_hyperedge(
                member_node_ids=set([forest_id] + tree_ids + window_ids),
                metadata={"creation_mode": "conversational", "cc": True},
            )
            if journal is not None:
                journal["hyperedges"].append(getattr(he, "hyperedge_id", None))
            status["hyperedge"] = True
        if len(window_ids) >= 2:
            import random as _rnd
            for i in range(len(window_ids) - 1):
                d = _rnd.randint(2, max(2, _CC_CONV_SYNAPSE_DELAY_MAX))
                _syn("window_chain", window_ids[i], window_ids[i + 1], 0.2, delay=d)
                status["window_chain"] += 1
        last_id = state.get("last_forest_id")
        if last_id and last_id in graph.nodes and last_id != forest_id:
            import random as _rnd
            d = _rnd.randint(2, max(2, _CC_CONV_SYNAPSE_DELAY_MAX))
            _syn("sequence_link", last_id, forest_id, 0.2, delay=d)
            status["sequence"] = "written"
        # #904 (iii): a forest born with NO predecessor, NO trees and NO windows has nothing binding it to anything
        # (#900's unbound_status counts it as NEW). ONE WARNING at birth: a fixed reason code and the three booleans,
        # never an id or text. Only for a forest this call CREATED (a repeat of an existing turn is not "born").
        if (not tree_ids and not window_ids and status["sequence"] != "written"
                and (journal is None or journal.get("forest_new", True))):
            logger.warning(
                "CC forest born UNBOUND: reason=forest_born_unbound has_predecessor=%s has_trees=%s has_windows=%s "
                "-- nothing links this forest to the rest of the graph",
                False, False, False)
        state["last_forest_id"] = forest_id
        # Anticipatory pre-activation (#256 port): this turn's forest+trees are
        # CC's "just fired" set — prime their synaptic neighborhood for the next
        # recall. state carries primed_nodes to the daemons' _recall(). (#358)
        cc_anticipate(graph, [forest_id] + tree_ids, state)
        return status


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


# ------------------------------------------------------------------------------------------------------
# THE HOST'S SHARE of the fair-chance window (Josh's ruling Exec P550 / P552; Exec P561; Exec P563; CC-CALLOSUM-TRUTH §8.13).
# FRAMING (Josh): the fair-chance window is SHARED MACHINERY being TESTED FIRST on the CC, not CC-specific code: the pioneer
# implementation of canonical §8.13 arrival protection; rollout to other NeuroGraphs (Syl's) is Josh's call, LAW 8 gate per host.
# The window logic itself (the step-keyed counters, the window test, the completion heartbeat) is CANONICAL and lives in
# neuro_foundation.Graph (enable_fair_chance_window / fair_chance_stamp / fair_chance_advance / fair_chance_heartbeat_stamp). This module keeps
# only what is the HOST'S: which nodes its advancer advances (below) and the calls its deposit and its advancer make into the canonical
# helpers. The registration (the switch) is the daemon's, made once its probation advance runs on its own autonomic clock (LAW 8).
# ------------------------------------------------------------------------------------------------------

# creation_mode values whose probation THIS host's advancer does NOT advance: the Ingestor's own sweep (universal_ingestor.py
# NodeRegistrar.update_probation) owns them, and on the laptop it runs only inside on_message, never on the autonomic pulse. The ONE definition:
# probation_population below reads it, and the daemon hands the same tuple to the canonical registration as the excluded modes.
PROBATION_UNADVANCED_CREATION_MODES = ("ingested",)


def probation_population(node) -> bool:
    """True iff cc_update_probation advances this node's graduation window (the POPULATION). The ONE definition (LAW 4):
    cc_update_probation's own skip consults it. Deliberately an EXCLUSION, not `== "conversational"`: nodes with no creation_mode (older
    checkpoints, seeds) are still advanced here. Byte-identical to the round-1 skip (`creation_mode != "ingested"`), including that a truthy
    non-dict metadata raises (a pre-existing shape the advancer's differential pins)."""
    return (node.metadata or {}).get("creation_mode") not in PROBATION_UNADVANCED_CREATION_MODES


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

    [P561 / P563] Graduation, the dampening ramp, the release at 0, `probation_remaining` and
    `probation_total` are BYTE-IDENTICAL to before: they stay on the per-pulse
    `probation_remaining` count, because step-keying THAT count would stall the ramp and
    graduation whenever the clock is held. The fair-chance window the orphan sweep honours
    is counted in graph STEPS on SEPARATE node fields by the CANONICAL helpers (Graph.fair_chance_advance,
    called once per node below; a no-op on an unregistered graph), and a COMPLETION heartbeat is
    stamped through Graph.fair_chance_heartbeat_stamp at the end of a non-raising pass. No window logic
    lives in this module (LAW 4).
    """
    with _cc_mutation_lock(graph):
        graduated = []
        base_threshold = graph.config.get("default_threshold", 1.0)
        # P563: the canonical helpers, resolved ONCE per pass; absent on a graph that predates them (then every call below is skipped and
        # this advancer is exactly the round-1 one).
        _advance = getattr(graph, "fair_chance_advance", None)
        _heartbeat = getattr(graph, "fair_chance_heartbeat_stamp", None)
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
            #
            # [P552/P563] The exclusion is probation_population(node): the ONE population
            # definition. It reads nothing about the fair-chance window or its heartbeat: the
            # advancer must never skip itself because its own heartbeat went stale (that would
            # be a deadlock).
            if not probation_population(node):
                continue
            # P561 / P563: the fair-chance window, advanced through the CANONICAL helper BEFORE any `continue` below so it is seeded /
            # advanced for every population node (even one whose old count is already 0 or absent). Never raises, touches only the two
            # window fields, a no-op when unregistered.
            if _advance is not None:
                _advance(node)
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
        # P561 P2 / P563: the COMPLETION heartbeat, through the canonical helper. Reached ONLY if the whole loop above ran without raising (a
        # raise anywhere, or this function never being called, leaves the old stamp: that staleness IS the signal).
        if _heartbeat is not None:
            _heartbeat()
        return graduated


def run_conversational_dual_pass(graph, vector_db, text: str, embedding, state: dict) -> bool:
    """Core dual-pass on one turn's text. Returns True on success, False on
    failure -- caller decides retry policy (this function does not enqueue).
    Mirrors canonical's _run_conversational_dual_pass, parameterized, EXCEPT that
    this one is atomic (below): the canonical (neurograph_rpc.py, Syl's) has no
    rollback and was not touched by #904.
    Every turn deposits raw (LAW 7) -- no redundancy check, dedup, or
    threshold runs at deposit; `target_id` is content-hashed (see below), so
    an exact-repeat turn's deposit naturally lands on the same node instead
    of creating a duplicate.

    ATOMIC (#904): the turn is written in separate lock acquisitions (forest,
    each tree, each window, then the bind). If anything raises AFTER the first
    write, everything this call wrote is rolled back (see
    _CCConversationalDualPassEco.rollback), state["last_forest_id"] /
    state["primed_nodes"] are restored, ONE WARNING is logged (ERROR if the
    rollback itself failed and a partial write REMAINS) with hardcoded text,
    the stage, the counts written and the exception CLASS NAME only (never
    str(exc), a node id or text), and False is returned: a retry then lands on
    a clean graph, so it equals one clean pass. False is never returned for a
    success and True is never returned after a partial write. A failure before
    the first write (nothing to undo) logs at debug, as before, and returns False.
    """
    if graph is None or embedding is None:
        return False
    eco = None
    stage = "setup"
    _absent = object()
    prior_last = prior_primed = _absent
    try:
        prior_last = state.get("last_forest_id", _absent)
        prior_primed = state.get("primed_nodes", _absent)
        from ng_embed import NGEmbed
        import hashlib
        target_id = "cc:conv::" + hashlib.sha1(text.encode()).hexdigest()
        embedder = NGEmbed.get_instance()
        windows = ()
        try:
            windows = embedder.embed_windows(text).windows  # () on short turns
        except Exception as win_exc:  # noqa: BLE001 — windows are extra topology
            logger.debug("CC embed_windows failed (non-fatal, no window nodes): %s", win_exc)
        meta = {"source": "cc_gateway", "creation_mode": "conversational",
                "_forest_content": text}
        eco = _CCConversationalDualPassEco(graph, vector_db)
        stage = "dual_record"
        _result = embedder.dual_record_outcome(
            ecosystem=eco,
            content=text,
            embedding=embedding,
            target_id=target_id,
            success=True,
            strength=1.0,
            metadata=meta,
        )
        window_ids = []
        stage = "windows"
        if windows:
            for i, w in enumerate(windows):
                wid = f"{target_id}::window::{i}"
                eco.deposit(
                    wid, w.embedding, w.text,
                    {**meta, "_window": True, "_window_index": i, "_forest_id": target_id},
                    index_in_recall=False, kind="window",
                )
                window_ids.append(wid)
        # dual_record_outcome raises DualPassIncompleteError on pass-2 failure
        # (R3 atomicity: no forest-only deposit). A return means forest+pass-2
        # completed (legitimate empty concepts produce extraction_failed=False).
        stage = "bind"
        eco.topology["forest_new"] = any(kind == "forest" and new for _nid, kind, new, _f in eco.writes)
        _bound = _cc_bind_conversational_topology(
            graph, target_id, _result or {}, embedding, state, window_ids=window_ids,
            journal=eco.topology,
        )
        # TRUTH RULE (#904 / Exec P491 (ii)): a turn that HAS trees must come out with EVERY forest<->tree synapse pair
        # AND its binding hyperedge. The bind raises on any failed write, so this can only trip if a write returned
        # without writing (or a swallow is ever restored): then the turn is NOT bound and True would be a lie. It is
        # all-or-nothing on purpose: with the rollback, a partial bind is never kept, so a retry equals one clean pass.
        if _bound and _bound.get("trees") and not (
                _bound.get("tree_pairs", 0) >= _bound["trees"] and _bound.get("hyperedge")):
            eco.topology["site"] = "bind_postcondition"
            raise RuntimeError("bind wrote less than a turn with trees needs")
        return True
    except Exception as exc:
        if eco is None or not eco.writes:
            # Nothing was written (e.g. pass-2 extraction failed before any deposit): nothing to undo.
            logger.debug("CC conversational dual-pass failed (non-fatal): %s", exc)
            return False
        n_forest, n_tree, n_window, n_syn, n_he = eco.counts()
        site = eco.topology.get("site") if stage == "bind" else None
        site_attempted = (eco.topology.get("attempts") or {}).get(site, 0) if site else 0
        try:
            rolled_back = eco.rollback()
        except Exception:  # noqa: BLE001 — the rollback must not mask the failure report
            rolled_back = False
        try:
            if prior_last is _absent:
                state.pop("last_forest_id", None)
            else:
                state["last_forest_id"] = prior_last
            if prior_primed is _absent:
                state.pop("primed_nodes", None)
            else:
                state["primed_nodes"] = prior_primed
            if site:   # COUNTED (caller-owned, in-process): failures per bind site, survives the rollback
                _failures = state.setdefault("bind_site_failures", {})
                _failures[site] = _failures.get(site, 0) + 1
        except Exception:  # noqa: BLE001
            rolled_back = False
        (logger.warning if rolled_back else logger.error)(
            "CC conversational dual-pass FAILED after the first write: stage=%s site=%s site_attempted=%d site_failed=%d "
            "exc_type=%s written forest=%d trees=%d windows=%d synapses=%d hyperedges=%d vdb_kept_unprovable=%d; "
            "unrestorable=%d; %s; the turn is NOT bound",
            stage, site or "-", site_attempted, 1 if site else 0, type(exc).__name__,
            n_forest, n_tree, n_window, n_syn, n_he, eco.vdb_unprovable, eco.unrestorable,
            ("rolled back, nothing of this turn remains" if not eco.vdb_unprovable
             else "rolled back; kept: %d pre-existing vdb entr%s that could not be proven (re-written by this call, not restored)"
             % (eco.vdb_unprovable, "y" if eco.vdb_unprovable == 1 else "ies")) if rolled_back
            else "ROLLBACK FAILED, a partial write REMAINS",
        )
        return False


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

    Doors (Chief B3 ruling 001; the drains were removed from the door set by
    P240(3)): the hook door -- the Stop-side _deposit(step=True)
    (cc_ng_host.py) -- is the only caller, and calls this once per turn even
    when the dual pass failed -- a failed turn is still a timestep. The
    prompt-side, pith-failure and PostToolUse deposits never call it; the two
    drains (drain_ingest_tract, Leg-1 drain_gateway_conduit) never step at all.

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


def _cc_drain_size_cap(batch_nodes) -> int:
    """Normalise drain_ingest_tract's `batch_nodes`: a positive int enables node-count pacing; None, <= 0 or
    anything non-integer means unpaced (0). The caller owns the loud warning for a bad value (the daemon names
    both env variables once); this stays a pure normaliser."""
    try:
        n = int(batch_nodes) if batch_nodes is not None else 0
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _cc_drain_receipt_write(receipt, **fields) -> None:
    """Guarded write of drain_ingest_tract's out-parameter receipt (756b-reporter style). Replaces the receipt's
    contents wholesale so a caller can never read a stale one. A receipt that raises changes NOTHING about the
    drain: the failure is logged with a hardcoded message and the exception CLASS NAME only -- never str(exc),
    never entry text."""
    if receipt is None:
        return
    try:
        receipt.clear()
        receipt.update(fields)
    except Exception as exc:  # noqa: BLE001 -- a reporter must never break the drain
        logger.warning("CC drain receipt write failed (ignored, drain unchanged): %s", type(exc).__name__)


def drain_ingest_tract(graph, vector_db, state: dict, tract_path: str = None,
                        return_consumed: bool = False, max_entries: int = 0,
                        batch_nodes: int = None, receipt: dict = None,
                        max_seconds: float = 0, retry_tract_path: str = None,
                        defer_tract_path: str = None, defer_over_bytes: int = 0,
                        hold_on_failure: bool = False):
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

    hold_on_failure=False (default) is byte-identical to the historical
    behaviour: every entry the reader yields is consumed whether or not its
    absorb succeeded, so an entry whose dual pass returned False or raised is
    truncated out of the file and lost (#794). hold_on_failure=True (opt-in;
    #794, Chief-003 ruling / Exec P386) advances the offset ONLY past entries
    that were absorbed (True) or legitimately filter-skipped (wrong type, wrong
    source, empty text -- looked at, nothing to absorb). At the FIRST entry
    whose absorb returns False or raises, the loop stops; only the prefix before
    that entry is truncated; that entry and everything after it stay in the file
    for the next call (a held entry is retried every cycle); ONE logger.warning
    is emitted (fixed reason code + exception CLASS NAME only); and the function
    RETURNS NORMALLY -- it never raises. If nothing precedes the held entry the
    file is not rewritten at all. max_entries still counts entries ATTEMPTED (a
    held entry was attempted). return_consumed still reports exactly the bytes
    this call removed.

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

    Locking: the caller holds graph._concurrent_lock (punchlist #643, the
    autosave-loop caller) for the whole call -- that lock is what makes the
    dual pass's mutation of the graph safe. UNCHANGED by D24: this function
    acquires and releases nothing, never steps and never consolidates.

    batch_nodes (D24, Exec P471/P476; default None = unpaced, byte-identical to
    the previous behaviour): a positive int paces the drain by NODES, the
    FatherGraph 25/250 rule. WHOLE turns are taken until the nodes created in
    this call (the len(graph.nodes) delta around each atomic dual pass) reach
    it; the turn that crosses it is absorbed whole and the batch ends, and a
    single turn alone over the size is ONE batch. The remainder stays in the
    file through the same partial truncate max_entries uses. Both caps may be
    set; whichever is reached first ends the call.

    receipt (D24; default None): an OUT-PARAMETER dict the caller reads after
    the call -- chosen over changing the return so the default return (int /
    (int, bytes)) stays byte-identical. Replaced wholesale on every call:
      nodes_created  int  -- len(graph.nodes) delta over the call
      ended_on_size  bool -- the batch ended on the batch_nodes rule
      arrivals       set  -- ids of the nodes this call landed (ids after
                             minus ids before, taken once per call); the
                             caller's guard (since the D24 FOLD, #896)
                             covers the WHOLE graph; these are the subset
                             that names this batch
      turns_taken    int
      reason         str  -- hardcoded: size_reached | entries_cap_reached |
                             tract_exhausted | parse_failed | no_batch
    Writes are guarded (a raising receipt changes nothing; class name only).
    The idle steps are NOT run here -- see the 2026-10-01 changelog entry:
    the caller releases its lock, then consolidates.

    Fails soft -- an ingest-tract drain failure must never break the
    daemon's autosave pulse.

    retry_tract_path (MVP 2026-10-03, Josh): with hold_on_failure, a failed
    entry's exact bytes are APPENDED to this tract and the pass moves on,
    instead of holding everything behind it. Nothing is dropped: the turn waits
    there whole. If the append itself fails, the pass holds exactly as before.
    Passing tract_path itself rotates a failed retry to the end of its own
    tract. None (default) = unchanged.

    defer_tract_path + defer_over_bytes (MVP 2026-10-03, overnight): a turn whose
    text is longer than defer_over_bytes is moved whole, unattempted, to this tract
    (not drained automatically) -- one atomic dual pass over a 120 KB turn ran ~40
    sequential extractions and held the pass (and every save) for 40+ minutes.
    Scheduling by size only; nothing dropped, nothing labelled. Unset = unchanged.
    """
    def _ret(absorbed_n: int, consumed: bytes = b""):
        return (absorbed_n, consumed) if return_consumed else absorbed_n

    size_cap = _cc_drain_size_cap(batch_nodes)
    # A fresh, empty-batch receipt BEFORE any early return, so a caller never reads a stale one.
    _cc_drain_receipt_write(receipt, nodes_created=0, ended_on_size=False, arrivals=set(),
                            turns_taken=0, reason="no_batch")
    ids_before = None   # set only when pacing or a receipt asked us to track arrivals
    ended = "tract_exhausted"

    def _fill_receipt(reason: str, taken_n: int) -> None:
        if receipt is None:
            return
        try:
            arrivals = (set(graph.nodes) - ids_before) if ids_before is not None else set()
        except Exception as exc:  # noqa: BLE001
            logger.warning("CC drain receipt arrivals unavailable (receipt left empty): %s", type(exc).__name__)
            arrivals = set()
        _cc_drain_receipt_write(receipt, nodes_created=len(arrivals), ended_on_size=(reason == "size_reached"),
                                arrivals=arrivals, turns_taken=taken_n, reason=reason)

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
    # hold_on_failure only (#794): offset just past the last entry that was
    # absorbed or legitimately filter-skipped. Unused (and never read) when the
    # flag is off, so the default path is unchanged.
    safe_offset = 0
    try:
        if size_cap or receipt is not None:
            ids_before = set(graph.nodes)
        _t_start = time.monotonic()
        reader = ng_tract.TractReader(data)
        for entry in reader:
            entry_start = consumed_offset
            # position() is a bound method on the Rust binding, not a property.
            # Read it BEFORE the filters so `continue` still consumes the entry.
            consumed_offset = reader.position()
            # Check entry type using ng_tract.ENTRY_EXPERIENCE (the real module constant)
            if entry.entry_type != ng_tract.ENTRY_EXPERIENCE:
                safe_offset = consumed_offset
                continue
            if entry.source != "cc_gateway":
                safe_offset = consumed_offset
                continue
            text = entry.content
            if not text or not text.strip():
                safe_offset = consumed_offset
                continue
            if defer_tract_path and defer_over_bytes and len(text) > defer_over_bytes:
                try:
                    with open(defer_tract_path, "ab") as dfh:
                        dfh.write(data[entry_start:consumed_offset])
                    safe_offset = consumed_offset
                    logger.info("CC ingest-tract defer: a %d-char turn moved whole to the deferred tract", len(text))
                    continue
                except Exception as dexc:  # noqa: BLE001 - attempt it normally instead
                    logger.warning("CC ingest-tract defer append failed (%s); attempting the turn", type(dexc).__name__)
            taken += 1
            hold_reason = None
            hold_exc_type = "-"
            try:
                if _apply_gateway_experience(graph, vector_db, state, entry):
                    absorbed += 1
                    safe_offset = consumed_offset
                else:
                    hold_reason = "absorb_returned_false"
            except Exception as exc:
                logger.debug("CC ingest-tract entry failed (non-fatal): %s", exc)
                hold_reason = "absorb_raised"
                hold_exc_type = type(exc).__name__
            if hold_on_failure and hold_reason is not None and retry_tract_path:
                try:
                    with open(retry_tract_path, "ab") as rf:
                        rf.write(data[entry_start:consumed_offset])
                    safe_offset = consumed_offset
                    logger.warning(
                        "CC ingest-tract retry: reason=%s exc_type=%s -- failed entry moved whole "
                        "to the retry tract; the pass continues", hold_reason, hold_exc_type)
                    hold_reason = None
                except Exception as rexc:  # noqa: BLE001 - falls through to the hold below
                    logger.warning("CC ingest-tract retry append failed (%s); holding instead",
                                   type(rexc).__name__)
            if hold_on_failure and hold_reason is not None:
                # #794: stop at the FIRST failed entry. Hardcoded text only: a
                # fixed reason code and the exception CLASS NAME -- never
                # str(exc), the entry text, a path or a secret.
                logger.warning(
                    "CC ingest-tract hold: reason=%s exc_type=%s -- failed entry "
                    "and everything after it kept in the tract for the next cycle",
                    hold_reason, hold_exc_type)
                break
            # D24: the node-count rule is checked AFTER the atomic dual pass returned, so the turn that crosses
            # the size is absorbed whole and only then does the batch end. Unset (0) -> never true.
            if size_cap and (len(graph.nodes) - len(ids_before)) >= size_cap:
                ended = "size_reached"
                break
            if max_entries and taken >= max_entries:
                ended = "entries_cap_reached"
                break
            if max_seconds and (time.monotonic() - _t_start) >= max_seconds:
                ended = "time_cap_reached"   # MVP 2026-10-03: bound one drain pass; the rest stays in the tract
                break
        if hold_on_failure:
            # Truncate only what precedes the first failed entry (or the whole
            # walked span when nothing failed: safe_offset == consumed_offset).
            consumed_offset = safe_offset
    except Exception as exc:
        # Parse failure -- truncate below never runs, so nothing was actually
        # consumed from the file. Elevated to warning (was debug): silent at
        # the default level, this is exactly how a laptop/VPS ng_tract format
        # skew would look -- every file failing the same way, invisibly.
        logger.warning("CC ingest-tract parse failed (non-fatal, file untouched): %s", exc)
        _fill_receipt("parse_failed", taken)
        return _ret(absorbed)  # consumed=b"" -- nothing was truncated

    _fill_receipt(ended, taken)
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


def _cc_callosum_consolidate(graph, idle_steps: int, *, guard=None, progress=None) -> bool:
    """FatherGraph Finding 3 sleep consolidation: run idle_steps of pure
    graph.step() with NO new input, so homeostatic regulation (threshold
    adaptation, synaptic scaling, excitability) can catch up before the next
    batch of foreign topology arrives. Measured 47%->74% accuracy in the
    FatherGraph training; the report calls it "not optional -- it's what
    makes merge work". Mirrors _handle_import (cc_ng_host.py) and
    import_trickle (cc-ng-sync.py), which already do exactly this.
    Returns True if the steps ran. Fails soft.

    PER-SLICE GUARD (#905 part D, Exec P490 -- both callers, the Leg 2 merge
    and the daemon's drain, get it from here, at the source):
      guard    zero-argument callable returning an iterable/set of the node ids
               that currently BLOCK the clock (empty = clear). Build it with
               cc_topology_merge.whole_graph_guard -- there is no predicate in
               this function. What "blocks" is defined ONCE, in cc_topology_merge:
               `held_unbound_nodes` defines the HOLD (#905-DELTA: the sweep-eligible
               unbound nodes whose binding is IN TRANSIT; with the in-transit variable
               unset or unusable, every sweep-eligible unbound node), over
               `_unbound_nodes`, which defines the sweep-eligible unbound BASE and so
               what "bound" means: bound = NOT sweep-eligible (Exec P493 R1), NOT "a
               complete turn"; turn completeness is not a gate condition and nothing
               here tests it. It is called BEFORE EACH slice (including the
               first), WITHOUT holding _concurrent_lock (it takes _step_lock only;
               the established order is _concurrent_lock -> _step_lock). Ids are
               used for COUNTING only and are NEVER logged. A non-empty result
               STOPS the pass at the slice boundary: ONE logger.error (fixed
               reason code, steps done/remaining, COUNT of blockers), progress
               held=True, return False ("held, not done"). A raising guard fails
               CLOSED (stop; same record with the exception CLASS name only;
               held=True, failed=True). A node that became unbound between
               slices is therefore never aged through the rest of the pass.
      progress a dict the CALLER owns; filled on every return path:
               {"done": steps run, "remaining": idle_steps - done,
                "held": bool, "failed": bool} (success: held/failed False,
               remaining 0). A step that raises: held False, failed True.
    guard=None / progress=None (the positional (graph, idle_steps) call) behaves
    exactly as before."""
    prog = progress if progress is not None else {}

    def _fill(done_steps: int, held: bool = False, failed: bool = False) -> None:
        prog.update(done=done_steps, remaining=max(0, idle_steps - done_steps),
                    held=held, failed=failed)

    if idle_steps <= 0 or graph is None:
        _fill(0)
        return False
    # Take the lock in SLICES, not for the whole 250 steps. cc_ng_host.py's
    # changelog records real hook timeouts caused by _recall() blocking on a
    # long _concurrent_lock hold ("_concurrent_lock in _recall() caused hook
    # timeouts (Tonic holds lock)"). Consolidation is exactly that shape --
    # hundreds of graph.step() calls -- so it yields between slices, letting
    # a waiting recall/deposit interleave. Homeostasis does not care whether
    # the steps were contiguous; the hooks care a great deal.
    slice_n = max(1, int(os.environ.get("CC_CALLOSUM_LOCK_SLICE_STEPS", "25")))
    done = 0
    try:
        lock = getattr(graph, "_concurrent_lock", None)
        while done < idle_steps:
            if guard is not None:
                # Outside every lock this function takes: the guard needs
                # _step_lock, and nothing may be held across a slice that it needs.
                try:
                    blockers = len(tuple(guard()))
                except Exception as exc:  # noqa: BLE001 - fail CLOSED, loud, class only
                    logger.error(
                        "CC callosum consolidation HELD at a slice boundary: "
                        "reason=guard_raised exc_class=%s steps_done=%d steps_remaining=%d. "
                        "The clock is not advanced further (fail closed); no node ids "
                        "are logged.", type(exc).__name__, done, idle_steps - done)
                    _fill(done, held=True, failed=True)
                    return False
                if blockers:
                    logger.error(
                        "CC callosum consolidation HELD at a slice boundary: "
                        "reason=unbound_nodes_present blocking_nodes=%d steps_done=%d "
                        "steps_remaining=%d. The clock is not advanced further: orphan "
                        "grace is denominated in it and a node that became unbound "
                        "between slices must not age through the rest of the pass "
                        "(#905); no node ids are logged.",
                        blockers, done, idle_steps - done)
                    _fill(done, held=True)
                    return False
            n = min(slice_n, idle_steps - done)
            if lock is not None:
                with lock:
                    for _ in range(n):
                        graph.step()
            else:
                for _ in range(n):
                    graph.step()
            done += n
        _fill(done)
        return True
    except Exception as exc:
        # D24 / Exec P476(e) (P370: fixed where the drain's phase 2 reuses it): this used to be logger.debug --
        # the silent class (plan-002 7A item 5c): a failed consolidation looked identical to a skipped one.
        # Loud now, hardcoded message, the exception CLASS NAME only (never str(exc)). The RETURN VALUE (False)
        # and every other behaviour are unchanged; the merge and Syl's process share this function.
        logger.error("CC callosum consolidation FAILED after %d of %d step(s) (%s); "
                     "this is the CAUSE record -- the caller logs what it did about it",
                     done, idle_steps, type(exc).__name__)
        _fill(done, failed=True)
        return False


def drain_gateway_conduit(graph, vector_db, state: dict, conduit_dir: str = None,
                           batch_size: int = None, idle_steps: int = None,
                           load_ceiling: float = None, exclude_prefix: str = None,
                           *, save_callback=None, journal_path=None) -> dict:
    """Receive immutable raw Leg1 input, acknowledge only a complete save receipt.

    SQLite stores transport identities, exact raw BTF bytes and attempt states,
    never embeddings or derived cognition. FULL synchronous transactions precede
    every mutation. A filesystem lock serializes deliveries, while graph locking
    remains one record (or save) at a time. No synthetic graph steps.

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
                                    preserve_graph_config: bool = False,
                                    on_error: Optional[Any] = None,
                                    whole_content: bool = False) -> List[Dict[str, Any]]:
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

    on_error: optional reporter, on_error(exc), called with the swallowed
    exception (after the existing debug log) just before the fail-soft [] is
    returned, so a caller can tell "Active Recall failed" from "nothing
    matched" (both are []). Not called for the legitimate empty returns (empty
    query / no graph). Guarded: a raising reporter does not change the return.
    Unset (default) = behaviour unchanged.

    whole_content (#813, default False = byte-identical for every existing caller):
    when True the snippet is NOT bounded to 300 chars -- it is the node's whole
    resolved text.  pith_provider_context passes True: its budget is met by dropping
    whole assemblies, so an upstream snippet cut would be a truncation it cannot see.
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

        # (#813: the Stage 4 proximity-keyed LOD staging -- a far promoted node shown
        # as a keyframe -- is removed.  It discarded the keyframe's delta, which is a
        # cut; a promoted node now stays whole at every distance and the budget
        # decides whether it survives.)
        out = []
        for r in surfaced:
            nid = r.get("node_id") or r.get("id")
            node = ng.graph.nodes.get(nid) if (nid and ng.graph) else None
            text = resolve_surface_content(
                node, r, allow_ingested=True,
                max_chars=(sys.maxsize if whole_content else 300))
            if not text:
                continue
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
        # #818: results beyond the root/result count k are dropped -- say so.
        _pith_log_drop("recall", "roots (result limit k=%d)" % k, "results",
                       [(item.get("node_id"), len(item.get("content") or "")) for item in out[k:]])
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
        _cc_report(on_error, exc)
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
    step only through cc_deposit_step or the Tonic's autostep; neither pushes those stats -- so CC dips the bucket at
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
# (#813: there is no per-node character cap -- a node is rendered WHOLE and the
# budget is met by dropping whole assemblies. The two caps below are REJECT-LOUDLY
# request guards: they refuse the whole request with a visible closed state.)
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
    basin builder and read by `_pith_render_connected_line` for the heading label.

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
        """Bump the fail-soft counter -- a failing Pith path falls back to the
        un-Pithed rendering in cc_assemble_recall, or returns an empty result
        (pith_prefetch_seed), so without this a 100%-failing Pith pass is
        indistinguishable from a working one. Call from the caller's
        except-handler."""
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
    "CC_PITH_PROVIDER_DEPTH",
    # #813: RETIRED knob.  Nothing reads it any more (resolved=None, authority
    # "retired").  It stays in this tuple because env == resolved == this tuple ==
    # cc_ng_host.PITH_SNAPSHOT_GATE_KEYS is asserted (tests/test_pith_metrics_concurrency.py,
    # tests/test_cc_host_pith_telemetry.py) and the host is not edited in this change;
    # and between the code deploy and the .bashrc removal a stale export is exactly what
    # an operator wants to see (env set, resolved None).  Remove here, in the resolved
    # dict, and in the host list together.
    "CC_PITH_PROVIDER_NODE_CHARS",
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
        # #813: retired -- no per-node cap exists; None = "not a live setting" (same
        # convention as the tonic_engine-unavailable entries below).
        "CC_PITH_PROVIDER_NODE_CHARS": None,
        "CC_PITH_PROVIDER_MAX_INSTRUCTION_CHARS": _CC_PITH_PROVIDER_MAX_INSTRUCTION_CHARS,
        "CC_PITH_PROVIDER_MAX_QUEST_CHARS": _CC_PITH_PROVIDER_MAX_QUEST_CHARS,
        "CC_PITH_PREFETCH_ENABLED": _CC_PITH_PREFETCH_ENABLED,
        "CC_PITH_PREFETCH_LOD_DIST": _CC_PITH_PREFETCH_LOD_DIST,
    }
    authority = {k: "cc_ng_organism" for k in resolved}
    authority["CC_PITH_PROVIDER_NODE_CHARS"] = "retired (#813: nodes render whole)"

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

    #813 (Exec P411/P413, Josh: no truncation) -- A KEYFRAME APPLIES ONLY WITH ITS
    DELTA.  `keyframe` + `delta` together are a lossless reordering of `content`;
    `keyframe` alone is a cut with a marker.  Model-facing budgeted output (the
    provider context, the Stage 3 L1 assembler, recall staging) has no room to carry
    the delta -- that is exactly what a binding budget refused -- so it never calls
    this function: an item is rendered whole or dropped whole.  Callers that use the
    keyframe on its own (today only pith_compress_history) own that loss.
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


# ---------------------------------------------------------------------------
# #813 -- THE ONE BUDGET RULE (Exec P411/P413/P416; checker-019 C3, le-017 F2/F8)
# Used by the provider admit, Stage 3 and the un-Pithed renderer, so no budgeted path can
# emit over budget, shorten an item, or drop one silently:
#   1. WHOLE OR ABSENT -- nothing is shortened to fit.
#   2. STRICT RANK PREFIX on the remaining envelope: the first unit that does not fit the
#      space left ends admission; a lower-ranked unit never jumps it.
#   3. NEVER-FIT (a unit that cannot fit an EMPTY envelope) is never emitted over budget
#      and does NOT end the prefix; it is skipped (the #819 reference form is tried first).
#   4. LOUD: every drop is ONE INFO line -- count, total chars, and the never-fit node ids
#      (bounded, first-time-seen only, so a recurring giant cannot flood the log).
# ---------------------------------------------------------------------------

_CC_PITH_DROP_LOG_IDS_PER_CALL = max(1, min(64, int(os.environ.get("CC_PITH_DROP_LOG_IDS_PER_CALL", "8"))))
_CC_PITH_DROP_LOG_SEEN_MAX = max(16, min(65536, int(os.environ.get("CC_PITH_DROP_LOG_SEEN_MAX", "4096"))))
_PITH_DROP_SEEN: Dict[str, None] = {}          # insertion-ordered; oldest evicted past the max
_PITH_DROP_SEEN_LOCK = threading.Lock()


def _pith_note_ids(ids: Any) -> tuple:
    """(shown, already_reported, more): the ids worth NAMING this call. An id is named the
    first time it is seen; repeats are only counted (flood-safe), and at most
    CC_PITH_DROP_LOG_IDS_PER_CALL new ids are named per call."""
    shown: List[str] = []
    already = more = 0
    with _PITH_DROP_SEEN_LOCK:
        for key in (str(i) for i in ids):
            if key in _PITH_DROP_SEEN:
                already += 1
            elif len(shown) < _CC_PITH_DROP_LOG_IDS_PER_CALL:
                shown.append(key)
                _PITH_DROP_SEEN[key] = None
                if len(_PITH_DROP_SEEN) > _CC_PITH_DROP_LOG_SEEN_MAX:
                    _PITH_DROP_SEEN.pop(next(iter(_PITH_DROP_SEEN)))
            else:
                more += 1
    return shown, already, more


def _pith_log_budget_drop(where: str, budget_label: str, unit: str, size_note: str, budget: int,
                          dropped: int, dropped_chars: int, kept: int, kept_chars: int,
                          never_fit: Any = ()) -> None:
    """The ONE INFO line for a budget drop (never called when nothing was dropped)."""
    message = ("pith %s: %s %d chars met by dropping %d whole %s (%d %s); kept %d (%d chars)"
               % (where, budget_label, budget, dropped, unit, dropped_chars, size_note,
                  kept, kept_chars))
    never_fit = list(never_fit)
    if never_fit:
        shown, already, more = _pith_note_ids(nid for nid, _chars in never_fit)
        sizes = {str(nid): chars for nid, chars in never_fit}
        named = ", ".join("%s (%d chars)" % (nid, sizes[nid]) for nid in shown)
        extra = "".join([" [%d already reported]" % already if already else "",
                         " [+%d more]" % more if more else ""])
        message += "; never-fit (cannot fit an empty envelope): %s%s" % (named or "-", extra)
    logger.info(message)


def _pith_log_drop(where: str, reason: str, unit: str, entries: Any) -> None:
    """#818: ONE INFO line for a structural (non-budget) drop -- a count limit or an overlap.
    `entries` is [(id, chars)].  Count, total chars and the reason are ALWAYS reported; ids are
    named first-time-seen only (bounded), so a recurring drop cannot flood the log (same shape
    as the #810 skip log).  Never called when nothing was dropped."""
    entries = list(entries)
    if not entries:
        return
    message = "pith %s: dropped %d %s (%d chars) - reason: %s" % (
        where, len(entries), unit, sum(chars for _i, chars in entries), reason)
    shown, already, more = _pith_note_ids("%s|%s" % (reason.split(" ")[0], i) for i, _c in entries)
    sizes = {"%s|%s" % (reason.split(" ")[0], i): c for i, c in entries}
    named = ", ".join("%s (%d chars)" % (k.split("|", 1)[1], sizes[k]) for k in shown)
    extra = "".join([" [%d already reported]" % already if already else "",
                     " [+%d more]" % more if more else ""])
    logger.info("%s; first seen: %s%s", message, named or "-", extra)


# ---------------------------------------------------------------------------
# #819 -- an OVER-BUDGET NODE surfaces through its TREES plus a whole-node REFERENCE
# (Exec P417; LAW 7: raw means complete; Josh P360: a long turn stays ONE node / one forest --
# windowed pooling, not chunking).  NO split at ingest, no new node type: the node still
# participates FULLY in activation and learning; ONLY its RENDERING takes this form.  A reference
# points to the whole; it is not a cut.  DEPENDENCY: for pre-PASS-2 forests the trees cover only
# the first 2,000 chars until PASS 2 (the laptop TID) runs; full coverage arrives with PASS 2.
# ---------------------------------------------------------------------------

def _pith_provider_node_limit(core: str, budget: int) -> int:
    """The per-node character limit ABOVE which a node is rendered as a reference (#819).

    An OPTIMISTIC bound, measured from the renderer itself (turn 6 / le-025 C-2; turn 5 had
    measured the WORST case, which in a tight budget fell BELOW what fits and dropped or
    referenced nodes that used to render whole):
      overhead = the SMALLEST envelope the renderer can wrap around one node -- ONE ordinary
      connected line (an alert-free coherence, no correction, no sources line, no anchors,
      no relations) inside the "Learned Situation" section, measured by rendering it with
      empty text and subtracting the core;
      limit    = budget - len(core) - overhead, floored at 1.
    A node ABOVE the limit cannot fit whole under ANY real assembly, so it takes the reference
    form.  A node at or below it is left WHOLE; if its real assembly needs more overhead than the
    minimum (alerts, sources, anchors, relations) it is caught at admit as a NEVER-FIT assembly:
    dropped whole, loudly, by id (THE ONE rule).  The two errors are not symmetric: a limit too
    large costs a loud whole-or-absent drop of a node that could not fit anyway; a limit too small
    (turn 5) turned a node that DID fit into a reference or nothing.  Whole-or-absent favours the
    first.  The ordinary-line choice reads _PITH_COHERENCE_STATES / _PITH_ALERT_COHERENCE, the
    same constants the renderer uses."""
    ordinary = [c for c in _PITH_COHERENCE_STATES if c not in _PITH_ALERT_COHERENCE]
    overheads = []
    for coherence in ordinary:
        line = CacheLine(node_id="", content="", coherence=coherence, stream="connected", sources=[])
        section = _pith_provider_sections(core, [line], [_pith_render_connected_line(line)])[0]
        overheads.append(len(section) - len(core))
    return max(1, budget - len(core) - (min(overheads) if overheads else 0))


def _pith_tree_nodes(graph: Any, node_id: str, limit: Optional[int] = None) -> List[tuple]:
    """[(tree_id, text)] for the concept trees linked to `node_id`, strongest link first.
    Trees are the forest's graph neighbours carrying `_tree_concept` (the real link made by
    _cc_bind_conversational_topology); each is small and is rendered WHOLE."""
    out: List[tuple] = []
    for nid, _kind, _strength in _pith_graph_neighbors(graph, node_id):
        node = graph.nodes.get(nid)
        if node is None or not (getattr(node, "metadata", None) or {}).get("_tree_concept"):
            continue
        text = _pith_node_raw_text(node)
        if text:
            out.append((nid, text))
    return out if limit is None else out[:limit]


def _pith_whole_node_reference(graph: Any, node_id: str, node: Any, text: str,
                               shown: Optional[int] = None) -> str:
    """The ONE-LINE reference to a node too large to render whole here: id, size, date, tree
    count.  It says the whole exists and where; it shows none of it.

    The trailing clause is TRUE, not a promise (turn 5 / le-022 N2): nothing about "concepts
    following" is said for a node with no concept trees; when the caller knows how many trees it
    will place (`shown`) the line says exactly that ("K of N concept trees follow"); when it does
    not (the provider path: the trees arrive as the basin's ordinary relations, subject to the
    member/depth caps) it says they follow "where they fit".  Concept trees are summaries, so the
    line also says they may cover only part of the node (for a pre-PASS-2 forest, only its first
    2,000 chars -- nothing on a node marks PASS 2, so the hedge is unconditional)."""
    size = len(text)
    size_text = ("\u2248%dk" % round(size / 1000.0)) if size >= 10000 else ("\u2248%d" % size)
    stamp = getattr(node, "creation_time", None)
    date = (time.strftime("%Y-%m-%d", time.gmtime(stamp))
            if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) and stamp > 0
            else "undated")
    trees = len(_pith_tree_nodes(graph, node_id))
    line = ("A long node (id %s; %s chars; %s; %d concept tree%s) is related to this cue; it is "
            "too large to render whole here."
            % (node_id, size_text, date, trees, "" if trees == 1 else "s"))
    if trees == 0:
        return line
    if shown is None:
        return line + " Its concept trees follow where they fit; they may cover only part of it."
    if shown <= 0:
        return line + " None of its concept trees fit here."
    if shown >= trees:
        return line + (" %d concept tree%s follow%s; they may cover only part of it."
                       % (trees, "" if trees == 1 else "s", "s" if trees == 1 else ""))
    return line + " %d of %d concept trees follow; they may cover only part of it." % (shown, trees)


def _pith_log_reference(where: str, entries: Any, limit: int) -> None:
    """One INFO line for nodes ABOVE the reference limit rendered as trees + reference (never a
    silent swap).  `limit` is the threshold that rejected them: the L1 budget on the recall paths,
    the measured per-node limit on the provider path -- so this says "above the reference limit",
    not "over budget" (a node the limit rejects need not exceed the budget)."""
    entries = list(entries)
    if not entries:
        return
    shown, already, more = _pith_note_ids("ref|%s" % i for i, _c in entries)
    sizes = {"ref|%s" % i: c for i, c in entries}
    named = ", ".join("%s (%d chars)" % (k.split("|", 1)[1], sizes[k]) for k in shown)
    extra = "".join([" [%d already reported]" % already if already else "",
                     " [+%d more]" % more if more else ""])
    logger.info("pith %s: %d node%s above the reference limit %d chars (%d chars) surfaced through "
                "their trees + a whole-node reference; first seen: %s%s",
                where, len(entries), "" if len(entries) == 1 else "s", limit,
                sum(c for _i, c in entries), named or "-", extra)


def _pith_reference_text(graph: Any, node_id: str, text: str, budget: int) -> Optional[str]:
    """The L1 / un-Pithed form of an over-budget item: the reference line plus as many of its
    strongest trees, WHOLE, as fit.  None when the node is unknown or even the reference cannot
    fit (the caller then drops the item loudly under THE ONE RULE)."""
    node = graph.nodes.get(node_id) if (graph is not None and node_id) else None
    if node is None:
        return None
    trees = _pith_tree_nodes(graph, node_id)
    candidates = trees[:_CC_PITH_PROVIDER_MEMBERS]

    def build(k: int) -> str:
        reference = _pith_whole_node_reference(graph, node_id, node, text, shown=k)
        return "\n".join([reference] + ["- concept: " + t for _tid, t in candidates[:k]])

    if len(build(0)) > budget:
        return None
    included = 0
    used = len(build(0))
    for _tid, tree_text in candidates:
        piece = "- concept: " + tree_text
        if used + 1 + len(piece) > budget:
            break
        used += 1 + len(piece)
        included += 1
    while included > 0 and len(build(included)) > budget:      # the "K of N" wording is a few chars longer
        included -= 1
    if included < len(trees):
        # Turn 4 / checker-022 C3: the node swap is loud, so the trees it could not carry are too
        # (whether the budget or the CC_PITH_PROVIDER_MEMBERS cap stopped them).  Counts only.
        logger.info("pith reference form: node %s shows %d of %d concept trees whole; %d left out "
                    "(budget %d chars, tree cap %d)", node_id, included, len(trees),
                    len(trees) - included, budget, _CC_PITH_PROVIDER_MEMBERS)
    return build(included)


def _pith_reference_lines(graph: Any, lines: List[CacheLine], budget: int) -> List[CacheLine]:
    """Stage 3 input: replace each UNPINNED line longer than the whole budget by its
    trees + reference form (new CacheLine copies; the originals -- and the nodes -- untouched)."""
    out: List[CacheLine] = []
    swapped: List[tuple] = []
    for cl in lines:
        if not cl.pinned and len(cl.content or "") > budget:
            form = _pith_reference_text(graph, cl.node_id, cl.content, budget)
            if form is not None:
                swapped.append((cl.node_id, len(cl.content)))
                out.append(_dc_replace(cl, content=form))
                continue
        out.append(cl)
    _pith_log_reference("L1", swapped, budget)
    return out


def _pith_reference_items(graph: Any, items: List[Dict[str, Any]], budget: int,
                          swapped: List[tuple], pinned: Any = None) -> List[Dict[str, Any]]:
    """Un-Pithed input: the same swap for recall dicts (copies; `swapped` collects entries).
    An identity-protected item (`pinned(node_id)`) is never swapped -- identity is indivisible."""
    out: List[Dict[str, Any]] = []
    for item in items:
        content = item.get("content", "") or ""
        if len(content) > budget and not (pinned is not None and pinned(item.get("node_id"))):
            form = _pith_reference_text(graph, item.get("node_id"), content, budget)
            if form is not None:
                swapped.append((item.get("node_id"), len(content)))
                out.append(dict(item, content=form))
                continue
        out.append(item)
    return out


def _pith_admit_strict_prefix(ordered: List[Any], budget: int, size_of: Any,
                              separator: int = 0) -> tuple:
    """The ONE rule over units already in rank order.  size_of(unit) is the unit's whole size
    in an empty envelope; `separator` chars are charged between admitted units.
    Returns (kept, dropped, never_fit, used) -- `dropped` includes `never_fit`."""
    kept: List[Any] = []
    dropped: List[Any] = []
    never_fit: List[Any] = []
    used = 0
    stopped = False
    for unit in ordered:
        alone = size_of(unit)
        if alone > budget:
            never_fit.append(unit)
            dropped.append(unit)
            continue
        if stopped:
            dropped.append(unit)
            continue
        cost = alone + (separator if kept else 0)
        if used + cost > budget:
            dropped.append(unit)
            stopped = True
            continue
        kept.append(unit)
        used += cost
    return kept, dropped, never_fit, used


def _pith_default_weights() -> Dict[str, float]:
    return {
        "pattern": _CC_PITH_W_RELEVANCE,
        "monitor": _CC_PITH_W_RECENCY,
        "recall": _CC_PITH_W_RELEVANCE,
        "victim": _CC_PITH_W_RECENCY,   # recovered drops: secondary prior, like recency
    }


def _pith_unified_rank(unpinned_lines: List[CacheLine],
                       weights: Optional[Dict[str, float]] = None) -> List[tuple]:
    """Stage 3 steps 2-4, factored out (no metrics, no side effects) so the un-Pithed
    renderer ranks by EXACTLY the same rule: per-stream min-max normalisation, stream weight,
    thermal fold, stable sort descending.  Returns [(unified, input_index, line)]."""
    if weights is None:
        weights = _pith_default_weights()
    stream_bounds: Dict[str, tuple] = {}
    for cl in unpinned_lines:
        lo, hi = stream_bounds.get(cl.stream, (cl.score, cl.score))
        stream_bounds[cl.stream] = (min(lo, cl.score), max(hi, cl.score))
    scored: List[tuple] = []
    for idx, cl in enumerate(unpinned_lines):
        lo, hi = stream_bounds.get(cl.stream, (cl.score, cl.score))
        norm = 1.0 if hi <= lo else (cl.score - lo) / (hi - lo)
        weight = weights.get(cl.stream, 1.0)
        # Stage 5 thermal fold: warm content (high Ca_i/firing) is gently preferred.
        # thermal defaults 0.0 -> multiplier 1.0 -> byte-identical to pre-Stage-5 ranking.
        unified = weight * norm * (1.0 + _CC_PITH_THERMAL_GAIN * cl.thermal)
        scored.append((unified, idx, cl))
    # Ties keep input order because idx (ascending) is the secondary sort key.
    scored.sort(key=lambda t: (-t[0], t[1]))
    return scored


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
    5. Budget fill by THE ONE BUDGET RULE (see _pith_admit_strict_prefix),
       accumulating len(content): whole or absent, never shortened. Keep
       lines while the running total stays <= budget_chars; the first line
       that does not fit the REMAINING budget ends the ranked prefix. A line
       longer than the whole (empty) budget -- including the FIRST, top-ranked
       one -- is skipped, never emitted over budget (#813 D8: the old "keep
       the first line even if it alone exceeds the budget" guard is gone; it
       was a silent overrun), and it does not end the prefix. Every drop is
       ONE INFO line: count, total chars, and the never-fit node ids. (An
       over-budget recall item normally reaches this function already in its
       trees + reference form, #819.) Pinned lines are outside the budget.
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

    # Step 1: split pinned vs unpinned.
    pinned_lines = [cl for cl in cache_lines if cl.pinned]
    unpinned_lines = [cl for cl in cache_lines if not cl.pinned]

    # Steps 2-4: normalise + weight + thermal fold + stable sort (shared with the
    # un-Pithed renderer -- one ranking rule).
    scored = _pith_unified_rank(unpinned_lines, weights)

    # Step 5: THE ONE BUDGET RULE (see the block above).  Strict rank prefix on the remaining
    # budget; a line is kept WHOLE or dropped WHOLE.  #813 (Exec P411/P413/P416, Josh: no
    # truncation): never replaced by a keyframe (a keyframe without its delta is a cut), and
    # -- checker-019 C3 / le-017 F2 -- no longer kept "even if it alone exceeds the budget":
    # that guard was a silent overrun and made this the only budgeted path that emitted over
    # budget.  A line that cannot fit an EMPTY budget is skipped and named in the INFO line.
    kept_unpinned, dropped_lines, never_fit, running_total = _pith_admit_strict_prefix(
        [cl for _unified, _idx, cl in scored], budget_chars,
        lambda cl: len(cl.content or ""), separator=0)
    dropped = len(dropped_lines)
    if dropped:
        _pith_log_budget_drop(
            "stage3", "L1 budget", "items", "chars", budget_chars, dropped,
            sum(len(cl.content or "") for cl in dropped_lines), len(kept_unpinned),
            running_total, [(cl.node_id, len(cl.content or "")) for cl in never_fit])

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
    """Resolve one node's own meaning, WHOLE, for a connected assembly.

    #813 (Exec P411/P413, Josh: no truncation): there is no per-node clip.  The
    budget is met by admitting fewer whole assemblies (_pith_provider_admit), never
    by shortening what a node says.

    Tree nodes keep their own concept while a forest keeps the lived turn.  This
    differs deliberately from standalone snippet display: an assembly already
    carries the forest keyframe, so repeating that forest for every tree would
    erase the relationships the cache line exists to preserve.
    """
    return _pith_node_raw_text(node, fallback)


def _pith_node_sources(node: Any) -> list:
    meta = getattr(node, "metadata", None) or {}
    values = []
    for key in ("source", "provenance", "creation_mode"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
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
                                      live_rails: Optional[Dict[str, str]] = None,
                                      node_limit: Optional[int] = None) -> List[CacheLine]:
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

    referenced: Dict[str, int] = {}      # #819: node id above the reference limit -> whole size

    def _display_text(node, fallback="", node_id=None):
        raw = _pith_node_raw_text(node, fallback)
        rail_label = rail_labels.get(_pith_normalize(raw))
        if rail_label:
            return raw, f"[{rail_label} is present exactly once in the live tail]", True
        if node_limit and node_id and len(raw) > node_limit:
            # #819: too large to render whole -> ONE reference line; its trees are this basin's
            # ordinary graph neighbours and arrive whole as relations.  Returning "" as the raw
            # text means no anchors are mined from the unshown whole (metadata anchors remain).
            referenced[node_id] = len(raw)
            return "", _pith_whole_node_reference(graph, node_id, node, raw), False
        return raw, _pith_node_text(node, fallback), False

    active_node_ids = {item.get("node_id") for item in surfaced if item.get("node_id")}
    count_declined: Dict[str, tuple] = {}      # #818: node id -> (reason, chars), across all basins
    root_scores = [float(item.get("score", 0.0) or 0.0) for item in surfaced]
    score_lo = min(root_scores) if root_scores else 0.0
    score_hi = max(root_scores) if root_scores else 0.0

    for root_item in surfaced:
        root_id = root_item.get("node_id")
        root = graph.nodes.get(root_id) if root_id else None
        if root is None or _pith_is_constitutional(graph, root_id):
            continue
        root_raw, root_text, root_is_live = _display_text(
            root, root_item.get("content", ""), root_id)
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
        declined: Dict[str, str] = {}      # #818: neighbour the walk reached but did not admit -> reason
        parent_id = root_id                # last expanded parent (defined even if the walk never runs)

        while frontier and len(members) < member_limit:
            parent_id, parent_text, parent_node, depth = frontier.pop(0)
            neighbors = _pith_graph_neighbors(graph, parent_id, active_node_ids)
            total_support += sum(strength for _nid, _kind, strength in neighbors)
            if depth >= depth_limit:
                for _cid, _k, _s in neighbors:
                    declined.setdefault(_cid, "depth_limit")
                continue
            for child_id, edge_kind, strength in neighbors:
                if child_id in visited or _pith_is_constitutional(graph, child_id):
                    continue
                child = graph.nodes.get(child_id)
                child_raw, child_text, child_is_live = _display_text(child, "", child_id)
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

        if len(members) >= member_limit:
            # The member budget is spent: everything still unexpanded or unvisited was declined.
            for _cid, _k, _s in _pith_graph_neighbors(graph, parent_id, active_node_ids):
                declined.setdefault(_cid, "member_limit")
            for f_id, _t, _n, f_depth in frontier:
                for _cid, _k, _s in _pith_graph_neighbors(graph, f_id, active_node_ids):
                    declined.setdefault(_cid, "depth_limit" if f_depth >= depth_limit else "member_limit")
        for _cid, _reason in declined.items():
            if _cid not in visited and not _pith_is_constitutional(graph, _cid):
                _dnode = graph.nodes.get(_cid)
                _dchars = len(_pith_node_raw_text(_dnode)) if _dnode is not None else 0
                if _dchars:                                   # a node with no text carries nothing
                    count_declined.setdefault(_cid, (_reason, _dchars))

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
    overlapped = []
    for line in raw_basins:
        members = set(line.member_node_ids)
        if members and len(members & covered) / len(members) >= 0.6:
            overlapped.append((line.node_id, len(_pith_render_connected_line(line))))
            continue
        selected.append(line)
        covered.update(members)
    # #818: nothing above vanishes silently.  A declined neighbour counts only if it is in NO
    # selected basin (it may have been admitted to another root's basin).
    _pith_log_drop("basins", "member_limit (CC_PITH_PROVIDER_MEMBERS=%d)" % member_limit,
                   "neighbour nodes",
                   [(nid, chars) for nid, (why, chars) in count_declined.items()
                    if why == "member_limit" and nid not in covered])
    _pith_log_drop("basins", "depth_limit (CC_PITH_PROVIDER_DEPTH=%d)" % depth_limit,
                   "neighbour nodes",
                   [(nid, chars) for nid, (why, chars) in count_declined.items()
                    if why == "depth_limit" and nid not in covered])
    _pith_log_drop("basins", "overlap (>=60% of its members already covered by a higher-ranked basin)",
                   "basins", overlapped)
    if referenced:
        _pith_log_reference("basins", [(nid, n) for nid, n in referenced.items() if nid in covered],
                            node_limit or 0)
    return selected


def _pith_render_connected_line(line: CacheLine) -> str:
    """Model-facing Markdown for one whole cache line; never renders scores/ids."""
    label = "learned from substrate" if line.epistemic == "learned" else line.epistemic
    lines = [f"### Connected assembly [{label}; coherence: {line.coherence}]",
             f"- Root: {line.content}"]
    for relation in line.relations:
        lines.append(f"- {relation['kind']}: {relation['content']}")
    if line.sources:
        lines.append("- Sources: " + ", ".join(line.sources))
    if line.anchors:
        lines.append("- Exact anchors: " + ", ".join(f"`{a}`" for a in line.anchors))
    return "\n".join(lines)


def _pith_provider_admit(lines: List[CacheLine], budget_chars: int) -> tuple:
    """Admit WHOLE relationship cache lines by THE ONE BUDGET RULE (see its block above).

    #813 (Exec P411/P413, Josh: no truncation): nothing is shortened.  A cache line -- prose,
    every relation, sources, coherence, exact anchors -- is admitted whole or not at all.
    Admission is a strict ranked prefix on the remaining envelope; an assembly that could not
    fit even an EMPTY envelope is skipped (it does not blank the rest) and named in the INFO
    line; every drop is reported there with its count and total rendered size.
    """
    ordered = sorted(lines, key=lambda line: (-line.score, line.node_id))
    sizes = {id(line): len(_pith_render_connected_line(line)) for line in ordered}
    kept_lines, dropped, never_fit, used = _pith_admit_strict_prefix(
        ordered, budget_chars, lambda line: sizes[id(line)], separator=2)
    kept = [_pith_copy_cache_line(line) for line in kept_lines]
    rendered = [_pith_render_connected_line(line) for line in kept]
    if dropped:
        _pith_log_budget_drop(
            "provider_context", "learned budget", "assemblies", "chars rendered", budget_chars,
            len(dropped), sum(sizes[id(line)] for line in dropped), len(kept), used,
            [(line.node_id, sizes[id(line)]) for line in never_fit])
    return kept, rendered


def _pith_line_is_correction(line: CacheLine) -> bool:
    return any("correction" in relation["kind"] or "failure" in relation["kind"]
               for relation in line.relations)


# The renderer's coherence vocabulary, defined ONCE (turn 6 / le-025 C-4).  The alert set is what
# _pith_provider_sections turns into an "Uncertainty and Conflicts" paragraph + a warning, and
# _pith_provider_node_limit derives its ORDINARY (alert-free) line from the same constants, so a
# change here moves the renderer and the measured limit together -- no second, drifting list.
_PITH_COHERENCE_STATES = ("exclusive", "shared", "modified", "uncertain", "stale", "conflict", "unknown")
_PITH_ALERT_COHERENCE = ("conflict", "stale", "uncertain", "unknown")


def _pith_provider_sections(core: str, lines: List[CacheLine],
                            blocks: List[str]) -> tuple:
    """Render complete provider context and its learned-state warnings."""
    situation = []
    corrections = []
    warnings = []
    alert_states = []
    for line, block in zip(lines, blocks):
        (corrections if _pith_line_is_correction(line) else situation).append(block)
        if line.coherence in _PITH_ALERT_COHERENCE:
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
            ng, cue, roots, state=recall_state, preserve_graph_config=True,
            whole_content=True)
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
            graph, surfaced, live_rails=live_rails,
            node_limit=_pith_provider_node_limit(core, budget))
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


def cc_pith_failure_text(exc: BaseException) -> str:
    """Raw text of a Pith recall failure: the exception type and message,
    as-is. No category, severity or tag (LAW 7) -- this is what the
    hemisphere deposit paths hand to the substrate."""
    return f"NeuroGraph recall Pith pass failed: {type(exc).__name__}: {exc}"


def cc_deposit_pith_failure(exc: BaseException, tract_path: Optional[str] = None) -> None:
    """Deposit a Pith recall failure raw onto the CC ingest tract -- the same
    tract, entry type and source ("cc_gateway") miniTID uses for conversational
    turns, so drain_ingest_tract absorbs it exactly as it absorbs a turn. The
    laptop hemisphere's on_pith_failure. Raises on write failure; the caller
    (cc_assemble_recall) logs it."""
    import ng_tract
    ng_tract.deposit_experience(cc_pith_failure_text(exc).encode("utf-8"),
                                "cc_gateway", [tract_path or cc_gateway_tract_path()])


def _cc_pin_guard_failed(node_id: Any, exc: BaseException, failed: Optional[Dict[Any, str]] = None) -> bool:
    """The identity-pin guard raised (or is missing): FAIL CLOSED -- the item is PINNED.

    Identity fails toward KEEPING content (#92 "nothing protected dies"; Duck Ethics): a broken
    guard must never make an identity item budget-droppable.  Loud, not silent: a WARNING naming
    the node id and the exception TYPE only (never node text), the id first-time-seen (bounded,
    flood-safe -- the same tracker as the drop lines).  `failed` is the calling probe's per-call
    record (node id -> exception type) that _cc_log_guard_pins turns into ONE count-only line.
    Returns True."""
    if failed is not None:
        failed[node_id] = type(exc).__name__
    shown, _already, _more = _pith_note_ids(["pin|%s" % (node_id,)])   # a LIST: a bare str iterates chars
    if shown:
        logger.warning("Pith identity-pin guard failed for node %s (%s); treated as PINNED "
                       "(fail closed: identity keeps its content, outside the budget)",
                       node_id, type(exc).__name__)
    return True


def _cc_pin_probe(ng: Any) -> Any:
    """The identity-pin test used by Stage 3 AND the un-Pithed renderer (turn 5 / le-022 N1):
    `ng.graph._is_identity_protected(node_id)`.

    Turn 6 (le-025 C-5, Chief ruling): the guard FAILS CLOSED.  When it raises, or does not exist
    (no `.graph`, no `_is_identity_protected`), the item is treated as PINNED and a WARNING is
    logged (_cc_pin_guard_failed) -- base failed soft to NOT pinned at DEBUG, which let a vanished
    guard make identity items budget-droppable.  The guarded call itself is unchanged from base
    (a test compares it with e4ebf982's closure).

    Turn 7 (le-027 N-2): one probe lives for ONE recall call, so it also carries that call's
    record of which items a FAILED guard pinned (`_pinned.guard_failed`), which the path turns
    into one count-only line (_cc_log_guard_pins)."""
    failed: Dict[Any, str] = {}

    def _pinned(node_id):
        try:
            return bool(ng.graph._is_identity_protected(node_id))
        except Exception as exc:
            return _cc_pin_guard_failed(node_id, exc, failed)
    _pinned.guard_failed = failed
    return _pinned


def _cc_log_guard_pins(where: str, pinned: Any, node_ids: List[Any], l1_chars: int, budget: int) -> None:
    """Turn 7 / le-027 N-2: ONE count-only line PER CALL when any item of this call was pinned
    BECAUSE the identity guard failed.

    Fail-closed keeps every such item whole and OUTSIDE the budget, so a dead guard can leave the
    L1 far over its budget with no drop line (drops are the only thing the other lines report).
    This is the signal that persists for as long as it costs the prompt.  Counts only: how many
    items, how many in total, the exception type(s), the L1 size vs the budget -- no node text and
    no ids (the first-seen per-id WARNING already names them once).  WARNING, not INFO: the drop /
    reference lines report designed behaviour; a dead identity guard is an operational fault."""
    failed = getattr(pinned, "guard_failed", None) or {}
    hit = [nid for nid in node_ids if nid in failed]
    if not hit:
        return
    logger.warning("pith %s: %d of %d items pinned because the identity guard failed (%s); "
                   "L1 %d chars vs budget %d", where, len(hit), len(node_ids),
                   "/".join(sorted({failed[nid] for nid in hit})), l1_chars, budget)


# INTERIM FORK (T1, le-022): _cc_monitor_items_whole + _format_cc_monitor_block below exist only
# because the producer's cut lives in SHARED surfacing.py / surface_resolver.py (Syl's /assemble,
# P329). The SOURCE fix -- a whole-content option there (#812, Josh's post-track go) -- DELETES both.
def _cc_monitor_items_whole(ng: Any, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """#816 CC-ONLY route for the SurfacingMonitor stream.

    The shared monitor resolves each fired node with the shared resolver's default 240-char
    bound (surfacing.py:192 -> surface_resolver.resolve_surface_item) BEFORE we see it.  Editing
    that default or surfacing.py would change Syl's live /assemble path (P329, #812), so instead
    this wrapper re-resolves each item's WHOLE content by node_id from the same node + vector-db
    entry the monitor used (Exec P410(c): the wrapper renders full stored content).  Returns new
    dicts -- the monitor's own items are never mutated.  Per item: one that is not re-resolvable
    by design (unknown node, image frame, filtered/empty) keeps what the monitor gave us.  One
    whose re-resolve RAISES is DROPPED -- whole-or-absent (turn 4 / checker-022 C2): keeping the
    shared 240-char snippet would leak a cut item.  The drop is a WARNING naming the node id and
    the exception TYPE (never node text), the id first-time-seen only (flood-safe).
    """
    graph = getattr(ng, "graph", None)
    vdb = getattr(ng, "vector_db", None)
    out: List[Dict[str, Any]] = []
    failed: List[tuple] = []
    for item in items or []:
        fresh = dict(item)
        nid = item.get("node_id")
        try:
            node = graph.nodes.get(nid) if (graph is not None and nid) else None
            if node is not None and not item.get("image_ref"):
                from surface_resolver import resolve_surface_content
                entry = vdb.get(nid) if vdb is not None else None
                text = resolve_surface_content(node, entry, max_chars=sys.maxsize)
                if text:
                    fresh["content"] = text
        except Exception as exc:
            failed.append((nid, type(exc).__name__))
            continue                         # whole-or-absent: never append the cut snippet
        out.append(fresh)
    if failed:
        shown, already, more = _pith_note_ids("monitor|%s" % nid for nid, _why in failed)
        why = {"monitor|%s" % nid: w for nid, w in failed}
        named = ", ".join("%s (%s)" % (k.split("|", 1)[1], why[k]) for k in shown)
        extra = "".join([" [%d already reported]" % already if already else "",
                         " [+%d more]" % more if more else ""])
        logger.warning("CC monitor whole-content re-resolve failed for %d item%s; DROPPED "
                       "(whole-or-absent -- a cut item is never kept): %s%s",
                       len(failed), "" if len(failed) == 1 else "s", named or "-", extra)
    return out


# INTERIM FORK (T1): deleted together with _cc_monitor_items_whole by the #812 source fix.
def _format_cc_monitor_block(items: List[Dict[str, Any]]) -> str:
    """The CC-side twin of SurfacingMonitor.format_context WITHOUT its 200-char cut.

    The layout is byte-identical for items the shared code would not cut (a parity test pins
    it against the real shared function): the `[NeuroGraph Surfaced Knowledge]` header is
    miniTID's rail marker and must not change.  Kept here rather than editing the shared
    formatter, which also serves Syl's /assemble."""
    if not items:
        return ""
    lines = ["[NeuroGraph Surfaced Knowledge]"]
    for item in items:
        content = item.get("content", "")
        score = item.get("score", 0.0)
        if not content and item.get("image_ref"):
            lines.append(f"- [something you saw \u2014 image attached] (salience: {score:.2f})")
            continue
        lines.append(f"- {content} (salience: {score:.2f})")
    return "\n".join(lines)


def _cc_render_unpithed(ng: Any, monitor_items: List[Dict[str, Any]],
                        pc_results: List[Dict[str, Any]], commons: Any,
                        pc_fired_ids: List[str], on_surfaced: Optional[Any] = None) -> str:
    """The un-Pithed recall rendering (gate OFF, or the Pith path raised): the SurfacingMonitor
    block then the Active Recall block, every item WHOLE.

    Size is controlled by HOW MANY (#816, Exec P410/P416): items are ranked by the same unified
    rank as Stage 3 and admitted by THE ONE BUDGET RULE against the existing L1 budget
    (cc_l1_budget); the lowest-ranked WHOLE items are dropped and ONE INFO line reports the
    count and total size.  Short items that all fit render exactly as before.  Identity-
    protected items (`ng.graph._is_identity_protected`, the same test Stage 3 uses) are outside
    the budget: rendered whole, never dropped, never swapped for a reference."""
    # Josh 2026-09-26: "When Pith fails, there HAS to be pass-through".  This is the fallback for
    # a failed Pith pass, so it must not depend on the machinery that may have just failed, and
    # it must never raise: if the ranking/budget step itself fails, every item is rendered WHOLE
    # and unbudgeted, LOUDLY (warning) -- never dropped, never cut.
    try:
        try:
            budget = cc_l1_budget(commons, getattr(ng, "graph", None), pc_fired_ids)
        except Exception as exc:                               # never let the budget sink recall
            logger.debug("un-Pithed budget lookup failed (static budget used): %s", exc)
            budget = _CC_PITH_L1_BUDGET
        swapped: List[tuple] = []
        graph = getattr(ng, "graph", None)
        pinned = _cc_pin_probe(ng)
        monitor_items = _pith_reference_items(graph, monitor_items, budget, swapped, pinned)
        pc_results = _pith_reference_items(graph, pc_results, budget, swapped, pinned)
        _pith_log_reference("recall (un-Pithed)", swapped, budget)
        tagged = (
            [(CacheLine(node_id=i.get("node_id") or "", content=i.get("content", "") or "",
                        score=float(i.get("score", 0.0) or 0.0), stream="monitor"), i)
             for i in monitor_items]
            + [(CacheLine(node_id=i.get("node_id") or "", content=i.get("content", "") or "",
                          score=float(i.get("score", 0.0) or 0.0), stream="pattern"), i)
               for i in pc_results])
        origin = {id(cl): item for cl, item in tagged}         # line -> its recall item
        # N1: identity-protected items sit OUTSIDE the budget exactly as in Stage 3 -- kept whole,
        # never ranked, never dropped, and never the reason a strict-prefix stop loses a smaller
        # unprotected neighbour.  Only the ordinary items are ranked and admitted.
        pinned_items = {id(item) for cl, item in tagged if pinned(cl.node_id)}
        ranked = _pith_unified_rank([cl for cl, item in tagged if id(item) not in pinned_items])
        kept_lines, dropped, never_fit, used = _pith_admit_strict_prefix(
            [cl for _u, _ix, cl in ranked], budget, lambda cl: len(cl.content or ""), separator=0)
        if dropped:
            _pith_log_budget_drop(
                "recall (un-Pithed)", "L1 budget", "items", "chars", budget, len(dropped),
                sum(len(cl.content or "") for cl in dropped), len(kept_lines), used,
                [(cl.node_id, len(cl.content or "")) for cl in never_fit])
        kept = {id(origin[id(cl)]) for cl in kept_lines} | pinned_items
        _cc_log_guard_pins(
            "recall (un-Pithed)", pinned, [cl.node_id for cl, _item in tagged],
            sum(len(item.get("content", "") or "") for _cl, item in tagged if id(item) in kept), budget)
    except Exception as exc:
        logger.warning("un-Pithed budget step failed; rendering every item whole and unbudgeted: %s",
                       exc)
        kept = {id(i) for i in list(monitor_items) + list(pc_results)}
    monitor_block = _format_cc_monitor_block([i for i in monitor_items if id(i) in kept])
    pc_block = _format_cc_recall_block([i for i in pc_results if id(i) in kept])
    if on_surfaced is not None:
        # [lane 812-813-onto-s4] report exactly what this render emitted (the WHOLE kept items,
        # monitor first; a monitor item counts only if its block rendered) and what the ONE
        # budget rule dropped (whole).  Guarded by _cc_report: never changes the return.
        _rendered = ([_cc_surfaced_item(i, 'monitor') for i in monitor_items
                      if id(i) in kept and monitor_block]
                     + [_cc_surfaced_item(i, 'pattern') for i in pc_results
                        if id(i) in kept and pc_block])
        _dropped = ([_cc_surfaced_item(i, 'monitor') for i in monitor_items if id(i) not in kept]
                    + [_cc_surfaced_item(i, 'pattern') for i in pc_results if id(i) not in kept])
        _cc_report(on_surfaced, _rendered, _dropped)
    if monitor_block and pc_block:
        return monitor_block + "\n\n" + pc_block
    return monitor_block or pc_block


def cc_assemble_recall(ng: Any, query: str, k: int, conv_state: dict, commons: Any,
                        allow_pattern_completion: bool = True,
                        on_monitor_error: Optional[Any] = None,
                        on_pith_failure: Optional[Any] = None,
                        on_degraded: Optional[Any] = None,
                        on_surfaced: Optional[Any] = None) -> str:
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
    unconditionally. Any exception anywhere in the Pith path is fail-soft --
    falls back to the pre-Pith monitor_ctx/pc_block rendering -- records a
    _PITH_METRICS failure, rate-limit-warns, and hands the raw exception to
    on_pith_failure (each hemisphere wires its own raw deposit there -- this
    query function does no write-side work itself, LAW 4). It never raises:
    a surfacing pass must not crash or time out the hook.

    on_degraded: optional reporter, on_degraded(code, exc), for the three
    swallowed failures that leave the returned text valid but INCOMPLETE; the
    wrapper (daemon / host) decides what to count, log or put on a wire.
    code is one of 'monitor_race' (the SurfacingMonitor harvest hit a dict-
    mutation race; the monitor block is missing), 'pattern_completion_failed'
    (Active Recall raised, either inside cc_pattern_completion_recall or at the
    call; the Active Recall block is missing) and 'pith_fallback' (the Pith
    pipeline raised and the un-Pithed text above was returned; reported AFTER
    on_pith_failure, which is still called exactly as before). exc is the
    exception object: log its class at most -- its message can carry the
    prompt, paths or secrets. Calls arrive in pipeline order (monitor, Active
    Recall, Pith) and one request can report several; 'pattern_completion_failed'
    may arrive twice only if the failure is reported below AND a later step of
    the same block raises. Guarded: a raising reporter changes neither the
    returned text, the existing callbacks, their order, nor the rate-limited
    WARNING/metrics. Unset (default) = behaviour byte-identical, and the call
    to cc_pattern_completion_recall carries no new kwarg.

    on_surfaced: optional reporter, on_surfaced(rendered, dropped), called once
    just before the return with what this pass actually surfaced. Each list
    holds {'stream', 'node_id', 'score', 'content'} dicts in render order;
    `dropped` is what the budget dropped WHOLE (Pith Stage 3, or the un-Pithed
    renderer's ONE budget rule since #813). `content` is the WHOLE text actually
    rendered (an over-budget item reports its trees + one-line reference form).
    Reporting only -- the caller decides where it goes (LAW 4). Guarded like
    on_degraded; unset (default) = behaviour byte-identical.

    Params only (ng/conv_state/commons) -- no module-global STATE access,
    so this function is process-agnostic (Syl's-Law) and safe to call from
    either hemisphere with its own isolated instances.
    """
    monitor_node_ids: set = set()
    monitor_items: List[Dict[str, Any]] = []
    try:
        monitor = getattr(ng, '_surfacing_monitor', None)
        if monitor is not None:
            monitor_items = monitor.get_surfaced()
            monitor_node_ids = {item.get('node_id') for item in monitor_items}
    except RuntimeError as exc:
        monitor_node_ids = set()  # dict mutation race during concurrent deposit
        monitor_items = []
        _cc_report(on_degraded, 'monitor_race', exc)
    except Exception as exc:
        logger.debug('Recall failed: %s', exc)
        if on_monitor_error is not None:
            try:
                on_monitor_error(exc)
            except Exception:
                pass  # the error-reporting hook itself must never break recall
        monitor_node_ids = set()
        monitor_items = []
    # #816 CC-ONLY ROUTE: the shared SurfacingMonitor cut each item at 240 chars before it
    # reached us (surfacing.py:192 -> surface_resolver default).  Re-resolve every item's WHOLE
    # content by node_id here; the shared modules stay untouched (Syl's /assemble, P329).
    monitor_items = _cc_monitor_items_whole(ng, monitor_items)
    # An item dropped by the whole-or-absent rule must not also suppress its pattern-stream twin
    # (which is whole): dedupe against the items that SURVIVED.
    monitor_node_ids = {item.get('node_id') for item in monitor_items}

    pc_results: List[Dict[str, Any]] = []
    # Everything pattern completion fired, before the display dedup against
    # the monitor below: the L1 budget's region is what fired, not what is new.
    pc_fired_ids: List[str] = []
    if allow_pattern_completion:
        try:
            pc_extra: Dict[str, Any] = {}
            if on_degraded is not None:
                # Reporting only: lets the swallow INSIDE cc_pattern_completion_recall
                # be seen. Not passed when unset, so the default call is unchanged.
                pc_extra['on_error'] = lambda exc: _cc_report(on_degraded, 'pattern_completion_failed', exc)
            # #816: WHOLE content for BOTH the Pith-ON stream and the gate-off block -- the
            # budget (Stage 3 / the un-Pithed renderer) decides how MANY items, never how
            # much of one (a 300-char snippet here was cut before any budget could see it).
            pc_results = cc_pattern_completion_recall(
                ng, query, k, state=conv_state, whole_content=True, **pc_extra)
            pc_fired_ids = [r.get('node_id') for r in pc_results if r.get('node_id')]
            pc_results = [r for r in pc_results if r.get('node_id') not in monitor_node_ids]
        except Exception as exc:
            logger.debug('Pattern-completion recall failed (non-fatal): %s', exc)
            pc_results = []
            pc_fired_ids = []
            _cc_report(on_degraded, 'pattern_completion_failed', exc)

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
        # Which Pith step is running -- named in the fallback log line if one raises.
        _stage = 'CacheLine build'
        try:
            _pinned = _cc_pin_probe(ng)      # the ONE pin test, shared with the un-Pithed renderer

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
            # #819: an item longer than the whole L1 budget surfaces through its trees + a
            # one-line whole-node reference (copies; the node and the victim buffer's lines
            # are untouched), instead of being dropped as never-fit.
            survivors = _pith_reference_lines(ng.graph, survivors, budget)
            # [lane 812-813-onto-s4] what Stage 3 actually chose from (reference copies
            # included), so on_surfaced's `dropped` never lists an item rendered as its reference.
            _l1_in = survivors
            _stage3_ids = [cl.node_id for cl in survivors]
            survivors = pith_stage3(survivors, budget_chars=budget)
            _cc_log_guard_pins("L1 (Pith-ON)", _pinned, _stage3_ids,
                               sum(len(cl.content or "") for cl in survivors), budget)
            # Pith Stage 5 (eviction): budget-dropped lines fall to the victim buffer.
            try:
                pith_victim_capture(survivors, _pre_l1)
            except Exception as exc:
                logger.debug('Pith victim capture failed (non-fatal): %s', exc)
            _stage = 'render'
            survivor_results = [{'score': cl.score, 'content': cl.content} for cl in survivors]
            if on_surfaced is not None:
                _kept = {id(cl) for cl in survivors}
                _cc_report(on_surfaced, [_cc_surfaced_item(cl) for cl in survivors],
                           [_cc_surfaced_item(cl) for cl in _l1_in if id(cl) not in _kept])
            return _format_cc_recall_block(survivor_results)
        except Exception as exc:
            # Fail-soft: fall back to the un-Pithed monitor_ctx/pc_block
            # rendering below. COUNT it and warn (rate-limited) so a failing
            # Pith path is observable, then hand the raw exception to the
            # caller's deposit (LAW 4: no write-side work in this query fn).
            _PITH_METRICS.record_failure()
            global _last_pith_warn_ts
            _now = time.time()
            if _now - _last_pith_warn_ts >= _PITH_WARN_INTERVAL_S:
                _last_pith_warn_ts = _now
                logger.warning('Pith %s failed, falling back to un-Pithed rendering: %s', _stage, exc)
            else:
                logger.debug('Pith %s failed (non-fatal), falling back to un-Pithed rendering: %s', _stage, exc)
            if on_pith_failure is not None:
                try:
                    on_pith_failure(exc)
                except Exception as cb_exc:
                    # The deposit hook must never break recall -- but a lost
                    # failure deposit is logged, not swallowed.
                    logger.warning('Pith failure deposit failed: %s', cb_exc)
            # Report the fallback AFTER the existing callback so its order and
            # the WARNING/metrics above are untouched (guarded: never raises).
            _cc_report(on_degraded, 'pith_fallback', exc)

    # Gate OFF, or the Pith path failed: the un-Pithed rendering -- two blocks, monitor first --
    # now WHOLE per item, size controlled by how MANY (the ONE budget rule, INFO on any drop).
    # on_surfaced is reported by the renderer: only it knows which WHOLE items were kept.
    return _cc_render_unpithed(ng, monitor_items, pc_results, commons, pc_fired_ids,
                               on_surfaced=on_surfaced)
