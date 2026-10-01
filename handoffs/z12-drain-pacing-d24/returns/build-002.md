<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, dispatch #12731) — return build-002 for D24 (re-scoped): BUILT on the Exec P476 ruling
# What: turn-2 return. READ (plain, first), STOP-condition status, what changed with file:line at each tip, signatures,
#   the byte-identity proof, printed preambles + run outputs + base failing counts, what was NOT verified, findings for the punchlist.
# Why: the brief's ADDENDUM (Exec P476 via Chief-003) accepted the turn-1 STOP (e) (build-001.md, 76d3b4e3) and ruled the lock case.
# How: NG commit 38febe6e (cc_ng_organism.py + tests/test_cc_drain_pacing.py), docs-daemon commit 55fb0e1f (scripts/cc-ng-daemon.py
#   + scripts/tests/test_cc_ng_daemon_drain_pacing.py). build-001.md is left intact as the accepted turn-1 record.
# -------------------
-->

# build-002 — D24 drain pacing: BUILT (two repos)

Lane `drain-pacing-d24` (Z12), dispatch #12731. Builder: Claude Sonnet 5.5. Related: [[NeuroGraph]], [[The River]], [[Dual-Pass Embedding]], [[FatherGraph]]. Turn-1 record: `build-001.md` (STOP (e), accepted).

**NG merges BEFORE the daemon** (a daemon passing `batch_nodes=`/`receipt=` to an older organism raises `TypeError`; the surrounding `except` at the daemon's drain site swallows it at DEBUG, so the drain would stop silently). **Merge held; nothing wired; no restart, no `.bashrc`/crontab edit, no PR.**

## Hashes (real `git rev-parse HEAD`)
| Repo / branch | Commit | What |
|---|---|---|
| NeuroGraph `cc-laptop-drain-pacing-d24-20261001` (base `e4ebf982b1989fd9066d610b94853bc68bf70d37`, no upstream, never pulled/rebased) | `38febe6e78197b85fa7af4f8814f5f632f1c3433` | the ONE code commit: `cc_ng_organism.py` + `tests/test_cc_drain_pacing.py` |
| same branch | (this return commit; see the final message for its hash) | docs-only: `handoffs/z12-drain-pacing-d24/returns/build-002.md` (+ one pointer line in `build-001.md`) |
| docs daemon `cc-laptop-daemon-d24-env-20261001` (base `e60524154f9cfd6a1be34d28d97afc174b61accb`, not rebased) | `55fb0e1f6364767d32c6be250a017cd58f89b30d` | the ONE code commit: `scripts/cc-ng-daemon.py` + `scripts/tests/test_cc_ng_daemon_drain_pacing.py` |

## STEP 1 — READ (plain; the full line-by-line is in `build-001.md`, restated here with what changed)
1. **Does `drain_ingest_tract` honour `CC_NG_BATCH_SIZE`/`CC_NG_IDLE_STEPS`? Not at the base; yes now, but only via the daemon.** The organism function itself reads NO environment variable (LAW 5 stays with the caller): it takes `batch_nodes` and the daemon passes `CC_NG_BATCH_SIZE`. `CC_NG_IDLE_STEPS` is read by the daemon too and used for phase 2. Base facts re-verified: `drain_ingest_tract` had only `max_entries` (turns); `drain_gateway_conduit`'s batch/idle args are INERT (docstring and `:2622-2623` at the base), so `_cc_callosum_consolidate`'s only live caller was `merge_cc_topology` (`cc_topology_merge.py:609`); this pacing is NEW daemon behaviour.
2. **Nodes per turn:** the `len(graph.nodes)` delta around the atomic dual pass (the caller holds the lock, nothing steps or deletes inside the drain; content-hashed ids make an exact-repeat turn create 0). The dual pass is untouched. A turn that crosses the size is absorbed whole and the batch ends; one turn alone over the size is one batch.
3. **Reuse and lock (RULED, P473/P476):** the merge's two functions, `cc_topology_merge._unbound_nodes` (`cc_topology_merge.py:631`) and `cc_ng_organism._cc_callosum_consolidate` (`:2649`), are called by the DAEMON in phase 2, lazily imported, not copied. The drain is caller-locked and deposit-only: it acquires/releases nothing, never steps, never consolidates. The lock is an `RLock` (daemon `:831` region unchanged); the daemon's held section (acquire `cc-ng-daemon.py` trylock at the top of the section, release in `finally` at `:2328`) is unchanged in scope and order.
4. **Between-batch bookkeeping:** nothing persisted. The remainder stays in the tract file (existing partial truncate); the only cross-cycle state is `STATE.pending_consolidation` (in-memory, lost on restart, acceptable per P476(b): the tract offset protects the raw turns).

## STOP conditions
- **(e) RESOLVED** by Exec P476 (two-phase, adopted as designed in build-001).
- **(c) does not hold.** Files changed at the NG tip vs base: `cc_ng_organism.py`, `tests/test_cc_drain_pacing.py`, `handoffs/.../build-001.md` (docs). No protected file (`neuro_foundation.py`, `openclaw_hook.py`, `stream_parser.py`, `activation_persistence.py`) and no vendored file (grep over the diff name list: none). Docs repo: only `scripts/cc-ng-daemon.py` + its test.
- **(d) does not hold, with proof.** The only other call site is the VPS host `cc_ng_host.py:1526` (`drain_ingest_tract(graph, vector_db, conv_state)`), unchanged (`cc_ng_host.py` is not in the diff); with `batch_nodes`/`receipt` at their defaults the function is byte-identical to `e4ebf982` (proof below), so that site is unaffected. Leg 2 is HELD (P416): no merge arrivals exist at S4. The one shared-function change that Leg 2 can see is `_cc_callosum_consolidate`'s failure now being logged at ERROR instead of DEBUG (return value and every other behaviour identical); its only caller `merge_cc_topology` is otherwise untouched.

## What changed (file:line at each tip)
**NeuroGraph `38febe6e` — `cc_ng_organism.py`**
- Changelog entry (top of the file, dated 2026-10-01; names D24 / Exec P471-P476 / this lane; records the KNOWN LIMIT).
- `_cc_drain_size_cap` `:2333`, `_cc_drain_receipt_write` `:2344` (guarded writer: never raises; logs the exception CLASS NAME only).
- `drain_ingest_tract(graph, vector_db, state, tract_path=None, return_consumed=False, max_entries=0, batch_nodes=None, receipt=None)` `:2358`. Both new parameters optional default `None`. The size check sits after the atomic dual pass inside the existing loop (`ids_before` `:2486`, `ended = "size_reached"` `:2509`); the remainder uses the existing partial-truncate (no second mechanism). Receipt keys: `nodes_created`, `ended_on_size`, `arrivals` (set), `turns_taken`, `reason` in {`size_reached`,`entries_cap_reached`,`tract_exhausted`,`parse_failed`,`no_batch`}.
- **Why an OUT-PARAMETER receipt (my call, P476 left it to me):** the default return (`int` / `(int, bytes)`) stays byte-identical; the daemon already uses the `status=None` pattern; the receipt is reset on entry so a stale one is never read.
- `_cc_callosum_consolidate` `:2649`: P476(e) fix, `logger.error(...)` at `:2687` with the exception class name only (was `logger.debug` with `str(exc)`); `done` is initialised before the `try` so the message can report steps run. Return value, slicing and the success path unchanged.

**docs daemon `55fb0e1f` — `scripts/cc-ng-daemon.py`**
- `_read_drain_pacing_env` `:831` / `DRAIN_PACING` `:847` (reads the two EXISTING variables; BOTH must be positive integers else `None` = unpaced; no `25`/`250` literal in code, asserted by an AST test), `DaemonState.pending_consolidation` `:882`, `_warn_drain_pacing_off_once` `:2157` (ONE WARNING naming both, at the first autosave).
- `_drain_consolidate(arrivals, idle_steps)` `:2169` (phase 2), `_run_pending_consolidation()` `:2238`, `_autosave_loop` `:2256`: pending runs first at the top of the cycle (`:2269`), BEFORE the held section; the drain call passes `batch_nodes=pacing[0], receipt=drain_receipt` (`:2309`) only when paced (unpaced = the pre-D24 call, no new arguments); phase 2 arms/runs after the `finally` release (`:2331`).
- **No re-entrancy assumption (the citation you asked for):** phase 2 runs after `finally ... release()` (`:2328`). `_drain_consolidate` still refuses to run if `lock._is_owned()` (outcome `skipped_lock_held`, ERROR, pending stays armed) and its busy probe `acquire(blocking=False)` is released before the consolidation is called. I demonstrated in turn 1 that an outer RLock hold makes the slices in `_cc_callosum_consolidate` a single unsliced hold, so this guard is load-bearing.
- The guard is evaluated under `graph._step_lock` exactly as the merge does (`cc_topology_merge.py:360-365`) and RELEASED before the steps (consolidation takes `_concurrent_lock`; the established order is `_concurrent_lock -> _step_lock`). It is arrival-scoped and filters to arrivals still in the graph (a reaped node has nothing to protect).
- Which record is which for a failed consolidation: `_cc_callosum_consolidate` logs the CAUSE (class name + steps run); the daemon logs the CONSEQUENCE ("pacing RE-ARMED"). Unbound-arrival skip: the merge's own `logger.error` form with `consolidation_skipped_unbound_arrivals=<n>` (+ counter `drain_consolidation_skipped_unbound_arrivals` in `status`). Busy: rate-limited WARNING via the existing `_report_recall(kind='drain')`, code `consolidation_skipped_graph_busy`.

## Byte-identity proof (params unset == `e4ebf982`)
`tests/test_cc_drain_pacing.py::test_unset_parameters_are_byte_identical_to_the_base_module` loads the base module from `git show e4ebf982b1989fd9066d610b94853bc68bf70d37:cc_ng_organism.py` into a temp dir (path printed in the run) and runs base and tip on byte-identical copies of one tract for: whole file; `max_entries=2`; `return_consumed`; `return_consumed`+`max_entries`; empty file; missing file; and a corrupt file (parse-failure path). It asserts equal return value, equal bytes left in the file, equal node ids, equal synapse count and equal `state`. It passes. Mechanically: with the defaults `size_cap == 0` and `receipt is None`, so every added statement is a no-op (`_cc_drain_receipt_write(None, ...)` returns on its first line, `_fill_receipt` returns on `receipt is None`, the loop check short-circuits on `size_cap`).

## Runs (all under a scratch HOME, `PYTHONDONTWRITEBYTECODE=1`, no daemon, no checkpoint, no `data/` path)
**Printed preambles (P379/#770):**
- NG session: `cc_ng_organism -> /home/josh/NeuroGraph-worktrees/z12-drain-pacing-d24-20261001/cc_ng_organism.py`, `cc_topology_merge -> .../cc_topology_merge.py`, worktree root printed; a test FAILS if either resolves outside the worktree. The base module path is printed too (temp dir).
- Daemon session: `daemon under test (resolved): /home/josh/docs/.claude/worktrees/daemon-d24-env-20261001/scripts/cc-ng-daemon.py`; `real NG modules in sys.modules before/after daemon load: none`; `PASS: no real NG module loaded`; at session end `none`. `cc_ng_organism`/`cc_topology_merge` are in-process fakes (`__fake__=True`).

| Run | Result |
|---|---|
| NG `tests/test_cc_drain_pacing.py` (tip) | **23 passed** |
| NG `tests/test_cc_recall_unification.py` (tip) | **25 passed** (needs `$HOME/docs` to resolve; see "scratch HOME" below) |
| NG `tests/test_cc_drain_pacing.py` on the BASE (files from `git archive e4ebf982`, new test copied in) | **19 failed, 4 passed**. Failures: `TypeError ... unexpected keyword argument 'batch_nodes'` (feature missing) and, for the consolidate test, "swallowed below WARNING" (the DEBUG log). The 4 passes are the identity guards that must pass on both: the worktree-path test, `test_drain_default_return_types_unchanged`, `test_consolidate_success_path_is_unchanged`, and `test_unset_parameters_are_byte_identical_to_the_base_module` (on the base it compares the base against itself). |
| docs `scripts/tests/test_cc_ng_daemon_drain_pacing.py` (tip) | **27 passed** |
| docs `scripts/tests/test_cc_ng_daemon_recall_status.py` (tip) | **41 passed** (unchanged file) |
| docs `test_cc_ng_daemon_drain_pacing.py` on the BASE daemon (`git show e60524154`, via `Z12_D24_DAEMON_UNDER_TEST`) | **25 failed, 2 passed** (missing `_read_drain_pacing_env`/`_drain_consolidate`/`pending_consolidation`, no warning, no pacing args; the 2 passes are identity guards: the held section is unchanged at the base, and no spurious warning in paced mode) |

**Mutation checks (tests are not vacuous):** daemon copies broken on purpose, each caught: consolidating inside the held section (9 tests fail), dropping the pending-first rule (1), whole-graph guard instead of arrival-scoped (2), re-arming on an unbound skip (1), not re-arming on failure (2). NG: `>` instead of `>=` (1 fails), arrivals = all nodes (3), consolidation failure still at DEBUG (1), size check at loop top which loses a turn (7). One mutant (a pre-check placed ahead of the post-check) survived and I confirmed it is EQUIVALENT (the post-check always fires first), not a gap.

**End-to-end seam check (ad hoc, NOT committed, `/tmp/d24_seam_check.py`):** the real daemon `_autosave_loop` + real `drain_ingest_tract` + real `_unbound_nodes` + real `_cc_callosum_consolidate` on a real in-memory `neuro_foundation.Graph` (embedder and dual pass stubbed; tract in a temp dir). 4 turns x 10 nodes, `CC_NG_BATCH_SIZE=25`/`CC_NG_IDLE_STEPS=250`: cycle 1 absorbed 3 turns (30 nodes) then ran 250 steps; cycle 2 absorbed the 4th (10 nodes) then ran 250; final timestep 500, tract empty, `pending_consolidation=None`; the autosave thread did NOT own the lock at either consolidation (`[False, False]`). This proves the daemon<->organism seam (kwargs, receipt keys, arrivals), which the two unit suites cover only with a fake on one side.

## Decisions I made that you should look at
1. **Consolidation runs after ANY non-empty batch, not only size-ended ones** (mirrors the merge's `if idle_steps > 0 and merge_landed`, `cc_topology_merge.py:597`). Consequence in steady state: a cycle that lands even one turn's nodes costs 250 idle steps (and the next cycle's drain waits for them if the graph was busy). If the intent was 250 steps only after a FULL 25-node batch, that is a one-line condition (`receipt["ended_on_size"]`); I did not narrow it because P476 says "after each batch".
2. **An unbound-arrival skip is NOT re-armed** (the merge's own semantics; re-arming would re-check unbindable nodes forever and stall the drain, the very failure P476(d) names). So after an unbound skip the next batch is taken without idle steps between: loud ERROR, integration quality only.
3. A pending set is filtered to arrivals still in the graph before the guard (a reaped node cannot protect anything).
4. Phase-2's busy WARNING reuses `_report_recall(kind='drain')`, whose text reads "drain failure reported: code=consolidation_skipped_graph_busy", slightly odd for a skip but the code is explicit and it gets the rate limit and `status` counter for free.

## KNOWN LIMIT (recorded as P476(d) requires; also in both code changelogs)
The guard is arrival-scoped, so the 250 idle steps CAN age a Leg 2 merge's unbound arrivals. Leg 2 is HELD (P416): none exist at S4. When Leg 2 resumes, merge and drain consolidation MUST be coordinated (one consolidator at a time, or a combined arrival set): recorded on #876 and the #806 post-track lane. The ~806 pre-existing laptop-unbound count is the zone manager's read-only re-check, not mine.

## Findings outside this task (for the punchlist; I did not touch them)
1. **Silent-failure class that remains at the daemon drain site:** the `try` around `drain_ingest_tract`/probation/wants/emergent-want still swallows every exception at DEBUG (`cc-ng-daemon.py`, the `ingest-tract/probation/surface_wants/emergent_want failed (non-fatal)` line). A kwarg `TypeError` from an older organism (merge-order mistake) would stop the drain with no visible trace. Recommend WARNING + class name only.
2. `tests/test_cc_refeed.py::test_feed_batch_absorbed_by_real_drain` FAILS on the base too: `cc_refeed.py:207` calls `ng_tract.deposit_experience(content=...)`, which the installed `ng_tract` rejects (`unexpected keyword argument 'content'`; the newer form is `raw=`/`tract_paths=`). The same stale call style is in `tests/test_cc_dual_pass.py`. Real production-code defect if the installed ng_tract is the live one.
3. `tests/test_cc_deposit_step.py` HANGS (killed at my 240 s cap; same on the base): the last test, `test_autosave_loop_drains_the_tract_under_concurrent_lock_643`, never returns (likely the #754 class). Also `test_drain_ingest_tract_never_steps` fails with `0 == 2` under a scratch HOME on both base and tip (the embedder model is not cached there, so the real `embed` yields nothing); I ran the other drain tests in that file in isolation: no new failures.
4. `tests/test_cc_recall_unification.py` loads `~/docs/scripts/cc-ng-daemon.py` through `expanduser`, so under a scratch HOME it ERRORS (7 errors) unless `$HOME/docs` exists. I ran it with `$HOME/docs` symlinked to the DAEMON WORKTREE (not the primary checkout) and it passed 25/25. The test silently depends on the primary docs checkout otherwise.
5. `cc_topology_merge.py:277` docstring cites `cc_ng_organism.py:1856` for `_cc_callosum_consolidate`; it is at `:2649` now (stale pointer).
6. Observation from the seam check: with nothing driving firing, 250 idle steps on a quiet synthetic graph tripped the substrate's own zero-fire breaker ("Emergency excitability boost applied to 30 nodes"). I did not investigate whether that is expected on the live CC graph; flagging it because the pacing makes idle-step runs routine.
7. Edge: a tree-less, window-less first turn right after a daemon restart (`last_forest_id` empty) leaves its forest node with no synapse and no hyperedge, so the arrival-scoped guard will (correctly, loudly) skip that batch's steps.

## NOT verified
- No run against the live daemon, live graph, real tract, TID or Syl's process (forbidden); the real dual pass and real embedder were stubbed everywhere. `ng_embed.dual_record_outcome`'s tree count was not executed.
- The wall-clock cost of 250 idle steps on the real ~16k-node CC graph (the seam check used 40 nodes), and how long hooks wait between 25-step lock slices at that size.
- The NG full suite (forbidden, #754), the other tests touching the drain beyond those listed above, and `cc_ng_host.py`'s other lock acquisitions.
- `orphan_node_grace_period` for the CC graph (taken from the merge's comment, not config).
- Nothing was run under the primary HOME; the S4 preflight line asserting both variables is plan-002's, not mine.
