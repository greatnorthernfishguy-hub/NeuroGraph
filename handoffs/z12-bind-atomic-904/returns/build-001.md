<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 fresh builder, lane dual-pass-atomic-904, dispatch #13058) — build-001 return for #904
#   What: the BUILD return for #904 (Exec Packet 487 via Chief-003, S4-GATING) plus Exec Packet 489 (surface_wants swallow) and Exec Packet 491 (the #257 write-path family: the persisted sequence pointer, every bind site loud and truthful, a born-unbound forest announced).
#   Why: #898 found the 147 conversational nodes were never bound: the local dual-pass write is non-atomic and silent, and every caller ignored the False.
#   How: ONE code commit per repo (NG and the docs daemon), local only, nothing merged; this file is the docs-only return commit in the NG worktree.
# -------------------
-->

# #904 build-001 — the dual-pass write path is atomic, loud and truthful; every caller honours the False

Links: [[NeuroGraph]] · [[Dual-Pass Embedding]] · [[The River]] · [[NG Tract Bridge]] · assignment `build-904-dual-pass-atomic.md` (docs `ede8b2e6` addendum 1, `887fa102` addendum 2) · analysis [[analysis-898-unbound-147-cause]] (docs `a14a2002`) · #898b `analysis-898-second-pass.md` (docs `656d5f77`).

## READ FIRST

**Hashes (real `git rev-parse HEAD`, all local, no upstream, nothing merged, no PR):**

| Repo / branch | Base | Code commit | |
|---|---|---|---|
| NG `cc-laptop-bind-atomic-904-20261001` (`/home/josh/NeuroGraph-worktrees/z12-bind-atomic-904-20261001`) | `c568433434a99154b9f7652b77cbe43752d34324` (the #794 tip) | `895a809a60ab86afacd8ef193b95c4ab39f6893f` | `cc_ng_organism.py`, `tests/test_cc_bind_atomic_904.py` |
| docs daemon `cc-laptop-daemon-deposit-904-20261001` (`/home/josh/docs/.claude/worktrees/daemon-deposit-904-20261001`) | `e2c8092bf54a1b9247f1316b9ab2107779187ce1` (the D24 tip) | `6b5555022b6a31dcca417a6e24e4fcaa92c2d150` | `scripts/cc-ng-daemon.py`, `scripts/tests/test_cc_ng_daemon_deposit_904.py` |
| NG, this return | | the docs-only commit that adds this file (see `git log -1` on the NG branch) | |

**One paragraph.** The dual pass is now **ATOMIC**: any exception after the first write rolls back exactly what that call wrote (synapses, the hyperedge, the nodes it created, the vdb entries it inserted fresh), restores the caller-owned pointer, logs ONE WARNING (ERROR if the rollback itself failed) naming the stage, the failing bind site and the counts with the exception CLASS name only, and returns a **truthful False**; a bind that wrote less than a turn with trees needs is also False. The daemon `_deposit` now REPORTS a False/raise (counter + rate-limited class-name-only WARNING); the daemon passes `hold_on_failure=True` to the drain **only when `CC_NG_DRAIN_HOLD_ON_FAILURE` is clearly truthy at start (absent = off, no default flipped)**. The **#805 temp-graph retry-safety proof PASSES**, including the case a LOUD-only fix cannot survive (the retry extracts DIFFERENT concepts). Addendum 1 (wants) and addendum 2 (persisted sequence pointer, every bind site loud, born-unbound warning) are folded in. **24 of 24 mutations are KILLED.** Existing files I ran show the SAME results on base and fix (identical failing sets where they fail for pre-existing reasons).

**Five things the Chief should read before anything else (all detailed below):**
1. **Do not set `CC_NG_DRAIN_HOLD_ON_FAILURE` before an NG that has `hold_on_failure` is the one running.** The D24 NG branch (`24a497dd`) has `batch_nodes`/`receipt` but NOT `hold_on_failure`; this NG branch has `hold_on_failure` but NOT `batch_nodes`/`receipt`. A daemon with the flag on against an NG without it raises `TypeError` inside the autosave section, which the section's outer `except` logs at **debug**: the drain would stop silently every cycle. NG (both) must merge before the flag is set (consequence 2).
2. **The brief contradicts itself on stepping** ("unchanged" vs "do not step on a failed dual pass"). I kept it UNCHANGED (a standing ruling, B3/001: a failed turn is still a timestep; no reward without `ingested`). One-line option for the Chief (consequence 5).
3. **The daemon keeps no counter for "the drain held a failed turn"**: neither the drain's return nor the D24 receipt carries a hold, and inferring one would be a second mechanism. The hold IS reported, loudly, once per cycle, by the drain's own #794 WARNING. A countable signal needs a receipt reason set at the source, which belongs to the #794/D24 merge (consequence 4).
4. **With the flag OFF (today's default) a failed turn is still truncated and lost** (the #794 default; LAW 7). What changed is that it now leaves NOTHING half-written behind. The flag is what keeps the bytes.
5. **Two adjacent silent failures I did not touch** (punchlist, below): the autosave section's outer `except Exception: logger.debug(...)` swallows ANY exception in the whole section, and `_cc_deposit_memory_node` logs `str(exc)` at WARNING.

## What changed (file:line, final code)

### NG `cc_ng_organism.py` (one commit)
- `_CCConversationalDualPassEco` (`:2035`): a **write journal** (`writes`, `topology`), `deposit()` (`:2051`: `_cc_deposit_memory_node`, journalled BEFORE the write so a write that raises midway is still journalled), `counts()` (`:2066`), `rollback()` (`:2074`). The journal lives on the eco class, not in a new module-level helper, because `tests/test_cc_capture_mutations_423.py` extracts exactly these five names by AST into a bare namespace. `_cc_deposit_memory_node` is **unchanged** (that file pins its "no fabricated rollback" contract).
- `_cc_bind_conversational_topology` (`:2132`): **nothing is swallowed any more** (the forest<->tree pair, the window synapses, the hyperedge, the window delay-chain, the sequence link all raise; a forest absent at the bind raises instead of returning silently). It tags the write SITE as it goes (`journal["site"]`, `journal["attempts"]`), journals every synapse/hyperedge id it creates, and **returns a status dict** (`trees`, `tree_pairs`, `window_pairs`, `hyperedge`, `window_chain`, `sequence`, `has_trees`, `has_windows`). Optional keyword `journal=None`; with `None` it is byte-identical on every success path. **Born-unbound** (Packet 491 (iii), `:2220`): ONE WARNING for a forest this call CREATED with no predecessor, no trees and no windows (`reason=forest_born_unbound`, the three booleans, no id/text).
- `run_conversational_dual_pass` (`:2348`): snapshot of `state["last_forest_id"]`/`["primed_nodes"]`, stage tracking (`dual_record` | `windows` | `bind`), the **truth rule** (`:2428`), the rollback, the state restore, the per-site counter `state["bind_site_failures"]` (`:2453`), and the one loud record. A failure BEFORE the first write (nothing to undo, e.g. TID down: `DualPassIncompleteError`) is unchanged: debug line, False.
- `surface_wants` (`:1611`, counter `:1667`) and `surface_wants_for_graph` (`:1215`, counter `:1276`) + `_cc_report_want_synapse_failures` (`:1199`): Packet 489.
- `drain_ingest_tract`, `ng_embed.py` and every protected file: **untouched** (a test pins `git diff --name-only` for the five).

### Docs daemon `scripts/cc-ng-daemon.py` (one commit)
- `_deposit` (`:2090`, report call `:2132`): `False` → code `dual_pass_false`, exception → `dual_pass_raised`, through the EXISTING `_report_recall(None, code, exc, kind='deposit')` (counter `STATE.stats['deposit_fail_<code>']`, WARNING rate-limited by the EXISTING `RECALL_WARN_INTERVAL_S`, class name only), made AFTER the graph lock is released (`_report_recall` takes `STATE.lock`). `stats['errors']`, the Commons deposit, the success path and `step=True` are unchanged.
- `_read_drain_hold_env` (`:959`) and `DRAIN_HOLD_ON_FAILURE` (`:980`): `CC_NG_DRAIN_HOLD_ON_FAILURE` read ONCE at start. `1/true/yes/on` = True; unset/empty/`0/false/no/off` = False; anything else = False + ONE WARNING naming the variable only. No other new environment variable (a test diffs the string constants against `e2c8092b`).
- `_autosave_loop` (`:2618`): `_hold_kw = {'hold_on_failure': True} if DRAIN_HOLD_ON_FAILURE else {}` spread into BOTH drain calls (unpaced and D24-paced). Flag off ⇒ the kwarg is NOT passed ⇒ the call is byte-identical to the D24 tip (the D24 tests' fake drain has no such parameter, and a drain that predates #794 keeps working).
- Packet 491 (i), the sequence pointer: `_CONV_POINTER_FILE` (`:748`), `_write_conv_pointer` (`:1217`), `_load_conv_pointer` (`:1233`), the pre-save read (`:1318`) and the single write (`:1345`) inside `_guarded_save`, the restore in `init_ng` (`:1110`).

## ATOMIC, not LOUD, and why
A retry re-extracts a DIFFERENT concept set (LLM output) and the bind covers only that attempt's `tree_ids`, so a kept partial write leaves stragglers on every retry (#898 bucket 5). A LOUD-only fix cannot satisfy the #805 proof unless the test pretends extraction is deterministic. ATOMIC needed no edit to `ng_embed.py` (vendored) or a protected file: it uses the public `Graph.remove_synapse` / `remove_hyperedge` / `remove_node` and `SimpleVectorDB.delete`.

**Exactly what rollback does and does not do.** It removes only what THIS call created: the synapse ids and hyperedge ids the bind reported, the nodes whose id was absent before the write, and the vdb entries that were absent before. A node that existed before the call is never removed (tested with an exact-repeat turn whose retry extracts different concepts). One effect is NOT undone: a PRE-EXISTING node's idempotent re-stamp (its probation counters/threshold, and its vdb entry re-written with identical content), which a successful repeat of the same turn does anyway. A vdb entry whose graph node was gone (the 5,757 orphans of #898) is not deleted. If the rollback itself fails the record is an ERROR that says "ROLLBACK FAILED, a partial write REMAINS" and the function still returns False (tested: the partial write really is still there). So **in the ATOMIC option a window-only or tree-only partial cannot remain** except when the rollback itself fails.

## The Packet 491 choices
- **(i) pointer.** Not inferred from the graph. Mechanism = the one this daemon already uses beside the checkpoint (`_HEALTHY_REF_FILE` written inside `_guarded_save`): a tiny JSON file `<checkpoints>/.cc_conv_last_forest`, written with tmp + `os.replace`, ONLY after a successful save (not on a refused save, not on a non-accepted receipt), read BEFORE the save (so it names a forest that existed when the save began), exactly ONE writer (an AST test pins it), restored in `init_ng`, validated on load (missing / corrupt / non-string / stale id / unverifiable ⇒ `None` + ONE WARNING with a fixed reason code and the class name only, never a crash). **No protected file and no checkpoint-format change were needed, so no STOP.** The pointer is only ever moved by the bind and only after all of a turn's writes succeeded; a rolled-back turn restores it (tested: a failed turn does not move the persisted pointer). The VPS host `cc_ng_host.py` (~`:512`) has its own per-process dict: Josh-gated, not edited, reported.
- **(ii) truth rule.** A turn that HAS trees must come out with **every** forest<->tree synapse pair **and** its hyperedge, else the dual pass raises inside its own try (site `bind_postcondition`), rolls back, returns False with its own record. All-or-nothing, not "any one tree pair": a bind that dropped most of its trees must not pass. Fail-fast rather than continue-and-count: after the first failed write the turn is rolled back anyway. The rule is a second layer behind the raising bind (it trips only if a write returns without writing, or a swallow is ever restored; that is exactly what N7/N11 test).
- **(iii) born unbound.** Only for a forest this call CREATED. Expected volume with (i) in place: **one per true first forest ever** (or a pointer to a vanished forest). Whether the laptop daemon has a reachable TID (which decides how many tree-less, window-less forests it writes at all) was NOT examined (the #898 analysis did not either); with no TID nothing is deposited, so no forest and no warning.

## Caller audit (every caller of `run_conversational_dual_pass`, both repos)

| Caller | What it did with the `False` BEFORE | Now |
|---|---|---|
| docs `cc-ng-daemon.py::_deposit` (`:2090`) | dropped it (only fed `cc_deposit_step`); its `except` was `logger.debug` | **FIXED**: counter + WARNING (class only), reported after the lock is released |
| `cc_ng_organism.drain_ingest_tract` via `_apply_gateway_experience` (`:2698`) | counted it as not-absorbed, then (flag off) truncated the turn's bytes | unchanged by default (#794); **held** when the daemon passes `hold_on_failure=True` |
| docs daemon drain call site (`_autosave_loop`) | called the drain with no flag | **FIXED**: passes the flag from `CC_NG_DRAIN_HOLD_ON_FAILURE` |
| `cc_ng_organism.drain_gateway_conduit` via `_apply_gateway_experience` (`:3070`, Leg 1 VPS conduit) | already honours it: a False raises `RuntimeError('dual-pass did not confirm full application')`, the file is marked `uncertain` and retained | unchanged; its retries are now clean because a failed record leaves nothing behind |
| `cc_ng_host.py::_deposit` (`:664`, the VPS half) | drops it (only feeds `cc_deposit_step`); `except` = `logger.debug` | **REPORT ONLY** (Josh-gated, not edited): the same silent shape as the laptop daemon had; it also keeps its own per-process `conv_state` (the sequence-pointer defect) |
| `neurograph_rpc._run_conversational_dual_pass` (Syl's canonical, a SEPARATE implementation) | returns a bool; its caller enqueues a retry on False | not this function, not touched. **Not verified:** whether its `_bind_conversational_topology` carries the same swallow pattern (the CC port was extracted from it): worth a read-only look |
| `cc_refeed.py` | docstring mention only (re-feeds through `drain_ingest_tract`) | none |

## Runs (scratch `HOME=/tmp/z12-904-scratch/home`, `PYTHONUSERBASE=/home/josh/.local`, `HF_HUB_OFFLINE=1`, `-p no:cacheprovider`, load gate < 6 before every run)
Final code, one run per file. "BASE" = the base module/daemon (`c5684334` NG, `e2c8092b` daemon) via a scratch copy (`Z12_904_ORG_UNDER_TEST` / `Z12_904_DAEMON_UNDER_TEST`) or a `git archive` export of the base tree. P379/#770: the preamble of both of my files printed every resolved module path; every NG module resolved under THIS worktree (the scratch org under `/tmp/z12-904-scratch/` for base/mutation runs only), the docs file loaded no real NG module.

| File | FIX | BASE |
|---|---|---|
| NG `tests/test_cc_bind_atomic_904.py` (with the real daemon loaded by `Z12_904_DAEMON_UNDER_TEST`) | **44 passed** | **32 failed, 12 passed** (base org + base daemon). The 12 are the tests that pin what must NOT change: signatures, `ng_embed.py`/protected files unedited, the success-path golden, the pre-write failure, no counter/state key on success, four "no born-unbound warning" cases, the two want goldens, `generate_emergent_want` untouched. All 32 failures are the defects: e.g. `bind_hyperedge` fails with `assert True is False` (base returns **True** after a half-bound turn), the tree/window cases leave the forest and trees in the graph, the wants cases show no record above debug, the restart round trip fails with `assert None == 'cc:conv::...'` |
| NG `tests/test_cc_bind_atomic_904.py` WITHOUT the daemon path | 40 passed, 4 skipped (the four daemon-gated tests: two daemon-loop, two restart; skipped with the reason printed) | n/a |
| NG `tests/test_cc_drain_hold_on_failure.py` | **49 passed** | 27 passed, 22 failed on the exported base tree: all 22 are `fatal: not a git repository` (the file shells out to `git show e4ebf982`, and a `git archive` export has no `.git`): an artefact of the export, not a base failure. The same file passes 49/49 on the fix branch (a git repo) |
| NG `tests/test_cc_drain_pacing.py` | not present on this base | |
| NG `tests/test_cc_capture_mutations_423.py` (the AST-extraction harness I designed around) | **10 passed** | 10 passed |
| NG `tests/test_cc_want_bounds.py` (real `surface_wants`) | **8 passed** | 8 passed |
| NG `tests/test_cc_gateway_durable.py` | 28 passed | 28 passed |
| NG `tests/test_cc_durable_integration.py` | 3 passed | 3 passed |
| NG `tests/test_cc_host_stop_door.py` (real `_deposit` -> `surface_wants_for_graph`) | 4 passed | 4 passed |
| NG `tests/test_conversational_window_chains.py` | 4 passed | 4 passed |
| NG `tests/test_cc_callosum_leg1.py` | 16 passed | 16 passed |
| NG `tests/test_cc_pattern_completion_recall.py` | 1 failed, 7 passed | 1 failed, 7 passed: **identical failing test** |
| NG `tests/test_cc_retrieval_enrichment.py` | 2 failed, 14 passed | 2 failed, 14 passed: **identical failing tests** (`ImportError: cannot import name 'cc_gsg_backfill'` and one assertion). With the shell's `NG_EMBED_REMOTE=hf` set it is 7 failed on BOTH (`HF token unavailable`) |
| NG `tests/test_cc_dual_pass.py` | 10 failed, 11 passed | 10 failed, 11 passed: **identical 10 failing tests** (TID-class: the dual pass returns False without TID extraction; plus `deposit_experience(content=...)` against an API that takes `raw=`) |
| docs `scripts/tests/test_cc_ng_daemon_deposit_904.py` | **62 passed** | **52 failed, 10 passed** (base daemon). The 10: success silent, `step=True` unchanged, a raising reporter never breaks the hook, the flag-off call and event sequence, the flag read per cycle, and four "writes nothing when ..." pointer tests that hold trivially on base |
| docs `scripts/tests/test_cc_ng_daemon_drain_pacing.py` | 46 passed with `CC_NG_BATCH_SIZE=25 CC_NG_IDLE_STEPS=250`, 46 passed with both unset | 46 / 46 |
| docs `scripts/tests/test_cc_ng_daemon_recall_status.py` | 41 passed set, 41 passed unset | 41 / 41 |

Not run, as instructed: the NG full suite and `tests/test_cc_deposit_step.py` (hangs). TID-class tests are "not run" in the sense that they cannot pass without TID, base or fix. **Load gate (< 6):** every run was gated; it tripped ONCE at 6.07 (other sessions) before a regression file and I stopped and waited for it to clear instead of running through; the figures printed beside the background runs are read AFTER each run.

## Mutations (scratch copies; each replacement asserted to match EXACTLY once; `/tmp/z12-904-scratch/mutate.py`)
| Verdict | Mutation | Result | Killed by (first two) |
|---|---|---|---|
| KILLED | N1 failure logged at debug again | 14 failed, 30 passed in 4.32s | 805_proof_a_turn_that_fails_every_cycle_stays_byte_identical_and_, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_not_report |
| KILLED | N2 returns True after a partial write | 24 failed, 20 passed in 4.10s | 805_proof_a_turn_that_fails_every_cycle_stays_byte_identical_and_, 805_proof_failed_turn_then_a_retry_with_different_concepts_equals |
| KILLED | N3 rollback skipped | 21 failed, 23 passed in 4.40s | 805_proof_failed_turn_then_a_retry_with_different_concepts_equals, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_not_report |
| KILLED | N4 vdb entries not deleted on rollback | 19 failed, 25 passed in 4.53s | 805_proof_failed_turn_then_a_retry_with_different_concepts_equals, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_not_report |
| KILLED | N5 state pointer not restored | 3 failed, 41 passed in 3.24s | a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_not_report, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_not_report |
| KILLED | N6 forest-absent returns silently again | 1 failed, 43 passed in 3.13s | partial_write_is_rolled_back_loudly_and_truthfully[bind_forest_ab |
| KILLED | N7 the post-condition (truth rule) removed | 3 failed, 41 passed in 4.15s | a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_not_report, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_not_report |
| KILLED | N8 site counter dropped | 10 failed, 34 passed in 3.68s | a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_not_report, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_not_report |
| KILLED | N9 born-unbound warning dropped | 1 failed, 43 passed in 2.48s | a_first_forest_with_no_predecessor_no_trees_no_windows_logs_one_b |
| KILLED | N10a swallow back: tree synapse | 1 failed, 43 passed in 4.88s | partial_write_is_rolled_back_loudly_and_truthfully[bind_tree_syna |
| KILLED | N10b swallow back: window synapse | 1 failed, 43 passed in 3.94s | partial_write_is_rolled_back_loudly_and_truthfully[bind_window_sy |
| KILLED | N10c swallow back: hyperedge (debug) | 2 failed, 42 passed in 2.45s | partial_write_is_rolled_back_loudly_and_truthfully[bind_hyperedge, the_site_counter_accumulates_per_site |
| KILLED | N10d swallow back: window chain | 1 failed, 43 passed in 2.62s | partial_write_is_rolled_back_loudly_and_truthfully[bind_window_ch |
| KILLED | N10e swallow back: sequence link | 1 failed, 43 passed in 3.32s | partial_write_is_rolled_back_loudly_and_truthfully[bind_prev_link |
| KILLED |  | N11 True after a failed tree-bind (swallow tree site AND no post-condition) 4 failed, 40 passed in 3.41s | a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_not_report, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_not_report |
| KILLED | N12 the drain truncates a failed entry with the flag on | 4 failed, 40 passed in 3.61s | 805_proof_a_turn_that_fails_every_cycle_stays_byte_identical_and_, 805_proof_failed_turn_then_a_retry_with_different_concepts_equals |
| KILLED | N13a want swallow back: surface_wants | 2 failed, 42 passed in 2.85s | ], ] |
| KILLED | N13b want swallow back: surface_wants_for_graph | 2 failed, 42 passed in 2.79s | ], ] |
| KILLED | D1 _deposit ignores the False | docs[3 failed, 59 passed in 7.51s] ng[44 passed in 2.42s] | a_false_from_the_dual_pass_is_reported_counted_and_warned, the_report_is_made_after_the_graph_lock_is_released |
| KILLED | D2 the daemon never passes the flag | docs[2 failed, 60 passed in 4.55s] ng[1 failed, 43 passed in 2.73s] | flag_on_the_drain_is_called_with_hold_on_failure_true[None-expect, flag_on_the_drain_is_called_with_hold_on_failure_true[pacing1-exp |
| KILLED | D3 pointer persisted but NOT loaded | docs[4 failed, 58 passed in 6.24s] ng[1 failed, 43 passed in 3.09s] | init_ng_restores_the_persisted_pointer_into_the_fresh_state_dict, init_ng_with_a_missing_stale_or_corrupt_pointer_starts_with_none_ |
| KILLED | D4 pointer loaded WITHOUT the id-exists check | docs[2 failed, 60 passed in 7.46s] ng[44 passed in 2.51s] | a_stale_id_not_in_the_graph_is_none_with_one_warning_and_the_id_n, init_ng_with_a_missing_stale_or_corrupt_pointer_starts_with_none_ |
| KILLED | D5 the default is flipped on in code (absent = True) | docs[2 failed, 60 passed in 7.59s] ng[1 failed, 43 passed in 4.30s] | hold_env_clear_values[env0-False], the_constant_is_read_once_at_start[None-False-0] |
| KILLED | D6 pointer read AFTER the save (runs ahead of the saved graph) | docs[1 failed, 61 passed in 6.83s] ng[44 passed in 2.50s] | guarded_save_reads_the_pointer_before_the_save |

**24 of 24 KILLED.** The brief's list is all covered: failure logged at debug again (N1), True after a partial write (N2), `_deposit` ignores the False (D1), the daemon never passes the flag (D2), the drain truncates a failed entry with the flag on (N12), the rollback skipped (N3); addendum 1: a want swallow back (N13a/b); addendum 2: pointer persisted but not loaded (D3), loaded without the id-exists check (D4), each of the five bind swallows back to `pass`/debug (N10a-e), the dual pass True after a failed tree-bind (N7, N11), the born-unbound warning dropped (N9); plus the default flipped on in code (D5), the pointer read after the save (D6), state not restored (N5), vdb entries left behind (N4), the forest-absent silent return (N6) and the counter dropped (N8). Note on N10a/N10c: with the post-condition behind the bind, a restored swallow at the tree or hyperedge site is still caught as False, so those two are killed by the per-site assertions (`site=tree_synapse` / `site=hyperedge`), not by a returned True.

## The #805 retry-safety PROOF: PASSES (`tests/test_cc_bind_atomic_904.py`, on a REAL in-memory `Graph` + a REAL `SimpleVectorDB` + the REAL vendored `NGEmbed.dual_record_outcome` + the REAL `drain_ingest_tract` on real tract bytes in tmp files)
- `test_805_proof_failed_turn_then_a_retry_with_different_concepts_equals_one_clean_pass`: three turns in the tract, `hold_on_failure=True`. Cycle 1: turn 2's dual pass fails partway (a tree write raises AFTER the forest, one tree and the windows were written). Result: turn 1 truncated, turn 2 and 3 byte-identical in the tract, the graph is exactly "turn 1 only". Cycle 2: the fault is gone AND the extractor returns DIFFERENT concepts for turn 2. Result: the tract is empty and the final graph equals ONE CLEAN PASS of the three turns (same node set, no orphan/straggler, no duplicate forest/tree/window, same synapses and hyperedge set, none of the abandoned first attempt's trees).
- `test_805_proof_a_turn_that_fails_every_cycle_...`: a turn that fails EVERY cycle stays in the tract byte-identical (with everything after it) for 4 cycles, leaks nothing into the graph, and logs exactly TWO WARNINGs per cycle (one from the dual pass, one from the drain hold); then recovers when the fault clears. **Volume:** the held entry is retried, re-embedded and re-extracted EVERY cycle: at the daemon's 60 s autosave that is 2 WARNING lines a minute (about 2,880 a day) plus one full dual-pass attempt a minute, for as long as an entry is held.
- `test_daemon_loop_with_the_flag_on_holds_...`: the same on the REAL daemon `_autosave_loop` (loaded by path) over real NG: the Leg 1 conduit receives ONLY the bytes that left the file, never the held turn.
- `test_flag_off_truncates_a_failed_turn_as_before`: the #794 default is unchanged.
Not a proof of: a real embedder, a real TID, a real daemon process (see "Not verified").

## Consequences (stated, not decided)
1. **WARNING volume of a held entry:** 2 lines/min while held (above). It is loud on purpose; a poison turn (one that can never succeed) would hold the whole tract behind it forever. That is the hold's defined behaviour (#794), and why the flag stays off until the preflight.
2. **What the S4 preflight must set:** `CC_NG_DRAIN_HOLD_ON_FAILURE=1` in the daemon's environment (`.bashrc`/unit), once, AFTER confirming the NG the daemon imports has `hold_on_failure` (merge both NG branches first: this one and D24). With the flag on and an NG without it, the drain `TypeError`s every cycle and the autosave section's outer `except` logs it at debug (silent). Nothing else needs setting; no default was flipped in code.
3. **ATOMIC removes the #896 wedge's partial-failure cause:** in a paced cycle a turn that failed after depositing nodes used to leave unbound nodes that blocked the 250 idle steps AND every later batch (the D24 seam test pins that wedge); a rolled-back turn leaves none. (A turn that fails before its first write never did.)
4. **No daemon counter for "held a failed turn"** (see READ FIRST 3). Needed at the source: a receipt `reason` (e.g. `held_failed_turn`) set by `drain_ingest_tract` when it stops on a failure; if the two NG branches are merged as they stand, a held drain would report `reason="tract_exhausted"` (the D24 `ended` default): inferred from reading both, not run (they were not merged). A decision for the #794/D24 merge.
5. **Stepping:** `cc_deposit_step(graph, ingested)` still runs on the Stop-side door with `ingested=False` after a failed pass (B3/001). If the Chief wants it not to step on a failure, it is a one-line change in `_deposit` (`if step and ingested:`); it is clock behaviour, so not mine to change.
6. **A failure now FAILS the turn where it used to silently half-bind:** the bind's swallows are gone. On the real graph they cannot fire (nodes are checked under the same lock; self-links are excluded), so no success path changes (the golden pins return value, graph, vdb, synapses, hyperedge). With the flag OFF, such a turn is dropped (rolled back + truncated) instead of kept half-bound; with the flag ON it is held and retried.
7. **The restore warns once on a fresh start** (a missing file is a WARNING by the brief). One line, once.
8. **Born-unbound, counter and pointer volumes** are above; none is per-turn on the success path.

## Defects noted outside my task (for the punchlist; not touched)
- The autosave section's outer `except Exception as exc: logger.debug('ingest-tract/probation/surface_wants/emergent_want failed (non-fatal): %s', exc)` swallows ANY exception from the whole section (drain, probation, wants, emergent want) at debug, with `str(exc)`.
- `_cc_deposit_memory_node` logs `logger.warning("CC recall insert failed: %s", exc)`: `str(exc)` at WARNING (pre-existing; my new records never carry it).
- `surface_wants_for_graph`'s outer `except Exception: logger.debug("Failed to create want node: %s", exc)` (a failed `create_node`) and `surface_wants`' unguarded `create_node` (a raise aborts the loop, so later wants are not materialized that pulse) remain.
- `cc_anticipate` swallows its own exception at debug (it only sets `state["primed_nodes"]`, a hint, not topology).
- Two existing tests are stale against the base: `tests/test_cc_dual_pass.py` calls `ng_tract.deposit_experience(content=...)` (the installed API takes `raw=`), and `tests/test_cc_retrieval_enrichment.py` imports `cc_gsg_backfill`, which the base organism does not define. Both fail IDENTICALLY on base.
- `NG_EMBED_REMOTE` is set in this shell (`.bashrc`), which forces remote HF embedding; offline with no token every real-embedder test fails with `HF token unavailable`. I unset it only for the regression runs (never touched or seeded the hub token file).

## Not verified
- Anything with the REAL embedder or a REAL TID: every test of mine replaces only the model/TID seams on the vendored `NGEmbed` instance (`_extract_concepts`, `embed_batch`, `embed_windows`); the real extraction path was not run (`tests/test_cc_dual_pass.py` needs TID: identical failures on base and fix).
- A real daemon process, a real checkpoint save/restore (the restore is tested through the real `DaemonState.init_ng` and the real `_guarded_save` with a fake `NeuroGraphMemory` and tmp paths), the live graph/tract/checkpoint (never opened), the VPS half, Leg 2, the S4b gate.
- Whether the hold interacts correctly with D24 pacing in the MERGED NG (the two NG branches were not merged; the daemon-side paced call is tested with a fake drain only).
- `tests/test_cc_deposit_step.py` (hangs, #754) and the NG full suite were not run, as instructed.
- Syl's `neurograph_rpc._bind_conversational_topology` (not read).

## Reproduce
`cd <NG worktree> && HOME=/tmp/z12-904-scratch/home PYTHONUSERBASE=/home/josh/.local HF_HUB_OFFLINE=1 Z12_904_DAEMON_UNDER_TEST=<docs worktree>/scripts/cc-ng-daemon.py python3 -m pytest -p no:cacheprovider -q tests/test_cc_bind_atomic_904.py` (add `Z12_904_ORG_UNDER_TEST=<scratch copy of cc_ng_organism.py>` to run against the base or a mutation). Docs: `python3 -m pytest -p no:cacheprovider -q scripts/tests/test_cc_ng_daemon_deposit_904.py` (`Z12_904_DAEMON_UNDER_TEST=<scratch daemon>` likewise).
