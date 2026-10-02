<!--
# ---- Changelog ----
# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane in-transit-hold-905d, dispatch #14776 + #14809) — build-002 RETURN: #976 shape guard (A), #978 comment fold (B), #981/#982 test-only commit D
#   What: the return for commits A + B + D on top of build-001's tips; docs-only, committed in the NG worktree on top of NG D.
#   Why: Chief-003 RE-RULED ADOPT #976 (checker-048 F3 HIGH); #978; Exec P559 (C6 knob isolation, C9/C10 tighten two weak tests).
#   How: every number is from a recorded gated run on the COMMITTED trees (raw results in /tmp/z12-905d/results3|4a|4b|5.jsonl, tables generated from them).
# -------------------
-->

# build-002: #976 redacted-id shape guard + #978 comment fold + #981/#982 test-only fixes

## 0. Status, first
**Done, committed, verified on the committed trees. Nothing uncommitted, nothing pushed or merged, no VPS, the real `in-transit-146.jsonl` / `rows.jsonl` never read or opened. No commit already reported was amended, rebased or reordered** (build-001's `b3f6695b`, `cc45679d`, `11293833`, `91cd208d`, `d0cde8a0` and its return `32569113` are untouched; the pair's reviewer worktrees were never touched). Every sweep/mutant result below is green / killed. Two things the Chief should read before the rest: (1) **d05's kill needs an environment note (§6):** in this shell `PYTHONPATH=/home/josh/NeuroGraph:` makes NG importable, so d05 already errors under the OLD tests (via the session preamble) and the new tests' value only shows with NG not importable (`PYTHONPATH` emptied), which I ran both ways. (2) **Test-only commit D is TWO commits** (one per repo) because its files live in two repos; each lists test files only.

## 1. Commits (all unpushed; read with `git rev-parse HEAD` / `git log` / `git status --porcelain` in each worktree; both worktrees clean at their tips when every run below executed)
| id | repo / worktree | commit | what |
|---|---|---|---|
| A | NG `/home/josh/NeuroGraph-worktrees/z12-in-transit-hold-905d-20261002` | `1225c4e25104ecf1dad4036422f9364dfd0f4bc6` | #976: `redacted_id_shape` corrupt class (`cc_topology_merge.py` + `tests/test_cc_in_transit_hold_905d.py`) |
| B (NG) | same | `734f18462943e3218fcea113c7c3b3c37066333c` | #978: `cc_ng_organism.py` docstring only |
| B (daemon) | docs `/home/josh/docs/.claude/worktrees/daemon-in-transit-hold-20261002` | `62cd1731350c392c68da4c767b22f207f0e3bd77` | #978: `scripts/cc-ng-daemon.py` comments/docstrings only |
| D (NG) | NG | `2aaff43b1db65449acce72d0f459c3ce327213ce` | #981 (C6): autouse knob-isolation fixture in the 2 failing NG test files |
| D (daemon) | docs daemon | `837b087c65a36a2d9048faead886bcb418907753` | #982 (C9/C10): the lazy-import test and the vacuous warning test tightened |
| return | NG | the commit containing this file (`git log -1` there) | docs-only `build-002.md` |
NG branch order: `32569113` (build-001 return) -> A -> B -> D -> this return. `git diff --name-only 734f184 2aaff43` (NG D) lists ONLY `tests/test_cc_emergent_want_bound_905.py`, `tests/test_cc_merge_whole_graph_guard.py`; the daemon D lists only `scripts/tests/test_cc_ng_daemon_in_transit_hold_905d.py`. **No production file changes in D.**

## 2. COMMIT A: the `redacted_id_shape` class
**Where:** `cc_topology_merge.py:943` `_IN_TRANSIT_REDACTED_SHAPE = re.compile(r"^[a-z]+:[0-9a-f]{12}$")`, a named module-level compiled constant beside the other `_IN_TRANSIT_*` constants, with a comment saying it is the shape `redact_node_id` (`:1132`) produces, `<kind>:<first 12 hex of sha256(id)>`; the check is `:1015` inside `_read_in_transit_ids` (`:959`), right after the `bad_id` check: `if _IN_TRANSIT_REDACTED_SHAPE.fullmatch(nid): raise _InTransitCorrupt("redacted_id_shape")`. ONE such id makes the WHOLE file corrupt, through the existing fail-CLOSED path (`held_unbound_nodes` == `_unbound_nodes`, hold ALL; `in_transit_ids()` == None; ONE loud ERROR naming the class, never the id or a line; cached once). The class is listed in the CORRUPT comment block and in `held_unbound_nodes`'s docstring. **Both Chief guards kept:** an empty intersection is NOT corrupt, and no count is pinned (ZERO ids stays the only size rule). Nothing else changed; `_unbound_nodes` is byte-identical (build-001's source-hash pin still passes at every seed).
**Tests (answers to (1)-(7); `tests/test_cc_in_transit_hold_905d.py`, 21 new, 86 total, scratch data only):**
1. **Real raw-id SHAPES pass:** 146 seeded ids (132 `cc:conv::<40 hex>::tree::<concept words with spaces, colons, unicode>` + 14 `cc:conv::<40 hex>`) -> VALID, `in_transit_ids()` is a 146-frozenset, `held_unbound_nodes` holds the cohort's unbound nodes (not the laptop-own one), one INFO and no ERROR.
2. **All-redacted file** (`forest:2dfa2d637643`, `tree:`/`want:`/`window:`/`node:` + 12 hex) -> corrupt: hold ALL (== `_unbound_nodes`) through `held_unbound_nodes`, `merge_landed` variants and `whole_graph_guard`; ONE ERROR naming `redacted_id_shape`; `caplog` carries no id fragment and no line; `in_transit_ids()` None.
3. **Mixed file** (146 good + ONE redacted id at the FIRST line, in the MIDDLE, at the LAST line; 3 params) -> the whole file is corrupt.
4. **Boundary shapes do NOT trip it** (10 params): 11 hex; 13 hex; uppercase hex; a digit in the kind; an underscore in the kind; `cc:want::<hex>`; `cc:conv::<12 hex>`; a trailing char; a trailing NEWLINE inside the id (this is what separates `fullmatch` from `search`/`match`); a leading space.
5. **Empty intersection stays quiet (guard 1):** a VALID file whose ids are all bound or absent -> `held_unbound_nodes` empty, no ERROR, the consolidation pass runs every step, and `in_transit_ids()` is still the whole set; a graph holding none of the cohort is quiet too.
6. **No count pinned (guard 2):** 1, 2 and 500 ids are all VALID (none is 146).
7. **Reported like `bad_id`:** through `in_transit_ids()`, `held_unbound_nodes` and `whole_graph_guard`: one read (spy), one ERROR, cached (a file fixed afterwards is not re-read until a restart). Plus a test that the constant is exactly `redact_node_id`'s output shape (every redacted form it can make matches; no raw form does).

### Mutants of A (scratch copies of the COMMITTED file `1225c4e2`, one per gated run, seed 0, `Z12_MERGE_UNDER_TEST`): all KILLED
| mutant | start load | result | KILLED by |
|---|---|---|---|
| s1_guard_removed | 2.64/3.83 | **KILLED** (5 failed, 80 passed, 1 skipped) | `test_976_2_an_all_redacted_shape_file_is_corrupt`, `test_976_3_one_redacted_id_poisons_a_file_of_146_good_ones`, `test_976_3_one_redacted_id_poisons_a_file_of_146_good_ones`, `test_976_3_one_redacted_id_poisons_a_file_of_146_good_ones` (+more) |
| s2_guard_only_on_the_first_line | 2.32/3.65 | **KILLED** (3 failed, 82 passed, 1 skipped) | `test_976_3_one_redacted_id_poisons_a_file_of_146_good_ones`, `test_976_3_one_redacted_id_poisons_a_file_of_146_good_ones`, `test_976_7_reported_like_bad_id_one_read_one_error_cached` |
| s3_re_search_instead_of_fullmatch | 2.14/3.48 | **KILLED** (1 failed, 84 passed, 1 skipped) | `test_976_4_boundary_shapes_do_not_trip_the_guard` |
| s4_regex_widened_to_trip_cc_conv_forms | 2.33/3.39 | **KILLED** (6 failed, 79 passed, 1 skipped) | `test_976_1_the_real_raw_id_shapes_pass_and_hold_as_before`, `test_976_4_boundary_shapes_do_not_trip_the_guard`, `test_976_4_boundary_shapes_do_not_trip_the_guard`, `test_976_4_boundary_shapes_do_not_trip_the_guard` (+more) |
| s5_count_pinned_not_146_is_corrupt | 3.04/3.50 | **KILLED** (32 failed, 53 passed, 1 skipped) | `test_976_4_boundary_shapes_do_not_trip_the_guard`, `test_976_4_boundary_shapes_do_not_trip_the_guard`, `test_976_4_boundary_shapes_do_not_trip_the_guard`, `test_976_4_boundary_shapes_do_not_trip_the_guard` (+more) |
| s6_empty_intersection_treated_as_corrupt | 3.62/3.63 | **KILLED** (6 failed, 79 passed, 1 skipped) | `test_a_a_laptop_own_unbound_node_does_not_hold_either_guard`, `test_a_the_clock_runs_when_the_only_unbound_node_is_laptop_own`, `test_b_a_merge_landed_arrival_holds_even_when_it_is_not_in_the_file`, `test_d_a_laptop_own_node_that_cofires_and_sprouts_survives_through_the_engines_o` (+more) |
| s7_the_id_printed_in_the_error | 2.74/3.41 | **KILLED** (4 failed, 81 passed, 1 skipped) | `test_976_2_an_all_redacted_shape_file_is_corrupt`, `test_976_3_one_redacted_id_poisons_a_file_of_146_good_ones`, `test_976_3_one_redacted_id_poisons_a_file_of_146_good_ones`, `test_976_3_one_redacted_id_poisons_a_file_of_146_good_ones` |

## 3. COMMITS B (#978): comment/docstring-only, PROVEN
`cc_ng_organism.py` (`_cc_callosum_consolidate`'s `guard` paragraph, `:~2613-2618`) said what blocks "is defined ONCE, in `cc_topology_merge._unbound_nodes`": it now says `held_unbound_nodes` defines the HOLD (#905-DELTA) over `_unbound_nodes`, which defines the sweep-eligible unbound base and what "bound" means. In the daemon: the `_autosave_loop` comment at the pre-check ("any unbound node exists") and `_log_unbound_block`'s docstring/comments now say the blocking set is the HELD set. **The log STRING `_log_unbound_block` emits is UNCHANGED: it still reads "While ANY unbound node exists the idle steps do not run and no batch is deposited ..."** (stale for a VALID in-transit source; I added a NOTE in the docstring and a comment above the `logger.error` saying so). **Follow-up for the Chief** (a log string's change is its own look).
**Proof method** (`/tmp/z12-905d/proof_comment_only.py <repo> <file> [<parent rev>]`, run gated against each B edit before it was committed, so HEAD was the parent): (a) both versions parse; (b) `ast.dump` of each with EVERY docstring (module/class/function) replaced by a placeholder is IDENTICAL, so every non-comment, non-docstring token, including every string literal that can reach a log, is equal; (c) every `+`/`-` line of `git diff -U0` is either a `#` comment line or lies inside a docstring span of ITS OWN version, so the multiset of the remaining (non-comment, non-docstring) +/- lines is EMPTY. A NEGATIVE CONTROL (a scratch repo where a string literal changes) is correctly reported NOT comment-only. Output, verbatim:
```
[gate] proof-comment-only z12-in-transit-hold-905d-20261002 START load 1m=4.46 5m=4.09 15m=3.49 at 00:48:40Z
(a) both versions parse: OK
(b) ast.dump equal with all docstrings replaced by a placeholder: True  (len 465441)
(c) changed lines: 3 '#' comment lines, 12 docstring-span lines, 0 OTHER
    multiset of non-comment, non-docstring +/- lines: EMPTY
RESULT: COMMENT/DOCSTRING-ONLY PROVEN
[gate] proof-comment-only z12-in-transit-hold-905d-20261002 exit=0

[gate] proof-comment-only daemon-in-transit-hold-20261002 START load 1m=3.69 5m=3.95 15m=3.46 at 00:49:07Z
(a) both versions parse: OK
(b) ast.dump equal with all docstrings replaced by a placeholder: True  (len 245670)
(c) changed lines: 12 '#' comment lines, 8 docstring-span lines, 0 OTHER
    multiset of non-comment, non-docstring +/- lines: EMPTY
RESULT: COMMENT/DOCSTRING-ONLY PROVEN
[gate] proof-comment-only daemon-in-transit-hold-20261002 exit=0

```
```
negative control (a string literal and a docstring changed in a scratch repo):
[gate] negative-control START load 1m=2.68 5m=3.68 15m=3.39 at 00:49:42Z
(a) both versions parse: OK
(b) ast.dump equal with all docstrings replaced by a placeholder: False  (len 267)
(c) changed lines: 0 '#' comment lines, 2 docstring-span lines, 2 OTHER
    multiset of non-comment, non-docstring +/- lines: {('-', 'return "log string"  # c'): 1, ('+', 'return "log string CHANGED"  # c2'): 1}
RESULT: NOT comment-only
[gate] negative-control exit=1
```

## 4. COMMIT D (test-only): #981 (C6) knob isolation
**Finding the affected files** (named files only, gated, `CC_NG_IN_TRANSIT_IDS_PATH` EXPORTED to a valid synthetic scratch file; the real file never read): I ran the 3 merge/guard files plus EVERY other NG test file that imports `cc_topology_merge` or `cc_ng_organism` (33 files in 3 groups; `test_cc_deposit_step.py` and the NG full suite NOT run). Only code in `cc_topology_merge.py` reads the variable, and production callers of the merge/guard/accessor outside it are the daemon (other repo); six test files reach it. **Result: exactly the 16 le-061 measured fail, in exactly two files.** The 22 failing/erroring items that appear in the other groups (10 in `test_cc_dual_pass.py`, plus 5 failures and 7 errors in group 01 incl. `test_cc_recall_unification.py`, `test_cc_retrieval_enrichment.py`, `test_cc_refeed.py`, `test_cc_pattern_completion_recall.py` and the pre-existing `poincare` test) are IDENTICAL with the variable unset AND on the exported base tree `f685a3a7` (pre-existing, unrelated to the variable and to this lane; flagged F-D2).
**Before / after (same two files, seed 0, variable EXPORTED; before = tip `734f184`, after = D's working tree = commit `2aaff43b`):**
| state | result |
|---|---|
| BEFORE D | **16 failed, 61 passed, 2 skipped** (7 in `test_cc_merge_whole_graph_guard.py`, 9 in `test_cc_emergent_want_bound_905.py`) |
| AFTER D | **77 passed, 2 skipped, 0 failed** (the same 16 now pass) |
| the new in-transit file `tests/test_cc_in_transit_hold_905d.py` with the variable EXPORTED | 86 passed (it sets the variable itself per test) |
The 16 that failed before and pass after:
```
  tests/test_cc_merge_whole_graph_guard.py::test_a_preexisting_unbound_node_blocks_the_steps_even_when_every_arrival_binds
  tests/test_cc_merge_whole_graph_guard.py::test_a_binding_the_orphan_releases_the_steps_for_the_next_frame_by_exactly_idle_steps
  tests/test_cc_merge_whole_graph_guard.py::test_b_consolidation_runs_at_the_batch_where_the_graph_first_has_no_unbound_node[2-2]
  tests/test_cc_merge_whole_graph_guard.py::test_b_consolidation_runs_at_the_batch_where_the_graph_first_has_no_unbound_node[3-2]
  tests/test_cc_merge_whole_graph_guard.py::test_b_consolidation_runs_at_the_batch_where_the_graph_first_has_no_unbound_node[3-3]
  tests/test_cc_merge_whole_graph_guard.py::test_c_mixed_a_preexisting_and_an_arrival_both_unbound_are_split_in_the_stats_and_the_record
  tests/test_cc_merge_whole_graph_guard.py::test_c_the_id_sample_is_bounded_to_three_sorted_ids
  tests/test_cc_emergent_want_bound_905.py::test_b_merge_an_unprotected_unbound_preexisting_node_blocks_the_steps_and_survives_with_one_error[cc_emergent_want]
  tests/test_cc_emergent_want_bound_905.py::test_b_merge_an_unprotected_unbound_preexisting_node_blocks_the_steps_and_survives_with_one_error[conversational]
  tests/test_cc_emergent_want_bound_905.py::test_b_merge_an_unprotected_unbound_preexisting_node_blocks_the_steps_and_survives_with_one_error[plain]
  tests/test_cc_emergent_want_bound_905.py::test_b_the_unbound_unprotected_node_survives_the_real_sweep_because_the_clock_was_held
  tests/test_cc_emergent_want_bound_905.py::test_c_the_897_error_for_an_unbound_TREE_node_prints_only_redacted_forms
  tests/test_cc_emergent_want_bound_905.py::test_d_a_node_that_becomes_unbound_between_slices_stops_the_pass_at_the_slice_boundary
  tests/test_cc_emergent_want_bound_905.py::test_d_whole_graph_guard_is_zero_arg_reads_the_whole_graph_under_the_step_lock_and_skips_protected
  tests/test_cc_emergent_want_bound_905.py::test_d_the_merge_passes_the_real_guard_and_a_progress_dict_to_the_consolidation
  tests/test_cc_emergent_want_bound_905.py::test_d_merge_a_pass_held_midway_is_reported_not_counted_the_next_batch_proceeds_and_the_outer_guard_still_blocks
```
**The fix:** an autouse fixture `_isolate_in_transit_knob_981` in each of the two files: `monkeypatch.delenv("CC_NG_IN_TRANSIT_IDS_PATH", raising=False)` and `tmg._reset_in_transit_cache_for_tests()` (via `getattr`, so an older module copy under `Z12_MERGE_UNDER_TEST`/BASE_REF still runs) before AND after each test. **No assertion changed**; per-file fixtures, no `conftest.py` (none exists; adding one could alter collection). I did NOT add the fixture to the other merge-reaching files (`test_cc_topology_callosum.py`, `test_cc_topology_frame.py`, `test_cc_topology_capture_423.py`): they did not fail with the variable set (capture_423 stubs the cohort); say if you want it there too.

## 5. COMMIT D (test-only): #982 (C9/C10)
**C9, lazy import** (`scripts/tests/test_cc_ng_daemon_in_transit_hold_905d.py`): (i) an AST walker over EVERY `Import`/`ImportFrom` reachable at module level (inside try/except/else/finally/if/with/for/while/class bodies; function bodies are the lazy path and skipped) plus module-level `__import__`/`import_module` calls, with its own self-test (`test_the_module_level_import_walker_sees_imports_inside_try_if_with_and_class`); (ii) `test_a_fresh_load_never_even_attempts_an_ng_import_whether_or_not_ng_is_importable`: a meta-path SPY records every import ATTEMPT during a fresh exec (a swallowed `ImportError` is still an attempt) and requires none; (iii) the same claim in a FRESH INTERPRETER (`test_a_fresh_interpreter_loads_the_daemon_with_no_ng_import_attempt`). **C10, the vacuous warning test:** split into `test_the_empty_graph_fallback_still_says_the_ng_is_old_once` and `test_the_fallback_warning_is_emitted_while_a_secret_named_node_is_in_the_graph_and_names_no_id`, which puts the SECRET-named unbound nodes in the graph, THEN resets the once-per-process flag (test-only monkeypatch of the module flag), so the WARNING is emitted in the very call that sees them; it asserts exactly one WARNING, the nodes were visible to the call, and no node id/substring appears in any record's message, args or rendered text, nor in the following block record.
**le-061's mutants, re-created exactly from the committed daemon `62cd1731` (scratch `/tmp/z12-905d/dmut3/`):** d05 = a try-wrapped module-level `import cc_topology_merge as _eager_ng` / `except ImportError` after `import os`; d08 = the production fallback WARNING text interpolating a node id (`... unbound node (e.g. %s) ...`, `sorted(unbound)[0]`). Each run is one gated invocation of the daemon file, seed 0; the "OLD" tests are the A/B-state daemon test file exported with `git archive 62cd1731` to `/tmp/z12-905d/dbase`.
| mutant | tests | NG importable? | result | by which test |
|---|---|---|---|---|
| d05 | OLD (pre-D) | NO (`PYTHONPATH` emptied) | **SURVIVED** (18 passed): reproduces le-061 | n/a |
| d05 | OLD (pre-D) | yes (this shell's `PYTHONPATH=/home/josh/NeuroGraph:`) | killed only at SESSION level (18 errors): the P379 preamble sees a real NG module in `sys.modules` | the session preamble, not the lazy test |
| d05 | **NEW** | NO | **KILLED** (3 failed, 19 passed) | `test_no_module_level_import_of_ng_anywhere_in_the_daemon_ast`, `test_a_fresh_load_never_even_attempts_an_ng_import_whether_or_not_ng_is_importable`, `test_a_fresh_interpreter_loads_the_daemon_with_no_ng_import_attempt` |
| d05 | **NEW** | yes | **KILLED** (22 errors: the preamble) | the session preamble (and the three above when the preamble is not fatal) |
| d08 | OLD (pre-D) | yes (not also run with NG unimportable: d08's behaviour does not depend on importability) | **SURVIVED** (18 passed) | n/a: the old test was vacuous |
| d08 | **NEW** | NO (and yes) | **KILLED** (1 failed, 21 passed) | `test_the_fallback_warning_is_emitted_while_a_secret_named_node_is_in_the_graph_and_names_no_id` |
| none (clean daemon `62cd1731`) | **NEW** | NO | 22 passed | n/a |
(The d08 NG-importable NEW run is in the pre-commit queue: 1 failed, 21 passed, the same test.)

## 6. Runs: PYTHONHASHSEED 0..31, starting loads, env SEEN
Gate: `gate.sh` = a fixed 25 s pause BEFORE each run, then refuse at 1-min >= 6 OR 5-min >= 5.0; a refused run is never retried while the gate is closed (the queue restarts only after a GENUINE re-open, logged as a fresh attempt). **Seen on every run:** `CC_NG_BATCH_SIZE=<unset> CC_NG_IDLE_STEPS=<unset> CC_NG_DRAIN_HOLD_ON_FAILURE=<unset> NG_EMBED_REMOTE=<unset> CC_NG_IN_TRANSIT_IDS_PATH=<unset> HOME=/tmp/z12-905d/home PYTHONUSERBASE=/home/josh/.local` for the standard runs (`env -u` the five); the C6 "exported" runs print `CC_NG_IN_TRANSIT_IDS_PATH=/tmp/z12-905d/scratch-in-transit.jsonl (EXPORTED)` with the other four unset; the d05 not-importable runs add `PYTHONPATH=` (empty). **Refusals during this build (all honoured, none retried while closed):** the first q3 attempt stopped at `ng-sweep-12`, then after genuine re-opens at `ng-sweep-13` (twice) and `ng-sweep-20`, each restart a fresh attempt (so 5 attempts total, no gap in the table); the first "AFTER D" run was REFUSED at 1m=6.45 and re-run after the gate re-opened; the q4a/q4b/q5 queues ran with no refusal.
| 32-seed sweep | files | tree |
|---|---|---|
| NG in-transit file (86 tests) | `tests/test_cc_in_transit_hold_905d.py` | A+B tip `734f184` |
| NG two D-touched files (77 passed, 2 skipped: the existing env-gated skips) | `test_cc_merge_whole_graph_guard.py` + `test_cc_emergent_want_bound_905.py`, variable UNSET | D tip `2aaff43b` |
| daemon D-touched file (22 tests) | `scripts/tests/test_cc_ng_daemon_in_transit_hold_905d.py` | D tip `837b087c` |
| seed | NG in-transit file @A+B tip: load / passed | NG two files @D (variable UNSET): load / passed | daemon file @D: load / passed |
|---|---|---|---|
| 0 | 1.49/3.18 / 86 passed | 3.17/3.47 / 77 passed, 2 skipped | 5.59/3.65 / 22 passed |
| 1 | 1.37/2.99 / 86 passed | 3.45/3.52 / 77 passed, 2 skipped | 5.40/3.82 / 22 passed |
| 2 | 1.73/2.92 / 86 passed | 2.94/3.37 / 77 passed, 2 skipped | 5.90/4.13 / 22 passed |
| 3 | 1.74/2.82 / 86 passed | 3.08/3.38 / 77 passed, 2 skipped | 4.95/4.07 / 22 passed |
| 4 | 5.18/3.52 / 86 passed | 3.38/3.45 / 77 passed, 2 skipped | 5.44/4.26 / 22 passed |
| 5 | 5.18/3.69 / 86 passed | 2.46/3.23 / 77 passed, 2 skipped | 4.11/4.05 / 22 passed |
| 6 | 4.54/3.71 / 86 passed | 2.45/3.16 / 77 passed, 2 skipped | 3.46/3.90 / 22 passed |
| 7 | 5.28/3.97 / 86 passed | 1.97/2.97 / 77 passed, 2 skipped | 2.77/3.70 / 22 passed |
| 8 | 5.19/4.10 / 86 passed | 1.72/2.83 / 77 passed, 2 skipped | 2.52/3.56 / 22 passed |
| 9 | 4.51/4.03 / 86 passed | 1.37/2.65 / 77 passed, 2 skipped | 3.92/3.78 / 22 passed |
| 10 | 5.34/4.32 / 86 passed | 1.57/2.58 / 77 passed, 2 skipped | 3.42/3.67 / 22 passed |
| 11 | 4.77/4.29 / 86 passed | 2.15/2.64 / 77 passed, 2 skipped | 3.62/3.68 / 22 passed |
| 12 | 5.05/4.66 / 86 passed | 2.03/2.57 / 77 passed, 2 skipped | 4.42/3.92 / 22 passed |
| 13 | 2.05/4.39 / 86 passed | 2.63/2.66 / 77 passed, 2 skipped | 3.83/3.83 / 22 passed |
| 14 | 1.70/4.12 / 86 passed | 3.56/2.92 / 77 passed, 2 skipped | 3.83/3.82 / 22 passed |
| 15 | 2.61/4.10 / 86 passed | 3.37/2.94 / 77 passed, 2 skipped | 3.00/3.61 / 22 passed |
| 16 | 2.46/3.93 / 86 passed | 3.41/2.98 / 77 passed, 2 skipped | 3.08/3.57 / 22 passed |
| 17 | 3.03/3.92 / 86 passed | 2.94/2.90 / 77 passed, 2 skipped | 3.47/3.63 / 22 passed |
| 18 | 3.30/3.90 / 86 passed | 2.15/2.72 / 77 passed, 2 skipped | 2.90/3.48 / 22 passed |
| 19 | 4.82/4.20 / 86 passed | 2.07/2.66 / 77 passed, 2 skipped | 2.46/3.32 / 22 passed |
| 20 | 4.20/4.30 / 86 passed | 2.06/2.61 / 77 passed, 2 skipped | 2.55/3.28 / 22 passed |
| 21 | 3.81/4.23 / 86 passed | 1.65/2.46 / 77 passed, 2 skipped | 2.17/3.11 / 22 passed |
| 22 | 2.63/3.92 / 86 passed | 2.49/2.57 / 77 passed, 2 skipped | 2.24/3.05 / 22 passed |
| 23 | 2.83/3.85 / 86 passed | 2.06/2.46 / 77 passed, 2 skipped | 1.76/2.86 / 22 passed |
| 24 | 3.86/3.95 / 86 passed | 2.60/2.54 / 77 passed, 2 skipped | 3.44/3.18 / 22 passed |
| 25 | 4.38/4.07 / 86 passed | 2.65/2.56 / 77 passed, 2 skipped | 2.38/2.95 / 22 passed |
| 26 | 4.26/4.08 / 86 passed | 3.02/2.65 / 77 passed, 2 skipped | 1.87/2.77 / 22 passed |
| 27 | 3.67/3.96 / 86 passed | 2.70/2.63 / 77 passed, 2 skipped | 1.38/2.58 / 22 passed |
| 28 | 5.42/4.32 / 86 passed | 2.71/2.66 / 77 passed, 2 skipped | 1.43/2.47 / 22 passed |
| 29 | 5.54/4.49 / 86 passed | 3.25/2.82 / 77 passed, 2 skipped | 1.11/2.30 / 22 passed |
| 30 | 4.23/4.27 / 86 passed | 3.20/2.86 / 77 passed, 2 skipped | 0.94/2.16 / 22 passed |
| 31 | 3.21/4.03 / 86 passed | 4.87/3.26 / 77 passed, 2 skipped | 0.99/2.05 / 22 passed |

**Carry-over (results obtained on the A/B tree for files D did NOT touch; still valid at the D tips because the D diffs are test-files-only and these files/production code are byte-identical):** (1) the NG in-transit test file `tests/test_cc_in_transit_hold_905d.py` (A's file; D does not touch it): the 32-seed sweep above is on the A+B tip, and one more run at the D tip with the variable exported (86 passed); (2) the A-mutant table; (3) the existing NG consumer files at seed 0 on the A+B tip: `test_cc_topology_capture_423.py`, `test_cc_topology_callosum.py`, `test_cc_topology_frame.py` (+ the two D now re-swept): **129 passed, 2 skipped, 1 failed = the pre-existing `test_poincare_dir_is_rederived_locally_not_transmitted`** (fails at the base `f685a3a7` too); (4) the daemon pacing + new file at seed 0 on the B tip: **82 passed** (the pacing file untouched by D); (5) the comment-only proofs (B commits are not touched by D). The #900 branch (`d0cde8a0`) and build-001's sweeps are untouched by this nudge.

## 7. What ran / did not / not verified
**Ran:** everything in §2-§6. **Did not run (on purpose):** `tests/test_cc_deposit_step.py`; the NG full suite; any `systemctl`; anything on the VPS or live; the real `in-transit-146.jsonl` / `rows.jsonl` (never read or opened: every set is synthetic and shaped like the real ids); no process was killed by a loose pattern (none killed at all this time). **Not verified:** (a) the real 146 ids through the real loader (synthetic shapes only; I cannot see the real file, so "none of the real 146 matches the regex" rests on the Chief's and Z12's measurement, not mine); (b) a sweep of the NG D files with the variable EXPORTED at every seed (I ran seed 0 exported before commit and all 32 seeds UNSET; the fixture makes the two agree by construction); (c) the redacted-shape guard against the real `redact_node_id` output beyond the shapes the test enumerates (it checks the constant against the real function's outputs for 5 id kinds).

## 8. Flags for the Chief (not decided by me)
- **F-D1 (same weakness, other branch):** the #900 Part 3 test (`scripts/tests/test_cc_ng_daemon_unbound_status_in_transit_905d.py::test_the_accessor_import_is_lazy_and_the_daemon_reads_no_new_env`, on `cc-laptop-daemon-status-900-20261001` @ `d0cde8a0`) has the same top-level-only `Import` walk. D covered the two files the addendum named; I did NOT touch that reviewed hash. Say if you want the same tightening as a new commit there.
- **F-D2 (pre-existing, for the punchlist):** 22 failing/erroring items in other NG test files (10 in `test_cc_dual_pass.py`; 5 failures + 7 errors in the group containing `test_cc_recall_unification.py` etc.) are identical at the base `f685a3a7`: not mine, not caused by the variable.
- **F-D3 (environment):** this shell exports `PYTHONPATH=/home/josh/NeuroGraph:` (the PRIMARY checkout), so any test that does not put its own worktree first can import the primary's NG. The NG test files here insert their worktree first and resolve inside it (their P379 preambles print it), and the daemon tests use fakes, but it is why d05 was already caught by the preamble in this shell and why the C9 behavioural half must not depend on NG being importable.
- **F-D4:** the daemon's `_log_unbound_block` ERROR string still reads "While ANY unbound node exists ..." (follow-up, §3).
- **Carried from build-001 (unchanged):** F1 (a concurrent drain cannot see an in-flight merge's arrivals), F2 (the drain's own deposits are laptop-own), F4 (the old-NG shim), F5 (now half-fixed: the organism docstring and the daemon comments, not the ERROR string), F8 (`poincare`).
- **A note on the guard's limit (it is a SHAPE check, not an identity check):** it catches a file built from redacted log forms; it does not catch other wrong-id files (a stale snapshot, a different graph's ids). That is the Chief's ruling (no empty-intersection check, no count pin), recorded here so nobody reads `redacted_id_shape` as "ids verified".
