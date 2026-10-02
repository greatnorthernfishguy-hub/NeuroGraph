<!--
# ---- Changelog ----
# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane in-transit-hold-905d, dispatch #14946) — build-003 RETURN: DL-1 (#987) strip guard (a), F-D1 #900 lazy-import walker (b), le-061 C9 walker limits + comment (c)
#   What: the return for THREE new commits in three worktrees; docs-only, committed in the NG worktree on top of the NG commit (a).
#   Why: Chief-003 APPROVED FOLLOW-UP 3 (bundle of three small commits).
#   How: every number is from a recorded gated run on the COMMITTED trees (raw results /tmp/z12-905d/results6b|6c|7.jsonl; tables generated from them).
# -------------------
-->

# build-003: DL-1 strip guard + #900 lazy-import walker + C9 walker limits

## 0. Status, first
**Done, committed, verified on the committed trees. Nothing uncommitted (all three worktrees clean at their tips when the queue ran), nothing pushed or merged, no VPS, the real `in-transit-146.jsonl` / `rows.jsonl` never opened. No commit already reported was amended, rebased or reordered.** All three PYTHONHASHSEED 0..31 sweeps are green (96/96); every mutant of the new behaviour is killed. **Two honest items first:** (1) the old a3 mutant (`search` instead of `fullmatch`) now **SURVIVES, and that is correct: after `strip()` it is an EQUIVALENT mutant** (§2); (2) a **residual DL-1 gap I measured but did not fix (out of the brief):** padding with a zero-width space (U+200B) or a BOM (U+FEFF) is NOT whitespace to `str.strip()`, so such a padded redacted id still passes the guard (§8 F-3-1).

## 1. The three commits (unpushed; read with `git rev-parse HEAD`, `git log`, `git status --porcelain` in each worktree)
| id | worktree | branch | commit | files |
|---|---|---|---|---|
| (a) NG | `/home/josh/NeuroGraph-worktrees/z12-in-transit-hold-905d-20261002` | `cc-laptop-in-transit-hold-905d-20261002` | `12c5148c1f852f5bbf53772f122b4e63f49bcadf` (on build-002's `e70b94b`) | `cc_topology_merge.py`, `tests/test_cc_in_transit_hold_905d.py` |
| (b) #900 | `/home/josh/docs/.claude/worktrees/daemon-status-900-20261001` | `cc-laptop-daemon-status-900-20261001` | `86a16aea70345d172e1b2d1a350086ae31536e8e` (on `d0cde8a0`) | `scripts/tests/test_cc_ng_daemon_unbound_status_in_transit_905d.py` ONLY (`git diff --name-only HEAD~1 HEAD`) |
| (c) docs daemon | `/home/josh/docs/.claude/worktrees/daemon-in-transit-hold-20261002` | `cc-laptop-daemon-in-transit-hold-20261002` | `280ac7b4b32e8baccab67fc13218d470f2ccd70f` (on `837b087c`) | `scripts/tests/test_cc_ng_daemon_in_transit_hold_905d.py` ONLY |
| return | NG | same as (a) | the commit containing this file (`git log -1` there) | docs-only `build-003.md` |
(b) and (c) are TEST-ONLY. (a) changes `cc_topology_merge.py` by ONE expression plus comments/changelog.

## 2. (a) NG: DL-1 (#987)
**The defect (le-061):** the shape check ran `fullmatch` on the id AS WRITTEN, so a copy-pasted log line with padding (` forest:2dfa2d637643`, a trailing `\n`/`\r\n`/tab/NBSP) did not match: the file was VALID and the hold EMPTIED, the exact failure #976 exists for. **The fix** (`cc_topology_merge.py`): `:1021` `if _IN_TRANSIT_REDACTED_SHAPE.fullmatch(nid.strip()):` (regex constant unchanged at `:949`, `r"^[a-z]+:[0-9a-f]{12}$"`), and `ids.add(nid)` still stores the ORIGINAL id (the next line, comment says so): raw tree ids contain internal spaces, are unaffected either way, and the stored cohort/intersection semantics are unchanged. `bad_id` / `zero_ids` and BOTH Chief guards are unchanged (an empty intersection is NOT corrupt; NO count pinned). Changelog entries in the file.
**Tests (`tests/test_cc_in_transit_hold_905d.py`, now 103; 17 new):**
- **The two pinned boundary params FLIPPED:** the trailing-newline-inside-the-id and leading-space cases were in the "boundary shapes that must NOT trip" list (that pinned the bug). They are removed from it (replaced by an internal-space case, `forest: 2dfa2d637643`, which still must not trip) and `test_987_the_two_formerly_pinned_boundary_ids_now_trip_the_guard` asserts each now makes the file corrupt (`class=redacted_id_shape`).
- **Padded variants (9 params, each a corrupt file: class, hold ALL, ONE ERROR, no id/line in `caplog`, `in_transit_ids()` None, cached):** leading space, trailing space, trailing `\n`, trailing `\r\n`, leading tab, trailing tab, leading NBSP, trailing NBSP, and a mixed ` \t tree:...  \r\n`.
- **A padded id as the FIRST / MIDDLE / LAST line of 146 good ones** (3 params): the whole file is corrupt.
- **The NEGATIVE side (5 params):** a RAW tree id with internal spaces and colons; one with leading/trailing spaces in the CONCEPT part; one with tabs and a trailing newline inside the concept; and a space-padded RAW forest id (both sides): each stays VALID and the cohort contains the ORIGINAL string, never its stripped form.
**Mutants of the COMMITTED `12c5148c` (scratch copies, one per gated run, seed 0, `Z12_MERGE_UNDER_TEST`):**
| mutant (scratch copy of COMMITTED `12c5148c`) | start load | result | KILLED by |
|---|---|---|---|
| s1_guard_removed | 1.27/2.43 | **KILLED** (18 failed, 84 passed, 1 skipped) | `test_976_2_an_all_redacted_shape_file_is_corrupt`, `test_976_3_one_redacted_id_poisons_a_file_of_146_good_ones`, `test_976_7_reported_like_bad_id_one_read_one_error_cached`, `test_987_a_padded_redacted_id_poisons_146_good_ones_at_any_position` (+2 more tests) |
| s2_guard_only_on_the_first_line | 0.89/2.25 | **KILLED** (15 failed, 87 passed, 1 skipped) | `test_976_3_one_redacted_id_poisons_a_file_of_146_good_ones`, `test_976_7_reported_like_bad_id_one_read_one_error_cached`, `test_987_a_padded_redacted_id_poisons_146_good_ones_at_any_position`, `test_987_a_padded_redacted_id_trips_the_guard` (+1 more tests) |
| s3_search_instead_of_fullmatch | 0.65/2.06 | **SURVIVED** (102 passed, 1 skipped): see s3 note | n/a (equivalent mutant) |
| s4_regex_widened_to_trip_cc_conv_forms | 0.74/1.96 | **KILLED** (8 failed, 94 passed, 1 skipped) | `test_976_1_the_real_raw_id_shapes_pass_and_hold_as_before`, `test_976_4_boundary_shapes_do_not_trip_the_guard`, `test_976_5_an_empty_intersection_is_not_corrupt_the_hold_goes_quiet_and_the_clock_run`, `test_976_the_constant_is_the_shape_redact_node_id_produces` (+1 more tests) |
| s5_count_pinned_not_146_is_corrupt | 1.22/1.96 | **KILLED** (36 failed, 66 passed, 1 skipped) | `test_976_4_boundary_shapes_do_not_trip_the_guard`, `test_976_5_an_empty_intersection_is_not_corrupt_the_hold_goes_quiet_and_the_clock_run`, `test_976_6_no_count_is_pinned_one_id_and_five_hundred_are_both_valid`, `test_987_raw_ids_with_spaces_stay_valid_and_are_stored_unstripped` (+18 more tests) |
| s6_empty_intersection_treated_as_corrupt | 1.26/1.90 | **KILLED** (6 failed, 96 passed, 1 skipped) | `test_a_a_laptop_own_unbound_node_does_not_hold_either_guard`, `test_a_the_clock_runs_when_the_only_unbound_node_is_laptop_own`, `test_b_a_merge_landed_arrival_holds_even_when_it_is_not_in_the_file`, `test_d_a_laptop_own_node_that_cofires_and_sprouts_survives_through_the_engines_own_dy` (+2 more tests) |
| s7_the_id_printed_in_the_error | 1.18/1.83 | **KILLED** (16 failed, 86 passed, 1 skipped) | `test_976_2_an_all_redacted_shape_file_is_corrupt`, `test_976_3_one_redacted_id_poisons_a_file_of_146_good_ones`, `test_987_a_padded_redacted_id_poisons_146_good_ones_at_any_position`, `test_987_a_padded_redacted_id_trips_the_guard` |
| s8_strip_removed_the_DL1_bug | 0.82/1.68 | **KILLED** (13 failed, 89 passed, 1 skipped) | `test_987_a_padded_redacted_id_poisons_146_good_ones_at_any_position`, `test_987_a_padded_redacted_id_trips_the_guard`, `test_987_the_two_formerly_pinned_boundary_ids_now_trip_the_guard` |
| s9_stored_id_is_the_stripped_one | 0.60/1.56 | **KILLED** (4 failed, 98 passed, 1 skipped) | `test_987_raw_ids_with_spaces_stay_valid_and_are_stored_unstripped` |

**Mutant s8 (`strip()` removed = the DL-1 bug) is KILLED, by `test_987_a_padded_redacted_id_trips_the_guard[...]` (all padded variants) and `test_987_a_padded_redacted_id_poisons_146_good_ones_at_any_position[...]` (13 failed in all) and the flipped-boundary test.** s9 (store the STRIPPED id) is an extra mutant I added: KILLED by the `test_987_raw_ids_with_spaces_stay_valid_and_are_stored_unstripped[...]` params. **s3 (the old a3: `search` instead of `fullmatch`) SURVIVES, and it is an EQUIVALENT mutant now**, not a test gap: it was killed before only by the trailing-newline-inside-the-id boundary case (`$` matches before a final `\n`); after `strip()` no id carries a trailing newline into the match, and the pattern is anchored `^...$` without `MULTILINE`, so `search(x.strip())` and `fullmatch(x.strip())` give the same answer for every string (I checked 11 probes including interior-newline and prefix/suffix cases; none differs). The brief said the a1-a5 family should stay killed: a1, a2, a4, a5 (and the older s6, s7) are; a3 is now equivalent, so the `fullmatch` choice is belt-and-braces, not a behaviour the tests can pin.

## 3. (b) the #900 branch: F-D1
`test_the_accessor_import_is_lazy_and_the_daemon_reads_no_new_env` (and its siblings) in the #900 Part 3 file walked only top-level `Import` nodes. It now uses a **COPY** of the walker (not imported across branches; a comment in the file says so) with its own self-tests: an AST walk over EVERY module-level `Import`/`ImportFrom` (inside try/except/else/finally/if/with/for/while/match/class bodies) plus module-level `__import__`/`importlib.import_module` calls, **using the le-061-tightened version** (compound-statement HEADER expressions and `match` subject/patterns/guards/bodies, decorators/defaults/class bases; function and lambda bodies skipped = the lazy path); self-tests (13 parametrised forms + a positive/negative source); and the behavioural half independent of NG being importable: a meta-path import-ATTEMPT spy during a fresh exec in-process (observe-only) AND in a fresh interpreter in which NG is genuinely BLOCKED (the spy records the attempt and raises `ImportError`). The existing no-new-env check is kept.
**d05-style mutant proof (scratch copies of the COMMITTED `d0cde8a0` `cc-ng-daemon.py`; `PYTHONPATH=` EMPTY = NG NOT importable; OLD = the pre-change test file exported with `git archive d0cde8a0`, NEW = `86a16aea`; one gated run each, seed 0):**
| mutant (`d05_try_wrapped_import`, a try-wrapped module-level import in the committed #900 daemon `d0cde8a0`) | tests | start load | verdict | result | failing tests |
|---|---|---|---|---|---|
| d05 | OLD (`d0cde8a0`) | 3.44/3.00 | **SURVIVED** | 18 passed | none |
| d05 | **NEW** (`86a16aea`) | 3.77/3.12 | **KILLED** | 4 failed, 31 passed | `test_a_fresh_interpreter_loads_the_daemon_with_no_ng_import_attempt`, `test_a_fresh_load_never_even_attempts_an_ng_import_whether_or_not_ng_is_importable`, `test_no_module_level_import_of_ng_anywhere_in_the_daemon_ast`, `test_the_accessor_import_is_lazy_and_the_daemon_reads_no_new_env` |
| d05d literal (`if __import__(...)`), OLD | | 3.72/3.29 | KILLED at session level (the daemon crashes at load: NG not importable) | 18 errors | n/a |
| d05d literal, NEW | | 3.72/3.17 | KILLED at session level (the daemon crashes at load: NG not importable) | 35 errors | n/a |
| d05e literal (`match` case-body import), OLD | | 3.31/3.22 | KILLED at session level (the daemon crashes at load: NG not importable) | 18 errors | n/a |
| d05e literal, NEW | | 3.74/3.22 | KILLED at session level (the daemon crashes at load: NG not importable) | 35 errors | n/a |

d05 reproduces F-D1 exactly: with NG not importable the OLD test **SURVIVES** (18 passed), the NEW test **KILLS** it (4 failed, by the tests named). The two LITERAL forms (`if __import__(...)`, a `match` case-body import) crash the daemon at load when NG is not importable, so they die at session level in OLD and NEW alike and prove nothing about the walker: for those the walker-vs-walker comparison in §4 is the evidence. I did **not** run the SWALLOWED (try-wrapped) d05d/d05e forms against the #900 branch (its walker is the same copy; §4's swallowed-form results are on the docs daemon branch): **not verified there**.
Runs on (b): the new file + the existing `test_cc_ng_daemon_unbound_status.py` together at seed 0: **75 passed** (with `Z12_900_REDACT_PIN_REV=f685a3a7715dda508c615109e63759b5512bee1f`, the reviewed #905 tip, as in build-001); the new file alone with NG not importable: 35 passed.

## 4. (c) the docs daemon branch: C9 walker limits + the comment
**(1) Walker tightened** (`_module_level_import_targets` in `scripts/tests/test_cc_ng_daemon_in_transit_hold_905d.py`): it now also walks the HEADER expressions of module-level compound statements (`if`/`while` test, `for` iterable, `with` items, `except` type), `match` subject / case patterns / guards / bodies, class bases, decorators and default arguments, plus comprehensions (which run at load); function and lambda bodies stay skipped. Its self-test gains one parametrised case per form (13) plus the positive/negative source. **Walker-vs-walker (the OLD walker extracted from the committed `837b087c` test file vs the new one, on the mutant forms; one gated run):**
```
module-level source                              OLD walker     NEW walker
literal d05d  if __import__(...): pass           missed         FOUND
swallowed d05d_s  try: if __import__(...)        missed         FOUND
literal d05e  match case body imports            missed         FOUND
swallowed d05e_s  try: match ... import          missed         FOUND
d05  try-wrapped import (control)                FOUND          FOUND
clean (lazy import in a function)                missed         missed
```
**(2) The fresh-interpreter comment:** it said NG was "deliberately NOT importable" while its spy returned `None` (an importable NG would still import after the attempt was recorded). **Chosen: make the spy BLOCK** (record the attempt, then raise `ImportError`), so the claim is TRUE and the test does not depend on `sys.path`; the in-process spy remains observe-only and its docstring now says exactly that.
**Mutants (scratch copies of the committed daemon `62cd1731`, `PYTHONPATH=` EMPTY; OLD = the test file at `837b087c` exported with `git archive`, NEW = `280ac7b4`; one gated run each, seed 0):**
| mutant / tests | start load | verdict | result | failing tests |
|---|---|---|---|---|
| d05 try-wrapped import (control): OLD tests (`837b087c`) | 3.60/3.26 | KILLED | 3 failed, 19 passed | `test_a_fresh_interpreter_loads_the_daemon_with_no_ng_import_attempt`, `test_a_fresh_load_never_even_attempts_an_ng_import_whether_or_not_ng_is_importable`, `test_no_module_level_import_of_ng_anywhere_in_the_daemon_ast` |
| d05 try-wrapped import (control): **NEW** tests (`280ac7b4`) | 3.45/3.25 | KILLED | 3 failed, 32 passed | `test_a_fresh_interpreter_loads_the_daemon_with_no_ng_import_attempt`, `test_a_fresh_load_never_even_attempts_an_ng_import_whether_or_not_ng_is_importable`, `test_no_module_level_import_of_ng_anywhere_in_the_daemon_ast` |
| d05d LITERAL `if __import__('cc_topology_merge'): pass`: OLD tests (`837b087c`) | 4.30/3.49 | KILLED at session level (load crash) | 22 errors | (session-level: 0 errors, the daemon failed to import) |
| d05d LITERAL `if __import__('cc_topology_merge'): pass`: **NEW** tests (`280ac7b4`) | 3.29/3.32 | KILLED at session level (load crash) | 35 errors | (session-level: 0 errors, the daemon failed to import) |
| d05d SWALLOWED (the same inside `try/except ImportError`): OLD tests (`837b087c`) | 3.32/3.32 | KILLED | 2 failed, 20 passed | `test_a_fresh_interpreter_loads_the_daemon_with_no_ng_import_attempt`, `test_a_fresh_load_never_even_attempts_an_ng_import_whether_or_not_ng_is_importable` |
| d05d SWALLOWED (the same inside `try/except ImportError`): **NEW** tests (`280ac7b4`) | 2.06/3.02 | KILLED | 3 failed, 32 passed | `test_a_fresh_interpreter_loads_the_daemon_with_no_ng_import_attempt`, `test_a_fresh_load_never_even_attempts_an_ng_import_whether_or_not_ng_is_importable`, `test_no_module_level_import_of_ng_anywhere_in_the_daemon_ast` |
| d05e LITERAL module-level `match` whose case imports it: OLD tests (`837b087c`) | 1.86/2.89 | KILLED at session level (load crash) | 22 errors | (session-level: 0 errors, the daemon failed to import) |
| d05e LITERAL module-level `match` whose case imports it: **NEW** tests (`280ac7b4`) | 3.09/3.08 | KILLED at session level (load crash) | 35 errors | (session-level: 0 errors, the daemon failed to import) |
| d05e SWALLOWED (the same inside `try/except ImportError`): OLD tests (`837b087c`) | 2.83/3.02 | KILLED | 2 failed, 20 passed | `test_a_fresh_interpreter_loads_the_daemon_with_no_ng_import_attempt`, `test_a_fresh_load_never_even_attempts_an_ng_import_whether_or_not_ng_is_importable` |
| d05e SWALLOWED (the same inside `try/except ImportError`): **NEW** tests (`280ac7b4`) | 3.78/3.22 | KILLED | 3 failed, 32 passed | `test_a_fresh_interpreter_loads_the_daemon_with_no_ng_import_attempt`, `test_a_fresh_load_never_even_attempts_an_ng_import_whether_or_not_ng_is_importable`, `test_no_module_level_import_of_ng_anywhere_in_the_daemon_ast` |

**What the table says, precisely** (I am not claiming more than it shows): for the SWALLOWED d05d/d05e forms the OLD test file is NOT silent (its two behavioural spy tests, `..._fresh_load_never_even_attempts...` and `..._fresh_interpreter_loads...`, already see the import ATTEMPT), but its AST walker test (`test_no_module_level_import_of_ng_anywhere_in_the_daemon_ast`) does NOT fail: that test's walker misses the form (the table above). The NEW file's AST test now fails on them too, so the static check and the behavioural check agree; the tightened walker is defence in depth, not the only line. The LITERAL forms crash the load with NG not importable (session-level errors in both files), so they are "killed" by the crash in OLD and NEW alike.
Runs on (c): the new file + the D24 pacing file (NG importable): **99 passed**; the new file alone with NG not importable: 35 passed.

## 5. Runs: PYTHONHASHSEED 0..31, START loads, env SEEN
Gate: `gate.sh` = a fixed 25 s pause BEFORE each run, refuse at 1-min >= 6 OR 5-min >= 5.0; a refused run is never retried while the gate is closed (the queue restarts only after a GENUINE re-open, logged as a fresh attempt). **Refusals in the final queue (all honoured):** stopped at `p900-sweep-21`, `daemon-sweep-3` and `daemon-sweep-21`, each restarted as a fresh attempt after a genuine re-open (4 attempts, no gap in the tables); the earlier pre-commit queues (6b, 6c) ran with no refusal. **Seen on every run:** `CC_NG_BATCH_SIZE=<unset> CC_NG_IDLE_STEPS=<unset> CC_NG_DRAIN_HOLD_ON_FAILURE=<unset> NG_EMBED_REMOTE=<unset> CC_NG_IN_TRANSIT_IDS_PATH=<unset> HOME=/tmp/z12-905d/home PYTHONUSERBASE=/home/josh/.local` (`env -u` the five), the P379/#770 preambles naming the module/daemon under test (no real NG module loaded); the "NG not importable" runs add `PYTHONPATH=` (empty; the shell's ambient value is `/home/josh/NeuroGraph:`); the knob-exported run prints `CC_NG_IN_TRANSIT_IDS_PATH=/tmp/z12-905d/scratch-in-transit.jsonl (EXPORTED)`; the #900 runs add `Z12_900_REDACT_PIN_REV=f685a3a7...`.
Sweeps (one invocation per seed, on the committed tips `12c5148c` / `86a16aea` / `280ac7b4`): NG in-transit file 103 tests; #900 new file 35 tests (alone; its `world` fixture comes from the existing #900 file, which is unchanged); docs daemon file 35 tests.
| seed | NG in-transit file: load / passed | #900 new file: load / passed | docs daemon file: load / passed |
|---|---|---|---|
| 0 | 1.57/2.68 / 103 passed | 4.50/3.00 / 35 passed | 5.28/4.31 / 35 passed |
| 1 | 1.06/2.45 / 103 passed | 3.82/2.97 / 35 passed | 5.68/4.51 / 35 passed |
| 2 | 3.27/2.80 / 103 passed | 3.41/2.95 / 35 passed | 5.34/4.59 / 35 passed |
| 3 | 2.65/2.69 / 103 passed | 2.89/2.89 / 35 passed | 4.59/4.65 / 35 passed |
| 4 | 3.21/2.83 / 103 passed | 2.22/2.73 / 35 passed | 4.39/4.62 / 35 passed |
| 5 | 3.01/2.84 / 103 passed | 2.35/2.72 / 35 passed | 3.37/4.35 / 35 passed |
| 6 | 2.42/2.72 / 103 passed | 2.37/2.71 / 35 passed | 2.66/4.09 / 35 passed |
| 7 | 2.23/2.66 / 103 passed | 1.68/2.51 / 35 passed | 2.79/4.00 / 35 passed |
| 8 | 1.79/2.52 / 103 passed | 1.70/2.44 / 35 passed | 3.71/4.11 / 35 passed |
| 9 | 2.38/2.59 / 103 passed | 1.62/2.35 / 35 passed | 3.40/4.00 / 35 passed |
| 10 | 2.87/2.68 / 103 passed | 1.29/2.21 / 35 passed | 3.31/3.92 / 35 passed |
| 11 | 3.31/2.81 / 103 passed | 1.04/2.06 / 35 passed | 3.35/3.87 / 35 passed |
| 12 | 3.67/2.92 / 103 passed | 0.88/1.94 / 35 passed | 2.68/3.67 / 35 passed |
| 13 | 3.29/2.90 / 103 passed | 0.90/1.85 / 35 passed | 3.34/3.71 / 35 passed |
| 14 | 2.66/2.82 / 103 passed | 1.34/1.87 / 35 passed | 4.82/4.03 / 35 passed |
| 15 | 2.01/2.66 / 103 passed | 1.59/1.88 / 35 passed | 4.74/4.10 / 35 passed |
| 16 | 2.03/2.61 / 103 passed | 3.07/2.26 / 35 passed | 4.89/4.17 / 35 passed |
| 17 | 2.25/2.61 / 103 passed | 4.14/2.59 / 35 passed | 4.58/4.16 / 35 passed |
| 18 | 1.73/2.45 / 103 passed | 3.42/2.56 / 35 passed | 4.48/4.18 / 35 passed |
| 19 | 1.49/2.33 / 103 passed | 2.63/2.46 / 35 passed | 5.01/4.31 / 35 passed |
| 20 | 1.34/2.21 / 103 passed | 2.76/2.49 / 35 passed | 5.85/4.56 / 35 passed |
| 21 | 2.02/2.29 / 103 passed | 5.37/3.69 / 35 passed | 3.10/4.40 / 35 passed |
| 22 | 3.33/2.59 / 103 passed | 5.16/3.85 / 35 passed | 2.69/4.16 / 35 passed |
| 23 | 3.41/2.67 / 103 passed | 5.01/3.90 / 35 passed | 2.05/3.87 / 35 passed |
| 24 | 3.79/2.83 / 103 passed | 5.14/4.08 / 35 passed | 1.67/3.64 / 35 passed |
| 25 | 3.82/2.94 / 103 passed | 4.06/3.93 / 35 passed | 1.74/3.48 / 35 passed |
| 26 | 2.48/2.71 / 103 passed | 4.28/3.99 / 35 passed | 1.18/3.18 / 35 passed |
| 27 | 2.65/2.72 / 103 passed | 4.21/3.99 / 35 passed | 1.61/3.11 / 35 passed |
| 28 | 2.55/2.69 / 103 passed | 3.83/3.93 / 35 passed | 1.33/2.91 / 35 passed |
| 29 | 1.90/2.53 / 103 passed | 4.53/4.10 / 35 passed | 1.49/2.79 / 35 passed |
| 30 | 1.78/2.44 / 103 passed | 4.05/4.03 / 35 passed | 1.60/2.71 / 35 passed |
| 31 | 2.02/2.44 / 103 passed | 3.56/3.92 / 35 passed | 1.23/2.51 / 35 passed |

Other gated runs on the committed trees: the existing NG consumer files `test_cc_topology_capture_423.py`, `test_cc_merge_whole_graph_guard.py`, `test_cc_emergent_want_bound_905.py`, `test_cc_topology_callosum.py`, `test_cc_topology_frame.py` at seed 0: **129 passed, 2 skipped, 1 failed (the pre-existing `poincare` test, which fails at the base `f685a3a7`)**; the two files with the C6 fixtures with the knob EXPORTED: **77 passed, 2 skipped** (the fixtures still hold after the strip change).
**Carry-over (earlier results for code these commits did not touch; SAID so):** (1) build-002's A/B/D sweeps and mutants cover everything outside the one `strip()` expression: the comment-only proofs for the B commits, the C6 16-before/after, the d05/d08 table for `62cd1731`'s production code (production daemon code is unchanged by (b)/(c): both are test-only); (2) the existing #900 test file is unchanged and ran green beside the new file (75 passed); (3) build-001's tables for `held_unbound_nodes`, the accessor, Part 2 and Part 3 production code are untouched. What WAS re-run on the final trees because these commits changed it: the NG in-transit file (the shape check and its tests), the #900 new file, the docs daemon file, and the A-family mutants (regenerated from `12c5148c`).

## 6. What ran / did not / not verified
**Ran:** everything in §2-§5. **Did not run (on purpose):** `tests/test_cc_deposit_step.py`; the NG full suite; any `systemctl`; anything on the VPS or live; the real in-transit file (never opened); no process was killed by a loose pattern (none killed). **Not verified:** (a) the SWALLOWED d05d/d05e forms against the #900 branch (§3); (b) the guard against the real 146 ids (synthetic shapes only); (c) a sweep of the knob-exported run at every seed (seed 0 exported, 32 seeds unset: the fixtures make them agree by construction); (d) whitespace classes beyond `str.strip()`'s 29 code points (§8).

## 7. Flags
- **F-3-1 (residual DL-1 gap, LOW; measured, NOT fixed because the brief says `nid.strip()`):** `str.strip()` removes the 29 code points `str.isspace()` accepts (space, `\t\n\r\v\f`, NBSP, U+2000-200A, U+3000...), but NOT U+200B (zero-width space) or U+FEFF (BOM/ZWNBSP): a redacted id padded with one of those still matches nothing and passes the guard. A copy-paste from a rich-text source can carry them. A stricter check (e.g. also strip `​﻿`, or reject any id containing a non-printable/zero-width code point) is a one-line follow-up; say if you want it.
- **F-3-2:** a3 (`search`) is an equivalent mutant now (§2): keep or drop it from the mutant family; `fullmatch` stays as written.
- **F-3-3 (environment, carried from build-002):** the shell exports `PYTHONPATH=/home/josh/NeuroGraph:` (the PRIMARY checkout), which is why every import-claim mutant proof here was run with `PYTHONPATH=` empty as well.
- **Carried from build-001/002 (unchanged):** F1 (a concurrent drain cannot see an in-flight merge's arrivals), F2 (the drain's own deposits are laptop-own), F4 (the old-NG shim), F-D4 (the `_log_unbound_block` ERROR string still says "ANY"), F-D2 (22 pre-existing failing items in other NG test files), F8 (`poincare`).
