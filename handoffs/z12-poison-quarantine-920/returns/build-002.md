<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane poison-quarantine-920, dispatch #13417) — build-002 return for #920 (Chief-003 ADDENDUM 1 + Exec P501 / ADDENDUM 2)
#   What: the literal retry count (max_retries = RETRIES AFTER the first failed attempt; attempts = max_retries + 1), stated in words in the quarantine ERROR
#     and the drain docstring, the operator lever, the tests moved to the new boundary, and the gated runs and mutants.
#   Why: Chief-003 rulings on build-001; Exec P501's one condition. The #920 pair cannot point at a final tip until this commits.
#   How: every number below comes from the run log in this directory (build-002-runs.log.txt) or the mutant output (build-002-mutants.out.txt).
# -------------------
-->

# #920 build-002 return: the literal retry count, "attempts = retries + 1" stated

Related: [[NeuroGraph]] · [[Dual-Pass Embedding]] · [[Vendored Files]] · [[The Laws]]

## 0. READ FIRST

1. **Delivered and committed (local, nothing pushed):** ADDENDUM 1 + ADDENDUM 2, one commit per repo on top of the build-001 tips.
2. **One gap, stated plainly: the mutant table is PARTIAL.** The load gate refused at M8 (START LOAD 6.22/4.06). **12 mutants ran, all 12 KILLED** (including both off-by-one directions and the relation-dropped mutant). **16 did NOT run:** M8, M9, M10, M11, M12, M13, M14 (NG) and D1 to D9 (daemon). They were killed in build-001 against the old boundary; their status against the build-002 tests is **unverified**. Your call whether they need a gated re-run.
3. **The climb was partly my own doing:** the 1-minute load went 3.67 to 5.89 across eleven back-to-back mutant runs (each is a pytest process), so I cannot attribute the 6.22 to other sessions.
4. **Question for the Chief:** ADDENDUM 1 lists `quarantined_total` in the status among the #928 items. It already exists from build-001 (accepted as delivered), so I **left it untouched** and did not build anything toward #928.

## 1. Commits (real `git rev-parse HEAD`)

| Repo | Branch | Previous tip | build-002 code commit |
|---|---|---|---|
| NG | `cc-laptop-quarantine-920-20261001` | `4df4a35dc20d1a02ee04443f344646d5ab4ef513` | `44e034fb4d1491696a5cdcc064133b3c24a9a70b` |
| docs daemon | `cc-laptop-daemon-quarantine-920-20261001` | `45f57f86390ac962949851bfd8a5757179bc517a` | `2e882d2f948b549a9223ba14e5e36f26e5cfcf56` |

Plus one docs-only commit in the NG worktree (this file, the gated mutant harness, and the raw run / mutant logs); its hash is in the builder's final message. No rebase, no push, no merge. `git diff --name-only 895a809a -- ng_embed.py neuro_foundation.py openclaw_hook.py stream_parser.py activation_persistence.py` is empty; the lock part (#922) is untouched.

## 2. What changed (file:line is in the commits; summary here)

**NG `cc_ng_organism.py`:**
- `_drain_bounded_hold`: the ONE boundary change `failures < max_retries` to `failures <= max_retries`. Failures 1..max_retries are held; the (max_retries + 1)-th consecutive failure quarantines in that same cycle. Default 5 = up to 6 attempts; `max_retries=1` = 2 attempts. No other behaviour changed.
- **ERROR** now reads `... failures=<n> max_retries=<N> file=... stored=... -- quarantining after <attempts> attempts (= max_retries + 1), max_retries=<N>: the raw frame is durably ...` (existing fields kept; entry tag, reason, class name only; no text, path or id).
- **Below-bound WARNING** now reads `attempt=<n> of <N+1> (max_retries=<N>)`, which is truthful under the new reading (it used to read `attempt=n/N`).
- **Docstring** (`drain_ingest_tract`): `attempts = max_retries + 1 (attempts = retries + 1)`, the formula with examples (5 means up to 6, 1 means 2), and the **operator lever**: a TID/embed/socket outage is indistinguishable at the drain from a poison turn, so every `max_retries + 1` failing cycles parks one good turn in the quarantine file; **while TID is flaky, RAISE `CC_NG_DRAIN_MAX_RETRIES`**. The infra-vs-data split and a documented re-ingest path are named as row #928, not built.
- A changelog entry, which states that it supersedes the count sentence of the build-001 entry below it.

**Docs daemon `scripts/cc-ng-daemon.py`** (comments, docstring and warning text only, no behaviour): the formula and the lever on `DRAIN_MAX_RETRIES_DEFAULT`, in `_read_drain_max_retries_env`'s docstring and at the read site; the garbage WARNING no longer says "that many failed attempts" (it names the default of 5 retries and failed attempt number 6, never the value). Per ADDENDUM 2 ("nothing else changes") the daemon files were not touched for the relation wording.

## 3. Tests: changed (file, test, why). No assertion weakened.

All in NG `tests/test_cc_poison_quarantine_920.py` (my own build-001 file). A `_attempts(n) = n + 1` helper makes the boundary explicit.

| Test | Why it changed |
|---|---|
| `test_a_turn_that_fails_every_cycle_is_retried_max_retries_times_then_quarantined_whole` (max_retries 1, 3, 5) | Boundary moved: hold for attempts 1..N, quarantine on N+1. The old `max_retries == 1` special case is **gone** (T1 and the quarantine can no longer share a call), so the assertion is now uniform, not weaker. WARNING assertion now `attempt=n of N+1 (max_retries=N)`; ERROR assertions now `failures=N+1` and the full `quarantining after N+1 attempts (= max_retries + 1), max_retries=N`. |
| `test_the_quarantine_error_carries_only_the_hash_form_class_counts_and_file_name` | Needed `_attempts(2)` = 3 drains to reach the quarantine. |
| `test_a_raising_dual_pass_is_quarantined_with_its_exception_class_name` | 2 drains became 3. |
| `test_poison_as_the_very_first_entry_is_quarantined_and_the_tract_advances` | Two held attempts, the third quarantines, the fourth drains T3. |
| `test_fsync_happens_before_the_tract_is_truncated` | Two held drains before the fsync spy is installed, then the quarantining drain. |
| `test_a_failed_quarantine_write_keeps_the_entry_held_and_is_one_error` (3 faults) | Pre-step is now 2 drains, otherwise the first faulted cycle would be a hold, not a failed quarantine. |
| `test_recovery_after_a_failed_quarantine_write_quarantines_exactly_once` (3 faults) | Same pre-step. |
| `test_a_frame_found_at_the_tail_is_fsynced_again_before_the_tract_advances` | Restructured: one held drain first (asserts no quarantine file), then the fsync fault; attempts 2, 3, 4. |
| `test_crash_between_the_quarantine_fsync_and_the_truncate_neither_duplicates_nor_loses` | **Restructured:** the truncate fault is disarmed for the first (held) drain so T1's prefix truncate works, then armed for the quarantining attempt; after the restart the count resets, so the entry takes two attempts again (held, then re-quarantined idempotently). The old comment about T1 being re-absorbed no longer applies and was removed. |
| `test_a_soft_truncate_failure_requarantines_idempotently_and_does_not_inflate_the_counter` | First drain is the held attempt (asserts no quarantine file); then three re-quarantines under the refused truncate. |
| `test_the_quarantine_file_is_created_no_wider_than_the_tract_directory_and_file` | `_attempts(2)` drains. |
| `test_max_entries_still_counts_the_poison_attempt` | Three calls: held, quarantined, T3. |
| `test_daemon_loop_quarantines_a_poison_turn_after_the_bound_and_counts_it` (real NG + docs daemon) | Four autosave cycles for `max_retries=3`; the conduit-trickle assertions index the new cycle count. |

**New tests:** `test_boundary_attempt_n_still_held_attempt_n_plus_1_is_quarantined_in_that_cycle` (max_retries 1, 2, 5); `test_boundary_max_retries_failures_then_a_success_on_the_last_attempt_is_absorbed_not_quarantined` (1, 2, 5); `test_the_hold_warning_and_the_quarantine_error_state_the_numbers_literally` (default-sized 5: `attempt=1 of 6`, `quarantining after 6 attempts (= max_retries + 1), max_retries=5`); **`test_the_quarantine_error_states_attempts_equal_retries_plus_one`** (Exec P501's condition: parses the ERROR text with a regex and asserts the relation is stated and that attempts is exactly max_retries + 1, for 1 and 5); `test_the_docstring_states_the_formula_and_the_operator_lever` (includes `attempts = max_retries + 1` and the lever). Docs daemon `test_cc_ng_daemon_poison_quarantine_920.py`: three NEW tests (reader docstring, constant and read-site comments, the truthful garbage warning). **No existing daemon assertion changed.** One of the three new daemon tests had a bug of mine: it scanned the whole file for the old phrase "that many failed attempts" and so matched my own changelog entry that quotes it; fixed by scoping the check to the code below the first `import` (that is why that file ran twice).

## 4. Runs (every run through `gated.py`; START LOAD = 1-minute / 5-minute load average; gate: refuse at 1m >= 6 or 5m >= 5.0; `NG_EMBED_REMOTE` unset, scratch HOME, `HF_HUB_OFFLINE=1`, printed-path preamble each time)

| Run | START LOAD | Result |
|---|---|---|
| NG new file (daemon env set) | 8.73 | **REFUSED, not run** |
| NG new file (daemon env set) | 6.41 | **REFUSED, not run** |
| NG `tests/test_cc_poison_quarantine_920.py` (daemon env set) | 5.62 (1m only; the 5m gate was added at the resume) | **53 passed** |
| NG `tests/test_cc_bind_atomic_904.py` | 5.93 (1m only) | **44 passed** |
| NG `tests/test_cc_drain_hold_on_failure.py` | 6.26 | **REFUSED, not run** |
| NG `tests/test_cc_drain_hold_on_failure.py` | 1.47 / 2.32 | **49 passed** |
| docs `test_cc_ng_daemon_poison_quarantine_920.py` | 1.64 / 2.32 | 1 failed, 58 passed (my test bug, section 3) |
| docs `test_cc_ng_daemon_poison_quarantine_920.py` (after the test-only fix) | 2.67 / 2.53 | **59 passed** |
| docs `test_cc_ng_daemon_deposit_904.py` pacing UNSET / SET | 2.71 / 2.54 ; 2.89 / 2.60 | **62 passed** / **62 passed** |
| docs `test_cc_ng_daemon_drain_pacing.py` pacing UNSET / SET | 2.97 / 2.61 ; 2.90 / 2.60 | **46** / **46 passed** |
| docs `test_cc_ng_daemon_recall_status.py` pacing UNSET / SET | 2.89 / 2.60 ; 3.31 / 2.69 | **41** / **41 passed** |

The 53-pass and 44-pass NG runs happened under the earlier 1-minute-only gate (starts 5.62 and 5.93, under 6). I did **not** record their 5-minute readings; the nearest readings I do have (4.96 just before them and 5.55 just after) straddle the stricter 5.0 line, so those two runs may not have passed the stricter rule. Every run after the resume was gated on both. Raw log: `build-002-runs.log.txt`. **Not run, by constraint:** `tests/test_cc_deposit_step.py` and the NG full suite.

## 5. Mutants (gated harness `build-002-mutants.py`; scratch copies under `/tmp/z12-920-scratch/mut/`; each replacement asserted to match exactly once)

**Ran, all KILLED (12):**

| Mutant | Failing tests | START LOAD |
|---|---|---|
| M1 no retry bound | 27 | 3.67/2.88 |
| **B1 off-by-one EARLY** (the old boundary `failures < max_retries`) | 26 | 3.72/2.92 |
| **B2 off-by-one LATE** (`failures <= max_retries + 1`) | 22 | 3.87/2.99 |
| **R1 relation dropped from the ERROR** | 6 | 5.40/3.39 |
| R2 WARNING `attempt=n of N` untruthful | 4 | 5.11/3.40 |
| R3 relation dropped from the docstring | 1 | 4.94/3.39 |
| M2 quarantine skipped (discard) | 23 | 5.03/3.43 |
| M3 tract truncated BEFORE the fsync | 23 | 5.86/3.73 |
| M4 unbounded table | 1 | 5.49/3.72 |
| M5 entry text in the ERROR | 8 | 5.61/3.78 |
| M6 failed quarantine advances the tract | 7 | 5.76/3.87 |
| M7 not append-only (truncating open) | 7 | 5.89/3.96 |

**NOT RUN (16), the gate refused at START LOAD 6.22/4.06:** NG M8 (no idempotency), M9 (`already_present` skips the fsync), M10 (key from text), M11 (count not cleared on success), M12 (quarantined but tract not advanced), M13 (default #794 WARNING text changed), M14 (counter also on `already_present`); daemon D1 to D9 (kwarg passed with hold off; default 5 with hold off; bound read per call; counter not mirrored; garbage warning at import; `main()` never warns; zero accepted; mirror inside the graph lock; stat seeded with the flag off). The harness already carries their (updated) replacement strings.

## 6. Not verified

- The 16 mutants above against the build-002 tests.
- The existing NG test files and the two edited build-001 pins were **not** run against the base code (no base counts for them), and the new files were **not** re-run against the BASE copies this round (build-001's base counts, 37/43 and 40/56, are for the old file contents).
- Everything build-001 listed as not verified still applies (power-loss durability, a real D24 drain, concurrent miniTID appends, the VPS host call).
- That a TID outage no longer parks good turns: it still does, every `max_retries + 1` failing cycles; only the documentation and the lever (raise the variable) changed. #928 is not built.
