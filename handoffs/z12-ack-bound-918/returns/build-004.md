<!--
# ---- Changelog ----
# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane ack-bound-918, dispatch #14238) — return build-004: the owed runs of fold-up 2 (TEST-RUNS ONLY)
# What: completes what the load gate stopped in build-003: N3 seed sweep 5..31, whole new file at seeds 14/21/28, mutants G2/G7 against the FIXED N3 test, guard / capture_423 / callosum once each.
# Why: the brief (docs worktree daemon-recall-756-20260930, handoffs/z12-silent-failure-lanes-20260930/assignments/build-918-sweep.md) requires this file; Chief-003 ASK C: YES.
# How: runs only (no code/test commit: no defect found); mutants are scratch copies of the COMMITTED merge (b1949155); every run was preceded by a 25 s pause and gated on its own START load; NO run was refused.
# -------------------
-->

# Return build-004 — #918 owed runs (sweep completion). NO code or test commit.

**Seat/authority:** worker, `worktree_write` (RESUMED my own thread). Test runs only. Nothing pushed or merged; `git status` shows no other change (clean before this file).

## Headline
- **The N3 test is seed-independent across the FULL 0..31 sweep: 32/32 seeds pass** (seeds 0-4 in build-003, seeds 5-31 here), each seed 3 cases passed; the whole new file also passes at seeds 0, 7 (build-003) and **14, 21, 28** (42 passed each).
- **G2 (budget ignored): KILLED** by the fixed N3 test, in all three X orderings.
- **FINDING, reported not fixed: G7 (the ack includes the unbound set) is NOT killed by the N3 test** (`-k n3`: 3 passed with the mutant active). **My build-003 sentence that G7 "ends in the exhausted branch with X unbound (its first assertion fails)" was reasoned, not run, and is WRONG for this scenario.** G7 IS killed by the whole file (12 tests, none of them N3), see below.
- Guard / capture_423: green. Callosum: only the one known #902 failure. No refusals, no other failure.

## Effective values SEEN (every run)
Each run printed an `EFFECTIVE` line: `CC_NG_BATCH_SIZE`, `CC_NG_IDLE_STEPS`, `CC_NG_DRAIN_HOLD_ON_FAILURE`, `NG_EMBED_REMOTE` all `'<unset>'` (the runner uses `env -u` on all four) plus `PYTHONHASHSEED` (the seed of that run, or `<unset>` for the three single-file runs). For the long sweep lines I collapsed the four unset flags to the words `flags-unset OK` with a `sed` that only substitutes when all four read `'<unset>'` (any set flag would have shown the full dict); the mutant runs show the full line including `Z12_MERGE_UNDER_TEST=/tmp/z12_918_scratch/mut4/<G2|G7>/cc_topology_merge.py`. Scratch HOME, `PYTHONUSERBASE=/home/josh/.local`, `HF_HUB_OFFLINE=1`, `-p no:cacheprovider`, `PYTHONDONTWRITEBYTECODE=1`.

## Runs (START load = 1-min / 5-min read after the 25 s pause; gate: refuse at 1-min >= 6 or 5-min >= 5.0; NONE refused; the box was at 6.7-7.0 right before the first run and had eased below the gate by the time its own check ran, so the first run was allowed by its own reading, not by waiting past a refusal)
### 1. N3 seed sweep, `-k n3` on `tests/test_cc_ack_bound_918.py` (3 cases), seeds 5..31: ALL 27 RAN, ALL PASSED
| seed | START load1/load5 | result | | seed | START load1/load5 | result |
|---|---|---|---|---|---|---|
| 5 | 5.10 / 4.44 | 3 passed | | 19 | 3.15 / 4.07 | 3 passed |
| 6 | 4.03 / 4.24 | 3 passed | | 20 | 2.48 / 3.82 | 3 passed |
| 7 | 3.77 / 4.14 | 3 passed | | 21 | 1.97 / 3.56 | 3 passed |
| 8 | 3.45 / 4.05 | 3 passed | | 22 | 1.35 / 3.29 | 3 passed |
| 9 | 3.09 / 3.91 | 3 passed | | 23 | 1.50 / 3.16 | 3 passed |
| 10 | 2.89 / 3.80 | 3 passed | | 24 | 1.50 / 3.02 | 3 passed |
| 11 | 4.54 / 4.08 | 3 passed | | 25 | 3.43 / 3.34 | 3 passed |
| 12 | 4.80 / 4.20 | 3 passed | | 26 | 3.39 / 3.37 | 3 passed |
| 13 | 5.06 / 4.35 | 3 passed | | 27 | 2.50 / 3.15 | 3 passed |
| 14 | 5.74 / 4.55 | 3 passed | | 28 | 2.50 / 3.08 | 3 passed |
| 15 | 5.82 / 4.70 | 3 passed | | 29 | 1.99 / 2.91 | 3 passed |
| 16 | 4.25 / 4.43 | 3 passed | | 30 | 1.53 / 2.71 | 3 passed |
| 17 | 3.83 / 4.31 | 3 passed | | 31 | 1.29 / 2.56 | 3 passed |
| 18 | 3.63 / 4.23 | 3 passed | | | | |
Seeds that ran: 5..31 (all). Seeds that did not run: none. With build-003's seeds 0-4 (all 3 passed) the sweep 0..31 is complete: **32/32**.

### 2. Whole new file (42 tests)
| seed | START | result |
|---|---|---|
| 14 | 1.79 / 2.57 | 42 passed |
| 21 | 1.45 / 2.41 | 42 passed |
| 28 | 1.28 / 2.30 | 42 passed |

### 3. Mutants vs the FIXED N3 test (scratch copies of the COMMITTED `b1949155` merge, `PYTHONHASHSEED=0`, one per gated run; the diffs are one line each)
| mutant | change | START | result | killed by |
|---|---|---|---|---|
| **G2** budget ignored | `budget = max_nodes_per_call` -> `budget = 10 ** 9` | 1.99 / 2.34 | **3 failed (all three X orderings: `[first]`, `[middle]`, `[last]`), 39 deselected** | the N3 test, at its per-call cap assertion `assert ms["absorbed_nodes"] <= 1, call` (`assert 2 <= 1`: the mutant lands both new nodes on call 1); **KILLED** |
| **G7** the ack includes the unbound set | `cc_ack_membership` returns `held` instead of `held - _unbound_nodes(...)` | 2.73 / 2.47 | **`-k n3`: 3 passed (SURVIVES the N3 test)** | **NOT killed by N3** |
| G7, whole file (extra run, to say what DOES kill it) | same mutant | 2.18 / 2.36 | **12 failed, 29 passed, 1 skipped** | KILLED by the whole file: `test_membership_bound_in_unprotected_unbound_out_protected_unbound_in_non_cc_out`, `test_membership_USES_the_imported_unbound_predicate_not_a_copy`, `test_membership_a_double_without_is_identity_protected_is_not_exempted_unbound_nodes_stay_out`, `test_the_ack_file_the_merge_writes_is_the_bound_membership`, `test_cc_ack_membership_is_cc_current_membership_minus_the_sweep_eligible_unbound_set`, `test_cc_ack_membership_composes_the_two_imported_functions_not_copies`, `test_q2_a_protected_unbound_cc_node_is_acked_and_never_re_offered_...`, `test_one_merge_binds_a_held_unbound_node_via_a_reoffered_hyperedge_and_the_ack_follows`, `..._via_a_reoffered_synapse_only`, `test_membership_stale_readmitted_counts_a_culled_BOUND_node_not_a_culled_unbound_one`, `test_the_loop_re_offers_held_unbound_nodes_with_their_hyperedge_and_binds_them`, `test_the_ack_after_every_tick_contains_only_bound_nodes` (the "1 skipped" is the module-resolution test, skipped whenever the merge is overridden) |

**Why G7 survives N3 (what I can show, not a production bug):** in the N3 scenario the VPS hyperedge `{C1, C2, X}` has an un-acked member (C2 is not held on the laptop until call 2). The sender's whole-hyperedge closure RE-ADDS already-acked members (`cc_topology_export.py` `_closeable_he`: "already-acked members are RE-added (idempotent)"), so with G7's over-wide ack (C1 and X acked) the frame is still `{C1, C2, X}` on every tick, exactly what the correct ack gives. The two acks only diverge when EVERY member of a hyperedge is acked-but-unbound (the #909 cohort shape), which the loop and ack-file tests cover and N3 does not. So N3 pins the BUDGET (G2), not the ack semantics; the ack semantics are pinned by the 12 tests above.
**Consequence for the record:** the build-003 statement and my code comment/docstring in `tests/test_cc_ack_bound_918.py` for the N3 test ("one whose ack includes the unbound set never re-offers X and ends in the exhausted branch with X unbound: both fail") claim more than the test does for G7. I did NOT edit it (this brief is test-runs only, no commit unless a defect; I am not deciding whether that overclaim in a docstring is a "defect" to fix). Zone manager: your call whether to ask for a one-line docstring correction; the mutant coverage of G7 itself is intact (12 killers).

### 4. The other files, once each (no seed set, `PYTHONHASHSEED` `<unset>`)
| file | START | result |
|---|---|---|
| `tests/test_cc_merge_whole_graph_guard.py` (#897) | 1.80 / 2.26 | **17 passed, 2 skipped** (the 2 skips need `Z12_MERGE_BASE_REF`, which is UNSET: not run) |
| `tests/test_cc_topology_capture_423.py` | 1.44 / 2.12 | **4 passed** |
| `tests/test_cc_topology_callosum.py` | 1.27 / 2.02 | **33 passed, 1 failed**: only `test_poincare_dir_is_rederived_locally_not_transmitted` (the known #902 failure); no other failure |

## What did not run
Nothing from the owed list was skipped or refused. Not run, forbidden as ever: the NG full suite and `tests/test_cc_deposit_step.py`.

## Not verified
- The 32-seed sweep covers the N3 test (`-k n3`); the WHOLE new file was run at seeds 0, 7, 14, 21, 28 (not 0..31), and no other test in it was swept.
- Nothing ran against any live graph/ack/conduit/VPS; all loops are synthetic.
- The G7 explanation above is read from the exporter's closure code and consistent with the observed pass/fail, but I did not instrument a G7 N3 run to print the frames.
- The two skipped #897 tests were not run (`Z12_MERGE_BASE_REF` unset).
