<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane dual-pass-atomic-904, dispatch #13259) — build-002 return: the #904 FOLD-UP
#   What: the return for the fold-up batch (le-050 C3/F11, C8/F8, C4/F1; checker-038 corrections 2 and 3; Chief-003 item 6; Exec Packet 497 (e)).
#   Why: the pair (le-050 ETHOS DRIFT + checker-038 PASS-WITH-NOTES) accepted the build as built and asked for one fold-up batch.
#   How: ONE code commit per repo on top of the build-001 commits (nothing rebased, no upstream, nothing merged); this file is the docs-only return.
# -------------------
-->

# #904 build-002 — fold-up: a failed attempt leaves NO trace; the hold flag is read after logging; a clean start is INFO

Links: [[NeuroGraph]] - [[Dual-Pass Embedding]] - [[The River]] - build-001 return `build-001.md` - brief `build-904-foldup.md` (docs `83069612`, item 6 at `3aa39b46`) - reviews `le-050-904.md`, `checker-038-904.md`.

## READ FIRST

**Hashes (real `git rev-parse HEAD`; local only, no upstream, nothing merged, no PR):**

| Repo / branch | Parent (build-001) | Fold-up code commit |
|---|---|---|
| NG `cc-laptop-bind-atomic-904-20261001` | `f14825d70590d6dba093f2229e7550e2754cdb00` (the build-001 return; its code is `895a809a60ab86afacd8ef193b95c4ab39f6893f`) | `12d6241b03fe4a35b050b73347cfaee145ffe518` |
| docs daemon `cc-laptop-daemon-deposit-904-20261001` | `6b5555022b6a31dcca417a6e24e4fcaa92c2d150` | `c3492dcfdac737c991db39ce9ac513e3157535c5` |
| NG, this return | | the docs-only commit that adds this file (`git log -1` on the NG branch) |

**All six items are done, each with a test that fails on build-001 or kills a mutant, and the whole table is re-run: 39 of 39 mutations KILLED** (build-001's 24 + the fold-up's). One mutant (F4) SURVIVED my first fold-up run: my tests could not see it (see item 3), I added a test, and re-ran the WHOLE table on the final files.

1. **C3/F11 (M20)** — `test_a_failed_exact_repeat_failing_after_the_hyperedge_create_removes_that_hyperedge_and_every_synapse_it_created`: every member pre-exists, the call creates its hyperedge (the test counts it), then fails at the window chain; the rollback removes that hyperedge and every synapse it created and leaves every pre-existing node intact. **M20 (rollback leaves the hyperedge) is now KILLED** (it survived le-050's 44-test run).
2. **C8/F8** — the `CC_NG_DRAIN_HOLD_ON_FAILURE` read and its bad-value WARNING moved off import into `_init_drain_hold()`, called once from `main()` right after `_setup_logging()`. `DRAIN_HOLD_ON_FAILURE` stays a module constant, **False at import**. A test asserts the warning through the daemon's REAL file handler, with a contrast test proving the old ordering never reached the file.
3. **C4/F1 (the defect)** — a failed attempt now leaves NO trace on a pre-existing node: **snapshot + restore**, not "move the stamp to commit time" (why below). A held exact-repeat poison turn retried 4 cycles leaves the pre-existing nodes' full metadata (compared by `repr`, so key order too), threshold, excitability and their vdb entries (embedding bytes, content, metadata) **byte-identical**.
4. **checker-038 correction 2** — a raising `vdb.get()` for a NEW node no longer orphans the entry the call inserted; for a pre-existing node whose probe raised nothing is deleted and the record says how many (`vdb_kept_unprovable=<n>`).
5. **checker-038 correction 3** — two tests where the REAL bind completes (the pointer is moved, `primed_nodes` replaced) and a LATER stage fails: the pointer is restored, and the persisted sidecar is not advanced.
6. **Chief-003 item 6** — a MISSING pointer sidecar is an INFO line; corrupt / non-string / stale-id / unverifiable stay WARNING, once each. Folded into the single docs commit (not committed earlier).

`tests/test_cc_capture_mutations_423.py` passes **UNMODIFIED** (10 passed; `git diff` on it is empty): `_cc_deposit_memory_node` is not edited. `ng_embed.py`, every protected file, and every vendored file are untouched. Nothing from the "do NOT touch" list (#915/#913 swallow sites, #921, #920, #914, the flag-off drain logging (C2), the VPS host (#919)) was touched.

## What changed (file:line, final code)

### NG `cc_ng_organism.py`
- `_CCConversationalDualPassEco.__init__` (`:2071`): `restores`, `unrestorable`, `vdb_unprovable`.
- `deposit` (`:2075`): the vdb probe now keeps the entry it fetched; a probe that RAISES marks the entry fresh when the node is NEW this call (`:2089`) and counts it unprovable when the node pre-existed (`:2092`); a pre-existing node is snapshotted BEFORE delegating (`:2094`).
- `_snapshot_existing` (`:2099`): threshold, excitability, a copy of the metadata dict in key order, and the references of the node's vdb entry. A snapshot that cannot be taken counts as `unrestorable`.
- `rollback` (`:2121`, restore loop `:2161`): after the synapses, the hyperedges, the new nodes and the fresh vdb entries, restores each snapshot LAST-IN-FIRST-OUT (a node deposited twice in one call ends at its ORIGINAL state): threshold, excitability, metadata cleared and refilled IN PLACE (the vdb entry shares that dict), then the original embedding / content / metadata of its vdb entry, the original embedding array put back bit-for-bit (`SimpleVectorDB.insert` re-normalises). Any restore failure, or an unrestorable snapshot, makes it return False, so the record is the ERROR "ROLLBACK FAILED, a partial write REMAINS".
- The failure record (`:2528`) gains `vdb_kept_unprovable=<n>` after the existing counts (every earlier substring is unchanged).
- `_cc_deposit_memory_node`: **unchanged**.

### Docs daemon `scripts/cc-ng-daemon.py`
- `DRAIN_HOLD_ON_FAILURE = False` (`:992`), `_init_drain_hold` (`:995`), its one call in `main()` (`:3101`, directly after `_setup_logging()` and the start-up log line; `main` is `:3097`). The warning now uses the daemon `logger`.
- `_load_conv_pointer` (`:1255`): the `FileNotFoundError` branch is `logger.info` (`:1267`), fixed reason code, no path or value; the four defect branches are unchanged WARNINGs.

## Why snapshot + restore, not "move the stamp to commit time" (C4/F1)
The stamp is inside `_cc_deposit_memory_node`, which `tests/test_cc_capture_mutations_423.py` extracts by AST and pins, and `dual_record_outcome` calls the eco in the MIDDLE of extraction (forest, then each tree). Deferring the stamp to commit time would mean editing or duplicating that function, or restructuring the vendored call order: both outside the brief. A snapshot taken by the eco immediately before it delegates needs neither, lives with the journal on the same class (so the AST harness still extracts it), and leaves the success path untouched (a SUCCESSFUL exact repeat still re-stamps exactly as before; a test pins that on build-001 and the fix alike).

## Item 3's survivor, stated plainly
My first full mutation run was 37 of 38: **F4 (the pre-existing node's vdb entry not restored) SURVIVED.** In my test world the re-written entry is identical to the original (same embedding bytes, same content, and its `metadata` is the very same shared dict), so skipping that restore was unobservable. The restore matters when the pre-existing entry DIFFERS from what the deposit writes (a checkpoint-loaded entry has its own metadata dict, older content, a non-unit embedding). I added `test_a_pre_existing_vdb_entry_that_DIFFERS_from_what_the_deposit_writes_is_put_back_exactly` (it also asserts object identity: the ORIGINAL array and dict, not copies), plus mutation F6 (the exact embedding put-back dropped), and re-ran the entire table.

## Item 6: the tests that pinned the old WARNING
Updated (both pinned "missing => one WARNING"): `test_a_missing_file_is_none_with_one_warning` (now `..._with_one_INFO_and_no_warning`, also asserting no path in the line) and `test_init_ng_with_a_missing_stale_or_corrupt_pointer_starts_with_none_and_one_warning_and_does_not_crash` (the `missing` parameter removed, function renamed `..._a_stale_or_corrupt_...`). Added: `test_init_ng_on_a_clean_start_with_no_sidecar_...` (INFO only, no warning, `None`, no crash), `test_the_four_defect_cases_WARN_exactly_once_and_never_at_info` (parametrized over corrupt / non_string / stale_id / unverifiable: exactly ONE record, at WARNING), `test_the_missing_case_does_not_warn_and_is_the_only_info`. Mutations: D10 (missing back at WARNING), D11 (corrupt downgraded to INFO), D12 (stale id downgraded to INFO): all KILLED.
Also updated by C8: `test_no_default_in_code_and_no_other_new_environment_variable` (the assignment shape is now `False` + `_read_drain_hold_env(environ)` inside `_init_drain_hold`; the no-new-variable diff is unchanged) and the old "read once at import" test replaced by `test_the_constant_is_false_at_import_whatever_the_environment_says`, `test_init_reads_the_environment_once_into_the_constant`, `test_main_configures_logging_first_and_reads_the_flag_exactly_once_after_it_and_nothing_reads_it_at_import` (AST), and the three real-file-handler tests.

## Runs (private scratch root `/tmp/z12-904-fold-scratch`, `HOME` under it, `PYTHONUSERBASE=/home/josh/.local`, `HF_HUB_OFFLINE=1`, `NG_EMBED_REMOTE` unset, `-p no:cacheprovider`, load gated < 6, one run per file)
"Build-001" = the build-001 code (`895a809a` / `6b555502`) via scratch copies (`Z12_904_ORG_UNDER_TEST` / `Z12_904_DAEMON_UNDER_TEST`) or a `git archive` export of `895a809a` (no `.git` in it). P379/#770: both of my files printed the resolved module paths; every NG module resolved under THIS worktree (scratch org only for build-001/mutation runs), the docs file loaded no real NG module.

| File | Fix | Build-001 |
|---|---|---|
| NG `tests/test_cc_bind_atomic_904.py` (with the real daemon loaded) | **54 passed** | 6 failed, 48 passed. The 6: the byte-identity held-turn test (x2: `bind_hyperedge`, `new_tree_vdb_insert`), the differing-vdb-entry test, the double deposit, and the two `vdb.get()` tests. The M20, completed-bind and sidecar tests PASS on build-001 (they are coverage tests whose value is the mutants they kill) |
| same, WITHOUT the daemon path | 49 passed, 5 skipped (the five daemon-gated tests) | n/a |
| NG `tests/test_cc_drain_hold_on_failure.py` | **49 passed** | not re-run (the #794 file, untouched; build-001: 49 passed) |
| NG `tests/test_cc_drain_pacing.py` | not present on this base | |
| NG `tests/test_cc_capture_mutations_423.py` | **10 passed, UNMODIFIED** | 10 passed |
| NG `tests/test_cc_want_bounds.py` | 8 passed | 8 passed |
| docs `scripts/tests/test_cc_ng_daemon_deposit_904.py` | **76 passed** | 15 failed, 61 passed: all the C8 and item-6 tests (the constant-at-import, `_init_drain_hold`, the AST pins, the file-handler tests, the missing-case INFO tests, the assignment-shape test) |
| docs `..._drain_pacing.py` | 46 set / 46 unset | 46 / 46 |
| docs `..._recall_status.py` | 41 set / 41 unset | 41 / 41 |
| NG `test_cc_gateway_durable` / `test_cc_durable_integration` / `test_cc_host_stop_door` / `test_conversational_window_chains` / `test_cc_callosum_leg1` | 28 / 3 / 4 / 4 / 16 passed | identical |
| NG `test_cc_pattern_completion_recall` | 1 failed, 7 passed | identical failing test |
| NG `test_cc_retrieval_enrichment` | 2 failed, 14 passed | identical failing tests (`cc_gsg_backfill` ImportError + one assertion) |
| NG `test_cc_dual_pass` | 10 failed, 11 passed | identical 10 (TID-class, plus the stale `deposit_experience(content=...)`) |

**Constraints (Exec Packet 500).** Not run, as instructed: the NG full suite and `tests/test_cc_deposit_step.py`. No protected, vendored or `ng_embed.py` edit was needed, so no STOP was triggered. **Load:** every run was gated at < 6 (the runners wait for < 5.5 and stop at >= 6); the highest figure printed beside a final run was 4.18 and no step was run above the ceiling. **One environment event, disclosed:** another session wiped my build-001 scratch root `/tmp/z12-904-scratch` (the reviewers used the same root); I rebuilt under a private `/tmp/z12-904-fold-scratch` and re-created the mutation runner there (build-001's `/tmp/z12-904-scratch/mutate.py` no longer exists; the new one is committed beside this return as `build-002-mutate.py`).

## Mutations (scratch copies; every replacement asserted to match EXACTLY once; the WHOLE table re-run on the final files)
| Verdict | Mutation | Result | Killed by (first two) |
|---|---|---|---|
| KILLED | N1 failure logged at debug again | 14 failed, 40 passed in 3.41s | 805_proof_a_turn_that_fails_every_cycle_stays_byte_iden, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_ |
| KILLED | N2 returns True after a partial write | 33 failed, 21 passed in 5.61s | 805_proof_a_turn_that_fails_every_cycle_stays_byte_iden, 805_proof_failed_turn_then_a_retry_with_different_conce |
| KILLED | N3 rollback skipped | 28 failed, 26 passed in 6.08s | 805_proof_failed_turn_then_a_retry_with_different_conce, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_ |
| KILLED | N4 vdb entries not deleted on rollback | 21 failed, 33 passed in 5.30s | 805_proof_failed_turn_then_a_retry_with_different_conce, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_ |
| KILLED | N5 state pointer not restored | 5 failed, 49 passed in 2.77s | a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_ |
| KILLED | N6 forest-absent returns silently again | 1 failed, 53 passed in 2.21s | partial_write_is_rolled_back_loudly_and_truthfully[bind |
| KILLED | N7 the post-condition (truth rule) removed | 5 failed, 49 passed in 3.12s | a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_ |
| KILLED | N8 site counter dropped | 10 failed, 44 passed in 3.05s | a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_ |
| KILLED | N9 born-unbound warning dropped | 1 failed, 53 passed in 2.33s | a_first_forest_with_no_predecessor_no_trees_no_windows_ |
| KILLED | N10a swallow back: tree synapse | 1 failed, 53 passed in 2.33s | partial_write_is_rolled_back_loudly_and_truthfully[bind |
| KILLED | N10b swallow back: window synapse | 1 failed, 53 passed in 2.24s | partial_write_is_rolled_back_loudly_and_truthfully[bind |
| KILLED | N10c swallow back: hyperedge (debug) | 2 failed, 52 passed in 2.55s | partial_write_is_rolled_back_loudly_and_truthfully[bind, the_site_counter_accumulates_per_site |
| KILLED | N10d swallow back: window chain | 2 failed, 52 passed in 2.33s | a_failed_exact_repeat_failing_after_the_hyperedge_creat, partial_write_is_rolled_back_loudly_and_truthfully[bind |
| KILLED | N10e swallow back: sequence link | 1 failed, 53 passed in 2.15s | partial_write_is_rolled_back_loudly_and_truthfully[bind |
| KILLED | N11 True after a failed tree-bind (swallow tree site AND no post-condition) | 6 failed, 48 passed in 2.89s | a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_ |
| KILLED | N12 the drain truncates a failed entry with the flag on | 6 failed, 48 passed in 2.57s | 805_proof_a_turn_that_fails_every_cycle_stays_byte_iden, 805_proof_failed_turn_then_a_retry_with_different_conce |
| KILLED | N13a want swallow back: surface_wants | 2 failed, 52 passed in 2.36s | ], ] |
| KILLED | N13b want swallow back: surface_wants_for_graph | 2 failed, 52 passed in 2.42s | ], ] |
| KILLED | M20 rollback leaves the hyperedge (C3/F11) | 1 failed, 53 passed in 2.46s | a_failed_exact_repeat_failing_after_the_hyperedge_creat |
| KILLED | F1 re-stamp restore skipped entirely (C4) | 4 failed, 50 passed in 3.06s | a_held_exact_repeat_poison_turn_leaves_the_pre_existing, a_held_exact_repeat_poison_turn_leaves_the_pre_existing |
| KILLED | F2 stamp left in place: no snapshot taken (C4) | 4 failed, 50 passed in 2.47s | a_held_exact_repeat_poison_turn_leaves_the_pre_existing, a_held_exact_repeat_poison_turn_leaves_the_pre_existing |
| KILLED | F3 restore metadata only (threshold/excitability left re-stamped) | 3 failed, 51 passed in 4.91s | a_held_exact_repeat_poison_turn_leaves_the_pre_existing, a_held_exact_repeat_poison_turn_leaves_the_pre_existing |
| KILLED | F4 vdb entry of a pre-existing node not restored | 1 failed, 53 passed in 4.60s | a_pre_existing_vdb_entry_that_DIFFERS_from_what_the_dep |
| KILLED | F5 metadata restored by replacing the dict (breaks the vdb-shared reference) | 2 failed, 52 passed in 2.54s | a_held_exact_repeat_poison_turn_leaves_the_pre_existing, a_held_exact_repeat_poison_turn_leaves_the_pre_existing |
| KILLED | F6 exact embedding put-back dropped (a re-insert re-normalises) | 3 failed, 51 passed in 2.53s | a_held_exact_repeat_poison_turn_leaves_the_pre_existing, a_held_exact_repeat_poison_turn_leaves_the_pre_existing |
| KILLED | V1 delete SKIPPED when the vdb probe raised for a NEW node (checker C2) | 1 failed, 53 passed in 2.29s | a_raising_vdb_get_for_a_NEW_node_leaves_no_orphan_after |
| KILLED |  | V2 a pre-existing node whose probe raised has its entry DELETED (never-delete violated) 1 failed, 53 passed in 2.37s | a_raising_vdb_get_for_a_PRE_EXISTING_node_never_deletes |
| KILLED | D1 _deposit ignores the False | docs[3 failed, 73 passed in 6.03s] ng[54 passed in 2.12s] | a_false_from_the_dual_pass_is_reported_counted_and_warn, the_report_is_made_after_the_graph_lock_is_released |
| KILLED | D2 the daemon never passes the flag | docs[2 failed, 74 passed in 5.68s] ng[1 failed, 53 passed in 2.28s] | flag_on_the_drain_is_called_with_hold_on_failure_true[N, flag_on_the_drain_is_called_with_hold_on_failure_true[p |
| KILLED | D3 pointer persisted but NOT loaded | docs[4 failed, 72 passed in 5.96s] ng[1 failed, 53 passed in 2.26s] | init_ng_on_a_clean_start_with_no_sidecar_starts_with_no, init_ng_restores_the_persisted_pointer_into_the_fresh_s |
| KILLED | D4 pointer loaded WITHOUT the id-exists check | docs[3 failed, 73 passed in 5.42s] ng[54 passed in 2.42s] | a_stale_id_not_in_the_graph_is_none_with_one_warning_an, init_ng_with_a_stale_or_corrupt_pointer_starts_with_non |
| KILLED | D5 the default is flipped on in code (absent = True) | docs[2 failed, 74 passed in 8.45s] ng[54 passed in 3.00s] | hold_env_clear_values[env0-False], init_reads_the_environment_once_into_the_constant[None- |
| KILLED | D6 pointer read AFTER the save (runs ahead of the saved graph) | docs[1 failed, 75 passed in 5.71s] ng[54 passed in 2.06s] | guarded_save_reads_the_pointer_before_the_save |
| KILLED | D7 the flag is read at IMPORT again (C8/F8) | docs[6 failed, 70 passed in 5.84s] ng[54 passed in 2.50s] | init_reads_the_environment_once_into_the_constant[garba, main_configures_logging_first_and_reads_the_flag_exactl |
| KILLED | D8 main() never reads the flag (C8/F8) | docs[1 failed, 75 passed in 7.14s] ng[54 passed in 2.52s] | main_configures_logging_first_and_reads_the_flag_exactl |
| KILLED | D9 the flag is read BEFORE logging is configured (order swapped) | docs[1 failed, 75 passed in 5.67s] ng[54 passed in 2.00s] | main_configures_logging_first_and_reads_the_flag_exactl |
| KILLED | D10 a MISSING sidecar is a WARNING again (item 6) | docs[3 failed, 73 passed in 6.55s] ng[54 passed in 2.22s] | a_missing_file_is_none_with_one_INFO_and_no_warning, init_ng_on_a_clean_start_with_no_sidecar_starts_with_no |
| KILLED | D11 a CORRUPT sidecar downgraded to INFO (item 6) | docs[4 failed, 72 passed in 5.97s] ng[54 passed in 2.48s] | a_corrupt_file_is_none_with_one_warning_and_the_content, a_corrupt_file_is_none_with_one_warning_and_the_content |
| KILLED | D12 a STALE id downgraded to INFO (item 6) | docs[3 failed, 73 passed in 5.64s] ng[54 passed in 2.19s] | a_stale_id_not_in_the_graph_is_none_with_one_warning_an, init_ng_with_a_stale_or_corrupt_pointer_starts_with_non |

**39 of 39 KILLED.** The fold-up's required ones: M20 (C3), the missing case back at WARNING (D10), a corrupt file downgraded to INFO (D11), a stale id downgraded to INFO (D12), the restore skipped (F1), the stamp left in place with no snapshot (F2), the delete skipped when the probe raised (V1), the restore skipped after a completed bind (N5, now also killed by the two completed-bind tests); plus the flag read at import again (D7), main() never reading it (D8), the read before logging (D9), a never-delete violation (V2), a partial restore (F3), the vdb restore skipped (F4, the survivor, now killed), the shared-dict reference broken (F5) and the exact embedding put-back dropped (F6).

## Consequences (stated, not decided)
1. **A poison turn is now truly traceless**, but it still blocks the tract behind it and still costs two WARNINGs and one dual-pass attempt per cycle (build-001's volume figures stand). Only the trace on pre-existing nodes changed.
2. **Cost:** one shallow copy of the metadata dict (references to its values, including the `poincare_dir` bytes; not a deep copy) per PRE-EXISTING node per deposit, taken under the mutation lock the deposit already holds. New nodes (the normal case) take no snapshot. I did not measure it on a real 6,000-node graph.
3. **A residual the record now states:** if the vdb probe raised for a PRE-EXISTING node, its entry is kept (never deleted) and was re-written by the deposit; it cannot be restored (there is no snapshot of what the probe could not read). That is what `vdb_kept_unprovable=<n>` reports. For a NEW node whose probe raised, the rule is the brief's (the entry the call inserted is deleted); if that id also happened to be one of the #898 orphan vdb entries (graph node gone, entry present), the call had overwritten it anyway, so the entry goes.
4. **C8:** the constant is `False` until `main()` runs. Nothing in this daemon reads it before `main()` (the autosave loop starts after `init_ng`); a test or tool that imports the module and drives the autosave loop without `main()` sees `False` unless it sets it (my tests do).
5. **Item 6:** a clean first boot now logs INFO, so anyone alarming on WARNING in `daemon.log` will not see "no sidecar yet". The four defect cases still alarm.
6. Everything else from build-001 stands unchanged: merge both NG branches first, set the flag only after the running NG has `hold_on_failure` and the rollback, the stepping contradiction, no hold counter, the flag-off LAW 7 exposure. The le-050 F9 start-up refusal (#921) and the other items listed as "do not touch" are NOT in this batch.

## Not verified
- A real embedder, a real TID, a real daemon process, a real checkpoint reload: the "pre-existing vdb entry that differs" is a SYNTHETIC entry shaped like a checkpoint-loaded one (its own metadata dict, older content, a non-unit embedding).
- What class the live daemon's `vector_db` is at runtime (the restore uses `get`/`insert`/`delete` and, guarded, the `embeddings` dict of `SimpleVectorDB`); a store without that dict still restores content/metadata through `insert`.
- Metadata values that are mutable and mutated IN PLACE by a deposit: I read `_cc_deposit_memory_node` (it only assigns keys, from a fresh dict per call), so a shallow copy is exact for it; any other writer of a pre-existing node's metadata between the snapshot and the rollback (a concurrent mutator) is the untested concurrency case from le-050 NV8.
- The merged stack (D24 + #794 + #904 + this fold-up), the VPS host, Leg 2, the S4b gate. Nothing was merged, started or wired.
- The LIGHT look (one fresh law enforcer) that follows the fold-up has not happened.

## Reproduce
NG: `cd <NG worktree> && HOME=<scratch home> PYTHONUSERBASE=/home/josh/.local HF_HUB_OFFLINE=1 Z12_904_DAEMON_UNDER_TEST=<docs worktree>/scripts/cc-ng-daemon.py python3 -m pytest -p no:cacheprovider -q tests/test_cc_bind_atomic_904.py` (add `Z12_904_ORG_UNDER_TEST=<scratch cc_ng_organism.py>` for a base or a mutant; the scratch copy must live under `/tmp/z12-904-scratch` or `/tmp/z12-904-fold-scratch`). Docs: `python3 -m pytest -p no:cacheprovider -q scripts/tests/test_cc_ng_daemon_deposit_904.py`. Mutations: `build-002-mutate.py` (edit its `SCR` path first; it asserts each replacement matches once).
