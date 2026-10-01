<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane dual-pass-atomic-904, dispatch #13437) — build-003 return: #904 ROUND 2 (le-053)
#   What: the return for the round-2 follow-up: F1-1 (the verdict sentence says what was KEPT), F1-2 (a NARROW, conditional restore), F1-3 (a tie test), F1-4 (the unrestorable count is printed), Q2 (an honest missing-sidecar message), Q1 (REPORT ONLY).
#   Why: le-053 (the light look on the build-002 fold-up) found ETHOS DRIFT (light), no LAW violation, no blocking correction; Chief-003 folded F1-1..F1-4 and Q2 and asked for a report on Q1.
#   How: ONE code commit per repo on top of the previous tip (no rebase, no push, nothing merged); this file is the docs-only return in the NG worktree.
# -------------------
-->

# #904 build-003 — round 2: the restore is narrow and conditional, the WARNING says what was KEPT

Links: [[NeuroGraph]] - [[Dual-Pass Embedding]] - [[The River]] - `build-001.md`, `build-002.md` - brief `build-904-foldup.md` ROUND 2 (docs `d0cb86f1`) - review `le-053-904-fold.md`.

## READ FIRST

**Hashes (real `git rev-parse HEAD`; local only, no upstream, nothing merged, nothing pushed):**

| Repo / branch | Parent | Round-2 code commit |
|---|---|---|
| NG `cc-laptop-bind-atomic-904-20261001` | `c0511940eb54bdcd70468a5f7eb16c04c3c8bf44` (the build-002 return, on top of the fold-up code `12d6241b03fe4a35b050b73347cfaee145ffe518`) | `59d332c754594755ac95bca5d827fc79eff3c6b6` |
| docs daemon `cc-laptop-daemon-deposit-904-20261001` | `c3492dcfdac737c991db39ce9ac513e3157535c5` | `8811e97a7d2a63d24f34059deacf8f7f2bbf580c` |
| NG, this return | | the docs-only commit that adds this file (`git log -1` on the NG branch) |

**All of it is done; one STOP was NOT needed:** narrowing the restore did not require any edit to `_cc_deposit_memory_node` (it stays AST-pinned and unedited; `tests/test_cc_capture_mutations_423.py` passes UNMODIFIED, `git diff` on it is empty). **47 of 47 mutations KILLED** on the final files.

1. **F1-1** — the caller's WARNING no longer says "rolled back, nothing of this turn remains" while `vdb_kept_unprovable > 0`; it says `rolled back; kept: <n> pre-existing vdb entr(y/ies) that could not be proven (re-written by this call, not restored)`. Class/count only.
2. **F1-2** — the restore is NARROWED to exactly what the failed call wrote and made CONDITIONAL: a field is put back only while it still holds the value this call wrote. A probation sweep that lands between the snapshot and the rollback is NOT erased, and neither is any key the deposit never writes.
3. **F1-3** — a test ties the restore's field list to `_cc_deposit_memory_node`'s write-set by AST.
4. **F1-4** — `unrestorable=<n>;` is printed (after the existing `vdb_kept_unprovable=<n>;`, so every earlier substring is unchanged).
5. **Q2** — the missing-sidecar INFO now says only "no pointer sidecar found" (the MESSAGE option, not the check; why below).
6. **Q1** — REPORT ONLY, below. Nothing was changed or folded.

**Process (Exec P500) — what happened with the gate.** Every run went through `grun.sh`, which RECORDS the start load and REFUSES at >= 6 (a gate: it does not wait, and a refusal stops the queue). It refused **seven** times and nothing was started on those: start load **9.32** (the first docs and NG test runs), **9.14**, then during the mutation table at **6.99**, **6.78**, **6.05** and **7.08** (the table is therefore split over five invocations; each resumed by name, none re-run after code changed: the code and tests were frozen before the first mutation run). The machine was loaded by other sessions (a full test suite and several `pytest` jobs that are not mine). Between refusals I waited with a notify-only check (it starts no test) and then started a fresh gated attempt by hand. One more event: my first launch of the gated runner hit a bug in the runner script itself (`env` option order); `env` failed instantly, so no test ran (it is not a run; I fixed the script and re-ran). Not run, as instructed: the NG full suite and `tests/test_cc_deposit_step.py`. No daemon, socket, VPS or live data.

## My own first reading (written BEFORE I opened le-053) and how le-053 compares

Re-reading my build-002 code at its tip I found, before any review: **(a)** the restore did `metadata.clear()` + `update(whole old dict)` and reset threshold/excitability unconditionally, so it could erase a change the probation sweep made between the snapshot and the rollback (the sweep rewrites `probation_remaining`, `graduated`, `probation_expired_unfired`, `threshold`, `intrinsic_excitability` under the same lock): a violation of #904's own "rollback exactly what the call wrote"; **(b)** the sentence "rolled back, nothing of this turn remains" was printed beside `vdb_kept_unprovable > 0`; **(c)** `unrestorable` was counted but never printed. le-053 found the same three (F1-2, F1-1, F1-4) and added F1-3 (no tie test; ROUND 2 already listed it), Q1, Q2 and the C8 residual. **Nothing in le-053 contradicted my reading; nothing I found was missing from it.**

## What changed (file:line, final code)

### NG `cc_ng_organism.py`
- `_CCConversationalDualPassEco`: the restore's field list as class constants `_STAMPED_ATTRS` (`:2076`) and `_STAMPED_META_KEYS` (`:2077`) (+ `_ABSENT`).
- `deposit` (`:2097`): the snapshot is now returned and, in a `finally`, `_record_written` captures what the deposit wrote (also when it raised).
- `_snapshot_existing` (`:2126`): only the write-set: the two attributes, the four stamped keys and every key of this call's `meta`; a key absent before is remembered as absent.
- `_record_written` (`:2149`): the values the deposit left (the "wrote" values).
- `rollback` (`:2168`; `same()` `:2210`; restore loop `:2215`): last-in-first-out; each field is restored only if `same(current, wrote)`; a key absent before is deleted; the vdb entry's original embedding/content/metadata references are put back as before. Any failure, or an unrestorable snapshot, still makes it return False.
- The failure record (`:2589` onward): the honest `kept:` sentence and `unrestorable=%d;`.
- `_cc_deposit_memory_node` (`:1969`): **unchanged**.

### Docs daemon `scripts/cc-ng-daemon.py`
- `_load_conv_pointer`'s `FileNotFoundError` branch (`:1276`): "no pointer sidecar found; the first forest after this start has no predecessor link" at INFO, fixed reason code, no path/value. The four WARNING branches are unchanged.

## Why the restore is conditional (F1-2), and its stated limit
"Restore only those keys" is not enough on its own: the deposit itself writes `threshold`, `intrinsic_excitability` and `probation_remaining`, the same fields the sweep rewrites, so an unconditional per-key restore would still overwrite a sweep that landed between the snapshot and the rollback. The eco therefore records what the deposit wrote and restores a field only while it still holds that value. **Limit, stated:** if the sweep did rewrite one of those fields in between (e.g. `probation_remaining` one lower than the call's reset), the field is left as the sweep left it, so a failed attempt's reset can survive in that field. That is the price of never erasing another writer; the other written fields are still restored. (le-053 NV3: its interleaving was not demonstrated; mine is SIMULATED inside the failing call, under the same lock, not observed from a real second thread.)

## Why the MESSAGE for Q2, not the check
The smallest honest option the brief named. A check ("the graph already holds `cc:conv::` forests") is a heuristic that can misfire (forests restored from a checkpoint that predates the sidecar are normal on the first boot after #904), would add behaviour to a function the Chief ruled INFO, and the function still could not tell a lost sidecar from a first boot reliably. The message now claims only what is known. A lost sidecar after a save is therefore logged at INFO, exactly like a first boot: stated, and pinned by a test; if the Chief wants a louder signal for that one case, the check is the follow-up.

## Q1 — REPORT ONLY (nothing was changed or folded): the success-path probation re-stamp

**Current behaviour (file:line at `12d6241b` + this round; unchanged by it).**
- The stamp is written by `_cc_deposit_memory_node` (`cc_ng_organism.py:1969`): for a node that ALREADY exists it takes the `else` branch (`node.metadata.update(meta)`, `:1979`) and then, for new and pre-existing nodes alike and with no condition, sets `node.threshold = default_threshold + _CC_CONV_THRESHOLD_BOOST` (`:1981`), `node.intrinsic_excitability = _CC_CONV_NOVELTY_DAMPENING` (`:1982`), `node.metadata["probation_remaining"] = _CC_CONV_PROBATION_PERIOD` and `["probation_total"] = _CC_CONV_PROBATION_PERIOD` (`:1983-1984`), `["novelty_dampening"]` (`:1985`), re-packs `poincare_dir`, and re-writes the node's vdb entry.
- It runs on EVERY successful dual pass, because every deposit goes `run_conversational_dual_pass` -> `_CCConversationalDualPassEco.deposit` (`:2536` windows; `:2250`/`:2253` forest and trees via `record_outcome`) -> `_cc_deposit_memory_node`. Because `target_id` is content-hashed (`cc:conv::<sha1(text)>`, trees `<forest>::tree::<concept>`), an exact-repeat turn lands on the SAME forest/tree/window nodes, so each of them goes back to the START of its probation window: threshold boosted, excitability dampened, `probation_remaining` back to the full period. The sweep then walks it down again: `cc_update_probation` (`:2395`) decrements `probation_remaining` once per autosave pulse, releases the dampening at 0, and stamps `graduated` only on evidence of firing (#93).

**Intended semantics, per the code's own words.**
- `tests/test_cc_dual_pass.py:181-186` (`test_raw_deposit_on_exact_repeat_reuses_content_hashed_node`): "LAW 7 (post-gate-removal): an exact-repeat turn deposits raw onto the same node -- via ordinary content-hashing, not via any redundancy check or reinforcement branch (both deleted). `_cc_deposit_memory_node` **resets probation/threshold/excitability unconditionally on every call**, so the second deposit looks identical to the first: no special-cased path." That is a statement that the unconditional reset on re-deposit is the DESIGNED consequence of "raw deposit, no special case".
- The Lane B changelog entry in `cc_ng_organism.py` (`:346-383`, Exec Packets 153(3)(a)/153(4) Q2/155): the KISS Delta Gate, the #523 deposit-time skip and `_cc_kiss_reinforce_node` were REMOVED so that "every turn now deposits raw straight to the fresh-deposit path that was already there. No replacement redundancy check, dedup, or threshold was added at deposit (Packet 154(6): fix, never pass or patch)", because a deposit-time check is "a LAW 7 break: Packet 112(4), 'Shaping IS a Law violation at deposit'". It added "tests asserting raw deposit on repeat".
- `_cc_deposit_memory_node`'s docstring: "Mirrors canonical's `_deposit_memory_node`"; the canonical (`neurograph_rpc.py:2480-2518`, Syl's) carries the same unconditional stamp and says it "Applies light novelty-dampening (graduated by `_update_probation`)".
- `cc_update_probation`'s docstring (`:2395-2410`): it "fades novelty-dampening over the probation window", "operates on ALL probationary nodes, not just ones just deposited", and was a near-verbatim port of canonical `_update_probation`.

**Do the two agree?** On the success path, yes: the code does exactly what its tests and the Lane B ruling describe (a repeat is an ordinary deposit, and an ordinary deposit starts probation). What I could NOT find anywhere is a statement of INTENT for the *probation window* specifically on a repeat ("re-experiencing a memory should (or should not) re-enter novelty dampening"): the only explicit sentence is the test docstring above, which reads as a description of a consequence ("so the second deposit looks identical to the first"), not a design goal for probation. The Lane B record settles that no repeat-aware special case may be added at deposit (LAW 7), but it does not say whether the stamp itself should be conditional on the node being new. So: current behaviour is documented and deliberate as "no special case"; whether the probation re-entry is DESIRED is not decided by any text I found. That is the Chief's ruling. **Relevance to #904:** the failure-path restore exists only because this success-path stamp exists; if the Chief rules the stamp should not re-run on a repeat, the fix lives at the stamp (LAW 4), `_cc_deposit_memory_node` is AST-pinned and the canonical (Syl's) twin would need the same decision, and most of the restore would then be unnecessary. Nothing was changed or folded either way.

## Tests changed
**NG `tests/test_cc_bind_atomic_904.py`** (+7): `test_a_probation_sweep_between_the_snapshot_and_the_rollback_is_not_erased` (F1-2), `test_a_key_the_failed_call_added_is_deleted_and_the_key_order_is_preserved`, `test_the_restores_field_list_is_tied_to_the_deposits_write_set` (F1-3, AST), `test_the_rolled_back_sentence_says_what_was_KEPT_...` (x2: plural 3 entries and singular 1 entry; F1-1), `test_the_nothing_remains_sentence_is_kept_when_nothing_was_kept`, `test_the_unrestorable_count_is_printed_and_a_snapshot_that_cannot_be_taken_is_an_error` (F1-4). Every earlier test is unchanged and passes (the byte-identity-over-N-cycles tests, the differing-vdb-entry test, the double deposit, the `vdb.get()` tests, M20).
**Docs `scripts/tests/test_cc_ng_daemon_deposit_904.py`** (+1, one updated): `test_a_missing_file_is_none_with_one_INFO_and_no_warning` now also pins the wording ("no pointer sidecar found", never "clean start"); new `test_a_sidecar_lost_after_an_earlier_save_is_logged_no_more_confidently_than_a_first_boot`.

## Runs (private scratch root `/tmp/z12-904-fold-scratch`; `HOME` under it; `PYTHONUSERBASE=/home/josh/.local`; `HF_HUB_OFFLINE=1`; `NG_EMBED_REMOTE` unset; `-p no:cacheprovider`; one run per file; START LOAD recorded; gate < 6)
"Build-002" = the previous tip (`12d6241b` NG / `c3492dcf` daemon) via scratch copies or a `git archive` export. P379/#770: both of my files print the resolved paths; every NG module resolved under THIS worktree (scratch org only for build-002 and mutation runs); the docs file loaded no real NG module.

| File | Start load | Fix | Start load | Build-002 |
|---|---|---|---|---|
| NG `tests/test_cc_bind_atomic_904.py` (real daemon loaded) | 3.63 | **61 passed** | 3.61 | 5 failed, 56 passed (the sweep test, the tie test (no constants yet), the KEPT sentence x2, the printed count) |
| same, WITHOUT the daemon path | 2.98 | 56 passed, 5 skipped | | |
| NG `tests/test_cc_drain_hold_on_failure.py` | 2.82 | **49 passed** | | not re-run (the #794 file, untouched) |
| NG `tests/test_cc_drain_pacing.py` | | not present on this base | | |
| NG `tests/test_cc_capture_mutations_423.py` | 2.92 | **10 passed, UNMODIFIED** | | |
| NG `tests/test_cc_want_bounds.py` | 2.93 | 8 passed | | |
| docs `scripts/tests/test_cc_ng_daemon_deposit_904.py` | 4.02 (unset) / 3.46 (both set) | **77 passed** / 77 passed | 3.88 | 2 failed, 75 passed (the two Q2 wording tests) |
| docs `..._drain_pacing.py` | 3.57 (unset) / 2.70 (BATCH_SIZE=25 + IDLE_STEPS=250) | 46 / 46 | | |
| docs `..._recall_status.py` | 3.45 (unset) / 2.81 (both set) | 41 / 41 | | |
| NG `test_cc_gateway_durable` | 3.45 | 28 passed | 3.23 | 28 passed |
| NG `test_cc_durable_integration` | 3.60 | 3 passed | 3.39 | 3 passed |
| NG `test_cc_host_stop_door` | 3.39 | 4 passed | 3.20 | 4 passed |
| NG `test_conversational_window_chains` | 3.20 | 4 passed | 3.59 | 4 passed |
| NG `test_cc_callosum_leg1` | 3.59 | 16 passed | 3.46 | 16 passed |
| NG `test_cc_pattern_completion_recall` | 3.46 | 1 failed, 7 passed | 3.23 | identical failing test |
| NG `test_cc_retrieval_enrichment` | 3.38 | 2 failed, 14 passed | 3.25 | identical failing tests |
| NG `test_cc_dual_pass` | 3.06 | 10 failed, 11 passed | 3.05 | identical 10 failing tests |

One correction of my own: my first queue gave the docs "env set" runs only `CC_NG_BATCH_SIZE`; I re-ran the three docs files with BOTH variables set (rows above; same numbers). The three red regression files' failing sets were compared BY NAME and are identical on both trees.

## Mutations (scratch copies; every replacement asserted to match EXACTLY once; the per-run start loads are in the third column; the whole table run on the frozen final files)
| Verdict | Mutation | Start load + result | Killed by (first two) |
|---|---|---|---|
| KILLED | N1 failure logged at debug again | start_load=3.31 17 failed, 44 passed in 4.52s | 805_proof_a_turn_that_fails_every_cycle_stays_byte_iden, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_ |
| KILLED | N2 returns True after a partial write | start_load=3.20 39 failed, 22 passed in 8.60s | 805_proof_a_turn_that_fails_every_cycle_stays_byte_iden, 805_proof_failed_turn_then_a_retry_with_different_conce |
| KILLED | N3 rollback skipped | start_load=3.94 31 failed, 30 passed in 6.18s | 805_proof_failed_turn_then_a_retry_with_different_conce, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_ |
| KILLED | N4 vdb entries not deleted on rollback | start_load=4.46 21 failed, 40 passed in 4.61s | 805_proof_failed_turn_then_a_retry_with_different_conce, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_ |
| KILLED | N5 state pointer not restored | start_load=4.19 5 failed, 56 passed in 2.83s | a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_ |
| KILLED | N6 forest-absent returns silently again | start_load=4.01 1 failed, 60 passed in 2.65s | partial_write_is_rolled_back_loudly_and_truthfully[bind |
| KILLED | N7 the post-condition (truth rule) removed | start_load=3.93 5 failed, 56 passed in 2.83s | a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_ |
| KILLED | N8 site counter dropped | start_load=3.70 10 failed, 51 passed in 5.90s | a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_ |
| KILLED | N9 born-unbound warning dropped | start_load=4.44 1 failed, 60 passed in 2.54s | a_first_forest_with_no_predecessor_no_trees_no_windows_ |
| KILLED | N10a swallow back: tree synapse | start_load=4.24 1 failed, 60 passed in 2.37s | partial_write_is_rolled_back_loudly_and_truthfully[bind |
| KILLED | N10b swallow back: window synapse | start_load=3.98 1 failed, 60 passed in 3.10s | partial_write_is_rolled_back_loudly_and_truthfully[bind |
| KILLED | N10c swallow back: hyperedge (debug) | start_load=3.83 3 failed, 58 passed in 4.23s | partial_write_is_rolled_back_loudly_and_truthfully[bind, the_rolled_back_sentence_says_what_was_KEPT_when_a_vdb_ |
| KILLED | N10d swallow back: window chain | start_load=4.08 2 failed, 59 passed in 2.96s | a_failed_exact_repeat_failing_after_the_hyperedge_creat, partial_write_is_rolled_back_loudly_and_truthfully[bind |
| KILLED | N10e swallow back: sequence link | start_load=3.99 1 failed, 60 passed in 2.71s | partial_write_is_rolled_back_loudly_and_truthfully[bind |
| KILLED | N11 True after a failed tree-bind (swallow tree site AND no post-condition) | start_load=3.99 6 failed, 55 passed in 4.12s | a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_, a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_ |
| KILLED | N12 the drain truncates a failed entry with the flag on | start_load=3.91 6 failed, 55 passed in 3.04s | 805_proof_a_turn_that_fails_every_cycle_stays_byte_iden, 805_proof_failed_turn_then_a_retry_with_different_conce |
| KILLED | N13a want swallow back: surface_wants | start_load=4.08 2 failed, 59 passed in 2.95s | ], ] |
| KILLED | N13b want swallow back: surface_wants_for_graph | start_load=3.99 2 failed, 59 passed in 2.77s | ], ] |
| KILLED | M20 rollback leaves the hyperedge (C3/F11) | start_load=3.75 1 failed, 60 passed in 2.55s | a_failed_exact_repeat_failing_after_the_hyperedge_creat |
| KILLED | F1 re-stamp restore skipped entirely (C4) | start_load=3.69 6 failed, 55 passed in 3.38s | a_held_exact_repeat_poison_turn_leaves_the_pre_existing, a_held_exact_repeat_poison_turn_leaves_the_pre_existing |
| KILLED | F2 stamp left in place: no snapshot taken (C4) | start_load=3.64 7 failed, 54 passed in 3.33s | a_held_exact_repeat_poison_turn_leaves_the_pre_existing, a_held_exact_repeat_poison_turn_leaves_the_pre_existing |
| KILLED | F3 restore metadata only (threshold/excitability left re-stamped) | start_load=3.43 3 failed, 58 passed in 3.74s | a_held_exact_repeat_poison_turn_leaves_the_pre_existing, a_held_exact_repeat_poison_turn_leaves_the_pre_existing |
| KILLED | F4 vdb entry of a pre-existing node not restored | start_load=3.71 1 failed, 60 passed in 3.22s | a_pre_existing_vdb_entry_that_DIFFERS_from_what_the_dep |
| KILLED | W1 restore UNCONDITIONALLY (a sweep between snapshot and rollback is erased) | start_load=3.74 1 failed, 60 passed in 3.26s | a_probation_sweep_between_the_snapshot_and_the_rollback |
| KILLED | W2 WHOLE-DICT restore (clear + refill: erases keys the call never wrote) | start_load=4.00 4 failed, 57 passed in 4.45s | a_held_exact_repeat_poison_turn_leaves_the_pre_existing, a_held_exact_repeat_poison_turn_leaves_the_pre_existing |
| KILLED | W3 a key the call ADDED is not deleted | start_load=4.24 1 failed, 60 passed in 4.36s | a_key_the_failed_call_added_is_deleted_and_the_key_orde |
| KILLED | W4 threshold/excitability restored but the metadata keys are not | start_load=5.40 5 failed, 56 passed in 3.47s | a_held_exact_repeat_poison_turn_leaves_the_pre_existing, a_held_exact_repeat_poison_turn_leaves_the_pre_existing |
| KILLED | S1 the "nothing of this turn remains" sentence is back beside kept entries (F1-1) | start_load=5.05 2 failed, 59 passed in 3.10s | the_rolled_back_sentence_says_what_was_KEPT_when_a_vdb_, the_rolled_back_sentence_says_what_was_KEPT_when_a_vdb_ |
| KILLED | U1 the unrestorable count is not printed (F1-4) | start_load=4.81 1 failed, 60 passed in 4.07s | the_unrestorable_count_is_printed_and_a_snapshot_that_c |
| KILLED | T1 the deposit gains a NEW stamped key (F1-3 tie) | start_load=4.66 1 failed, 60 passed in 2.72s | the_restores_field_list_is_tied_to_the_deposits_write_s |
| KILLED | T2 the restore list drops a key the deposit writes (F1-3 tie) | start_load=4.77 3 failed, 58 passed in 2.75s | a_key_the_failed_call_added_is_deleted_and_the_key_orde, a_probation_sweep_between_the_snapshot_and_the_rollback |
| KILLED | F6 exact embedding put-back dropped (a re-insert re-normalises) | start_load=4.47 3 failed, 58 passed in 2.61s | a_held_exact_repeat_poison_turn_leaves_the_pre_existing, a_held_exact_repeat_poison_turn_leaves_the_pre_existing |
| KILLED | V1 delete SKIPPED when the vdb probe raised for a NEW node (checker C2) | start_load=4.35 1 failed, 60 passed in 2.96s | a_raising_vdb_get_for_a_NEW_node_leaves_no_orphan_after |
| KILLED | V2 a pre-existing node whose probe raised has its entry DELETED (never-delete violated) | start_load=4.35 1 failed, 60 passed in 3.10s | a_raising_vdb_get_for_a_PRE_EXISTING_node_never_deletes |
| KILLED | D1 _deposit ignores the False | docs[start_load=4.32 3 failed, 74 passed in 8.42s] ng[start_load=4.82 61 passed in 4.82s] | a_false_from_the_dual_pass_is_reported_counted_and_warn, the_report_is_made_after_the_graph_lock_is_released |
| KILLED | D2 the daemon never passes the flag | docs[start_load=4.76 2 failed, 75 passed in 8.85s] ng[start_load=4.65 1 failed, 60 passed in 9.55s] | flag_on_the_drain_is_called_with_hold_on_failure_true[N, flag_on_the_drain_is_called_with_hold_on_failure_true[p |
| KILLED | D3 pointer persisted but NOT loaded | docs[start_load=5.97 4 failed, 73 passed in 9.91s] ng[start_load=5.75 1 failed, 60 passed in 5.91s] | init_ng_on_a_clean_start_with_no_sidecar_starts_with_no, init_ng_restores_the_persisted_pointer_into_the_fresh_s |
| KILLED | D4 pointer loaded WITHOUT the id-exists check | docs[start_load=4.41 3 failed, 74 passed in 6.94s] ng[start_load=4.63 61 passed in 2.34s] | a_stale_id_not_in_the_graph_is_none_with_one_warning_an, init_ng_with_a_stale_or_corrupt_pointer_starts_with_non |
| KILLED | D5 the default is flipped on in code (absent = True) | docs[start_load=4.34 2 failed, 75 passed in 6.93s] ng[start_load=4.31 61 passed in 3.91s] | hold_env_clear_values[env0-False], init_reads_the_environment_once_into_the_constant[None- |
| KILLED | D6 pointer read AFTER the save (runs ahead of the saved graph) | docs[start_load=4.36 1 failed, 76 passed in 8.78s] ng[start_load=4.50 61 passed in 7.96s] | guarded_save_reads_the_pointer_before_the_save |
| KILLED | D7 the flag is read at IMPORT again (C8/F8) | docs[start_load=5.94 6 failed, 71 passed in 11.04s] ng[start_load=5.88 61 passed in 3.88s] | init_reads_the_environment_once_into_the_constant[garba, main_configures_logging_first_and_reads_the_flag_exactl |
| KILLED | D8 main() never reads the flag (C8/F8) | docs[start_load=4.45 1 failed, 76 passed in 8.03s] ng[start_load=5.06 61 passed in 2.85s] | main_configures_logging_first_and_reads_the_flag_exactl |
| KILLED | D9 the flag is read BEFORE logging is configured (order swapped) | docs[start_load=5.06 1 failed, 76 passed in 6.87s] ng[start_load=5.27 61 passed in 3.22s] | main_configures_logging_first_and_reads_the_flag_exactl |
| KILLED | D10 a MISSING sidecar is a WARNING again (item 6) | docs[start_load=5.65 4 failed, 73 passed in 11.50s] ng[start_load=5.88 61 passed in 5.58s] | a_missing_file_is_none_with_one_INFO_and_no_warning, a_sidecar_lost_after_an_earlier_save_is_logged_no_more_ |
| KILLED | D11 a CORRUPT sidecar downgraded to INFO (item 6) | docs[start_load=4.26 4 failed, 73 passed in 5.86s] ng[start_load=3.83 61 passed in 2.77s] | a_corrupt_file_is_none_with_one_warning_and_the_content, a_corrupt_file_is_none_with_one_warning_and_the_content |
| KILLED | Q2 the INFO claims "clean start" again (round 2) | docs[start_load=3.84 2 failed, 75 passed in 6.92s] ng[start_load=4.08 61 passed in 2.33s] | a_missing_file_is_none_with_one_INFO_and_no_warning, a_sidecar_lost_after_an_earlier_save_is_logged_no_more_ |
| KILLED | D12 a STALE id downgraded to INFO (item 6) | docs[start_load=4.08 3 failed, 74 passed in 5.51s] ng[start_load=3.61 61 passed in 2.15s] | a_stale_id_not_in_the_graph_is_none_with_one_warning_an, init_ng_with_a_stale_or_corrupt_pointer_starts_with_non |

**47 of 47 KILLED.** Build-002's 39 (re-run: F5 "metadata restored by replacing the dict" is retired because the restore no longer clears the dict; the F2/F3 texts follow the new code) plus the round-2 mutants: **W1** restore unconditionally (a sweep is erased), **W2** whole-dict restore (clear + refill: erases keys the call never wrote), **W3** a key the call added is not deleted, **W4** the attributes restored but not the metadata keys, **S1** the "nothing of this turn remains" sentence back beside kept entries, **U1** the unrestorable count not printed, **T1** the deposit gains a new stamped key (the tie test), **T2** the restore list drops a key the deposit writes (the tie test), **Q2** the INFO claims "clean start" again. The brief's required "restore of the whole dict, KILLED" is W2 (killed by the sweep test and the byte-identity tests).

## Consequences (stated, not decided)
1. **A failed attempt can still leave one trace**, by design now: a field the sweep rewrote between the snapshot and the rollback keeps the sweep's value (see the stated limit). The byte-identical-over-N-cycles guarantee holds for every field nobody else touched.
2. **The KEPT sentence is information, not a failure**: the level stays WARNING and `rollback()` still returns True for a pre-existing node whose vdb probe raised (the entry was re-written by this call and cannot be restored). If the Chief wants that to count as "a partial write remains" (ERROR, `ROLLBACK FAILED`), it is a one-line change to `ok`.
3. **Q2:** a sidecar lost after a save is INFO, like a first boot.
4. Everything from build-001 and build-002 stands (merge both NG branches first, the flag only after the running NG has `hold_on_failure` and the rollback, the stepping contradiction, no hold counter, the flag-off LAW 7 exposure, the C8 residual that the flag exists only after `main()`).

## Not verified
- The sweep interleaving as a REAL second thread: my test simulates the sweep's writes inside the failing call under the same lock; no concurrency was exercised (le-053 NV3 stands).
- A real embedder, a real TID, a real daemon process, a real checkpoint reload (every test uses fakes inside the tests); the live daemon's `vector_db` class at runtime.
- The cost of the extra per-deposit read of the write-set (one extra pass over ~6 fields plus the call's meta keys, only for PRE-EXISTING nodes): not measured on a real 6,000-node graph.
- The merged stack, the VPS half, Leg 2, the S4b gate; nothing was merged, started or wired.
- Q1: I found no text that states the INTENDED probation semantics for a repeat beyond the test docstring and the Lane B ruling cited; whether re-entering probation on re-experience is DESIRED is not decided by anything I read.

## Reproduce
NG: `cd <NG worktree> && HOME=<scratch home> PYTHONUSERBASE=/home/josh/.local HF_HUB_OFFLINE=1 Z12_904_DAEMON_UNDER_TEST=<docs worktree>/scripts/cc-ng-daemon.py python3 -m pytest -p no:cacheprovider -q tests/test_cc_bind_atomic_904.py` (add `Z12_904_ORG_UNDER_TEST=<scratch cc_ng_organism.py>` under `/tmp/z12-904-scratch` or `/tmp/z12-904-fold-scratch` for a base or a mutant). Docs: `python3 -m pytest -p no:cacheprovider -q scripts/tests/test_cc_ng_daemon_deposit_904.py`. Mutations: `build-003-mutate.py` (edit its `SCR`; it refuses to start any run at load >= 6 and records every start load).
