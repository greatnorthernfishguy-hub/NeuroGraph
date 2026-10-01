<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane ack-bound-918, dispatch #13930) — return build-002 for the #918 fold-up
# What: builder return for build-918-foldup.md (N6, T1, N3, N4 + ADDENDUM 1: Exec P522 Q1 and Q2).
# Why: the brief (docs worktree daemon-recall-756-20260930, handoffs/z12-silent-failure-lanes-20260930/assignments/build-918-foldup.md) requires this file.
# How: faithful report; every hash is a real `git rev-parse HEAD`; every run's START load is recorded; nothing pushed, nothing merged.
# -------------------
-->

# Return build-002 — #918 fold-up (Q1, Q2, N4 = behaviour; N6, T1, N3 = docs/test)

**Seat/authority:** worker, `worktree_write` (RESUMED my own thread). Did not self-accept; nothing pushed or merged; no rebase (no upstream anywhere).

## Hashes (real `git rev-parse HEAD`)
| repo | branch | commit | what |
|---|---|---|---|
| NG `/home/josh/NeuroGraph-worktrees/z12-ack-bound-918-20261001` | `cc-laptop-ack-bound-918-20261001` | `86cf5afd91806b4ee79a87313edfe11c0b41fc72` (code) / `678d03dea56f350b40afbbe62ecb133da9ef92c6` (return 001) | start (on top: the build-001 return commit) |
| NG | same | **`c3c76432e9b2d5fbad3db71e04be1945012c2aaf`** | the ONE NG fold-up commit (`cc_topology_merge.py` + 4 test files) |
| NG | same | the commit carrying THIS file (docs-only; `git log -1` of the branch: a file cannot hold its own commit's hash) | return |
| docs `/home/josh/docs/.claude/worktrees/callosum-truth-918-20261001` | `cc-laptop-callosum-truth-918-20261001` | `008a10affdc2f104dcd0cc4f5f163e33643f41ac` | start |
| docs | same | **`d8dc4f9b2772d9643594577369cdd2d36b1072ba`** | the ONE docs fold-up commit (`CC-CALLOSUM-TRUTH.md` only, +49/-24) |

The docs worktree now reports `ahead 2, behind 132` vs `origin/main` (other lanes' handoffs); it was NOT rebased, as instructed.

## READ first (my own code at `86cf5afd`, before any review text; I did not open `reviews/le-056-918.md`)
- T1: `tests/test_cc_ack_bound_918.py` (build-001 `:476`) was `assert ack <= tmg.cc_current_membership(lg) | ack`: a set is always a subset of itself unioned with anything, so it cannot fail whatever the code does.
- N4: build-001 `merge_cc_topology` reset the streak table only inline at the two early returns (`no_conduit`, `empty`, each a copy of `_reoffer_streaks = {}`); the four `raise TopologyMergeAbort` sites (MACHINE_ID unset, no header, own export, embedding-model mismatch) did not reset it.
- Q1: in `86cf5afd` the narrowed set lived under the name `cc_current_membership` (the ack writer's argument).

## What changed (NG `c3c76432`, `cc_topology_merge.py`; lines at that tip)
1. **Q1 (LAW 4, the NAME).** `cc_current_membership` `:250` is restored to the base meaning: `graph.nodes` filtered ONLY by `is_cc_provenance`. **Verified by AST: its code (docstring removed) is identical to `ee94f7d2`'s.** Its docstring is the base text verbatim plus ONE appended pointer paragraph ("the ACK is `cc_ack_membership`, this is 'what is currently held'"); I added that one paragraph because the base docstring's "authoritative source for the SENDER's exclude_ids" would otherwise be a false statement about this function (flag for the Chief: the brief said the docstring stays unchanged; I kept every base word and only appended). NEW pure function **`cc_ack_membership(graph)` `:277`** = `cc_current_membership(graph)` minus `_unbound_nodes(graph, held)` (called, never copied; no counter/log/state/write). **The ack writer calls it** (`_write_membership(membership_path, cc_ack_membership(graph))`, `:934`). The merge docstring (`:~507`), the `_track_reoffers` docstring, the Tier-1 "(After #110 ...)" comment and the end-of-call comment now name `cc_ack_membership`. The ack file's contents are unchanged from `86cf5afd` (the retargeted tests keep their assertions).
2. **Q2.** The docstring sentence "Protected nodes are deliberately excluded from re-offer (not sweep-eligible; H-1: Leg 2 never writes to identity nodes)." is in `cc_ack_membership`'s docstring and in the merge's end-of-call comment (`:~927-933`).
3. **N4 (behaviour).** ONE helper `_reset_reoffer_streaks()` `:319` (a module-level reassignment to a fresh `{}`: the table is replaced, never mutated). It is now used by the two early returns (`:531`, `:539`: the two inline copies are GONE) and by the four abort sites (`:526` MACHINE_ID unset, `:544` no header, `:551` own export, `:560` model mismatch). `merge_cc_topology`'s `global` line is gone (the helper owns the assignment).
4. Changelog header entry (top of the file); the build-001 entry is marked "NAME SUPERSEDED".

## Caller audit of the OLD name `cc_current_membership` (Q1 item 2), one by one — NONE wanted the narrowed meaning outside the ack role, so no STOP
Searched by symbol: the NG worktree (all `.py/.md/.sh/.json` outside `handoffs/`), the NG main checkout `/home/josh/NeuroGraph/{*.py,scripts,tests}`, `scripts/cc-ng-sync.py` and `scripts/cc-ng-daemon.py` (both the `heartbeat-source` and the `docs/scripts` copies), and `git grep` over the docs repo.
| caller | what it wanted | disposition |
|---|---|---|
| `cc_topology_merge.py` ack writer (was `:881`) | the ACK (what the sender reads) | moved to `cc_ack_membership` |
| `cc_topology_merge.py` Tier-1 comment (was `:609`) | prose about the ack | now names the ack function |
| `cc_topology_merge.py` `:173`-area #110 changelog entry | history | untouched (dated record) |
| `scripts/cc-ng-sync.py` (both copies) | 0 hits (it uses `_load_membership` only) | none |
| `scripts/cc-ng-daemon.py` (both copies) | 0 hits (it passes `membership_path`, reads stats) | none |
| NG main checkout `/home/josh/NeuroGraph` | the unmodified base: the writer + the same tests | none beyond the above |
| `tests/test_cc_topology_callosum.py` x5 (`test_journaled_but_culled_node_is_readmitted` x2, `test_membership_snapshot_feeds_sender_exclude_ids`, `test_culled_node_drops_out_of_exclude_ids_and_is_resent` x2) | each pins the ACK (the membership FILE or the sender's `exclude_ids`) | retargeted to `cc_ack_membership` |
| `tests/test_cc_topology_capture_423.py` (AST `names` set) | harness extraction | `cc_ack_membership` + `_reset_reoffer_streaks` added to the extracted names |
| `tests/test_cc_merge_whole_graph_guard.py` (spy caller-name pin) | "the second caller of the one predicate" | now `cc_ack_membership` (it is the function that calls `_unbound_nodes`; `cc_current_membership` no longer does) |
| `tests/test_cc_ack_bound_918.py` x14 | ack contents | retargeted; plus new tests for the base function |
| docs `CC-CALLOSUM-TRUTH.md` (3 passages + the new §9 entry) | prose about the ack | updated in the docs commit |
| docs `punchlist/open/ecosystem.md:11` (#918 row) and `:22` (#909 row), `handoffs/executive/EXECUTIVE-TODO.md` | name mentions | NOT edited (outside the truth-doc-only scope); only the start of the #918 row ("the ack must exclude sweep-eligible-unbound nodes") was seen |
| docs `ng_topology/*.tract` (binary) | the string appears in recorded conversation text | not code |
No production caller other than the ack writer exists.

## Tests (`tests/test_cc_ack_bound_918.py`: 35 test functions, 40 collected)
- **Moved/retargeted (assertions unchanged, never weakened):** every test that pins ack contents now calls `cc_ack_membership` (14 call sites; the in/out membership test, byte-identical-on-no-unbound, the `_unbound_nodes` spy test, the double-without-`_is_identity_protected` test, the ack-file test, the hyperedge/synapse binding tests, the loop and ack-after-every-tick tests, the counter-never-touches-ack test); the in-file CONTROL now patches `cc_ack_membership` back to the old "every CC node held".
- **Parametrized over both functions:** the pure-query test and the signature/return-type test.
- **New (Q1):** `:293` `cc_current_membership` is back to every CC node currently held (equal to the inlined base formula on a graph with unbound + bound + protected + non-CC nodes; also asserts `_unbound_nodes` does not appear in its source); `:307` `cc_ack_membership` = held minus the sweep-eligible-unbound set; `:312` composition spy (stand-ins for both functions are OBEYED: no copy of either); `:323` the ack writer calls `cc_ack_membership` (a mutant that calls the old name FAILS).
- **New (Q2) `:335`:** a constitutional node and a `*_authored` want, both unbound and connected on the VPS, are IN the ack and NOT in any frame's candidates (real exporter, that ack as `exclude_ids`: `candidates == 1`, frame == only the unprotected unbound node U); U is NOT in the ack and IS the candidate; after the merge U is bound and acked, the next tick has 0 candidates (`exhausted`); no edge was written onto either identity node.
- **T1 `:565`:** the tautology is replaced by `ack <= set(lg.nodes)` (the ack lists only nodes the receiver holds: a phantom id fails it) plus a monotone-growth check across ticks (`earlier <= later`; nothing is culled in that world, so a bound node never drops out); the final `acks[-1] == cc_ack_membership(lg)` check is kept.
- **N3 `:813` (reported, not papered over):** real exporter -> real merge, `max_nodes_per_call=1`, a frame of one held-unbound X + two NEW nodes in one VPS hyperedge. What the budget does: only NEW nodes spend budget (the call absorbs at most 1); a PRESENT node costs none, but the per-node budget check runs BEFORE the present check, so a present X placed after the spent budget is counted in `deferred_by_budget` (an accounting artefact, `completed` False) yet its edges are NOT lost because Tier 2/3 test `graph.nodes`: tick 1 leaves X unbound (a new member is still absent, the hyperedge is skipped), tick 2 lands the second new node and the hyperedge: **X is BOUND after the 2nd call**, everything is present and `completed` by the 3rd, and the ack then holds all three. The in-frame order follows the hyperedge's member-set order (hash-seed dependent), so the test asserts only the order-independent facts (my first draft pinned an exact per-tick log and FAILED on its first run for exactly that reason; I rewrote it). **This corrects a remark in my build-001 return** ("a deferred present node's hyperedge waits for the next call"): it waits only while a NEW member is still absent.
- **N4 `:881` (4 abort kinds) and `:900` (control):** N-1 = 4 consecutive re-offers, then a call that raises `TopologyMergeAbort` at each of the four sites (MACHINE_ID unset, no header, own export, model mismatch), then one re-offer: the table is empty after the abort, NO WARNING after it, streak 1. The control (no abort) keeps its streak and warns at N=5 (unchanged behaviour).
- **Existing-test expectation changes (file, test, why):** (1) `tests/test_cc_topology_callosum.py`: the 5 sites above, all pin the ACK, none weakened (its changelog header names them; one name there is imprecise: it says `test_readmitted...` where the test is `test_journaled_but_culled_node_is_readmitted`; not amended, to keep the ONE commit and the hashes the docs cite). (2) `tests/test_cc_topology_capture_423.py`: harness names only. (3) `tests/test_cc_merge_whole_graph_guard.py::test_d_the_guard_is_asked_about_every_node_in_the_graph_under_the_step_lock`: `("cc_current_membership", True)` -> `("cc_ack_membership", True)`: the caller of the one predicate is now the ack function; the guard's own assertions are unchanged.

## Runs (scratch HOME, `PYTHONUSERBASE=/home/josh/.local`, `HF_HUB_OFFLINE=1`, `NG_EMBED_REMOTE` unset, `-p no:cacheprovider`; EVERY run preceded by a 25 s pause and gated on its own START load: refuse at 1-min >= 6 or 5-min >= 5.0; NONE was refused; the gate script and the runner live in `/tmp/z12_918_scratch/`, outside the repo)
| run | START load1 / load5 | result |
|---|---|---|
| `test_cc_ack_bound_918.py` (first, before the N3 rewrite) | 2.23 / 3.24 | 1 failed (my N3 exact-log draft), 39 passed |
| `test_cc_ack_bound_918.py` (final) | 3.75 / 3.51 | **40 passed** |
| `test_cc_emergent_want_bound_905.py` | 4.06 / 3.61 | **60 passed** |
| `test_cc_merge_whole_graph_guard.py` (#897) | 4.53 / 3.76 | **17 passed, 2 skipped** (the 2 need `Z12_MERGE_BASE_REF`, unset: not run) |
| `test_cc_topology_callosum.py` | 5.52 / 4.08 | **33 passed, 1 failed**: `test_poincare_dir_is_rederived_locally_not_transmitted` (the stale #902 `poincare_dir`-bytes failure; it also fails on the untouched base, shown in build-001) |
| `test_cc_topology_frame.py` | 3.99 / 3.86 | **15 passed** |
| `test_cc_topology_capture_423.py` | 2.72 / 3.57 | **4 passed** |
| N3 hash-seed check, `-k n3`, `PYTHONHASHSEED` 1 / 2 / 3 | 2.02/2.45, 2.41/2.51, 1.89/2.38 | 1 passed each |
NOT run (forbidden): the NG full suite and `tests/test_cc_deposit_step.py`. The commit was made after the runs above passed and BEFORE the fail-first and mutant runs (a commit is not a run).

### N4 fail-first on `86cf5afd` (the old tip's `cc_topology_merge.py`, scratch copy, `Z12_MERGE_UNDER_TEST`; START 1.40 / 3.03)
`-k n4`: **4 failed (machine_id_unset, no_header, own_export, model_mismatch), 1 passed (the no-abort control)** -> the abort-reset tests FAIL on `86cf5afd` and pass on the fix (the full file passes on `c3c76432`).

### Mutants (scratch copies of `c3c76432`'s merge in `/tmp/z12_918_scratch/mut2/`; whole new file each; ALL 8 KILLED; none refused, none unrun)
| mutant | START load1 / load5 | killed by |
|---|---|---|
| G1 (T1) a phantom id added to the ack (`... \| {"cc:conv::phantom-not-held"}`) | 2.05 / 3.04 | 9 failed incl. `test_the_ack_after_every_tick_contains_only_bound_nodes` (its first assertion `ack <= set(lg.nodes)` is the T1 replacement) |
| G2 (N3) budget ignored (`budget = 10**9`) | 2.65 / 3.13 | 1 failed: the N3 test |
| G3 (N4) the abort-site resets removed | 2.69 / 3.09 | 4 failed: the 4 N4 abort cases |
| G4 (N4) the reset applied on the normal path too | 2.16 / 2.92 | 11 failed: warning-at-N, redaction, reset-on-bind/gap, env, cap, N4 x4 + control |
| G5 (Q1) the ack writer calls the old name | 1.78 / 2.75 | 9 failed incl. the writer-spy test, ack-file test, loop |
| G6 (Q1) `cc_current_membership` narrowed again | 1.76 / 2.66 | 2 failed: the "back to every CC node held" test, the `_unbound_nodes` spy test |
| G7 (Q1) `cc_ack_membership` includes the unbound set | 1.35 / 2.48 | 12 failed incl. the Q2 test, composition spy, loop |
| G8 (Q2) the protected node dropped from the ack (a predicate copy without the protection leg) | 1.74 / 2.48 | 7 failed incl. the Q2 test, byte-identical, composition spy, N3 |
Interpretation note: the brief's "the T1 replacement made vacuous again" cannot be a KILLED mutant if it means editing the test back to a tautology, so I took it as a CODE mutant that violates what the replacement pins (G1).

## Docs (docs commit `d8dc4f9b`, `CC-CALLOSUM-TRUTH.md` only; lines post-change)
- **N6(a), stale line references (each a dated bracketed note after the old reference; history stays readable):** merge `:575` -> `:934` (+ `_write_membership` `:411`); `:6,11-13` -> `:109-121`; `:118` -> `:227` (and the default is 25 since #88 §10.4-C); `:236-249, 542-557`, `:541-554`, `:552-567`, `:552`, `:479-498` (x2 passages) -> the guard `:833` and the skip/consolidate branch `:872-907`; `:59-62` -> the #108 header entry `:162-`; `:587` (`_unbound_nodes`) -> `:946`; export `:128` -> `:264`; export `:204` -> `:351`. Left as a record: the §3 pre-fix quote `:198-204` (code that no longer exists).
- **N6(b):** the §10.4-A VPS-tick bullet ("if the ack advanced, export; else no-op and wait") replaced with what the sender code does (read `export_cc_topology_frame` and `scripts/cc-ng-sync.py leg2-tick`, read only): no ack-advanced comparison; the same candidates are re-selected and the same frame re-exported when the ack has not moved; no-op only on unavailable daemon, resource gate, 0 candidates (exhausted), or none placeable. Josh's verbatim quote is kept and labelled as the design intent.
- **Q1/Q2/N4 naming and substance:** the banner (`:2`), the §3 *Update (#918)* paragraph (`cc_ack_membership` = `cc_current_membership` minus `_unbound_nodes`; the Q2 sentence; the abort reset), §10.4-C (`cc_ack_membership` + `_write_membership`), the §9 2026-10-01 entry (the function name), and a NEW §9 entry "2026-10-01 (later) ... fold-up (build-002)" at the end (Q1/Q2/N4/N3/N6 account, including the build-001 correction).

## Consequences (stated, not decided)
- Behaviour: the ack file is unchanged; the only runtime behaviour change is N4 (an aborted merge call now clears the streak table, so a streak can no longer span an abort). Not touched: the sender, the daemon, `cc-ng-sync.py` wording (#931), N1/N2/N5 (LOW, report-only; the env read was not moved).
- A `cc_current_membership` caller added later will get "every CC node held", never the ack: the narrowed set exists only under `cc_ack_membership`.
- The Q2 pin makes the identity-node exclusion explicit: a protected unbound node can never be bound by a Leg 2 re-offer. That is the ruling (H-1); it means such a node stays unbound if nothing else wires it.

## Not verified
- Nothing ran against any live graph/ack/conduit/VPS; all loops are synthetic. The N3 order-independence rests on three hash seeds (1, 2, 3) plus the default run, not on exhaustive orders.
- The delta look (Q1 refactor, N4) is the reviewers' to give; I did not self-review.
- The two skipped #897 tests (need `Z12_MERGE_BASE_REF`) were not run.
- I did not re-measure the ~9 ms whole-graph-pass cost; `cc_ack_membership` adds only the `cc_current_membership` pass that was already inside the 86cf5afd function.
- The docs punchlist rows for #918/#909 were not edited (out of the truth-doc-only scope); I saw only the first part of those rows (via a `git grep` hit), not the full text, so I make no claim about their accuracy.
