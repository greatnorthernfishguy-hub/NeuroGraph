<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane ack-bound-918, dispatch #14104) — return build-003 for the #918 fold-up 2
# What: builder return for build-918-foldup2.md (F1 deterministic N3 test, D1 docstring, Exec P534 wording, cosmetic nits).
# Why: the brief (docs worktree daemon-recall-756-20260930, handoffs/z12-silent-failure-lanes-20260930/assignments/build-918-foldup2.md) requires this file.
# How: faithful report. THE SWEEP IS PARTIAL: the gate REFUSED seed 5 (load), so I stopped; the unrun items are listed, not papered over.
# -------------------
-->

# Return build-003 — #918 fold-up 2 (PARTIAL: the load gate refused the seed sweep at seed 5)

**Seat/authority:** worker, `worktree_write` (RESUMED my own thread). Did not self-accept; nothing pushed or merged; no rebase.

## Hashes (real `git rev-parse HEAD`)
| repo | branch | commit | what |
|---|---|---|---|
| NG `/home/josh/NeuroGraph-worktrees/z12-ack-bound-918-20261001` | `cc-laptop-ack-bound-918-20261001` | start tip `7c21bb7acb6637e9659c6cfef7d1ddf9f9a7f0ac` (my code `c3c76432`, return 002 on top) | start |
| NG | same | **`b1949155b8e280b82f4911a8880ed0f9a731afa1`** | the ONE NG commit (test + docstring/comment only) |
| NG | same | the commit carrying THIS file (docs-only; `git log -1` of the branch) | return |
| docs `/home/josh/docs/.claude/worktrees/callosum-truth-918-20261001` | `cc-laptop-callosum-truth-918-20261001` | start `d8dc4f9b2772d9643594577369cdd2d36b1072ba` | start |
| docs | same | **`3b2c49fbf334e17b821a1d791420b5126e2db541`** | the ONE docs commit (`CC-CALLOSUM-TRUTH.md` only; status `ahead 3, behind 154` vs `origin/main`, not rebased) |

`git diff --stat 7c21bb7 HEAD` (NG, before this return commit) lists ONLY: `cc_topology_merge.py` (+56/-, docstrings/comments), `tests/test_cc_ack_bound_918.py`, `tests/test_cc_topology_callosum.py` (comment text). No other file.

## What did NOT run (read this first)
The standing clause says refuse at 1-min >= 6 OR 5-min >= 5.0, never wait-then-proceed. The box's load climbed during the sweep (other work; mine is light): **seed 5 was REFUSED at START load1=6.18 / load5=5.19, so I STOPPED the sweep there and ran nothing after it** (I re-read `/proc/loadavg` twice afterwards, 6.15/5.23 then 7.03/5.51, only to record it; no run was started). NOT RUN, so owed (a re-dispatch when the box is quiet):
1. the N3 seed sweep for **seeds 5..31 (27 seeds)**;
2. the whole new file at seeds **14, 21, 28**;
3. mutants **G2** (budget ignored) and **G7** (ack includes the unbound set) against the FIXED N3 test;
4. `test_cc_merge_whole_graph_guard.py`, `test_cc_topology_capture_423.py`, `test_cc_topology_callosum.py` (its test-file comment edits were made but the file was not re-run).
A PARTIAL table is a valid terminal state per the brief; I am not claiming the sweep.

## READ first (my own test and docstring before any review text; I did not open `reviews/le-056-918.md`)
- N3 (build-002 `:813-840`): the loop asserted `set(st["frame_node_ids"]) == {C1,C2,X}` on EVERY tick. When X sits last in the frame, call 2 completes the binding but `completed` is still False (X counted deferred), so a 3rd tick happens: by then everything is acked, the sender returns the `exhausted` shape WITHOUT `frame_node_ids`, and `st["frame_node_ids"]` raises `KeyError`. Whether X is last follows the exporter's iteration over a hyperedge's member SET (`_closeable_he` builds `add` from the set), i.e. the hash seed. My 3-seed check (1/2/3) happened to avoid it.
- D1: `cc_current_membership`'s docstring opened "the authoritative source for the SENDER's exclude_ids (#110)" and ended with my pointer paragraph saying the opposite.

## What changed
1. **F1 (test only), deterministic by construction.** `tests/test_cc_ack_bound_918.py`: a helper `_reorder_conduit_nodes` rewrites the REAL conduit's batch so its node records follow an order the TEST chooses; the N3 test is parametrized over X **first / middle / last** and pins the exact per-call tuple `(call, absorbed_nodes, deferred_by_budget, completed, X bound)` for each (`_N3_EXPECTED`): first `[(1,1,1,F,F),(2,1,0,T,T)]`, middle `[(1,1,2,F,F),(2,1,0,T,T)]`, last `[(1,1,2,F,F),(2,1,1,F,T),(3,0,0,T,T)]`. The sender's `exhausted` / no-frame shape is handled EXPLICITLY: that branch asserts `st["exhausted"] is True`, that X is actually BOUND, and that the ack is exactly `{C1,C2,X}` (so the shape can only appear once everything is bound); the receiver then re-merges the conduit left on disk (as `cc-ng-sync.py` does). **It cannot pass vacuously:** the loop's end state is asserted separately (the exact log, X bound, the hyperedge `{C1,C2,X}` present, ack `== {C1,C2,X}`, and `exhausted_seen == (x_position == "last")`), so a missing `frame_node_ids` is never treated as success. No assertion or loop condition depends on which candidate the sender picks first or on the frame order (the order is the test's). Same intent and strength: ignoring the budget lands both new nodes on call 1 (the call-1 tuple and `log[0][4] is False` fail); an ack that includes the unbound set stops the re-offer and ends in the exhausted branch with X unbound (its first assertion fails) — **both reasoned from the code; the G2/G7 runs themselves did NOT happen (see above).**
2. **D1 + the P534 wording (`cc_topology_merge.py`, comments/docstrings ONLY).** `cc_current_membership`'s docstring is rewritten TRUE and not contradictory: every CC-provenance node currently held, bound or not, protected or not; explicitly NOT the ack; the ack / `exclude_ids` source is `cc_ack_membership`. The superseded H-1 clause is DROPPED where it was live: the c3c76432 changelog entry (its two lines), `cc_ack_membership`'s docstring (`:~299-303`), the end-of-call comment (`:~946`), the Q2 pin test's docstring; the new sentence (words exactly as given; line-wrapped in comments/docstrings, so a single-line `grep` for the whole sentence matches only in the docs): *"Protected nodes are excluded from re-offer because they are not sweep-eligible (they survive at any degree). Identity-touching binding structure still crosses per #147 (identity crosses the callosum)."* A new changelog entry (dispatch #14104) states the change; it describes the dropped clause as "a blanket no-write-to-identity-nodes rule" so no live sentence asserts it.
3. **The Q2 pin test's assertions: all kept, none weakened.** The closing "no edge on the protected nodes" assertions hold only because of the SCENARIO (the one frame node has a synapse from its own acked anchor, none from the frame node to a protected node); the docstring now says so ("a fact of this scenario, NOT an invariant: such a synapse would cross by design (#147)"). **Reporting as the brief asked:** that pair of assertions would stop holding in a scenario with a sender synapse to an acked protected node, by design.
4. **Cosmetic nits.** The new test file's garbled header substitution ("every `tmg.cc_ack_membership(` became `tmg.cc_ack_membership(`") now reads `tmg.cc_current_membership(` became; the two callosum comments that said "current CC membership" over `cc_ack_membership` now say it is the ACK (the CC nodes held BOUND); the callosum header now names `test_journaled_but_culled_node_is_readmitted` (and the other two names in full).

### Proof: only comments/docstrings changed in `cc_topology_merge.py`
`ast.dump` of the module with docstrings stripped (Module / FunctionDef / ClassDef leading string expressions removed), `7c21bb7` (= the `c3c76432` code) vs `b1949155`: **IDENTICAL** (both dumps 61294 chars); the raw sources differ (docstrings/comments). Re-run against the committed blobs after the last edit.

### grep: no live "never writes to identity" left
- NG: `grep -rn "never writes to identity\|never write to identity" cc_topology_merge.py tests/` -> **0** (the remaining "H-1" mentions, 2 in the merge module and 1 in the test header, are all inside the dispatch-#14104 changelog entries, which say the clause was superseded / dropped).
- docs: `grep -c "never writes" CC-CALLOSUM-TRUTH.md` -> **0**; the P534 sentence appears in the banner, §3 *Update (#918)* and the §9 entries (4 matches). The two docs mentions of "H-1" that remain describe the supersession (the §9 Q2 paragraph and the new fold-up 2 entry).

## Runs (what DID run). `env -u CC_NG_BATCH_SIZE -u CC_NG_IDLE_STEPS -u CC_NG_DRAIN_HOLD_ON_FAILURE -u NG_EMBED_REMOTE`, EFFECTIVE values printed each time (all four `<unset>`), scratch HOME, `PYTHONUSERBASE=/home/josh/.local`, `HF_HUB_OFFLINE=1`, `-p no:cacheprovider`; each run preceded by a 25 s pause and gated on its own START load (refuse at 1-min >= 6 or 5-min >= 5.0); the effective `PYTHONHASHSEED` is printed in each run's `EFFECTIVE` line and listed here
| run | seed | START load1 / load5 | result |
|---|---|---|---|
| **FAIL-FIRST**: the OLD test file (scratch copy of `tests/test_cc_ack_bound_918.py` at `7c21bb7`, only its `_WORKTREE` line repointed to this worktree; the filename contains "n3" so `-k n3` selected all 40) | 7 | 4.04 / 4.16 | **1 failed, 39 passed**: the old N3 test dies with `KeyError: 'frame_node_ids'` at `:837`, as le-056/Z12 found; the other 39 old tests pass at seed 7 |
| whole NEW file | 7 | 3.96 / 4.15 | **42 passed** |
| whole NEW file | 0 | 4.44 / 4.26 | **42 passed** |
| sweep, new file `-k n3` (3 cases) | 0 | 5.20 / 4.54 | 3 passed |
| sweep | 1 | 5.06 / 4.58 | 3 passed |
| sweep | 2 | 5.64 / 4.74 | 3 passed |
| sweep | 3 | 5.59 / 4.82 | 3 passed |
| sweep | 4 | 5.72 / 4.93 | 3 passed |
| sweep | 5 | **6.18 / 5.19** | **REFUSED (gate). Sweep stopped. Seeds 5..31 NOT run.** |
Not run, forbidden as ever: the NG full suite and `tests/test_cc_deposit_step.py`. The commit `b1949155` was made after the fail-first and the two whole-file runs passed and BEFORE the sweep (a commit is not a run). Seed independence of the N3 test rests on its construction (the order is rewritten by the test) plus seeds 0, 1, 2, 3, 4 and 7 (the whole-file ones included); it is NOT demonstrated by a 32-seed sweep. I did not look for other seed-dependent tests beyond the whole file passing at seeds 0 and 7.

## Consequences (stated, not decided)
- Production behaviour is unchanged (AST identity). The only thing that moves is what the docstrings and the truth doc SAY about protected nodes: excluded from re-offer because not sweep-eligible; identity-touching binding structure still crosses per #147.
- The N3 test now PINS exact `deferred_by_budget` counts per ordering, including that a present X after the spent budget is counted deferred; if a later change to the merge's budget accounting is intended, this test will (correctly) need its table updated.

## Not verified
- Everything in "What did NOT run". In particular no 32-seed sweep, no G2/G7 run, and the callosum/423/#897 files were not re-run after the comment edits (the merge module is AST-identical, so I expect no change, but I did not run them).
- Nothing ran against any live graph/ack/conduit/VPS; all loops are synthetic. I did not read le-056, so I cannot say whether it lists other findings beyond those in the brief.
