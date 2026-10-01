<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane ack-bound-918, dispatch #13352) — return build-001 for #918
# What: builder return for "the ack means I hold it BOUND" (Exec Packet 496 via Chief-003, RULED ADOPT, S4b-GATING).
# Why: the assignment (docs worktree daemon-recall-756-20260930, handoffs/z12-silent-failure-lanes-20260930/assignments/build-918-ack-bound.md, docs 6326a9de) requires this file.
# How: faithful report; every hash is a real `git rev-parse HEAD`; nothing pushed, nothing merged.
# -------------------
-->

# Return build-001 — #918: the ack means "I hold it BOUND"

**Seat/authority:** worker, `worktree_write`. BUILD ONLY. Did not self-accept; nothing merged or pushed.

## Hashes (real `git rev-parse HEAD`)
| repo | branch | commit | what |
|---|---|---|---|
| NG `/home/josh/NeuroGraph-worktrees/z12-ack-bound-918-20261001` | `cc-laptop-ack-bound-918-20261001` | base `ee94f7d2c516cce40ff81bd32e4febb9781e8e04` (the #905 tip; **not** the #897 tip `ac5800fd`, so no STOP) | start HEAD |
| NG | same | **`86cf5afd91806b4ee79a87313edfe11c0b41fc72`** | the ONE NG code commit (merge + 5 test files) |
| NG | same | the commit that carries THIS file (docs-only; its hash is `git log -1` of this branch, which cannot be written inside the file it commits) | return |
| docs `/home/josh/docs/.claude/worktrees/callosum-truth-918-20261001` | `cc-laptop-callosum-truth-918-20261001` | base `fbbb53e78ed9543209a67b4b623c797825b612f5` (origin/main at creation) | start HEAD |
| docs | same | **`008a10affdc2f104dcd0cc4f5f163e33643f41ac`** | the ONE docs commit (`CC-CALLOSUM-TRUTH.md` only) |

Both branches have NO upstream; `git fetch -q` only; no `pull --rebase`; no push. The docs worktree reports `ahead 1, behind 4` vs `origin/main` (the 4 are other lanes' handoff commits); `git diff HEAD origin/main -- CC-CALLOSUM-TRUTH.md` is empty, so none of them touched the truth doc.

## READ first (base `ee94f7d2`, what I cited)
- `cc_topology_merge.py` base `:203-223` `cc_current_membership` = `graph.nodes` filtered ONLY by `is_cc_provenance`; `:727` `_write_membership(membership_path, cc_current_membership(graph))` at the end of EVERY call. (The brief's `~:188-208` / `~:684` were the older base; same code.)
- Only callers of `cc_current_membership` besides tests: none. `merge_cc_topology` callers: tests, and (read-only) `cc-ng-daemon.py` `handle_merge_topology` (`:1357-1367`, passes `membership_path`, reads stats) driven by `cc-ng-sync.py leg2-tick` (`:542-649`: `_load_membership` -> sender `exclude_ids`; `_git_commit_membership` commits the ack file).
- Sender `cc_topology_export.py:811-814` candidates = connected ∧ eligible ∧ `nid not in exclude_ids`; `:826` `present = set(exclude_ids)`; `:831-896` the (a) synapse-anchor / (b) whole-HE closure / (d) co-add rules; `:970` `syn_member_set = frame_set | exclude_ids`. Confirms the brief's mechanism: an acked node is never a candidate and is re-added only through a whole-HE closure with an un-acked connected member.
- Merge Tier 1 `:451-454` (present -> `skipped_present`, `batch_landed.add`, a usable endpoint), `:455-477` the `nid in held` branch, `:553-559` `membership_stale_readmitted`, Tier 2 `:565-588` (both endpoints in `graph.nodes`, `_synapse_exists`), Tier 3 `:591-639` (`members.issubset(graph.nodes)`, `_hyperedge_exists` member-set equality).
- `_unbound_nodes` (`:739` on base) and `redact_node_id` (`:796`) are #905's; `neuro_foundation._is_identity_protected` (`:3551-3572`) is pure.
- Analysis #909 `analysis-909-sender-candidate-predicate.md/.py` (docs `c5957ea7`): I copied its synthetic VPS/laptop loop pattern.

## What changed (all in `cc_topology_merge.py`; line numbers are post-change)
1. **`cc_current_membership` `:234-279`**: CC-provenance nodes `cc` minus `_unbound_nodes(graph, cc)` (`:279`). Same name, signature, `Set[str]`. Pure query (no counter/log/state/write). Docstring and the merge's end-of-call comment (`:872-881`) and merge docstring now state the ack = "I hold it BOUND (not sweep-eligible)" and why (the deadlock). **Cost: one whole-graph pass per merge call** (`_unbound_nodes` over the CC set; the #897 guard measured ~9 ms at 16k nodes; not re-measured here). `_write_membership` / `_load_membership` code unchanged.
2. **Starvation counter**: `_track_reoffers` `:282-` (a separate function; the query stays pure), called from `merge_cc_topology` after all batches and before the ack write (`:864`); `reoffered_present` collected at Tier 1's `skipped_present` branch (`:562`, `:591`). Constants `:222-225`.
3. **Stats** (initialised with the others, `:543-545`): `reoffered_unbound`, `reoffer_streak_warnings`, `reoffer_streak_nodes_at_or_over`.
4. Changelog header entry (top of file).

### The merge-lands-edges finding (item 2)
**The merge ALREADY does it; no merge fix was needed.** Read first (Tier 1 `:451-454` keeps a present node as an endpoint, Tier 2 `:565`, Tier 3 `:591`), then tested with the REAL exporter -> REAL merge: a held-UNBOUND node + a frame carrying a NEW hyperedge containing it is BOUND after ONE merge call; same for a node bound by a frame SYNAPSE only; the ack of THAT call includes it and the previous call's did not; when every node is already present (`skipped_present` 3) the synapse and hyperedge still land (`tests/test_cc_ack_bound_918.py::test_one_merge_binds_...hyperedge...`, `..._synapse_only`, `::test_the_skipped_present_path_does_not_skip_the_nodes_edges...`). Condition: receiver `max_nodes_per_call` generous (`BIG=200` in the tests). Observation, NOT changed (not a defect of this slice, not asked): inside Tier 1 the per-node `budget <= 0` check (`:582`, post-change; base `:445`) runs BEFORE `nid in graph.nodes` (`:588`; base `:451`), so once new nodes have spent the whole budget in a batch, later PRESENT nodes of that batch are deferred (counted in `deferred_by_budget`) instead of `skipped_present`, and their hyperedge waits for the next call (the same conduit is re-merged idempotently, so it converges). Present nodes themselves never spend budget.

### The counter (item 3)
- **Constant:** `_REOFFER_WARN_STREAK_DEFAULT = 5` (N). **Env variable (the ONLY new one, LAW 5): `CC_TOPOLOGY_REOFFER_WARN_STREAK`** (read per call; non-integer/empty -> default 5; `<=0` -> clamped to 1).
- **Streak semantics:** per merge call, the ids the frames carried that the receiver already held (`skipped_present`) are tested with the shared `_unbound_nodes` AFTER all batches; still sweep-eligible-unbound => streak+1; every other entry is dropped (reset on bind, on a call that did not re-send it, on protected). A call that returns early (`no_conduit` / `empty`) also resets the table. **ONE WARNING per node when its streak == N** (not repeated at N+1...; allowed again only after a reset). Record text has no ids/words: the redacted `<kind>:<12 hex>` from the ONE helper `redact_node_id`, the streak length, N, the env name, and "how many nodes now at/over N".
- **Table:** in memory only (module dict, process lifetime), key = the redacted form, **hard cap `_REOFFER_TABLE_CAP = 1024`**; when more nodes qualify than the cap the LONGEST streaks are kept and the rest start over; the dict is rebuilt and swapped as a whole each call (copy-on-write, no lock). Nothing persisted, no flag in the ack, ack contents unchanged.
- The laptop-only unbound forest (unconnected on the VPS) is never a candidate, so it cannot starve anything; not coded for.

## Tests
New `tests/test_cc_ack_bound_918.py` (27 tests; REAL `Graph`, REAL `SimpleVectorDB`, REAL `export_cc_topology_frame` -> REAL `merge_cc_topology`; `Z12_MERGE_UNDER_TEST` swaps the merge module so the SAME file runs on the base and on mutants; P379/#770 printed preamble names every module and all resolve inside the worktree).

### Runs (scratch HOME `/tmp/z12_918_scratch/home`, `PYTHONUSERBASE=/home/josh/.local`, `HF_HUB_OFFLINE=1`, `NG_EMBED_REMOTE` unset, `-p no:cacheprovider`, `PYTHONDONTWRITEBYTECODE=1`; load1 before every run 2.4-3.9, always < 6; no model/tokenizer seeding was needed)
| file | result |
|---|---|
| `tests/test_cc_ack_bound_918.py` (final) | **27 passed** (an earlier dev run of a prior version of this file, one test since rewritten, also 27 passed) |
| `tests/test_cc_emergent_want_bound_905.py` | **60 passed** |
| `tests/test_cc_merge_whole_graph_guard.py` (#897) | first run **2 failed, 15 passed, 2 skipped** (spy scope, see below); after the update **17 passed, 2 skipped** (the 2 skips need `Z12_MERGE_BASE_REF`, unset: not run) |
| `tests/test_cc_topology_callosum.py` | first run **3 failed, 31 passed** (2 = ack-meaning, 1 = poincare); after the update **1 failed, 33 passed**: the remaining failure is `test_poincare_dir_is_rederived_locally_not_transmitted` (#902, `poincare_dir` stored as bytes), which **also fails on the untouched base** (run on an exported `ee94f7d2` tree in scratch: 1 failed) |
| `tests/test_cc_topology_frame.py` | **15 passed** |
| `tests/test_cc_topology_capture_423.py` | **4 passed** (run once, after the harness update) |
| NEW tests against the **BASE** `cc_topology_merge.py` (`ee94f7d2:cc_topology_merge.py`, byte-compared) | **20 failed, 6 passed, 1 skipped**. The 6 that pass on base: pure query, byte-identical on no-unbound, signature, skipped_present-edges (the merge already landed them), the in-file CONTROL (it patches the old membership in, so it passes on both), counter-never-touches-the-ack (vacuous without a counter). **The end-to-end loop test and the ack-after-every-tick test FAIL on the base**; the CONTROL shows why: with the old ack the cohort, the pair and the (iii) nodes stay unbound and are never offered |
Run counts honestly: the #897 and callosum files were each run twice (before/after the expectation updates); my new file once final + the dev run; the base and mutant runs are extra runs of my file as the brief requires. NOT run: the NG full suite (#754), `tests/test_cc_deposit_step.py`, any TID-class test. One read-only background `grep -rln` over `/home/josh` (looking for other readers of the ack) ran ~90 s and finished; load stayed < 6 (4.85 seen).

### Mutations (scratch copies in `/tmp/z12_918_scratch/mut/`, each run via `Z12_MERGE_UNDER_TEST`; every one KILLED)
| mutant | killed by (count) |
|---|---|
| M1 old membership (`return cc`) | 9 failed: membership, spy, double, ack-file, hyperedge-binding, synapse-binding, stale-readmitted, loop, ack-after-every-tick |
| M2 copied predicate inside `cc_current_membership` | 2 failed: the "USES the imported predicate" spy test, the double-without-`_is_identity_protected` test |
| M3 age term added to `_unbound_nodes` | 1 failed: membership (the age-0 unbound node must be OUT) |
| M4 streak never resets (`streaks = dict(prior)`) | 2 failed: reset-on-bind, reset-on-gap |
| M5 WARNING every call (`>= n`) | 1 failed: exactly-once-at-N |
| M6 raw id in the record/table (`key = nid`) | 3 failed: redaction/privacy test, one-helper test, protected-node test |
| M7 unbounded table (cap removed) | 1 failed: cap test |
| M8a membership logs a line | 1 failed: pure-query test |
| M8b membership touches the counter state | 8 failed (pure-query + the counter tests that see the clobbered table) |

## Existing tests whose expectation changed (file, test, why; no assertion weakened to hide a failure)
1. `tests/test_cc_topology_callosum.py::test_membership_snapshot_feeds_sender_exclude_ids` -- **encoded "the ack includes unbound nodes"**: its sender fixture holds an isolated forest turn `f2` (no synapse, no hyperedge) that lands UNBOUND; it asserted `held == all four ids` and `exported_nodes == 0`. Now: `held == the three bound ids`, the sender re-offers exactly `{f2}` (new helper `_resent_ids`); the intent (current membership = exclude_ids, bound nodes are not re-sent) is still asserted.
2. `::test_culled_node_drops_out_of_exclude_ids_and_is_resent` -- same cause: pre-cull `exported_nodes == 0` -> `== 1` (`{f2}`); after the cull the re-send `{t1}` -> `{t1, f2}`; "the culled node drops out and is re-sent" is still asserted exactly. (This file had no changelog header; one added.)
3. `tests/test_cc_topology_capture_423.py::test_budget_and_unbound_arrival_guard_unchanged` -- `calls['membership'] == [set(graph.nodes)]` encoded the old ack; now parameter `expected_ack`: budget 1 -> `set()` (a landed, its edges skipped because b was deferred), unbound arrivals -> `set()`, bound -> `{'a','b'}` (the first test's `[{'a','b'}]` is untouched and passes). Plus harness plumbing only: `_track_reoffers` joins the AST-extracted names and the new constants/state join the namespace (same kind of change #905 made for `whole_graph_guard`).
4. `tests/test_cc_merge_whole_graph_guard.py::test_d_idle_steps_zero_never_evaluates_the_guard_and_never_blocks` and `::test_d_the_guard_is_asked_about_every_node_in_the_graph_under_the_step_lock` -- NOT about the ack's meaning: they spy on `tmg._unbound_nodes` and counted every call; the ack and the counter are now two more callers (after the batches, also at `idle_steps=0`). The spies now count only the GUARD's callers (`merge_cc_topology`, `_guard`), the guard assertions are unchanged (0 asks at `idle_steps=0`; exactly 2 whole-graph asks under `_step_lock`, steps unlocked), and the second test additionally pins the two new callers (`_track_reoffers`, `cc_current_membership`, both under `_step_lock`).

## Consequences (stated, not decided)
- **Ack file / `exclude_ids` size:** the file loses every unprotected sweep-eligible-unbound CC node the laptop holds (the 147 cohort among them) until it binds, then grows back; the sender's `exclude_ids` shrinks by the same set. Format unchanged (sorted ids, one per line, atomic swap).
- **Readers of the membership as "everything held":** none in code. `scripts/cc-ng-sync.py` (read-only here) reads it only through `_load_membership` -> `exclude_ids` and commits the file (`_git_commit_membership`); `cc-ng-daemon.py` passes `membership_path` and reads stats (the three new keys are additive). **Punch-list candidates, NOT touched:** stale wording in `cc-ng-sync.py` headers (`:22-34` "ack covers every connected CC node", the `_git_commit_membership` docstring "what the laptop now holds", `:232`) and a `cc-ng-daemon.py` comment (`:1308-1311`); they predate #918 and say "held" where it now means "held bound".
- **`membership_stale_readmitted` (`:692-697` post-change; base `:553-559`) meaning narrows:** `held` is the last ack, which no longer lists unbound nodes, so the stat now counts a node acked BOUND and later culled; an unbound node culled before it was ever acked is re-absorbed as a plain new node (`absorbed_nodes`, not the stale stat). Test: `test_membership_stale_readmitted_counts_a_culled_BOUND_node_not_a_culled_unbound_one` (fails on the base: it reports 1 where the new code reports 0). Nothing else in that branch changes.
- **Still holding the clock:** a laptop-only unbound node (unconnected on the VPS) is never a candidate and never in the ack, so it keeps holding the #897/#905 clock exactly as before (#905's docstring: it is ruled on by class, #909). #918 un-deadlocks only the VPS-connected ones.
- **Re-offer every tick until bound:** the sender now re-offers a held-unbound candidate with its structure every tick until it binds; a node that cannot bind is what the receiver-side WARNING reports. A conduit file persists and is re-merged idempotently (`cc-ng-sync.py`), so the receiver cannot tell an identical re-merged file from a fresh re-send; both count as a re-offer.
- LAW check by me: no vendored/protected file touched; ONE definition of the predicate (called, spy-tested); the query is pure (LAW 4, mutation-tested); ONE new env variable (LAW 5); the table keys/records carry only `redact_node_id` forms (LAW 7 privacy); no live graph/tract/checkpoint/ack file, no VPS contact, no sender change, no restart, no `.bashrc`/crontab/unit.

## Truth doc (docs commit `008a10af`, `CC-CALLOSUM-TRUTH.md` only; +61/-11, lines are post-change)
`:2` top banner (the changelog line + wikilinks, newest first); `:516` §3 #110 update (narrowed); `:521` NEW §3 *Update (#918)* paragraph; `:607` §4 #110 row (narrowed); `:608` NEW §4 #918 row; `:635` §4 caveat; `:1623` the §8/§10 cull-resend loop note (converse case); `:2561` §10 Phase 3; `:2643` §10.2 #110 row; `:2703` §10.4 table "resume / ack"; `:2712` §10.4-A "this write IS the ack"; `:2744` §10.4-C membership-as-ack bullet; `:2888` §10.4 Leg A candidate order (`∉ exclude_ids`); `:2906` ack wiring (`handle_merge_topology`); `:3339` NEW §9 2026-10-01 entry. READ and deliberately LEFT as dated records (the file's own convention, §8.5.3): the §9 2026-08-28 lines "Membership-as-ack (#110) is populated" and "already-acked members are idempotently re-sent"; the banner states they now mean "held BOUND".

## Not verified
- Nothing ran against the live laptop graph, the live ack file, the live conduit or the VPS; every loop result is on SYNTHETIC graphs with fake ids/embeddings.
- NOT estimated (the Executive's forward simulation): how many Leg 2 ticks the 146 VPS-connected cohort nodes need; whether the 147th (unconnected on the VPS) is as the Executive found.
- The ~9 ms whole-graph-pass cost is the #897 guard's measurement, not re-measured on this change.
- The real `cc-ng-daemon.py handle_merge_topology` body and the full `leg2-tick` flow were read only at the lines cited, not executed; the new stats keys are additive.
- The in-memory table lives per process (the laptop daemon): it resets on a restart, by design (no persisted per-node state); behaviour with two concurrent merge calls was reasoned (copy-on-write swap), not tested.
- The one-shot `export_cc_topology` (not the paced frame sender; production uses only the frame sender per `cc-ng-sync.py`) now re-ships unbound isolated nodes when given the new ack: only the two callosum tests above exercise that.
- `tests/test_cc_merge_whole_graph_guard.py`'s 2 skipped tests (need `Z12_MERGE_BASE_REF`) were not run.
- Pending, by design: the ONE FULL PAIR on this build, and the S4b gate/guards are unchanged.
