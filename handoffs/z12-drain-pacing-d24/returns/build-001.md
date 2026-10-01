<!--
# ---- Changelog ----
# [2026-09-30] Claude Sonnet 5.5 (Z12 builder, dispatch #12669) — return build-001 for D24 (re-scoped, Exec P471/P472/P473): READ + STOP (e)
# What: STEP 1 READ findings and a short design note. NO code was built in either repo.
# Why: STOP condition (e) holds: the daemon's autosave loop calls drain_ingest_tract inside ONE held
#   _concurrent_lock section that cannot be split without a lock-semantics change (Exec P473 sends exactly this case back to the Executive).
# How: read-only against NG e4ebf982 and docs daemon e6052415; two small Python demos of RLock behaviour; no daemon, no graph, no checkpoint.
# -------------------
-->

# build-001 — D24 drain pacing: READ, and STOP (e)

Lane `drain-pacing-d24` (Z12), dispatch #12669. Builder: Claude Sonnet 5.5. Related: [[NeuroGraph]], [[The River]], [[Dual-Pass Embedding]], [[FatherGraph]].

**Outcome: STOP (e). Nothing was built.** No code commit in either repo. The docs daemon worktree is untouched (tip still `e60524154f9cfd6a1be34d28d97afc174b61accb`). The owner of the ruling is the Executive (Exec P473: this is "the ONLY lock case that goes back to the Executive").

## STEP 1 — READ

### 1. Does `drain_ingest_tract` honour `CC_NG_BATCH_SIZE` / `CC_NG_IDLE_STEPS` on the path the daemon uses?
**No.** Signature at `cc_ng_organism.py:2299`: `drain_ingest_tract(graph, vector_db, state, tract_path=None, return_consumed=False, max_entries=0)`. Its only bound is `max_entries` (`:2400-2401`, `if max_entries and taken >= max_entries: break`), which counts turns *taken*, not nodes. No env read, no step, no idle logic in the function.

Across the NG worktree the only runtime read of either name is `cc_topology_merge.py:286` (`CC_NG_IDLE_STEPS`, default 250). `CC_NG_BATCH_SIZE` is read only by `scripts/cc-ng-sync.py:220` (docs repo; `:221` for IDLE_STEPS), default 25/250. The daemon `scripts/cc-ng-daemon.py` reads neither (only comments at `:200`, `:1387`, `:1388`). Both names are exported in `~/.bashrc` (`:207-208`; names only).

**Correction to the brief's Facts:** `drain_gateway_conduit` does NOT pace by `batch_size`/`idle_steps` at this base. Its docstring says "Legacy batch/idle arguments remain inert" (`:2594`), it only logs a warning if they are passed (`:2622-2623`), and the header marks the old policy "superseded 2026-09-11: Leg1 is raw experience; no sleep" (`:467`). So `_cc_callosum_consolidate` has exactly ONE live caller: `merge_cc_topology` (`cc_topology_merge.py:609`). The merge docstring's pointer "cc_ng_organism.py:1856" (`cc_topology_merge.py:277`) is stale; the function is at `:2535`.

### 2. How are nodes-per-turn counted without splitting the dual pass?
`run_conversational_dual_pass` (`:2137`) creates, per turn, through `_cc_deposit_memory_node` (`:1837`; `graph.create_node` at `:1845`, only if absent):
- 1 forest node `cc:conv::<sha1(text)>`, plus tree-concept nodes deposited by `ng_embed.dual_record_outcome` through `_CCConversationalDualPassEco.record_outcome` (`:1945-1959`; trees below `_cc_concept_passes_floor` are skipped, `:1951`), plus window nodes `<forest>::window::<i>` (`:2170-2179`, short turns yield none).
- Then `_cc_bind_conversational_topology` (`:1965`) adds synapses and ONE binding hyperedge over forest+trees+windows (`:1991`) and the delayed prev->current forest link (`:2010`).

Node ids are content-hashed, so an exact-repeat turn creates 0 new nodes. A per-turn count is knowable with no change to the dual pass: `n_before = len(graph.nodes)` before the atomic call, `n_after` after it. The drain's caller holds `graph._concurrent_lock` (`:2344`), no `graph.step()` runs inside the drain, and nothing in the dual pass deletes nodes, so the delta is exact and non-negative. For the unbound-arrivals guard the arrival IDs are a set difference (`set(graph.nodes)` before the batch vs after), taken once per batch, not per turn. A turn that overshoots 25 is absorbed whole and the batch ends; a single turn alone over 25 is one batch. Feasible with no split of the dual pass.

### 3. The merge's step-and-guard code, and how the drain would call it
- Guard predicate: `cc_topology_merge._unbound_nodes(graph, node_ids)` at `:631-654` (no outgoing, no incoming synapse, no hyperedge membership; mirrors the orphan sweep).
- The guard+skip+consolidate composition is INLINE in `merge_cc_topology` at `:563-611` (`merge_landed |= batch_landed`; `unbound = _unbound_nodes(...)`; if unbound: `consolidation_skipped_unbound_arrivals += len(unbound)` + `logger.error` `:598-608`; `elif _cc_callosum_consolidate(graph, idle_steps)` `:609`). It is not a standalone function. Reuse therefore means calling the two real functions (`_unbound_nodes`, `_cc_callosum_consolidate`), not copying; extracting a shared helper would edit the Leg 2 merge, which I treated as out of scope. `merge_landed` is MERGE-scoped (`:341-347`, `:563`, comment `:584-596`); a drain would keep a per-CALL set the same way.
- `_cc_callosum_consolidate` (`cc_ng_organism.py:2535-2570`): `lock = graph._concurrent_lock`; loops `with lock:` over slices of `CC_CALLOSUM_LOCK_SLICE_STEPS` (default 25) steps (`:2553-2566`). It acquires `_concurrent_lock` itself, per slice. Import is acyclic: `cc_topology_merge` imports `cc_ng_organism` only lazily inside `merge_cc_topology` (`:281`), so the drain could lazily import `_unbound_nodes`.

**THE LOCK, in full (the reason for the STOP):**
- Type: `threading.RLock` (daemon `cc-ng-daemon.py:831`; also `cc_ng_host.py:1859`, `neurograph_rpc.py:2359`).
- Deposit-side acquisitions: each node deposit takes `graph._step_lock` via `_cc_mutation_lock` (`:1824-1834`, `:1842`), nested inside the caller's `_concurrent_lock` (the established `_concurrent_lock -> _step_lock` order; merge comment `cc_topology_merge.py:360-362` forbids the reverse).
- Hook ops take `_concurrent_lock` BLOCKING (`cc-ng-daemon.py:1836`; header `:1810`), so any long hold blocks every hook (the P459 class).
- **The call site:** `_autosave_loop`, `cc-ng-daemon.py:2077-2131`. ONE non-blocking acquire at `:2084` (`acquired = ...acquire(blocking=False)`; skip the whole cycle if busy, `:2085-2087`), released only in the `finally` at `:2128-2129`. Everything runs inside that one hold: `_guarded_save("autosave")` (`:2089`), **`drain_ingest_tract(..., return_consumed=True)` (`:2111-2113`)**, `trickle_gateway_conduit` (`:2114`), `cc_update_probation` (`:2115`), `surface_wants` (`:2116`), `generate_emergent_want` (`:2117`), `persist_cc_commons` (`:2125`).
- Consequence, demonstrated (Python demo, this session): with the caller holding the RLock once, the callee's `with lock:` slice re-enters and the outer hold persists (`_is_owned()` stayed True; a second thread could NOT acquire between slices). So calling `_cc_callosum_consolidate` from inside this section does NOT slice anything: it is one 250-step hold blocking every hook. And a drain that tries to "release first" releases the caller's single acquisition; the caller's `finally` release at `:2129` then raises `RuntimeError: cannot release un-acquired lock` (demonstrated), caught only by the outer `except` at `:2130` as "Autosave failed", with the rest of the section (probation, wants, commons) running unlocked.
- Contrast: the Leg 2 merge works precisely because it is NOT called under `_concurrent_lock` (docstring `cc-ng-daemon.py:1395-1398`: "deliberately NOT from the autosave pulse"; it takes only `_step_lock` per batch and releases before consolidating, `cc_topology_merge.py:360-363`).

**Second call site:** `cc_ng_host.py:1525-1526` (`with _STATE.cc_ng.graph._concurrent_lock: drain_ingest_tract(...)`; drain-only scope, VPS host). Same shape: the drain is inside the `with`, so it cannot release either. With default-`None` parameters and no pacing passed there it would be byte-identical, so (d) for THAT site is provable; it is not paced.

### 4. Where can between-batch bookkeeping live?
Nowhere persistent is needed. A drain call takes whole turns up to one batch, then (design) runs the steps; the remainder is already left in the tract file by the existing partial-truncate (`max_entries` mechanism, `:2434-2447`, reusable as-is, LAW 3). The next call needs to know nothing: it re-reads the file. The arrival set / guard accumulation is per-call, exactly like the merge's `merge_landed`. **Residual (a limit, not a blocker):** like the merge, the guard is not carried ACROSS calls. A node an earlier call left unbound (steps skipped) is not in the next call's set, so a later pass could step past its grace. Dual-pass binding is atomic within a turn, so this should not arise from the drain's own arrivals.

## STOP condition that holds: (e), with exact location
**File:line:** `scripts/cc-ng-daemon.py:2084` (acquire) ... `:2111` (the drain call) ... `:2129` (release), `STATE.ng.graph._concurrent_lock`, an RLock (`:831`).

**Why it cannot be split without a lock-semantics change:** the drain sits in the middle of a single acquisition shared with the save, the Leg 1 trickle, probation, wants and the Commons persist. For the drain to deposit under the lock, RELEASE, then consolidate (Exec P473), the lock must be released between those two phases, and either:
1. the drain releases a lock it does not own (breaks the caller's `finally`, demonstrated), or
2. the drain re-acquires inside the caller's hold (no release; consolidation runs at full 250-step length with the lock held; the exact P459-class degradation P473 forbids; relying on RLock re-entrancy is the forbidden assumption), or
3. the daemon's held section is restructured so the drain is called with the lock NOT held and takes it for the deposit span itself. That changes the documented drain contract (`:2344`, "the caller holds `_concurrent_lock`"), the trylock/skip-the-cycle semantics of `:2084-2087` (is the drain's own acquire blocking, or a trylock?), and where trickle / probation / wants / persist sit relative to the lock. Those are lock-semantics decisions, not mine.

I did not build around it.

## Short design note (for the Executive's ruling; nothing implemented)
- **Recommended shape (two-phase, no change to the drain's lock contract):** keep `drain_ingest_tract` caller-locked and deposit-only, adding the optional default-`None` pacing parameters (stop at >= `CC_NG_BATCH_SIZE` nodes on a whole-turn boundary; partial truncate reused) and returning a small receipt (batch-ended-on-size flag + arrival id set + a hardcoded-reason code). The daemon's autosave section is split so the drain's `with` scope ends, the lock is released, and the caller then runs `_unbound_nodes` guard -> `_cc_callosum_consolidate(graph, idle_steps)` (the merge's own two functions, which slices the lock for real because nothing outer holds it). This departs from "the drain ITSELF runs the steps" only in who makes the call; the steps still run through the merge's code.
- **Alternative:** the drain owns the lock in paced mode (contract change `:2344`). Larger blast radius (second call site, tests that assume caller-holds).
- **Rulings needed:** (a) which of the two shapes; (b) the held section split: does the paced drain get its own trylock (skip-if-busy as today) or a blocking acquire; (c) do trickle (`:2114`, needs the drain's consumed bytes), probation, wants and Commons persist stay under a second held section; (d) Leg 2 interaction: the drain's guard sees only the drain's own arrivals, but 250 steps per paced cycle is a NEW step source and could march a node a MERGE left unbound (merge skipped consolidation) past `orphan_node_grace_period` (25). Today only ~1 step per turn ages such a node. I cannot prove this unaffected without a ruling on guard scope (drain-landed only vs a whole-graph unbound check, whose O(N) cost and permanently-blocking-on-pre-existing-orphans behaviour are not known to me); I flag it as a second hold-point under (d).
- **Whichever shape:** the daemon env-read (LAW 5, no new variable, unset/invalid/<=0 = unpaced + ONE WARNING naming both variables) and the tests specified in the brief are unaffected and ready to write once the shape is ruled.

## What I did NOT verify
- No test or build was run (nothing built). The two Python demos above are the only executions, in `/tmp`, no graph, no checkpoint, no daemon.
- The `dual_record_outcome` tree-creation count was read through `_CCConversationalDualPassEco` and `_cc_deposit_memory_node` but not executed; I did not run the dual pass on a synthetic graph to check the node-count delta empirically.
- The `RuntimeError` / re-entrancy behaviour is shown on a bare `threading.RLock`, not on the live graph's lock object (same type per `:831`).
- `cc_ng_host.py` (VPS host) was read at the drain call site only; its other lock acquisitions were not audited.
- Whether `orphan_node_grace_period` is 25 for the CC graph was taken from the merge's own comment (`cc_topology_merge.py:577-581`), not from config.

## Delivery
- NG branch `cc-laptop-drain-pacing-d24-20261001` (base `e4ebf982b1989fd9066d610b94853bc68bf70d37`, NO upstream; never pulled/rebased): this file is the only change (docs-only return commit).
- Docs daemon branch `cc-laptop-daemon-d24-env-20261001`: no commit.
- NG merges BEFORE the daemon (a daemon passing a new kwarg to an older organism raises `TypeError`). Merge held; nothing wired; no restart, no `.bashrc`/crontab edit, no PR.
