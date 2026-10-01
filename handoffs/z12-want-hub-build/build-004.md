<!--
# ---- Changelog ----
# [2026-09-30] Claude Sonnet 5.5 (Z12 builder, lane want-hub-engine-d-build-20260930, dispatch #12361) — build-004
#   What: return for PG-1-with-the-fold (four real-graph default-path comparisons) + the _step_lock HOLD measurement. NEW file, docs only.
#   Why: Chief-003 / Exec P448 scheduled the PRE-MERGE real-graph golden for the (d) engine fold; plan-005 §4A.5 "PG-1" is the spec of record.
#   How: every hash/number is from the harness records (copied by code into the artifact), git rev-parse / git hash-object / sha256sum. No secret involved.
# -------------------
-->

# build-004 — PG-1 with the fold: BASE vs FOLD identical on BOTH real copies (builder CLAIM: PASS ×2); the fold's `_step_lock` hold measured (~12–15 s per call)

**Not self-accepted.** A separate fresh law-enforcer ACCEPTOR verifies `pg1/pg1-artifact.md` next, then the delta pair reviews the procedure. Nothing merged, nothing armed, nothing replaced or settled by me (Exec P454). Syl's own graph was never loaded; the daemon stayed down (unit `inactive`, zero daemon processes at every reading and again at the end). No `save()`, no write to any checkpoint or live path, nothing under `~/NeuroGraph/data/checkpoints` opened, live tract never opened, `~/.bashrc` not touched, no unit/daemon started or stopped, no primary checkout and neither engine worktree edited. No secret involved.

## 0. Headline

| | |
|---|---|
| Artifact commit (tests branch `cc-laptop-want-hub-build-20260930`) | **`2475dcf05b589eab84b5d8e9a2c283002d43723c`** — `handoffs/z12-want-hub-build/pg1/` (artifact, harness, generator, compare JSON, 18 record files) |
| This return | `handoffs/z12-want-hub-build/build-004.md`, committed after the artifact (its own commit hash is in `git log`; it cannot cite itself) |
| Engine under test | FOLD `29f47f65058790240b2f9c6a0a5bc4d82171b42d` (blob `5e8945ac…`) vs BASE `e4ebf982b1989fd9066d610b94853bc68bf70d37` (blob `53494b7c…`); Part 2 also FIRST `8e5785322f910aefaf781fa3525bea2831fedd31` (blob `96d12f50…`) |
| **Copy (a)** laptop CEREMONY-BACKUP copy | BASE = FOLD on all 14 compared fields; **removed 0 / 0 (empty set — comparison of counters, state and bytes, not of a removal)**; full state digest `4626743e2ccbcf7070c3dc3126a51a96c0eedd4bbb41a9f4d401661166eec9a0`; checkpoint sha256 `902ad15a532ae0ec214dd50516c487b5dd9e0eee1691554967619cd7f90470ce`. **Builder claim: PASS.** |
| **Copy (b)** staged VPS bundle copy | BASE = FOLD on all 14 compared fields; **removed 10,433 / 10,433** (the plan's expected count), identical removal order (hash `e0df6abbc1c4…`, NOT id-sorted), identical `pruned` event `[[10433, 80259]]`; full state digest `9a96ef480a2ec592d3d2f648590ad93381129d6092551428298401c818674fa9`; checkpoint sha256 `6e0a36a565bbec1e57280478b2af9a5a42ea781dc28c3e4a4a12337b122b2a9d`. **Builder claim: PASS.** |
| Part 2 hold (laptop copy, K=50, B=5000) | FIRST: 12.99 / 12.26 / 12.27 s per call; FOLD: 14.82 / 14.19 / 12.06 s. ~10 s of each call is the orchestrator before `_prune_synapses`. Isolated validation CPU: FIRST 1.288 s → FOLD 1.961 s (**+0.673 s**). Wall fold-minus-first is the same size as the call-to-call noise (call 3 negative). |
| Memory (heap, org rule) | copy (a) whole-procedure heap 1.073 / 1.075 GiB (restore-only 0.995, = 007c's 0.992); copy (b) 0.589 GiB; Part 2 0.995 GiB. Cache-inclusive `memory.peak` 0.81–1.77 GiB. **No OOM; no load near the 6 GiB cap.** |
| Gate | every load met every gate by a wide margin (MemAvailable 9.14–9.33 GiB at all six loads; load 2.84–3.72; daemon inactive, 0 processes) |

## 1. What I read and set up (Step Zero for this dispatch)
- Read IN FULL: the brief `pg1-want-hub-engine-fold.md` (docs worktree `cc-laptop-daemon-recall-756-20260930`, not edited); plan-005 §4A.5 (PG-1 and the post-merge dry-run items, read at the NG worktree `cc-laptop-want-hub-d-20260930`, read-only) with §4.2 and §4.3; `tests/want_hub_golden_driver.py` in full; the fold's and the first commit's `_prune_synapses` and `compete_protected_links`. `git diff 8e578532..29f47f65 -- neuro_foundation.py` is `_prune_synapses` + a header only — **the orchestrator is byte-identical**, so Part 2's fold-minus-first isolates the fold's `_prune_synapses` changes.
- Tests branch: `git pull --rebase` ("Already up to date", branch had no upstream set; pulled by name from `origin`); the branch head before my commit was `2d860fb5f061ed2ac166e9a2072e9db4c4b17328`.
- NEW read-only detached worktrees (blob re-verified with `git hash-object` against `git rev-parse <commit>:neuro_foundation.py`, before and after all runs, status empty incl. ignored): `z12-pg1-engine-29f47f65` and `z12-pg1-engine-first-8e578532`; BASE `z12-want-hub-base-e4ebf982` pre-existing, verified.
- Scratch dir `/home/josh/backups/z12-pg1-20261001T021808Z/` (`records/`, `scratch/` — the four scratch `.msgpack` checkpoints + the ruled-copy `bundle-copy/`). The `T` stamp is UTC (the host's local clock reads 18:18 the previous day).

## 2. The procedure, as executed (one command per load; no retry of anything)
Order: harness smoke tests on a tiny SYNTHETIC checkpoint (no real graph, not part of the artifact, kept in scratch) → hash copy (a) whole directory → **(a)×BASE (the FIRST run = the heap measurement)** → (a)×FOLD → hash (a) again → stage copy (b) with sha256 before/after → (b)×BASE → (b)×FOLD → Part 2 FIRST → Part 2 FOLD → hash (a), (b), the staged VPS source again → independent `sha256sum` of the four scratch checkpoints → worktree integrity check. Six real-graph loads, each its own process and `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0`, one at a time. Each gate was a separate prior command (`pg1_harness.py gate`, exit non-zero on failure) chained with `&&`.

## 3. PG-1 results (details and every printed path/rev/sha256 are in `pg1/pg1-artifact.md` §3–§4 and `pg1/records/`)
- Every run printed its OWN checkout's `neuro_foundation.__file__`, git rev, blob, `sys.path[0]`, `ng_tract` file/version (`/home/josh/.local/lib/python3.12/site-packages/ng_tract/__init__.py`, 0.1.0 — an installed wheel shared by both checkouts) and `Graph.synapses` type (`builtins.SynapseStore`). None VOID. BASE reports `new_api_present` False, FOLD True — the two sides really run different code.
- Copy (a): 7,253 nodes / 138,753 synapses / 517 hyperedges / timestep 33,637. Source `main.msgpack` sha256 `7e4577868631de4389e5a6ef994f8b97d447690d0d1621e27f944189042c3a77` = the ruled value, equal before/after EVERY load (in-process) and across the whole directory (incl. the 1 GB `vectors.msgpack`) before Part 1, after Part 1 and after Part 2.
- Copy (b): staged copy of `vps-pull-staged` (`main.msgpack` 141,784,662 B); source stable across a 3 s re-stat and its newest mtime ~27 h old; copy = source sha256 for all four files; unchanged after both loads and the staged source is still equal to the copy.
- Compared per copy (BASE vs FOLD): return, removed count, removed-id sha256 (in removal order and sorted), `pruned` events, synapses before/after, pre-prune state digest, state digest before checkpoint, **full state digest in Test G's order**, serialized checkpoint sha256 and size, counts after restore and at the end. The audit hook shows exactly one write-mode open per process (the scratch `.msgpack`) and no mutating calls.
- The random-uuid-shaped-id class (M07a): real ids exercise it — on copy (b) the removal order differs from id-sorted order and is identical across BASE and FOLD. I did not verify the uuid4 shape of the real ids.

## 4. Part 2 results (table: `pg1-artifact.md` §5)
- Competing set built by the orchestrator: `competing_ids` 106,825 (= the plan's K=50 figure), `excluded_ids` 21,534, `order_key` 106,825; `F_links` 4,127, `protected_nodes` 183 (both = the plan's figures); call-1 `eligible` 102,145 → 21 cycles at B = 5,000.
- Wall per call (hold of `_step_lock`): FIRST 12.9853 / 12.2614 / 12.2691 s; FOLD 14.8211 / 14.1947 / 12.0647 s. CPU: FIRST 12.942 / 12.265 / 11.807; FOLD 13.654 / 13.315 / 11.713. Fold-minus-first wall +1.836 / +1.933 / −0.204 s; CPU +0.712 / +1.049 / −0.094 s. One process per variant at host load ~3–4: **the wall difference is the same size as the observed spread, so I do not read a steady fold cost from it**; the steady piece is the isolated validation CPU, +0.673 s (1.288 → 1.961 s, ~107k ids).
- Before `_prune_synapses` is reached the orchestrator spends ~10.2 s (FIRST) / ~11.3 s (FOLD, CPU 10.5 s) — identical code; that, not the prune, is most of the hold.
- Meaning for `step()` / Door B / StreamParser / Commons: all wait on `graph._step_lock` for the whole call (~12–15 s on this copy, uncontended); the pass is idle-gated (≥ 1,800 s) so a turn arriving mid-pass waits that long. **No judgement on acceptability — the Executive decides (C8).**
- Calls 2–3 run on the already-pruned scratch graph (5,000 removed per call; `low_weight_steps` advanced); nothing persisted. `compete_protected_links` removed 5,000 on every call.

## 5. Gates and the re-derivation (what I did and the one interpretation I made)
Heap per the org rule (`ru_maxrss` / sampled `RssAnon`, NOT the cache-inclusive `memory.peak`); re-derived gate = heap + 2.0 GiB: (a)×BASE 3.0734, (a)×FOLD 3.0754 (both above the ruled 3.0 → **the gate rose**), (b) 2.589 (below ruled → ruled 3.0 stands), Part 2 2.995 (below ruled 6.0 → 6.0 stands). I applied 3.074 to (a)×FOLD and 3.08 to both (b) loads. **Interpretation (flagged):** I read "if a gate is off STOP" as "if MemAvailable is below the re-derived gate" and "it rises" as the gate rising for later loads; I did NOT read a rise as a stop (the first load already ran at the ruled 3.0, with 9.154 GiB available). Had the brief meant otherwise, the only effect is that I continued; every reading was ≥ 9.1 GiB, ≥ 3× the highest derived gate.

## 6. Observations (not rulings) — for the plan owner / Chief
1. **Plan figure vs engine count (flag):** plan-005 §4A.3 states 16 held-back partners at K=50 (competing 106,841 → 106,825). Both engine variants report `held_back_last_link` = **15** while `competing_ids` = 106,825 (equal to the plan). A difference of one in the pre-hold set and one in the held-back count, unexplained here. PG-1 (default path) is unaffected; it matters for the dry run's "last-link held-back count". Not added to the punchlist by me (no punchlist access in this lane) — **to be routed**.
2. Re-serialized copy-(a) checkpoint = 230,539,966 B, the same size as the source but a different sha256 (expected for a re-save; noted because the size match is exact).
3. The `restore()` I read opens only the path given; the sibling-file opens in the audit are my harness hashing them (`src_sidecars`), not the engine.

## 7. What I did NOT verify
Real arming behaviour; the daemon slice and its dream-loop wiring; **#825** (PG-1 never calls `save()` and does not restore the checkpoint it writes); concurrency (nothing contended `_step_lock`, the hold is uncontended); the post-merge dry run (§4A.5 items 1–12); run-to-run variance of Part 2 (one process per variant, as briefed — a repeat process is the way to separate the fold's wall cost from the noise); the uuid4 SHAPE of real ids; that copy (b) holds nothing of Syl's (it is the staged VPS-bundle copy the plan names; I did not inspect its contents beyond counts and hashes).

## 8. Deviations (every one)
- Checkpoint-file sha256 computed in 8 MiB chunks, not with the driver's whole-file `sha256_file` (same value; `sha256sum` cross-check done) — to avoid a 230 MB heap spike distorting the heap measurement.
- Added extras the brief did not list: the pre-prune state digest, the pre-checkpoint post-state digest, a sampled-anon heap sampler, a Python audit hook, per-stage `ru_maxrss`. None changes what is compared.
- A harness smoke test on tiny SYNTHETIC checkpoints (built with the driver's `build_graph`) before the real loads, including two synthetic Part 2 runs; kept in scratch, not committed.
- The gate re-derivation interpretation in §5.
- The tests-branch had no upstream configured, so `git pull --rebase origin cc-laptop-want-hub-build-20260930` was used.

## 9. State left behind
- Tests branch `cc-laptop-want-hub-build-20260930` carries the artifact commit `2475dcf05b589eab84b5d8e9a2c283002d43723c` and this return; pushed by name, then I STOP. No PR, no merge.
- Scratch under `/home/josh/backups/z12-pg1-20261001T021808Z/` (four scratch checkpoints: a-base/a-fold 230,539,966 B each, b-base/b-fold 137,365,123 B each; the 0.52 GB bundle copy; smoke outputs; 1.2 GB in all) — left in place for the acceptor to re-hash; not cleaned up. Two new detached worktrees left in place for the acceptor (`z12-pg1-engine-29f47f65`, `z12-pg1-engine-first-8e578532`).
- Daemon: `inactive`, 0 processes, at the end. No unit, timer or `~/.bashrc` change.

## 10. Next (not mine)
The fresh law-enforcer ACCEPTOR verifies `pg1-artifact.md` against `pg1/records/` and records PASS or FAIL (plan §4A.5: in `handoffs/z12-want-hub-d/returns/pg1/` + `reviews/pg1-accept.md` per the plan's designation, or the paths the Chief names); then the delta pair reviews the procedure; the Executive rules on the hold (C8) and routes observation 6.1.
