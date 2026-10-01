# PG-1 artifact — the PRE-MERGE real-graph golden for the (d) engine FOLD, plus the `_step_lock` HOLD (dispatch #12361)

Lane `want-hub-engine-d-build-20260930` · builder: a FRESH thread independent of the engine author · tests branch `cc-laptop-want-hub-build-20260930` · spec of record plan-005 §4A.5 "PG-1" + brief `pg1-want-hub-engine-fold.md`.
**Everything below is a MEASUREMENT by the builder. The per-copy VERDICT lines are CLAIMS for the separate acceptor to verify; this builder does not accept its own artifact.** ids / hashes / counts / booleans only — no want text, no node metadata.

## 0. What was run
- Checkouts (read-only, detached; `git hash-object` of each `neuro_foundation.py` re-checked after all runs, `git status --porcelain --ignored` empty): BASE `/home/josh/NeuroGraph-worktrees/z12-want-hub-base-e4ebf982` (`e4ebf982b1989fd9066d610b94853bc68bf70d37`, blob `53494b7c56896d25040f3e7fd7c4046da7d0ab05`); FOLD `/home/josh/NeuroGraph-worktrees/z12-pg1-engine-29f47f65` (`29f47f65058790240b2f9c6a0a5bc4d82171b42d`, blob `5e8945accb476b0727bf07a3ab2890dccc87f650`); FIRST engine commit `/home/josh/NeuroGraph-worktrees/z12-pg1-engine-first-8e578532` (`8e5785322f910aefaf781fa3525bea2831fedd31`, blob `96d12f507746ce168fffe6feba56e097c8832353`).
- Harness: `pg1_harness.py` (git blob `8bda680d3830a03756ac15f8b8210fd5792f2d48`) in this directory; it imports `tests/want_hub_golden_driver.py` BY FILE PATH and reuses its `state_digest`, `instrument`, `ng_tract_info`, `git_rev`, `new_api_present` — so the digest compared is the SAME one Test G makes (`state_digest(g, events)` computed AFTER `Graph.checkpoint()`, as the driver does; the pre-checkpoint digest is recorded as an extra). Only deviation: the checkpoint FILE's sha256 is computed in 8 MiB chunks (a sha256 of the same bytes is the same value; verified independently with `sha256sum`, §2).
- Every load: its OWN fresh process inside `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0` (the process read back `memory.max` = 6442450944 and `memory.swap.max` = 0 from its own cgroup), under `env -u NG_EMBED_REMOTE -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTHONHASHSEED=0`; `sys.path[0]` pinned to its own checkout before import; a printed path outside its checkout would be VOID (none was). Canonical `Graph().restore(<copy>/main.msgpack)`; `_prune_synapses()` at ALL new parameters at defaults; `Graph.checkpoint()` to a scratch temp `.msgpack` under `/home/josh/backups/z12-pg1-*/scratch/` (never `save()`, never a live path). A Python audit hook recorded every write-mode `open`/mutating call in each process.
- Copy (a) = the CEREMONY BACKUP `/home/josh/backups/z12-s3-restore-bundle-20260929T222529Z/pre-placement-laptop-cc/` used READ-ONLY. Copy (b) = `vps-pull-staged/` copied to the NEW scratch dir (`/home/josh/backups/z12-pg1-20261001T021808Z`), sha256 before/after equal.
- The random-uuid-shaped-id class (M07a, le-036/checker-029 C1) is exactly what REAL graph ids exercise: on copy (b) the removal ORDER is not id-sorted order (recorded boolean in §3) and is identical base vs fold, so an unconditional sort-by-id on the default path would have shown. I did not separately verify the uuid4 SHAPE of the real ids.

## 1. Gate readings immediately before each load (ruled: Part 1 MemAvailable ≥ 3.0 GiB, Part 2 ≥ 6 GiB; load < 6; `cc-ng-daemon.service` inactive and no daemon process)
| load | UTC | MemAvailable GiB | load 1/5/15 | daemon unit | daemon procs | gate applied (GiB) | gate met |
|---|---|---|---|---|---|---|---|
| (a) BASE | 2026-10-01T02:20:41Z | 9.154 | 3.60/3.02/3.70 | inactive | 0 | 3.0 | True |
| (a) FOLD | 2026-10-01T02:21:41Z | 9.139 | 2.84/2.95/3.63 | inactive | 0 | 3.074 | True |
| (b) BASE | 2026-10-01T02:23:15Z | 9.203 | 3.02/2.96/3.57 | inactive | 0 | 3.08 | True |
| (b) FOLD | 2026-10-01T02:23:44Z | 9.218 | 2.96/2.95/3.55 | inactive | 0 | 3.08 | True |
| Part 2 FIRST 8e578532 | 2026-10-01T02:24:15Z | 9.265 | 3.72/3.13/3.58 | inactive | 0 | 6.0 | True |
| Part 2 FOLD 29f47f65 | 2026-10-01T02:25:26Z | 9.329 | 3.66/3.29/3.61 | inactive | 0 | 6.0 | True |

Gate at load 1 (a-base) was the ruled 3.0; its first-run heap re-derived the gate to 3.0734 → **applied 3.074 (a-fold) and 3.08 (b loads)** per "if that exceeds the ruled gate it rises". Interpretation stated: I read "if a gate is off STOP" as "if MemAvailable is below the (re-derived) gate"; every reading was ≥ 9.1 GiB, so no stop condition arose.

## 2. Memory — the org rule is HEAP (`ru_maxrss` / sampled anon), NOT the cache-inclusive cgroup `memory.peak`; both reported
| load | restore-only ru_maxrss GiB | whole-procedure ru_maxrss GiB | sampled RssAnon peak GiB | cgroup memory.peak GiB (cache-incl.) | heap used for gate (max of ru_maxrss, anon) | re-derived gate = heap + 2.0 | ruled gate | effective gate | oom_kill |
|---|---|---|---|---|---|---|---|---|---|
| a-base | 0.9951 | 1.0734 | 1.0564 | 1.7724 | 1.0734 | 3.0734 | 3.0 | 3.0734 | oom_kill 0 |
| a-fold | 0.9950 | 1.0754 | 1.0574 | 1.3712 | 1.0754 | 3.0754 | 3.0 | 3.0754 | oom_kill 0 |
| b-base | 0.5095 | 0.5889 | 0.5718 | 0.9305 | 0.5889 | 2.5889 | 3.0 | 3.0000 | oom_kill 0 |
| b-fold | 0.5094 | 0.5890 | 0.5719 | 0.8095 | 0.5890 | 2.5890 | 3.0 | 3.0000 | oom_kill 0 |
| Part 2 first 8e578532 | 0.9950 | 0.9950 | 0.9770 | 0.9809 | 0.9950 | 2.9950 | 6.0 | 6.0000 | oom_kill 0 |
| Part 2 fold 29f47f65 | 0.9951 | 0.9951 | 0.9770 | 0.9808 | 0.9951 | 2.9951 | 6.0 | 6.0000 | oom_kill 0 |

The restore-only column reproduces build-tool-007c's 0.992 GiB (0.995 here); the whole-procedure heap on copy (a) is ~1.075 GiB because PG-1 also builds two state digests and serializes the checkpoint. No load approached the 6 GiB cap. Independent cross-check: `sha256sum` of the four scratch checkpoints equals the in-process values (a-base = a-fold = `902ad15a532ae0ec214dd50516c487b5dd9e0eee1691554967619cd7f90470ce`; b-base = b-fold = `6e0a36a565bbec1e57280478b2af9a5a42ea781dc28c3e4a4a12337b122b2a9d`).

## 3. Part 1 — one record per copy × checkout

### (a) laptop CEREMONY-BACKUP copy × BASE
- printed `neuro_foundation.__file__`: `/home/josh/NeuroGraph-worktrees/z12-want-hub-base-e4ebf982/neuro_foundation.py`; git rev: `e4ebf982b1989fd9066d610b94853bc68bf70d37`; blob: `53494b7c56896d25040f3e7fd7c4046da7d0ab05`; `sys.path[0]`: `/home/josh/NeuroGraph-worktrees/z12-want-hub-base-e4ebf982`; VOID: False
- `ng_tract` file: `/home/josh/.local/lib/python3.12/site-packages/ng_tract/__init__.py`, version `0.1.0` (an installed wheel, not a file of either checkout); `Graph.synapses` type: `builtins.SynapseStore`; `_prune_synapses` new keyword-only API present: False
- PYTHONHASHSEED=`0`; env: NG_EMBED_REMOTE=None PYTHONPATH=None PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1; python 3.12.3; pid 3371677; cgroup `/user.slice/user-1000.slice/user@1000.service/app.slice/run-u77827.scope`
- source `main.msgpack`: sha256 before `7e4577868631de4389e5a6ef994f8b97d447690d0d1621e27f944189042c3a77`, after `7e4577868631de4389e5a6ef994f8b97d447690d0d1621e27f944189042c3a77` (equal: True); all sibling files equal before/after: True; load start `2026-10-01T02:20:41Z` MemAvailable 9.155 GiB, load 3.60
- after restore: nodes 7253, synapses 138753, hyperedges 517, timestep 33637; after prune/checkpoint: synapses 138753
- `_prune_synapses()` return: **0**; removed ids: 0; removed-id sha256 in removal order `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; sorted `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (equal: True); `pruned` events `[]`; default-path wall 0.86 s
- pre-prune state digest `ba6a6fcf1187a07e6879690fbfbaf3d8ee6b86a9948e04994b1a80f6747faed3`; post-state digest before checkpoint `4626743e2ccbcf7070c3dc3126a51a96c0eedd4bbb41a9f4d401661166eec9a0`; **full state digest (Test G order) `4626743e2ccbcf7070c3dc3126a51a96c0eedd4bbb41a9f4d401661166eec9a0`**
- serialized `Graph.checkpoint()` → scratch temp `.msgpack`: 230539966 bytes, sha256 `902ad15a532ae0ec214dd50516c487b5dd9e0eee1691554967619cd7f90470ce`
- audit: write-mode opens `['/home/josh/backups/z12-pg1-20261001T021808Z/scratch/a-base.msgpack']`; mutating os/shutil calls `[]`; programs spawned `['git', 'pgrep', 'systemctl']`

### (a) laptop CEREMONY-BACKUP copy × FOLD
- printed `neuro_foundation.__file__`: `/home/josh/NeuroGraph-worktrees/z12-pg1-engine-29f47f65/neuro_foundation.py`; git rev: `29f47f65058790240b2f9c6a0a5bc4d82171b42d`; blob: `5e8945accb476b0727bf07a3ab2890dccc87f650`; `sys.path[0]`: `/home/josh/NeuroGraph-worktrees/z12-pg1-engine-29f47f65`; VOID: False
- `ng_tract` file: `/home/josh/.local/lib/python3.12/site-packages/ng_tract/__init__.py`, version `0.1.0` (an installed wheel, not a file of either checkout); `Graph.synapses` type: `builtins.SynapseStore`; `_prune_synapses` new keyword-only API present: True
- PYTHONHASHSEED=`0`; env: NG_EMBED_REMOTE=None PYTHONPATH=None PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1; python 3.12.3; pid 3380737; cgroup `/user.slice/user-1000.slice/user@1000.service/app.slice/run-u77849.scope`
- source `main.msgpack`: sha256 before `7e4577868631de4389e5a6ef994f8b97d447690d0d1621e27f944189042c3a77`, after `7e4577868631de4389e5a6ef994f8b97d447690d0d1621e27f944189042c3a77` (equal: True); all sibling files equal before/after: True; load start `2026-10-01T02:21:41Z` MemAvailable 9.137 GiB, load 2.84
- after restore: nodes 7253, synapses 138753, hyperedges 517, timestep 33637; after prune/checkpoint: synapses 138753
- `_prune_synapses()` return: **0**; removed ids: 0; removed-id sha256 in removal order `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; sorted `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (equal: True); `pruned` events `[]`; default-path wall 0.95 s
- pre-prune state digest `ba6a6fcf1187a07e6879690fbfbaf3d8ee6b86a9948e04994b1a80f6747faed3`; post-state digest before checkpoint `4626743e2ccbcf7070c3dc3126a51a96c0eedd4bbb41a9f4d401661166eec9a0`; **full state digest (Test G order) `4626743e2ccbcf7070c3dc3126a51a96c0eedd4bbb41a9f4d401661166eec9a0`**
- serialized `Graph.checkpoint()` → scratch temp `.msgpack`: 230539966 bytes, sha256 `902ad15a532ae0ec214dd50516c487b5dd9e0eee1691554967619cd7f90470ce`
- audit: write-mode opens `['/home/josh/backups/z12-pg1-20261001T021808Z/scratch/a-fold.msgpack']`; mutating os/shutil calls `[]`; programs spawned `['git', 'pgrep', 'systemctl']`

### (b) staged VPS bundle copy × BASE
- printed `neuro_foundation.__file__`: `/home/josh/NeuroGraph-worktrees/z12-want-hub-base-e4ebf982/neuro_foundation.py`; git rev: `e4ebf982b1989fd9066d610b94853bc68bf70d37`; blob: `53494b7c56896d25040f3e7fd7c4046da7d0ab05`; `sys.path[0]`: `/home/josh/NeuroGraph-worktrees/z12-want-hub-base-e4ebf982`; VOID: False
- `ng_tract` file: `/home/josh/.local/lib/python3.12/site-packages/ng_tract/__init__.py`, version `0.1.0` (an installed wheel, not a file of either checkout); `Graph.synapses` type: `builtins.SynapseStore`; `_prune_synapses` new keyword-only API present: False
- PYTHONHASHSEED=`0`; env: NG_EMBED_REMOTE=None PYTHONPATH=None PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1; python 3.12.3; pid 3394445; cgroup `/user.slice/user-1000.slice/user@1000.service/app.slice/run-u77878.scope`
- source `main.msgpack`: sha256 before `8cf6ef22f0e75756fc0d0a7e706258ef390ca030c9f24c0bbc80f6203d3d90a1`, after `8cf6ef22f0e75756fc0d0a7e706258ef390ca030c9f24c0bbc80f6203d3d90a1` (equal: True); all sibling files equal before/after: True; load start `2026-10-01T02:23:15Z` MemAvailable 9.194 GiB, load 3.02
- after restore: nodes 19224, synapses 19390, hyperedges 2179, timestep 80259; after prune/checkpoint: synapses 8957
- `_prune_synapses()` return: **10433**; removed ids: 10433; removed-id sha256 in removal order `e0df6abbc1c4e7ea5fa051613d31476e08b93dbb40144d8af7c6cd25e7d40c49`; sorted `6078096850405870ba93de87e1e3f9eafb0f874802bd0ca29f31f7d7441c1645` (equal: False); `pruned` events `[[10433, 80259]]`; default-path wall 0.58 s
- pre-prune state digest `fe57b0dfdaa79125f3021b33a52da8004dfa3f405bf9b73b8cbea3bf6fd5c782`; post-state digest before checkpoint `9a96ef480a2ec592d3d2f648590ad93381129d6092551428298401c818674fa9`; **full state digest (Test G order) `9a96ef480a2ec592d3d2f648590ad93381129d6092551428298401c818674fa9`**
- serialized `Graph.checkpoint()` → scratch temp `.msgpack`: 137365123 bytes, sha256 `6e0a36a565bbec1e57280478b2af9a5a42ea781dc28c3e4a4a12337b122b2a9d`
- audit: write-mode opens `['/home/josh/backups/z12-pg1-20261001T021808Z/scratch/b-base.msgpack']`; mutating os/shutil calls `[]`; programs spawned `['git', 'pgrep', 'systemctl']`

### (b) staged VPS bundle copy × FOLD
- printed `neuro_foundation.__file__`: `/home/josh/NeuroGraph-worktrees/z12-pg1-engine-29f47f65/neuro_foundation.py`; git rev: `29f47f65058790240b2f9c6a0a5bc4d82171b42d`; blob: `5e8945accb476b0727bf07a3ab2890dccc87f650`; `sys.path[0]`: `/home/josh/NeuroGraph-worktrees/z12-pg1-engine-29f47f65`; VOID: False
- `ng_tract` file: `/home/josh/.local/lib/python3.12/site-packages/ng_tract/__init__.py`, version `0.1.0` (an installed wheel, not a file of either checkout); `Graph.synapses` type: `builtins.SynapseStore`; `_prune_synapses` new keyword-only API present: True
- PYTHONHASHSEED=`0`; env: NG_EMBED_REMOTE=None PYTHONPATH=None PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1; python 3.12.3; pid 3398839; cgroup `/user.slice/user-1000.slice/user@1000.service/app.slice/run-u77893.scope`
- source `main.msgpack`: sha256 before `8cf6ef22f0e75756fc0d0a7e706258ef390ca030c9f24c0bbc80f6203d3d90a1`, after `8cf6ef22f0e75756fc0d0a7e706258ef390ca030c9f24c0bbc80f6203d3d90a1` (equal: True); all sibling files equal before/after: True; load start `2026-10-01T02:23:44Z` MemAvailable 9.215 GiB, load 2.96
- after restore: nodes 19224, synapses 19390, hyperedges 2179, timestep 80259; after prune/checkpoint: synapses 8957
- `_prune_synapses()` return: **10433**; removed ids: 10433; removed-id sha256 in removal order `e0df6abbc1c4e7ea5fa051613d31476e08b93dbb40144d8af7c6cd25e7d40c49`; sorted `6078096850405870ba93de87e1e3f9eafb0f874802bd0ca29f31f7d7441c1645` (equal: False); `pruned` events `[[10433, 80259]]`; default-path wall 0.58 s
- pre-prune state digest `fe57b0dfdaa79125f3021b33a52da8004dfa3f405bf9b73b8cbea3bf6fd5c782`; post-state digest before checkpoint `9a96ef480a2ec592d3d2f648590ad93381129d6092551428298401c818674fa9`; **full state digest (Test G order) `9a96ef480a2ec592d3d2f648590ad93381129d6092551428298401c818674fa9`**
- serialized `Graph.checkpoint()` → scratch temp `.msgpack`: 137365123 bytes, sha256 `6e0a36a565bbec1e57280478b2af9a5a42ea781dc28c3e4a4a12337b122b2a9d`
- audit: write-mode opens `['/home/josh/backups/z12-pg1-20261001T021808Z/scratch/b-fold.msgpack']`; mutating os/shutil calls `[]`; programs spawned `['git', 'pgrep', 'systemctl']`

## 4. Part 1 comparison and per-copy VERDICT (a CLAIM — for the acceptor to verify against `records/` and `pg1-compare.json`)

**(a) laptop ceremony-backup copy** — field-by-field BASE vs FOLD equal: `return`=True, `removed_count`=True, `removed_ids_sha256_in_order`=True, `removed_ids_sha256_sorted`=True, `pruned_events`=True, `synapses_before`=True, `synapses_after`=True, `pre_state_digest`=True, `state_digest_before_checkpoint`=True, `state_digest`=True, `checkpoint_sha256`=True, `checkpoint_size`=True, `counts_after_restore`=True, `counts_final`=True
Preconditions: base_path_is_own_checkout=True; fold_path_is_own_checkout=True; different_code_under_test (blob differs, new_api False vs True)=True; same_ng_tract_wheel=True; same_pythonhashseed_pinned=True; source_main_sha256_equal_before_after (base, fold)=True; all_sidecars_equal_before_after (base, fold)=True; only_write_mode_open_was_the_scratch_checkpoint (base, fold)=True; no_oom (cgroup oom_kill 0, base and fold)=True; removal_set_non_empty (the removal comparison is NOT vacuous)=False; removal_order_differs_from_id_sorted_order (an unconditional sort-by-id would be visible)=False
Non-vacuity note: on this copy the default path removes **0** synapses (as the plan predicts: 0 eligible), so the removed-id and removal-order comparisons here are comparisons of two EMPTY sets; what this copy proves is that every counter the function advances (`low_weight_steps` on all 138753 evaluated non-protected synapses) and the full state and serialized bytes are identical. The load-bearing removal comparison is copy (b).
**VERDICT (claim, builder): PASS** — all 14 compared fields identical between BASE `e4ebf982` and FOLD `29f47f65` and every precondition above holds.

**(b) staged VPS bundle copy** — field-by-field BASE vs FOLD equal: `return`=True, `removed_count`=True, `removed_ids_sha256_in_order`=True, `removed_ids_sha256_sorted`=True, `pruned_events`=True, `synapses_before`=True, `synapses_after`=True, `pre_state_digest`=True, `state_digest_before_checkpoint`=True, `state_digest`=True, `checkpoint_sha256`=True, `checkpoint_size`=True, `counts_after_restore`=True, `counts_final`=True
Preconditions: base_path_is_own_checkout=True; fold_path_is_own_checkout=True; different_code_under_test (blob differs, new_api False vs True)=True; same_ng_tract_wheel=True; same_pythonhashseed_pinned=True; source_main_sha256_equal_before_after (base, fold)=True; all_sidecars_equal_before_after (base, fold)=True; only_write_mode_open_was_the_scratch_checkpoint (base, fold)=True; no_oom (cgroup oom_kill 0, base and fold)=True; removal_set_non_empty (the removal comparison is NOT vacuous)=True; removal_order_differs_from_id_sorted_order (an unconditional sort-by-id would be visible)=True
Non-vacuity note: this copy removes **10433** synapses on BOTH checkouts with identical removal order, identical `pruned` event and identical post-state and serialized bytes (the plan's expected count is 10,433).
**VERDICT (claim, builder): PASS** — all 14 compared fields identical between BASE `e4ebf982` and FOLD `29f47f65` and every precondition above holds.

## 5. Part 2 — the `_step_lock` HOLD on the laptop copy (K = 50, B = 5000; in-memory scratch call, nothing saved)
Each variant in its own fresh process/scope after ONE canonical restore of copy (a) (`main.msgpack` sha256 equal before/after in both). `compete_protected_links` holds `graph._step_lock` for its whole body, so its wall time IS the hold seen by `step()` / Door B / StreamParser / Commons. Calls 2–3 run on the already-pruned scratch graph (each call removed 5,000 and advanced `low_weight_steps` — nothing is persisted). `perf_counter` = wall, `process_time` = CPU. One process per variant, host load ~3–4 with other threads active: wall and CPU are both given because CPU is less sensitive to contention.
| variant | call | wall s | cpu s | inside `_prune_synapses` wall s | outside it (orchestrator) wall s | eligible | removed | conducting removed | held-back last-link | floors_ok | synapses after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FIRST 8e578532 | 1 | 12.9853 | 12.9422 | 2.6529 | 10.3324 | 102145 | 5000 | 0 | 15 | True | 133753 |
| FIRST 8e578532 | 2 | 12.2614 | 12.2654 | 2.5596 | 9.7018 | 97145 | 5000 | 220 | 15 | True | 128753 |
| FIRST 8e578532 | 3 | 12.2691 | 11.8073 | 2.4599 | 9.8092 | 92145 | 5000 | 484 | 15 | True | 123753 |
| FOLD 29f47f65 | 1 | 14.8211 | 13.654 | 4.5015 | 10.3196 | 102145 | 5000 | 0 | 15 | True | 133753 |
| FOLD 29f47f65 | 2 | 14.1947 | 13.3146 | 3.8209 | 10.3738 | 97145 | 5000 | 220 | 15 | True | 128753 |
| FOLD 29f47f65 | 3 | 12.0647 | 11.7133 | 2.8182 | 9.2465 | 92145 | 5000 | 484 | 15 | True | 123753 |

Fold-minus-first (same call index): call 1: wall +1.836 s, cpu +0.712 s, inside-prune wall +1.849 s; call 2: wall +1.933 s, cpu +1.049 s, inside-prune wall +1.261 s; call 3: wall -0.204 s, cpu -0.094 s, inside-prune wall +0.358 s.

**Orchestrator before the prune call** (two `_plan()` builds + floor check + capture map, measured by aborting at the `_prune_synapses` call WITHOUT mutating anything; identical code in both variants): FIRST wall 10.1941 s / cpu 10.1261 s; FOLD wall 11.2918 s / cpu 10.5091 s.

**The validation pass in isolation** (`_prune_synapses` called with the orchestrator's own captured `competing_ids` ∪ one absent id that sorts last, so it raises at the END of its pre-loop validation having validated every real id; state fingerprint unchanged — a refusal mutates nothing; harness-side only, no engine edit; ×3 each):
| variant | run | wall s | cpu s | raised at the absent id | state fingerprint unchanged |
|---|---|---|---|---|---|
| FIRST 8e578532 | 1 | 1.3174 | 1.2501 | True | True |
| FIRST 8e578532 | 2 | 1.8592 | 1.2965 | True | True |
| FIRST 8e578532 | 3 | 1.6865 | 1.3173 | True | True |
| FOLD 29f47f65 | 1 | 1.9235 | 1.9342 | True | True |
| FOLD 29f47f65 | 2 | 1.9848 | 1.9788 | True | True |
| FOLD 29f47f65 | 3 | 1.9764 | 1.9706 | True | True |

Mean validation CPU: FIRST 1.288 s, FOLD 1.961 s → fold-minus-first **+0.673 s** (the fold adds per-id tuple/kind checks to the pre-loop validation; the first commit checks only membership in `order_key`).  Counts (identical for both variants): `competing_ids` = 106825, `excluded_ids` = 21534, `order_key` entries = 106825, `max_removals` passed = 5000; call-1 `eligible` = 102145 (⇒ ceil(eligible/B) = 21 cycles); `F_links` = 4127; `protected_nodes` = 183; `held_back_last_link` = 15.

**What the hold means (no judgement on acceptability — the Executive decides, C8):** while `compete_protected_links` runs, every other holder of `graph._step_lock` waits — `step()`, Door B, StreamParser's `_nudge_nodes`/`_trigger_completions`, the Commons leg-2 read. On the laptop copy a call holds the lock for roughly 12–15 s (table above), of which about 10 s is the orchestrator's own set-building before `_prune_synapses` is reached. The pass is idle-gated (≥ 1,800 s idle), so a turn arriving mid-pass waits up to that long. The fold's marginal cost is the validation delta above plus sort/inside-prune differences; the fold-minus-first wall difference is of the SAME SIZE as the call-to-call spread within one process (call 3 of the FIRST variant was slower than call 3 of the FOLD), so only the CPU-time validation delta is steady.

## 6. Observations to flag (not rulings)
- **Plan figure vs engine count:** plan-005 §4A.3 says 16 partners are held back at K = 50 (competing 106,841 → 106,825). Both engine variants report `held_back_last_link` = 15 and `competing_ids` = 106825 (equal to the plan's 106,825) — i.e. one fewer held-back link and, implicitly, one fewer link in the pre-hold set than the plan's probe figures. Not explained here; PG-1 (default path) is unaffected. For the plan owner / dry run.
- The re-serialized checkpoint of copy (a) has the SAME byte size as the source `main.msgpack` (230,539,966) but a different sha256 (`902ad15a532ae0ec…` vs source `7e4577868631de43…`): expected for a re-save, noted only because the size coincidence is exact.

## 7. NOT verified here
Real arming behaviour; the daemon slice and its dream-loop wiring; #825 (pruned links staying pruned across save AND restore — PG-1 never calls `save()` and does not restore the checkpoint it writes); concurrency (no other thread contends for `_step_lock` in these runs, so the table is an uncontended hold); the post-merge dry run (items 1–12); a repeat of Part 2 for run-to-run variance (the brief asked for one process per variant); the uuid4 SHAPE of real ids; that Syl's own graph is never loaded (copy (b) is a staged VPS-bundle copy named by the plan; nothing from `~/NeuroGraph/data/checkpoints` was opened).


> **Z12 erratum (PG-1 delta pair, 2026-10-01):** the sentence on the fold-minus-first wall difference above is over-stated. Corrected wording: `acceptance-record-notes.md` note 3 (fold adds at least +0.67 s CPU in validation; the remaining wall, up to +1.9 s, is unexplained, not shown to be noise).
