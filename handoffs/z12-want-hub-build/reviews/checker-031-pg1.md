# checker-031 ROLE A — PG-1 procedure/results with the (d) engine fold

STATUS: INCOMPLETE - review in progress

- Seat: checker-031 (fresh cross-family, grok-4.6)
- Lane: want-hub-engine-d-build-20260930
- Dispatch: #12464
- Zone manager: Z12 (session 52d39aba-db92-4bf2-b3b1-0e4c13f77d8c)
- Authority: report_only
- Tests worktree: `/home/josh/NeuroGraph-worktrees/z12-want-hub-build-20260930`
- Tests branch: `cc-laptop-want-hub-build-20260930`
- Artifact commit (starting material): `2475dcf05b589eab84b5d8e9a2c283002d43723c`
- Plan pin: `24bcd335a9c1e265f4c9d562c5b9c68a7bd92711` §4A.5 PG-1 (`handoffs/z12-want-hub-d/returns/plan-005.md`)
- Packet: `/home/josh/docs/.claude/worktrees/daemon-recall-756-20260930/handoffs/z12-silent-failure-lanes-20260930/assignments/review-packet-want-hub-engine.md` ADDENDUM 2 (binding isolation) + ADDENDUM 3
- This commit: first findings. Forbidden reads (`build-004.md`, `pg1/acceptance-le-039.md`, other `reviews/` files, tests-branch `git log` / `git status -sb`) have not been opened.
- Isolation: searches confined to named paths; verdict created by exact path; `git fetch -q` / `git pull --rebase -q`; `git show <hash>:<path>` for the artifact.
- Verdict so far: **PASS-WITH-NOTES** (check 6 deferred until after this commit)

## P379 (this review process; no NG import, no graph load)

- python_executable: `/usr/bin/python3`
- sys.path[0] empty; `/home/josh/NeuroGraph` present via parent `PYTHONPATH=/home/josh/NeuroGraph:`
- NG-related sys.modules: none (neuro_foundation not imported)
- Parent `NG_EMBED_REMOTE` was set; this review did not spawn a targeted NG run. Builder records unset both `NG_EMBED_REMOTE` and `PYTHONPATH` inside each load.

Dispatch gate (packet): 03:23:39 UTC load 3.48, MemAvailable 8761132 kB; 03:24:39 UTC load 3.56, MemAvailable 8214592 kB; other worker turns []. No real-graph load in this review.

## Isolation / exposure

- Did not open `build-004.md`, `pg1/acceptance-le-039.md`, any file under `handoffs/z12-want-hub-build/reviews/` except this verdict, or tests-branch `git log` / `git status -sb`.
- Plan-005 heading scan for `4A.5` / `PG-1` printed other lines in the same file that mention PG-1 (changelog bullets and a truncated Test G paragraph). That is an over-read of plan-005 beyond §4A.5. Disclosed. No reviews/ bodies, no other-leg filenames.
- Tests HEAD at quiet pull: `c2c6b7552ca6500c24883da1e87bca9e8c6bde89` (hash only).
- Named-file reads after the stub: artifact commit blobs under `handoffs/z12-want-hub-build/pg1/` (including `records/`), plan-005 §4A.5, and `tests/want_hub_golden_driver.py` (the harness's hardcoded DRIVER path; blob at the artifact commit `b5d439f6e68d46e3d7598de1d1176c0cc53a30c0`, equal to the worktree file).

## ADDENDUM 3 checks

### (1) Procedure vs the spec

**PASS-WITH-NOTES**

Spec (plan-005 §4A.5 PG-1 at `24bcd335`): read-only copies; never `data/checkpoints/`; nothing written back; ONE load at a time; each load `MemAvailable ≥ ~8 GB`; copy handling with sha256 before/after; `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0`; `env -u NG_EMBED_REMOTE`; canonical `Graph().restore(copy)`; never `save()`; `checkpoint()` only to a scratch temp `.msgpack`; four loads (copy a/b × base/branch) each in its own fresh process; `PYTHONHASHSEED` pinned; `sys.path[0]` pinned; printed `neuro_foundation.__file__` + git rev (void if wrong file); `_prune_synapses()` at all new parameters at defaults; compare removed-id hash, return value, full post-state hash (Test G field list, incl. `items()` order), serialized-bytes (Test G exclusions); one record per copy × checkout; named acceptor; PASS recorded before Josh is asked.

Harness (`pg1_harness.py` blob `8bda680d3830a03756ac15f8b8210fd5792f2d48`) and records:

| clause | evidence | rule |
|---|---|---|
| inputs (a) laptop-pair copy, (b) staged VPS-bundle copy | (a) ceremony backup `/home/josh/backups/z12-s3-restore-bundle-20260929T222529Z/pre-placement-laptop-cc/` used in place; (b) copied from `.../vps-pull-staged` → `.../z12-pg1-20261001T021808Z/bundle-copy` (`copy-b.json` `all_equal` True, `source_stable_across_3s` True) | (a) deviation: in-place copy, not a fresh scratch copy. Harmless: before/after sha256 equal (independent rehash below). (b) matches. |
| base `e4ebf982` vs fold `29f47f65` | part1 headers: BASE rev `e4ebf982b1989fd9066d610b94853bc68bf70d37` blob `53494b7c56896d25040f3e7fd7c4046da7d0ab05`; FOLD rev `29f47f65058790240b2f9c6a0a5bc4d82171b42d` blob `5e8945accb476b0727bf07a3ab2890dccc87f650`. Checkouts still at those HEAD/blob values (`git hash-object`). `new_api_present` False vs True. | match |
| four loads, each own fresh process/scope | distinct pids and cgroup scopes; non-overlapping UTC windows: a-base 02:20:41–02:21:17Z pid 3371677 `run-u77827`; a-fold 02:21:41–02:22:17Z 3380737 `run-u77849`; b-base 02:23:15–02:23:38Z 3394445 `run-u77878`; b-fold 02:23:44–02:24:06Z 3398839 `run-u77893`. Each `memory.max`=6442450944, `memory.swap.max`=0. | match |
| PYTHONHASHSEED pinned | `"0"` on all four part1 headers | match |
| sys.path[0] + printed module path/rev; void if wrong | each `sys.path[0]` equals its checkout; `void` False; dirname(nf file) is that checkout | match |
| `_prune_synapses()` at all new parameters at defaults | harness `ret = g._prune_synapses()` (no kwargs) | match |
| compared fields | report `COMPARED` is 14 keys: return, removed_count, removed_ids_sha256_in_order, removed_ids_sha256_sorted, pruned_events, synapses_before, synapses_after, pre_state_digest, state_digest_before_checkpoint, state_digest, checkpoint_sha256, checkpoint_size, counts_after_restore, counts_final. Digest is the driver's `state_digest` (items() order, weight/peak/low_weight_steps/inactive_steps/salience/creation_time, dirty set, confirmation history, node set, timestep, pruned events) after `checkpoint()`, matching Test G order. Serialized comparison is sha256 of `Graph.checkpoint()` bytes (sidecars not in that file). | match the packet's 14-field list; extra pre-checkpoint digest is additional |
| per-copy record + named acceptor | records exist; this pair's acceptor is a fresh law enforcer (`le-039`), not yet read | path/acceptor deviations below |
| MemAvailable ≥ ~8 GB | standing gate used 3.0 GiB (part1) / 6.0 GiB (part2). Actual readings 9.154–9.329 GiB. | this run would also have passed the plan's ~8 GB text. Standing lowered gate: packet-named deviation; **needs an Executive note** if it remains the reusable PG-1 gate. Harmless for these six loads. |
| artifact path | tests branch `handoffs/z12-want-hub-build/pg1/` vs plan designation `handoffs/z12-want-hub-d/returns/pg1/` | plan allows rename in the recording commit. Harmless if the merge ask cites the artifact commit and the acceptance commit. **Executive note** that the designated path moved. |
| acceptor identity | plan: Chief/Executive with the delta pair. Packet: fresh law enforcer. | packet substitution. **Executive note**. Not a procedure fail of the builder. |

Independent recompute from the six part1 JSONs: all 14 fields equal BASE vs FOLD on both copies; `pg1-compare.json` byte-matches that recompute (`all_compared_fields_identical_and_preconditions_hold` True for a and b).

Independent sha256/stat of sources (this review; no live checkpoint opened):

| path | size | sha256 |
|---|---|---|
| ceremony-backup `main.msgpack` | 230539966 | `7e4577868631de4389e5a6ef994f8b97d447690d0d1621e27f944189042c3a77` |
| ceremony-backup `vectors.msgpack` | 1000876671 | `93ed891fa2a0812382dbb7da8b287108fbc54fdb1560c169c683244f43bcb05e` |
| `vps-pull-staged/main.msgpack` | 141784662 | `8cf6ef22f0e75756fc0d0a7e706258ef390ca030c9f24c0bbc80f6203d3d90a1` |
| `vps-pull-staged/vectors.msgpack` | 376141985 | `bb3479b4e2b3795f4777b39292269e664fcd216f4a8816921449170b2ad4b019` |
| `bundle-copy/main.msgpack` | 141784662 | `8cf6ef22f0e75756fc0d0a7e706258ef390ca030c9f24c0bbc80f6203d3d90a1` |
| leftover scratch `a-base.msgpack` / `a-fold.msgpack` | 230539966 | `902ad15a532ae0ec214dd50516c487b5dd9e0eee1691554967619cd7f90470ce` (equal) |
| leftover scratch `b-base.msgpack` / `b-fold.msgpack` | 137365123 | `6e0a36a565bbec1e57280478b2af9a5a42ea781dc28c3e4a4a12337b122b2a9d` (equal) |

Those match the artifact records. Scratch files were hashed in place; not written.

### (2) Evidence sufficient for the hot-path-unchanged claim on real graphs

**PASS-WITH-NOTES**

Claim PG-1 exists for: the shared hot path Syl's process runs is unchanged on the default path, on REAL graphs.

What the 14-field identity **proves**:

1. **Copy (a), empty removal set.** Restore 7253 nodes / 138753 synapses / 517 hyperedges / timestep 33637. `_prune_synapses()` return 0; removed-id sha256 is the empty digest `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; `pruned` events `[]`; synapse count unchanged. `pre_state_digest` (`ba6a6fcf…`) ≠ post digest (`4626743e…`): counters moved with no removals. BASE and FOLD share that post digest and the serialized checkpoint sha256 `902ad15a…` (230539966 bytes, same size as the source, different sha256 — a re-serialize). This is the non-vacuous empty-removal proof: every evaluated non-protected synapse's bookkeeping, full state, and checkpoint bytes match.
2. **Copy (b), 10,433 real removals.** Restore 19224 / 19390 / 2179 / 80259. Return 10433; synapses 19390 → 8957; `pruned` `[[10433, 80259]]`. Removal-order sha256 `e0df6abb…` ≠ sorted-id sha256 `60780968…`, so default-path order is not an unconditional sort-by-id, and that order is identical BASE vs FOLD. Post digest `9a96ef48…` and checkpoint sha256 `6e0a36a5…` (137365123 bytes) identical. This is the load-bearing removal comparison the plan named (10,433).

What it **does not** prove (an Executive/Josh reading PASS must still know):

1. Real synapse ids were not checked for uuid4 *shape*; only that (b)'s removal order is not id-sorted.
2. Neither copy is the live CC checkpoint. (a) is a ceremony-backup laptop pair (hashes above). (b) is the staged VPS bundle (`sprout_degree_cap` / wants / rim as the plan described that artifact), not Syl's graph.
3. A live graph whose shape differs from both copies (including a different hyperedge set than (a)'s 517) is untested.
4. `ng_tract` is the installed wheel `/home/josh/.local/lib/python3.12/site-packages/ng_tract/__init__.py` version `0.1.0`, shared by both checkouts. Provenance of that wheel is unverified. A store-implementation difference would be invisible here.
5. Serialized-byte identity is under `PYTHONHASHSEED=0`. The plan already says `_serialize_hyperedge` uses unsorted sets. Another seed is untested.
6. Competing/armed path, daemon dream-loop wiring, `#825` save-and-restore of deletions, and contended `_step_lock` are outside PG-1.
7. `state_digest` does not list hyperedges as its own field; hyperedge identity rides on `counts_*` plus checkpoint bytes.
8. The digest implementation is the one Test G uses (same driver file, blob `b5d439f6…`). That is not a second digest implementation.

### (3) Independence of the acceptor's re-run (its own §9.5)

**PASS-WITH-NOTES** (procedure-level; the acceptance file is unread until after this commit)

A re-run that shares the checkouts, the `ng_tract` wheel, CPython 3.12.3, `PYTHONHASHSEED=0`, the copies, and the driver's `state_digest`:

- **is** independent of the builder's processes, PIDs, transcription, and this-run wall times.
- **is** a confirmation that those recorded hashes are reproducible on the same artifacts.
- **is not** independent of the digest implementation, the wheel, the interpreter, or the copies. It cannot by itself catch a shared-tool bug that would make BASE and FOLD look identical for the wrong reason.
- Shared-wheel + shared-digest is the right comparison *for* "same default path on two engine blobs." It is the wrong comparison *for* "the digest matches a second implementation of Test G's field list."

Acceptor-file specifics (whether they re-timed Part 2, which scratch they used, five corrections) wait on check 6.

### (4) Part 2 numbers and the wall-difference sentence

**PASS-WITH-NOTES**

Acceptability of the `_step_lock` hold is **not judged** (Executive decided).

Recomputed from `part2-first.json` / `part2-fold.json` (FIRST `8e578532` blob `96d12f50…`; FOLD `29f47f65` blob `5e8945ac…`; each own process; copy (a) `main.msgpack` sha256 equal before/after; sources unchanged):

| variant | call | wall s | cpu s | inner prune wall s | outside (orch) wall s | eligible | removed | conducting | held-back | floors_ok | synapses after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FIRST | 1 | 12.9853 | 12.9422 | 2.6529 | 10.3324 | 102145 | 5000 | 0 | 15 | True | 133753 |
| FIRST | 2 | 12.2614 | 12.2654 | 2.5596 | 9.7018 | 97145 | 5000 | 220 | 15 | True | 128753 |
| FIRST | 3 | 12.2691 | 11.8073 | 2.4599 | 9.8092 | 92145 | 5000 | 484 | 15 | True | 123753 |
| FOLD | 1 | 14.8211 | 13.654 | 4.5015 | 10.3196 | 102145 | 5000 | 0 | 15 | True | 133753 |
| FOLD | 2 | 14.1947 | 13.3146 | 3.8209 | 10.3738 | 97145 | 5000 | 220 | 15 | True | 128753 |
| FOLD | 3 | 12.0647 | 11.7133 | 2.8182 | 9.2465 | 92145 | 5000 | 484 | 15 | True | 123753 |

Fold call 1 conducting = 0 (table typo guard: records say 0). `outside = wall - inner` matches on every row. Synapses 138753 − 3×5000 = 123753. Eligible drops by 5000 per call.

Packet's rounded series 12.99/12.26/12.27 vs 14.82/14.19/12.06 matches the records.

Orchestrator-before-prune (abort at `_prune_synapses`, no mutation): FIRST wall 10.1941 / cpu 10.1261; FOLD wall 11.2918 / cpu 10.5091. Call-1 outside-prune ~10.33 / 10.32 s. **Supported:** ~10 s of every call is the orchestrator before `_prune_synapses`.

Isolated validation CPU (×3, absent id sorts last, fingerprint unchanged, raise at that id): FIRST 1.2501 / 1.2965 / 1.3173, mean **1.287967 → 1.288**; FOLD 1.9342 / 1.9788 / 1.9706, mean **1.9612 → 1.961**; delta **+0.673233 → +0.673**. Arithmetic matches the artifact.

Fold-minus-first:

| call | Δ wall | Δ cpu | Δ inner prune wall |
|---|---|---|---|
| 1 | +1.8358 | +0.7118 | +1.8486 |
| 2 | +1.9333 | +1.0492 | +1.2613 |
| 3 | −0.2044 | −0.094 | +0.3583 |

Builder sentence: "the wall difference is within the noise; the steady piece is the isolated validation CPU +0.673 s."

- **Supported:** the validation-CPU delta is steady across three runs; call 3 wall is within the FIRST intra-process spread (12.9853−12.2614 = 0.724 s); outside-prune on call 1 is essentially identical.
- **Over-stated** if applied to all three calls: calls 1 and 2 wall deltas (~1.84 / 1.93 s) exceed FIRST's call-to-call spread, and inner-prune wall is higher on the fold on every call. Call 3 of FIRST being slower than call 3 of FOLD is true and is not a license to treat calls 1–2 as noise. The +0.673 s validation CPU is the steadiest fold-only piece; it does not account for the full call 1–2 wall delta.

Counts identical across variants: competing_ids 106825, excluded_ids 21534, order_key 106825, max_removals 5000, F_links 4127, protected_nodes 183, held_back_last_link 15, call-1 eligible 102145 (ceil/B = 21).

### (5) Scope / safety

**PASS**

- Audit write-mode opens on part1: only the named scratch checkpoint under `/home/josh/backups/z12-pg1-20261001T021808Z/scratch/`. Mutating os/shutil calls empty. Spawned: `git`, `pgrep`, `systemctl`.
- Part2 write-mode opens empty (in-memory only). `save()` is not in the harness.
- Checkpoint path assertion requires `.msgpack`, `/z12-pg1-`, and forbids `/data/checkpoints`.
- Source `main.msgpack` sha256 equal before/after on every load; sidecar dicts equal; independent rehash of leftover sources matches.
- Gate records: `cc-ng-daemon.service` inactive, daemon process count 0, on all six loads.
- non_library_read_opens are the ceremony backup / bundle-copy / scratch checkpoint. No `~/NeuroGraph/data/checkpoints`, no live tract path, no `.bashrc`.
- Artifact and records: ids, hashes, counts, booleans. No want text in the named artifact files.

### (6) Acceptor five corrections and builder two observations

**DEFERRED** — unread until after this first-findings commit, per ADDENDUM 3.

Builder observations already visible in the artifact (not yet adopted/disputed against the acceptor):

1. Plan §4A.3 probe: 16 last-link holds; both engine variants report `held_back_last_link` = 15 with `competing_ids` = 106825 (the plan's post-hold competing count). PG-1 default path is unaffected.
2. Copy (a) re-serialized checkpoint: same byte size as source (230539966), different sha256. Independent rehash confirms.

### (7) Numbered corrections; numbered "not verified"

**Corrections**

1. The Part 2 wall-difference sentence is over-stated for calls 1 and 2. Keep the +0.673 s isolated-validation CPU as the steady fold delta; do not describe the +1.8 s call-1/2 wall deltas as noise.
2. Copy (a) was the ceremony backup in place rather than a fresh scratch copy. Harmless here (hashes equal); a future PG-1 should copy (a) the way (b) was copied, or record the Executive note that in-place read-only is accepted.
3. `pg1_harness.py` imports the golden driver by live worktree path. The blob at the artifact commit equals the current file (`b5d439f6e68d46e3d7598de1d1176c0cc53a30c0`), but part1 headers do not record that driver blob. Record it.
4. Standing MemAvailable gate 3.0/6 GiB vs plan `≥ ~8 GB`: harmless for this run (readings ≥ 9.1 GiB). Needs an Executive note if it is the reusable gate.
5. Artifact directory is on the tests branch under `handoffs/z12-want-hub-build/pg1/`, not the plan's `handoffs/z12-want-hub-d/returns/pg1/`. Harmless if the merge ask cites both full hashes.
6. Acceptor identity is a fresh law enforcer (packet) vs plan "Chief / the Executive together with the delta pair." Executive note.

**Not verified**

1. uuid4 *shape* of real synapse ids.
2. Live Syl graph ≡ either copy (ceremony backup or staged VPS bundle).
3. Provenance of the shared `ng_tract` 0.1.0 wheel.
4. Default-path identity under a PYTHONHASHSEED other than 0.
5. `#825` (pruned ids absent after save and restore). PG-1 never calls `save()` and does not restore its scratch checkpoint.
6. Daemon slice, arming, contended `_step_lock`.
7. A second digest implementation of Test G's field list.
8. Part 2 run-to-run variance (one process per variant, as the brief asked).
9. Acceptor re-run details and five corrections (isolation; next commit).
10. Hyperedge *membership* identity beyond counts + seed-0 checkpoint bytes.

## Overall verdict

**PASS-WITH-NOTES** on checks 1–5 from the artifact + plan + packet. Check 6 and Agree/Disagree/Add wait on the post-isolation reads. No `_step_lock` acceptability judgement.
