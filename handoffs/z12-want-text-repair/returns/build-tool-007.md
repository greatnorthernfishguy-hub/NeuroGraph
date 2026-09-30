```
---- Changelog ----
[2026-09-30] Claude Code (claude-sonnet-5-5, Z12 BUILD worker seat, dispatch #11771, TURN B-1) - build-tool-007: the four memory
             probes were NOT RUN. The pre-flight for P1 failed on load (6.91 >= 6). The streamed checkpoint copy and the
             off-repo probe script were completed first and are kept. Readings and evidence only; no measurement exists yet.
-------------------
```

# build-tool-007 - TURN B-1 (probes P1-P4) - **NOT RUN: pre-flight for P1 failed on load**

Lane `z12-s3-restore-bundle-20260929` - Z12 - dispatch #11771 - tool branch `cc-laptop-want-repair-tool-20260930`.
Related: [[NeuroGraph]] - [[The Laws]]

**Status: STOPPED at the P1 pre-flight, as the brief requires ("any off = STOP, run nothing, no retry-until-pass").** No probe ran: no `systemd-run` scope was started for a probe, no `memory.peak`, sampler log, `ru_maxrss`, key count or byte total exists, and **no peak was measured, so no per-pass pre-flight is re-derived** (section 4). `returns/build-tool-006.md` section 5 stands as the last derivation. Nothing was retried, the cap was not touched, and the vectors were never loaded. The only repo write is this file (docs-only).

## 1. The pre-flight readings (script `/tmp/z12_preflight.sh`, printed before each step)

| Time (UTC) | Before | load1 | MemAvailable | SwapFree | daemon unit | daemon process | Result |
|---|---|---|---|---|---|---|---|
| 18:30:56 | (first try) | 3.54 | 7,395,320 kB (7.05 GiB) | 9,354,368 kB | inactive | filter printed 1 match | **my filter was wrong**: the match was my own `bash -c` wrapper, whose command text contains the filter pattern (same thing as #11707). No daemon process exists. |
| 18:31:13 | (fixed filter, see below) | 4.08 | 7,306,480 kB (6.97 GiB) | 9,354,396 kB | inactive | 0 | PASS |
| 18:31:26 | the streamed copy | 3.54 | 7,366,304 kB (7.03 GiB) | 9,354,556 kB | inactive | 0 | PASS |
| **18:34:23** | **probe P1** | **6.91** | 7,522,256 kB (7.17 GiB) | 7,356,788 kB | inactive | 0 | **FAIL: load 6.91 >= 6 -> STOP** |

- The daemon-process filter was changed, once, between 18:30:56 and 18:31:13. The first version matched on the whole command line (`pgrep -f`). The second matches only processes whose name (`comm`) starts `python` or `cc-ng` and whose command line names the daemon or sidecar. This fixes a self-match; the gate still requires zero daemon processes. It did not loosen any threshold.
- One reading was taken at 18:34:23 and I did not sample again to wait for a pass. The gate is the Chief's; re-measuring until it clears is what the brief forbids. A later `/proc/loadavg` (informational, not a retry) read 1-min 6.99, 5-min 4.88, 15-min 3.00.
- What moved the load: the load was 3.5-4.1 before the copy and 6.9 about three minutes after it. The copy step I ran (three passes over the two large files, about 4.4 GB of sequential reads plus 1.2 GB of writes) is a likely contributor, alongside the desktop. I cannot separate them with one reading. Top CPU at the later reading (informational): `opencode` 18%, `t3code` 10.5% and 5.8% (RSS about 0.6 GB and 3.3 GB), xfce4-taskmanager 9%. `SwapFree` fell from about 9.35 GB to 7.36 GB between 18:31 and 18:34, so the desktop was paging.
- **Suggestion for the re-dispatch (a decision for the Chief, not something I changed):** the copy already exists, so the re-dispatch needs no further heavy IO from me; a quiet-moment read of the load before dispatch is all that is needed.

## 2. What WAS done before the stop (read-only with respect to the checkpoint; all under the off-repo directory)

Off-repo directory (kept, do not delete): `/home/josh/backups/z12-want-text-repair-feasibility-20260930T183126Z/`

**Streamed copy of the six CC-laptop files** from `~/.claude/plugins/neurograph/checkpoints` by name (no directory listing, no generations directory, nothing under Syl's `~/NeuroGraph/data/checkpoints`). 4 MiB chunks, `open(..., "xb")` (never overwrites), sha256 of each source BEFORE, AFTER, and of the copy: **all three equal for all six files; the script printed `ALL_OK`**. The source inode, size and mtime were checked unchanged. Inode evidence (judged by inode, not by link count):

| File | Bytes | sha256 (source before = after = copy) | Source inode before = after | Source nlink | Copy inode |
|---|---|---|---|---|---|
| `main.msgpack` | 230,539,966 | `7e4577868631de4389e5a6ef994f8b97d447690d0d1621e27f944189042c3a77` | 19413931 = 19413931 | 2 | 22167962 |
| `vectors.msgpack` | 1,000,876,671 | `93ed891fa2a0812382dbb7da8b287108fbc54fdb1560c169c683244f43bcb05e` | 19413942 = 19413942 | 2 | 22166327 |
| `main.msgpack.activations.json` | 754,451 | `0b6fe94e5266ff5eccf412731275b089117fac2cdcad0567c2a62647c9ed3bdb` | 19408030 = 19408030 | 1 | 22165370 |
| `main.msgpack.guard_state.json` | 191 | `c3520eff093687b8a73a2d93b3859b8d233c97b47aea34b5db4050d60acb3932` | 19413930 = 19413930 | 1 | 22167957 |
| `main.msgpack.manifest.json` | 217 | `6e7a2ed8a7889d8bb232cb7fe4e2cc91a541532b1f4f59cc2bf96f14dce69325` | 19413954 = 19413954 | 1 | 22167963 |
| `commons.msgpack` | 115,754 | `986c8e93670e6d47d8db2cc18836abc0ea6df0e8cf97feba59bf8fea53495918` | 19405068 = 19405068 | 1 | 22167964 |

The live `main`/`vectors` having nlink 2 is consistent with the Exec P428 finding (hard-linked into a generations directory, which I did not open). The copy files are new inodes (fresh files, not links), so the probes cannot write through to the live checkpoint. The copy manifest is `copy_manifest.json` in the directory.

**Off-repo scripts (sha256 by `sha256sum`):**
- `copy_six.py` `957788f6b48a8ded118b1375a204442aac81f3c704acf8055831f4989955e77c` (the streamed copy, ran).
- `probe.py` `7e0bceb77c1a61eba0dbdf22fb1e6f75ecbcb7d55622570bc4afb960a62d4037` (P1-P4 as modes; **it has never been executed**, it was only syntax-checked with `ast.parse`; I also confirmed `systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 true` works in this user session).

What `probe.py` does, so the re-dispatch reviews the same thing: each mode reads the COPY only; a sampler thread appends a line every 0.25 s to `sampler_<P>.log` (cgroup `memory.current`, `memory.stat` anon and file, `memory.events` `oom`/`oom_kill`, running maxima) so an OOM-kill leaves a last-seen value; a surviving run writes `result_<P>.json` with `memory.peak` (which includes page cache, so the anon maximum from the sampler is reported beside it), `ru_maxrss`, wall time and counts/bytes only. Modes: `P1` keys plus per-field byte totals (`skip()` with `tell()` deltas, nothing decoded); `P3` descends the `main.msgpack` maps with one entry decoded at a time, retaining node metadata only, and writes the ids-only file `want_source_node_ids.json` (the `source_node` of every `kind=want` node, a superset of the S sources); `P2` reads that ids file as its keep set, so **P3 must run before P2** (a change of order from the list in build-tool-006, with the same four probes, one at a time); `P4` imports `neuro_foundation` from the read-only pin worktree (`python3 -B`, no bytecode written) and runs `Graph().restore` on the copy. The Unpacker buffer is capped at 256 MiB in `probe.py`; if a top-level `skip()` needs more, that shows as a `BufferFull` finding rather than a larger buffer.

## 3. Heads and unchanged-tool evidence

- Pin/stack head (frozen): `c7921b8436fb174c3f70fcf02827f16bb16deff0`. Pin worktree HEAD (`git rev-parse`): `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a`; `git status --porcelain --ignored` = 0 lines (clean; never edited).
- Tool worktree HEAD before this commit: `95c39e57ec38bb6fb15d1e76e281f66cda5bf937` (the build-tool-006 commit); clean (0 porcelain lines); `git diff --stat 95c39e5..HEAD -- oneshot-tool tests` is empty. Tool sha256 `cbc38bf4a02adfcab6ae05c5796bb20c818103a71c23ab20570835460e42565b`, tests sha256 `9b89f687764c6ef90fa7df01a34d9c94210b8a170abe73643086fb13901e2ab6` (both equal build-tool-004/005/006).
- No file under the tool's `oneshot-tool/` or `tests/` was edited; no Phase 1 step, `--apply`, `phase2-backup`, `--step rollback`, `--josh-go*` or P3(b) was run; no daemon or unit was started or stopped; no `~/.bashrc`; no raw want text or excerpt in this file.

## 4. The re-derived per-pass pre-flight - NOT DONE (nothing measured)

Every number in the derivation needs a measured peak P, and none exists. Until P1-P4 run, build-tool-006 section 5 stands unchanged: light hard-capped probes 3.0 GiB (1.0 + 1.1 + 0.3 + 0.6); heavy loads (canonical `Graph.restore`, the unchanged tool) keep 8 GiB; Pass V and the graph-metadata stream have **no stated number**. I am not estimating a figure in place of a measurement.

## 5. What I did NOT verify

The key count and per-field byte split of `vectors.msgpack`; the content-subset size; whether the top-level `skip()`s of `main.msgpack` fit a 256 MiB buffer; the canonical `Graph.restore` peak under 1 GiB (or its OOM); any `memory.peak`, `ru_maxrss` or wall time; that `probe.py` runs at all (never executed). Also not verified: what drove the load to 6.91 (one reading, section 1).

## 6. What I need

A re-dispatch of P1-P4 at a moment when the pre-flight passes (1-min load < 6, `MemAvailable` >= 3 GiB, daemon inactive). The copy and scripts above are ready, so the re-run is four short scope launches (order P1, P3, P2, P4), and I will run the same pre-flight before each. I have stopped.
