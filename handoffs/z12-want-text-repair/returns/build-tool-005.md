```
---- Changelog ----
[2026-09-30] Claude Code (claude-sonnet-5-5, Z12 BUILD worker seat, dispatch #11707, TURN B) - build-tool-005: TURN B was NOT RUN.
             The pre-flight memory reading was below the gate; per the brief nothing was run. Readings only.
-------------------
```

# build-tool-005 - TURN B (Phase 1 dry run on a COPY) - **NOT RUN: pre-flight failed**

Lane `z12-s3-restore-bundle-20260929` - Z12 - dispatch #11707 - tool branch `cc-laptop-want-repair-tool-20260930`.
Related: [[NeuroGraph]] - [[The Laws]]

**Status: STOPPED at pre-flight, as the brief requires ("if ANY is off: STOP, run nothing, report the readings").** No copy was made, no classify, no rewrite, no verifier, no systemd-run scope was started, nothing was written under `/home/josh/backups/`, and no checkpoint file was opened, hashed or stat'ed. No `--apply`, `phase2-backup`, rollback, `--josh-go*` or P3(b) was touched. Nothing was retried, and the cap was not raised. The only write is this file (docs-only, on the tool branch).

## 1. The pre-flight readings, taken at 14:04:28 UTC (before anything else)

| Reading | Value | Gate | Result |
|---|---|---|---|
| **`MemAvailable`** | **7,768,848 kB (7.41 GiB)** | >= 8,388,608 kB (8 GiB) | **OFF - 619,760 kB (about 0.59 GiB) short** |
| 1-min load (`/proc/loadavg`) | 1.27 (5 min 0.90, 15 min 0.93; 4 runnable / 1004 tasks) | < 6 | ok |
| `systemctl --user is-active cc-ng-daemon.service` | `inactive` (rc 3) | inactive | ok |
| `cc-ng-daemon-recover.timer` (extra, informational) | `inactive` | (not a listed gate) | ok |
| a `cc-ng-daemon` / `cc-ng-service` / `neurograph_rpc` process | none (the one line the process filter printed was my own shell wrapper, whose command text contains the pattern) | none | ok |
| `SwapFree` (informational) | 4,096,892 kB | - | - |

The dispatch's own readings were `MemAvailable` 8,455,360 kB at 14:02:12 UTC and 8,625,900 kB at 14:03:12 UTC; about a minute later it was 7,768,848 kB. I took ONE reading only and did not sample again to wait for it to pass: the gate is the Chief's conditional, and re-measuring until it clears would be working around it. Largest resident processes at that moment (informational, `ps` by RSS): an `opencode serve` (about 1.2 GB), the T3 Code server (about 0.56 GB), and two `claude` processes (about 0.3 GB each, one of them this session).

## 2. The rest of pre-flight (all fine - so a re-dispatch only needs the memory gate)

- **Tool worktree** `/home/josh/NeuroGraph-worktrees/z12-want-repair-tool-20260930`: `git rev-parse HEAD` = `95b0bd66ce5a5f0a463f4fbd51d69a20d8b379eb` (the A2c return), `git pull --rebase` up to date, clean; tool sha256 `cbc38bf4a02adfcab6ae05c5796bb20c818103a71c23ab20570835460e42565b`, tests sha256 `9b89f687764c6ef90fa7df01a34d9c94210b8a170abe73643086fb13901e2ab6` - both equal build-tool-004; `git diff --stat 95b0bd66..HEAD` on `oneshot-tool/` and `tests/` is empty.
- **Pin worktree** `/home/josh/NeuroGraph-worktrees/z12-want-repair-pin-ae798b9`: HEAD `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a`, `git status --porcelain --ignored` = 0 lines, `cc_ng_organism.py` sha256 `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2`, blob `a3aa8a0ddb6a89fe468a9a20beadc5e381624cab`. The frozen stack head the run record must also carry is `c7921b8436fb174c3f70fcf02827f16bb16deff0` (P1 records both by itself).
- **Inputs for the eventual run:** the daemon script (docs worktree `daemon-recall-756-20260930`, HEAD `c298571e`) exists and still assigns `CC_NG_WORKSPACE` (`:579`) and `CHECKPOINT_DIR` (`:588`) in the recorded two-line shape; `/home/josh/backups` exists; the home filesystem has 187 GB free.

## 3. What is needed to run TURN B

`MemAvailable` >= 8 GiB again with the rest of the gate unchanged (load < 6, daemon inactive and no daemon process, alone). Then ONE run:
`systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0` around the tool's two Phase 1 steps (`--step classify`, then `--step rewrite --run-dir <that run> --provisional-approve-all`), with the peak read from the scope (`MemoryPeak`) before it is reaped. No tool or test change is needed.

## 4. Not verified

Everything TURN B was for: the real-file copy, classify, rewrite, V1-V19 (V13 fidelity on the real synapses map, V15/V16 presence), the PRE-node report line, the peak memory under the 6 GB cap, the review-file entry and hint mark counts. The P3(b) real run and c026-C4 remain as the brief maps them (c026-C4 -> the P3(b) step on the COPY before Phase 2; not this turn, not any `--apply`).

I have stopped.
