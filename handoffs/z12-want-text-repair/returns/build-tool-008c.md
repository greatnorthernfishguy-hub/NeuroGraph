```
---- Changelog ----
[2026-10-01] Claude Code (claude-sonnet-5-5, Z12 BUILD worker seat, dispatch #12322, 12 ANCHOR_FAILED extraction) - build-tool-008c: NOT RUN. The pre-flight
             failed on load (6.01 >= 6). Nothing was extracted; no output file exists. Readings only. No want text anywhere in this file.
-------------------
```

# build-tool-008c - the 12 ANCHOR_FAILED extraction: **NOT RUN, pre-flight failed on load**

Lane `z12-s3-restore-bundle-20260929` - Z12 - dispatch #12322 (same-thread resume, Exec P454) - tool branch `cc-laptop-want-repair-tool-20260930`. Related: [[NeuroGraph]] - [[The Laws]]

**Status: STOPPED at the pre-flight, as the brief requires ("any off = STOP, no retry-until-pass").** No `systemd-run` scope was started for the extraction, the extraction script never executed, the run directory's copy was not opened, and **no output file exists** (the run directory's `review/` still holds only the two files from build-tool-008). Nothing was retried and no cap was touched.

## 1. The pre-flight reading (one reading, taken immediately before the launch)

| Time (UTC) | load1 | MemAvailable | SwapFree | `cc-ng-daemon.service` | daemon process | Gate | Result |
|---|---|---|---|---|---|---|---|
| (dispatch gate) 01:51:38 / 01:52:38 | 3.88 / 3.86 | 9,377,908 / 9,734,012 kB | - | inactive | none | quoted; other-worker-turns 1 | |
| **01:53:48** | **6.01** | 9,166,876 kB (8.74 GiB) | 9,109,748 kB | inactive | 0 | load < 6 | **OFF: 6.01 >= 6 -> STOP** |

Every other gate was fine (MemAvailable >= 3 GiB, daemon inactive, no daemon process). I took one reading and did not sample again to wait for it to clear: that would be the retry-until-pass the brief forbids. A later informational `/proc/loadavg` read 5.64 / 5.13 / 4.92 (1 / 5 / 15 min); the top CPU consumers at that moment were a `find` (62%) and `opencode` (29%), neither mine. The load rose between the dispatch's two readings and mine; I cannot say what drove it from one reading.

## 2. What exists, so a re-dispatch only needs the gate

- **Off-repo script, ready, never executed:** `/home/josh/backups/z12-want-text-repair-feasibility-20260930T183126Z/anchor_failed_extract.py`, sha256 `bb659a33c7c8623dec8421e27ed8764172bae75d9b9be33296387afb3a127e10` (syntax-checked only). Design, so the Chief can read it before the re-run (no text is involved in this description):
  - it re-verifies all six files of `.../z12-want-text-repair-20261001T011502Z/copy/` against `copy-hashes.json` (size and a full sha256 stream) and STOPS if any differs;
  - takes the ids from `reports/outcome-table.json` with `outcome == ANCHOR_FAILED` and STOPS if the count is not 12;
  - uses the tool's OWN functions (read-only import of `oneshot-tool/want_text_repair_oneshot.py`): `stream_graph_nodes`, `load_content_subset`, `load_pinned`, and `Classifier.classify_node` to re-derive each record (asserting it is still `ANCHOR_FAILED` with the recorded `detail`), `text_class` and `sha16`; it does not re-implement the anchor logic;
  - per want it records the id, `source_node`, class, `detail`, want-text length and `sha16`, source-content length, the tool's `str.count` and an overlapping-scan occurrence count, all 0-based character offsets, gaps between occurrences, adjacency / overlap / identical-context booleans, and (in the PRIVATE file only) 160 characters before and after each occurrence, the first 160 characters of the want text, and the verbatim left-list entry for that id;
  - writes the one file `review/anchor-failed-12-<UTC>.md` with `os.open(..., O_EXCL, 0o600)`; stdout carries only ids, counts, lengths, offsets, hashes and booleans; a numeric-only facts JSON goes beside the script.
  - expected resource use is light (the P3 + P2 + pin measurements: about 0.5 GiB heap), well under the 1 GiB scope.
- **Tool and trees unchanged:** tool worktree HEAD `f470e861db89de603df2e16ba79898419b55254f` (0 porcelain lines), tool sha256 `bf78defaadd28a91d935d7999ab4f136904aafea296afe3a2c6865e05e6f883c` (this return is the only repo change). The extraction was not started, so no source or COPY reading was taken this turn.

## 3. What I did NOT do / verify

Everything the dispatch is for: the 12 ids' count check, the per-id offsets, lengths and occurrence counts, the bounded excerpts, the output file (path, sha256, size, mode), and the copy re-verification. No `--apply`, rewrite, `phase2-backup`, rollback or `--josh-go*` was touched; no daemon, tract, Syl directory or `~/.bashrc`.

## 4. What is needed

A re-dispatch of the same extraction once the 1-minute load is below 6 (the pre-flight prints the readings; the script above is the whole job: one scope, one run). I have stopped.
