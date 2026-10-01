```
---- Changelog ----
[2026-10-01] Claude Code (claude-sonnet-5-5, Z12 BUILD worker seat, dispatch #12296, INDEPENDENT CANONICAL-PATH CHECK) - build-tool-008b: the PRE-DELTA
             tool (canonical Graph().restore + SimpleVectorDB().load; sha256 cbc38bf4...) was run ONCE on the same real bytes. Its 8 reports and 2
             review files are BYTE-IDENTICAL to build-tool-008's streamed run. DECISION TABLE: (A). Nothing patched, nothing concluded beyond
             the table. Ids / counts / hashes / booleans only; no want text. Not self-accepted.
-------------------
```

# build-tool-008b - the independent canonical-path check of the zero-candidate classify result

Lane `z12-s3-restore-bundle-20260929` - Z12 - dispatch #12296 (same-thread resume, Exec P454) - tool branch `cc-laptop-want-repair-tool-20260930`. Related: [[NeuroGraph]] - [[The Laws]]

## 0. Result, and the decision-table branch

**Branch (A).** The canonical path ALSO yields all-NONE: **0 of 118 candidates (106 `NONE`, 12 `ANCHOR_FAILED`), and every one of the 8 report artifacts and both review files is BYTE-IDENTICAL to the streamed run's** (same file sha256, same sizes). There is no divergence of any kind, so the streamed reader is not the cause of the zero-candidate result on this checkpoint. Per the Chief's table I **stop Phase 1 here and report; I do not conclude anything about the repair** (the Chief escalates to the Executive and the plan-004 author). What this does and does not establish is in section 6.

## 1. Pre-flight (printed first)

| Time (UTC) | load1 | MemAvailable | SwapFree | `cc-ng-daemon.service` | daemon process | Result |
|---|---|---|---|---|---|---|
| (dispatch gate) 01:39:01 / 01:40:01 | 4.00 / 3.12 | 9,194,308 / 9,834,252 kB | - | inactive | none | quoted; other-worker-turns 1 |
| **01:41:14 (immediately before the launch)** | **5.35** | **9,563,100 kB (9.12 GiB)** | 9,151,944 kB | inactive | 0 | **PASS** (>= 4 GiB, load < 6, daemon inactive, no daemon process) |

One reading, no retry. (Load 5.35 is under the builder's < 6 gate; it is higher than the dispatch's 3-4 because my own previous step's I/O had just ended.) Tool worktree HEAD `bbf75ee927c7f036fc6aa84078a4e88c5fedad6e` (clean, 0 porcelain lines before and after); pin worktree `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a` (0 porcelain lines, `--ignored`, before and after).

## 2. What was run (exactly as build-tool-008, with only the tool file changed)

- **The pre-delta tool**, extracted with `git show 09a032c3426baf8307cb968496c65f779894304b:handoffs/z12-want-text-repair/oneshot-tool/want_text_repair_oneshot.py` to `/home/josh/backups/z12-want-text-repair-feasibility-20260930T183126Z/pre_delta_09a032c/want_text_repair_oneshot.py` (mode 0444, never edited): **sha256 `cbc38bf4a02adfcab6ae05c5796bb20c818103a71c23ab20570835460e42565b`** (the expected value; recorded again by the tool in its run record and by the wrapper). Its `analyze()` is the canonical path: `load_pair` = `pinned.nf.Graph().restore(main.msgpack)` then `pinned.ui.SimpleVectorDB().load(vectors.msgpack)` (the old file, lines 1547-1554, checked by reading). The current tool was not edited, nothing was patched, no guard was bypassed.
- **Command** (identical to 008 except the tool path): `env -u PYTHONPATH -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 python3 -B classify_wrap.py <feasibility-dir> <feasibility-dir>/pre_delta_09a032c/want_text_repair_oneshot.py --pin-root ~/NeuroGraph-worktrees/z12-want-repair-pin-ae798b9 --target-dir ~/.claude/plugins/neurograph/checkpoints --daemon-script ~/docs/.claude/worktrees/daemon-recall-756-20260930/scripts/cc-ng-daemon.py --scope-min-len 600 --expect-wants 182 --expect-protected 183 --expect-scope 118 --step classify --provisional-approve-all`. The same wrapper `classify_wrap.py` sha256 `bc9a3fe0c38a6d6a244e605248b3ce2a7661f163430445ffb74d8123c342eba5`, the same pin, the same recorded CC checkpoint directory read-only (the tool's own verified copy into the new run directory, as in 008). `--provisional-approve-all` is inert in classify, as in 008.
- **New run directory:** `/home/josh/backups/z12-want-text-repair-20261001T014116Z/` (1.2 GB; created by the tool, nothing of mine written into it). Tool rc 0, 130.13 s, scope events `low 0 high 0 max 0 oom 0 oom_kill 0`, `MemorySwapMax` 0, `MemoryMax` 6,442,450,944.
- **Heads (the tool's P1, in the run record):** pin/stack `c7921b8436fb174c3f70fcf02827f16bb16deff0`; actual pin-worktree HEAD `ae798b94...`; `cc_ng_organism.py` sha256 `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2`; the run record's `tool.sha256` is `cbc38bf4...` (not `bf78defa...`).
- **Measurement files:** the wrapper writes fixed file names, so before the run I copied build-tool-008's four (`classify_wrap_result.json`, `sampler_classify.log`, `classify_stdout.txt`, `classify_stderr.txt`) to `streamed_008/`, and after the run moved this run's outputs to `canonical_008b/` and restored 008's originals to their original names; the restored files' sha256 equal build-tool-008's recorded values (`455e9a67...`, `be945097...`, `c139a0fb...`, `e3b0c442...`). This run's: `classify_wrap_result.json` `547e44d087afbf2c3870cb1a3c31947c9c719ccc9b89d2f73266bd25ba710d2f`, `sampler_classify.log` `6944337f7e332d15532e70a4271d6d35453daffa8e6d5268a79844628319b39e`, `classify_stdout.txt` `a593a21b6edd7c47bba4a4e74a889f7565893ba218a5c39afeb8ebdf26058cf2`, `classify_stderr.txt` (empty) `e3b0c442...`; all under `canonical_008b/`.

## 3. The comparison

**Step 1 - file sha256s (computed from the files on disk in both run directories, not from the tools' own reports; streamed = `.../z12-want-text-repair-20261001T011502Z`, canonical = `.../z12-want-text-repair-20261001T014116Z`):**

| Artifact | streamed sha256 = canonical sha256 | Same? |
|---|---|---|
| `reports/outcome-table.json` | ec999a1fd1f025c299ad1b6fb421e2cbd917b51f2bc9a1519a5ca19fe776f72d | **identical** |
| `reports/histograms.json` | 181515c21028a6de72252fd1f93677c6a0498e44fe92f5f8e2208a54f4249d76 | **identical** |
| `reports/marker-bearing-minted.json` | 7dd13d2a6afc0c191f3e97a71629b55cf70a3381bff862e32e8bc9b2089a35da | **identical** |
| `reports/residual-classes.json` | 0c7bf9eaadff85deb9523db5edd68c8762db6798f25d4a571e1d1b6a5d87d032 | **identical** |
| `reports/repair-list.json` | f7f8e3a063c572abcc000925ed7d667dcde11ba88d634948b0aa822f21f7d8e9 | **identical** |
| `reports/scope-ids.json` | 279d2474394d3a4ed905fe710bc9ad1736751219582a2414e8db272f5b1b144f | **identical** |
| `reports/pre-node-report.json` | a3e2663133543d58e10cd18d88e0bbb4bea854426900109ec464c00816821a9a | **identical** |
| `reports/id-map.json` | 270af25cfb905da9ed68b023fc350c13fa04b3a1ef1b07c6485b3e6d5b009cc6 | **identical** |
| `review/left-list-<UTC>.md` (0600) | 59aacdf7c6b442f0bc05c69ca891908f488fe0ade1ea0654020128f318a07b7d | **identical** (118 entries; the file name carries each run's UTC, the bytes do not) |
| `review/review-excerpts-<UTC>.md` (0600) | 960018c7d8f15ad26d3f560af026ca9c9ef4d2bcfb6d7b68292bfbef0f771191 | **identical** (0 entries) |

All 10 equal; file sizes equal. The review files were compared by hash only; their contents were not printed or read into this return.

**Step 2 - canonical JSON after dropping run-specific fields: no field needed dropping**, because the 8 reports are byte-identical, so no JSON-level difference exists (also verified: `json.dumps(sort_keys=True)` of each pair is equal). The only differences in the whole run directories are in the two bookkeeping files, listed so the claim is checkable (`copy-hashes.json`: 54 fields each, **0 differ**; `run-record.json`: 63 fields each, **7 differ**, all run-specific):

| `run-record.json` field | streamed (008) | canonical (008b) | Why it is run-specific |
|---|---|---|---|
| `/utc` | `20261001T011502Z` | `20261001T014116Z` | the run's own timestamp |
| `/tool/sha256` | `bf78defa...` | `cbc38bf4...` | the independent variable of this check (which tool ran) |
| `/isolation/sys_path_head[1]` | the current tool's directory | the pre-delta tool's directory | the directory the tool file was run from (an absolute path) |
| `/artifacts_sha256/review/left-list-<UTC>.md` (two keys, one per run) | `59aacdf7...` under the 011502Z name | the same `59aacdf7...` under the 014116Z name | the key embeds the run's UTC; **the value is identical** |
| `/artifacts_sha256/review/review-excerpts-<UTC>.md` (two keys) | `960018c7...` under the 011502Z name | the same `960018c7...` under the 014116Z name | same: key differs by UTC, value identical |

**Step 3 - side-by-side counts (from the two outcome tables, histograms, residuals; identical):**

| Quantity | streamed (008) | canonical (008b) |
|---|---|---|
| nodes / wants / protected | 7,253 / 182 / 183 | 7,253 / 182 / 183 |
| scope size (expected 118) | 118 | 118 |
| GENUINE / SEPARATE / OVERLAP / ANOMALY / SOURCE_MISSING | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 |
| **NONE** | **106** | **106** |
| **ANCHOR_FAILED** | **12** | **12** |
| by disposition | unchanged 118 | unchanged 118 |
| by class / outcome | A/NONE 60, A/ANCHOR_FAILED 7, B/NONE 16, C/NONE 30, C/ANCHOR_FAILED 5 | the same five cells, the same numbers |
| reason histogram, 49 S sources (nonzero) | `in_json_string` 1,024; `in_code_span` 9 | the same |
| reason histogram, 86 conversational marker nodes (nonzero) | `in_json_string` 1,219; `in_code_span` 14; `opener_unclosed` 2; `closer_without_opener` 1 | the same |
| residual classes | 47 (`in_json_string` 47) | 47 (`in_json_string` 47) |
| marker-bearing minted / `repair-list` candidates / `scope-ids` / `id-map` pairs | 0 / 0 / 118 / 0 | 0 / 0 / 118 / 0 |
| review entries (left-list / excerpts) / hint marks | 118 / 0 / 0 of 118 | 118 / 0 / 0 of 118 |
| PRE-node line (synapses 138,753) | both Choice Clause wants and the constitutional node: not a PRE-node of a synapse to a mapped id, count 0 each | the same (no mapped ids) |

**First divergence: none** (nothing to localise: no node, field or report differs).

## 4. The two peaks of the canonical path, next to the streamed run's (heap is the ruled floor reading; cache shown too)

| Reading | streamed (008) | **canonical (008b)** |
|---|---|---|
| **heap, `ru_maxrss` (tool + children)** | 959,132 kB = 0.915 GiB | **3,694,064 kB = 3.523 GiB** |
| **heap, sampled anon maximum** | 970,674,176 B = 0.904 GiB | **3,771,338,752 B = 3.512 GiB** (at t = 107.9 s) |
| scope `MemoryPeak` (kernel `memory.peak`, cache-inclusive) | 3,406,495,744 B = 3.173 GiB | 4,716,417,024 B = 4.393 GiB |
| sampled page-cache (`file`) maximum | 3,326,193,664 B (3.098 GiB) | 1,513,783,296 B (1.410 GiB) |
| wall | 128.14 s | 130.13 s |
| scope events | `max 0 oom 0 oom_kill 0` | `max 0 oom 0 oom_kill 0` |

The canonical path's heap is **3.85 times** the streamed path's. Its timeline: copy phase to t ~ 87 s (anon under 50 MiB, as in 008), then the canonical `Graph.restore` (anon 784 MiB at t = 90 s) and the whole-file `SimpleVectorDB.load` of the 1 GB vectors (anon 1.54 GiB at 100 s, 2.64 GiB at 110 s, peak 3.512 GiB) while the Graph stays resident, then the streamed run's same classification tail. By the ruled rule (heap + 2.0 GiB) the canonical path would need **5.52 GiB** (3.523 + 2.0) against the 4.0 GiB interim classify gate, which is set for the streamed classify (0.915 + 2.0 = 2.915): **an observation, not a ruling.** This run was permitted by the gate it was dispatched under (MemAvailable >= 4 GiB, 9.12 GiB available) and fit under the 6 GB cap with no event.

## 5. Source and COPY unchanged

- **Live CC source** (read-only): the tool's `copy_six` sha256s each file before, of the copy, and after, and STOPS on disagreement (rc 0; `source_unchanged: true` in the run record). Afterwards size and inode equal `copy_manifest.json`, and mtime equals what the tool recorded before the copy. The new run directory's copy has sha256 equal to the kept COPY's `sha_before` for all six files (the same bytes as the probes, as the streamed run).
- **Kept COPY** (`.../z12-want-text-repair-feasibility-20260930T183126Z/`): size + inode + mtime equal the manifest and the earlier readings (`vectors.msgpack` mtime_ns `1790793124837064596`, `main.msgpack` `1790793092513541531`); not read.
- Pin worktree HEAD `ae798b94...`, 0 porcelain lines after; tool worktree 0 porcelain lines (this return is the only repo change); the daemon unit stayed `inactive`. Nothing under Syl's directories, the live tract, `~/.bashrc`, any generations directory or any daemon/unit was touched. No `--apply`, `phase2-backup`, rollback, `--josh-go*`, P3(b) or rewrite step.

## 6. What this establishes, and what I did NOT verify

- **Established (on this checkpoint, with these bytes):** the streamed reader and the canonical reader give the pinned parser the same inputs and produce the same reports to the byte. So a reader divergence is excluded as the cause of the zero-candidate result: the pre-delta tool, which never had the streamed reader, also finds 0 of 118.
- **NOT established / not verified:**
  - **Whether the zero-candidate result is the right answer.** Both runs use the same pinned parser (#810, ae798b94) on the same bytes; agreement between two readers says nothing about the parser's judgement (the premise question: why every marker in these 49 sources is classified `in_json_string`). I did not examine the parser, the sources' content, or the plan's premise; reading excerpts is the review flow, not mine.
  - the rewrite step, V11, the T6 would-mint set, and the provisional packet (rewrite-step artifacts; not run, as ordered);
  - the plan-004 "~3.6 GiB" wording item and TURN B-rewrite+V11 (HELD by the ruling);
  - anything on Syl's checkpoint or the VPS.
- **Per branch (A): Phase 1 stops here. I have not concluded the repair, patched nothing, and changed no gate or constant.** I have stopped.
