```
---- Changelog ----
[2026-10-01] Claude Code (claude-sonnet-5-5, Z12 BUILD worker seat, dispatch #12268, TURN B-CLASSIFY) - build-tool-008: ONE run of the tool's
             `--step classify` on the CC checkpoint bytes, under the 6 GB / no-swap scope. It completed (rc 0). Heap peak 0.915 GiB; the
             scope's kernel MemoryPeak 3.173 GiB is page cache from the tool's own copy. RESULT: 0 repair candidates (106 NONE, 12
             ANCHOR_FAILED, every marker skipped as in_json_string). Ids / counts / bytes / hashes only. Not self-accepted.
-------------------
```

# build-tool-008 - TURN B-CLASSIFY (the light classify step, ONE run)

Lane `z12-s3-restore-bundle-20260929` - Z12 - dispatch #12268 (resume of the same thread, Exec P454) - tool branch `cc-laptop-want-repair-tool-20260930`. Related: [[NeuroGraph]] - [[The Laws]]

**Headline.**
1. The classify step ran ONCE to completion under `MemoryMax=6G` / `MemorySwapMax=0`: rc 0, 128 s, no OOM, no cap event, daemon untouched.
2. **Real classify peak: 0.915 GiB heap (`ru_maxrss`), 0.904 GiB sampled anon; the scope's kernel `MemoryPeak` is 3.173 GiB, but that is page cache from the tool's own 1.2 GB copy** (anon was 41 MiB at that moment). So "peak + 2.0 GiB" is **2.92 GiB** on the heap number and **5.17 GiB** on the literal `MemoryPeak` number: which input the rule means is the Chief's ruling (section 4). I changed no gate.
3. **The outcome is not what the plan's candidates assumed: 0 repair candidates.** All 118 nodes of S are `unchanged`: 106 `NONE` (every marker in the span skipped by the pinned parser, reason `in_json_string`) and 12 `ANCHOR_FAILED` (`text_count=2`). `repair-list.json` has 0 candidates, `id-map.json` 0 pairs, 0 review excerpts. This is the pinned function's answer on this checkpoint; I did not verify it against the pre-delta (canonical) path on the real data (section 6).

## 1. Pre-flight (printed first; the gate is the ruled interim 4 GiB)

| Time (UTC) | load1 | MemAvailable | SwapFree | `cc-ng-daemon.service` | daemon process | Result |
|---|---|---|---|---|---|---|
| (dispatch gate) 01:12:06 / 01:13:06 | 5.60 / 3.63 | 9,545,084 / 9,778,184 kB | - | inactive | none | quoted; other-worker-turns 1 |
| 01:14:34 (stand-alone print, `z12_preflight4.sh`) | 2.33 | 9,814,588 kB (9.36 GiB) | 9,029,128 kB | inactive | 0 | PASS |
| **01:14:59 (immediately before the launch)** | **1.74** | **9,777,516 kB (9.32 GiB)** | 9,029,184 kB | inactive | 0 | **PASS** (>= 4 GiB, load < 6, daemon inactive, no daemon process) |

One gate reading before launch; no retry, no re-sample. The daemon-process check matches process name (`python*`/`cc-ng*`) plus the daemon/sidecar names. Also before launch: the pin worktree `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a` clean (0 porcelain lines, `--ignored`), the tool tree clean (0), tool sha256 `bf78defaadd28a91d935d7999ab4f136904aafea296afe3a2c6865e05e6f883c`.

## 2. IMPORTANT deviation from the dispatch wording: the tool cannot take the kept COPY as its target

The dispatch says to run `--step classify` "on the kept COPY ... (size/inode/mtime re-check only)". **The tool does not allow that, by design:** `guard_target` (`want_text_repair_oneshot.py:384-401`, hard refusals with no override, P7) requires `--target-dir` to equal the RECORDED CC checkpoint directory (`RECORDED_CC_CHECKPOINT_DIR` `:145` = `~/.claude/plugins/neurograph/checkpoints`) AND the daemon script's `CHECKPOINT_DIR`; and `stage_classify` always copies the target's six files into the new run directory (`copy_six` `:2633`, sha256 before / copy / after) and analyses that copy. Pointing it at the kept COPY would need the tests' trick of patching the recorded constants, which is bypassing the guard, so I did not. I ran the tool **exactly as it defines the step**, as the original TURN B paragraph (item 3) describes it: target = the recorded CC directory, read-only; the tool made its own verified copy in the new run directory. **The bytes are the same as the kept COPY's:** the run-directory copy's sha256 equals `copy_manifest.json`'s `sha_before` for all six files (section 7), and the live source's size, inode and mtime still equal the manifest. The kept COPY itself was only stat'ed (size + inode + mtime, before and after), not read. Cost: the tool's copy is about 1.2 GB of reads (three passes) and writes, which is what put the page cache in the peak (section 4).

## 3. The run

- **Wrapper (off-repo, does NOT edit or patch the tool):** `classify_wrap.py` sha256 `bc9a3fe0c38a6d6a244e605248b3ce2a7661f163430445ffb74d8123c342eba5`, in `/home/josh/backups/z12-want-text-repair-feasibility-20260930T183126Z/`. It runs the tool as a subprocess INSIDE the scope, samples the scope's cgroup every 0.1 s (so heap and page cache are separable), and after the tool exits reads the kernel `memory.peak` (= the scope's MemoryPeak, before the scope is reaped) and `RUSAGE_CHILDREN` `ru_maxrss`.
- **Command (paths relative to the tool worktree; no secrets):**
  `env -u PYTHONPATH -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 python3 -B classify_wrap.py <feasibility-dir> handoffs/z12-want-text-repair/oneshot-tool/want_text_repair_oneshot.py --pin-root ~/NeuroGraph-worktrees/z12-want-repair-pin-ae798b9 --target-dir ~/.claude/plugins/neurograph/checkpoints --daemon-script ~/docs/.claude/worktrees/daemon-recall-756-20260930/scripts/cc-ng-daemon.py --scope-min-len 600 --expect-wants 182 --expect-protected 183 --expect-scope 118 --step classify --provisional-approve-all`
  (the `--expect-*` values are the ones P3 measured: 182 wants, 183 protected, 118 in scope). The wrapper then ran `python3 -B <tool> <the same arguments>`.
- **`--provisional-approve-all` is inert in `--step classify`:** the tool reads it only in the rewrite step (`:2812`); classify writes no provisional packet (that packet is a rewrite-step artifact). I passed it as the dispatch said; nothing in the classify outputs depends on it, and **no provisional packet exists** from this run.
- **New run directory:** `/home/josh/backups/z12-want-text-repair-20261001T011502Z/` (1.2 GB: `copy/` six files with new inodes, `copy-hashes.json`, `reports/` 8 stamped artifacts, `review/` 2 files mode 0600, `run-record.json`). The directory was created by the tool; I wrote nothing into it. The wrapper's measurement files (sampler log, result JSON, captured stdout/stderr) went to the feasibility directory (off-repo, same backups tree, before the run directory existed), not the run directory: a small departure from "nothing outside the run directory", stated.
- **Heads recorded by the tool's P1 (and in `run-record.json`):** pin/stack `c7921b8436fb174c3f70fcf02827f16bb16deff0`; actual pin-worktree HEAD `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a`; `cc_ng_organism.py` sha256 `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2` (blob `a3aa8a0ddb6a89fe468a9a20beadc5e381624cab`); frozen test-file sha256 `04b1a494e4a5762b1e3f97d07e12c0937a50bf5baac09e5208b2bd9f82a46c53`; tool sha256 `bf78defa...` (the dispatch's), tool worktree HEAD `9efb1ad17126520db2a900474d15d47fb537dc16`. P379 isolation printed: `cc_ng_organism`, `neuro_foundation`, `universal_ingestor`, `checkpoint_guardian` all resolve under the pin worktree; `PYTHONPATH None ; NG_EMBED_* none`.

## 4. The MEASURED peak, which number is which, and the floor re-derivation

| Reading | Value | What it is |
|---|---|---|
| **kernel `memory.peak` = the scope's `MemoryPeak`** (read after the tool exited, before reaping) | **3,406,495,744 B = 3.173 GiB** | cache-INCLUSIVE high-water mark |
| sampled `file` (page cache) maximum | 3,326,193,664 B (3.098 GiB), at t = 47.3 s | the tool's own copy and three sha256 passes over 1.2 GB; anon was **41 MiB** at that sample |
| **`ru_maxrss` (the tool process + git children)** | **959,132 kB = 0.915 GiB** | the heap high-water the pass needs |
| sampled anon maximum (0.1 s) | 970,674,176 B = **0.904 GiB**, at t = 111.5 s | the `synapse_stats` moment (analysis phase) |
| scope events at the end | `low 0 high 0 max 0 oom 0 oom_kill 0` | the cap was never touched; `MemorySwapMax` 0; `MemoryMax` 6,442,450,944 |
| wall time | 128.14 s | copy phase to t ~ 87 s (anon < 50 MiB, cache up to 3.1 GiB), analysis phase 87-128 s (anon 337 -> 495 MiB -> peak 0.904 GiB) |

**It closes the P5 gap.** P5 (build-tool-007c) measured the classify sequence without the classification, reports, review files and `residual_classes`: 0.906 GiB. The full step's heap peak is 0.915 GiB: the tail added **9 MiB** (959,132 - 950,148 kB). The peak is still the `synapse_stats` whole-file read (+0.43 GiB transient), as predicted.

**Re-derivation (the ruled rule: floor = measured peak + 2.0 GiB; if it exceeds 4.0 it RISES), every input named:**
- using the heap peak, **P = 0.915 GiB** (`ru_maxrss`; the process's own need): 0.915 + 2.0 = **2.915 GiB** -> below 4.0 -> the interim 4.0 GiB gate **stands, it does not rise**;
- using the literal scope `MemoryPeak`, **P = 3.173 GiB** (cache-inclusive): 3.173 + 2.0 = **5.173 GiB** -> above 4.0 -> by the rule as worded it would **rise to about 5.2 GiB**.

I report both and **do not choose**: the Chief's wording says "the REAL MemoryPeak", and the dispatch says to say which. My reading, for the Chief's ruling only: the 3.173 GiB occurs at t = 47 s during the tool's streamed copy, when the heap was 41 MiB; it is clean/dirty page cache the kernel reclaims under pressure (as in build-tool-007c, where the same effect pinned `memory.peak` at the cap). The heap number is what the pass needs; the cache number says what the copy touches. If the Chief wants a gate that also covers the copy's cache, 5.2 GiB is the arithmetic; if the gate is meant to cover what the pass holds, 2.9 GiB (so 4.0 stands). The gate is unchanged either way until ruled. The next measurement of the same kind (the rewrite step's first run) will show the same heap-versus-cache split, so the ruling is reusable.

## 5. Results (ids / lengths / counts / hashes / booleans only; no want text)

**Counts (run record and outcome table):** 7,253 nodes; 182 want nodes; 183 protected; scope size **118** (`scope_matches_expected` true; the rule-derived scope equals the list, count 118, `--scope-min-len 600` as the cross-check); 49 distinct S sources.

**Outcome table:** by outcome **NONE 106, ANCHOR_FAILED 12** (GENUINE 0, SEPARATE 0, OVERLAP 0, ANOMALY 0); by disposition **unchanged 118**; by class/outcome `A/NONE 60`, `A/ANCHOR_FAILED 7`, `B/NONE 16`, `C/NONE 30`, `C/ANCHOR_FAILED 5` (class A = 67, B = 16, C = 35); accounting sum 118; `assert_failed` 0; `collision_dropped` 0. Detail of the unchanged nodes: all 106 NONE are `every_marker_in_span_skipped` with reason `in_json_string` (2 to 33 skipped markers per span; two also with `in_code_span` 1-2); all 12 ANCHOR_FAILED are `text_count=2` (the want text occurs twice in its source).

**Reason histograms (11 reasons, reported for both populations):** the 49 S sources: `in_json_string` 1,024, `in_code_span` 9, every other reason 0; hand-review nodes `in_json_string` 49 of 49. All 86 conversational nodes containing a marker: `in_json_string` 1,219, `in_code_span` 14, `closer_without_opener` 1, `opener_unclosed` 2, rest 0; hand-review `in_json_string` 81.

**Other lists:** marker-bearing minted list **0**; residual classes **47** base-minted wants the pinned function does not mint, all `in_json_string`; `repair-list.json` **0 candidates, 0 dropped**; `scope-ids.json` **118** ids (expected 118, matches); `id-map.json` **0** pairs; **would-mint count: not produced by classify** (the T6 replay is a rewrite-step artifact; no value exists from this run).

**Review files (off-repo, mode 0600, referenced by path + sha256):** `review/left-list-20261001T011502Z.md` **118 entries** (every S node, all LEFT/unchanged); `review/review-excerpts-20261001T011502Z.md` **0 entries** (495 bytes, header only: no candidates). **Advisory hint counts: 0 of 118 entries marked**, no term counts.

**THE PRE-NODE REPORT LINE (plan-004 section 4.2-7; synapses total 138,753):**

| Node | Is a PRE-node of a synapse to a mapped id | Count |
|---|---|---|
| `cc:want::3eecfa18710e3b6b` (Choice Clause want) | **no** | 0 |
| `cc:want::7bd0f5fdca6eb404` (Choice Clause want) | **no** | 0 |
| `constitutional::rim::choice_clause` (the constitutional node) | **no** | 0 |

(There are no mapped ids at all - `id-map.json` is empty - so a `pred_weights` remap cannot occur from this classification. Other counts in the same report: `touch_scope` 81,999; `scope_to_any_want` 13,203; `rim_to_scope` 131; `rim_to_mapped` 0.)

**Artifact sha256 (file bytes; each equals the value in `run-record.json`'s `artifacts_sha256`, checked) and the pin-stamp check:**

| Artifact | sha256 |
|---|---|
| `reports/outcome-table.json` | ec999a1fd1f025c299ad1b6fb421e2cbd917b51f2bc9a1519a5ca19fe776f72d |
| `reports/histograms.json` | 181515c21028a6de72252fd1f93677c6a0498e44fe92f5f8e2208a54f4249d76 |
| `reports/marker-bearing-minted.json` | 7dd13d2a6afc0c191f3e97a71629b55cf70a3381bff862e32e8bc9b2089a35da |
| `reports/residual-classes.json` | 0c7bf9eaadff85deb9523db5edd68c8762db6798f25d4a571e1d1b6a5d87d032 |
| `reports/repair-list.json` | f7f8e3a063c572abcc000925ed7d667dcde11ba88d634948b0aa822f21f7d8e9 |
| `reports/scope-ids.json` | 279d2474394d3a4ed905fe710bc9ad1736751219582a2414e8db272f5b1b144f |
| `reports/pre-node-report.json` | a3e2663133543d58e10cd18d88e0bbb4bea854426900109ec464c00816821a9a |
| `reports/id-map.json` | 270af25cfb905da9ed68b023fc350c13fa04b3a1ef1b07c6485b3e6d5b009cc6 |
| `review/left-list-20261001T011502Z.md` | 59aacdf7c6b442f0bc05c69ca891908f488fe0ade1ea0654020128f318a07b7d |
| `review/review-excerpts-20261001T011502Z.md` | 960018c7d8f15ad26d3f560af026ca9c9ef4d2bcfb6d7b68292bfbef0f771191 |
| `run-record.json` | eb81d99e7626f98e4c5b7b3b15f7011dab9fb8575ecfbb3e5b7f887471afbaa5 |
| `copy-hashes.json` | 89bf5ca1684e8570b7d107df0f0b6cc9deb2708948cd33a2f55b8a23a3d6f2e1 |

**Pin-stamp check:** all 8 report artifacts carry a `pin_stamp` equal to the run record's: `branch_head` `c7921b84...`, `cc_ng_organism_blob` `a3aa8a0d...`, `cc_ng_organism_sha256` `8ad0f69e...`, `scrub_version` `scrub-1`, `test_file_sha256` `04b1a494...`: the same values the tool's P1 asserted (the frozen pin tuple). The pin worktree is still HEAD `ae798b94`, 0 porcelain lines after the run.

## 6. The zero-candidate result: what it is, and what I did NOT verify

- **What it is:** the pinned function (#810, the FINAL parser) treats every marker in every one of the 49 source nodes as being inside a JSON string literal or a code span, so it mints nothing in the old spans (NONE), and for 12 nodes the stored want text is not unique in its source (ANCHOR_FAILED). plan-004 anticipated a fall ("SEPARATE <= 102 ... the realized number may fall: a nested legitimate pair that lies in the same JSON string region is masked, so the node ends NONE"; sources are "JSON-escaped tool results"); it did not anticipate 0. Under this result a Phase-2 `--apply` would have nothing to write (the tool's own rule: a zero-write apply STOPs and retires nothing). That is a finding for the Executive and the Chief; I draw no conclusion about the repair.
- **Consistency with the plan's own figures (supporting, not proof):** 182 wants, 183 protected, 118 in scope, 49 distinct sources all match plan-004's counts (V1: 182 wants / 183 protected; 49 S sources).
- **NOT verified:** that the streamed-reader classification equals the pre-delta canonical (`Graph.restore` + `SimpleVectorDB.load`) classification **on the real files**. That equality was proven on synthetic worlds (build-tool-007b/007d, le-035, checker-028) and not on this checkpoint. Because the result is all-NONE, a reader divergence on real data (for instance in how a source's content is decoded) would matter more than usual; an independent check on the COPY, running the pre-delta path and comparing the report sha256s, is an option I did NOT run (it needs the canonical vectors load, a heavier cap than this step, and the Chief's go).
- **Also not verified:** why the sources are JSON-escaped; whether the pinned function's JSON-string masking is intended to swallow these; the rewrite step, V11, the T6 would-mint set and the provisional packet (all rewrite-step artifacts, not run, as ordered); anything on the VPS or Syl's checkpoint.

## 7. Source and COPY unchanged

- **Live CC source** (`~/.claude/plugins/neurograph/checkpoints`, read-only): the tool's `copy_six` sha256s each file before the copy, of the copy, and after, and STOPS if they disagree (rc 0, `source_unchanged: true`). After the run: size, inode and mtime equal what the tool recorded before the copy, and equal `copy_manifest.json` (the same files as at the kept COPY's creation; link counts 2 for `main`/`vectors` as before: the generation hard links, never opened).
- **Run-directory copy** (new inodes `22169606` main, `22169630` vectors, `22169631`, `22169632`, `22169634`, `22169637`, each different from its source inode): **sha256 equals the kept COPY's `sha_before` for all six files** (so the analysed bytes are the probed bytes): main `7e457786...3a77`, vectors `93ed891f...b05e`.
- **Kept COPY** (`.../z12-want-text-repair-feasibility-20260930T183126Z/`): size + inode + mtime re-checked before and after against `copy_manifest.json`; equal; not read.
- Nothing under Syl's directories, the live tract, `~/.bashrc`, any generations directory, or any daemon/unit was touched; the daemon unit stayed `inactive`. No `--apply`, `phase2-backup`, `--step rollback`, `--josh-go*`, P3(b) or rewrite step was run. The repo changed only by this return.

## 8. Measurement artifacts (feasibility directory, off-repo)

`classify_wrap.py` bc9a3fe0c38a6d6a244e605248b3ce2a7661f163430445ffb74d8123c342eba5; `classify_wrap_result.json` 455e9a67b01e556c44200ac619e784652630fe8d3020cb19cdacda620c5962d8; `sampler_classify.log` (1,240 lines) be945097971901c074972dc7b6f0d7abbe2b2d3b0364ebaba3c0007cfa16415d; `classify_stdout.txt` (P379 lines + the tool's result JSON) c139a0fbb1f8316a170f9f757818309b9e774aed13f66f14fa4b30d85d2627a4; `classify_stderr.txt` (empty) e3b0c442...; `z12_preflight4.sh` 5e0018ec679a108765deeff997f4ae5189a13c020a0398d8b7dba942e53f8130.

I have stopped.
