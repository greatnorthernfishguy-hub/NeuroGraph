```
---- Changelog ----
[2026-09-30] Claude Code (claude-sonnet-5-5, Z12 BUILD worker seat, dispatch #12217, TURN B-1 PROBES) - build-tool-007c: P1, P3 (+P3s, P3b),
             P2, P4 RAN, one at a time, each under a hard 1 GiB / no-swap scope, on the FOLDED tool. No probe was OOM-killed: the canonical
             restore (P4, the V11 measurement) peaked at 0.98 GiB, NOT the ~3.6 GiB plan-004 cites. Per-pass pre-flight re-derived from the
             measured peaks; nothing lowered. Ids / counts / bytes only. Not self-accepted.
-------------------
```

# build-tool-007c - TURN B-1 PROBES on the folded tool (DELTA 4)

Lane `z12-s3-restore-bundle-20260929` - Z12 - dispatch #12217 - tool branch `cc-laptop-want-repair-tool-20260930`. Related: [[NeuroGraph]] - [[The Laws]]

**Headline.** Nothing under the 1 GiB cap was OOM-killed. Measured on the CC-laptop checkpoint COPY (`main.msgpack` 230,539,966 B: 7,253 nodes, 138,753 synapses, 517 live + 14 archived hyperedges; `vectors.msgpack` 1,000,876,671 B: 18,502 entries):

| Pass | What ran | Peak that matters (process RSS / anon, NOT page cache) | OOM? |
|---|---|---|---|
| classify-light, one process (the sequence `analyze()` runs) | P5 | **0.906 GiB** `ru_maxrss` (anon max 0.889 GiB) | no |
| V11 canonical `Graph().restore(main.msgpack)` | P4 | **0.992 GiB** `ru_maxrss`; kernel `memory.peak` **0.980 GiB**; never reached the cap (`max` events 0) | **no** |
| rewrite step | **not measured** (no probe covers it) | static sum of measured parts ~2.0 GiB (section 6) | - |

The V11 result contradicts plan-004's "measured restore ~= 3.6 GiB" for this checkpoint (section 5). No honest derivation reaches 8 GiB (section 6). The gates themselves are unchanged pending the Chief's ruling.

## 1. Heads, scripts, what was and was not touched

- Tool worktree HEAD `d05863052aad181c892ca6832b871f5984e6a858` (the le-037 review commit on top of the fold; `oneshot-tool/` and `tests/` byte-identical to code commit `2c31077`): tool sha256 `bf78defaadd28a91d935d7999ab4f136904aafea296afe3a2c6865e05e6f883c` (recorded in every result JSON; equals the dispatch's), test file sha256 `079e95f1d430de4199d170303a42cf298089f3b973d1355ddef5924c752ca1fd`. Tool tree 0 porcelain lines before and after.
- Pin worktree HEAD `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a`, `git status --porcelain --ignored` 0 lines before and after P4 and P5 (the only probes that import the pin); `cc_ng_organism.py` sha256 `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2`. Pin/stack head `c7921b8436fb174c3f70fcf02827f16bb16deff0`.
- **Scripts (all off-repo in `/home/josh/backups/z12-want-text-repair-feasibility-20260930T183126Z/`):** `probe2.py` sha256 **`3653767f50044dce16aebe4de631de277941cf4b265009f271b16736fd593a80`** (new, this turn); `probe.py` sha256 `7e0bceb77c1a61eba0dbdf22fb1e6f75ecbcb7d55622570bc4afb960a62d4037` (the 007 script, kept for the record, NOT used); `copy_six.py` `957788f6...` (007, not re-run); the runner `z12_runprobe.sh` `4cdc432a991026824c748f4261ac2941778ccc00928d1381250c08e17bfcc7ec` and the gate script `z12_preflight.sh` `8cec6a18d1cb1a082dc5946bc0ab78a984456939c0e141c282bf241ffb8721d0` (copied into that directory). Result JSONs, sampler logs and run logs are there too (sha256 list in section 9).
- Each probe: `systemd-run --user --scope -q -p MemoryMax=1G -p MemorySwapMax=0 python3 -B probe2.py <MODE> <dir>`, run under `env -u PYTHONPATH -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1`, one at a time, READ-ONLY on the COPY. P2, P3, P3b and P5 import the TOOL'S OWN functions (`load_content_subset`, `stream_graph_nodes`, `stream_incident_figures`, `synapse_stats`, `derive_scope`, `load_pinned`) from the tool worktree: no mirror. Nothing written in the repo except this return; no Phase 1 step, `oneshot-tool/` and tests untouched; no copy re-made; Syl's directories, the live tract, `~/.bashrc` untouched; no raw want text anywhere (only counts, byte sizes and a sha16 of one entry id).
- **Two probes added to the four** (same cap, same pre-flight, same read-only rules): **P3s** (largest single element per section, section 4) and **P5** (the classify steps in ONE process, because DELTA 4 asks for "P3 + P2 peaks, run in one process as the tool's classify step would"). P3b is the "separate line" for `synapse_stats`. Run order: P1, P3, P3s, P3b, P2, P4, P5.

## 2. Pre-flight, printed before EACH probe (load < 6 by the builder's rule; every reading also < 5, the dispatch's)

| Probe | Time (UTC) | load1 | MemAvailable | SwapFree | daemon unit | daemon process | Result |
|---|---|---|---|---|---|---|---|
| (dispatch gate) | 00:37:13 / 00:38:13 | 2.49 / 2.10 | 9,583,036 / 8,985,716 kB | - | inactive | none | quoted |
| P1 | ~00:41:3x | **printed, but my output filter cut the line off; not recorded** | - | - | - | - | PASS (the runner stops unless every gate holds, and P1 ran) |
| P3 | 00:41:43 | 2.98 | 9,996,412 kB (9.53 GiB) | 9,232,060 kB | inactive | 0 | PASS |
| P3s | 00:41:57 | 2.52 | 10,040,504 kB (9.58 GiB) | 9,232,092 kB | inactive | 0 | PASS |
| P3b | 00:42:10 | 4.22 | 9,950,236 kB (9.49 GiB) | 9,232,092 kB | inactive | 0 | PASS |
| P2 | 00:42:36 | 3.83 | 9,995,896 kB (9.53 GiB) | 9,232,132 kB | inactive | 0 | PASS |
| P4 | 00:42:58 | 4.62 | 9,959,688 kB (9.50 GiB) | 9,232,144 kB | inactive | 0 | PASS |
| P5 | 00:43:27 | 4.16 | 9,886,336 kB (9.43 GiB) | 9,232,152 kB | inactive | 0 | PASS |

Every gate passed; no probe was retried; no cap was raised. The daemon-process check matches on process name (`comm` `python*`/`cc-ng*`) plus the daemon/sidecar names, not on wrapper command text (the self-match I hit in 007). "Alone" is the dispatch's reading (other-worker-turns 0); I cannot see other sessions myself.

## 3. The copy (not re-made)

`/home/josh/backups/z12-want-text-repair-feasibility-20260930T183126Z/`: all six files re-checked by **size + inode + mtime only (no re-hash, as ruled)** against `copy_manifest.json` at the start and the end of the turn: sizes equal, inodes equal (`22167962`, `22166327`, `22165370`, `22167957`, `22167963`, `22167964`), link count 1, `main.msgpack` mtime_ns `1790793092513541531` and `vectors.msgpack` `1790793124837064596` identical at start and end. The probes only read it.

## 4. Results per probe (ids / counts / bytes only)

**What a "peak" means here, stated once.** A probe that READS a file puts that file in the page cache, and the page cache is charged to the scope. So the kernel `memory.peak` of P1, P2 and P5 reads exactly the cap (1,073,741,824) because the cache filled the cap and was reclaimed (`max` events 2,212 / 3,841 / 630, `oom_kill` 0): that is cache, not heap, and it is NOT what the process needs. The numbers that matter are the process's own: `ru_maxrss` and the sampler's anon maximum. `memory.peak` is meaningful only where it stays below the cap (P3, P3b, P4).

| Probe | ru_maxrss | sampled anon max | `memory.peak` (incl. cache) | wall | OOM / `max` events |
|---|---|---|---|---|---|
| P1 vectors sizes | 21,124 kB (0.020 GiB) | 10.8 MB | 1.0 GiB (cache) | 8.1 s | 0 / 2,212 |
| P3 tool's graph side | 339,480 kB (0.324 GiB) | 335.3 MB | 800,960,512 (0.746 GiB) | 4.6 s | 0 / 0 |
| P3s sizes of every section | 21,100 kB | 10.8 MB | 11.7 MB | 2.6 s | 0 / 0 |
| P3b tool's `synapse_stats` | 776,060 kB (0.740 GiB) | 783.9 MB | 786,108,416 (0.732 GiB) | 9.3 s | 0 / 0 |
| P2 tool's `load_content_subset` | 178,984 kB (0.171 GiB) | 172.5 MB | 1.0 GiB (cache) | 13.2 s | 0 / 3,841 |
| P4 canonical restore (V11) | **1,039,960 kB (0.992 GiB)** | 1,045,778,432 (0.974 GiB) | **1,052,557,312 (0.980 GiB)** | 7.5 s | **0 / 0** |
| P5 classify, one process | 950,148 kB (0.906 GiB) | 954,748,928 (0.889 GiB) | 1.0 GiB (cache) | 12.6 s | 0 / 630 |

**P1 (vectors).** 18,502 entries (declared 18,502; top keys `version`, `count`, `entries`; every entry has exactly `content,embedding,metadata`). Field bytes: **content 560,819,198 (56%)**, **metadata 381,686,093 (38%)**, **embedding 56,893,650 (5.7%)** (each embedding 3,075 B), entry-id keys 922,635. So the "1 GB file" is 94% text-and-metadata, 6% vectors; the tool decodes only `content` and skips the 38% metadata and the embeddings.

**P3 (the tool's own graph side).** `stream_graph_nodes` -> 7,253 nodes, 182 want nodes, 183 protected; `derive_scope(.., 600)` -> **118** (the S of the plan); the three protected ids are all present as nodes; `stream_incident_figures` over S + watch -> 121 figures (sums out/in/hyperedge 72,866 / 21,599 / 153). Keep set written ids-only (`want_source_node_ids.json`, sha256 `a3653d61f794513d21e324b7e9ee92ed9fe8e1ffc1672c01ec4d6c84014a934c`): the `source_node` ids of S = **49** distinct (all wants' distinct source ids, a superset: 60). Resident cost: anon 22.2 MB after importing the tool, **322.8 MB after `stream_graph_nodes`** (the retained `nodes_meta`: the 164 MB `nodes` map decodes to a larger Python structure), 326.1 MB after the incident pass. The incident pass skips the whole 164 MB `nodes` value and descends 59 MB of synapses and added only +13.8 MB to `ru_maxrss`.

**P3b (the tool's `synapse_stats`, with `nodes_meta` resident as in `analyze()`).** 138,753 synapses seen (touch S 81,999; S to any want 13,203; rim to S 131). anon 322.8 MB resident -> **553.4 MB after `read_bytes` (+230.6 MB, the whole file)** -> **783.9 MB during `synapse_stats` (+230.5 MB, the Unpacker's feed copy)**; ru_maxrss 0.740 GiB; everything freed afterwards (anon back to 322.8 MB). This is the largest allocation left in classify, as DELTA 3 expected: two whole-file copies, 0.43 GiB.

**P2 (the tool's `load_content_subset`, keep = P3's 49 ids).** **504 entries retained, 76,827,479 bytes (73.3 MiB) of content**, largest retained content 482,991 B. All 49 keep ids are present; **all 504 retained entries contain `WANT]`** (the 49 S sources are among them, none is retained by keep alone), so 504 entries = the `WANT]`-bearing entries of the whole vdb. anon 22.2 MB -> **172.3 MB** after the call (the retained 73 MiB plus decode transients and allocator slack: ru_maxrss 0.170 GiB). **The new entry-id set's cost, measured on the real ids:** 18,502 ids, **2,206,018 bytes of Python objects (2.1 MiB, about 119 B per id)** - negligible beside the 172 MB. (That rebuild ran after the peak snapshot was taken; the snapshot before it is in the JSON.)

**P4 (V11: canonical `Graph().restore(main.msgpack)` from the read-only pin worktree).** Imported `neuro_foundation` (anon 21.9 MB), restored: 7,253 nodes, 138,753 synapses, 517 hyperedges. **Peak: `ru_maxrss` 1,039,960 kB = 0.992 GiB; kernel `memory.peak` 1,052,557,312 B = 0.980 GiB, i.e. 21,184,512 B (20.2 MiB) below the 1 GiB cap; `max` events 0, `oom_kill` 0, swap max 0.** After the restore the Graph holds 747,925,504 B (0.697 GiB) anon. The sampler's anon maximum was 1,045,778,432 B (0.974 GiB). **No OOM: the restore fits under a 1 GiB cap by a 20 MiB margin.**

**P5 (classify in one process, in `analyze()` order, tool's own functions).** `load_pinned` OK (+6.1 MB: the pin imports are small); graph pass + incident figures + `render_wants` (render length 18,647 B) -> anon 344.5 MB; `load_content_subset` -> 493.7 MB; `synapse_stats` -> **954.7 MB sampled anon max, ru_maxrss 950,148 kB (0.906 GiB)**. Survived; the cache reclaim (`max` events 630) shows the file pages were being squeezed, not the heap. **NOT run in P5:** classification (`Classifier` / `parse_wants` over the S sources and the 504 marker nodes), the reports, the review files, `residual_classes`, T6 (section 7).

## 5. Largest single element per streamed section (P1, P3s) and what `skip()` buffers

| Section | Elements | Total bytes | **Largest single element (key + value)** | Notes |
|---|---|---|---|---|
| `main` `nodes` (decoded one entry at a time) | 7,253 | 164,073,722 | **439,598 B** (value 439,547; id key 123) | avg 22.6 KB; 71% of the file |
| `main` `synapses` (decoded one at a time) | 138,753 | 58,976,284 | **591 B** | avg 425 B |
| `main` `hyperedges` (decoded one at a time) | 517 | 1,518,373 | **13,301 B** | avg 2.9 KB |
| `main` `archived_hyperedges` (skipped by the tool) | 14 | 150,332 | 13,301 B | |
| `vectors` entry (decoded only for `content`) | 18,502 | 1,000,876,671 | **976,245 B** (content 482,996 + metadata 490,095 + embedding 3,075) | largest single FIELD anywhere: content **768,582 B**, metadata 490,095 B, embedding 3,075 B |

**Every skipped top-level value of `main.msgpack`** (29 top-level keys; the tool skips all but `nodes`, `synapses`, `hyperedges`): **there is NO single huge leaf.** All are maps or arrays of small elements or scalars. The largest skipped section is `novel_sequence_log`, an array of 356 elements, **4,767,386 B in total, largest element 199,986 B**; then `prediction_outcomes` (array, 1,000 elements, 409,857 B), `recent_spikes` (map of 4,056 small leaves, 353,431 B, avg 87 B), `synapse_confirmation_history` (map of 4,184 small leaves, 220,937 B), `archived_hyperedges` 150,332 B, `delay_buffer` (3 elements, 24,307 B), `reward_history` 40,638 B, `config` 2,450 B (90 small leaves), the rest under 1 KB. **`he_prediction_window_fired` is an EMPTY map (1 byte)** and `active_predictions` is empty: the "tens of millions of slots" case that motivated the buffer worry does not exist in this checkpoint. So the worst-case buffered leaf is bounded by 4.8 MB (x about 2.1 per le-035's synthetic measurement, about 10 MB), and the measured fact agrees: skipping the 164 MB `nodes` map whole in `stream_incident_figures` moved `ru_maxrss` by only 13.8 MB.

## 6. Re-derivation of the per-pass pre-flight (rule (c); arithmetic shown, every input named)

**The rule (as in build-tool-006 section 5, unchanged):** required `MemAvailable` = measured peak P + desktop swing (~1.1 GiB, the Chief's 0.6-1.1 GB upper) + builder / tool growth outside the measured scope (~0.3 GiB) + slack for a swing arriving during the pass (~0.6 GiB). It is a guard against paging the box into a freeze (earlyoom fires only after heavy paging). Units: GiB; "P" is `ru_maxrss` (process RSS, what the pass actually needs; page cache is reclaimable and is not counted).

| Pass | P (input) | + swing | + growth | + slack | **Derived floor** | Status |
|---|---|---|---|---|---|---|
| **classify-light** (P5: pinned import, graph pass, incident, content subset, `synapse_stats`, in one process) | **0.906** (P5 `ru_maxrss` 950,148 kB) | 1.1 | 0.3 | 0.6 | 0.906 + 1.1 + 0.3 + 0.6 = 2.906 -> **3.0 GiB** (rounded up; equals the hard-capped light-probe floor 1.0 + 1.1 + 0.3 + 0.6) | **MEASURED, with one gap**: classification / reports / review files / `residual_classes` were not run (P5). The growth + slack terms (0.9 GiB together) are the allowance for that gap; the first real classify run, under the ruled 6 GB scope, records its own `MemoryPeak` and re-derives. |
| **V11 restore alone** (P4) | **0.992** (P4 `ru_maxrss` 1,039,960 kB; `memory.peak` 0.980) | 1.1 | 0.3 | 0.6 | 0.992 + 1.1 + 0.3 + 0.6 = 2.992 -> **3.0 GiB** | **MEASURED** for the restore on its own. **But the tool never runs V11 alone**: it runs inside the rewrite step (next row). |
| **rewrite step** (`build_outputs` + `Verifier.run`, V11 inside) | **NOT MEASURED.** Static sum of measured parts, below | 1.1 | 0.3 | 0.6 | static: 2.02 + 1.1 + 0.3 + 0.6 = **4.02 GiB** | **NOT MEASURED - a claim, not a measurement. The gate stays 8 GiB.** |

**The rewrite step's static sum (components measured elsewhere in this return; the assembly itself is not measured):**
- the `analyze()` result still resident (`nodes_meta` + content subset + pin modules): P5 anon after the content subset 493,658,112 B = **0.460 GiB**;
- `build_outputs` holds the input bytes (`read_bytes`, `:2538` at base; P3b measured +230.6 MB) = **0.215 GiB**;
- `rewrite_main` (`_rewrite_main_into`) streams the output to the file entry by entry (no output buffer: read in the code), with one feed copy of the input inside `iter_sections` (P3b measured +230.5 MB) = **0.215 GiB** while it runs;
- V11 inside `build_outputs`: `Graph().restore` of the output: P4 transient peak **0.992 GiB** (it includes its own read of the file), after which the Graph stays resident at **0.697 GiB** (`ctx["g2"]` lives through the verifier);
- `Verifier.run` -> `lockstep`: re-reads the output (`raw_b`, 0.215 GiB) and runs two `iter_sections` generators over the two buffers (two feed copies, 0.43 GiB) while `raw_a` (0.215 GiB, counted above) and `g2` (0.697) are resident.
- **Moment A (V11):** 0.460 + 0.215 + 0.992 = 1.667 GiB. **Moment B (lockstep):** 0.460 (A) + 0.215 (raw_a) + 0.697 (g2) + 0.215 (raw_b) + 0.429 (two feed copies) = **2.015 GiB**, the larger. So the static estimate is ~2.0 GiB, and 2.02 + 1.1 + 0.3 + 0.6 = 4.02 GiB.
- **What is NOT measured in this sum:** the real peak of the whole step in one process (allocator slack, a component alive when I assumed it freed), `Verifier` bookkeeping beyond the buffers (V1-V19 dictionaries, `diff_modulo`), `census_set` (an `mmap`, file-backed), `meta_after` / `protected_after` (references into `g2`), the T6 replay (`surface_wants` over the 504 marker nodes' content), and the review/report writers. **What it would need:** a probe or the TURN B-rewrite run itself under a cap above 1 GiB (the static sum alone exceeds 1 GiB, so a 1 GiB probe would be OOM-killed by construction, which is the reason I did not run one): the ruled 6 GB `systemd-run` scope on the COPY, recording `MemoryPeak` (and `ru_maxrss`) from the real step. That needs the Chief's word for a heavier-than-1 GiB measurement.

**Is any floor >= 8 GiB? No.** Every derivation above is below 8 GiB (3.0, 3.0, 4.02). I do NOT propose lowering any gate on this return: the 8 GiB heavy floor stands for the rewrite step (unmeasured), and the classify-light and V11-alone figures are PROPOSED values for the Chief's acceptance, not changes I made. The 6 GB cap, `MemorySwapMax=0`, load < 6 and the daemon gates are unchanged. If the Chief accepts the derived classify-light floor, TURN B-classify's measured gap is the classification term, and the first run measures it.

## 7. What I found that the brief did not predict (flagged, not acted on)

1. **V11's restore is ~1.0 GiB on this checkpoint, not ~3.6 GiB.** plan-004 section 6.6 says "measured restore ~= 3.6 GiB" and the 6G scope was sized to it. P4 restored this laptop's checkpoint (7,253 nodes, 138,753 synapses) under a 1 GiB cap without reaching it. I do not know what checkpoint or code plan-004's figure came from (a different, larger checkpoint such as the 644K-synapse graph mentioned in `Graph.restore`'s comments, or the restore before its "#RAM footprint" change, are the candidates; I did not check either). The plan's number is not reproduced here; it should be corrected or sourced by whoever owns plan-004.
2. **The 1 GiB cap is nearly the whole V11 restore** (20 MiB headroom) and **P5 used 0.89 of the 1 GiB**. A probe cap of 1 GiB therefore cannot measure anything bigger than these two, which is why the rewrite step (about 2 GiB static) needs a different cap to be measured.
3. **`nodes_meta` is the biggest resident item of the new classify (0.30 GiB for 7,253 nodes)** and `synapse_stats`' two whole-file copies the biggest transient (0.43 GiB). Both are levers (a streamed `synapse_stats`; keeping only the metadata keys classify reads) if the Chief wants classify smaller still. Neither is proposed here and I changed nothing.
4. The P3 keep set (49 S-source ids) retains **504** entries (every `WANT]`-bearing vdb entry), 73 MiB: the content subset is not "a handful of entries", it is the whole marker-bearing text. Still 7.7% of the file.

## 8. What I did NOT verify

- Any pass under a cap above 1 GiB: the rewrite step (section 6), the classification and reports of the classify step, and the assembled step as one process.
- P1's pre-flight values (my display cut them off; the gate itself passed).
- That the sampler (0.25 s) caught each true anon peak: `ru_maxrss` and the kernel `memory.peak` are the high-water marks and are what I quote; the sampled anon max is a lower bound.
- Anything about Syl's checkpoint or the VPS; plan-004's 3.6 GiB source (item 1 above).
- That "alone" held: the dispatch's reading, not mine.

## 9. Artifacts (all in the feasibility directory; sha256)

`want_source_node_ids.json` a3653d61f794513d21e324b7e9ee92ed9fe8e1ffc1672c01ec4d6c84014a934c; `result_P1.json` 90b862fec6817205e21c31a1aacc6f13905fbc2ebd5ed04b2fb1e95b404f380d; `result_P2.json` 9959efbadd8b2fd97a5d50065459e564a22958078c02ed268b009abc9811417d; `result_P3.json` 97707ed0ef94b25bc8e7ee7c0d80eba92b8c4ace9b78cfbeaca1a9221228847e; `result_P3b.json` 4c73d3db752f0e032de29f78f6ee9017c9af18384474442c94f1be58444bb82d; `result_P3s.json` 5186a46ce81cd6a24a39dc2c8ee1678ce4869a757c0599aa4b996bd996bc3af9; `result_P4.json` 17e827e4161c772c56c715bccada9ce87febda7fc048a643300d68ef4d475970; `result_P5.json` dc47f6c308c4bfca91406559fde447ae4e059f9b29db860618ea7cb640a8f73d; sampler logs `sampler_P1.log` 185066487975d3fee4d3acabcc76e6d6593ed23b33d37c18bae2ae89a1fb92e9, `sampler_P2.log` 91193dc80b75caf369d56a66047325195337c918237d331ec65229a364378327, `sampler_P3.log` 4568080a41197199fdc63429cb58f4baa290e80af448721b9d0ad43dc98b54c7, `sampler_P3b.log` 4bafd6b0f51b86414e6e60fa0c5733bbe6a61b0f873cde88a068664c82afaffc, `sampler_P3s.log` 8bf9d7f8f83c8780680e3202dbbc2d37566e018689ea4a90be4ab897c89483e1, `sampler_P4.log` 2951252044151977e992aca937c7deec429afbc6b5ede1978a5d6c83644dc4a2, `sampler_P5.log` 6b8ae9e78342c73bc71c2b7cc9a09d6a88245201575fe3aee0f585ed6fe7eaf4; run logs (pre-flight lines) `run_P2.log` 57a19f798bea4c1f50d8fc1e3fb00ad111e553e980a1ea8b35ffc2730ee957c7, `run_P3.log` cc17709e53c4a896fad30dde5920877bd44e724837c05151ff55a9b74ce03a31, `run_P3b.log` 0246acfc4526c30c986e9e3f5db46f12a93625d4dbd84e0c637e18759be2baa7, `run_P3s.log` c61ca07ac08d46fb2babb7521c6765f24b2bb554ecb27a7e93373c4bc7731ec8, `run_P4.log` 806eda374c6c05bb2e7aaa00bd965ccbea69221c8ea57df52a4ec069b9c32985, `run_P5.log` cb425106c73ac480febe6fb625ecd2d655ae56598de5bb1eb769f8f645162da8. (There is no `run_P1.log`: P1 ran before I started teeing logs.) Result JSONs carry every snapshot (anon / file / `memory.peak` / `ru_maxrss` at each phase boundary).

I have stopped.
