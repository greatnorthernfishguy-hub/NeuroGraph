```
---- Changelog ----
[2026-09-30] Claude Code (claude-sonnet-5-5, Z12 BUILD worker seat, dispatch #11737, TURN B-0) - build-tool-006: the read-only memory-feasibility
             measurement. PART DONE: (a) the static code-path analysis is COMPLETE (it needs no load gate); the PROBES were NOT RUN because
             the pre-flight LOAD reading was off. No Phase 1, no tool step, no edit to oneshot-tool/ or the tests. Ids/counts/hashes only.
-------------------
```

# build-tool-006 - TURN B-0 (memory feasibility of Phase 1) - static analysis DONE, probes NOT RUN (load gate off)

Lane `z12-s3-restore-bundle-20260929` - Z12 - dispatch #11737 - tool branch `cc-laptop-want-repair-tool-20260930`.
Related: [[NeuroGraph]] - [[The Laws]] - [[The Choice Clause]]

**Status.** The brief's pre-flight for the light, hard-capped probes is `load < 6`, `cc-ng-daemon.service` inactive, no daemon process, `MemAvailable` >= 3 GiB. The 1-minute **load was 6.77: off**, so, as in #11707, **nothing was run**: no copy, no probe, no `systemd-run` scope, no feasibility directory, no probe script, and no checkpoint file opened, hashed or even stat'ed (nothing exists under `/home/josh/backups/` named `z12-want-text-repair-*`). I did not sample again to wait for a pass (that would be working around the gate). The code-path analysis (a) is pure reading, so it is delivered in full below; everything that needs a measurement is marked **[unmeasured]** and is NOT guessed. **No edit to `oneshot-tool/` or the tests; one code change is PROPOSED (section 4) and I stop.**

## 1. Pre-flight readings, 16:52:25 UTC (one reading, before anything else)

| Reading | Value | Gate (this light probe only) | Result |
|---|---|---|---|
| 1-min load | **6.77** (5 min 4.61, 15 min 2.99; 6 runnable / 723 tasks) | < 6 | **OFF** |
| `MemAvailable` | 6,903,720 kB (6.58 GiB) | >= 3 GiB (1 GiB cap + the desktop's ~1.1 GB swings + the builder's ~0.3 GB + ~0.6 GB slack = 3.0 GiB) | ok |
| `cc-ng-daemon.service` | `inactive` (rc 3) | inactive | ok |
| daemon process | none (the only filter match was my own shell wrapper) | none | ok |
| `SwapFree` (info) | 9,344,504 kB | - | - |

Later, while writing this (informational, not a second pre-flight): `MemAvailable` 6,068,652 kB, load 6.24 / 4.90 / 3.23. cgroup v2 is present (`memory` controller; `memory.peak` is available on this 6.17 kernel) and `systemd-run` is 255, so the probes themselves are feasible once the gate clears.

Heads: pin/stack head (frozen, recorded by P1) `c7921b8436fb174c3f70fcf02827f16bb16deff0`; ACTUAL pin-worktree HEAD `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a` (clean, 0 porcelain lines, `cc_ng_organism.py` sha256 `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2`); tool worktree HEAD `3f62534a0986be962e6c82cad810a574cf619a57` (tool sha256 `cbc38bf4a02adfcab6ae05c5796bb20c818103a71c23ab20570835460e42565b`, unchanged since A2c).

## 2. (a) Static code-path analysis - what Phase 1 needs from `vectors.msgpack`

**What the file is.** `SimpleVectorDB.write_state` (pin `universal_ingestor.py:452-527` (`capture_state` `:452`, `write_state` `:490`)) writes one map `{"version", "count", "entries": {id: {"embedding": <float32 bytes>, "content": <str>, "metadata": <dict>}}}`. The tool never reads `embedding` or `metadata`; it reads `content` (the conversation TEXT of each node - which is where the `[WANT]` markers live).

**How the tool loads it today (`load_pair`, tool `:1547-1554`).** `Graph().restore(main)` then `SimpleVectorDB().load(vectors)`. The canonical `load` (pin `universal_ingestor.py:548-605`) does `msgpack.unpack(f, raw=False)` over the WHOLE ~1 GB file into one Python structure, then for every entry builds `np.frombuffer(...).copy()` and an L2-normalised `float32` array - i.e. the whole file is resident **and** a second full copy of every embedding is created on top of it, before `data` is released. Two structural consequences visible in the code:
1. **Order / sum of peaks.** `analyze` (`:1600`) restores the Graph FIRST (plan-004 measured that restore at ~3.6 GiB - cited, not re-measured here) and only then loads the vectors while the Graph is still resident: peak = Graph + vectors-load-peak, not max of the two.
2. **Retention.** `content = vdb.content` (`:1603`) keeps EVERY conversation text for the whole run (`A["content"]`, `:1633`), then `build_outputs` loads the output Graph (`:2545-2546`, V11) on top of it. `vdb.embeddings = {}` (`:1604`) frees the vectors only after the load peak has already happened.

**Per-step "needs" table** (tool `want_text_repair_oneshot.py` at `3f62534`; pin `cc_ng_organism.py` at `ae798b94`):

| Step | file:line | What it reads from the vectors file | Verdict |
|---|---|---|---|
| Phase-1 copy (`copy_six`) | `:2480-2510`; `sha256_file` `:150` | bytes, streamed in 4 MiB chunks (hash before / copy via `shutil.copyfile` / hash after) | **needs nothing in memory** (streaming) |
| classify A0/A1 source text | `Classifier.classify_node` `:1028` (`self.content.get(src)`), `:1016` (`parse(self.content[src])`) | `content[src]` for the **S source nodes only** (49 distinct per plan-004 4.4 [derived]) | **needs `content` VALUES - only of S sources** |
| marker-node enumeration | `conversational_marker_nodes` `:1182-1184` | filters every conversational node on `"WANT]" in content` | **needs `content` values, used only as a filter** (every node that fails the filter is dropped) |
| histograms / marker-bearing list / residuals | `reason_histograms`, `marker_bearing_minted` (over `marker_nodes`); `residual_classes` `:1258-1283` (`content[nid]`, `content[src]`) | `content` of the marker nodes | **needs `content` values of the marker nodes only** |
| excerpts / review files | `excerpt_anchors` `:1355`, `attach_excerpt_hashes` `:1380-1386`, `write_review_files` `:1443`; `mention_shape_flags(content[...])` `:1617`, `:1643` | `content` of the candidate / LEFT sources | **needs `content` values of S sources and marker nodes** |
| T6 replay | `_minted_by_surface_wants` `:1305-1311` -> pin `surface_wants` `cc_ng_organism.py:2261` (`vector_db.content.get(nid)`, then `"WANT]" not in content -> continue`) | only `.content`; `_FakeVDB` (tool `:982`) has NO `.embeddings` / `.metadata` | **needs `content` of the marker nodes only** (a node without a marker is skipped by the pinned function itself) |
| V9 verbatim | `Verifier.run` `:1910` (`A["content"][source]`) | repaired S sources' `content` | **needs `content` of S sources** |
| V12 re-derivation | `cl.parse(src)` -> `content[src]` | repaired S sources | **needs `content` of S sources** |
| V8 "other files unchanged" | `:1903` (`sha256_file(vectors)` vs the copy hash) | bytes, streamed | **needs nothing in memory** |
| census (`census_set`, `:627-636`; in `build_outputs` `:2534`) | `census_msgpack_file` `:600-611`: a compiled byte-pattern regex over an `mmap` of the file | byte patterns only | **needs nothing deserialized** (mmap pages are file-backed page cache, reclaimable) |
| writer, V1-V7, V10, V11, V13-V19 | operate on `main.msgpack` / the sidecar / the mapping only | nothing | **needs nothing** |
| the pinned parser | `parse_wants` is a pure function of a string | nothing | **needs nothing** |
| `embedding` and `metadata` fields | (grep of the whole tool: no use; `vdb.embeddings = {}` `:1604` discards them at once) | - | **needed by NOTHING in Phase 1** |

**Answer to (a), stated plainly:** Phase 1 does **NOT** need `vectors.msgpack` fully deserialized - it needs none of the embedding arrays and none of the vdb metadata. But it does **NOT** get away with "keys only" either: it needs the `content` **values** (the conversation text), and only those of (i) the S source nodes and (ii) the entries whose content contains `WANT]` - and it needs them **streamed one entry at a time**, not via a whole-file load. The keys alone (what the brief hypothesised) cannot carry the classification: the text is the data.

**Behaviour-equivalence of a filtered content set (the claim a code change would rest on).** Every consumer in the table touches `content` only for S sources or for nodes whose content contains `WANT]`; `surface_wants` and `conversational_marker_nodes` skip every other node. So a dict holding {content of the S sources} union {content of every entry containing `WANT]`} yields byte-identical reports, mapping and verifier results. The one edge: an S source whose content exists but has no `WANT]` is reported by A0 with a `detail` string naming the failed conditions (`content_has_marker` vs `content_present,content_has_marker`); keeping the S sources' content in the dict (not only marker-bearing ones) preserves it exactly.

## 3. What was NOT measured (and why) - the probes of (a) and (b)

The probes were planned and are still wanted; none ran:
- **P1 (a), keys + sizes, under `MemoryMax=1G`:** stream the file with a file-like `msgpack.Unpacker`, read the top-level keys, then each entry id, then the entry's three field names with `up.tell()` deltas around `skip()` - so the per-field BYTE sizes (embedding / content / metadata) and the key count are known without building any value. This answers the one open factual question: **how much of the ~1 GB is `content` text versus embeddings** (if the content dominates, the streamed pass still reads it all but retains almost none of it). **[unmeasured]**
- **P2 (b-ii), the streamed content-subset pass, under `MemoryMax=1G`:** per-entry decode, keep `content` only if the id is in a keep set or the text contains `WANT]`; report the number and bytes retained. **[unmeasured]**
- **P3 (b-i), a graph-only metadata stream of `main.msgpack`, under `MemoryMax=1G`:** file-like `Unpacker`, descend `nodes` / `synapses` / `hyperedges` as `iter_sections` does, keep only node metadata; note the top-level values the tool `skip()`s whole (for example the large `recent_spikes` / `delay_buffer` maps) must fit the Unpacker buffer - the probe would show it. **[unmeasured]**
- **P4 (b-i'), the canonical `Graph().restore(main)` under `MemoryMax=1G`:** expected to be OOM-killed (plan-004 cites ~3.6 GiB); that would be the finding (do not raise the cap). **[unmeasured; the ~3.6 GiB is plan-004's prior figure, not mine]**

Each probe script would record its own cgroup `memory.peak` (and a sampler log so an OOM-kill still leaves a last-seen value) plus `ru_maxrss`, and print only counts. They can run in a few minutes once the load gate clears.

## 4. (b) The proposed split and the minimal code change (PROPOSAL ONLY - nothing edited; Chief decides, a changed tool gets its own delta review)

The design the static analysis supports, in `analyze` (`:1596-1640`) and `load_pair` (`:1547`):
1. **Pass G (graph only):** `Graph().restore(main)` as today; extract everything that needs the live Graph object - `nodes_meta`, `existing_ids`, the scope, the counts, `incident_figures` for **S union the three protected ids** (a superset of the candidate ids, computed BEFORE classification so it no longer depends on `cand_old`), and `render_wants` before; then `del g; gc.collect()`.
2. **Pass V (vectors, streamed):** a new tool function (for example `load_content_subset(path, keep_ids)`) that streams `entries` with a file-like `Unpacker` and retains `content` only for `keep_ids` (the S source ids, known from pass G) and for entries containing `WANT]`. It replaces `SimpleVectorDB().load` in this tool only.
3. Classification, reports, T6, V-items unchanged (they read the subset dict). `build_outputs`'s V11 `Graph().restore(out)` is unchanged and then runs with only a small content subset resident.

**Is a code change required? Yes** if the whole Phase 1 must fit a 6 GB cap: as written, peak = Graph (~3.6 GiB cited) + the canonical vectors load (resident file + float32 copies) cannot be assumed to fit - that addition is exactly what the change removes. **Minimal change:** (i) one new streaming function of about 25 lines; (ii) `load_pair` no longer calls `SimpleVectorDB.load`; (iii) `analyze` reorders so the Graph is released before the vectors pass and computes `incident_figures` for S union watch. **Costs to state:** it is a NEW reader of a canonical format (the TURN A return recorded "the analysis-001 loader - nothing forked" as a LAW 3 choice); a streaming reader is read-only and is verified equal to the canonical loader only by a test on synthetic data (the canonical `SimpleVectorDB.save/load` round-trips the same file format in the existing tests, so a test can assert the subset equals the filtered canonical `content`), and the real 1 GB file would still be only read, never written. The Graph side (`Graph.restore`, ~3.6 GiB cited) is **unchanged and remains the heavy pass**: V11 requires the canonical restore of the output by design (plan 7), so this measurement cannot make Phase 1 light - it can only stop the vectors from adding to it.

**Static estimate of the other memory holders (NOT measured):** `synapse_stats` (`:1563`, called from `analyze` `:1627`) and `build_outputs` (`:2538`) read the whole `main.msgpack` (~230 MB) into `bytes`, and `_unpacker` (`:517-520`) `feed()`s it, which copies it again; `Verifier.lockstep` (`:1763`) holds two such buffers (input and output) plus two unpackers' copies. That is roughly 1 GB of working set for the rewrite/verify pass alone, on top of whatever Graph is resident. A `file`-backed streaming Unpacker would cut it, but that is a second change I do NOT propose here without a measurement.

## 5. (c) The per-pass pre-flight - what can and cannot be derived now

The derivation rule (stated so it is applied, not bent): required `MemAvailable` = measured peak P of the pass + the desktop's observed swing (~1.1 GB upper, from the Chief's 0.6-1.1 GB) + the builder's own growth and the tool's working set outside the measured scope (~0.3 GB) + a slack that covers a swing arriving during the pass (~0.6 GB, the same slack the 3 GiB light-probe floor carries). **It is a guard against paging the box into a freeze (earlyoom fires only after heavy paging) - never lowered for convenience.**

| Pass | P | Derived floor | Status |
|---|---|---|---|
| light probes, hard-capped 1 GiB (the Chief's 3 GiB) | 1.0 GiB (the cap) | 1.0 + 1.1 + 0.3 + 0.6 = **3.0 GiB** | reproduced, unchanged |
| Pass V (streamed content subset) | **[unmeasured]** | cannot be derived without P2 | **no number stated** |
| graph-metadata stream (if ever proposed) | **[unmeasured]** | cannot be derived without P3 | **no number stated** |
| Pass G and V11: canonical `Graph().restore` | ~3.6 GiB is plan-004's prior figure for the restore, **not re-measured**; plus ~0.7 GiB working set (raw `main` + unpacker copy) | 3.6 + 0.7 + 1.1 + 0.3 + 0.6 = **6.3 GiB** from a cited number | **keep the unchanged 8 GiB** for the heavy load: the honest derivation from an UNmeasured input is not a basis for lowering it (the 1.7 GiB between 6.3 and 8 is the allowance for that input being wrong; it is re-derived from a real `MemoryPeak` inside the 6G scope, which TURN B will record). Do not read 6.3 as a proposal to lower the gate. |

If the canonical vectors load stayed in place, the derivation for Pass G + V would be Graph 3.6 + vectors-load-peak (resident file + float32 copies; **[unmeasured]** but by construction at least the file size, i.e. >= ~1 GiB plus the copies) + 1.1 + 0.3 + 0.6 - **clearly >= 8 GiB**, which is the plain reading of (d) for the UNCHANGED tool.

## 6. (d) If the vectors truly must be fully loaded

They do **not** have to be, on the code's own evidence (section 2): no step reads an embedding or vdb metadata, and the content is consumed per source/marker node. The only thing that loads them fully is the canonical `SimpleVectorDB.load` (pin `universal_ingestor.py:548-605`) as called by `load_pair` (`:1553`). With the UNCHANGED tool the full load is unavoidable and TURN B then waits for a naturally quieter moment with `MemAvailable` comfortably above the >= 8 GiB derivation; with the section-4 change it is avoidable.

## 7. What I did NOT verify

Everything that needs a measurement: the size split of `vectors.msgpack` (content vs embeddings), the key count, the real peak of any pass, the canonical `Graph.restore` peak on this file (~3.6 GiB is plan-004's number), whether the top-level `skip()`s of the large `main.msgpack` maps fit a streaming buffer, and the wall time. Also not verified: that the byte-identical-report claim for the content subset holds on the real data (it is a code-reading claim, to be tested on synthetic data in a delta review); the real copy step (no checkpoint file was touched this turn, not even `stat`); whether the load will be below 6 when the probes are re-dispatched.

## 8. What I need

Either a re-dispatch of the four probes once the 1-minute load is below 6 (they are light and hard-capped at 1 GiB; I will write them off-repo under `/home/josh/backups/z12-want-text-repair-feasibility-<UTC>/` with their sha256 and run them one at a time), or a Chief ruling on the section-4 change so the tool is fixed before any measurement of the full pass. I have stopped.
