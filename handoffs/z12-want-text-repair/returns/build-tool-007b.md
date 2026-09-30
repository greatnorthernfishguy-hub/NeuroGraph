```
---- Changelog ----
[2026-09-30] Claude Code (claude-sonnet-5-5, Z12 BUILD worker seat, dispatch #11805, DELTA BUILD) - build-tool-007b.
             Section 1 (item 8, the P437 finding) was written and pushed BEFORE any code (b187ce0). Sections 2-9 are the
             build: a streamed content-subset vectors read REPLACES the whole-file vectors load, and analyze() is
             Graph-free. SYNTHETIC data only; final run 235 passed. NOT self-accepted: a fresh cross-family +
             law-enforcer pair follows.
-------------------
```

# build-tool-007b - DELTA BUILD: streamed content-subset vectors read + (item 8) Graph-free Phase-1 analysis

Lane `z12-s3-restore-bundle-20260929` - Z12 - dispatch #11805 - tool branch `cc-laptop-want-repair-tool-20260930`; base = `09a032c3426baf8307cb968496c65f779894304b` (build-tool-007). Pin worktree `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a` (read-only throughout), pin/stack head `c7921b8436fb174c3f70fcf02827f16bb16deff0`.
Related: [[NeuroGraph]] - [[The Laws]]

## 1. ITEM 8 (Exec P437) - does Phase 1 need a LIVE `Graph`? FINDING, written before any code

**Answer: the classify / id-mapping / report side does NOT need a live `Graph`. Every value `analyze()` takes from the Graph is a DIRECT stored field of `main.msgpack` or a pure function of stored fields. One Phase-1 step DOES still need the canonical `Graph.restore`: V11, the verifier's restore of the OUTPUT, which the plan puts in Phase 1.**

### 1.1 Every value `analyze()` (tool `:1595-1640`) and its callees take FROM the Graph

The canonical restore is `Graph.restore` (pin `neuro_foundation.py:5039`; it already streams the outer map, slicing `synapses` as raw bytes and skipping `he_prediction_window_fired`) then `Graph._deserialize` (pin `:5368`).

| Value the tool takes | Tool file:line | Where it comes from in `Graph._deserialize` (pin) | Stored or derived? | Streamable? |
|---|---|---|---|---|
| `nodes_meta = {nid: n.metadata}` | `:1601` | `Node(metadata=nd.get("metadata", {}))` `:5403`, `self.nodes[nid] = node` `:5412` (key = the `nodes` map key) | **stored** (`nodes[nid].metadata`, default `{}` applied at `:5403`) | yes: decode one node entry at a time, keep `.get("metadata", {})` |
| `existing_ids = set(g.nodes)` | `:1602` (used by `apply_collision_rule` `:1111`) | the `nodes` map keys `:5387-5412` | **stored** (map keys) | yes |
| `counts` (nodes / wants / protected) | `:1622-1625` | `len(g.nodes)`; `kind == "want"` and `is_protected(md)` over `nodes_meta` | **stored** (function of metadata) | yes |
| scope, `derive_scope` | `:999`, `:1607-1609` | function of `nodes_meta` only | **derived from stored metadata** | yes |
| `incident_figures(g, ids)` = (`len(_outgoing[i])`, `len(_incoming[i])`, `len(_node_hyperedges[i])`) | `:1557-1560`, `:1620` | `_outgoing`/`_incoming` rebuilt from the synapse store `:5454-5457` (`pre_node_id`, `post_node_id` of each synapse, keyed by synapse id, `bulk_load_msgpack` `:5451`); `_node_hyperedges` rebuilt from `hyperedges[*].member_nodes` (`set(hd["member_nodes"])` `:5463`) at `:5487` - NOT from `archived_hyperedges` (`:5495-5517` never touches `_node_hyperedges`) | **derived indexes** (plan-004 section 3.2, line 189: "Derived, never written") but each is a count of **stored** fields: synapse entries with `pre_node_id == i`, with `post_node_id == i`, and non-archived hyperedges listing `i` in `member_nodes` | yes: one streamed pass over `synapses` and `hyperedges` counting for a small id set |
| `render_len_before = len(org.render_wants(g).encode())` | `:1621` (`render_wants` pin `cc_ng_organism.py:2285`) | reads only `graph.nodes` items: `node.metadata` and `node.creation_time` (`:2303-2312`); `creation_time=nd.get("creation_time", 0)` `:5410` | **stored** | yes: a graph-like object holding only the want nodes (metadata + stored `creation_time`), in `nodes` map order (the sort is stable, so equal timestamps keep the same order) |
| `synapse_stats(raw_main, ...)` | `:1563`, `:1627` | already a streaming pass over the raw bytes, never the Graph | - | already Graph-free |

**Derived state the tool does NOT need:** restored defaults for the ~15 node fields other than `metadata` / `creation_time` / key (`voltage`, `threshold`, `spike_history` and the rest), the spike deques, the delay buffer, prediction validation (`:5368+`), the native synapse store itself, the `config` merge. Nothing in `analyze()` reads any of them. `is_protected` (tool `:565`) is a mirror of a metadata predicate.

**The one place a "restore semantic" could differ from a raw stream** is the native synapse store's (`ng_tract.SynapseStore`, a Rust extension not in the pin tree) treatment of the synapse map key versus the inner fields. I cannot read that source, so the claim rests on a TEST: the equivalence test builds the OLD result from the canonical `Graph.restore` (real native store) INSIDE the test and asserts the streamed figures equal it on a synthetic graph with several synapses, including a synapse whose endpoints are not nodes (the `setdefault` case `:5456-5457`) and archived/non-archived hyperedges. If the real store normalised something the synthetic world does not exercise, only the real-file run would show it (that is on the "not verified" list).

### 1.2 Which step still needs the canonical restore, and whose phase it is

`build_outputs` (tool `:2545-2546`) runs `pinned.nf.Graph().restore(out_main)` on the OUTPUT - "the canonical restore IS the verifier (V11)" - and V11 (verifier `:1926-1950`) compares per-id `_outgoing/_incoming/_node_hyperedges` figures, dangling ids and the `render_wants` byte length before/after. **By the plan's own text this is a Phase 1 step:** plan-004 (worktree `z12-want-text-repair-20260930` at `d18323e`, `returns/plan-004.md` line 286) section 6.6 "Phase 1 (COPY ...): ... `systemd-run ... MemoryMax=3G` for classify/rewrite **and 6G for V11 (measured restore ~= 3.6 GiB)** ... rewrite to a tmp file; V1-V19", and plan section 7: "V11 canonical restore (per-id `_outgoing/_incoming/_node_hyperedges` figures equal; no dangling id; the rendered wants block's byte length before/after reported)". So V11 is Phase 1 (the rewrite step of TURN B), it is deliberately canonical (plan 7: the restore is the verifier), and I do **not** change it: replacing V11's canonical restore with a stream would remove the very check it exists for. `Graph().restore` is therefore removed from **`analyze()`** (both Phase-1 steps call it; the Phase-2 `--apply` path calls it too on the live bytes), and stays in `build_outputs` (V11), where the `Graph` is the only thing resident besides the small content subset.

### 1.3 CONSEQUENCE, stated plainly (nothing lowered)

- With `analyze()` Graph-free and the vectors streamed, the **classify step** (`--step classify`) no longer constructs a `Graph` or loads the vectors: its memory is the copy/hash streaming, the node-metadata dicts, a streamed content subset, and the existing `synapse_stats` pass (which reads `main.msgpack` into `bytes`, ~230 MB, plus the Unpacker's copy). **[unmeasured - a real peak is a TURN B / probe fact, not claimed here.]**
- The **rewrite step** is still heavy: it holds the raw main bytes plus the rewritten output (`build_outputs` `:2538-2542`, verifier lockstep `:1763`) **and then V11's canonical `Graph.restore` of the output (~3.6 GiB, plan-004's cited figure, not re-measured)**. The heavy pass is therefore reduced to **V11 alone** (the input Graph and the 1 GB vectors load are gone from it), but it is **not** moved out of Phase 1.
- So the dry run does NOT become light end to end. The classify step may become light; the rewrite step keeps one canonical restore. **No pre-flight is lowered:** the 8 GiB heavy floor stands until a MEASURED peak re-derives it (and this turn measures nothing real); the 6 GB cap and `MemorySwapMax=0` are unchanged. If the Chief wants the dry run to be light, the lever left is to split TURN B into the light classify step and a separately-scheduled V11 step; that is a scheduling decision, not a code change I am making.

### 1.4 What this changes for the probes (for the re-dispatch, `probe.py` NOT edited here)

- **P1** (vectors keys + field byte sizes) and **P2** (streamed content subset): still the right probes; they mirror the new vectors reader. Keep.
- **P3** (graph-metadata stream of `main.msgpack`): now the **relevant** probe for the classify step. It should additionally do what the new pass does: one decode per node entry retaining `metadata` (plus `creation_time` for want nodes), then a second streamed pass over `synapses` / `hyperedges` counting incident figures for a small id set. The current P3 already decodes each entry transiently and retains node metadata; it does not do the incident-figure pass, and it does not read the `synapses` value as the tool's `synapse_stats` does (whole `bytes`).
- **P4** (canonical `Graph().restore(main)` under 1G): **no longer a Phase-1 classify probe**, but still the right measurement of **V11** (the output restore is the same call on a same-sized file). Keep it, relabelled "V11 restore", expected OOM under 1G as the finding.
- A probe of the tool's own `synapse_stats` whole-bytes read (~230 MB + copy) would be worth adding to P3 because it is now the largest allocation left in the classify step.

(Tool line numbers in section 1 are at the base `09a032c`; section 2 onward cite the post-change file.)

## 2. Commits, diff stat, the canonical-API answer

**Commits on `cc-laptop-want-repair-tool-20260930`** (base `09a032c3426baf8307cb968496c65f779894304b`, the head when I started), in order:

| Commit | What |
|---|---|
| `b187ce0560af66655caec2d99d75a085f984c18f` | docs only: the item-8 finding (section 1), pushed before any code |
| `504d89276d49bbf0e145030497f5b42f4f89fcf3` | **tests, failing-first**: streamed reader / Graph-free analyze / equivalence / fail-closed / no-embedding / memory shape |
| `6e19687679a67a66f3d5105ef4360a09607ad7e5` | tests: equivalence test writes under the patched backups root (harness fix) |
| `751f191f3662a2e8ce305e0ba52acfe3b7f9fb4d` | **tool**: `analyze()` Graph-free + streamed content-subset read (replaces `load_pair` / the whole-file vdb load) |
| `8007904145501901fd6767913d0bd1dd86dbfdfe` | tool: `Unpacker.read_size` must not exceed `max_buffer_size` (a bug the first full run found on small files) |
| `4e32efde48c88772c6d7ef0d5fb2d995f9aa68c5` | tests: equivalence run directories carry the run-directory prefix (harness fix) |
| `15e6976fd15dfbacec79d96c229913a179ec0035` | tests: streamed render view orders/truncates like the canonical Graph |
| `6d12488ba936826ee1b6daddad99eb3c78307819` | tests: that test's limit assertion counts the header and the "older" line (my assertion was wrong, the tool was right) |

Tests were in their own commits and pushed before the tool change; every commit was pushed before the test run that followed it.

**Diff stat `09a032c..6d12488`** (`git diff --numstat`): `oneshot-tool/want_text_repair_oneshot.py` +132 -27; `tests/test_want_text_repair_oneshot.py` +346 -1; `returns/build-tool-007b.md` +52 (this file, then grown). Nothing else: no pin file, no vendored or protected file, no pre-flight code (the tool has none: `grep -ciE "MemAvailable|loadavg|preflight|MemoryMax|MemorySwap"` over the tool diff = 0, and over the tool = 0; the pre-flight is an operator step).

**Item 1, the canonical API question: NONE EXISTS, so the minimal tool-local reader was written.** In the pin, `universal_ingestor.py` `SimpleVectorDB` (`:349`) has only the in-memory dict methods (`insert :367`, `search :386`, `get :413`, `delete :424`, `count :433`, `all_ids :437`, all over `self.embeddings/content/metadata`, i.e. they need a full load first) and the two-step writer `capture_state :452` / `write_state :490` / `save :529`; its one reader is `load :548` (whole-file `msgpack.unpack` + a float32 copy of every embedding). `grep -n "def iter\|def stream\|Unpacker\|def entries\|def items" universal_ingestor.py` = 0 lines: no iterator, streaming or per-entry API. On the Graph side `Graph.restore` (`neuro_foundation.py:5039`) already streams the outer map but always builds the whole `Graph`; it has no iterator either. So, as the brief allows: the minimal tool-local readers, `load_content_subset` (`:1577`), `stream_graph_nodes` (`:1616`), `stream_incident_figures` (`:1637`), over a shared `_streamed` file-like Unpacker guard (`:1560`), about 100 lines in total.

## 3. What changed in the tool (item 2 and item 4)

- **REPLACED, not added (item 2).** `load_pair` and `incident_figures(g, ids)` are deleted. Grep of the tool for the old call after the change (every line, including comments):
  `grep -n "SimpleVectorDB\|load_pair\|vdb\.load\|vdb\.content\|incident_figures(g"` -> 4 hits, ALL in the `#` changelog header (lines 5, 11, 14 describe the change and quote the old name; line 29 is the original TURN A header sentence that the new entry expressly supersedes). **Zero hits in code or docstrings.** `test_the_old_whole_file_paths_are_gone_from_the_tool_source` asserts that on the non-comment lines and that exactly ONE `.restore(` remains in code: `g2.restore(out_main)` (V11, `:2651`).
- **The `analyze()` reorder (item 4)** (`:1696`): pass G first (`stream_graph_nodes`, then the V11 'before' figures, the render length, the counts), then the streamed vectors pass with the keep set known from pass G (the `source_node` of every node of S), then the unchanged classification and reports. No `Graph` is constructed, so there is no `del g` / `gc.collect()` step left to reorder: the Graph is never resident.
- **`incident_figures` no longer depends on `cand_old`.** It is computed for **S + the three protected ids** BEFORE classification (a superset of the old candidates + watch set). **Why the output is identical:** `A["before_figures"]` is read in exactly one place, V11 (`A["before_figures"].get(o)` for `o` in the mapping, verifier `:1935` at base), and the mapping is a subset of the candidates, which are a subset of S; so every key V11 reads has the same value. It is never serialised into a report (the only other hits are the assignment into `A`). The test asserts the new dict equals the canonical figures over S + watch AND equals the old figures on the old keys.
- **Deliberate, disclosed behaviour differences** (all on inputs the canonical path also fails on, or on order of errors only): (a) the scope check (`not want nodes in the graph`) now runs before the vectors file is opened, where it ran after the load; (b) the streamed reader `skip()`s the embedding and metadata bytes without validating them, where the canonical loader would raise on, for example, a non-string map key inside a metadata dict or a malformed embedding dtype; it still requires every entry to be complete (truncation) and to carry an `embedding` field (as the canonical `entry["embedding"]` does); (c) a top-level value other than `entries` (`version`, `count`) is skipped, not read (the canonical loader ignores both as well, it never checks `count`).

## 4. Fail-closed (item 5)

`_streamed` turns any `OutOfData` / `ValueError` / `KeyError` / `TypeError` from the Unpacker into a `Stop` and raises a `Stop` if the top-level map ends before the end of the file (trailing bytes). `test_a_truncated_or_malformed_vectors_file_fails_closed`: the intact synthetic file reads; truncation at 7 cut points (one byte short, 37 bytes short, a half, a third, inside the header at 30 / 5 / 1 byte) each raises `Stop`; one trailing byte raises; a non-map top-level raises; an entry without `embedding` raises. No cap was raised anywhere; the reader's buffer limit is `file size + 1`, the canonical restore's own convention, never a bigger memory allowance.

## 5. Equivalence: exact coverage and results (item 5)

The OLD path is built INSIDE the test file (`_old_load_pair`, `_old_figures`, `_old_analyze` at `:2848`, a copy of the pre-delta `analyze` over `Graph().restore` + `SimpleVectorDB().load`); the NEW path is the tool's `analyze`. The synthetic world (`build_world(..., extras=True)`, `_add_extras`) is a real `Graph` + `SimpleVectorDB` + the four small files and adds, on top of the existing world, exactly the cases the brief named:

| Case | In the extras world |
|---|---|
| S source with content but NO `WANT]` | `NOMARK` (source `cc:conv::plain`); old detail string `content_has_marker` asserted |
| S source with NO vdb entry at all | `NOCONTENT`; detail `content_present,content_has_marker` asserted |
| empty / odd content | `EMPTYC` (content `""`), a vdb-only entry with empty content |
| non-ASCII text | `NONASCII`: a SEPARATE **candidate** with non-ASCII content and text, carried through classify, mapping, rewrite and V1-V19 |
| every `WANT]`-bearing node, nodes without a marker | existing world + a conversational marker node that is nobody's source (`cc:conv::markeronly`), `cc:conv::plain` |
| vdb-only entries | marker-bearing, plain non-ASCII, empty with rich metadata |
| rich metadata / non-unit vectors | `{"a": {"b": [1, 2, 3]}}`, `{"rich": {"deep": [...]}}`, non-normalised vectors |
| an ARCHIVED hyperedge that lists an S id | moved into `archived_hyperedges` in the checkpoint (the canonical restore does not index it; asserted) |

Assertions (each passes at the final run): the streamed content dict equals `{k: v for old content if k in keep or "WANT]" in v}` in the same key order; `nodes_meta`, node order, `existing_ids`, the render block length (and, in a dedicated test with more wants than `WANT_RENDER_LIMIT`, tied timestamps and text over `WANT_MAX_CHARS`, the render TEXT) equal the canonical Graph's; `stream_incident_figures` equals `_old_figures` for EVERY node plus a non-node id; `analyze` equals `_old_analyze` on `scope`, `scope_derived`, `dropped`, `nodes_meta`, `existing_ids`, `counts`, `render_len_before`, `syn` (the PRE-node report), `s_sources`, `marker_nodes`, `histograms`, `marker_bearing`, `residuals` and all `records` (compared as canonical JSON); `build_reports` for all 7 report artifacts has identical sha256 and identical JSON; then, on both A dicts through the same tool functions: the review files (`write_review_files`: returned sha256s, hint counts AND the bytes of every written file), `prepare_outputs` (the rewritten `main.msgpack` bytes, `out_hashes`, `id_map_sha256`, writer and sidecar stats, written ids, V1-V19 results **all 19 pass in both**, the T6 replay set, the walk counts) are equal, `==` on the whole structures. Equivalence covers the Graph side too (the OLD Graph is built inside the test).

**The tests discriminate (mutation checks, disclosed):** with the tool temporarily broken and reverted by `git checkout` (tree verified clean, 0 porcelain lines, after each): (A) also indexing `archived_hyperedges` -> 3 tests fail (graph stream, analysis equivalence, verifier equivalence); (B) a keep set that ignores the S sources -> 3 fail (reader, analysis equivalence, verifier equivalence); (C) `creation_time` replaced by 0 in the render view -> the first version of the suite did NOT notice (the synthetic wants fit under the render limit), so I added `test_the_streamed_render_view_orders_and_truncates_like_the_canonical_graph`, which fails under (C) and under (D) reversed node order and passes on the real code.

**Two more tests:** `test_analyze_needs_no_live_graph_and_no_vdb_load` (monkeypatches `Graph` and `SimpleVectorDB.load` to raise, then runs `analyze`: passes), and `test_the_streamed_reader_never_decodes_an_embedding_or_metadata` (a tracking `Unpacker` subclass records every `unpack()` result; only `str` values ever come out, so no embedding `bytes` and no metadata `dict` is materialised).

## 6. Failing-then-passing evidence, and every test run disclosed

1. **Failing-first, before any tool change** (commit `504d892`, run with the `-k` of the new tests): **10 failed, 2 passed.** The failures were `AttributeError: ... has no attribute 'load_content_subset'` / `'stream_graph_nodes'` for eight of them; `test_analyze_needs_no_live_graph...` failed with "analyze() built a Graph / loaded the whole vectors file"; `test_the_old_whole_file_paths_are_gone...` failed on `SimpleVectorDB`; the analysis-equivalence test failed on the before-figures set (old tool: candidates + watch, not S + watch). The 2 that passed were the extras-world coverage test (it tests the OLD path) and an existing test selected by the `-k`. One test (review/verifier) failed for a harness reason (run directory outside the backups root) and was fixed in `6e19687` and `4e32efd`; I ran the same selection a second time with `--tb=line -rA` to read the reasons.
2. **First full run after the tool change** (commit `751f191`): **64 failed, 97 passed, 73 errors** - a real bug of mine: `msgpack.Unpacker` rejects `read_size > max_buffer_size`, which every small synthetic file hit. Fixed in `8007904`.
3. Targeted runs of the new tests after the fix: 12 passed, 1 failed (run-directory prefix, test harness) -> fixed -> passed. Mutation runs (section 5) and the render-test runs.
4. **The final run, ONCE, on the clean tree at `6d12488ba936826ee1b6daddad99eb3c78307819`: `235 passed in 80.17s`** (223 at A2c + 12 new). P379 preamble printed: `sys.executable /usr/bin/python3`; `cc_ng_organism.__file__` = the PIN worktree copy, sha256 `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2`; `neuro_foundation`, `universal_ingestor`, `checkpoint_guardian` all resolve under the pin; `ng_lite`, `ng_embed`, `neurograph_rpc`, `cc_ng_host`, `activation_persistence` not loaded; `PYTHONPATH None ; NG_EMBED_* none`. Command: `env -u PYTHONPATH -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 python3 -B -m pytest tests/test_want_text_repair_oneshot.py -s -q -p no:cacheprovider`.

## 7. Memory shape (item 6) - SHAPE CHECK ONLY, synthetic, `tracemalloc` (not `ru_maxrss`)

`test_memory_shape_...` (off-repo: the synthetic files live under pytest's tmp directory, not in the repo) writes canonical-format vectors files of 8,000 and 16,000 entries (768-dimension float32 embeddings, about 800 characters of content, a small metadata dict; 8 and 16 marker-bearing entries) and compares the canonical `SimpleVectorDB().load` with `load_content_subset`, measuring `tracemalloc` peak and, for the new path, the bytes still held afterwards:

| Entries | File | OLD peak (canonical whole-file load) | NEW peak (streamed subset) | NEW bytes retained after | Entries kept |
|---|---|---|---|---|---|
| 8,000 | 31.6 MB | 67.2 MB | 4.25 MB | 0.01 MB (6,694 content bytes) | 8 of 8,000 |
| 16,000 | 63.7 MB | 137.0 MB | 4.25 MB | 0.02 MB (14,018 content bytes) | 16 of 16,000 |

The OLD peak doubles with the file (67.2 -> 137.0 MB, 2.04x); the NEW peak is **constant** (4.25 MB at both sizes, the 1 MiB read buffer plus transient per-entry decode) and what it retains is the size of the keep set. The assertions: new < old/10, new < file/5, old grows more than 1.6x, new grows less than 2.5x (+1 MB). **This is a shape check, not the TURN B peak:** `tracemalloc` counts Python and numpy allocations, not native Unpacker internals beyond what they route through the Python allocator; the real-file peak is not claimed.

## 8. The consequence statement (item 8), with the derivation inputs named - nothing lowered

- **Structure, proven on synthetic data:** `analyze()` (both Phase-1 steps call it, and so does the Phase-2 `--apply` path on the live bytes) needs no live `Graph` and no whole-file vectors load; it is byte-identical to the canonical path on the synthetic world, including the Graph-derived index figures.
- **Phase 1 is NOT yet light end to end.** The classify step's remaining large allocation is the existing `synapse_stats` pass, which reads all of `main.msgpack` into `bytes` and feeds an Unpacker (`:1664`; the file is 230,539,966 bytes per the `stat` in build-tool-007, so about 0.23 GB plus a copy by `_unpacker`, about 0.46 GB: a STATIC estimate, unmeasured, plus the node-metadata dicts and the small content subset). The rewrite step still holds the raw main + output + verifier buffers (~1 GB static estimate, build-tool-006 section 4) and **V11's canonical `Graph.restore` of the output (~3.6 GiB, plan-004's cited figure, not re-measured)**. V11 is a Phase 1 step by the plan's text (section 1.2 above), so the dry run's heavy pass is reduced to that one restore, not removed.
- **Derived pre-flight numbers:** by the derivation rule recorded in build-tool-006 section 5 (required `MemAvailable` = measured peak + desktop swing ~1.1 GB + builder/tool growth ~0.3 GB + slack ~0.6 GB): **no per-pass number can be re-derived, because no real peak exists yet.** The one input I would use for the classify step is a STATIC estimate (~0.5 GB), which would give 0.5 + 1.1 + 0.3 + 0.6 = 2.5 GiB, below the 3 GiB floor that the light hard-capped probes already use; I am not stating 2.5, and I am not proposing a value below the 3 GiB light-probe floor on an estimate. **Unchanged: the 8 GiB heavy pre-flight (the rewrite step and any `Graph.restore`), the 3 GiB floor for the light hard-capped probes, the 6 GB cap, `MemorySwapMax=0`, the `load < 6` and daemon gates.** A classify-only floor is to be re-derived from the measured peak of the streamed classify step (TURN B / a follow-up hard-capped probe on the kept COPY), not from this return.
- **What would make the dry run dispatchable at ordinary free memory (a scheduling decision for the Chief, not a code change I made):** run the classify step alone under its own measured floor, and schedule the rewrite step (V11) separately at the 8 GiB gate.

## 9. What the probes should do now (for the re-dispatch; `probe.py` NOT edited, the COPY and `probe.py` untouched)

- **P1, P2:** unchanged in purpose; better, P2 should call the tool's own `load_content_subset` (read-only import of the tool module from the tool worktree) with the keep set (S sources, or for a probe, every want node's `source_node`: a superset), so the probe measures the shipped code, not a mirror.
- **P3:** becomes the **classify step's graph side**: call `stream_graph_nodes` then `stream_incident_figures` (S + watch) and, separately, the existing `synapse_stats` whole-bytes read, each under the 1 GiB cap: the `synapse_stats` read is the largest allocation left in classify and should be measured on its own line.
- **P4:** no longer a classify probe; keep it as the **V11 restore** measurement (same call on an equal-size file), expected OOM under 1 GiB as the finding. A separate run of `build_outputs`' `read_bytes` + rewrite + verifier working set is not covered by any of P1-P4 and is the next unmeasured piece of the rewrite step.

## 10. What I did NOT verify

- **Anything on the real data.** The real 1 GB `vectors.msgpack`, the real 230 MB `main.msgpack`, and the kept COPY were NOT opened, hashed or statted this turn (the `stat` sizes quoted in section 8 are from build-tool-007). No probe and no `probe.py` ran. No real peak exists.
- **The native synapse store on the real file.** `incident_figures` equivalence rests on the canonical restore of a synthetic graph (real native store, including an archived hyperedge and multi-synapse nodes). I could not read the Rust source of the store (`ng_tract`, not in the pin tree) and did not synthesise a synapse whose endpoint is not a node (the graph API validates endpoints); the real file could still surprise in that corner.
- **The Unpacker skip buffer on the real `main.msgpack`:** `max_buffer_size` is `file size + 1`, so a huge top-level value (for example the ballooning `he_prediction_window_fired` map the canonical restore skips) would be buffered whole while it is `skip()`ped; the real peak of that is exactly what a P3-style probe measures.
- **Unchanged and unmeasured:** `synapse_stats`, `build_outputs`, the rewrite writer and the verifier still read whole-file `bytes`; V11 still runs the canonical restore; Phase 2 paths were not exercised beyond the existing synthetic tests.
- The fresh cross-family + law-enforcer review: not done (Chief's call). I have not self-accepted.

## 11. Pin and housekeeping (recomputed this turn)

Pin worktree `git rev-parse HEAD` = `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a`; `git status --porcelain --ignored` = 0 lines; `git diff --stat HEAD` = empty; `cc_ng_organism.py` sha256 `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2` (re-checked). Pin/stack head `c7921b8436fb174c3f70fcf02827f16bb16deff0`. Tool worktree HEAD at the final test run `6d12488ba936826ee1b6daddad99eb3c78307819` (clean, 0 porcelain, 0 untracked). Tool sha256 at that head `59ed9f827cc8e7b8ea5a71f3b768809c647a0bd7b6f113541d2631c2659a2b56` (was `cbc38bf4a02adfcab6ae05c5796bb20c818103a71c23ab20570835460e42565b`); test file sha256 `82a243c2663a9df5e92e207a701b2b24bd42adb17a52630a27897441b2a3363e` (was `9b89f687764c6ef90fa7df01a34d9c94210b8a170abe73643086fb13901e2ab6`). No real checkpoint, the COPY, Syl's directories, any tract, `~/.bashrc`, or any checkpoint directory was opened, listed or written; nothing applied, merged, deployed or restarted; no raw want text in any pushed file (the synthetic text is invented filler).

I have stopped. A fresh cross-family + law-enforcer pair is next at the Chief's call.
