# checker-028 ROLE A — delta review of build-tool-007b (streamed content-subset vectors reader + Graph-free analyze())

STATUS: INCOMPLETE - review in progress

Seat: checker-028 (cross-family, report_only). Dispatch #11899. Lane `z12-s3-restore-bundle-20260929`. Zone manager Z12 session `52d39aba-db92-4bf2-b3b1-0e4c13f77d8c`. Packet ADDENDUM 5 of `review-packet-want-text-repair-118.md` (docs worktree `/home/josh/docs/.claude/worktrees/daemon-recall-756-20260930`, branch `cc-laptop-daemon-recall-756-20260930`; packet read, not edited). Diff pin `09a032c3426baf8307cb968496c65f779894304b` → `3cd271b2a79657afe5a6feeee7454f04b4e479d3` on the tool and test files. Return `handoffs/z12-want-text-repair/returns/build-tool-007b.md` treated as a claim. Output only this file. Scope = the delta only.

Independence: `le-035-want-repair-delta-stream.md` contents have not been opened. Accidental listing: `ls` of `handoffs/z12-want-text-repair/reviews/` showed the filename present, and `git log --oneline` showed the four le-035 commit subjects (`stub`, `first findings`, `ROLE B … COMPLIANT, PASS-WITH-NOTES`, `final status COMPLETE; independence log (checker-028 absent at completion)`). The file body has not been read.

## Pins and run evidence

- Tool worktree `/home/josh/NeuroGraph-worktrees/z12-want-repair-tool-20260930`, branch `cc-laptop-want-repair-tool-20260930`. Tool and test blobs at HEAD equal `3cd271b` (`git rev-parse HEAD:handoffs/z12-want-text-repair/oneshot-tool/want_text_repair_oneshot.py` = `52a322ba6964bccad50b6ee89c55535e28c3f9c3`; test blob `e48c3a10af6a9abeb787811ccbccaf7dfe9f7235`).
- Tool sha256 `59ed9f827cc8e7b8ea5a71f3b768809c647a0bd7b6f113541d2631c2659a2b56`. Test-file sha256 `82a243c2663a9df5e92e207a701b2b24bd42adb17a52630a27897441b2a3363e`.
- Diff stat `09a032c..3cd271b --` tool + tests: `+159/-28` tool, `+347/-1` tests (numstat 132/27 and 346/1). Commits in order: `b187ce0` finding, `504d892` failing-first tests, `6e19687` harness, `751f191` tool, `8007904` Unpacker fix, then test commits, `3cd271b` return.
- Frozen pin worktree `/home/josh/NeuroGraph-worktrees/z12-want-repair-pin-ae798b9`: `git rev-parse HEAD` = `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a`; `git status --porcelain --ignored` = 0 lines; `cc_ng_organism.py` sha256 `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2`; blob `a3aa8a0ddb6a89fe468a9a20beadc5e381624cab`. Pin/stack head recorded by P1 as `c7921b8436fb174c3f70fcf02827f16bb16deff0`.
- Test file ONCE, from the tool worktree: `env -u PYTHONPATH -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 python3 -B -m pytest tests/test_want_text_repair_oneshot.py -s -q -p no:cacheprovider` → **235 passed in 100.57s**. Parent shell had `PYTHONPATH=/home/josh/NeuroGraph:` and `NG_EMBED_REMOTE` set; the test env cleared both.
- P379 (session-start, no NG import): `sys.executable` `/usr/bin/python3`; parent `sys.path` included `/home/josh/NeuroGraph`; no NG names in `sys.modules`. P379 (test run): pin worktree is `sys.path[0]`; `cc_ng_organism.__file__` = pin copy; `neuro_foundation`, `universal_ingestor`, `checkpoint_guardian` under the pin; `ng_lite`, `ng_embed`, `neurograph_rpc`, `cc_ng_host`, `activation_persistence` not loaded; `PYTHONPATH None`; `NG_EMBED_*` none.
- Own world: `/tmp/checker-028-own-world.py` sha256 `7ffa87c1a772bfa67f1cc181bd289d94b7a57b65e6e529051ca98cd77719e7a1` (after dropping a self-loop the Graph API refuses). Ran under the same env. RESULT PASS. World deleted after the run (`/tmp/checker028-world-*`).

No real checkpoint directory opened, listed or hashed. Live tract unopened. Pin, tool, tests, and plan unedited. No raw want text below.

## A1 Byte-identity / equivalence of old vs new path

**PASS-WITH-NOTES**

The test-only OLD path is `_old_load_pair` / `_old_figures` / `_old_analyze` at `tests/test_want_text_repair_oneshot.py:2834-2895`. Diff against `git show 09a032c:handoffs/z12-want-text-repair/oneshot-tool/want_text_repair_oneshot.py`:

- `_old_load_pair` is the pre-delta `load_pair` (`:1547-1554`): `Graph().restore` + `SimpleVectorDB().load`.
- `_old_figures` is the pre-delta `incident_figures` (`:1557-1560`): `(len(_outgoing), len(_incoming), len(_node_hyperedges))` for ids that are nodes.
- `_old_analyze` follows the pre-delta `analyze` (`:1595-1646`) through derive-scope, classify, collision, excerpt hashes, `cand_old`, figures for `cand_old | watch`, `render_wants`, counts, `synapse_stats`, and the report attachments.

Two test-only deltas, neither of which retunes the compared artefacts to match:

1. The pre-delta missing-scope `Stop` (`09a032c:1607-1609`) is omitted from `_old_analyze`. On every world used for equivalence the scope ids are want nodes, so that `Stop` does not fire.
2. `_old_analyze` also computes `before_figures_scope` (S + the three protected ids). The production old `before_figures` set remains `cand_old | watch`. The extra dict is the comparison handle for the intended reorder.

The NEW path is the tool's `analyze` (`want_text_repair_oneshot.py:1696-1751`): `stream_graph_nodes` then `stream_incident_figures` for S+watch, then `load_content_subset`, then the same classify/report pipeline.

Equivalence tests (worker extras world, `extras=True`): classify report, candidate records, reason histograms, T6, review-file bytes, stamped artefact sha256s, and V1-V19 (all 19 pass on both paths) are asserted identical in `test_the_streamed_analysis_is_identical_to_the_canonical_graph_and_vdb_analysis` (`:2997`) and `test_the_review_files_and_the_verifier_and_t6_are_identical` (`:3016`). Coverage includes S sources with content and no marker (`NOMARK`, detail `content_has_marker`), S sources with no vdb entry (`NOCONTENT`), empty content (`EMPTYC`), non-ASCII SEPARATE candidate (`NONASCII`), marker-bearing non-source (`cc:conv::markeronly`), and an archived hyperedge listing an S id (not indexed).

Own world (seed/shape the extras world lacks): 18 long wants sharing `creation_time=7` (stable-sort tie), an S–S 2-cycle, three parallel synapses on the same endpoints, a live hyperedge whose members include an S want and the rim, an archived hyperedge on a no-marker S want, an S source with content and no marker, and a marker-bearing conversational node that is nobody's source. Graph-free `analyze` matched `_old_analyze` on every compared key; `before_figures` equalled `before_figures_scope`; report sha256s equal; render length 6580 on both; archived hyperedge count 0 on the no-marker S want. `Graph.create_synapse` refused a self-loop (`ValueError: Self-connections not allowed`); that pattern was dropped.

Note: `_old_analyze` is a near-faithful copy with the two test-only deltas above. It was not rewritten to hide a mismatch on the keys the tests compare.

## A2 incident_figures equivalence

**PASS**

Pin `neuro_foundation.py` `_deserialize` (`ae798b94:5368-5517`):

- Nodes: `self.nodes[nid] = node` (`:5412`); `_outgoing/_incoming/_node_hyperedges` start as empty sets per node (`:5413-5415`).
- Synapses: `bulk_load_msgpack` / `bulk_load` (`:5450-5453`), then for each synapse id `sid`, `_outgoing.setdefault(ref.pre_node_id, set()).add(sid)` and `_incoming.setdefault(ref.post_node_id, set()).add(sid)` (`:5454-5457`). Counts are counts of stored synapse ids, keyed by the synapse map key. `setdefault` creates an index slot for an endpoint that is not a node.
- Hyperedges: for each live `hyperedges` entry, `for nid in he.member_nodes: _node_hyperedges.setdefault(nid, set()).add(hid)` (`:5486-5487`). `archived_hyperedges` (`:5491-5517`) is restored into `_archived_hyperedges` and never touches `_node_hyperedges`.

`stream_incident_figures` (`:1637-1661`) matches that: synapse ids accumulated in per-endpoint sets; hyperedges from the `hyperedges` map only; `archived_hyperedges` skipped; result keys restricted to ids that are nodes (`want = {i for i in ids if i in existing_ids}`), which is the same filter as `_old_figures` (`if i in g.nodes`). V11 (`:2033-2055`) reads `A["before_figures"].get(o)` only for `o` in the mapping. The mapping is a subset of the candidates, which are a subset of S, so every key V11 reads is in the new S+watch dict and equals the canonical count.

The new figures are computed for S + the three protected ids before classification (`:1713-1715`), independent of `cand_old`. Own-world and extras-world both showed `N["before_figures"] == O["before_figures_scope"]` and equality on the old keys.

The native-store / non-node-endpoint corner remains synthetic-only: the Graph API validates endpoints, and `create_synapse` refused a self-loop. Equivalence on synthetic `write_checkpoint` still uses the real native store.

## A3 Fail-closed / LAW 7 (raw means complete)

**PASS-WITH-NOTES**

`_streamed` (`:1560-1574`): `max_buffer_size=size+1`, `read_size=min(1<<20, size+1)` (the `8007904` fix). After the map, `up.tell() != size` is a `Stop` (trailing bytes). `Stop` is re-raised; every other `Exception` becomes `Stop("%s: truncated or malformed (%s)" % (what, type(e).__name__))`. Truncated / trailing-byte / non-map / missing-`embedding` vectors files raise in `test_a_truncated_or_malformed_vectors_file_fails_closed` (`:2960-2972`): intact file reads; seven cut points each `Stop`; one trailing byte `Stop`; a list top-level `Stop`; an entry without `embedding` `Stop`.

The comment names OutOfData/ValueError/KeyError/TypeError. The code is `except Exception`. That still fails closed (it raises `Stop`; it does not return a shorter content dict). A programming defect inside the reader (AttributeError, MemoryError) is labelled as a malformed file. It does not swallow a truncated read into a successful smaller set.

Disclosed behaviour differences (return section 3):

- (a) Scope check before the vectors file is opened. Harmless: the same `Stop` as `09a032c:1607-1609`, earlier. A bad scope no longer pays the vectors open.
- (b) `skip()` of embedding and metadata without validating them. Harmless for this tool's Phase-1 contract (Phase 1 never reads those fields). Residual: a vectors file whose embedding/metadata are valid msgpack and invalid as the canonical loader would parse them (`entry["embedding"]` + `np.frombuffer`) will classify on the new path and raise on the old path. Truncation inside those fields still `OutOfData` → `Stop`.
- (c) Top-level keys other than `entries` are skipped. Harmless: canonical `SimpleVectorDB.load` (`universal_ingestor.py:548-605`) uses `data.get("entries", {})` and ignores `version`/`count`. Residual: two `entries` keys in one map would be merged by the streaming loop and last-wins as a whole map under `msgpack.unpack`.

No size limit silently truncates. The buffer cap is `file size + 1`, the restore convention.

`test_the_streamed_reader_never_decodes_an_embedding_or_metadata` (`:2937`): a tracking `Unpacker.unpack` sees only `str` values.

## A4 Old path gone, no shrapnel (LAW 3)

**PASS**

Non-comment tool source: `SimpleVectorDB` 0, `load_pair` 0, `vdb.load` 0, `vdb.content` 0, `incident_figures(g,` 0. Exactly one `.restore(` in code: `g2.restore(out_main)` at `:2651` (V11). `test_the_old_whole_file_paths_are_gone_from_the_tool_source` (`:3055`) asserts the same. `test_analyze_needs_no_live_graph_and_no_vdb_load` (`:3044`) monkeypatches `Graph` and `SimpleVectorDB.load` to raise and still runs `analyze`. The replacement lives under `handoffs/z12-want-text-repair/oneshot-tool/`. Diff vs `09a032c` is that file plus the test file (plus the return, outside this diff).

## A5 V11 untouched and still a GATE

**PASS**

The V11 verifier block is byte-identical across the delta (sha256 prefix `fe955e9698f12701` of the `# V11 canonical restore` … `self.check("V11", ok11, v11)` span at `09a032c:1928-1950` and `3cd271b:2033-2055`). `build_outputs` still does `g2 = pinned.nf.Graph(); g2.restore(out_main)` with the same comment. `git diff 09a032c 3cd271b` on the tool has four hunks: changelog, the `load_pair`/`incident_figures` replacement, the `analyze` docstring, and the `analyze` body reorder. Nothing stubs, weakens, or skips V11. It remains a Phase-1 verifier step of the rewrite (plan-004 6.6/7, as the return's section 1 states). `test_v11_fails_without_a_canonical_restore_of_the_output` is still in the suite and was part of the 235-pass run.

## A6 LAW 4 / LAW 2 (canonical loaders, pin, vendored, pre-flight)

**PASS**

Canonical `SimpleVectorDB.load` (`universal_ingestor.py:548-605`) and `Graph.restore` (`neuro_foundation.py:5039`) are read from the pin, not edited. Pin worktree `git status --porcelain --ignored` empty, HEAD `ae798b94`, organism sha256 and blob as frozen. No vendored file in the diff. Tool has zero matches for `MemAvailable|loadavg|preflight|MemoryMax|MemorySwap|8 GiB|3 GiB|6 GB`. Item 1 (canonical streaming API): pin `SimpleVectorDB` has insert/search/get/delete/count/all_ids/capture_state/write_state/save/load and no iterator; `Graph.restore` streams the outer map and still builds a full Graph. The tool-local reader is the allowed fallback.

## A7 Memory evidence honesty

**PASS**

`test_memory_shape_the_streamed_reader_scales_with_the_keep_set_not_the_file` (`:3094`) is labelled a SHAPE check (`tracemalloc`, not `ru_maxrss`). This run printed:

| Entries | File | OLD peak | NEW peak | retained content | kept |
|---|---|---|---|---|---|
| 8,000 | 31.6 MB | 67.2 MB | 4.25 MB | 0.01 MB (6694 bytes) | 8 |
| 16,000 | 63.7 MB | 137.0 MB | 4.25 MB | 0.02 MB (14018 bytes) | 16 |

Those numbers match the return's table. Assertions: new < old/10, new < file/5, old grows >1.6×, new grows <2.5× + 1 MB. The return states this is not the TURN B peak and does not re-derive a floor. 8 GiB heavy floor, 3 GiB light floor, 6 GB cap, `MemorySwapMax=0` are unchanged (absent from the tool).

## A8 Mutation / discrimination

**PASS-WITH-NOTES**

Read of the committed tests, plus one in-memory mutation on the own world. The worktree tool was not broken and reverted (report_only).

- Archived-hyperedge indexing: `test_the_graph_stream_equals_the_canonical_restore` (`:2984`) asserts `len(g._node_hyperedges[NONASCII]) == 0` and stream figures equal canonical figures over every node. Own world: real nomark figures `(0, 1, 0)`; a wrapper that also indexed `archived_hyperedges` produced `(0, 1, 1)`. The tests fail under that mutation.
- Keep set ignoring S sources: `test_the_streamed_content_reader_equals_the_filtered_canonical_load` (`:2926`) requires `cc:conv::plain` (NOMARK's source, no `WANT]`) to be kept when it is an S source, and `orph:plain` (no marker, not an S source) dropped. A keep set that ignored S sources would fail that test and the analysis-equivalence test (NOMARK would become `SOURCE_MISSING` instead of `content_has_marker`).
- `creation_time` zeroed / reversed node order: `test_the_streamed_render_view_orders_and_truncates_like_the_canonical_graph` (`:3115`) builds more wants than `WANT_RENDER_LIMIT`, scrambled times with many ties, and texts over `WANT_MAX_CHARS`, and asserts the render TEXT equal. The worker's disclosure that the first suite missed (C) until this test was added matches the commit `15e6976` sitting after the tool change.

Note: those mutations are not separate committed "break-the-reader" tests; discrimination is carried by the equivalence assertions. That is enough for the cases named.

## A9 Worker disclosures and not-verified

**PASS** (accept, with the same holes left open)

Accept:

- Nothing on the real 1 GB `vectors.msgpack` / 230 MB `main.msgpack` / kept COPY. This seat did not open, list, or hash those directories.
- Native synapse store on a synapse whose endpoint is not a node: Graph API validates endpoints; this seat also hit `Self-connections not allowed`. Equivalence uses the real native store on synthetic graphs that the API will write.
- Unpacker `skip()` buffer on a real `main.msgpack` (`he_prediction_window_fired` and other large top-level values): `max_buffer_size` is `file size + 1`, so a huge skipped value is still buffered. Unmeasured here.
- `synapse_stats` still reads `main.msgpack` into `bytes` (`:1732`). Classify is Graph-free and vectors-streamed; it is not yet light end-to-end. V11 remains a Phase-1 canonical restore. No pre-flight lowered.

Challenge, small: return section 3's `_streamed` except list is written as the four types; the code is `except Exception` (`:1573`). Fail-closed still holds (A3).

## A10 Independence

**PASS** (pending ROLE B comparison)

Own findings above were drafted without reading `handoffs/z12-want-text-repair/reviews/le-035-want-repair-delta-stream.md`. Accidental listing of the filename and of four git-log subjects is disclosed at the top. After this draft is committed, that file will be read and a comparison appended.

## Overall

**PASS-WITH-NOTES** (ROLE A)

The delta does what ADDENDUM 5 / items 1-8 asked: a tool-local streamed content-subset reader and a Graph-free `analyze()` replace `load_pair` / whole-file `SimpleVectorDB.load` and the live Graph in the classify path; V11 stays canonical and a gate; pin and canonical loaders untouched; byte-identity proven on the extras world and on one independent world; fail-closed on truncated/malformed vectors; old path gone; memory numbers are a shape check; nothing lowered.

## Numbered corrections

None that block the delta. Optional follow-ups (not required to accept the replacement):

1. Narrow `_streamed`'s `except Exception` to the four named types so a reader bug is not relabelled as a malformed file.
2. If the pair wants fail-closed vs canonical embedding/metadata well-formedness, decode those fields enough to raise the same errors `SimpleVectorDB.load` raises, still without retaining them.

## Numbered not-verified

1. Real checkpoint files (explicitly out of scope).
2. Native synapse store behaviour for an endpoint that is not a node, and for a self-loop (Graph API refuses both on the write path).
3. Unpacker skip-buffer peak on a real `main.msgpack`.
4. Re-execution of the worker's failing-first pytest at `504d892` (git order confirms tests-before-tool; this seat ran the suite once at the delta tip).
5. ROLE B (`le-035`) comparison — after this draft commit only.
