```
---- Changelog ----
[2026-09-30] Claude Code (claude-sonnet-5-5, Z12 BUILD worker seat, dispatch #11952, FOLD) - build-tool-007d: the ONE FOLD of the
             delta pair's corrections (le-035 COMPLIANT / PASS-WITH-NOTES, checker-028 PASS-WITH-NOTES): items 1-5 and 7 done,
             item 6 not this turn. SYNTHETIC data only. Final run 244 passed. NOT self-accepted.
-------------------
```

# build-tool-007d - FOLD of the delta pair's corrections (items 1-5 and 7)

Lane `z12-s3-restore-bundle-20260929` - Z12 - dispatch #11952 - tool branch `cc-laptop-want-repair-tool-20260930`; base `3cd271b2a79657afe5a6feeee7454f04b4e479d3` (the 007b return; the pair's seven review commits sit on top of it and touch only `reviews/`). Pin worktree `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a`, pin/stack head `c7921b8436fb174c3f70fcf02827f16bb16deff0`.
Related: [[NeuroGraph]] - [[The Laws]]

I read in full: the FOLD section of `build-want-text-repair-118-delta-stream-reader.md` (docs worktree, not edited), `reviews/le-035-want-repair-delta-stream.md` and `reviews/checker-028-want-repair-delta-stream.md`.

## 1. Commits, diff stat

| Commit | What |
|---|---|
| `e6dc6a4254e929a419674589982397df87b84e50` | **tests first (failing-first)**: duplicate-key tests; the extras world gains the fold's shapes; the OLD path gains the missing-scope `Stop`; the last-wins duplicate test is replaced |
| `0bb45baad332ee57eb8c08ccc178e0094658c66e` | **tool**: repeated map keys are a `Stop`; two stale comments edited in place; the disclosure header entry; the `_streamed` ruling recorded |
| `2c31077b02a7d5a491327dac7648b04e3c7401d4` | tests: incident figures count a non-node endpoint on BOTH sides and a self-loop (kills a mutant my first world let through) |

`git diff --numstat 3cd271b..2c31077` (tool + tests): `oneshot-tool/want_text_repair_oneshot.py` +58 -10; `tests/test_want_text_repair_oneshot.py` +162 -10. Nothing else from me (the pair's two review files are theirs). No pin, vendored, protected, canonical-loader or pre-flight file touched.

## 2. Item 1 - duplicate map keys raise `Stop` (le-035 C1, checker-028 C1)

**What the tool does now** (`_once` `:1597`, `stream_graph_nodes` `:1652`, `stream_incident_figures` `:1678`, `load_content_subset` `:1606`):
- `main.msgpack`: a repeated top-level `nodes` / `synapses` / `hyperedges` key is a `Stop` in BOTH graph readers (every top-level key is checked whether descended or skipped); a repeated node id inside `nodes` is a `Stop` (`stream_graph_nodes`); a repeated hyperedge id inside `hyperedges` is a `Stop` (`stream_incident_figures`).
- `vectors.msgpack`: a repeated top-level `entries` key and a repeated entry id are a `Stop`. The earlier "a later duplicate id wins" behaviour of my 007b reader is REMOVED (it was a last-wins emulation; the ruling is to refuse, never to pick or merge), and the test that asserted it is replaced.
- Messages carry only the three fixed key names and no file text.

**Failing-first (commit `e6dc6a4`, before the tool change):** `6 failed, 13 passed` on the selection of new and equivalence tests. The 6 failures were exactly the duplicate-key tests: `test_a_repeated_node_id_inside_nodes_is_a_stop_never_a_merge`; `test_a_repeated_top_level_graph_key_is_a_stop_in_both_graph_readers[nodes|synapses|hyperedges]` (3); `test_a_repeated_hyperedge_id_is_a_stop_in_the_incident_pass`; `test_a_repeated_entries_key_or_entry_id_in_the_vectors_file_is_a_stop`. **After the tool change (`0bb45ba`): `21 passed`** on the selection. Each test builds the duplicate BYTE BY BYTE (a Python dict cannot emit one), asserts the canonical last-wins view with `msgpack.unpackb` first (so the test documents what the canonical path would have produced) and includes a no-duplicate control (`test_a_clean_minimal_graph_file_reads`, and the single-entry vectors case) so the `Stop` cannot come from some other malformation.

**Costs and what this does NOT cover (stated):**
- `load_content_subset` now keeps a set of every entry id it has seen, so it is O(entries) in ids, no longer only O(keep set). The synthetic shape check (`tracemalloc`, NOT the real peak) printed `new_peak` 4.27 MB at 8,000 entries and 4.53 MB at 16,000 (it was 4.25 MB at both before this fold); the retained bytes after the call are unchanged (0.01 / 0.02 MB). The real vdb's entry count is unknown to me (nothing real was opened), so I cannot state the real addition; it is ids, not embeddings or content.
- NOT covered: a repeated synapse id inside `synapses` (would need a set of every synapse id, hundreds of thousands on the real file; not asked for and not free), ids inside `archived_hyperedges` (skipped wholesale, never indexed by the canonical restore), and repeated top-level keys other than the four named. If the Chief wants the synapse-id check it is a separate, measured decision.

## 3. Item 2 - the equivalence-world gaps, as tests (le-035 C3, checker-028 C4)

Built into the extras world (`build_world(..., extras=True)`, `_add_extras`) and compared old-path vs new-path by the existing equivalence tests, **including all 19 verifier checks, which still pass on both paths**:

| Gap | How it is in the world |
|---|---|
| a `hyperedges` entry whose STORED `is_archived` is true, still in `hyperedges` | a fourth hyperedge (members: the `GEN` want and `n1`) with `is_archived` set true after the capture; the canonical restore indexes it (`GEN` has 1 hyperedge) |
| a node in two or more non-archived hyperedges | `SEP1` is in `he` and a new `he3` (figure 2); `NONASCII` is in one live (`he3`) and one archived hyperedge (figure 1, so the archived one is proven not to count) |
| a synapse with a non-node endpoint and a self-loop | byte surgery on the native synapse sub-map (`msgpack.unpackb` of `cap["synapses"]`, two entries added with fresh ids and `synapse_id`, re-packed; round trip of the unmodified map is byte-exact, checked): a ghost-PRE synapse `ghost:not-a-node -> SEP1` and a self-loop on `GEN`; asserted on the canonical restore (`"ghost:not-a-node" in g._outgoing`, a synapse with `pre == post`) |
| two S wants sharing ONE source | `SHARED1` / `SHARED2`, source `cc:conv::shared`, both markers in one content (asserted GENUINE with the same `source_node`) |
| `analyze(frozen_scope=[missing])` raises the same `Stop` before the vectors file is opened, and the OLD path's message equals it | the test's `_old_analyze` now carries the pre-delta missing-scope check (copied from `09a032c:1607-1609`); `test_analyze_a_missing_scope_id_stops_with_the_old_paths_message_before_the_vectors_file_is_opened` runs an absent id and a present-but-not-a-want id (`n1`), asserts `str(new) == str(old)` with the text "not want nodes in the graph", and patches `load_content_subset` to raise `AssertionError` so any earlier open fails the test |

**A limit of the verifier world:** a synapse from a REAL node to a non-node POST endpoint cannot live there (V11's dangling sweep would fail on both paths, which is correct behaviour). So the outgoing-side ghost and self-loops on a small graph are a separate direct test, `test_the_incident_figures_count_a_non_node_endpoint_on_both_sides_and_a_self_loop` (canonical restore vs `stream_incident_figures`, values `a: (3, 2, 0)`, `b: (1, 2, 0)`).

**These were coverage tests, not failing-first:** run at `e6dc6a4` against the UNCHANGED tool they PASSED - no tool defect was found; the streamed figures and every equivalence key already agreed on the new shapes. So I proved they DISCRIMINATE by mutation instead (tool edited uncommitted, `git checkout` after each, tree verified clean, 0 porcelain lines, at the end):

| Mutant | Result |
|---|---|
| M5 - skip a `hyperedges` entry whose stored `is_archived` is true | 2 failed (graph-stream, analysis equivalence) |
| M6b - ignore a ghost PRE endpoint on the incoming side | 3 failed (graph-stream, analysis equivalence, verifier equivalence) |
| M9 - skip self-loops | 2 failed |
| M8 - open the vectors file before the scope check | 1 failed (the missing-scope test) |
| M6 - ignore a ghost POST endpoint (outgoing side) | **survived** the verifier world (for the reason above); the new direct test fails it (`1 failed`), passes on the real code |

## 4. Item 3 - DISCLOSURE, and the Phase-2 apply-path answer

**Disclosed** in this return and as a forward entry in the tool header (`# [2026-09-30] ... dispatch #11952`, lines 3-15): since #11805, Phase-1 `analyze()` no longer canonical-validates the INPUT. `main.msgpack` and `vectors.msgpack` are streamed with `skip()` of the embedding, vdb metadata and every unread top-level value (no validation), and with `strict_map_key=False`. A file the canonical `Graph.restore` / `SimpleVectorDB.load` would REJECT (le-035 measured 88 of 1,200 mutated `main.msgpack` files and 58 of 6,000 mutated vectors files that the canonical path rejects and the streamed path accepts) can be classified. Not a change to a Phase-1 output on a canonically-written checkpoint.

**Which later step proves canonical loadability of the INPUT? Read from the code: NONE in the tool.** The only canonical restore left in code is `g2.restore(out_main)` (`:2699`, V11 in `build_outputs`), and it restores the OUTPUT of a rewrite of the input. The Phase-2 path:
- `stage_apply` (`:3004`): `analyze(pinned, tdir, ..., frozen_scope=scope, full_reports=False)` at `:3043` - "the LIVE bytes" - now STREAMS the live input (before #11805 this same call did `Graph().restore` + `SimpleVectorDB().load` of the live files, so **Phase 2 used to canonical-load the live input and no longer does**); then `prepare_outputs(pinned, A, approvals, in_dir=tdir, ...)` at `:3053` -> `build_outputs` -> V11 restores the STAGED output; the post-apply check (`live_ok`, `:3094`) is sha256 equality plus the old-id census, not a restore; `gate_p3(pinned, run_pinned_tests=True)` runs the pinned PARSER tests, not a data restore.
- `vectors.msgpack` is not rewritten in Phase 2 (hash-equal, V8), so nothing in the tool canonical-loads it after #11805.
- **Bounding argument (le-035), for the Chief's weighing, not my claim:** the rewriter copies every untouched value as a raw slice, so a corruption in a region the canonical restore reads passes into the OUTPUT and V11 rejects it; the exposure is regions the restore skips or never reads (for example the skipped `he_prediction_window_fired`) and the whole vectors file. The first canonical load of the live files after an apply is the daemon's own at the S4 start, outside this tool (which waits on gate P10).
- **Decision for the Chief, not built here:** whether to reinstate a canonical restore of the live INPUT at a Phase-2 step (for example once in `phase2-backup`, on the files it is about to hash) and whether the dry run should add one canonical `SimpleVectorDB.load` of the COPY, or to accept the bounded exposure above. Either is a separate change with a memory cost (a full canonical restore is the ~3.6 GiB pass). I changed nothing for this.

## 5. Item 4 - stale comments edited IN PLACE (le-035 C4, checker-028 C5)

| Where | Before | After |
|---|---|---|
| section banner above the readers (was `:1556`, now `:1571`) | `# the analysis stage: the analysis-001 loader (canonical readers) -> classification -> reports` | `# the analysis stage: tool-local STREAMED readers (pass G: main.msgpack; pass V: the vectors content subset; no Graph, no whole-file load, no canonical validation of the input - see the #11952 header entry) -> classification -> reports` |
| the original TURN A header "How" sentence (was `:28-30`) | `The checkpoint is read with the canonical readers (Graph.restore + SimpleVectorDB.load = the analysis-001 loader, analyze_pair.py:84-86; nothing forked) and written value-granularly ...` | `[Edited in place, #11952: this sentence originally said the checkpoint is read with Graph.restore + SimpleVectorDB.load, the analysis-001 loader; since #11805 analyze() READS it with the tool-local streamed readers stream_graph_nodes / stream_incident_figures / load_content_subset - no Graph, no whole-file vectors load, no canonical validation of the INPUT - and the canonical Graph.restore is used only by V11.] The checkpoint is written value-granularly ...` |
| the #11805 header entry's "SUPERSEDES the How sentence below" line | said the sentence below was superseded | now says the sentence below "was edited in place by #11952 to say so" (one truth, no pointer to a sentence that no longer says the old thing) |

Same lines then mention the old names only as history inside `#` changelog lines; the code and docstrings still have zero old-path hits (`test_the_old_whole_file_paths_are_gone_from_the_tool_source` passes).

## 6. Item 5 - `_streamed`'s `except Exception` stays BROAD (Z12 ruling)

The two seats differed: checker-028 C3 would narrow it to the four named types (OutOfData, ValueError, KeyError, TypeError) so a reader bug is not relabelled a malformed file; le-035 read the broad form as fail-closed and acceptable. **RULED by the Z12 zone manager in dispatch #11952: KEEP it broad; do NOT narrow.** Why, as ruled: both seats agree it is fail-closed (it raises `Stop`, it never returns a shorter successful result), and the `Stop` text names the exception type (`truncated or malformed (AttributeError)`), so an operator can tell a reader bug from a malformed file. Recorded in the tool at the `except` line itself (a comment edit only; the code is unchanged) and in the header entry. The earlier in-line comment that listed four types as the set "that means not the file we expect" was the mismatch checker-028 saw; it now says what the code does.

## 7. Item 7 - re-stamp, tests, pin

- **Tool sha256 at `2c31077`: `bf78defaadd28a91d935d7999ab4f136904aafea296afe3a2c6865e05e6f883c`** (was `59ed9f827cc8e7b8ea5a71f3b768809c647a0bd7b6f113541d2631c2659a2b56` at `3cd271b`; `cbc38bf4...` before #11805). **Test file sha256 `079e95f1d430de4199d170303a42cf298089f3b973d1355ddef5924c752ca1fd`** (was `82a243c2...`).
- **Every artifact stamp binds to the new hash:** `tool_sha256()` (`:2579`) is computed from the running file (`sha256_file(os.path.realpath(__file__))`), not a stored constant, and is written into the run record, the post-apply receipt and the RETIRED refusal; there is no constant to update and no earlier real run exists whose stamp could be stale (no real Phase 1 has ever run).
- **Untouched, checked by hash:** the V11 verifier block (`# V11 canonical restore` through `self.check("V11", ...)`) and `build_outputs` are byte-identical between `3cd271b` and `2c31077` (sha256 prefixes `f05eece279e45bb5` and `c3f73a73ae4b6371` at both); `grep -ciE "MemAvailable|loadavg|preflight|MemoryMax|MemorySwap"` over the tool is 0 and over the tool diff is 0 (no pre-flight code exists in the tool and none was touched); exactly one `.restore(` in code, V11's at `:2699`; canonical `Graph.restore` and `SimpleVectorDB.load` are not edited or wrapped.
- **Pin:** `git rev-parse HEAD` `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a`; `git status --porcelain --ignored` 0 lines; `cc_ng_organism.py` sha256 `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2`.
- **Test runs, all disclosed:** (1) failing-first at `e6dc6a4`, selection of new and equivalence tests: 6 failed, 13 passed (section 2). (2) After the tool change `0bb45ba`, same selection plus memory-shape, old-path-gone and no-graph tests: 21 passed. (3) Mutation runs (section 3), a targeted run of the added direct figure test (one wrong expected value of mine, `a: (3, 3, 0)`, corrected to the canonical `(3, 2, 0)` that I had mis-added by hand; the assertion that matters is stream == canonical), and its mutation run. (4) **The final run, ONCE, clean tree at `2c31077b02a7d5a491327dac7648b04e3c7401d4`: `244 passed in 117.17s`** (235 at 007b, minus the replaced last-wins duplicate test, plus 10 new: the clean control, repeated node id, 3 repeated top-level keys, repeated hyperedge id, the vectors duplicates, the missing-scope Stop, the extras-world shapes, the both-sides ghost/self-loop figures). P379 printed: `sys.executable /usr/bin/python3`; `cc_ng_organism.__file__` = the PIN copy, sha256 `8ad0f69e...a5e2`; `neuro_foundation`, `universal_ingestor`, `checkpoint_guardian` under the pin; `ng_lite`, `ng_embed`, `neurograph_rpc`, `cc_ng_host`, `activation_persistence` not loaded; `PYTHONPATH None ; NG_EMBED_* none`. Command: `env -u PYTHONPATH -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 python3 -B -m pytest tests/test_want_text_repair_oneshot.py -s -q -p no:cacheprovider`.

## 8. What I did NOT verify

- **Anything on real data.** No checkpoint file, directory or the kept COPY was opened, listed, hashed or statted; `probe.py` did not run; no real peak exists. The added id-set cost on the real vdb is unmeasured (section 2).
- **Item 6** (the largest-single-element requirement for the probes, le-035 C5 / checker-028 C6) is not in this turn by instruction; it goes into the 007c probes brief.
- **The native synapse store on a real file**; the pure-Python msgpack fallback; a repeated synapse id (not covered, section 2); whether the bounded-exposure argument in section 4 holds on the real file.
- I did not run the two seats' own scripts; their results are quoted as theirs. Not self-accepted: a tiny re-look is the Chief's call.

I have stopped.
