STATUS: COMPLETE

# le-037 — LIGHT RE-LOOK at the FOLD (build-tool-007d), ROLE B law enforcer

Zone manager Z12, lane z12-s3-restore-bundle-20260929, dispatch #11984. Packet: review-packet-want-text-repair-118.md ADDENDUM 6 (read in the docs worktree, not edited).
Reviewer agent file `/home/josh/.claude/agents/neurograph-law-enforcer.md`, sha256 `6daf1621b844b9b72d567b329f2c9f40ca0b4516744608c125147e601c4acf23` — VERIFIED equal to the expected value before any other step.
Tool worktree head at start `76b75fa35cef09b58c9293d1c85fd455f957542d` (`git pull --rebase`: already up to date). Reviewed diff `3cd271b2a79657afe5a6feeee7454f04b4e479d3..2c31077b02a7d5a491327dac7648b04e3c7401d4`, tool + tests only (the only non-review files changed in that range: those two). Return `returns/build-tool-007d.md` treated as a CLAIM. le-035 and checker-028 read (allowed: a re-look). Report only: nothing in the tool, tests, pin or plan edited. SYNTHETIC data under /tmp/le037 only; no real checkpoint, backup directory or the kept COPY opened, listed or hashed; no raw want text.

## COMPLIANCE VERDICT: COMPLIANT (no Law violation; three documentation corrections, none blocking)

| # | Check | Verdict |
|---|---|---|
| 1 | Duplicate keys fire on real duplicates, never on clean files; byte-identity holds; synapse-id / archived-id gap | COMPLIANT (gap ruled: NOT a LAW 7 hole; see 1.4) |
| 2 | Extras-world additions discriminate; OLD-path missing-scope Stop faithful | COMPLIANT |
| 3 | V11 untouched and still a GATE; one `.restore(`; old-path grep; no pre-flight; pin clean | COMPLIANT |
| 4 | Comments edited in place; `_streamed` `except` unchanged code | COMPLIANT |
| 5 | Phase-2 finding confirmed, with one sharpening and placement advice | CONFIRMED (with sharpening, corrections C2 and C3) |
| 6 | Re-stamp honesty | COMPLIANT |
| 7 | Not verified | listed below |

Environment of every run (Exec P379): interpreter `/usr/bin/python3`; tool `handoffs/z12-want-text-repair/oneshot-tool/want_text_repair_oneshot.py` (worktree path above); pin `/home/josh/NeuroGraph-worktrees/z12-want-repair-pin-ae798b9`, HEAD `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a`; `cc_ng_organism`, `neuro_foundation`, `universal_ingestor`, `checkpoint_guardian` all resolved under the pin; `ng_lite`, `ng_embed`, `neurograph_rpc`, `cc_ng_host`, `activation_persistence` NOT loaded; `PYTHONPATH None`, no `NG_EMBED_*` names; `env -u PYTHONPATH -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1`. Pin `cc_ng_organism.py` sha256 `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2`; pin `git status --porcelain --ignored`: 0 lines, before and after my runs.

## 1. Duplicate keys — COMPLIANT

Code read: `_once` `:1597`, `load_content_subset` `:1606`, `stream_graph_nodes` `:1652`, `stream_incident_figures` `:1678`.
My own scratch (`/tmp/le037/scratch_le037.py` sha256 `97c282e1acd1868f2339469d45271d29d7f2abb6d7c21269d9bdd2c6ba580d89`; `scratch2_le037.py` sha256 `6951feb591a7772d647357830fad1d17e8c882b1802660e5ab329b9309a26caf`), built on the test module's `build_world` + its OLD-path helpers, with byte-level surgery on copies. The "canonical" column is the pin's `Graph.restore` + `SimpleVectorDB.load` on the same bytes, run by me.

| Case (repeat placed at the END of its map; none is a shape the worker tested except where marked) | canonical | tool `analyze()` |
|---|---|---|
| D1 repeated NON-want node id `fill:007`, different metadata | loads (last-wins) | **Stop** "a node id is repeated inside nodes" |
| D2 repeated S-want node id, changed `want_text` | loads | **Stop** (same) |
| D3 repeated top-level `hyperedges` at END OF FILE | loads | **Stop** "top-level key 'hyperedges' is repeated" |
| D4 repeated hyperedge id at end of `hyperedges` (identical bytes) | loads | **Stop** |
| D7 repeated vdb entry id at end of `entries` | loads | **Stop** |
| D8 repeated vdb top-level `entries` at end | loads | **Stop** |
| CLEAN control: same split/join surgery path, one NEW unique node id | loads | no Stop |
| CLEAN control: top-level split/join round trip, nothing changed | loads | no Stop |
| CLEAN control: one NEW unique vdb entry id | loads | no Stop |
| D5 repeated top-level `archived_hyperedges` / `version` (keys neither reader uses) | loads | no Stop (correct: skipped by both readers, never read by the canonical restore's indexes) |
| D6 repeated id INSIDE `archived_hyperedges` | loads | no Stop (worker's stated non-coverage; harmless, never indexed) |
| D9 repeated vdb top-level `version` | loads | no Stop (unread key) |

1.1 **False positives: none.** Real duplicates all fire, at the end of map and end of file as well as adjacent; every clean variant (my control world, the round trip, unique additions) passes; the 244-test file includes the worker's own controls.
1.2 **Does the new id set change any non-duplicate output? No.** I rebuilt the extras world with a shape the worker did not build (30 filler nodes, 15 S-incoming synapses on two S wants, one outgoing S synapse, a 4-member hyperedge holding three S wants + a filler; canonical rewrite of both files) and ran OLD (`_old_analyze`) vs NEW (`tool.analyze`) and then `prepare_outputs` on each: analyze keys differing `[]`; V11 "before" figures equal; every stamped report sha256 equal; review-file bytes, verifier results (**19**), `failed` (`[]` / `[]`), T6, out hashes, id-map sha256 and the rewritten `main.msgpack` bytes all equal. The worker's own world is covered by the suite's equivalence tests (all pass in the full run below).
1.3 **Memory side of the id set:** `load_content_subset` now retains every entry id (`seen_ids`), O(entries). The suite's shape test printed `new_peak=4.27MB` at 8,000 entries and `4.53MB` at 16,000 (matches the return; it is a `tracemalloc` SHAPE check, not the real peak). On the real vdb the addition is ids only (tens of MB even at several hundred thousand entries, an estimate, unmeasured), nowhere near the 3 GiB/8 GiB floors, and nothing was lowered. Accepted.
1.4 **The worker's non-coverage of a repeated SYNAPSE id and of ids inside `archived_hyperedges`: ACCEPTED, with a measured correction to the stated reason.** `archived_hyperedges` ids: never indexed by the canonical restore, D6 confirms no effect. Repeated synapse id: the worker framed it as a cost decision. I ran it (D10: a repeated synapse id, last copy re-pointed at another S want): the **canonical restore does not take last-wins there: it RAISES `ValueError: duplicate synapse_id: ...`** (native synapse store, `neuro_foundation.py:5451`). So it is not a "raw means complete" divergence (the streamed pass does not silently pick or merge a value into any output): `analyze()` streams past it (the streamed figures over-count by one on the first endpoint; the same validation-gap class the return already discloses), and `prepare_outputs` then **raises at `build_outputs:2699`, the V11 canonical restore of the OUTPUT** (the rewriter raw-copies the repeated synapse into the output), before any write, so the run fails closed. Observed: `prepare_outputs RAISED ValueError duplicate synapse_id`. Consequence for LAW 7: no smaller or merged content set ever reaches an artifact or a live file on this path; V11 is exactly the guard that catches it. Two residues, both stated in correction C1: the failure surfaces as an uncaught `ValueError` (`main()` catches only `Refusal`/`Stop`, `:3288-3291`), not a clean `Stop`; and `--step classify` alone (no V11) would still emit Phase-1 reports from such a file (the disclosed class). Not worth the O(synapses) id set the worker declined.

## 2. Extras-world additions, OLD-path Stop — COMPLIANT

2.1 Mutants re-run by me on a /tmp COPY of the tool + tests (`/tmp/le037/mut`, source worktree untouched, `git status --porcelain` 0 lines after): **M5** (skip a `hyperedges` entry whose stored `is_archived` is true) -> `2 failed, 2 passed` (graph-stream equality; streamed-vs-canonical analysis); **M9** (skip self-loops) -> `3 failed` (the two above plus the direct both-sides test); **drop the repeated node-id and repeated hyperedge-id Stops** -> `2 failed` (exactly the two matching duplicate tests). These match the worker's claims (M5 2, M9 2 + the later direct test). M8 (open the vectors file before the scope check) reasoned from code: the missing-scope check is at `:1758-1760`, before `load_content_subset`; the test replaces `load_content_subset` with an `AssertionError` raiser, so any earlier open fails it. M6/M6b: reasoned from the diff (the ghost POST side lives only in the direct test, for the reason the return gives).
2.2 **OLD-path missing-scope Stop is a faithful copy.** `git show 09a032c:...` lines 1607-1609: `missing = [i for i in scope if i not in nodes_meta or not isinstance(nodes_meta[i].get("want_text"), str)]` / `if missing:` / `raise Stop("scope: %d id(s) of S are not want nodes in the graph (first: %s)" ...)`. The test's `_old_analyze` copy and the new tool `:1758-1760` are character-for-character the same, in the same position relative to the loads (after the load, before `Classifier`). It was not bent to agree.

## 3. V11 untouched and still a GATE — COMPLIANT

3.1 AST comparison of every top-level function/class between `3cd271b` and `2c31077` (docstrings included): changed = `_once` (new), `load_content_subset`, `stream_graph_nodes`, `stream_incident_figures` ONLY. `build_outputs`, `Verifier`, `stage_apply`, `analyze`, `_streamed` are AST-identical. Text hashes of `build_outputs` (`ca6cbf5a0349cd03`) and of the V11 block (marker `V11 canonical restore` through `self.check("V11"`, `fe955e9698f12701`) equal at both commits; all diff hunks lie in the header and `:1556-1700`.
3.2 Exactly ONE `.restore(` in code: `:2699` (`g2.restore(out_main)`); V11 (`:2080-2102`) still compares the streamed "before" figures to the canonical output's and counts dangling; `ok11` still drives `self.check("V11", ...)` into the verifier result and `failed()`.
3.3 Old-path grep (`SimpleVectorDB|load_pair|vdb.load|vdb.content|incident_figures(g`): hits only on `#` comment/changelog lines (`:9, :18, :27, :42`); `test_the_old_whole_file_paths_are_gone_from_the_tool_source` passes. Pre-flight grep (`MemAvailable|loadavg|preflight|MemoryMax|MemorySwap`): 0 in the tool, 0 in the diff. Canonical `SimpleVectorDB.load` / `Graph.restore` unedited and unwrapped; no vendored/protected file in the diff; tool still under `oneshot-tool/`; nothing added in parallel (LAW 3, 4, 2 clear). Pin untouched (HEAD `ae798b94`, 0 porcelain lines).

## 4. Comments edited in place — COMPLIANT

The section banner (`:1571`) and the TURN A "How" sentence (`:41-46` region, bracketed `[Edited in place, #11952: ...]`) now say one thing; the #11805 entry no longer points at a sentence that says the opposite. Remaining stale-claim grep (`canonical reader`, `analysis-001 loader`, `later duplicate`, `duplicate id wins`) finds only the bracketed edited sentence (history, labelled). `_streamed` is AST-identical to `3cd271b`: the broad `except Exception` is unchanged code (comment lines only); the Z12 ruling is recorded at the line. LAW 3 clear (edit in place, no second implementation).

## 5. The Phase-2 finding — CONFIRMED, with a sharpening (advice only, nothing edited)

5.1 **Confirmed.** Reading `stage_apply` (`:3004-3116`): gates P1-P9 (`gate_p3` runs the pinned PARSER tests, not a data load); `analyze(pinned, tdir, ...)` at `:3043` STREAMS the live bytes; `prepare_outputs(... in_dir=tdir ...)` at `:3053` -> `build_outputs` -> the V11 restore of the STAGED OUTPUT (`:2699`); `live_ok` (`:3094-3096`) is sha256 equality plus an old-id census. The only `.restore(` is `:2699`; there is no `SimpleVectorDB` anywhere in code. **No step canonical-loads the live (or copy) INPUT.**
5.2 **Sharpening (correction C2).** Two facts the return's bounded-exposure paragraph should state: (a) for `main.msgpack`, the file the daemon will load after apply is, byte for byte, the staged output V11 already canonically restored (`live_ok` proves live == staged by sha256), so the daemon-bound main bytes ARE canonically accepted; what a restore of the INPUT would add for main is an early, attributed failure, not protection of the live load. (b) `vectors.msgpack` is the actual gap: it is never canonical-loaded anywhere in the tool now (neither input nor output; it is hash-equal, V8), so a vectors file that `SimpleVectorDB.load` would reject is first discovered by the daemon at the S4 start. That exposure is pre-existing data health, not something the repair causes.
5.3 **Placement advice if the separate change is built** (Chief-003 ruled it a separate change):
- **Primary home: `stage_phase2_backup` (`:2910`), after the independent re-read passes (`:2926-2928`) and BEFORE the manifest is written (`:2929`)**, against the BACKUP copy (`<run>/backup/main.msgpack`, `vectors.msgpack`; hash-equal to live, daemon mechanically down, hold in place). A non-loadable input then produces NO manifest, so Josh has nothing to quote in his go. The transfer to the live input is already mechanical: `gate_p4` (`:2469-2470`, `six_files_equal_the_start_of_phase2_backup`) requires the live six files to equal the manifest hashes at apply, and `recheck()` re-asserts it immediately before `os.replace`. Record the result (node/synapse/hyperedge counts, vdb entry count, "canonical load of the input: ok") inside the manifest (its sha is what the go quotes) and echo it into the FINAL receipt (`:3105`); `tool_sha256()` already binds it.
- **Main and vectors SEQUENTIALLY, with the first freed (`del` + `gc.collect()`) before the second**: the old full `SimpleVectorDB.load` inflates the ~1 GB file and copies every embedding, the very peak the delta removed from analysis; the ~3.6 GiB main restore and the vectors load must not be co-resident, and this step must run under the same 8 GiB heavy floor / 6 GB cap / `MemorySwapMax=0` as V11's restore (under `MemorySwapMax=0` an overshoot is an OOM kill, so the runner's pre-flight, which lives outside the tool, must precede it exactly as for V11).
- **If Chief-003 prefers the write step itself: in `stage_apply` between the P5 checks (`:3047`) and `prepare_outputs` (`:3053`)**, never inside `build_outputs` (V11 territory, must stay untouched) and never after `prepare_outputs` returns but sharing residency: `prepare_outputs` pops `g2` and `gc.collect()`s (`:2743-2744`), so the V11 graph is freed on return, but anything restored before it must also be freed BEFORE `build_outputs` starts, or two ~3.6 GiB graphs stack.
- **It must NOT break:** (i) the failure must be a clean `Stop` raised before `atomic_file_write` (`:3092`) and before `write_retired_receipt` (`:3113`): a failed check must not consume the one-shot (same principle as the zero-write rule at `:3049`); (ii) no write to, rename in, or second open-for-write of the live directory (read-only open; ideally the backup copy); (iii) V11, its position in the verifier list, and `build_outputs` stay byte-identical (this review's check 3); (iv) the receipt/manifest schema addition changes the manifest sha256 Josh quotes, so tests that pin manifest shapes need updating in the same change; (v) the canonical loader is called as-is, never wrapped or patched (LAW 4/2).
- **Dry run:** one canonical `SimpleVectorDB.load` of the COPY belongs in the `rewrite` step AFTER `prepare_outputs` has returned (graph freed), not in `classify` (keeps the classify step's streamed footprint).

## 6. Re-stamp honesty — COMPLIANT

- `sha256sum` of the tool: `bf78defaadd28a91d935d7999ab4f136904aafea296afe3a2c6865e05e6f883c` (equals the return); tests: `079e95f1d430de4199d170303a42cf298089f3b973d1355ddef5924c752ca1fd` (equals the return). `git status --porcelain` empty before and after my runs.
- **Full test file, run ONCE by me** (`env -u PYTHONPATH -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 python3 -B -m pytest tests/test_want_text_repair_oneshot.py -s -q -p no:cacheprovider`, real HOME, cwd the tool worktree): **`244 passed in 133.33s`**, exit 0 (the return says 117.17 s; wall time differs with load, the count matches). The suite's P379 preamble printed the pin paths and "not loaded" for the five others; `MEMSHAPE` lines (`4.27MB` at 8,000; `4.53MB` at 16,000) equal the return. My mutant runs (section 2) were targeted `-k` subsets on a /tmp copy, separate from that once-only run.
- **Stamps bind to the running file:** `tool_sha256()` (`:2579`) is `sha256_file(os.path.realpath(__file__))`, used 7 times (run record, post-apply receipts, RETIRED receipt); no stale stamp constant remains (`59ed9f82`, `cbc38bf4`, `bf78defa`: 0 hits as literals in the tool or the tests). The `test_file_sha256` at `:101` is the PIN's own test file, not this tool's. No earlier real run exists whose stamp could be stale (per the return).

## Corrections (numbered; none blocks acceptance)

- **C1.** Return section 2 "NOT covered: a repeated synapse id" should add the measured fact: the canonical restore RAISES `ValueError: duplicate synapse_id` on it (not last-wins), so it belongs to the disclosed "canonical rejects, streamed accepts" class and is closed fail-closed by V11 (`build_outputs:2699`, before any write). Optional hardening, only if the Chief wants a clean exit code rather than a traceback: a `ValueError` from V11's restore surfaces as an uncaught exception (`main()` catches only `Refusal`/`Stop`); V11 itself must not be touched to do it.
- **C2.** Return section 4: state that the live `main.msgpack` the daemon will load after apply equals the staged output V11 restored (proved by `live_ok`), so the real canonical-load gap is `vectors.msgpack` (never canonical-loaded in any step), not `main.msgpack`.
- **C3.** When the separate Phase-2 change is scoped, use the placement/"must not break" list in 5.3 (backup step, sequential, freed, before `atomic_file_write` and before retirement).

## Correct implementations (brief)

Repeated keys are refused rather than merged: the right LAW 7 posture ("raw means complete", never pick a winner); the old last-wins emulation was removed rather than kept beside the new check (LAW 3); V11/the canonical loaders/the pin are untouched (LAW 4, 2); every artifact stamp derives from the running file; the stale comments were edited in place, not annotated beside.

## Remediation priority

CRITICAL: none. HIGH: none. MEDIUM: none. LOW: C1-C3 (documentation/scoping); ethos drift: none (no conventional-pattern drift; nothing wired into a live consumer).

## Recommended next steps

Fold C1/C2 into the return's disclosure text and scope the separate Phase-2 canonical-load change per 5.3; the tool itself needs no further change for this fold.

## 7. Not verified (numbered)

1. Anything on real data: no real checkpoint, backup directory or the kept COPY was opened, listed, hashed or statted; the real vdb entry count and the real peak of the added id set are unmeasured (1.3 is an estimate and the suite's shape check).
2. The native synapse store corner beyond the synthetic cases (the `ValueError` in 1.4 was observed on synthetic bytes only); the pure-Python msgpack fallback; the Unpacker `skip` buffer behaviour on the real `main.msgpack`.
3. Mutants M6 / M6b / M8 were reasoned from the diff and code, not re-run; M5, M9 and the dup-Stop removal were re-run on a /tmp copy.
4. I did not re-run le-035's or checker-028's own scripts; their measurements (88/1,200 and 58/6,000 mutated-file counts) are quoted as theirs.
5. The Phase-2 change in section 5 is advice from reading `stage_apply`/`stage_phase2_backup`; nothing of it was built or run, and the actual memory peak of a canonical restore plus vectors load under `MemorySwapMax=0` was not measured.
6. `stage_rollback` (`:3119`) and the runner-side pre-flight (outside this tool) were not re-read for this fold (not in its diff).
7. The `ValueError` exit shape on a repeated synapse id (1.4) was observed through `prepare_outputs`, not through the CLI `main()` end to end.
