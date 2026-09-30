```
---- Changelog ----
[2026-09-29] Claude Code (claude-sonnet-5-5, worker seat) — plan-001: the 118 mis-parsed want-text repair
What: PLAN ONLY for the Josh-approved separation of the 118 oversized `cc_authored` want texts
      (Exec Packet 392 B.2; acceptance row #801, Exec Packet 393). No code, no data load, no write.
Why:  assignments/plan-want-text-repair-118.md (docs worktree branch cc-laptop-daemon-recall-756-20260930).
How:  read cc_ng_organism.py / cc_ng_host.py / neuro_foundation.py / checkpoint_guardian.py /
      cc_topology_export.py / cc_topology_merge.py at base e4ebf982 and the DERIVED analysis JSON only.
-------------------
```

# plan-001 — the 118-want TEXT repair (separate each real want from swallowed prose; delete nothing)

Lane `z12-s3-restore-bundle-20260929` (dispatch #10634) · branch `cc-laptop-want-text-repair-20260930` · base `origin/main` `e4ebf982b1989fd9066d610b94853bc68bf70d37`
Status: **PLAN.** Next gates: cross-family (non-glm) + LE pair review → a build lane for the scratch script + synthetic tests → a copy-first run under separate authority.

## 0. What I did and did not do

- Read (full regions, not grep hits): `cc_ng_organism.py` 330-400, 1090-1200, 1495-1625, 1720-1775, 3386-3485, 4620-4670; `cc_ng_host.py` 637-705, 1500-1540; `neuro_foundation.py` 3500-3575, 4960-5110; `checkpoint_guardian.py` 112-141, 340-363, 418-481; `cleanup_cc_tool_noise.py` 1-140; `cc_topology_export.py` 228-295; `tests/test_cc_want_bounds.py` (via grep); the assignment; the want-hub d rev3 assignment; `protected_census.py` head.
- **No** graph or checkpoint load, no build, no daemon/unit start or stop, no write to any checkpoint, `~/.claude/plugins/neurograph`, `~/NeuroGraph/data/checkpoints`, or the live tract. Nothing under Syl's directories was opened.
- Derived data used (all under `/home/josh/backups/z12-s3-restore-bundle-20260929T222529Z/analysis-scratch/`): `summary-laptop.json`, `probe-laptop.json` (per-node `[creation_time, creation_mode, provenance, constitutional, source[:24]]`; vdb `[len, sha1, first100, last60]`), **and `want-rows-laptop.json`** — a third derived file NOT on the assignment's list. I used it because `probe-laptop.json` carries no want text (want nodes have no vdb entry) and `summary-laptop.json` has only aggregates; `want-rows-laptop.json` is the same prior worker's read-only stream output (id, len, backtick-lead, has-marker, 70-char head, key set, creation_time, `source_node`). No file was opened in a way that loads a graph. Flagging the deviation so the reviewers can rule on it.
- Machine state when written: `MemAvailable` ≈ 4 GiB, load ≈ 5.6 → **not executable now** (§5 needs ≥ 8 GiB and load < 6).
- Facts below are tagged **[code]** (read at the cited line), **[derived]** (from the analysis JSON), or **[unverified]** (needs data or a running system; stated with the check that would settle it).

## 1. Where the mis-parse happens, and whether new deposits reproduce it

**Origin [code].** Before 2026-09-16 the want bucket used `\[WANT\](.*?)\[/WANT\]` with `re.DOTALL`: a `[WANT]` that is only *mentioned* (prose, a code span, a pasted tool result) opens a span that runs to the next `[/WANT]` however far away. Changelog `cc_ng_organism.py:352-368` records the incident (182 want-nodes, 118 over 600 chars, largest 136,449, "## What I Want" = 2.27 MB per turn).

**What the parser calls a genuine want now [code].**
`WANT_MAX_CHARS = 600`, `WANT_RENDER_LIMIT = 40`, `_WANT_RE = r"\[WANT\](.{1,600}?)\[/WANT\]"` (`:1512-1514`), then in `surface_wants` (`:1517-1572`): skip if the character before the opener is a backtick (`:1549`); skip if `.strip()`ped inner is empty (`:1552`); skip if the inner contains another `[WANT]`/`[/WANT]` (`:1556` — the comment at `:1554-1555` states the pairing rule: a closing tag belongs to the *nearest preceding* opener with no marker between); id `cc:want::` + `sha1(inner)[:16]` (`:1558`); dedupe `if want_id in graph.nodes` (`:1559`); create node with `kind/want_text/want_state/provenance/source_node/creation_mode` (`:1561-1565`), synapse source→want weight 0.3 (`:1567`). Regression tests: `tests/test_cc_want_bounds.py`.

**Is the fix in the running bytes? [derived+git].** The bound-fix commit `d75efeb` (2026-09-16) is an ancestor of `18a090e`, the git stamp in the laptop checkpoint's manifest (saved 2026-09-23). The checkpoint still holds 182 wants, all `cc:want::`, all `cc_authored`, all `open` **[derived]**. So the bound is in, and it does not repair nodes that already exist (nothing rewrites text; the guard only prevents *new* mints).

**Two ways the mis-parse can still recur — the parser is only half-fixed.**

1. **`surface_wants_for_graph` (`cc_ng_organism.py:1128-1195`) is the OLD unbounded implementation, still live.** Regex `\[WANT\](.*?)\[/WANT\]` (`:1163`), no backtick guard, no marker guard, id prefix `want::` (`:1167`, a different namespace from `cc:want::`), provenance `cc_authored`. It is called from `cc_ng_host.py:696-704` inside `_deposit` (`:637`) **after every turn deposit** (after `run_conversational_dual_pass`, so the new conversational node is already in graph and vdb). Any turn that merely *mentions* `[WANT]` alongside a far `[/WANT]` mints a fresh oversized, prune-protected `want::…` node. `tests/test_cc_host_stop_door.py:163-208` exercises exactly this real path.
   - **Observed [derived]:** zero `want::`-prefixed nodes exist in the 09-23 checkpoint (all 182 are `cc:want::`). So it has not been *seen* to fire. **[unverified]** whether it is reachable/failing at runtime (its `except` logs at DEBUG only, `cc_ng_host.py:703-704`). §5 T6 replays it on the copy to show what the next deposit *would* mint.
   - This is two implementations of one bucket (LAW 3 shrapnel). The fix is at the source (LAW 4) and it is a **separate code lane**: make the per-turn path use the bounded/guarded function (or drop the call), not a patch on the repaired data. **This plan does not do it.** Recommended as a hard prerequisite of the S4 start (not of the offline repair).
2. **Canonical `neurograph_rpc.py:4902` carries the same unbounded regex** for Syl's `syl_authored` path (already noted in the `cc_ng_organism.py:364-368` changelog as pending Josh's approval). Out of scope here; listed in §8.

**Would a repair be undone by the next pulse? [code] — mostly no, with one visible side effect.**
- The oversized texts themselves can never be re-emitted (bounded + guarded) and nothing recomputes them.
- `_autosave_loop` runs `surface_wants` (`cc_ng_host.py:1531`) after `drain_ingest_tract` (`:1526`), inside one `try` whose `except` only logs DEBUG (`:1533-1534`). If the drain or probation call raises, `surface_wants` is skipped that pulse. **[unverified]** whether it runs at all on the laptop; the count staying at 182 for a week (09-16 → 09-23) is consistent with either "nothing new to mint" or "never reached".
- **The one interaction:** dedupe is by *id*, id is `sha1(text)`. After an in-place repair the node keeps `sha1(old text)`, so if the production parser (bounded) *does* emit the repaired text `X` from the source node, `cc:want::sha1(X)` is not in `graph.nodes` and a **duplicate** want with text `X` is minted (protected, cc_authored). This is independent of the repair (the parser would mint it today), but after the repair the duplicate becomes visible as a twin. §3 defines the classes E1/E2/E3 and §5 T6 measures them; it does not block the repair but Exec must know (§7 Q4).

## 2. Node identity vs text (the crux)

**Claim: editing `want_text` in place does not change any id, and nothing in the code reads a want's id back from its text.** Evidence [code]:

- `sha1(text)` is computed only at **creation**: `cc_ng_organism.py:1167` (legacy), `:1558` (`surface_wants`), `:1749`/`:1760` (`generate_emergent_want`, keyed on `tonic-concept::<label>` or the emergent text; no `cc_emergent` node is among the 182 **[derived]**), `:2151`/`:4374` (`cc:conv::` + full `sha1(turn text)` — the *source* nodes, unrelated to want text). After creation the id is an opaque dict key.
- Everything that references a want does so **by id**: `graph.nodes` key, synapse `pre_node_id`/`post_node_id`, hyperedge `member_nodes`, the activations sidecar (`main.msgpack.activations.json`, keyed by node id), `guardian_nodes` (a *count*). Wants have **no vdb entry** (0 of 182 ids in `probe-laptop.json` vdb **[derived]**), so no vector-db key is derived from want text.
- The only lookup that hashes text is the creation-time dedupe (`:1559`) discussed in §1.
- The `cc:conv::` mapping is the want's `source_node` metadata (all 182 present **[derived]**) plus one synapse source→want; neither depends on `want_text`.
- The `cc_topology_merge` receiver dedupes by `node_id in graph.nodes` only (`cc_topology_merge.py:373-374`; its own comment at `:61`), so a merge never overwrites repaired text — but a peer that lacks the repair keeps the old text under the same id (§6).

**Fields — exactly one changes per repaired node:**

| Field | After repair |
|---|---|
| node id (dict key), `node_id` | **unchanged** (no longer equals `sha1(want_text)[:16]`; nothing depends on that equality) |
| `metadata["want_text"]` | **CHANGED** → new text `X` |
| `provenance`, `kind`, `want_state`, `source_node`, `creation_mode`, `constitutional` (absent) | unchanged |
| `metadata["poincare_dir"]` (packed float32 bytes) | **unchanged** — see below |
| voltage/threshold/spike state/`creation_time`/every dynamics field | unchanged |
| every synapse, hyperedge, sidecar, vdb entry | unchanged (byte-identical) |

**The embedding.** A want has no vdb embedding; its only embedding-derived field is `poincare_dir`. All 182 carry one **[derived: key set on 182/182]**, stamped by `cc_stamp_missing_geometry` from `want_text` (`cc_ng_organism.py:3388-3484`, source order at `:3453-3454`), i.e. from the **old bloated text** (how much of a 136 KB string the embedder actually consumed is **[unverified]**; I did not read `ng_embed`'s input bounding). That function stamps **only nodes missing** the field (`:3419-3420`, `:3469`), so nothing will refresh it after a text edit. `poincare_dir` feeds GSG geometry (`neuro_foundation.py:2259`, `:3692`; `cc_ng_organism.py:3206-3222`, `:3268-3275`) and is banned from the export path (`cc_topology_export.py:264-265`).
- Recomputing it needs the embedding model (`ng_embed`) in the repair tool and changes geometry-dependent behaviour on 118 protected nodes. Leaving it is the **minimal change and is exactly today's state** (no regression). **Recommendation: leave `poincare_dir` byte-identical in this pass** and let Exec rule separately (§7 Q3) on a follow-up that clears the field on repaired ids so the daemon's own `cc_stamp_missing_geometry` re-derives it from the new text. I did not choose either silently: acceptance 2/3 do not forbid touching it, but "UNTOUCHABLE except the approved separation" argues for leaving it.

## 3. The mechanical separation rule

### 3.1 Scope
`S` = nodes with `kind == "want"`, `provenance == "cc_authored"`, `len(want_text) > WANT_MAX_CHARS` (import the constant from the module in the tool; do not copy the number). **Expected |S| = 118** [derived]. The other 64 wants (≤ 600) are **never touched** (§3.5).

### 3.2 The rule (parser's own terms, no guessing)
For `T = want_text` and `C = vdb.content[source_node]`, apply gates **in order; the first failing gate leaves the node UNCHANGED and LISTED with its reason code.**

- **G1 source present.** `source_node` is a graph node that is not a want, not constitutional, not `*_authored`, and `C` exists. (All 118 pass [derived].)
- **G2 T is exactly an old-parse span of C.** `C.count(T) == 1` at position `p`; `C[:p].rstrip()` ends with `[WANT]` (the outer opener at index `i`); `C[p+len(T):].lstrip()` starts with `[/WANT]`. (This is what the old regex plus `.strip()` produced. A T from any other writer, or a repeated T, fails and is listed.)
- **G3 the outer opener is documentation by the parser's own guard.** `i > 0 and C[i-1] == "`"` — the same condition as `cc_ng_organism.py:1549`. (Proxy in derived data: `T[0] == "`"`, 83 of 118. Exact G3 count only known from the dry run.) If the outer opener is *not* backtick-preceded, the text before the inner opener could itself be a genuine want whose closing tag was forgotten — that cannot be told mechanically, so the node is left unchanged.
- **G4 a well-formed inner pair exists.** `k = T.rfind("[WANT]")` exists; `k == 0 or T[k-1] != "`"`; `X_raw = T[k+6:]`; `_WANT_RE.fullmatch("[WANT]" + X_raw + "[/WANT]")` matches (this re-uses the module's own bound); `X = X_raw.strip()` is non-empty. (`T` cannot contain `[/WANT]` by construction of the old regex; `X` cannot contain `[WANT]` because `k` is the last opener.)
- **Result:** `new_text = X`. Post-conditions asserted per node, or the node is not written: `new_text in T`; `T.endswith(new_text)`; `0 < len(new_text) <= WANT_MAX_CHARS`; `new_text == new_text.strip()`; no marker inside.

This is the parser's own definition of a well-formed want (a closed pair, nearest opener, no marker inside, opener not backtick-preceded, ≤ 600). It selects the *suffix* because the closing tag that ended the old match belongs to the last opener before it.

### 3.3 Documented failure modes (what the rule cannot know)
1. **Parser-well-formed is not the same as author-intended.** A quoted example (`"[WANT]like this[/WANT]"` inside a code listing or tool output) is a well-formed pair and will be selected; the rule cannot tell a real want from a quoted fixture. The result is still a verbatim, bounded substring, nothing is deleted, and the parser itself would treat identical fresh text as a want. Mitigation: the report prints every proposed `X` in full for Exec review before apply (§5, "frozen list").
2. **Head-genuine unclosed wants** (a real `[WANT]` whose closing tag was never written, followed by prose to the next `[/WANT]`) are exactly the non-backtick class; G3 leaves them unchanged and lists them. They are never guessed.
3. **Production may not emit `X`** (it skips a match that contains a nested marker; it only emits `X` when the leftmost opener within the 600-char window is `k`). Classes: **E1** production would not emit `X`; **E2** it would, and `cc:want::sha1(X)` already exists; **E3** it would and the id is absent (a duplicate would be minted on the next pulse, §1). The repair does not change E1/E2/E3 membership (§5 T6 proves the minted set is identical pre/post).

### 3.4 Class enumeration from derived data (118 oversized, `want-rows-laptop.json`)

| Class | Definition (derived flags) | Count | Rule outcome |
|---|---|---|---|
| **A** backtick-leading + contains `[WANT]` | `T[0]=="`"`, marker present | **67** | Candidates. Repaired iff G1-G4 pass. Upper bound 67. |
| **B** backtick-leading, no marker | `T[0]=="`"`, no `[WANT]` | **16** | No inner opener → **UNCHANGED, LISTED** (certain, from flags). |
| **C** non-backtick, contains marker | outer opener not backtick-led | **35** | Fail G3 (assuming G3 ≈ the `T[0]` proxy) → **UNCHANGED, LISTED**. |
| total | | **118** | |

Nothing else in `S`: no oversized node is "neither". 83 backtick-led matches the parser changelog's "83 of the 118".

**Expected outcome:** repaired ≤ 67; LEFT UNCHANGED ≥ 51 (16 certain + 35 proxy-dependent) plus the class-A nodes that fail G2/G4. **The exact split is not knowable without the text** — the dry-run classification produces it, and the plan deliberately does not estimate it. All 118 have a source conversation node that is in the graph, conversational, unprotected (`source in nodes` = 118/118; 49 distinct sources) **[derived]**.

### 3.5 Out of scope, but you should know
The 64 wants ≤ 600 chars are mostly **code and test-fixture fragments** (e.g. lengths 1, 3, 4; strings such as `\") {\n let after = &search[start + 6..]`), plus a few real-looking wants; 21 contain a marker and 8 begin with a backtick [derived]. They are outside the approved "118 oversized" scope and are **not touched**. Listed for Exec (§7 Q2).

## 4. Where the swallowed prose goes

**Nowhere new — it is already preserved, verbatim, as ordinary non-want text, and the plan proves it per node instead of copying it.** By construction (G2) `T` is a contiguous substring of the source conversation node's content `C`; the removed remainder (`T[:k]`, the prose before the inner opener) is therefore a substring of `C`. The source node is an ordinary `cc:conv::` node (`creation_mode=conversational`, no provenance, not protected) whose text lives in vdb, and the vdb is untouched. A new "prose" node would (a) need a full Node record and an embedding, (b) be an orphan the next `_collect_orphan_nodes` pass may collect (it is not protected), i.e. the prose would be *deleted* by the graph's own hygiene, and (c) change non-want node counts and the SaveGate reference. So:

- **Acceptance 5 is checked as provenance:** for each repaired id, `source_node` ∈ graph nodes, is `kind != want`, has no `_authored`/`constitutional` flag, and `vdb.content[source_node]` (unchanged bytes) contains the old text `T` verbatim — so `T[:k]` and `X` are both still present.
- No synapse of any want is created, altered or removed; the source→want synapse is untouched.
- If Exec wants belt-and-braces, an *off-graph* receipt (old text per id, sha256) is written next to the backup, never into the repo (§5).

## 5. Execution procedure (offline, on a copy first; separate authority)

**Tool:** a **new scratch script** (suggested `scripts/want_text_repair.py`; no protected file, no vendored file touched), built in a later code lane with synthetic tests. It is a **byte-level splice**, not a Graph round-trip, for a specific reason: `Graph.restore()`+`checkpoint()` cannot promise acceptance 3 (byte-identical non-want nodes and synapses) — restore deserializes, may migrate fields, and re-serializes everything, and it costs ~3.6 GiB RSS (`summary-laptop.json` `process_maxrss_MiB` = 3614). A splice copies every other byte verbatim.

**Why a splice is safe [code].** `Graph.write_checkpoint` (`neuro_foundation.py:4972-5021`) writes a msgpack map with `Packer(use_bin_type=True)`: `pack_map_header`, then each top-level key and value. Node values sit in the `nodes` map (`_serialize_node`, `:5089-5109`, `metadata` at `:5102`). msgpack maps carry *element counts*, not byte lengths, so replacing one node value changes no parent header; only the modified node's own str/map length prefixes (re-emitted by the packer). Reading uses `Unpacker.tell()`/`skip()` exactly as `restore` does (`:5050-5061`).

### 5.1 Preflight (both phases)
`MemAvailable ≥ 8 GiB`, `loadavg < 6`, **one loader at a time**; run as `systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0` for the splice/classify passes (peak ≈ raw main.msgpack ~231 MB (sum of `saved_top_level_keys_and_value_bytes`, **[derived]**) + streamed vdb content; the analysis probe with far more retained state peaked at 2.0 GiB). The optional canonical-reader check A9 needs **6G** (`protected_census.py` used 6G; canonical restore measured 3.6 GiB) and the full 8 GiB headroom. Env `NG_EMBED_REMOTE` unset and `HF_HUB_OFFLINE=1` (the tool does no embedding).

### 5.2 Phase 1 — dry-run on a COPY (daemon may stay up; nothing live is touched)
1. Copy the six checkpoint files (`main.msgpack`, `vectors.msgpack`, `.activations.json`, `.guard_state.json`, `.manifest.json`, `commons.msgpack`) to `/home/josh/backups/z12-want-text-repair-<UTC>/copy/`; record `sha256` of source and copy; abort on mismatch. (A copy taken under a live daemon is a consistent snapshot because every writer replaces atomically, but it goes stale — see 5.3.)
2. **Classify (read-only):** stream `nodes`; for every node with `kind=="want"` record id/len/hashes; for `S` apply G1-G4 using `C` streamed from `vectors.msgpack` (embeddings skipped, as `overlap_probe.py` does). Emit `repair-list.json` `[{id, class, decision, gate_failed, old_len, old_sha16, new_len, new_sha16, new_text}]` and a LISTED table (id, reason, len, first/last 100 chars).
3. **Freeze:** Chief/Exec review the proposed `X` texts and strike any id they do not want (a struck id becomes UNCHANGED). Recommended, not required by row #801; it turns failure mode 3.3-1 into a human-signed list that the tool then applies mechanically.
4. **Splice to a tmp file** on the copy through `checkpoint_guardian.atomic_file_write` (reusing the canonical atomic writer; `:343-363`): for each id in the frozen list, `unpack` that node's value, set `metadata["want_text"]`, `Packer(use_bin_type=True).pack(...)`; everything else is a `raw[a:b]` slice. Before any edit, **A8**: `pack(unpack(span)) == span` for every target node; a node that does not round-trip is not edited (LISTED "encoder not faithful").
5. Run **A1-A9 + T6** (below) on `copy_before` vs `copy_after`; write `report-<UTC>.json`/`.md` in the backup dir (not the repo: it contains want text).

### 5.3 Acceptance checks (all mechanical; row #801)

| # | Check | How |
|---|---|---|
| A1 | want count unchanged (**182**) | count `kind=="want" & provenance=="cc_authored"` in before/after; also total node count and the node-id *set* equal (covers the constitutional node) |
| A2 | ids, `provenance`/`constitutional`/`kind`/`want_state`/`source_node`/`creation_mode`/`poincare_dir` unchanged | dict-compare each of the 182 wants before/after; the only differing key is `want_text` on ids in the frozen list |
| A3 | every non-want node and synapse **byte-identical** | with spans from `Unpacker.tell()`: `raw_before[span] == raw_after[span']` for every node not in the frozen list; the `synapses`, `hyperedges` and every other top-level value byte-equal; and `sha256(before minus repaired-node spans) == sha256(after minus repaired-node spans)` |
| A3b | rim and sidecars | `constitutional::rim::choice_clause` node bytes and each of its incident synapses (4,127 in the 09-23 analysis copy **[derived: `probe-laptop.json`]**; re-counted at run time) byte-equal; `vectors.msgpack`, activations, guard_state, commons: sha256 equal |
| A4 | new text verbatim inside old | `new in old and old.endswith(new)` per repaired id |
| A5 | prose preserved as ordinary non-want text | §4 provenance check per repaired id; vdb `sha256` unchanged |
| A6 | protected census | re-run `protected_census.py` (plan-scratch) before/after: protected count 183, provenance `cc_authored` 182, constitutional 1, incident synapse counts, want states all equal; only `want_text_len`/`want_text_sha16` differ, and only for frozen-list ids |
| A7 | idempotence | running the tool again on the output finds no `len > 600` in `S` it can change → zero edits |
| A8 | encoder fidelity | above |
| A9 | canonical read path | fresh process, `Graph().restore(after)` under 6G: node/synapse/hyperedge counts equal the manifest; `render_wants(graph)` byte length before vs after (report only) |
| **T6** | **no new behaviour on the next pulse** | build a *stub graph* (the pattern of `tests/test_cc_want_bounds.py`: fake nodes with `metadata`, `_step_lock`, recording `create_node`/`create_synapse`) from the copy's `(id, kind, creation_mode, want_text)`, stub vdb `content` from `vectors.msgpack`, and call the **real** `surface_wants` and the **real** `surface_wants_for_graph` on *before* and *after*. Pass iff the minted-id set is identical before/after (the repair does not change what production mints). Report `|minted|` for each function and the E1/E2/E3 split; a non-empty `surface_wants_for_graph` set is the empirical evidence for the §1 code lane. |

Stop conditions: any A-check fails → no live write, report and stop. Any id where `old_sha16` no longer matches → LEFT UNCHANGED.

### 5.4 Phase 2 — live apply (only after Phase 1 passes and Chief/Exec authorise)
1. **Daemon down** (the unit/process that owns the laptop CC checkpoint — name to be confirmed by Chief at execution; `cc-ng-daemon.py` is named in the `cleanup_cc_tool_noise.py` header). Confirm **no process holds the checkpoint** and no `neurograph_rpc.py`/CC-daemon PID survives (Practical Notes: an orphan silently autosaves).
2. **Full backup first**: the six files to `/home/josh/backups/z12-want-text-repair-<UTC>/backup/` with sha256 recorded; re-hash the backup; independent read of the backup (counts vs manifest). Keep the daemon's `generations/` ring untouched (it hardlinks old inodes; `os.replace` preserves them, `checkpoint_guardian.py:575-628`).
3. **Re-run the same tool against the live bytes — do not copy the Phase-1 output over live.** A Phase-1 copy is stale the moment the daemon autosaves; installing it would silently revert everything learned since. The live run consumes the frozen list and refuses any id whose `old_sha16` no longer matches, or if the set of 182 want ids changed.
4. Inside the `atomic_file_write` writer function run **A1-A8** against (backup, tmp); on any failure raise — `atomic_file_write` leaves the final file untouched and unlinks the tmp (`checkpoint_guardian.py:354-363`). Only a fully verified tmp is `os.replace`d onto the live `main.msgpack`.
5. **Manifest / guard state: leave them alone.** Counts are unchanged and `SaveGate.permit` keys off `guardian_nodes` then `nodes` (`checkpoint_guardian.py:451`), so #743 is not triggered. Verify `manifest.nodes/synapses/hyperedges` equal the post-repair counts and record it. (Precedent scripts refresh the manifest only when counts change: `cleanup_cc_tool_noise.py:127-139`.)
6. Do **not** start the daemon (the S4 start owns that). Fresh-process re-verify of live vs backup (A1-A8, sha256 of untouched sidecars) and write the receipt.
7. **First two daemon autosave cycles after S4 start:** re-check wants (count 182, repaired texts unchanged, minted set == the T6 prediction). Report the observation window, not a single-point "done".

**Rollback:** with the daemon down, `atomic_file_write` the backed-up `main.msgpack` over the live one and compare its sha256 to the recorded value (only `main.msgpack` differs); the hardlinked generation is a second copy.

**Writer choice, stated:** `checkpoint_guardian.atomic_file_write` (canonical, tmp + `os.replace`, used by the #379 precedent) with a splice writer function; **not** `Graph.checkpoint()` (§5 intro).

## 6. Ripple

| Reader / consumer | Reads want text? | Effect of repair |
|---|---|---|
| `render_wants` (`cc_ng_organism.py:1575-1610`, called `cc_ng_host.py:1141-1144`) | yes (`:1595`; newest 40, each clamped to 600 at `:1603-1604`) | Bounded already. **[derived]** 28 of the 40 newest wants are in the 118; the top-40 block is ≈ 18.4 KB. Repair shortens the rendered entries for repaired ids only; ordering (by `creation_time`) unchanged |
| identity block `render_constitutional_core` | reads `core_text`, not want text | none |
| `_pith_node_raw_text` (`:4640-4652`, want_text at `:4646`) | yes | shorter provider context for repaired ids; benign |
| `cc_stamp_missing_geometry` (`:3388`) | yes, only when `poincare_dir` missing | none (all 182 have it) — see §2 for the follow-up decision |
| `surface_wants` / `surface_wants_for_graph` | reads `want_text` only for open-want listing; dedupe is by id | §1 duplicate side effect (E3) |
| `generate_emergent_want` | writes `want_text` for its own emergent ids only | none (no emergent among the 182) |
| callosum export/merge (`cc_topology_export.py:259-295`, `cc_topology_merge.py:373-374`) | `want_text` rides `_portable_metadata`; identity nodes cross | receiver skips present ids → a peer keeps the old text under the same id (**divergence, not undoing**) |
| `protected_census.py` | records `want_text_len` and sha16 (`:53-56`) | census values change for repaired ids only; **protected-set membership does not change** (flags untouched) — A6 |
| `guardian_nodes` / #743 / manifest | counts only | no refresh needed (counts unchanged) |
| want-hub d (`assignments/plan-want-hub-d-rev3.md`) and competition plan (d) | work on ids, degrees and synapses, not text | none; ids and synapses unchanged |
| `neurograph_rpc.py:3240` (Syl's path) | reads `want_text` | unaffected (laptop-only repair; Syl's directories untouched) |

## 7. Questions for Chief / Executive (none needs Josh: no product/ethics change)

1. **Parser code lane (LAW 4).** Authorise a separate lane to retire the unbounded `surface_wants_for_graph` (delegate to the bounded, guarded function or drop the `_deposit` call), and to decide on `neurograph_rpc.py:4902`. Recommended: hard prerequisite of the S4 start, not of the offline repair.
2. **Scope of "LEFT UNCHANGED".** Are 51+ untouched oversized wants acceptable (16 without an inner opener; the non-backtick 35; any class-A node failing a gate), plus the 64 short fragments? Alternative: Exec hand-specifies a per-id span list; the tool applies it mechanically (an explicit decision, not a guess).
3. **`poincare_dir`.** Leave (recommended) vs clear-on-repaired-ids-and-let-the-daemon-restamp vs recompute in the tool.
4. **Twins (E3).** If the dry run shows E3 > 0, accept the visible duplicates, or open a code lane to dedupe by text as well as id? Deleting or merging any want is forbidden, so the repair cannot resolve them.
5. **Peer copies.** The VPS/bundle copy may hold the same 118 ids with old text; is a peer repair wanted? **[unverified]** whether the peer has those ids (I did not open `summary-bundle.json`; it is outside the named derived files).
6. **Frozen-list review** (5.2 step 3): required or optional? Who signs?
7. **Slot and unit.** Which daemon unit to stop, which slot (single loader, load < 6, MemAvailable ≥ 8 GiB), and how long to retain the backup and the report (which contains want text).
8. **The extra derived file** `want-rows-laptop.json` (§0): acceptable to reuse for the class table?

## 8. Not verified without data or a running system

- The exact repaired / left-unchanged split (needs the texts; §3.4 gives bounds only).
- Whether `surface_wants_for_graph` and `surface_wants` actually run on the laptop daemon (silent DEBUG failures); T6 shows what they *would* do, not whether they ran.
- Whether the live laptop checkpoint still equals the 09-23 analysis copy (Phase 2 re-derives everything from the live bytes).
- `T` occurs uniquely in `C` for each of the 118 (G2) and that `pack(unpack(span)) == span` for their spans (A8).
- The AUTOSAVE interval and the daemon unit name.

**Flagged for the punchlist (not mine to file):** (1) `surface_wants_for_graph` unbounded twin, live per turn (`cc_ng_host.py:696-704`); (2) `neurograph_rpc.py:4902` unbounded regex (already noted in the 09-16 changelog); (3) `_autosave_loop` bundles `drain_ingest_tract`, `cc_update_probation`, `surface_wants` and `generate_emergent_want` in one DEBUG-swallowed `try` (`cc_ng_host.py:1517-1534`) — a raise in the drain silently skips want surfacing; (4) the 64 short code/fixture-fragment wants; (5) dedupe-by-id-only for text-hashed ids.
