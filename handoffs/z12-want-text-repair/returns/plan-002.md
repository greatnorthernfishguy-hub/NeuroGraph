```
---- Changelog ----
[2026-09-29] Claude Code (claude-sonnet-5-5, worker seat) — plan-002: FOLD of plan-001 (92af5822)
What: (R2·P402) the amended #801 acceptance: the want id FOLLOWS the text (option ii). §2 and §5 rewritten,
      §1/§3/§6 restated, §7 pruned. (R2·P401) the crux ruling: every reference site to a want id enumerated
      with file:line, the canonical-path (LAW 3) analysis, the out-of-graph reference sweep.
Why:  assignments/plan-want-text-repair-118-fold.md (Chief-003 crux ruling P401 + Exec P402). I have read the fold
      assignment, NOT the text of P401/P402 themselves; I tag [R2·P402] the passages that implement the amended
      acceptance items listed in the fold and [R2·P401] the passages that implement its crux items (reference-site
      sweep, LAW-3 canonical path, references outside the graph). Unmarked passages are carried from plan-001 unchanged.
How:  read-only. New code reading: neuro_foundation.py 1900-2060, 3296-3356, 4960-5285, 5368-5667;
      activation_persistence.py 100-170; cc_topology_export.py 340-416; cc_topology_merge.py 170-250;
      plus derived analysis JSON and two small id-only files (§0). plan-001 stays in place, untouched.
-------------------
```

# plan-002 — the 118-want TEXT repair, FOLDED: the id follows the text

Lane `z12-s3-restore-bundle-20260929` (dispatch #10683) · branch `cc-laptop-want-text-repair-20260930` · base of plan-001 `92af58222f1898c617256fb429c0c4c29258d9e5`
Status: **PLAN.** A pair (cross-family + law enforcer) reviews this BEFORE any build or offline execution. No build here.

## 0. What I did and did not do

- **No** graph or checkpoint load, no build, no daemon/unit start or stop, no write to any checkpoint, `~/.claude/plugins/neurograph`, or `~/NeuroGraph/data/checkpoints`; the live tract was never opened; nothing under Syl's directories was opened.
- Derived data (all under `/home/josh/backups/z12-s3-restore-bundle-20260929T222529Z/analysis-scratch/`): `summary-laptop.json`, `probe-laptop.json` (per-node tuples; per-synapse `(pre, post, weight, peak, creation)`; hyperedge member lists; vdb `[len, sha1, first100, last60]`), and `want-rows-laptop.json` (id, len, backtick-lead, has-marker, 70-char head, key set, creation_time, `source_node`; **not on the assignment's list, flagged in plan-001, still flagged**).
- **Two further inputs, both flagged for the reviewers.** (a) `checker-scratch/checker003-roleA/laptop-copy/main.msgpack.activations.json` — another worker's copy of the laptop activation sidecar (JSON, not a graph load); I read only its key set and one entry's field names to confirm the sidecar is keyed by node id. (b) `ng_topology/laptop_cc_membership.json` in the docs worktree (committed `3aed187f`, 2026-09-13) — a list of ids, opened only to see whether want ids appear. I also grepped the docs worktree, `quest-tracker`, and this repo for `cc:want::` (§6).
- Machine state when I last looked: `MemAvailable` ≈ 4 GiB, load ≈ 5.6 → not executable now.
- Tags: **[code]** read at the cited line · **[derived]** from the analysis JSON · **[unverified]** needs data or a running system, with the check that settles it.

## 1. Where the mis-parse happens, and whether new deposits reproduce it

**Origin [code].** Before 2026-09-16 the bucket used `\[WANT\](.*?)\[/WANT\]` with `re.DOTALL`; a merely-*mentioned* `[WANT]` opens a span running to the next `[/WANT]`. Changelog `cc_ng_organism.py:352-368`: 182 want-nodes, 118 over 600 chars, largest 136,449; "## What I Want" was 2.27 MB per turn.

**The parser's own definition of a genuine want [code].** `WANT_MAX_CHARS = 600`, `_WANT_RE = r"\[WANT\](.{1,600}?)\[/WANT\]"` (`:1512-1514`); in `surface_wants` (`:1517-1572`): skip if the character before the opener is a backtick (`:1549`); skip if the stripped inner is empty (`:1552`); skip if the inner contains another marker (`:1556`; the comment `:1554-1555` gives the pairing rule: a closing tag belongs to the nearest preceding opener with no marker between); **id = `"cc:want::" + hashlib.sha1(inner.encode("utf-8")).hexdigest()[:16]` (`:1558`)**; dedupe **`if want_id in graph.nodes: continue` (`:1559-1560`)**; create node (`:1561-1565`), synapse source→want weight 0.3 (`:1567`).

**The fix is in the bytes, but only half of it [derived + git + code].** `d75efeb` (2026-09-16) is an ancestor of `18a090e`, the git stamp in the laptop checkpoint's manifest (saved 2026-09-23). That checkpoint holds 182 wants, all `cc:want::`, all `cc_authored`, all `open`. The bound prevents new mints; it repairs nothing.

**Two ways the mis-parse can still recur (unchanged from plan-001).**
1. **`surface_wants_for_graph` (`cc_ng_organism.py:1128-1195`) is the old unbounded implementation, still live**: regex `(.*?)` (`:1163`), no guards, id prefix `want::` (`:1167`, a different namespace, so it never dedupes against `cc:want::`), provenance `cc_authored`. It runs after every turn deposit (`cc_ng_host.py:696-704`, inside `_deposit` `:637`, after the dual pass). No `want::` node exists in the 09-23 checkpoint [derived]; whether it is reachable/failing at runtime is **[unverified]** (its `except` logs DEBUG only, `:703-704`). This is two implementations of one bucket (LAW 3 shrapnel); the fix is at the source (LAW 4) and is a **separate code lane**, recommended as a prerequisite of the S4 start.
2. Canonical `neurograph_rpc.py:4902` has the same unbounded regex for Syl's path (already noted in the 09-16 changelog); out of scope.

**[R2·P402] What a repair now does to the reproduction risk.** Plan-001 found that an *in-place* text edit leaves the id at `sha1(old text)`, so the first re-parse of the source (`:1558` computes `sha1(X)`, `:1559` finds no such id) would mint a silent duplicate. Under the chosen course the repaired node's id **is** `sha1(X)`, computed by the same expression, so `:1559` dedupes onto the repaired node. **The E3 duplicate risk is removed by construction for every repaired node.** What remains:
- **Collision cases (formerly E2)** — the repaired text's id already exists, or two repairs collapse to one text: the rule in §2.6 leaves both unchanged and lists them.
- **E1** — production may never emit `X` from the source at all (it skips a match containing a nested marker); then nothing changes either way.
- **Wants left unchanged keep their old ids.** The bounded parser can never re-emit their old text, so those ids are inert; but if such a node's source contains a parser-well-formed pair `X'` whose id is absent, production may mint `cc:want::sha1(X')` as a separate node. That is pre-existing and independent of the repair; §5 T6 measures it.
- `_autosave_loop` runs `surface_wants` after `drain_ingest_tract` inside one DEBUG-swallowing `try` (`cc_ng_host.py:1517-1534`); whether it runs at all on the laptop is **[unverified]**.

## 2. [R2·P402][R2·P401] The chosen course: the id follows the text

### 2.1 Decision and evidence
Exec P402 rules option (ii). The evidence stays as plan-001 gave it: an in-place edit changes exactly one field and nothing recomputes an id from text (`sha1` only at creation: `:1167`, `:1558`, `:1749`/`:1760` emergent, `:2151`/`:4374` conv ids), but the dedupe at `:1559` is keyed on the hash of the *new parse*, hence the duplicate. Under (ii) the repaired node carries the id the parser itself would compute, and the dedupe works.

**New id = `cc:want::` + `sha1(new_text)[:16]`, by the same expression.** The expression is inline at `:1558` (also `:1749`, `:1760`); there is no helper to import. So (1) the tool defines one `_want_id(text)` that is character-for-character that expression, and (2) a test **asserts equality against production**: it calls the real `surface_wants` on a stub graph (the `tests/test_cc_want_bounds.py` pattern) with a battery of texts (ASCII, multi-byte UTF-8, edge whitespace, exactly-600-char, a real-shaped sentence) and compares each minted id to `_want_id`; a second test reads `cc_ng_organism.py`, finds the line containing `"cc:want::" + hashlib.sha1(inner.encode("utf-8")).hexdigest()[:16]` and fails if it moved or changed (drift alarm). *Optional, for Exec (§7 Q1):* extract a one-line helper in `cc_ng_organism.py` used by all three sites — a code-lane change, not part of this plan.

### 2.2 What changes per repaired node
**Changes:** the node's dict key and `node_id` (old id → new id); `metadata["want_text"]` (→ `X`); and **every reference site to the old id** (§2.3). **Unchanged:** `provenance` (`cc_authored`), `kind`, `want_state`, `source_node`, `creation_mode`, `poincare_dir` (carried byte-identical; Exec Q3 stays), every dynamics field, `creation_time`, and every synapse/hyperedge field other than the endpoint/member id.

### 2.3 Every reference site to a want id (persisted; the rewrite set)
Enumerated from `Graph._serialize_full` (`neuro_foundation.py:5164-5285`) and `_deserialize` (`:5368-5667`). Volumes are **[derived]** from the 09-23 analysis copy (118 repair candidates upper bound; the true set is smaller, §3.4).

| # | Structure | Where | Effect of the re-key | Volume [derived] |
|---|---|---|---|---|
| S1 | `nodes` map key + `node_id` | `_serialize_node :5089-5090`; `:5188`; restore `:5387-5416` | key and field rewritten in place (map order preserved) | ≤ 118 |
| S2 | `Node.pred_weights` — a dict keyed by **post-node id inside the PRE node** | `:673`, written `:3463`/`:3473`, serialized `:5106`, restored `:5407` | keys equal to an old id rewritten inside **non-want** nodes too | ≤ 1,735 nodes (1,561 non-want) have a synapse into an S id |
| S3 | `synapses` `pre_node_id` / `post_node_id` (native store; synapse ids are UUIDs, `:698`, so **no synapse id changes**) | `create_synapse :2020-2034`; `:5183`/`:5194`; restore `:5449-5457` | endpoint re-pointed; weight/peak/ages/counters/`metadata` untouched | **81,999** of 138,753 touch S (67,535 as pre, 20,561 as post; **13,203** want↔want, 6,097 with both ends in S); **131** are rim↔want |
| S4 | `synapse.metadata["expected_target"]` (a node id inside surprise-driven synapses) | written `:3308` | rewritten if it equals an old id | 5,172 surprise-driven synapses exist; how many reference a want is **[unverified]** |
| S5 | `hyperedges` and `archived_hyperedges`: `member_nodes` (list), `member_weights` (map keyed by node id), `output_targets` (list) | `_serialize_hyperedge :5112-5136`; `:5192-5199`; restore `:5460-5517` | ids substituted in place, list order preserved (a set is serialized in arbitrary order; restore rebuilds sets `:5463`) | 48 hyperedges contain S ids (145 memberships); archived (14 exist) **[unverified]** |
| S6 | `active_predictions` (source/target) | `:5201-5204`, `:5138-5151`; restore `:5563-5584` (dropped if an endpoint is missing) | substituted; else the prediction would be silently dropped | tiny (the value is ~1 byte in the copy) |
| S7 | `prediction_outcomes` (prediction source/target, `actual_firing_nodes`) | `:5206-5214`; restore `:5588-5608` (**not validated — stale ids persist**) | substituted | 409,857 bytes in the copy; occurrences unknown |
| S8 | `he_active_predictions` (`predicted_targets`, `confirmed_targets`) | `:5224-5227`; restore `:5627-5647` | substituted | 516 bytes |
| S9 | `he_output_candidates` (`hid → {node id: count}`) | written `:2396-2434`; `:5265`; restore `:5558` | inner map keys substituted | 787 bytes |
| S10 | `novel_sequence_log` (`source`, `firing_nodes`) | written `:3346-3353`; `:5221`; restore `:5620` | substituted (history is carried) | 4.77 MB; occurrences unknown |
| S11 | `delay_buffer` (`[[node id, current]]`) | `:5271-5274`; restore `:5427-5435` (drops unknown ids) | substituted | 24 KB |
| S12 | `recent_spikes` (keyed by node id) | `:5275-5278`; restore `:5420-5422` (drops unknown ids) | key substituted | 353 KB |
| S13 | activation sidecar `main.msgpack.activations.json`, `entries` keyed by node id (`voltage`, `last_spike_time`, `excitability`, `timestamp`) | `activation_persistence.py:120-136`; `restore()` skips entries whose node is not in the graph | keys re-keyed, values byte-equal, `saved_at`/`timestep` kept so the decay clock continues; **without this the repaired wants lose their saved voltage and excitability** | all 182 present in the analysis copy of the sidecar |

**Not id-bearing** (no rewrite): `synapse_confirmation_history` (keyed by *synapse* id, `:5216-5219`, `:2050`, restore `:5613-5617`), `reward_history` (`:3891`), `telemetry` (`:5240`), `config`, `he_last_fired_step` (keyed by hyperedge id), `he_prediction_window_fired` (skipped on restore `:5062-5070`), the counters.

**Derived, NOT persisted — the tool must never write them; the canonical restore rebuilds them:** `_outgoing`/`_incoming` (`:1948-1949`, rebuilt `:5454-5457`), `_node_hyperedges` (`:1950`, `:5486-5487`), the `_recent_spikes` deque objects (`:1951`, `:5416`), the dirty sets (`_dirty_nodes/_synapses/_hyperedges`, `:1952`, `:2034`; used only by INCREMENTAL serialization `:5287-5308`, never written into a FULL checkpoint).

**Other references to a want id, inside the checkpoint set but outside the structures above, are not expected — and the tool refuses to guess (the census, §2.4).** The vdb has no want key (0 of 182 [derived]); a vdb *content* string that merely quotes an id is text, not a reference, and is counted but never rewritten.

### 2.4 [R2·P401] The canonical path (LAW 3) — and why the tool is a value-granular rewrite
- **No canonical re-key exists.** A scan of every `*.py` in the repo for `rename|rekey|repoint|remap|relabel|change_id|move_node|merge_nodes|swap_node` finds only `remove_node` (`neuro_foundation.py:1955`), which **deletes**: it cascades over every incident synapse (`:1961-1966`) and strips hyperedge membership (`:1968-1979`). `create_synapse` builds a fresh `Synapse` (weight, peak, ages reset, `:2020-2030`). Remove+create would lose exactly what the amended acceptance says must carry.
- **The Graph route (restore → mutate → checkpoint) was considered and rejected.** It also needs a hand-written re-key of the in-memory structures, and restore normalizes state (drops stale delay-buffer entries `:5427-5435`, drops predictions with missing endpoints `:5563-5570`, skips window-fired `:5062-5070`, re-caps spike deques `:5420-5422`), so non-want bytes would change and acceptance 3 could not hold; it also costs ≈ 3.6 GiB (`summary-laptop.json`).
- **Chosen:** a **value-granular rewrite of the decoded persisted structures** in a scratch script, using the canonical serialization settings (`Packer(use_bin_type=True)`, `write_checkpoint :5013-5021`) and reading with `Unpacker.tell()/skip()` as `restore` does (`:5050-5061`). Every value that does not contain an old id is copied as raw bytes. Node ids are 25-character strings (both old and new ids), so an id substitution **does not change any encoded length**; only the 118 nodes' `want_text` strings change length. The derived indexes are never written; **the canonical `Graph.restore` of the output is the verifier** (§5, V11): it rebuilds `_outgoing/_incoming/_node_hyperedges` from the persisted structures, and the verifier compares them with the pre-repair per-id figures. So derived-index consistency is guaranteed by the canonical path, not by hand-patching.
- **Unenumerated-site guard (the raw-occurrence census).** Before writing, scan the raw bytes of all six checkpoint-set files for each old id (its 25-char string) and classify each hit: (i) a whole string at a position in S1–S13 → rewritten; (ii) a whole string anywhere else → **STOP**, report as an unenumerated reference site to the Executive; (iii) embedded in longer text (vdb/content) → counted, not rewritten. After the rewrite: zero whole-string old ids remain, and the number of substitutions equals the census (i) count. (The scan matches decoded strings, not raw byte patterns, to avoid false hits on UTF-8 continuation bytes.)

### 2.5 The mapping (off-path, hash-verified, reported)
`id-map-<UTC>.json` in the backup directory (never the repo): `[{old_id, new_id, old_len, new_len, old_sha16, new_sha16, class}]`, plus its own `sha256` recorded in the report. Verified before use (Phase 1 and again in Phase 2): every `new_id` recomputed from the new text by `_want_id`; the mapping is injective; no `new_id` ∈ the pre-repair node-id set (§2.6); the file is re-read and re-hashed. The report lists the mapping's hash to Chief.

### 2.6 Collision rule (per P402: never merge automatically)
A repair is **dropped, both parties left unchanged and LISTED to the Executive via Chief**, when its `new_id` (i) already exists **as any node** in the pre-repair graph (another want, or a non-want) or (ii) is produced by two repairs (same `X`). The check is against the **pre-repair** id set, so a chain (`A → B` where `B` is the old id of another node in the repair set) is also a collision: no ordering games, both untouched. Nothing is merged, deleted or re-tagged. After the pass, ids are asserted unique.

### 2.7 Protection and the wants ruling
`_is_identity_protected` (`neuro_foundation.py:3551-3572`) is keyed on the **metadata flag** (`constitutional`, `provenance` ending `_authored`), not on an id list, and `_prune_synapses` skips by the same test (`:3517-3519`). The repaired node keeps `provenance = cc_authored`, so **the protected count stays 183 (182 wants + the constitutional node)** and no synapse becomes prunable. Nothing is de-flagged or re-tagged. The old id ceases to exist — that is the approved separation (P402) — and the mapping keeps it traceable.
**Rim, needs an explicit yes (§7 Q8).** 131 synapses connect the Choice Clause node and S wants. The rim **node's bytes do not change**, but each such synapse's want-side endpoint field is re-pointed. The original brief said nothing may touch "the Choice Clause node or its rim synapses"; the amended acceptance says every synapse is re-pointed. I read the two together as: rim synapses are carried whole, only the want-side id changes — and I ask, not decide.

## 3. The mechanical separation rule (unchanged except G5)

**Scope** `S`: `kind == "want"`, `provenance == "cc_authored"`, `len(want_text) > WANT_MAX_CHARS` (import the constant). Expected **|S| = 118** [derived]. The 64 wants of ≤ 600 chars are never touched.

Gates, in order; the first failure leaves the node UNCHANGED and LISTED with its reason code. `T = want_text`, `C = vdb.content[source_node]`.
- **G1** the source node is a non-want, non-constitutional, non-`*_authored` graph node and `C` exists (118/118 [derived]).
- **G2** `T` is exactly an old-parse span of `C`: `C.count(T) == 1` at `p`; `C[:p].rstrip()` ends with `[WANT]` (outer opener at `i`); `C[p+len(T):].lstrip()` starts with `[/WANT]`.
- **G3** the outer opener is documentation **by the parser's own guard**: `i > 0 and C[i-1] == "`"` (`:1549`). Proxy in derived data: `T[0] == "`"`, 83 of 118. If not, the head could itself be a real want whose closing tag was forgotten — not mechanically decidable → unchanged.
- **G4** a well-formed inner pair: `k = T.rfind("[WANT]")` exists, `k == 0 or T[k-1] != "`"`, `X_raw = T[k+6:]`, `_WANT_RE.fullmatch("[WANT]" + X_raw + "[/WANT]")` matches (the module's own bound), `X = X_raw.strip()` non-empty. Post-conditions asserted per node: `X in T`, `T.endswith(X)`, `0 < len(X) <= 600`, no marker inside.
- **[R2·P402] G5 the new id is free** (§2.6): `_want_id(X)` is not in the pre-repair node-id set and no other repair yields it.

**Failure modes (unchanged).** (1) Parser-well-formed ≠ author-intended: a quoted example (`"[WANT]like this[/WANT]"` in a listing) is a well-formed pair and will be selected; mitigation is the frozen list (§5). (2) Head-genuine unclosed wants sit in the non-backtick class; G3 lists them, never guesses. (3) Production may not emit `X` (E1) — harmless.

**Class table (118 oversized, [derived], `want-rows-laptop.json`)**

| Class | Definition | Count | Outcome |
|---|---|---|---|
| A | `T[0]=="`"` and contains `[WANT]` | **67** | candidates; repaired iff G1–G5 pass (upper bound 67) |
| B | `T[0]=="`"`, no `[WANT]` | **16** | no inner opener → UNCHANGED, LISTED (certain from flags) |
| C | not backtick-led, contains `[WANT]` | **35** | fail G3 (assuming the proxy) → UNCHANGED, LISTED |

Repaired ≤ 67; left unchanged ≥ 51 (16 certain + 35 proxy-dependent) plus any class-A node failing G1–G5. **The exact split is not knowable without the texts** (§8). All 118 sources are in the graph, conversational, unprotected (49 distinct) [derived]. Out of scope: the 64 short wants are mostly code/test-fixture fragments (§7 Q2).

## 4. Where the swallowed prose goes (unchanged)

It never leaves the graph: `T` is a contiguous substring of the source conversation node's content `C` (G2), that node is an ordinary `cc:conv::` node (no provenance, not protected) and the vdb is untouched, so the removed prefix and `X` both remain verbatim there. A new "prose" node would be an unprotected orphan the next `_collect_orphan_nodes` may collect, i.e. the prose would be deleted by the graph's own hygiene. Acceptance 5 is checked as **provenance** (V9). The source→want synapse is carried to the new id like every other synapse (§2.3 S3).

## 5. [R2·P402] Execution procedure and verification (offline, copy-first, separate authority)

**Tool:** a new scratch script (suggested `scripts/want_text_repair.py`; no protected or vendored file), built in a later code lane with synthetic tests. **Writer:** `checkpoint_guardian.atomic_file_write` (`:343-363`; tmp + `os.replace`; used by the #379 precedent `cleanup_cc_tool_noise.py:127-139`), with a writer function that performs the rewrite; **not** `Graph.checkpoint()` (§2.4). The sidecar is written with the **same JSON layout as `write_state`** (`json.dump(captured, f)`, `activation_persistence.py:138-169`) but through `atomic_file_write` — `write_state` itself writes in place (`open(path,"w")`), which can tear.

### 5.1 Preflight
`MemAvailable ≥ 8 GiB`, load < 6, one loader at a time; rewrite/classify passes under `systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0` (raw main.msgpack ≈ 231 MB [derived]; values are decoded one at a time); V11 (canonical restore) under 6G (measured restore ≈ 3.6 GiB), one restore at a time. `NG_EMBED_REMOTE` unset, `HF_HUB_OFFLINE=1` (no embedding is done).

### 5.2 Phase 1 — dry-run on a COPY (nothing live is touched)
1. Copy the six checkpoint files (`main.msgpack`, `vectors.msgpack`, `.activations.json`, `.guard_state.json`, `.manifest.json`, `commons.msgpack`) to `/home/josh/backups/z12-want-text-repair-<UTC>/copy/`; record and compare sha256.
2. **Classify (read-only):** stream `nodes`; apply G1–G5 (streaming `C` from `vectors.msgpack`, embeddings skipped). Emit `repair-list.json` and the LISTED table (id, reason, len, first/last 100 chars).
3. **Freeze** (recommended): Chief/Exec review each proposed `X`, strike any id; a struck id becomes UNCHANGED. The tool then applies the frozen list mechanically and refuses anything not on it.
4. **Census** (§2.4) → STOP on any unenumerated whole-string site.
5. **Fidelity (V13)** before any edit: for every value that will be re-encoded — the ≤ 118 nodes, ≤ 1,735 pre-nodes, ~82 K synapses, the hyperedges and log entries containing an old id — `pack(unpack(raw)) == raw`. **Any failure stops the whole run** (a partial re-point would leave dangling endpoints); reported, not worked around.
6. **Rewrite to a tmp file** via `atomic_file_write` (main) and to a tmp sidecar; every other value is a raw slice. Then run V1–V13 on `copy_before` vs `copy_after`; write `report-<UTC>.json/.md` (contains want text → backup dir, never the repo).

### 5.3 The verifier (all mechanical; a failing check → no live write)

| # | Check | How |
|---|---|---|
| V1 | counts | 182 wants before and after; total node count equal; protected census 183 (182 `cc_authored` + 1 constitutional); constitutional node untouched |
| V2 | ids | each `new_id == _want_id(new_text)`; old ids absent, new ids present; mapping injective; post ids == (pre ids − old) ∪ new; each new id sits at its old id's position in the node map |
| V3 | node carry-over | for each repaired node, `decode(post) == decode(pre)` with only `node_id`, `metadata.want_text` (→ `X`) and its own `pred_weights` keys mapped; **`poincare_dir` bytes equal**; `provenance`/`want_state`/`source_node`/`creation_mode`/`creation_time` equal |
| V4 | synapses | set of synapse ids equal (138,753 in the copy; re-counted); for every synapse, all fields equal except endpoints, which equal the mapped pre-repair endpoints; per-old-id vs per-new-id incident counts (out/in) equal — **no synapse lost**; rim incident count unchanged (4,127 [derived]), with exactly the rim↔S ones (131) differing in the want-side endpoint |
| V5 | hyperedges | ids equal; `member_nodes`/`member_weights`/`output_targets` equal under the mapping (list order preserved); other fields equal; same for archived |
| V6 | other id sites | S4, S6–S12: decoded equality modulo the mapping; every other top-level value **byte-equal** |
| V7 | non-want nodes | raw bytes equal, **except** nodes whose only difference is `pred_weights` key ids mapped (S2) — proven by an independent *diff walker* that compares the two decoded trees and permits only `(old id → new id)` differences at string positions, asserting each pair is in the mapping (independent of the rewrite code). For every re-encoded non-node value additionally `len(new) == len(old)` |
| V8 | sidecar & files | sidecar keys == mapped key set, per-entry values byte-equal, `saved_at`/`timestep`/`version` equal; sha256 equal for `vectors.msgpack`, `guard_state`, `commons`; `manifest` counts equal the post counts (§5.4) |
| V9 | text and prose | `X in T and T.endswith(X)` per repaired id; prose provenance (§4): source node ∈ graph, non-want, unprotected, vdb bytes unchanged and containing `T` |
| V10 | census | zero whole-string old ids remain in the output; substitutions == census (i) |
| V11 | canonical read path | fresh process, `Graph().restore(before)` → record per-want `len(_outgoing)`, `len(_incoming)`, `_node_hyperedges`, `render_wants` block; free it; `restore(after)` → `_outgoing/_incoming` sizes for each new id equal the old id's, every synapse endpoint and hyperedge member resolves, no dangling id, `_node_hyperedges` equal under the mapping; `render_wants` differs only for repaired entries (ordering by `creation_time` unchanged) |
| V12 | idempotence | re-running the tool on the output finds no `len > 600` it can repair → zero edits; collisions listed, none merged |
| **T6** | no new behaviour on the next pulse | stub graph (fake nodes + `_step_lock`, recording `create_node`/`create_synapse`, the `test_cc_want_bounds.py` pattern) built from `(id, kind, creation_mode, want_text)` plus stub vdb content; call the **real** `surface_wants` and the **real** `surface_wants_for_graph` on before and after. Pass iff `minted(after) ⊆ minted(before)` and `minted(before) − minted(after)` ⊆ {new ids} — i.e. the re-parse now lands on the repaired node and dedupes. Report `|minted|` for each function and any pre-existing mint (§1) |
| V13 | encoder fidelity | as 5.2 step 5 |

**Unit tests for the build lane (synthetic fixtures, no real data):** BT + nested well-formed; BT no opener; non-BT; inner opener backtick-preceded; suffix > 600; source missing; `T` repeated in `C`; each collision shape (existing id, two repairs one text, chain); a want↔want synapse with both ends repaired; a rim synapse; `pred_weights` in a non-want node; hyperedge list order; sidecar re-key; unenumerated-site STOP; fidelity STOP; `_want_id` equals production; idempotence.

### 5.4 Phase 2 — live apply (only after Phase 1 passes and Chief/Exec authorise)
1. **Daemon down** (unit/process name to be confirmed by Chief; `cc-ng-daemon.py` is named in `cleanup_cc_tool_noise.py`); confirm nothing holds the checkpoint and no `neurograph_rpc.py`/CC-daemon PID survives.
2. **Full backup first:** the six files to `/home/josh/backups/z12-want-text-repair-<UTC>/backup/`, sha256 recorded, backup re-hashed and read independently (counts vs manifest). The daemon's `generations/` ring is left alone (hardlinks keep old inodes, `checkpoint_guardian.py:575-628`).
3. **Re-run the same tool on the live bytes; never copy the Phase-1 output over live** (a copy taken under a running daemon is stale and installing it would revert everything learned since). The live run recomputes the mapping and **refuses to proceed unless it equals the frozen, hash-verified Phase-1 mapping** (same old ids, new ids, `old_sha16`) or the want-id set changed.
4. Inside the `atomic_file_write` writer functions run V1–V10 and V13 against (backup, tmp) for **both** `main.msgpack` and the sidecar; a failure raises, the final files stay untouched and the tmps are removed (`:354-363`). Then `os.replace` **main first, sidecar second**. The two files cannot swap atomically together; if the process dies between them (main re-keyed, sidecar old), the tool's idempotent re-run detects that state from the mapping and completes the sidecar only; otherwise roll back both from the backup.
5. **Manifest and guard state: leave them.** Counts are unchanged and `SaveGate.permit` keys off `guardian_nodes` then `nodes` (`:451`) → #743 is not triggered; the report records manifest counts == post counts. (The precedent scripts refresh the manifest only when counts change.)
6. Do **not** start the daemon (S4 owns that). Fresh-process re-verification of live vs backup, then the receipt.
7. **First two daemon autosave cycles after S4 start:** re-check want count 182, repaired ids/texts unchanged, minted set == the T6 prediction; report the observation window, not a single-point "done".

**Rollback:** daemon down; `atomic_file_write` the backed-up `main.msgpack` and sidecar back (sha256 == recorded). Only those two files change.

## 6. [R2·P401] Ripple and references outside the graph

| Consumer / location | Refers to want ids or text? | Consequence of the repair |
|---|---|---|
| `render_wants` (`cc_ng_organism.py:1575-1610`; `cc_ng_host.py:1141-1144`) | text; orders by `creation_time` | 28 of the 40 newest wants are in the 118 [derived]; top-40 block ≈ 18.4 KB now; shortens repaired entries; order unchanged |
| identity block `render_constitutional_core` | `core_text` only | none |
| `_pith_node_raw_text` (`:4640-4652`) | text | shorter provider context for repaired ids |
| `cc_stamp_missing_geometry` (`:3388-3484`) | text, only when `poincare_dir` missing | none (carried byte-identical); Exec Q3 |
| `surface_wants` / `surface_wants_for_graph` | dedupe by id | §1 (dedupe now lands on repaired nodes; the legacy twin is a separate lane) |
| **activation sidecar** | keyed by node id | re-keyed (S13) |
| `guardian_nodes` / manifest / guard state (#743) | counts only | none; equal counts |
| protected census (`plan-scratch/protected_census.py:53-56`) | records id, `want_text_len`, `want_text_sha16` | protected count 183 unchanged (flag-based); a post-repair run compares by joining through the mapping |
| want-hub d / competition plan (d) | ids, degrees, synapses | any analysis keyed by old want ids must be re-keyed through the mapping; synapse and degree figures carry over (endpoints follow) |
| analysis JSONs (`want-rows-laptop.json`, census outputs) | keyed by old ids | derived, not live; superseded — join through the mapping |
| vault docs (docs worktree) | scan of `cc:want::` | **none of the 118 candidate ids is cited.** Two vault files cite ids from the other 64 (`punchlist/open/neurograph.md` 1 id; `zone-return-011-packet374-f3-origin-trace.md` 2 ids); the remaining hits are the `cc:want::` prefix or ids that are not among the 182. Nothing points at a vanished id. Scope: the docs worktree only |
| Quest tracker | `grep` of `quest-tracker` for `cc:want::` | no hits. Other board/Quest records **[unverified]** |
| `#799 export` | the fold names it | I could not locate what it refers to **[unverified]** — Chief to name |
| tract content | frames carry exported node ids | history is immutable and is never opened or rewritten; see the next row |
| **callosum export/merge** (`cc_topology_export.py:284-295`, `:351-416`; `cc_topology_merge.py:373-374`, membership ledger) | the exporter sends every CC-provenance node whose id is **not in the peer's membership ledger** (`exclude_ids`, `:371-373`); `want_text` rides `_portable_metadata`; wants have no vdb embedding so each is logged as `missing_embedding_DEFECT` (`:400-407`) | **after the re-key the 118 new ids are absent from the ledger, so they will be re-sent as new nodes, and the receiver (idempotent by id, `:373-374`) will admit them next to its old bloated copies → duplicate protected wants on the peer, which the rules forbid deleting.** A committed ledger snapshot (`ng_topology/laptop_cc_membership.json`, 2026-09-13) lists all 182 want ids (118 of them candidates); whether it equals the live peer state, and where its live counterpart is, is **[unverified]** |
| `neurograph_rpc.py:3240` (Syl's path) | reads `want_text` | unaffected (laptop-only) |

## 7. Questions for the Executive (via Chief) — the ruled ones have left the list

*Ruled and removed:* the id follows the text; the collision rule (both unchanged, listed, never merged).
1. **Parser code lane (LAW 4).** Retire the unbounded `surface_wants_for_graph` (delegate to the bounded, guarded function, or drop the `_deposit` call); decide `neurograph_rpc.py:4902`; optionally extract the one-line id helper (§2.1). Recommended as a hard prerequisite of the S4 start, not of the offline repair.
2. **Scope of LEFT UNCHANGED.** Are ≥ 51 untouched oversized wants acceptable (16 with no inner opener; the 35 non-backtick; any class-A node failing G1–G5), plus the 64 short fragments? Alternative: Exec hand-specifies a per-id span list which the tool applies mechanically.
3. **`poincare_dir`.** Carried byte-identical here (recommended). Leave, or clear on repaired ids so the daemon's own `cc_stamp_missing_geometry` re-derives it from the new text, or recompute in the tool.
4. **Peer copies.** Given §6 (the exporter re-sends the new ids as new nodes and the peer will hold both), apply the same frozen mapping to the peer copy offline, or hold the export until it is applied? Which side owns the ledger?
5. **Frozen-list review** (§5.2 step 3): required or optional? Who signs?
6. **Slot, unit, retention.** Which daemon unit to stop, which slot (one loader, load < 6, `MemAvailable ≥ 8 GiB`), how long to keep the backup and the report (which contains want text).
7. **The extra inputs:** `want-rows-laptop.json` (plan-001), and now the checker's sidecar copy and the membership-ledger snapshot (§0).
8. **Rim synapses (§2.7).** Confirm that re-pointing the want-side endpoint of the 131 rim↔want synapses is within "every synapse re-pointed", the rim node itself staying byte-identical.
9. **"Byte-identical apart from re-pointing".** Confirm the carve-out also covers the `pred_weights` keys inside ~1.7 K non-want nodes (S2), verified by the independent diff walker rather than by raw bytes.

## 8. What this plan still cannot know without the texts

- **The exact repaired / left-unchanged split** (bounds only: repaired ≤ 67, left ≥ 51). The dry run produces it; the plan does not estimate it.
- How many candidates collide (§2.6), how many census hits fall in each site (S4, S7, S10 especially), and how many values will actually be re-encoded.
- Whether `surface_wants_for_graph` and `surface_wants` actually run on the laptop daemon; T6 shows what they would do, not whether they did.
- Whether the live laptop checkpoint still equals the 09-23 analysis copy (Phase 2 re-derives everything from the live bytes), and the daemon unit name and autosave interval.
- Whether the analysis copy's vdb or the Commons file holds a want id as a whole string (the census settles it).

**The LEFT list (id, reason code, length, first/last 100 chars) and the collision list go to the Executive via Chief.**

**Flagged for the punchlist (not mine to file):** (1) unbounded `surface_wants_for_graph`, live per turn (`cc_ng_host.py:696-704`); (2) `neurograph_rpc.py:4902` unbounded regex; (3) `_autosave_loop` bundles drain, probation, `surface_wants` and `generate_emergent_want` in one DEBUG-swallowed `try` (`cc_ng_host.py:1517-1534`); (4) the 64 short code/fixture-fragment wants; (5) dedupe-by-id-only for text-hashed ids; (6) `ActivationPersistence.write_state` writes the sidecar in place, not atomically (`activation_persistence.py:162-164`).
