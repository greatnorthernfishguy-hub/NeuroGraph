```
---- Changelog ----
[2026-09-29] Claude Code (claude-sonnet-5-5, worker seat) — plan-003: FOLD of plan-002 (55fdb2b)
What: (R3·P406) the separation rule IS the parser's structural-legitimacy test (Exec P406, Josh): §3 replaced
      (G1-G4 and the backtick proxy removed); the 600-char bound and every `<= WANT_MAX_CHARS` check removed;
      ONE shared function (#810's `parse_wants` / `want_id_for_text`) called by the parser and the repair; classes
      recounted; genuine long wants handled; the SEQUENCING RULING recorded as RULED (§0.1, §5.1); §7/§8 updated.
Why:  assignments/plan-want-text-repair-118-p406.md (Exec P406 via Chief-003, docs 6b8d0792; sequencing ruling docs b2e9e7bd).
How:  read-only. New reading: the #810 build plan on origin (`cc-laptop-want-legitimacy-810-20260930` @ 2ab4a851,
      a PLAN document, no code yet), the laptop daemon's want call (docs branch 155343e4), plus re-derivation from the
      analysis JSON. Passages changed by P406 are marked [R3·P406]; unmarked passages are carried from plan-002 unchanged.
      One CORRECTION to plan-001/002 is marked [R3·CORRECTION] in §1. plan-001/002 stay in place.
-------------------
```

# plan-003 — the 118-want TEXT repair: the separation rule is the parser's legitimacy test

Lane `z12-s3-restore-bundle-20260929` (dispatch #10745) · branch `cc-laptop-want-text-repair-20260930` · previous plan `55fdb2b15531741868d457408630c27412e43ae8`
Status: **PLAN.** A pair (cross-family + law enforcer) reviews this BEFORE any build or offline execution. No build here.

## 0. What I did and did not do

- **No** graph or checkpoint load, no build, no daemon/unit start or stop, no write to any checkpoint; the live tract was never opened; nothing under Syl's directories was opened or listed.
- Derived data (`/home/josh/backups/z12-s3-restore-bundle-20260929T222529Z/analysis-scratch/`): `summary-laptop.json`, `probe-laptop.json`, and — flagged since plan-001 and still flagged — `want-rows-laptop.json`, plus the two id-only inputs flagged in plan-002 §0 (a copy of the laptop activation sidecar's key set; the 2026-09-13 membership-ledger snapshot).
- **New this turn:** (a) the #810 build plan, read with `git show origin/cc-laptop-want-legitimacy-810-20260930:handoffs/z12-want-legitimacy-810/returns/build-001.md` (commit `2ab4a851`, **a plan document only; no code exists yet**, so every statement about the function below is "as planned at `2ab4a851`"); (b) `scripts/cc-ng-daemon.py` lines 2088-2120 on the docs branch `cc-laptop-daemon-recall-756-20260930` (HEAD `155343e4`), read-only.
- Tags: **[code]** read at the cited line · **[derived]** from the analysis JSON · **[#810-plan]** as planned in build-001 · **[unverified]** needs data or a running system, with the check that settles it.

### 0.1 [R3·P406] Ruled — not questions
1. **A want** is the text between a real `[WANT]` opener and its paired `[/WANT]` closer, **no length limit**. The parser checks only structural **legitimacy**: the tag is not a mention (not in inline code, not in a fenced block, not in a quoted/escaped span), the pair contains no other live marker, and the closer pairs with the nearest opener. A failing marker is skipped and logged at INFO with its reason. **That legitimacy test is this repair's separation rule.**
2. **The id follows the text** (P402): new id = `cc:want::` + `sha1(text)[:16]`, by the shared expression; every reference carries to the new id; collisions leave both unchanged and are listed, never merged.
3. **Sequencing (Exec via Chief-003, docs b2e9e7bd):** the #801 repair may be **written and pair-reviewed now** but must **not be applied live** until #810's shared legitimacy function is fixed, reviewed/paired **and is the code the laptop daemon runs**. Apply order: **#810 merged and deployed on the laptop daemon → #801 applied live (daemon down, full backup first) → S4.** The offline dry-run on the COPY may use the fixed function from the #810 branch. Reason: the repaired ids must equal what the running parser mints on the next re-parse (S5 drains the tract, including pre-fix turns); if the daemon still ran the old capped parser, ids could diverge and the dedupe would break.

## 1. Where the mis-parse happens, and whether new deposits reproduce it

**Origin [code].** Until 2026-09-16 the bucket used an unbounded `\[WANT\](.*?)\[/WANT\]`; a merely-mentioned `[WANT]` opened a span running to the next `[/WANT]` (changelog `cc_ng_organism.py:352-368`: 182 wants, 118 over 600, largest 136,449). The 09-16 fix bounded the pattern (`WANT_MAX_CHARS = 600`, `_WANT_RE :1512-1514`) and added two guards (`:1549`, `:1556`); it repaired nothing, and the bound itself **silently drops a genuine > 600-character want** and `render_wants` clamps every want to 600 (`:1604`) — the LAW 7 violations P406 names. The fix commit `d75efeb` is an ancestor of the laptop checkpoint's manifest commit `18a090e`, so the 09-23 checkpoint's 182 wants are all pre-fix survivors [derived + git].

**[R3·P406] What #810 changes [#810-plan].** `parse_wants(content)` (pure) replaces `_WANT_RE`; `surface_wants` calls it and `want_id_for_text`; the id expression `"cc:want::" + sha1(text.encode("utf-8")).hexdigest()[:16]` moves out of the loop (currently inline at `:1558`) into that one named function. `WANT_MAX_CHARS` **stays defined** until #810's second turn (`render_wants` still reads it); `render_wants` is not touched by #810's first turn (P408 retires the standing block instead).

**[R3·CORRECTION] The legacy twin is not on the laptop daemon's path.** Plan-001/002 said `surface_wants_for_graph` runs "after every turn deposit". That is true of **`cc_ng_host.py:696-704`** (the NeuroGraph-side host), but the **laptop daemon** — `scripts/cc-ng-daemon.py`, docs branch `155343e4` — calls only `surface_wants` (`:2116`, inside the same DEBUG-swallowing `try` as `drain_ingest_tract`, `:2088-2119`) and has no reference to `surface_wants_for_graph` [code]. The twin (`cc_ng_organism.py:1128-1195`: unbounded `(.*?)`, no guards, `want::` ids) is **parked by #810 as #755** and is not changed. Consequence for this plan: on the laptop the twin is not a re-minting path; on any graph hosted by `cc_ng_host.py` it still is. If it were ever run over these sources it would mint `want::` copies of the very mis-parsed spans this repair separates, so the repair's durability depends on the twin staying uncalled on the repaired graph (§7 Q1). No `want::`-prefixed node exists in the 09-23 checkpoint [derived].

**Would a re-parse undo or duplicate a repair? No, for repaired nodes.** The repaired node's id is `want_id_for_text(X)`, which is exactly what `surface_wants` mints for the same legitimate span, so `if want_id in graph.nodes: continue` (`:1559-1560`) lands on the repaired node. That holds **only if the daemon runs the same function** — the sequencing ruling (§0.1, §5.1).
What remains: collisions (§2.6); wants left unchanged keep their old ids (the new parser can never re-mint their old spans as one want); and the first pulse after #810 deploys will mint **other** legitimate wants that were never minted before (long ones the cap dropped; a genuine want an unbounded parse swallowed) — pre-existing behaviour of #810, measured by T6 (§5.3) and reported so the Executive knows what the first S4 pulse will add.

## 2. The id follows the text — reference sites (carried from plan-002 §2, with P406 edits)

### 2.1 [R3·P406] The id expression
There is exactly one implementation: `want_id_for_text` from #810 (LAW 3/4). The repair tool **imports it; it contains no copy**. A test still asserts equality against production: the real (#810) `surface_wants` on a stub graph mints, for a battery of texts (ASCII, multi-byte UTF-8, edge whitespace, a 5,000-character text, a real-shaped sentence), the same ids `want_id_for_text` returns.

### 2.2 What changes per repaired node
The dict key and `node_id` (old → new id); `metadata["want_text"]` (→ `X`, **any length**); and every reference site below. Unchanged: `provenance` (`cc_authored`), `kind`, `want_state`, `source_node`, `creation_mode`, `creation_time`, `poincare_dir` (carried byte-identical; Exec Q3 stays), every dynamics field, every synapse/hyperedge field other than the id.

### 2.3 Every persisted reference site (the rewrite set) — [derived] volumes are upper bounds over the 118
| # | Structure | Where [code] | Volume [derived] |
|---|---|---|---|
| S1 | `nodes` key + `node_id` | `neuro_foundation.py:5089-5090`, `:5188`; restore `:5387-5416` | ≤ 118 |
| S2 | `Node.pred_weights` (keyed by post-node id inside the PRE node, incl. non-want nodes) | `:673`, `:3463`/`:3473`, `:5106`, `:5407` | ≤ 1,735 nodes (1,561 non-want) |
| S3 | `synapses` `pre_node_id`/`post_node_id` (synapse ids are UUIDs, `:698`; none change) | `:2020-2034`, `:5183`/`:5194`, restore `:5449-5457` | **81,999** of 138,753 touch S (13,203 want↔want; **131** rim↔want) |
| S4 | `synapse.metadata["expected_target"]` | written `:3308` | unknown |
| S5 | `hyperedges`/`archived_hyperedges`: `member_nodes`, `member_weights`, `output_targets` | `:5112-5136`, `:5192-5199`, restore `:5460-5517` | 48 hyperedges, 145 memberships |
| S6-S12 | `active_predictions`, `prediction_outcomes` (**not validated on restore**, `:5588-5608`), `he_active_predictions`, `he_output_candidates`, `novel_sequence_log`, `delay_buffer`, `recent_spikes` | `:5201-5278`, restore `:5427-5435`, `:5557-5647` | occurrences unknown until the census |
| S13 | activation sidecar `entries` keyed by node id (restore skips unknown ids) | `activation_persistence.py:120-136`, `restore()` | all 182 present in the copy |

Not id-bearing: `synapse_confirmation_history` (keyed by synapse id), `reward_history`, `telemetry`, `config`, `he_last_fired_step`. **Derived, never written:** `_outgoing/_incoming` (`:5454-5457`), `_node_hyperedges` (`:5486-5487`), the `_recent_spikes` deques, the dirty sets — rebuilt by the canonical restore.

### 2.4 The canonical path (LAW 3)
No canonical re-key exists (repo-wide scan: only `remove_node`, `:1955`, which cascades and deletes; `create_synapse` resets weight/ages, `:2020-2030`). A Graph round-trip would normalize non-want state at restore (`:5427-5435`, `:5563-5570`) and cannot keep bytes identical. **Chosen:** a value-granular rewrite of the decoded persisted structures with the canonical packer settings (`write_checkpoint :5013-5021`); untouched values are raw slices; ids are 25 characters both before and after, so id substitutions never change an encoded length (only the repaired `want_text` strings do — and they may now be longer or shorter than 600); the derived indexes are rebuilt by the **canonical `Graph.restore` of the output, which is the verifier** (V11). A **raw-occurrence census** (whole-string occurrences of each old id across the six checkpoint-set files) STOPS the run on any occurrence outside S1-S13.

### 2.5 The mapping and 2.6 the collision rule (unchanged)
`id-map-<UTC>.json` off-path, hash-verified before every use, every `new_id` recomputed by `want_id_for_text`, injective. A repair is **dropped, both parties left unchanged and LISTED via Chief**, when its `new_id` already exists as any node in the pre-repair graph (checked against the *pre-repair* id set, so chains collide too) or two repairs yield the same text. Nothing merged.

### 2.7 Protection and the rim (unchanged)
`_is_identity_protected` keys on the metadata flag (`neuro_foundation.py:3551-3572`); `provenance` stays `cc_authored`, so the protected count stays 183 (182 wants + the constitutional node). The Choice Clause node's bytes do not change; its 131 want-side synapse endpoints are re-pointed — asked in §7 Q7, not decided.

## 3. [R3·P406] The separation rule IS the legitimacy test

### 3.1 What the repair relies on from the shared function (the contract; #810 defines it)
The repair calls the function #810 ships, on each node's **source conversation content**, and takes whatever it returns. Planned shape [#810-plan `build-001.md` §2]:
`parse_wants(content) -> WantParse(wants: tuple[WantSpan(text, want_id, open_start, close_end)], skipped: tuple[SkippedMarker(marker, start, reason)])` and `want_id_for_text(text)`.
Guarantees the repair needs (any departure in the shipped code invalidates §3.3):
- **G-pure:** a pure, deterministic function of the content string (no graph, vdb, clock, env, logging).
- **G-text:** each `WantSpan.text` is exactly what `surface_wants` stores and hashes: `.strip()` of the inner text, **unbounded and never truncated**; `want_id == want_id_for_text(text)`.
- **G-offsets:** `open_start` is the char offset of the live `[WANT]` and `close_end` the offset just past its paired `[/WANT]`, into the same string that was passed in (so `content[open_start+6 : close_end-7].strip() == text`).
- **G-reasons:** every marker that is not part of a returned pair appears in `skipped` with a reason from a closed set (planned: `in_fence`, `in_code_span`, `code_adjacent`, `escaped`, `quoted`, `opener_unclosed`, `empty_pair`, `closer_without_opener`); never silent.
- **G-pair:** the closer pairs with the **nearest** live opener; a returned want contains **no live marker**.
- **G-single:** the parser and the repair call the same function object (same module path), not copies.
- **G-fingerprint:** #810's golden test file (fixed inputs → fixed outputs) is what the apply gate re-runs (§5.1 P3).
The function's home is the #810 build's decision; I only note that the repair tool imports it under the memory cap, and that #810 currently places it in `cc_ng_organism.py` (a large module).

### 3.2 Scope
`S` = the approved population: `kind == "want"`, `provenance == "cc_authored"`, `len(want_text) > 600` (the tool takes the threshold as a required, reported parameter `--scope-min-len 600`, since `WANT_MAX_CHARS` is no longer a parser constant). Expected **|S| = 118** [derived]. The 64 shorter wants are **never touched** (§7 Q3). The old checks that assumed a bound (`0 < len(new_text) <= WANT_MAX_CHARS`, the `_WANT_RE.fullmatch` gate) are **removed**.

### 3.3 The algorithm, per node `N` in `S` (`T = want_text`, `C = vdb.content[source_node]`)
1. **Source (A0).** `source_node` is a graph node that is not a want, not constitutional, not `*_authored`, and `C` exists; else `source_missing` → unchanged, listed.
2. **Anchor (A1)** — reproduce the old unbounded parse of `N`: `C.count(T) == 1` at `p`; `C[:p].rstrip()` ends with `[WANT]`, the outer opener at `i`; after `T`, skipping whitespace, `[/WANT]` follows at `j`; `closer_end = j + 6 + 1`. Else `anchor_failed` → unchanged, listed (T came from another writer or is repeated).
3. **Parse.** `wp = parse_wants(C)` — once per distinct source (49 distinct [derived]; cached).
4. **Decide by the pair that closes at `j`** (`w` = the entry of `wp.wants` with `w.close_end == closer_end`):
   - **GENUINE** — `w` exists, `w.open_start == i`, `w.text == T`: the old span **is** a legitimate want (a real long want that the old parse got right). **UNCHANGED.** Assert `want_id_for_text(T) == N.id` (else `id_mismatch`, listed).
   - **SEPARATE** — `w` exists, `w.open_start > i`: the legitimate want is a nested pair; `X = w.text`, `T.endswith(X)`, `X in T` asserted; the swallowed prefix stays in `C` (§4). **Repair candidate**, subject to G5 (collision, §2.6) and fidelity (V13). No length test of any kind; `X` may exceed 600.
   - **ANOMALY** — `w` exists but `w.open_start < i` (the legitimate pair reaches back beyond the old span, so `X` would not be a substring of `T`), or `w.text != T` while `w.open_start == i`: `pair_spans_beyond_old_span` / `function_disagrees_with_anchor` → unchanged, listed.
   - **NONE** — no legitimate want closes at `j`: every marker in `[i, closer_end]` was skipped by the function; the reasons are recorded (histogram + the reason for the closer `j`). The whole span is discussion; there is nothing to separate. **UNCHANGED, LISTED** with those reasons — this is mechanical evidence, not a guess.
5. **No guessing:** the tool never chooses a span itself; `X` is always `w.text` as returned. No node is decided by a proxy (the backtick-lead test is gone).
6. **Post-conditions** for every SEPARATE node, or it is not written: `X == C[w.open_start+6 : w.close_end-7].strip()` (no truncation), `X in T`, `T.endswith(X)`, `X` non-empty, `want_id_for_text(X)` recomputed equal to the mapping's `new_id`.
7. **Report fields per node:** class, reason(s), old/new length and sha16, `w.open_start`/`close_end` relative to `i`, the skip reason of the **outer opener** `i` (a real but *unclosed* `[WANT]` is, under P406's paired rule, not a want — the report shows it as `opener_unclosed` so the Executive can see what was separated away), the number of masked markers inside `X`, and a ±80-character context excerpt at both tags to help the frozen-list review.

### 3.4 Where "mention vs real tag" cannot be told mechanically (LEFT UNCHANGED or accepted-and-listed; never guessed)
1. **A bare, unquoted mention that pairs cleanly** ("use [WANT] to mark one, and [/WANT] closes it") is structurally a want [#810-plan failure mode 1]. It **will be classified GENUINE or SEPARATE** and is not detectable; it reaches the Executive only through the frozen-list review of the printed `X` and excerpts.
2. **Quoted/escaped spans in pasted tool output.** The planned `quoted` reason requires the quote characters *immediately adjacent on both sides of the token* (`"[WANT]"`), and `escaped` requires an odd run of backslashes right before `[`. But the sources are JSON-escaped tool results (heads show literal `\n` and `\"` sequences [derived]); a Python/JSON string literal such as `(\"[WANT]I want to feel …[/WANT]\"` has a quote before the opener and text (not a quote) after it, so it matches **neither** planned reason and would count as a live marker. Evidence is thin: only four oversized heads contain a marker within their first 70 characters [derived]. Three (lengths 18067, 18357, 35579) show `('[WANT]')`, a token-wrapped quote that the planned `quoted` reason **does** catch; **one** (length 642) shows `(\"[WANT]I want …`, the uncovered idiom. A 70-character head shows only the first marker, so this is an illustration of the risk, not a count. Recommendation to the #810 pair: recognise a backslash-escaped or plain quote character immediately before a marker as a mention, as P406's "quoted/escaped span" implies (§7 Q2).
3. **Not recognised as mentions by the plan:** blockquote lines, indented code blocks, bold/emphasis, HTML/`<code>`, a longer quoted span that merely contains a marker [#810-plan table]. These stay live.
4. **Fences inside JSON-escaped text** appear as literal `\n```rust\n` sequences on one giant line; the line-based fence step cannot see them, so they are handled only by the CommonMark inline-code pairing (a 3-backtick run pairs with the next 3-backtick run). That may or may not match the author's intent; the dry run's reason histogram is what shows it **[unverified]**.
5. **A real `[WANT]` whose closer was never written** is not a want under the paired rule; the repair separates the nested pair away from it and reports the outer opener as `opener_unclosed`.

### 3.5 Recount of the classes from derived data only (plan-001 §3.4 classes A/B/C)
The derived flags are the same (118 oversized: **A** 67 = starts with a backtick and contains a marker; **B** 16 = starts with a backtick, no marker; **C** 35 = no leading backtick, contains a marker) and they agree with #810's own counts (123 rows with a marker in their text; 91 start with a backtick; 16 marker-free rows over 600) [derived]. What **changes** is what the flags may be used for: the backtick proxy is no longer the test, and the marker flag no longer decides genuineness (the planned function treats a *masked* marker inside a legitimate pair as text, so a long want that merely discusses the marker syntax is still a want).

| Class | Outcomes now possible | Outcomes excluded |
|---|---|---|
| A (67) | GENUINE, SEPARATE, NONE, ANOMALY | — |
| B (16) | GENUINE, NONE, ANOMALY | SEPARATE (needs a nested opener; none in `T`) |
| C (35) | GENUINE, SEPARATE, NONE, ANOMALY | — |

**Counts that change against plan-002:** repaired upper bound **67 → 102** (class C is no longer auto-left); "left unchanged ≥ 51" is **withdrawn** — derived data supports **no lower bound** on any outcome; GENUINE upper bound **≤ 118** (was 0: plan-002 assumed every oversized want was a mis-parse); class B is no longer "certainly left" (its 16 nodes are exactly the rows where the no-limit parser can re-derive the same id **if** the span is legitimate). The backtick-led population is *expected* to be dominated by NONE (an opener inside inline code, `in_code_span`), which is what the 09-16 changelog and #810 both infer, but that is an inference from the heads; the dry run states it. **The exact split is not computable from derived data** (§8).

### 3.6 Genuine long wants
The plan does not assume the 118 are all mis-parses: any of them may be a real, well-formed want longer than 600 characters, and the legitimacy test decides each (GENUINE → unchanged). Their ids equal `want_id_for_text(T)` by construction of the old parse, which is why a GENUINE node needs no repair and a no-limit re-parse dedupes onto it.

## 4. Where the swallowed prose goes (unchanged)
`T` is a contiguous substring of the source conversation node's content `C`; the removed prefix and `X` both remain verbatim in that ordinary, unprotected `cc:conv::` node (vdb untouched). A new "prose" node would be an unprotected orphan that `_collect_orphan_nodes` may collect. Acceptance 5 is checked as provenance (V9). NONE and GENUINE nodes keep their text and lose nothing.

## 5. [R3·P406] Execution procedure and verification (offline, copy-first, separate authority)

### 5.1 RULED apply order and the mechanical gates for it
**#810 merged and deployed on the laptop daemon → #801 applied live (daemon down, full backup first) → S4.** Written and reviewed now; **never applied live earlier.** Mechanical preconditions, checked by the tool and recorded in the report, any failure = STOP:
- **P1 identity of the function.** The #810 commit is recorded; the git blob of the file that defines `parse_wants` is recorded.
- **P2 same code the daemon runs.** `sha256` of the daemon's deployed defining file (the NeuroGraph checkout the unit runs; `cc-ng-daemon.service` imports it, #810-plan §1) equals `sha256` of the file the tool imports from. Read-only hash at run time; no data directory is touched.
- **P3 behavioural fingerprint.** #810's golden battery produces the recorded outputs when run through the imported function.
- **P4 the daemon is down** and has been since the deploy: any pulse between deploy and apply would have minted the SEPARATE nodes' texts as new ids (collision → both listed, the repair defeated). Verified by the pre-apply census: no `new_id` already present.
- **P5 frozen list pinned** to the P1-P3 values: the Phase-1 mapping and frozen list carry the function's hashes; Phase 2 refuses if they differ (a changed function invalidates the list).

### 5.2 Tool, writer, phases
**Tool:** a new scratch script (suggested `scripts/want_text_repair.py`), built in a later lane against the shipped function (a test double may stand in only inside unit tests). **Writer:** `checkpoint_guardian.atomic_file_write` (`:343-363`), value-granular rewrite (§2.4); the sidecar is written in the same JSON layout as `ActivationPersistence.write_state` (`json.dump(captured, f)`, `activation_persistence.py:138-169`) but through `atomic_file_write` because `write_state` writes in place.
**Preflight:** `MemAvailable ≥ 8 GiB`, load < 6, one loader at a time; classify/rewrite passes under `systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0`; V11 under 6G (measured restore ≈ 3.6 GiB); `NG_EMBED_REMOTE` unset, `HF_HUB_OFFLINE=1`.
**Phase 1 (COPY, nothing live touched; may use the #810-branch function, pinned by commit + P3):** copy the six checkpoint files with sha256; classify per §3.3 (streaming `C` from `vectors.msgpack`, embeddings skipped); emit `repair-list.json` (outcome table: GENUINE n, SEPARATE n (of which collisions n), NONE n by reason, ANOMALY n by code — the counts must sum to `|S|`) and the LISTED table; **freeze** (Chief/Exec strike ids; recommended); census (§2.4); fidelity V13; rewrite to a tmp file; run V1-V14; write `report-<UTC>` in the backup dir (contains want text; never the repo).
**Phase 2 (live):** only after Phase 1 passes, the §5.1 gates pass and Chief/Exec authorise. Daemon down; full backup of the six files (sha256, re-hash, independent read); **re-run the tool on the live bytes — never copy the Phase-1 output over live** (it is stale once the daemon autosaves); the recomputed mapping must equal the frozen one; run V1-V10 and V13 inside the writer functions (a failure raises and `atomic_file_write` leaves the final file untouched); `os.replace` main, then the sidecar (if the process dies between them the idempotent re-run completes the sidecar from the mapping, else roll back both); leave manifest and guard state alone (counts unchanged; `SaveGate` keys off `guardian_nodes` then `nodes`, `checkpoint_guardian.py:451`; #743 not triggered); do not start the daemon. **Rollback:** restore the two files from the backup, sha256 == recorded. **After S4 starts:** watch **≥ 2 autosave cycles** — want count, repaired ids/texts unchanged, minted set == the T6 prediction — and report the window.

### 5.3 The verifier (all mechanical; any failure → no live write)
| # | Check |
|---|---|
| V1 | 182 wants before and after; total node count equal; protected census 183; constitutional node untouched |
| V2 | for each mapping: `new_id == want_id_for_text(new_text)`; old ids absent, new ids present; injective; post ids == (pre ids − old) ∪ new; each new id at its old id's map position; **outcome accounting:** exactly one outcome per node in `S`, counts sum to `|S|` |
| V3 | per repaired node, `decode(post) == decode(pre)` with only `node_id`, `want_text` (→ `X`) and its own `pred_weights` keys mapped; `poincare_dir` bytes equal; `provenance`/`want_state`/`source_node`/`creation_mode`/`creation_time` equal |
| V4 | synapse-id set equal (138,753 in the copy; re-counted); every synapse equal except mapped endpoints; per-old-id vs per-new-id incident counts equal — **no synapse lost**; rim incident count unchanged (4,127 [derived]), exactly the 131 rim↔S ones differ in the want-side endpoint |
| V5 | hyperedges and archived: `member_nodes`/`member_weights`/`output_targets` equal under the mapping, list order preserved |
| V6 | S4, S6-S12 equal modulo the mapping; every other top-level value byte-equal |
| V7 | non-want nodes byte-equal except those differing only in mapped `pred_weights` keys, proven by an independent diff walker that permits only `(old id → new id)` differences at string positions; re-encoded non-node values keep `len(new) == len(old)` |
| V8 | sidecar keys == mapped key set, per-entry values byte-equal, `saved_at`/`timestep`/`version` equal; other files sha256-equal; manifest counts equal post counts |
| V9 | **[R3·P406] verbatim and untruncated:** `X in T`, `T.endswith(X)`, `X == C[open+6 : close-7].strip()` from the function's offsets; **no length assertion** (the `<= WANT_MAX_CHARS` check is removed); prose provenance (§4): source node ∈ graph, non-want, unprotected, vdb bytes unchanged and containing `T` |
| V10 | census: zero whole-string old ids remain; substitutions == census count |
| V11 | canonical read path: fresh `Graph().restore(before)` → per-want `_outgoing/_incoming/_node_hyperedges` sizes; free; `restore(after)` → same figures under new ids, every endpoint/member resolves, no dangling id; report the byte length of the rendered wants block before/after (**it may grow**: repaired texts are no longer clamped by the parser, and the render clamp `:1604` is #810/P408's, not this plan's) |
| V12 | idempotence: re-running finds nothing left to repair that it has not already listed |
| V13 | encoder fidelity: `pack(unpack(raw)) == raw` for every value to be re-encoded; any failure stops the whole run |
| V14 | **[R3·P406] shared-function proof:** the function used by the tool is the P1-P3 pinned one (hashes in the report) |
| **T6** | stub graph from the copy (`(id, kind, creation_mode, want_text)` + vdb content); call the **real #810 `surface_wants`** on before and after: `minted(after) ⊆ minted(before)` and `minted(before) − minted(after) ⊆ {new ids}`. A non-empty `minted(after)` is **not a repair defect** — it is the set of other legitimate wants the first S4 pulse will add (long ones the cap dropped, swallowed ones); report its size and the lengths. Also replay the legacy twin over the same stub and report what it would mint (§1) |

Unit tests for the build lane (synthetic fixtures): every outcome (GENUINE with a masked marker inside, SEPARATE longer than 600, SEPARATE with an unclosed outer opener, NONE by each planned reason, ANOMALY both codes, `anchor_failed`, `source_missing`); each collision shape; want↔want and rim synapses; `pred_weights` in a non-want node; hyperedge list order; sidecar re-key; census STOP; fidelity STOP; P1-P5 STOP; `want_id_for_text` equals the production mint; idempotence; **a 5,000-character `X` is neither truncated nor rejected**.

## 6. Ripple (carried; additions marked)
| Consumer | Effect |
|---|---|
| `render_wants` (`:1575-1610`, calls `cc_ng_host.py:1141-1144`) | **[R3·P406]** clamps each want to 600 (`:1604`) and shows 40 (`:1603`); #810 turn 1 does not touch it and P408 retires the standing block. Until then a repaired or GENUINE text over 600 characters is rendered clamped (in the graph it is whole). 28 of the 40 newest wants are in the 118 [derived] |
| `_pith_node_raw_text` / `_pith_node_text` (`:4640-4669`) | text; provider assembly bounds a node to `_CC_PITH_PROVIDER_NODE_CHARS` (default 700, `:3512`) — a bound at another layer, not the graph; flagged, not decided |
| `cc_stamp_missing_geometry` (`:3388-3484`) | none (`poincare_dir` carried); Exec Q4 |
| activation sidecar | re-keyed (S13) |
| `guardian_nodes`/manifest/guard state (#743) | none; equal counts |
| protected census (`plan-scratch/protected_census.py:53-56`) | flag-based, 183 unchanged; post-repair comparison joins through the mapping |
| want-hub d / analyses keyed by old ids | re-key through the mapping; degree/synapse figures carry over |
| vault docs and Quest tracker | none of the 118 candidate ids is cited (scan of the docs worktree and `quest-tracker`); other board records **[unverified]**; `#799 export` not located **[unverified]** |
| **callosum export/merge** (`cc_topology_export.py:351-416`; `cc_topology_merge.py:373-374`) | the exporter sends CC nodes whose id is not in the peer's membership ledger; the 118 new ids are absent from it, so they will be re-sent as new nodes and the receiver (idempotent by id) will admit them beside its old copies → duplicate protected wants on the peer (Exec Q5) |
| **legacy twin** `surface_wants_for_graph` | **[R3·CORRECTION]** not on the laptop daemon's path; live in `cc_ng_host.py`; parked (#755) — §1, Q1 |

## 7. [R3·P406] Questions for the Executive (via Chief) — the ruled ones have left the list
*Ruled and removed:* the P406 rule; the id follows the text; the collision rule; **the sequencing** (now §0.1/§5.1).
1. **The legacy twin (#755).** #810 parks `surface_wants_for_graph`; on the laptop it is not called, but if a graph hosted by `cc_ng_host.py` ever runs it over these sources it mints unbounded, unguarded `want::` copies of the spans this repair separates. Must the twin be routed through the shared function or retired **before S4** (or is "uncalled on the laptop" enough)? Same for Syl's `neurograph_rpc.py:4902` (canonical file; Josh's approval).
2. **What #810's legitimacy test must cover, because it decides the repair's outcomes.** (a) Escaped/plain quote immediately before a marker (`\"[WANT]`, `\'[WANT]\'`, §3.4-2) — recognise as a mention? (b) Masked markers inside a legitimate pair are **inert text** in the plan (a want may discuss the syntax); the earlier literal reading ("any marker inside disqualifies") would change some outcomes — which does the Executive want? (c) Blockquote lines, indented code and emphasis are not recognised — acceptable? (d) A backtick right *after* the opener is not rejected — acceptable? The repair uses whatever the shipped function does; these are #810's, but they change which of the 118 are repaired.
3. **Scope of LEFT UNCHANGED.** NONE nodes (entire span is discussion, nothing to separate) and ANOMALY nodes stay as they are and are listed; the 64 wants ≤ 600 are outside the approved 118 and are mostly code/test-fixture fragments — leave, or a follow-up? Alternative: the Executive hand-specifies per-id spans, applied mechanically.
4. **`poincare_dir`.** Carried byte-identical (recommended). A repaired want's geometry was stamped from the *old* text; leave, clear-and-restamp, or recompute.
5. **Peer copies.** Given the export re-send (§6), apply the same frozen mapping to the peer copy offline, or hold the export? Which side owns the ledger?
6. **Frozen-list review** (§5.2): required or optional; who signs; and does the Executive also want the listed **bare-mention risk** (§3.4-1) reviewed from the excerpts?
7. **Rim synapses.** Confirm re-pointing the want-side endpoint of the 131 rim↔want synapses is within "every synapse re-pointed" (rim node bytes unchanged) — and the `pred_weights` carve-out inside ~1.7 K non-want nodes, verified by the diff walker rather than raw bytes.
8. **First-pulse additions.** T6 will show how many other legitimate wants the #810 parser mints on the first S4 pulse (and their sizes). Who reviews that set before S4, given wants are never deleted?
9. **Slot, unit, retention.** Which daemon unit to stop, which slot, how long to keep the backup and the report (which contains want text).
10. **Extra inputs** flagged in §0 (`want-rows-laptop.json`, the sidecar copy's keys, the ledger snapshot) — acceptable to reuse?
11. **Long texts downstream.** A repaired or GENUINE want over 600 characters is clamped by `render_wants` until P408 retires it, and bounded per node by the pith provider (700). Flagged for the Executive; not decided here.

## 8. What this plan still cannot know without the texts
- **The exact outcome split** (GENUINE / SEPARATE / NONE / ANOMALY, collisions, the reason histogram): derived data gives only bounds (§3.5). It is produced by the Phase-1 dry run on a COPY with the real function; the plan does not estimate it.
- What the shipped #810 function actually does: everything in §3 is "as planned at `2ab4a851`"; a change to it changes the outcomes and invalidates a frozen list.
- Whether the JSON-escaped sources are classified as intended (§3.4-2, -4); how many census hits fall in S4, S7 and S10; how many values get re-encoded.
- Whether the live laptop checkpoint still equals the 09-23 analysis copy (Phase 2 re-derives everything from the live bytes), the daemon unit name and the autosave interval; and whether the deployed daemon file will byte-match the merged #810 file (P2).

**The LEFT list (id, outcome code, reasons, length, first/last 100 characters) and the collision list go to the Executive via Chief.**

**Flagged for the punchlist (not mine to file):** (1) the unbounded `surface_wants_for_graph` in `cc_ng_host.py:696-704` (parked #755) and `neurograph_rpc.py:4902`; (2) the laptop daemon's want surfacing shares one DEBUG-swallowing `try` with `drain_ingest_tract` (`cc-ng-daemon.py:2088-2119`; same shape at `cc_ng_host.py:1517-1534`) — a raise in the drain silently skips `surface_wants`; (3) `#810`'s planned `quoted`/`escaped` reasons do not match a JSON-escaped string literal such as `(\"[WANT]text` in pasted tool output (§3.4-2; one derived head shows it); (4) `ActivationPersistence.write_state` writes the sidecar in place, not atomically (`activation_persistence.py:162-164`); (5) dedupe-by-id-only for text-hashed ids.
