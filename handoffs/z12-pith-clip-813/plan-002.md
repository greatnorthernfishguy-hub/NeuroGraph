<!--
# ---- Changelog ----
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 TURN 2 plan/inventory (pre-code)
# What: the exact change list for the pair's corrections + #816 (both Pith-ON streams AND the
#   gate-off path) + F2/F3a/F3b/F6/F8, then #817, #818, #819, with the ONE budget rule.
# Why: dispatch #10952; brief build-813-pith-clip.md TURN 2 + ADDENDUM + ADDENDUM 2; reviews
#   checker-019 (PASS-WITH-NOTES) and le-017 (PASS-WITH-NOTES), both read in full.
# How: line numbers read at branch head bc4ae7a. Nothing in this file is code.
# -------------------
-->

# #813 TURN 2 — plan/inventory (committed BEFORE code)

Related: [[NeuroGraph]] · [[Pith]] · [[NeuroGraph Is a Mind, Not a Database]] · [[Format-for-Purpose Principle]] · previous: `plan-001-audit.md`, `returns/build-001.md`, reviews `checker-019-813-pith.md`, `le-017-813-pith.md`.

## 0. Order of commits (each its own commit, pushed by name)
NG branch: **(0)** this plan → **(1a)** F3a/F3b staged-script hardening → **(1b)** pair corrections + #816 + F2/F6/F8 + the ONE rule → **(1c)** audit/plan corrections (row #7, consequence 4, D7) → **(2a)** #817 retire `pith_compress_history` *(DEFERRED/reverted in turn 3)* → **(2b)** #818 loud drops → **(2c)** #819 over-budget node → **(3)** `returns/build-002.md`.
docs branch: **(2a')** retire the daemon's `compress_history` handler (same change as #817) → return pointer.

## 1. Facts I re-verified at head `bc4ae7a` (not taken from the reviews)
* **F1 is real and wider than I said.** `cc_assemble_recall` calls `cc_pattern_completion_recall(ng, query, k, state=conv_state)` with `whole_content` defaulted `False` (300-char cut + `…`) at the single call that feeds BOTH the Pith-ON `pattern` stream and the gate-off `## Active Recall` block. The `monitor` stream is cut earlier still, inside SHARED code: `surfacing.py:187-214` resolves each fired node with `resolve_surface_item(node, db_entry)` (default `max_chars=240`) and stores that as `_SurfacedItem.content`; `format_context` then cuts again at 200. I do NOT edit `surfacing.py` / `surface_resolver.py` (Syl's `/assemble`, P329, #812).
* **Trees link to their forest by real synapses** (`_cc_bind_conversational_topology`, `cc_ng_organism.py:2003-2016`: forest→tree 0.2, tree→forest 0.15, plus a hyperedge over forest+trees+windows) and trees carry `_tree_concept` + `_concept`. So a basin rooted at a forest already pulls that forest's trees in through normal expansion; #819 needs no new node type and no schema change.
* `pith_compress_history` references: organism (function, `PithMetrics.history_*` ×7 fields + `record_history_compression` + snapshot/reset, docstring mentions), `cc_ng_host.py` (`_handle_compress_history`, dispatch table entry), docs `scripts/cc-ng-daemon.py` (handler `:1617`, table `:1722`, changelog `:240`), `docs/PITH_HOST_CONTRACT.md` (the `compress_history` section + history counters + healthy/failed-host bullets), tests `test_cc_host_compress_history.py`, `test_pith_history_metrics.py`. **Callers on Condensate `master` (`4086540`): none** (`git grep compress_history` → only the header comment at `minitid.rs:12,15`).

## 2. THE ONE BUDGET RULE (C3, F2, F8) — used by the provider admit, Stage 3, and the un-Pithed renderer
1. **Whole or absent.** Nothing is shortened to fit.
2. **Strict rank prefix on the remaining envelope**: the first unit that does not fit the space left ends admission; lower-ranked units never jump it.
3. **A unit that cannot fit an EMPTY envelope ("never-fit") is never emitted over budget.** It takes the #819 form (whole-node reference + its trees) when that fits; otherwise it is skipped. It does not end the prefix.
4. **Every drop / skip / substitution is loud**: ONE INFO line per call with count, total chars and reason; never-fit **node ids** (bounded, first-time-seen only, flood-safe).
Consequence, decided here so the reviewer can veto it (**D8**): `pith_stage3`'s "keep the first unpinned line even if it alone exceeds the budget" guard is REMOVED. That guard is exactly the silent overrun le-017 F2 found and the rule split checker-019 C3 found; keeping it would make Stage 3 the only path that emits over budget. The price: a recall whose every item is never-fit yields an empty L1 unless #819 supplies the reference form — loudly. Existing `oversized_top_line_kept` tests are rewritten with that reason.

## 3. Change list

### (1a) F3a / F3b — `returns/bashrc-drop-node-chars.sh` (+ tests)
* `apply`: after the edit, run its own `verify`; **if verify fails, restore the just-made backup byte-exact and exit non-zero** (le-017 reproduced: the sole line inside `if true; then … fi` → `bash -n` rc=2, file left broken). Also verify before finishing that the restored file is identical to the backup (sha).
* **F3b:** `sed -i --follow-symlinks` (a symlinked `~/.bashrc` stays a symlink; backup/reverse/rollback use `cp`, which follows the link, so the target is the file edited and restored). Tested with a symlink.
* Header + `apply` output: the script **cannot check** "after both merges are deployed" — that is a **checklist item in the S4 batch**; `apply` prints the reminder.
* Re-rehearse on locked `/tmp` copies only (real `~/.bashrc`: hash only, never written).

### (1b) #816 + F1 + F2 + F6 + F8 + C3/C4 — `cc_ng_organism.py`
* **Pattern stream:** `cc_assemble_recall` → `cc_pattern_completion_recall(..., whole_content=True)` (organism-local). Covers Pith-ON AND gate-off (one call).
* **Monitor stream, CC-ONLY route:** a new organism helper re-resolves each monitor item's whole content **by `node_id`** from `ng.graph` + `ng.vector_db` with `resolve_surface_content(..., max_chars=sys.maxsize)` (the same node/vdb inputs `surfacing.py` used; the shared resolver's default and `surfacing.py` are untouched). Fail-soft: an item that cannot be re-resolved keeps its (cut) text and is counted/logged at DEBUG — never dropped for that.
* **Un-Pithed renderer (gate-off AND the Pith-failure fallback):** `monitor.format_context` (shared, cuts at 200) is replaced in this path by a CC-side formatter with the identical layout (header `[NeuroGraph Surfaced Knowledge]` — miniTID's rail marker — one `- content (salience: x.xx)` line per item, the image-only line), kept byte-identical for short items by a parity test against the real `SurfacingMonitor.format_context`. Size is controlled by HOW MANY: items ranked with the SAME unified rank as Stage 3 (`_pith_unified_rank`, factored out of `pith_stage3`, no metrics side effect), admitted by the ONE rule against `cc_l1_budget(...)` (existing budget, LAW 3), INFO line with count and total chars.
* **The ONE rule** as `_pith_admit_strict_prefix` + `_pith_log_budget_drop`, used by `_pith_provider_admit`, `pith_stage3` (first-line guard removed) and the un-Pithed renderer. `_pith_provider_admit` gains `graph` (needed by #819). F8: never-fit ids in the INFO line.
* **F6 / C4:** the AST guard is rewritten to walk the **complete caller set** of `pith_stage2_keyframe` / `_pith_cut_at_word_boundary` (Name **and** Attribute calls, whole module) and to assert no `resolve_surface_content` call passes a literal `max_chars`; behaviour tests: a >300-char pattern item and a >240-char monitor item survive Stage 3 whole, or are dropped whole with the INFO line, on Pith-ON, gate-off and the failure fallback.
* Golden vs BASE `e4ebf982` extended: `cc_assemble_recall` on short items, Pith-ON and gate-off, byte-identical.

### (1c) documents
Audit row #7 corrected (it is budgeted on the Pith-ON path), consequence 4 = "cannot fit `learned_budget`" (C2), D7 names Condensate `origin/cc-laptop-minitid-card7-quest-removal-20260929` (`88ddfec`) and keeps `MAX_QUEST_CHARS` until it and BOTH hosts move together (C5), `PITH_HOST_CONTRACT.md` updated, and the plan-001 amendment.

### (2a) #817 — retire `pith_compress_history` (LAW 3)
> **[TURN 3, dispatch #11011] DEFERRED — REVERTED.** **DEFERRED to the post-track VPS/daemon lane: the function + BOTH live Python handlers (`cc_ng_host.py:974`, `cc-ng-daemon.py:1617`) to be removed TOGETHER (LAW 3).** Chief ruling docs `084b4161`; the turn-2 removal was reverted in turn 3 (`82cbbcd` NG, `336954c3` docs). Everything below in this subsection is the turn-2 plan as executed and then reverted; it is kept for the record and is NOT current.

Removed (every reference listed in the return): `pith_compress_history`; `PithMetrics.history_*` fields, `record_history_compression`, their snapshot/reset lines; `cc_ng_host._handle_compress_history` + its dispatch entry; docs `cc-ng-daemon.py` `handle_compress_history` + entry; the contract's `compress_history` section, `history_*` table and healthy/failed-host bullets; `tests/test_cc_host_compress_history.py`, `tests/test_pith_history_metrics.py`. **Kept and flagged (D9):** `pith_stage2_keyframe` (+ `CC_PITH_KEYFRAME_CHARS`) — a pure primitive with **zero** callers afterwards; it is the piece a lossless (keyframe + delta) rebuild would use, and deleting a tested primitive is a decision above this brief. A future caller MUST carry the delta.
A test pins that the socket now answers `unknown event: compress_history` for both hosts (the failure mode the contract described for a stale host).

### (2b) #818 — every drop is loud
INFO with **count, total chars and reason**, one summary per call, per-item detail only first time seen (bounded set, env-sourced sizes, LAW 5 — same shape as the #810 skip log): `member_limit` (neighbours beyond `CC_PITH_PROVIDER_MEMBERS`), `depth_limit` (neighbours beyond `CC_PITH_PROVIDER_DEPTH`), `overlap` (basin skipped at ≥60 % coverage), `roots` (harvest results beyond the root count `k`). Not changed (and listed): Stage-1 clutter/dedup drops are counted in `_PITH_METRICS` but not logged — adjacent, named in the return.

### (2c) #819 — an over-budget node surfaces through its TREES plus a one-line whole-node reference
No split at ingest (Exec P417; LAW 7: raw = complete; P360: one node, one forest). Only the **rendering** changes; the node still activates and learns in full. When a node's whole text cannot fit the largest usable envelope (the budget minus the constitutional core — `learned_budget` is smaller still; a node between the two is caught at admit as a loud never-fit), the basin renders ONE line — `A long node (id …; ≈N chars; YYYY-MM-DD; T concept trees) is related to this cue; too large to render whole here, so its concepts follow.` — and its trees arrive as the assembly's normal whole relations (they are graph neighbours). Text-derived anchors of the giant are not extracted (they belong to the unshown whole; metadata anchors are kept). INFO line per call (count, size, ids bounded). Same substitution for the L1 path (recall items larger than the L1 budget) with the giant's strongest trees whole under the reference. **Dependency, stated plainly:** for **pre-PASS-2 forests the trees cover only the first 2,000 chars** until PASS 2 (the laptop TID) runs; full coverage arrives with PASS 2. Until then the reference is honest that the whole exists and is not shown.

## 4. Not in this lane (filed by the Chief; untouched)
#821 (F4 dead code incl. `CC_PITH_PREFETCH_LOD_DIST`, `_CC_PITH_PREFETCH_SUMMARY_CHARS`, `_cc_node_query_distance` + host/daemon allow-lists), #822 (F5 the Rust proposals + spec text), #823 (F7 `~/.bashrc` mode 644, Josh FYI), the #812 family (Syl's siblings: `neurograph_rpc.py handle_assemble`, `kiss_filter.py`, `tonic_thread.py:649`). **The 35-row audit is NOT a repo-wide cap census** — it covers the Pith provider/L1 path and the paths it feeds.
