<!--
# ---- Changelog ----
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 TURN 2 return build-002
# What: the pair's corrections + #816/#817/#818/#819 folded; audit-table updates; what still cuts;
#   decisions left; evidence; the removed-reference list for #817.
# Why: dispatch #10952; brief TURN 2 + ADDENDUM + ADDENDUM 2; reviews checker-019, le-017 (read in full).
# How: every number below is from a command run in this session (see section 8 for what was and
#   was NOT run). Supersedes build-001 where they differ; build-001 carries pointers back here.
# -------------------
-->

# #813 TURN 2 — RETURN build-002

> **SUPERSEDED IN PART by `build-003.md` (turn 3, dispatch #11011): #817 was REVERTED and is DEFERRED.** **DEFERRED to the post-track VPS/daemon lane: the function + BOTH live Python handlers (`cc_ng_host.py:974`, `cc-ng-daemon.py:1617`) to be removed TOGETHER (LAW 3).** Chief ruling docs `084b4161`; the turn-2 removal was reverted in turn 3 (`82cbbcd` NG, `336954c3` docs). In this file §1 (`afc9b3e`, `771f006a`), §3 "#817", §4 row #10, §5 and §6 D9 describe the retirement and are no longer true; everything else here (#816, the ONE budget rule, #818, #819, F2/F3/F6/F8) stands. The claim "no live caller" was wrong: it was true of Rust only.

Lane `pith-clip-removal-813` · dispatch #10952 · worker seat (`worktree_write`) · returned **unreviewed**; a DELTA pair reviews the turn-2 diff (`bc4ae7a..HEAD`).
Related: [[NeuroGraph]] · [[Pith]] · [[NeuroGraph Is a Mind, Not a Database]] · [[Format-for-Purpose Principle]] · plan `../plan-002.md` · prior `build-001.md`.

**Nothing merged, wired, restarted or installed. The real `~/.bashrc` was never written (sha256 prefix `72f2e7133cce652a` before, during and after — hash only). Condensate read-only. No protected, vendored or shared (`surfacing.py`, `surface_resolver.py`, `neurograph_rpc.py`, `kiss_filter.py`, `tonic_thread.py`) file edited.**

## 1. Commits (each its own commit, pushed by name; remote head = local head at close)

| Repo · branch | Commit | Step |
|---|---|---|
| NeuroGraph `cc-laptop-pith-clip-813-20260930` | `f6c2650` | (0) turn-2 plan/inventory, **before code** |
| NeuroGraph | `b058237` | (1a) F3a/F3b staged-script hardening |
| NeuroGraph | `37f6bf7` | (1b) #816 + F1/C1 + F2 + F6 + F8 + C3/C4 + the ONE budget rule |
| NeuroGraph | `0052bc0` | (1c) audit corrections (row #7, consequence 4, D7) |
| NeuroGraph | `afc9b3e` | (2a) #817 retire `pith_compress_history` |
| NeuroGraph | `66695bc` | (2b) #818 every drop is loud |
| NeuroGraph | `bc36b0e` | (2c) #819 over-budget node → trees + reference |
| NeuroGraph | *this return's commit (branch head)* | (3) this file |
| docs `cc-laptop-pith-clip-813-20260930` (base `7cf85149`) | `771f006a` | (2a') retire the laptop daemon's `compress_history` handler |
| docs | *return-pointer commit (branch head)* | vault pointer |

## 2. The pair's findings → what I did

| Finding | Sev | Disposition |
|---|---|---|
| **F1 / C1** Pith-ON L1 path still cut (300-char pattern snippet; 240-char monitor items from SHARED code) | HIGH | **Fixed** (`37f6bf7`). Pattern stream: `cc_assemble_recall` asks `whole_content=True` (covers Pith-ON *and* gate-off: one call). Monitor stream: **CC-ONLY route** `_cc_monitor_items_whole` re-resolves each item's whole content by `node_id` (node + vdb entry, same inputs the monitor used); `surfacing.py` / `surface_resolver.py` **not edited** (test asserts it). Gate-off and the Pith-failure fallback render through `_cc_render_unpithed` + `_format_cc_monitor_block` (layout parity-tested against the real shared `format_context`). Regression tests: a >300-char pattern item and a >240-char monitor item come out whole, or are dropped whole with the INFO line, on Pith-ON, gate-off and the failure fallback. |
| **Audit row #7** "no budget" | (part of F1) | **Corrected in place** in `plan-001-audit.md`, marked `[CORRECTED turn 2]`; `build-001.md` annotated (`0052bc0`). |
| **C2** consequence 4 wording | LOW | Corrected: "cannot fit `learned_budget`" (total budget − core − section shell), not "cannot fit 40,000". |
| **C3 / F2** two budget rules; Stage 3 silent over-budget first line | LOW | **One rule** (below). Stage 3's "keep the first line even if it alone exceeds the budget" guard is **removed** — decision **D8**. |
| **C4 / F6** AST guard incomplete | LOW | Rewritten to the **complete caller set** (Name *and* Attribute calls, whole module) of `pith_stage2_keyframe` (must be empty) and `_pith_cut_at_word_boundary` (only inside the keyframe primitive); no `resolve_surface_content`/`resolve_surface_item` call may pass a literal `max_chars`; `cc_assemble_recall` must pass `whole_content=True` as a literal. |
| **C5** D7 naming | NOTE | `MAX_QUEST_CHARS` stays REJECT-LOUDLY; Condensate `origin/cc-laptop-minitid-card7-quest-removal-20260929` (`88ddfec`) is the Quest-removed blob; keep the guard until it **and both hosts** move together. |
| **F3a** script leaves a broken `.bashrc` when its own verify fails | MEDIUM | **Fixed** (`b058237`). `apply` runs `verify`; on failure it restores the just-made backup byte-exact and exits non-zero. Reproduced le-017's case (sole export inside `if true; then … fi`) against the **old** script (file left broken, rc=2) and against the new one (byte-identical after, rc≠0). `verify` now returns failure *explicitly* (`set -e` is suspended inside `if ! verify` — that trap would have let a broken file report success). |
| **F3b** symlink flattened | LOW | `sed -i --follow-symlinks`; a symlinked target stays a symlink; tested. |
| Script cannot check "after both merges deployed" | note | Stated in the script header and printed by `apply` as an **S4-checklist item**. |
| **F8** never-fit assemblies unidentified | LOW | The INFO line names never-fit **node ids** (bounded, first-time-seen, flood-safe; repeats are counted "already reported"). |
| F4 / F5 / F7 | — | **Not mine** (filed #821 / #822 / #823). Untouched. |

### The ONE budget rule (used by the provider admit, Stage 3 and the un-Pithed renderer)
1. **Whole or absent.** 2. **Strict rank prefix** on the remaining envelope. 3. **A never-fit unit** (cannot fit an *empty* envelope) is never emitted over budget and does not end the prefix; it takes the #819 reference form if that fits, else it is skipped. 4. **Loud**: one INFO line per call with count, total chars and never-fit ids.
Ranking is shared too: `_pith_unified_rank` was factored out of `pith_stage3` (no behaviour change).

## 3. #816 – #819

* **#816** as above. Gate-off size is controlled by *how many* items under the existing `cc_l1_budget` (LAW 3, reuse), lowest-ranked whole items dropped, INFO with count/size. **Fail-open:** my first fallback renderer used `CacheLine.from_surfaced`, and the existing test that injects a failure *there* also broke my fallback — a real design catch (Josh 2026-09-26: "when Pith fails there HAS to be pass-through"). `_cc_render_unpithed` no longer depends on the machinery that may have failed and, if its own budget step breaks, renders everything whole and unbudgeted with a WARNING (tested).
* **#817** retire `pith_compress_history` (LAW 3). **Callers verified again at my head:** Condensate `master` `4086540` — `git grep compress_history` → only the header comment (`minitid.rs:12,15`); docs `scripts` → only `cc-ng-daemon.py`; NG → host + tests. **Every reference removed:**
  * `cc_ng_organism.py`: `pith_compress_history`; `PithMetrics.history_calls / history_turns_in / history_turns_compressed / history_chars_in / history_chars_out / history_chars_saved / history_failures` (fields, `_reset_locked` lines, `snapshot()` lines); `PithMetrics.record_history_compression`; docstring mentions.
  * `cc_ng_host.py`: `_handle_compress_history` and the `"compress_history"` dispatch entry.
  * docs `scripts/cc-ng-daemon.py`: `handle_compress_history` and the `'compress_history'` DISPATCH entry.
  * tests deleted: `tests/test_cc_host_compress_history.py`, `tests/test_pith_history_metrics.py`.
  * `docs/PITH_HOST_CONTRACT.md`: the `compress_history` event example, the whole `compress_history → pith_compress_history` section, the outbound `history_*` counter table, the "history compression" intro/`Live sources`/tests mentions, and the healthy/failed-host bullets that read it.
  * Now: the socket answers `{"ok": false, "error": "unknown event: compress_history"}` (test drives the real connection handler over a socketpair). **Kept and flagged (D9):** `pith_stage2_keyframe` + `CC_PITH_KEYFRAME_CHARS` — a pure primitive with **zero** callers (AST-asserted); a future caller MUST carry the delta.
* **#818** `_pith_log_drop`: one INFO line per call and reason — count, total chars, reason, ids first-time-seen (bounded, env-sourced `CC_PITH_DROP_LOG_IDS_PER_CALL`=8 / `CC_PITH_DROP_LOG_SEEN_MAX`=4096 with defaults, LAW 5) — for `member_limit`, `depth_limit`, `overlap`, `roots`. Counts only candidates the walk **reached and declined** and that appear in **no** selected basin. What is selected is unchanged.
* **#819** an over-budget node → **its trees (whole) + one reference line**: `A long node (id …; ≈300k chars; 2026-09-21; 3 concept trees) is related to this cue; it is too large to render whole here, so its concepts follow.` **No split at ingest, no new node type, the node is never modified** (asserted) and still activates and learns in full; only its **rendering** changes. Provider path: the trees are the basin's ordinary graph neighbours (forest↔tree synapses made by `_cc_bind_conversational_topology`). Text-derived anchors of the unshown whole are not mined; metadata anchors are. Same swap on the Pith-ON L1 path and the un-Pithed path; one INFO line.
  **Dependency, stated plainly:** for **pre-PASS-2 forests the trees cover only the first 2,000 chars** until PASS 2 (the laptop TID) runs; full coverage arrives with PASS 2. Until then the reference is honest that the whole exists and is not shown.
  *Residual:* the per-node limit is `budget − core − 800` (a fixed shell allowance); a node between that and the true `learned_budget` is still caught — loudly, by id — as a never-fit assembly at admit, without its trees.

## 4. Audit-table updates since build-001 (full table: `../plan-001-audit.md`; §10 there records these)

| Row | Was (build-001) | Now |
|---|---|---|
| #7 recall snippet 300 | CUT, provider path only; "no budget" elsewhere | **Fixed for both Pith-ON streams and gate-off**; the 300 default remains only as the unused `whole_content=False` default — both production callers (`cc_ng_organism.py` provider + `cc_assemble_recall`) pass `True` |
| #3 admit tail drop | DROP-SILENTLY → loud | one rule, never-fit named by id |
| #9 Stage 3 tail `break` | counted, INFO added | never-fit skipped not emitted; first-line overrun guard removed (D8) |
| #10 `pith_compress_history` keyframe | CUT, no live caller, not changed (D2) | **Retired** (#817) |
| #18 member/overlap/`ROOTS`/`DEPTH` silent drops | COUNT / DROP-SILENTLY, not changed (D5) | **Loud** (#818) |
| #22 `surfacing.py` 200-char cut | CUT, shared, not changed (D6) | **Still cut in the shared file (untouched, by rule); no longer on any CC path** (the CC renderer never calls `format_context`; the monitor's 240-char item cut is bypassed by the CC re-resolve). Syl's `/assemble` still uses both (#812) |
| #14 wants `_WANT_RE` 600 / `render_wants` `[:600]` | CUT / DROP-SILENTLY, not changed (D3) | unchanged — not in this lane |
| #15/#20/#27 deposit-path clips | CUT, not changed (D4) | unchanged — #741 |
| consequence 4 | "cannot fit 40,000" | "cannot fit `learned_budget`"; such a node now surfaces via #819 |
| MAX_QUEST_CHARS (D7) | REJECT-LOUDLY, not dead | unchanged; card-7 blob named |

## 5. What still cuts or drops (honest list)
* **Shared / Syl's path (untouched, #812 family):** `surfacing.py:293-295` 200-char `"..."`; the shared monitor's 240-char item resolve; `surface_resolver` defaults (240/300); `neurograph_rpc.py handle_assemble` 300/`[:297]+"..."`; `kiss_filter.py` 60-char; `tonic_thread.py:649`.
* **Deposit path (#741, LAW 7):** Commons metadata `text[:2000]` (`cc_ng_organism.py:1122`); host `cc_ng_host.py:1190-1203` and daemon `cc-ng-daemon.py:1156-1169` tool-experience clips.
* **Wants (D3):** `_WANT_RE` 600-char span silently un-captures longer `[WANT]` blocks; `render_wants` `t[:600]`; `WANT_RENDER_LIMIT` 40 (that one is loud: `- ... and N older open wants`).
* **Rust (#822, read-only):** `PITH_NOTICE_WHY_MAX` 200 + `...` (`minitid.rs:1264-1265`) is a cut of the provider-facing notice; the 40,000-char rejection says "not a fresh envelope" rather than "oversized"; the tool-tail byte envelope drops whole messages visible only in aggregate counts.
* **Still silent, adjacent to #818:** Stage-1 clutter/dedup drops (counted in `_PITH_METRICS`, not logged); the graph engine's own `max_surfaced` cap inside `_harvest_associations` (shared `neuro_foundation.py`).
* **Reject-loudly guards (not cuts):** `MAX_INSTRUCTION_CHARS` 8000, `MAX_QUEST_CHARS` 8000.
* **A never-fit assembly with no trees** is absent (loud, by id) — the price of whole-or-absent; #819 covers nodes, not an assembly whose many *individually fitting* members exceed the budget.

**The 35-row audit is NOT a repo-wide cap census.** It covers the Pith provider/L1 path and the paths it feeds. Sibling extraction caps on Syl's path are #812-family and are not in it.

## 6. Decisions left (not made by me)
* **D8 (new)** — Stage 3's "keep the first unpinned line even if it exceeds the budget" guard is **removed** (ONE rule; it was a silent overrun). Consequence: a recall whose every item is never-fit yields an empty L1 unless #819's reference form supplies content — loudly. *Reviewer: confirm, or tell me to restore the guard as "emit + INFO" for Stage 3 only.*
* **D9 (new)** — `pith_stage2_keyframe` (+ `CC_PITH_KEYFRAME_CHARS`) now has **zero callers**; kept as the primitive a lossless rebuild would use. Delete it, or keep? (LAW 3 says shrapnel; it is also the one tested piece of "keyframe + delta").
* **D3** wants · **D4** deposit-path clips (#741) · **D6** shared `surfacing.py` (#812) — unchanged, not in this lane.
* **D7** — keep `MAX_QUEST_CHARS` until card-7 (`88ddfec`) and both hosts move together.
* Log volume: the INFO lines fire on every recall where something is dropped/referenced (that is "visible, never silent"); a rate limit is the operator's call, silence is not.

## 7. Contract
`docs/PITH_HOST_CONTRACT.md` updated in the same commits: whole-recall rule for both streams and both paths, the ONE rule and its INFO line, #818's loud drops, #819's reference form **with the pre-PASS-2 dependency**, and `compress_history` / `history_*` removed.

## 8. Evidence — and what I did NOT run
* **P379:** `test_p379_module_under_test_is_the_worktree_copy` fails if `cc_ng_organism` is not this worktree's file; `NG_EMBED_*` / `PYTHONPATH` unset for every run (`env -u`).
* **RED first** for each step (e.g. (1b): 13 failed / 28 passed against the unchanged code; (2b): 4 failed / 1 vacuous guard; (2c): 6 failed / 2 guards).
* **Golden vs BASE `e4ebf982`** (`tests/fixtures/pith_clip_813_golden_base.json`, generated **from BASE** by `gen_pith_clip_813_golden.py`; turn 2 added +8 lines, removed none): 4 provider scenarios + **6 `cc_assemble_recall` scenarios** (Pith-ON and gate-off × both streams / pattern only / monitor only, short items) are **byte-identical**.
* **Final targeted run, no daemon env, worktree code only — (A): `254 passed, 3 skipped`** (the 3 = host/daemon parity, need `CC_DAEMON_UNDER_TEST`). **(B) parity tests alone with `CC_DAEMON_UNDER_TEST` → docs worktree daemon: `3 passed`.**
* **docs:** `scripts/tests/test_cc_ng_service.py` + `test_pith_provider_context_wrapper.py` → `25 passed`.
* **Staged script:** rehearsed on locked `/tmp` copies of the real file (apply → diff is exactly `291d290 < export CC_PITH_PROVIDER_NODE_CHARS=700` → syntax ok → 5 keepers present → reverse byte-exact; symlink variant keeps the link and restores the target); real `~/.bashrc` hash unchanged.
* **Honest process notes**
  * I did **not** run "the targeted files once": I ran subsets while iterating (each RED/GREEN step) and the full set once at the end — plus one extra invocation because my first final command used `-rs` and did not name failures. Those mid-course runs are why the counts moved from 223 (turn 1) to 254.
  * **A P379 hazard I found:** setting `CC_DAEMON_UNDER_TEST` *inside* the full run makes `test_pith_provider_context.py` import the docs daemon, whose import puts the **primary checkout** (`/home/josh/NeuroGraph`) at the front of `sys.path`; Stage 4's fixtures then re-import modules and later tests silently ran the primary's *old* `cc_ng_organism` (9 spurious failures, all from my tripwire `format_context` firing against stale code). So the parity tests must be run **alone** — as in (B). The reviewers' packet should say so. (Turn 1's run didn't hit it because it wasn't mixed.)
  * Existing tests changed (each has a changelog reason): `test_cc_recall_unification.py`, `test_cc_recall_dedup.py`, `test_cc_region_confidence.py` (doubles accept `whole_content`; a double with the old signature raised inside the fail-soft `try` and silently emptied the pattern stream), `test_pith_stage2.py`, `test_pith_stage3.py` (never-fit skipped, not emitted). Two test files deleted (#817). One of my own turn-1 tests (the name-list AST guard) was replaced by the complete-caller-set tests.
  * The docs branch is still based at `7cf85149` (now 22 behind `origin/main`); the merger rebases. I did **not** `pull --rebase` the docs branch again (that rewrites pushed commits, which I don't force-push).
* **Not verified:** no live service, socket, sidecar, checkpoint, tract or real graph; the INFO wording is asserted by tests, never seen in a running daemon; Rust read, not built; PASS 2 coverage of real forests unmeasured (I did not load a graph); the effect of the L1 budget on real recall sizes is unmeasured.

## 9. Undo
NG: revert `bc36b0e`, `66695bc`, `afc9b3e`, `37f6bf7` (in that order) — `b058237`/`0052bc0`/`f6c2650` are script/docs only. docs: revert `771f006a`. `.bashrc`: never applied.

## 10. Order for S4 (unchanged from build-001 §8, plus)
NG merges first (P329, merge = deploy) → docs merge → `bashrc-drop-node-chars.sh apply` (**checklist item: confirm both merges are deployed — the script cannot**) → restart. Because the laptop daemon (docs) and the VPS host (NG) both lose `compress_history`, **any client still sending it now gets `unknown event`** — Condensate master sends none.

Pairing: the DELTA pair (cross-family non-glm + LE) reviews the turn-2 diff before anything merges; this worker does not self-accept.
