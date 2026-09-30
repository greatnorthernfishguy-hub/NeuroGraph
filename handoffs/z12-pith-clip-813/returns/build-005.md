<!--
# ---- Changelog ----
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 TURN 5 return build-005
# What: le-022 N1 (fix, failing-first), N2/N3 (fixes), LOWs, N4/N5/T1/C4 recorded, #817 record completed.
# Why: dispatch #11114; Chief ruling docs 3604cfb1 on le-022 (PASS-WITH-NOTES, final diff bc4ae7a..1a347f9c).
# How: NG repo only. Every number below is from this session. The failing output in section 2 is
#   quoted from /tmp files written at the time (before the fix existed).
# -------------------
-->

# #813 TURN 5 — RETURN build-005 (small)

> **CORRECTION (turn 6, 2026-09-30; history below is left as written):** the row "LOW — constants 800 / 200" and the `_pith_provider_node_limit` docstring said the measured limit is conservative and that a node between it and the true budget "is still caught loudly ... as a never-fit assembly at admit". **That was wrong.** The limit measured the WORST-case shell, so in a tight budget it fell *below* what fits and the node was dropped or turned into a reference instead (le-025 C-2: core 800 / budget 1200 / node 30 → `capacity_empty`; 800/1500/60 and 400/1000/30 → reference). Fixed in turn 6 (`77a18ce`): the limit is now the optimistic bound. See `build-006.md`.

Lane `pith-clip-removal-813` · dispatch #11114 · worker seat · returned **unreviewed**.
Related: [[NeuroGraph]] · [[Pith]] · [[Duck Ethics]] · previous `build-004.md` · review `../reviews/le-022-813-delta.md` (read in full).

**Nothing merged, wired, restarted or installed. The real `~/.bashrc` was never written (sha256 prefix `72f2e7133cce652a`, unchanged). Docs repo, `cc_ng_host.py`, `surfacing.py`, `surface_resolver.py` not touched. Condensate not touched.**

## 1. The items

| Item | Result |
|---|---|
| **N1 (MEDIUM) — FIX, failing-first** | **Done.** `_cc_render_unpithed` (`cc_ng_organism.py:5771`) now exempts identity-protected items from the budget **exactly as Stage 3 does**, through **one shared test**, `_cc_pin_probe` (`:5688`), which `cc_assemble_recall` also uses (`:5930`, replacing its private closure — same `ng.graph._is_identity_protected`, same fail-soft-to-unpinned). A pinned item is kept whole (`pinned_items`, `:5810`; added to `kept`, `:5819`), is **never ranked, never dropped, never swapped for a reference** (`_pith_reference_items(..., pinned)`, `:4764`), and is never the reason the strict-prefix stop loses a smaller unprotected item. Ordinary items behave exactly as before (golden unchanged). **Pins are NOT folded into THE ONE rule** (C4 stays the Exec's open note). See §2 for the failing-then-passing evidence. |
| **N2 (MEDIUM/LOW) — FIX** | **Done.** `_pith_whole_node_reference` (`:4662`) no longer ends "so its concepts follow" unconditionally. **0 trees:** the line ends "too large to render whole here." and says nothing else. **L1 / un-Pithed path (it knows what it placed):** `K of N concept trees follow` / `N concept trees follow` / `1 concept tree follows` / `None of its concept trees fit here`. **Provider path (the trees arrive as the basin's ordinary relations, subject to the member/depth caps, so the count placed is unknown when the line is written):** `Its concept trees follow where they fit`. Every non-empty case carries `they may cover only part of it`. **Limit I could not remove:** nothing on a node marks PASS 2 (PASS 2 runs in the TID), so the code cannot say *whether* a forest is pre-PASS-2 or state "first 2,000 chars" only where it applies; the hedge is therefore unconditional, and the contract/changelog keep the exact 2,000-char statement. `_pith_reference_text` (`:4712`) was reworked so the line and the trees agree exactly (it drops a tree if the longer "K of N" wording would overflow the budget). |
| **N3 (LOW) — FIX** | **Done.** `docs/PITH_HOST_CONTRACT.md` now describes C2 (a monitor item whose re-resolve raises is DROPPED + one WARNING with count, first-seen ids, exception *type*, no text), C3 (the `pith reference form: node … shows K of N concept trees whole; L left out …` INFO line, budget **or** member cap), N1 (identity outside the un-Pithed budget, with the open note), the truthful N2 wording, and a table of the two knobs: `CC_PITH_DROP_LOG_IDS_PER_CALL` default `8` clamp `[1, 64]`; `CC_PITH_DROP_LOG_SEEN_MAX` default `4096` clamp `[16, 65536]` (neither a mandatory export). A test reads the defaults out of the code and asserts the contract states them. |
| **LOW — constants 800 / 200** | **Done.** `_PITH_SHELL_ALLOWANCE` and the `max(200, …)` floor are gone. `_pith_provider_node_limit` (`:4626`, used at `:5564`) **measures the renderer's own output**: the worst-case section shell (`_pith_provider_sections` with a candidate of every alert coherence and one correction, empty blocks) and the minimum line overhead (`_pith_render_connected_line` of an empty line); `limit = budget − core − shell − line`, floored at **1** (a `0` would silently disable the reference form). It stays the conservative (worst-case-shell) direction; a node between it and the true learned budget is still caught loudly, by id, as a never-fit assembly at admit. |
| **LOW — #817 deferred record (le-022 gaps a–c)** | **Done** in `../plan-001-audit.md` §11 ("COMPLETE checklist") and pointed to from `../plan-002.md`: (a) **traffic evidence first** — "two live callers" means two live *handlers*; nobody has shown any client still *sends* `compress_history`; grep/measure the senders (miniTID, Anima, other hosts, hooks) before removing the event; (b) the `pith_stage2_keyframe` docstring (`:4267`) and the AST guard `keyframe_callers <= {"pith_compress_history"}` (`tests/test_cc_pith_clip_813.py:659`) change in the same removal; (c) the 3 parity tests (`tests/test_pith_provider_context.py:499`, run alone) and the contract's `history_*` table/bullets (`docs/PITH_HOST_CONTRACT.md:380-389`, `:427`, `:441`) are re-run/updated; audit row 10 stays open. Every line number in it was re-verified at this commit (two stale ones from earlier turns caught and fixed). |
| **T1 — record; pointer comments added** | `_cc_monitor_items_whole` and `_format_cc_monitor_block` are an **INTERIM fork** of the shared monitor/formatter. The **source fix** — a whole-content option in `surfacing.py` / `surface_resolver.py` (**#812**, Josh's post-track go) — **deletes both** in the same change. One-line pointer comments now sit at both functions (`:5702`, `:5750`). The zone manager writes the #812 row. |
| **N4 — record only** | The C3 INFO line has no first-seen suppression, so it repeats every recall while a giant stays in the cue. Accepted: it is counts-only, one line per swap, per the Chief's ruling. |
| **N5 — record only (le-022)** | `on_monitor_error` is not called for re-resolve drops (host telemetry sees them only in the WARNING); and if `surface_resolver` cannot be *imported*, every monitor item is dropped (loudly, by WARNING) where the shared monitor itself would fall back to the vdb path. Consistent with whole-or-absent; not changed. |
| **C4 — open note (Exec's call)** | Pinned Stage-3 lines sit off-budget (identity by design): a large pin can push L1 past `budget_chars` with no drop INFO. With N1 the un-Pithed path now behaves the same way. Not folded into THE ONE rule. |

## 2. N1 — failing first, then passing

**Written first** (`test_n1_*`, four tests, in `tests/test_cc_pith_clip_813.py`) and run on head `b2f3d18` **before any fix existed**:

```
FAILED test_n1_identity_protected_item_is_exempt_from_the_gate_off_budget
FAILED test_n1_a_pin_larger_than_the_whole_budget_is_still_rendered_whole_never_referenced
FAILED test_n1_the_pin_exemption_also_holds_on_the_pith_failure_fallback
FAILED test_n1_pith_on_and_gate_off_agree_on_which_items_survive
4 failed, 64 deselected
```
(the fourth is the sharpest: Pith-ON keeps the identity item, gate-off loses it — the two paths disagreed.)

**le-022's probe reproduced on head** (module confirmed to be the worktree copy): 3 items, budget 4000 — `a` 2500 @ 9.0, `b` 2500 @ 8.0, identity-protected 1200 @ 0.1:
```
BEFORE  INFO: pith recall (un-Pithed): L1 budget 4000 chars met by dropping 2 whole items (3700 chars); kept 1 (2500 chars)
        ident protected: True | ident kept: False | a kept: True | b kept: False
AFTER   INFO: pith recall (un-Pithed): L1 budget 4000 chars met by dropping 1 whole items (2500 chars); kept 1 (2500 chars)
        ident protected: True | ident kept: True  | a kept: True | b kept: False
```
The strict-prefix stop at `b` no longer costs the small protected item; only the ordinary item `b` is dropped, and the pin is not counted in the kept budget either. After the fix all four N1 tests pass (74/74 in the file at that point). **Honest note:** N2 and the measured-limit tests were written alongside their fixes, not shown failing first (only N1 was required to be).

## 3. Commits, diff, proofs

| | |
|---|---|
| base of this turn | `b2f3d18` = turn-4 return `1a347f9c2a4f0246012eb9f1f50c5cefe9aa22cb` + le-022's review file only |
| **code commit** (the commit before this return file) | **`a26c77410db8f065d2c6fd7e7a91953208c99f6e`** |
| **return commit** = final NG branch head | the commit that adds **this file** together with the audit/plan record edits (parent `a26c774`); a file cannot name its own commit — the exact hash is in the reply and equals `git rev-parse HEAD` / `git ls-remote origin refs/heads/cc-laptop-pith-clip-813-20260930` |
| docs branch head (untouched) | `fd0ccc9c08b965e6b9c68baebe6f78a09ed36931` |

`git diff 1a347f9 HEAD --stat` at the code commit: `cc_ng_organism.py | 139 +++++++++++++++-----` · `docs/PITH_HOST_CONTRACT.md | 42 +++++-` · `tests/test_cc_pith_clip_813.py | 150 +++` · `…/reviews/le-022-813-delta.md | 97 +++` (the reviewer's file, not mine) → 4 files, 393 insertions, 35 deletions. **Mine: 3 files, 296 insertions, 35 deletions** (this return + the two plan/audit edits add on top).

* **`git diff e4ebf982 HEAD --stat -- cc_ng_host.py` → empty** (byte-identical to base).
* Protected / vendored / shared (`surfacing, surface_resolver, cc_ng_host, neurograph_rpc, kiss_filter, tonic_thread, neuro_foundation, openclaw_hook, stream_parser, activation_persistence, ng_*`) touched vs base: **none**.
* Golden vs BASE `e4ebf982` unchanged and passing; P379 preamble passes (`env -u NG_EMBED_REMOTE -u PYTHONPATH`; the probe scripts confirmed the module path is this worktree).

## 4. Tests — once
**(A)** the turn-4 union (`test_cc_pith_clip_813, test_pith_provider_context, test_pith_stage1…5, test_pith_l1_provenance, test_pith_metrics_concurrency, test_cc_host_pith_telemetry, test_cc_recall_dedup, test_cc_recall_unification, test_cc_region_confidence, test_pith_history_metrics, test_cc_host_compress_history`; worktree code only, no daemon env) → **282 passed, 3 skipped** (270 at turn 4 + 12 new: 4 N1, 6 N2/limit, 2 N3; the 3 skips are the host/daemon parity tests). **(B)** parity tests **alone** with `CC_DAEMON_UNDER_TEST` → the docs worktree daemon → **3 passed**.
**Not verified:** anything live (no service, socket, sidecar, checkpoint, tract, real graph); the N1 behaviour is proven with a fake in-memory graph and le-022's probe shape, not against a real identity-protected node set; log wording is asserted by tests, not seen in a running daemon.

## 5. Process notes
* `git pull --rebase` refused once mid-turn because my audit/plan edits were uncommitted; the (fast-forward) push of the code commit was unaffected. No history was rewritten.
* Open notes carried forward, unchanged: C4 (pins off-budget; Exec) · #817 deferred (record completed above) · T1/#812 (source fix deletes the fork; Josh's post-track go) · D3 wants · D4 deposit-path clips (#741) · D7 `MAX_QUEST_CHARS` until card-7 (`88ddfec`) and both hosts move together · #821 (dead LOD constants / phantom `CC_PITH_PREFETCH_LOD_DIST`) · the 35-row audit is not a repo-wide cap census.
