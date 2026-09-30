<!--
# ---- Changelog ----
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 TURN 4 return build-004
# What: checker-022 C1/C2/C3 done with tests; C4 recorded; real line numbers; proofs; final heads.
# Why: dispatch #11061; Chief ruling docs 2a3b5fbf on checker-022 PASS-WITH-NOTES (turns 2+3).
# How: NG repo only (cc_ng_organism.py + tests + this file). Every number below is from this session.
# -------------------
-->

# #813 TURN 4 — RETURN build-004 (small)

Lane `pith-clip-removal-813` · dispatch #11061 · worker seat · returned **unreviewed**.
Related: [[NeuroGraph]] · [[Pith]] · previous `build-003.md` · review `../reviews/checker-022-813-delta.md`.

**Nothing merged, wired, restarted or installed. The real `~/.bashrc` was never written (sha256 prefix `72f2e7133cce652a`, unchanged). Docs repo, `cc_ng_host.py`, `surfacing.py`, `surface_resolver.py` not touched. Condensate not touched.**

## 1. The four items

| Item | Result |
|---|---|
| **C1 (LOW)** stale `pith_stage3` docstring | **Done.** Step 5 of the docstring (real lines **`cc_ng_organism.py:4815-4825`** at head; the Chief's `4794-4798` was the right spot before my edit grew it) now says what the body does: THE ONE BUDGET RULE, whole or absent; an over-budget line, **including the first, top-ranked one**, is skipped and never emitted over budget; it does not end the prefix; every drop is one INFO line (count, chars, never-fit ids); pinned lines are outside the budget. **D8 CONFIRMED: the first-line overrun guard stays REMOVED** — I did not restore it. Tests: `test_c1_stage3_docstring_describes_the_skip_not_the_removed_guard` (the removed sentence and "never emit an empty L1" are gone, "skipped"/"INFO" present) and `test_c1_a_first_unpinned_line_longer_than_the_budget_is_absent_with_the_info_line` (rank-1 giant absent, its id and size in the INFO line, a pinned 2,000-char line untouched). |
| **C2 (NOTE)** never keep a cut item | **Done.** `_cc_monitor_items_whole` (**`:5632`**): an item whose re-resolve **raises** is now **dropped** (`:5662-5663`, `failed.append(...)` then `continue` — it is never appended) and reported by a **WARNING** (`:5671`) that names the node id and the exception **type** and nothing else (no node text, no exception message). Repeat failures still warn (with the count) but name the id once (`_pith_note_ids`, flood-safe, "already reported"). `surfacing.py` untouched. Non-error "keep as given" cases (unknown node, image frame, resolver returns nothing) are unchanged. Tests: resolver that raises (`test_c2_a_monitor_item_whose_re_resolve_raises_is_dropped_and_warned_without_text` — asserts the secret exception text, the node body and the shared snippet appear nowhere in the log), flood-safety, and the twin test below. |
| **C3 (NOTE)** leftover trees loud | **Done.** `_pith_reference_text` (**`:4663`**) now logs **one INFO line** (`:4683-4688`) — `pith reference form: node <id> shows N of T concept trees whole; L left out (budget B chars, tree cap M)` — whenever it leaves any out, whether the **budget** or the `CC_PITH_PROVIDER_MEMBERS` **cap** stopped them (the second cause was checker-022's other half; the old `_pith_tree_nodes(..., limit)` slice hid it). Counts only, no text; silent when every tree is included. Tests: budget stops it (`2 of 5`, `3 left out`, no tree text in the line), member cap stops it (`3 of 5`), and the all-included guard. |
| **C4 (NOTE)** record only | **Recorded, not changed:** *Pinned Stage-3 lines sit off-budget by design (identity/constitutional); a large pin can make L1 exceed `budget_chars` with no drop INFO. Folding pins into THE ONE rule is an Exec call — an **open note**, not done here.* |

### One thing I found that the brief did not list (fixed as part of C2)
Dropping a monitor item on error would have **lost the node from both streams**: `cc_assemble_recall` de-duplicates the pattern stream against `monitor_node_ids`, which was computed **before** the whole-content re-resolve, so the dropped item's ID still suppressed its (whole) pattern-stream twin. The set is now recomputed from the **surviving** monitor items (`cc_ng_organism.py:5815`; the pre-existing computation at `:5796` is unchanged). Test: `test_c2_a_dropped_monitor_item_does_not_take_its_pattern_twin_with_it` on Pith-ON **and** gate-off (the node appears exactly once, whole, no `…`).

## 2. Commits, diff, proofs

| | |
|---|---|
| base of this turn | `5a634ca2db9ddadcd7c2babec56f18e67caf08b2` (turn 3); the branch head when I started was `b7ee697` = `5a634ca` + checker-022's review file only (`git diff 5a634ca b7ee697` is that one file) |
| **code commit** (the commit before this return file) | **`ed3c1037fa32fadb00099edf56bbf76aacde17aa`** |
| **return commit** = final NG branch head | the commit that adds **this file and nothing else** (parent `ed3c103`); a file cannot name its own commit — the exact hash is in the reply and equals `git rev-parse HEAD` / `git ls-remote origin refs/heads/cc-laptop-pith-clip-813-20260930` |

`git diff 5a634ca HEAD --stat` at the code commit: `cc_ng_organism.py | 64 +++++-` · `tests/test_cc_pith_clip_813.py | 137 +++` · `…/reviews/checker-022-813-delta.md | 242 +++` (the reviewer's file, not mine) → 3 files, 432 insertions, 11 deletions. **Mine: 2 files, 190 insertions, 11 deletions.**

* **`git diff e4ebf982 HEAD --stat -- cc_ng_host.py` → empty** (still byte-identical to base).
* Protected / vendored / shared files touched vs base (`surfacing, surface_resolver, cc_ng_host, neurograph_rpc, kiss_filter, tonic_thread, neuro_foundation, openclaw_hook, stream_parser, activation_persistence, ng_*`): **none**.
* Docs repo: not touched (its branch stays at `fd0ccc9c08b965e6b9c68baebe6f78a09ed36931`).
* Golden vs BASE `e4ebf982` unchanged and passing; P379 preamble passes (`env -u NG_EMBED_REMOTE -u PYTHONPATH`).

## 3. Tests — once
**(A)** the turn-3 union (`test_cc_pith_clip_813, test_pith_provider_context, test_pith_stage1…5, test_pith_l1_provenance, test_pith_metrics_concurrency, test_cc_host_pith_telemetry, test_cc_recall_dedup, test_cc_recall_unification, test_cc_region_confidence, test_pith_history_metrics, test_cc_host_compress_history`; worktree code only, no daemon env) → **270 passed, 3 skipped** (261 at turn 3 + the 9 new; the 3 skips are the host/daemon parity tests). **(B)** parity tests **alone** with `CC_DAEMON_UNDER_TEST` → the docs worktree daemon → **3 passed** (kept separate on purpose; see `build-002.md` §8). RED first: before implementing, the 9 new tests were 7 failed / 2 passed (the two that passed are guards: C1's behaviour was already right — only its docstring lied — and C3's "no line when everything fits").
**Not verified:** anything live (no service, socket, checkpoint, tract, real graph); the WARNING/INFO wording is asserted by tests, not seen in a running daemon.

## 4. Open notes carried forward (unchanged)
D8 confirmed and closed · C4 pins off-budget (Exec call) · #817 deferred to the post-track VPS/daemon lane (function + both handlers together) · D3 wants · D4 deposit-path clips (#741) · D6/#812 shared `surfacing.py` · D7 `MAX_QUEST_CHARS` until card-7 (`88ddfec`) and both hosts move together · the 35-row audit is not a repo-wide cap census.
