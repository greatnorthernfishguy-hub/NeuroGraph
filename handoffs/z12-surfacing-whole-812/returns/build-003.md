<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 lane surfacing-whole-812, dispatch #12684) — build-003 return for #812 TURN 2 PART 2
#   What: return for Part 2: #813 re-based onto the #812 tip on a NEW branch, the interim fork deleted, the :2993 conflict resolved, (g)(3) proven.
#   Why:  Chief-003 ruling (h) + Exec P468/P474. "We fix stuff correctly, not monkey patch or work around."
#   How:  docs-only commit after the Part 2 code commit b1295580. Part 1's return is build-002.md (on the #812 branch).
# -------------------
-->

# build-003 — #812 turn 2, PART 2 return (#813 re-based onto #812, interim fork deleted)

Related: [[NeuroGraph]], [[The Laws]] (LAW 3 reuse, LAW 4 fix at the source, LAW 5 env names), [[Vendored Files]].
Lane `surfacing-whole-812`, dispatch #12684. Worker seat. **Nothing merged, restarted, wired or connected; no PR.**

| | |
|---|---|
| new worktree / branch | `/home/josh/NeuroGraph-worktrees/z12-813-onto-812-20261001` / `cc-laptop-813-onto-812-20261001` |
| base = Part 1 tip (the #812 branch, unmodified) | `e07b9b8c196aec1163e04fddfca82d17646e3b26` |
| replayed range | `e4ebf982b1989fd9066d610b94853bc68bf70d37..84a0968a3622a4ccf793761b50fd6c0cdeee13d1` (36 commits, no merges), now `77f7c9d053e6…d76b6f1c8adbe5da9ca959f6a905ccc4fe33a549` |
| **Part 2 code commit** | `b12955801b044f1c9e854116858b92838a8e82a6` |

The original `cc-laptop-pith-clip-813-20260930` and `cc-laptop-surfacing-whole-812-20261001` published commits were not amended, rebased or force-pushed. The new branch is unpublished until the push at the end of this turn.

---

## The replay and every conflict I resolved

`git cherry-pick` of the 36-commit range onto Part 1's tip stopped **twice**; the other 34 commits auto-merged. Only `cc_ng_organism.py` overlapped with my #812 commits. I resolved with a small policy script that printed every hunk (`/tmp/z12_812/resolve.py`, outside the repo). Policy for the **replay**: keep each replayed commit faithful to #813 as written, so the full (h) resolution lands in the ONE new commit, as briefed.

| # | At (original → replayed) | Hunk | Resolution |
|---|---|---|---|
| 1a | `5a8120a` "#813 build: remove the Pith per-node clip…" → `b67da22` | changelog header (`cc_ng_organism.py:6`) | **both** entries kept, newest first: my 2026-10-01 #812 entry above #813's 2026-09-30 entry |
| 1b | same commit | the `:2993` call site (now `:3155`) | took **#813's** form `max_chars=(sys.maxsize if whole_content else 300)` for the replay (its `whole_content` parameter still exists at that commit); my one-line #812 form was restored by the Part 2 commit (below) |
| 2 | `37f6bf7` "#813 turn 2 (1b): #816 both Pith-ON streams + gate-off, the ONE budget rule…" → `b892201` | changelog header | **both** kept, newest first |

**Defect I introduced and repaired:** I continued each conflict with `GIT_EDITOR=true git cherry-pick --continue`. Git then applied its default `--cleanup=strip`, which **deleted the subject lines of those two commits** because they begin with `#813 …` (a `#` line is a comment). I caught it when listing the replay by subject, compared **all 36** replayed messages against their originals (exactly those two differed), and repaired them on the unpublished branch with a scripted rebase that inserted `git commit --amend --cleanup=verbatim -F <original message>` after each. Verified afterwards: **all 36 replayed messages equal their originals; the tree is byte-identical before and after the repair** (`git diff` empty), so every test result below applies to the final tree. The repair re-hashed the replayed range (and my Part 2 commit); the hashes in this document are the final ones.

---

## What changed in the Part 2 commit `b1295580` (file:line at the tip)

`cc_ng_organism.py`:

| Site | Change |
|---|---|
| `:3027` `cc_pattern_completion_recall` | the `whole_content: bool = False` parameter and its docstring paragraph **deleted**; the docstring now says content is the node's WHOLE resolved text |
| `:3155` | the **merged form** of the #812/#813 conflict: ONE plain `text = resolve_surface_content(node, r, allow_ingested=True)` (no bound, no flag, no `sys.maxsize`) |
| `:5615` (`pith_provider_context`) | the `whole_content=True` argument **deleted** |
| `:5820` `_cc_render_unpithed` | the monitor block is rendered by the **shared** `monitor.format_context(kept_monitor)` (`:5883`), **fail-soft**: an exception loses only the monitor block, with one WARNING naming the exception TYPE, never item text (`:5885`). The old code guarded the same call; "a surfacing pass must not crash the hook" |
| `:5893` `cc_assemble_recall` | the fork's re-resolve call and its pattern-twin dedupe recompute **deleted**; `:5962` the call is `cc_pattern_completion_recall(ng, query, k, state=conv_state)` (the second `whole_content=True` argument deleted); `:6086` still `return _cc_render_unpithed(...)` |
| (deleted) | `_cc_monitor_items_whole` and `_format_cc_monitor_block` with their `INTERIM FORK` pointer comments: 69 lines |
| header | the fork-pointer sentences in four older #813 entries are replaced by a bracketed "superseded 2026-10-01 (#812 turn 2)" marker (history stays honest; no pointer to a deleted function), plus my new turn-2 entry on top |

Not touched: the `_CC_PITH_*` caps, `pith_stage2_keyframe`, `pith_stage3`, `_pith_admit_strict_prefix` (`:4844`), `_pith_log_budget_drop` (`:4634`), `_pith_reference_*`, `cc_l1_budget` (`:4424`). No protected file, no vendored file, no new module, no new environment variable.

### What the fork did, and why nothing is lost

The fork (T1, le-022) existed only because the producer's cut lived in shared `surfacing.py` / `surface_resolver.py`. It did two things:
1. `_cc_monitor_items_whole` re-resolved each SurfacingMonitor item **whole** by `node_id` (from the same node + vdb entry the monitor used), kept what the monitor gave for items not re-resolvable by design (unknown node, image frame, filtered/empty), and **DROPPED, with a WARNING naming the node id and exception type, an item whose re-resolve raised** ("whole-or-absent": never keep the shared 240-char snippet).
2. `_format_cc_monitor_block` was a CC-side twin of `SurfacingMonitor.format_context` without its 200-char cut.

**Nothing is lost:** since Part 1/turn 1, `after_step` stores the resolver's **WHOLE** text and `format_context` renders it whole, so no cut item can reach the CC; the re-resolve and the whole-or-absent drop guarded a state that can no longer occur. The size budget (`_pith_admit_strict_prefix` against `cc_l1_budget`, ONE INFO line) is now the **only** place items are dropped, whole.

### Fork-pinning tests: deleted, rewritten, restored

In `tests/test_cc_pith_clip_813.py` (header entry added there):
- **DELETED:** `test_provider_context_asks_recall_for_whole_content`; `test_816_shared_surfacing_and_resolver_are_not_edited` (false by design now: #812 edits those files); `test_cc_monitor_block_is_layout_identical_to_the_shared_format_context` (the parity test); `test_monitor_items_are_re_resolved_whole_by_node_id_and_fail_soft`; `test_cc_assemble_recall_asks_recall_for_whole_content_by_literal_true`; the three `test_c2_*` monitor re-resolve tests (`…is_dropped_and_warned_without_text`, `…repeated_failures_still_warn_but_name_the_id_once`, `…a_dropped_monitor_item_does_not_take_its_pattern_twin_with_it`) and their helpers `_Boom` and `_raising_resolver`; the now-unused `_shared_surfacing` import.
- **REWRITTEN:** `test_recall_default_still_bounds_the_snippet_and_whole_content_does_not` → `test_recall_default_is_whole`; `_honest_recall` and `_long_world` now model the real (whole) behaviour; the AST cap guard needs `>= 1` resolver call (the CC monitor route is gone).
- **RESTORED byte-identical to the stack base `e4ebf982`** (verified with `git diff --quiet`): `tests/test_cc_recall_unification.py` (so it runs "unchanged", as briefed), `tests/test_cc_recall_dedup.py`, `tests/test_cc_region_confidence.py`. Their only #813 edits existed for the fork (`_format_cc_monitor_block` in the expected strings; `**kw` for the `whole_content` flag).
- Every #813 test not about the fork still passes (97 in `test_cc_pith_clip_813.py`).

---

## (g) the size budget on the CC side: KEY CHECK answered and PROVEN

**Names/defaults as ruled (Exec P474), recorded again here:**
- CC paths, Pith-ON **and** Pith-OFF: **reuse `CC_PITH_L1_BUDGET`**, default **4000**, clamp **500–40000** (`cc_ng_organism.py:3672-3673`), via `cc_l1_budget` (`:4424`). **No new variable.**
- Shared/Syl paths: `NG_SURFACE_BUDGET_CHARS` = 3000, clamp 500–40000: a **ROLLOUT item** (Josh sets it via `.bashrc` / `openclaw.json`); **its mechanism is the STOPPED sub-item, see `build-002.md`** (on the #812 branch). Nothing in this Part touches it.

**Key-check answer: YES, the CC Pith-OFF DEFAULT path is already budgeted by #813's `_cc_render_unpithed`; after the re-base no gap remains.** `cc_assemble_recall` skips the Pith block when `_CC_PITH_ENABLED` is off (its code default is `"0"`) and ends with `return _cc_render_unpithed(...)` (`:6086`). The renderer takes `budget = cc_l1_budget(commons, graph, pc_fired_ids)` (fallback `_CC_PITH_L1_BUDGET`), ranks all items by `_pith_unified_rank`, admits by `_pith_admit_strict_prefix`, drops the lowest-ranked WHOLE items with ONE INFO line (count, chars, kept), exempts identity-protected items, and swaps an item that alone exceeds the budget for the P417 reference form (its trees whole + a one-line whole-node reference). So the ruled mechanism, including "always keep the top item", is the existing one.

**Proof (`tests/test_cc_pith_off_budget_812.py`, new, 7 tests, real source chain, gate at its code default OFF):** a real `SurfacingMonitor` fired on real (fake-graph) nodes + the real `cc_pattern_completion_recall` + the real `cc_assemble_recall`; only the GSG re-score and novelty are stubbed (they need the embedder).
- 1 monitor item + 3 pattern items, each 1500 chars WHOLE (6000 > 4000): the monitor item arrives whole **at the source**; top-ranked kept WHOLE; the two lowest-ranked dropped WHOLE (`p1[:20]`/`p2[:20]` absent, never a piece); **exactly one** INFO record: `L1 budget 4000 chars … dropping 2 whole items (3000 chars) … kept 2 (3000 chars)`; no `…`/`...` in the output.
- Nothing dropped → no INFO line, everything whole, output starts with the shared marker `[NeuroGraph Surfaced Knowledge]`.
- **Top item alone over the budget** (a ~30000-char node): kept as the reference form (not dropped, never cut mid-text), its three trees whole, `GIANT-START`/`GIANT-END` absent, the reference-limit INFO line emitted.
- A raising `monitor.format_context` loses only the monitor block; one WARNING naming `RuntimeError`, no item text.
- (h): the fork functions and the `whole_content` parameter are gone; the recall call is one plain resolver call (a 2000-char node comes back whole).
- The gate default is OFF in the source (regex on the `os.environ.get("CC_PITH_ENABLED", "0")` line).

---

## Tests and runs

**Environment note (names only):** my shell exports 17 `CC_PITH_*` variables, including `CC_PITH_ENABLED=1` and `CC_PITH_PREFETCH_ENABLED=1` (the laptop's live config has Pith ON; the code default is OFF). Every Part 2 run **scrubbed all `CC_PITH_*`** so the module defaults are what is tested, and used `PYTHONUSERBASE=/home/josh/.local PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, `EXPECT_NG_ROOT` = the worktree.

**Printed-path preamble (as printed by the session; all 8 loaded NG root modules under the worktree, `OK`):**
```
[#770 preamble] test root (must contain every NG module under test): /home/josh/NeuroGraph-worktrees/z12-813-onto-812-20261001
[#770 preamble]   surface_resolver.__file__ = …/z12-813-onto-812-20261001/surface_resolver.py
[#770 preamble]   surfacing.__file__ = …/surfacing.py
[#770 preamble]   neurograph_rpc.__file__ = …/neurograph_rpc.py
[#770 preamble]   cc_ng_organism.__file__ = …/cc_ng_organism.py
[#770 preamble]   ces_config.__file__ = …/ces_config.py
[#770 preamble]   tonic_thread.__file__ = …/tonic_thread.py
[#770 preamble]   cc_ng_host.__file__ = …/cc_ng_host.py
[#770 preamble]   ng_salience_gate.__file__ = …/ng_salience_gate.py
[#770 preamble] OK: all 8 loaded NG root modules are under the test root
```
(`…` = `/home/josh/NeuroGraph-worktrees/z12-813-onto-812-20261001`.)

### What happened with the "one targeted run" (full disclosure: it took 3 more runs, and a side effect)

1. **The one authorised run hung** (my `timeout 500` → exit 124, no summary; 172 results printed). `tests/test_cc_pith_clip_813.py` scrubs `NG_EMBED_*` at import (its own P379 preamble), which for the whole session flips the embedder from `NG_EMBED_REMOTE` (fast "HF token unavailable") to **loading the local model**; with the scratch `HOME` empty, `ng_embed._ensure_model` → `hf_hub_download` **fetched the public ONNX model over the network** (~106MB, no token). Test #173 (`test_pith_stage4::test_promotion_noop_when_gated_off`) sat in that load.
2. **Diagnostic runs (beyond the one authorised):** that test alone passes in 4.3s (order-dependent, not a code error); the ordered 6-file prefix with `faulthandler_timeout` showed the stall inside `ng_embed._ensure_model` and completed: **189 passed, 2 xfailed in 188s**, all 20 `test_pith_stage4` tests passing with the model reachable.
3. **Side-effect cleanup:** the two diagnostic scratch `HOME`s that held the model download (`/tmp/tmp.XK0jiowpfD` 106MB, `/tmp/tmp.fQvv31Year`) were identified by birth time = my run start times and **removed**, with the 21:36:25 single-test one; scratch dirs I could not attribute to my runs were left alone.
4. **The final complete run** used no network: a fresh scratch `HOME` seeded by copying **only** the `models--Snowflake--snowflake-arctic-embed-m-v1.5` directory from `~/.cache/ng_embed` (read-only; the live `failed_embeds.jsonl` was not copied or touched), `HF_HUB_OFFLINE=1`. It is removed after the run.

**Final run** (ended 05:42:13 UTC, 16.35s):
`tests/test_surfacing_whole.py tests/test_cc_pith_off_budget_812.py tests/test_cc_pith_clip_813.py tests/test_pith_stage2.py tests/test_pith_stage3.py tests/test_pith_stage4.py tests/test_pith_provider_context.py tests/test_cc_recall_unification.py tests/test_cc_recall_dedup.py tests/test_cc_region_confidence.py`
→ **4 failed, 253 passed, 3 skipped, 2 xfailed, 7 errors.** None is caused by this change:

| Count | Tests | Cause (evidence) |
|---|---|---|
| 7 errors | all of `test_cc_recall_unification.py` that use the `daemon_mod` fixture | `FileNotFoundError: …/docs/scripts/cc-ng-daemon.py`: the test loads `~/docs/scripts/cc-ng-daemon.py`, which does not exist under a scratch `HOME`. **Supplementary run** (below) supplies that one script, read-only, and all 25 pass |
| 4 failed | `test_pith_stage4::test_promotion_already_surfaced_node_not_double_counted`, `…_is_pure_additive_not_a_hard_override`, `…_lod_keeps_near_content_full`, `…_lod_far_content_stays_whole` | `ng_embed.EmbeddingUnavailableError: embedding model unavailable`, from a bare `embed(...)` in each test's own setup: the **#882 class** (not mine). In the diagnostic run with the model reachable all 20 stage4 tests passed |
| 3 skipped | `test_pith_provider_context.py:487` | "cross-repo parity requires `CC_DAEMON_UNDER_TEST` to name the candidate daemon" (environmental) |
| 2 xfailed | the two shared-budget SPECS in `test_surfacing_whole.py` | the STOPPED sub-item (`build-002.md`) |

Passes per file: `test_cc_pith_clip_813` 97, `test_surfacing_whole` 47, `test_pith_provider_context` 22, `test_cc_region_confidence` 24, `test_pith_stage4` 16, `test_pith_stage2` 9, `test_pith_stage3` 9, `test_cc_pith_off_budget_812` 7, `test_cc_recall_dedup` 4, `test_cc_recall_unification` 18 (the other 7 are the errors above).

**Supplementary run** (the unification file needs its daemon script; one read-only copy of `~/docs/scripts/cc-ng-daemon.py` placed in a scratch `HOME`; its `main()` is behind `if __name__ == '__main__'` and the test loads it under another module name, so nothing starts): `tests/test_surfacing_whole.py tests/test_cc_recall_unification.py` → **72 passed, 2 xfailed in 7.98s; `test_cc_recall_unification.py` 25/25 unchanged.**

### Fail-before (the new Part 2 tests against a `git archive` of the replay tip `301f6823…`, `data/` `docs/` excluded, outside the repo; preamble `OK` for its root)

**5 failed, 49 passed, 2 xfailed in 2.96s.** The 5, and why:

| Test | Fails on the replay tip because |
|---|---|
| `test_cc_pith_off_budget_812::test_h_the_interim_fork_and_the_whole_content_flag_are_gone` | the fork functions and the `whole_content` parameter exist there |
| `…::test_h_the_recall_call_is_one_plain_resolver_call` | at the replay tip the recall site is #813's `max_chars=(sys.maxsize if whole_content else 300)`: default clips at 300 |
| `…::test_pith_off_a_failing_monitor_formatter_never_crashes_the_hook` | the renderer used the CC-side formatter and never called `monitor.format_context` |
| `test_surfacing_whole::test_cc_pattern_completion_recall_renders_whole` ×2 (turn-1 test, >300 and >1000) | **the `:2993` conflict in action:** the replayed #813 form clips at 300 by default, undoing turn 1's fix until the Part 2 commit restores the plain call |

The other 4 new tests (budget binds, nothing dropped, top item over budget, gate default) **pass on both**, which is exactly the key-check answer: #813 already budgets the Pith-OFF default; Part 2 adds the proof, not the mechanism.

---

## Not verified / limits

- No real graph, daemon, checkpoint or `data/` path; no Syl data; real `_forest_content` lengths unknown. The CC-side proof uses fake graphs with the real functions.
- **Shared/Syl budget (`NG_SURFACE_BUDGET_CHARS`) is not implemented** (the STOPPED sub-item, `build-002.md`); the #812 and this branch's Syl-facing blocks are count-bounded only until the Executive rules.
- The 4 stage4 HF-class tests and the 7 unification errors in the **final** run are environmental (above); they passed in the diagnostic/supplementary runs, which were not the single final run.
- The three runs beyond the one authorised, and the one network model download (public model, no token, scratch dir, removed), are disclosed above.
- The Anima (Rust) gateway side was not read. Vault docs/wikilinks and the punchlist were not updated by me (outside this brief's commits).
- `tests/test_cc_pattern_completion_recall.py` and `tests/test_cc_retrieval_enrichment.py` (embedder-dependent) were not in this brief's Part 2 list and were not run on this branch.
