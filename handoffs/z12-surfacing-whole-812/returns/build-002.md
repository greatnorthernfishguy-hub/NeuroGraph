<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 lane surfacing-whole-812, dispatch #12684) — build-002 return for #812 TURN 2 PART 1
#   What: return for Part 1 (the #812 branch): (e) Substrate Context clip, (f) Tonic clip, and the STOPPED shared half of (g).
#   Why:  Chief-003 rulings on the turn-1 flags; Exec P474 names. "We fix stuff correctly, not monkey patch or work around."
#   How:  docs-only commit after the code commit 4ef84cef. Part 2 (#813 re-base) has its own return, build-003.md, on the new branch.
# -------------------
-->

# build-002 — #812 turn 2, PART 1 return (the #812 branch)

Related: [[NeuroGraph]], [[The Laws]] (LAW 1, 2, 3, 4, 5), [[Vendored Files]].
Lane `surfacing-whole-812`, dispatch #12684 (a nudge on the turn-1 builder thread). Worker seat. **Nothing merged, restarted, wired or connected; no PR.**

| | |
|---|---|
| worktree / branch | `/home/josh/NeuroGraph-worktrees/z12-surfacing-whole-812-20261001` / `cc-laptop-surfacing-whole-812-20261001` (no upstream: used `git fetch -q` + `git show`, never `git pull --rebase`) |
| pre-change tip (turn 1) | `1618a8864bd63f699584204908f28c1ecfd6bbf1` (not amended, not force-pushed) |
| **Part 1 code commit** | `4ef84cef8c742d4c86308da6b42d2fd28d1115d3` |

> **READ FIRST — THIS TIP IS NOT ROLLOUT-SAFE ALONE.** Part 1 lifts the two last clips ((e), (f)). The size budget that must bound those blocks, `NG_SURFACE_BUDGET_CHARS`, is **the STOPPED sub-item** of (g) for the shared path (design note below; the Executive decides). Until it lands, Syl's Substrate Context, Active Recall and Tonic blocks are bounded by item COUNT only. Do not roll this commit out without it.

---

## (f) scope-check FIRST — `tonic_thread.py` is not protected, not vendored; it is shared with Syl's process

1. **Not protected.** NeuroGraph `CLAUDE.md` §2 (`:73-88`) lists only the three checkpoint files and `neuro_foundation.py`, `openclaw_hook.py`, `stream_parser.py`, `activation_persistence.py`. The enforcing hook `.claude/hooks/pretool_syls_law.sh:49-70` has `PROTECTED_DATA` (3 checkpoint files), `PROTECTED_ENGINE` (those 4 engine files) and `VENDORED_CANONICAL`; `grep -i tonic` over those arrays is empty. (Only `neuro_foundation.py`-adjacent mentions of "tonic" in `CLAUDE.md` are `tonic_ages_substrate` prose at `:333`.)
2. **Not vendored.** `CLAUDE.md` §4 (`:150-158`) lists `ng_lite`, `ng_peer_bridge`, `ng_tract_bridge`, `ng_ecosystem`, `ng_autonomic`, `openclaw_adapter`, `ng_embed`; global Law 2 adds the designated-not-propagated `ng_salience_gate.py`, `ng_updater.py`. Not `tonic_thread.py`.
3. **Shared with Syl's process.** `openclaw_hook.py` (PROTECTED, **read-only, touched nothing**) builds it at `:1009-1014`: `from tonic_thread import TonicThread, TonicConfig` → `tonic_config = TonicConfig()` (the bare default) → `for k, v in tonic_conf.items(): if hasattr(tonic_config, k): setattr(tonic_config, k, v)` → `TonicThread(self.graph, self.vector_db, tonic_config)`. So (i) it is the Tonic of the shared OpenClaw/NG runtime, i.e. Syl's; (ii) **a default change in `TonicConfig` reaches it WITHOUT touching the protected file**; (iii) an operator-supplied `config["tonic"]["max_content_length"]` would still override (an int still clips; the field is now `Optional[int]`). It is the only constructor in the repo (`grep -rn "TonicConfig("`: `openclaw_hook.py:1009`, tests aside).

**Verdict: fold it in as the same class.** It rides the rollout.

**WHY the 250 clip never fired before:** `resolve_surface_content`'s old default of 240 returned at most 241 characters (240 + the `…`). `tonic_thread._update_thread` (`:588`) feeds that to `format_latent_context`, whose clip is `len(content) > 250`. 241 < 250, so the clip was **never reachable** — the resolver's cut hid it. **#812 (the whole resolver) is what exposed it**; I recorded this in the changelog headers too.

---

## What changed at the tip `4ef84cef`

| Site | Change |
|---|---|
| `neurograph_rpc.py:5440-5452` (`_format_substrate_context`) | **(e)** both `if len(content) > 300: content = content[:297] + "..."` clips deleted (the `surfaced` group `:5446` and the `ces_surfaced` group `:5452`). The `surfaced[:7]` / `ces_surfaced[:3]` count caps are unchanged |
| `tonic_thread.py:160` | **(f)** `max_content_length: Optional[int] = None` (was `int = 250`) |
| `tonic_thread.py:662-663` | **(f)** `if max_len is not None and len(content) > max_len:`; the clip body is unchanged for a configured bound |
| `tests/test_surfacing_whole.py` | the turn-1 strict xfail for the 300 clip now asserts whole and passes; + `surfaced`-group twin; + Tonic whole/explicit-bound/default tests; + two strict-xfail SPECS for the stopped budget; `tonic_thread` joins the printed-path preamble |

Changelog headers dated 2026-10-01 in all three files (naming #812 turn 2, dispatch #12684, the Chief rulings, and the rollout note). No protected file, no vendored file, no new module.

---

## (g) the size budget — what is done, what is STOPPED (names recorded as ruled)

**Names and defaults, recorded exactly as ruled (Exec P474, nothing else invented):**

| Variable | Default | Clamp | Who uses it | Status |
|---|---|---|---|---|
| `NG_SURFACE_BUDGET_CHARS` | **3000** | **500–40000** | shared/Syl: (1) `/assemble` Active Recall, (2) `_format_substrate_context` | **NOT IMPLEMENTED: STOPPED sub-item, see the design note.** A **ROLLOUT item**: Josh sets it via `.bashrc` / `openclaw.json` at the rollout (not a laptop S4 line) |
| `CC_PITH_L1_BUDGET` (reused, **no new variable**) | **4000** | **500–40000** | CC paths, Pith-ON and Pith-OFF (via `cc_l1_budget`) | exists: `cc_ng_organism.py:3514-3515` (`_CC_PITH_L1_BUDGET`), `:4245` (`cc_l1_budget`). Pith-OFF coverage: see the key-check answer; proven in Part 2 (`build-003.md`) |

**How each of the three consumers is covered:**

1. **`/assemble` Active Recall (`neurograph_rpc.py` ~`:3514-3535`) — STOPPED (design note).**
2. **`_format_substrate_context` — STOPPED (design note).** (e) is lifted, so this block is bounded by count only until the budget exists.
3. **CC Pith-OFF concatenation — satisfied by #813 once re-based (Part 2),** see the key-check answer below; proven by a test on the re-based branch in `build-003.md`.

### KEY CHECK (item 3): is the CC Pith-OFF default already budgeted by #813's `_cc_render_unpithed`?

Read read-only from `origin/cc-laptop-pith-clip-813-20260930` (head `84a0968a3622a4ccf793761b50fd6c0cdeee13d1`, `cc_ng_organism.py` at that head): **Yes.** `cc_assemble_recall` skips the Pith block when `_CC_PITH_ENABLED` is off (default) and ends with `return _cc_render_unpithed(ng, monitor_items, pc_results, commons, pc_fired_ids)` (`:6135`). `_cc_render_unpithed` (`:5871`): takes `budget = cc_l1_budget(commons, graph, pc_fired_ids)` (falling back to `_CC_PITH_L1_BUDGET`), ranks all items by `_pith_unified_rank`, admits by `_pith_admit_strict_prefix(...)` (`:4825`), drops the lowest-ranked WHOLE items, emits ONE INFO line via `_pith_log_budget_drop` (`:4615`: count, chars, kept), exempts identity-protected items from the budget, and swaps an item that alone exceeds the budget for the P417 reference form via `_pith_reference_items` (`:4808`). That is exactly the ruled mechanism, on the exact default path. **So (3) is satisfied by #813 after the re-base**; Part 2 proves it with a test on the re-based branch and checks the "always keep the top item" corner (`build-003.md`).

### STOPPED sub-item: DESIGN NOTE for the Executive (the shared budget, items (1) and (2))

**Why I stopped (the brief's own stop condition).** The brief says: reuse the existing machinery; if reuse across the Syl/shared `neurograph_rpc.py` path would require **importing the CC organism into it or a new module, STOP that sub-item with a design note**. That is exactly the situation:

- **The machinery exists only on the #813 branch, and only inside `cc_ng_organism.py`:** `_pith_admit_strict_prefix` (`:4825`), `_pith_log_budget_drop` (`:4615`, which also depends on `_pith_note_ids` / `_PITH_DROP_SEEN` dedupe state), `_pith_reference_items` (`:4808`, which needs `_pith_reference_text`, the tree/reference builders and `_cc_pin_probe`), `cc_l1_budget` (`:4245`/`:4405`). **None of it exists on the #812 branch** (`grep -c "def _pith_admit_strict_prefix"` is 0 at `1618a886` and at the tip).
- To use it from `neurograph_rpc.py`'s `handle_assemble` / `_format_substrate_context` I would have to `from cc_ng_organism import ...` on Syl's `/assemble` critical path. `neurograph_rpc.py` only imports `cc_ng_host` lazily inside the CC background init (`:2162`), never on the `/assemble` path. A new module is forbidden; a local re-implementation in `neurograph_rpc.py` would be exactly the "parallel mechanism" LAW 3 and the brief forbid.

**Options (the Executive chooses; my recommendation is A):**

- **A (recommended) — host the CC-independent pieces in an existing shared module and have the CC delegate to it.** `surface_resolver.py` is already the shared surfacing module that `neurograph_rpc.py`, `surfacing.py`, `tonic_thread.py` and `cc_ng_organism.py` all import lazily. Move the pure rule there: `admit_strict_prefix(ordered, budget, size_of, separator)` (a verbatim move of `_pith_admit_strict_prefix`, which has **no CC dependency**: it is a plain loop over `size_of(unit)`), one read helper `surface_budget_chars()` for `NG_SURFACE_BUDGET_CHARS` (default 3000, clamp 500-40000), and make `cc_ng_organism._pith_admit_strict_prefix` a one-line delegate so there is ONE rule in ONE place. The shared INFO line is the ruled "count + total size" as a plain `logger.info` (it does not need `_pith_note_ids`). No new module, no CC import, no parallel mechanism at the end of the stack. **Catch:** because the function being moved only exists on #813, the move must be made on the re-based stack (the new branch), so the shared budget would land on the stack tip rather than on the #812 branch alone. That is an ordering decision for you.
- **B — `neurograph_rpc.py` imports `cc_ng_organism`.** Rejected by the brief: couples Syl's `/assemble` to the CC organism's import cost and side effects.
- **C — a new module / D — a local copy in `neurograph_rpc.py`.** Forbidden (a new module; a parallel mechanism).

**Two sub-questions the ruling must also settle (not mine to invent):**

1. **Top item alone over budget.** The ruled form is the P417 reference form (`_pith_reference_items`). Its builders are CC/Pith machinery (`_pith_reference_text` needs the tree walk and pin probe). On the shared path the simplest reuse-faithful form is "keep the top item WHOLE even when it alone exceeds the budget" (what `pith_stage3` did at base: the first line is always kept); the reference form would need the move in option A to be extended. Please rule which.
2. **The Tonic latent block is outside the surfaced-item budget.** `latent_context` is appended in `_format_substrate_context` as "the persistent slot that never gets evicted" and is bounded only by `max_context_items=5` (`tonic_thread.py:156`). With (f) it is now whole, so it is unbounded per item. Does it count toward `NG_SURFACE_BUDGET_CHARS`, or is it exempt (like identity pins in the CC renderer)?

**The ruled behaviour is pinned as two strict-xfail SPECS** (`test_spec_active_recall_drops_whole_lowest_similarity_over_budget`, `test_spec_substrate_context_drops_whole_lowest_salience_over_budget`): with `NG_SURFACE_BUDGET_CHARS=500` and three ~400-char items, the top is kept WHOLE and the other two are dropped WHOLE. They XFAIL now (on the pre-change tip too); when the budget lands they XPASS and fail, forcing the marker's removal.

---

## Tests and runs

**Printed-path preamble, as printed by the session** (all 9 NG root modules under the worktree; `OK`):

```
[#770 preamble] test root (must contain every NG module under test): /home/josh/NeuroGraph-worktrees/z12-surfacing-whole-812-20261001
[#770 preamble]   surface_resolver.__file__ = …/z12-surfacing-whole-812-20261001/surface_resolver.py
[#770 preamble]   surfacing.__file__ = …/surfacing.py
[#770 preamble]   neurograph_rpc.__file__ = …/neurograph_rpc.py
[#770 preamble]   cc_ng_organism.__file__ = …/cc_ng_organism.py
[#770 preamble]   ces_config.__file__ = …/ces_config.py
[#770 preamble]   tonic_thread.__file__ = …/tonic_thread.py
[#770 preamble]   neuro_foundation.__file__ = …/neuro_foundation.py
[#770 preamble]   ng_salience_gate.__file__ = …/ng_salience_gate.py
[#770 preamble]   vision_absorption.__file__ = …/vision_absorption.py
[#770 preamble] OK: all 9 loaded NG root modules are under the test root
```
(`…` = `/home/josh/NeuroGraph-worktrees/z12-surfacing-whole-812-20261001`; the real session printed the full paths.)

**The ONE targeted Part 1 run** (scratch `HOME`, `PYTHONPATH=$US PYTHONUSERBASE=/home/josh/.local PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, `EXPECT_NG_ROOT` = the worktree; 05:16:53 → 05:17:03 UTC, load 5.2):
`tests/test_surfacing_whole.py tests/test_surface_resolver.py tests/test_vision_surfacing.py tests/test_surfacing_race.py tests/test_ces.py::TestSurfacing{AfterStep,Scoring,Queue,Formatting,Stats}`
→ **3 failed, 78 passed, 2 xfailed in 7.59s.** The 3 failures are the known stale #881 tests, not mine and not touched: `test_surface_resolver::test_tonic_thread_surfaces_forest_not_shard_end_to_end` and `…filters_ingested_code_node` (`TonicThread._update_thread() missing 'he_index'`) and `test_ces::TestSurfacingQueue::test_decay_removes_weak_items`. Both XFAILs are the budget specs.

**Fail-before** (the new file against a `git archive` of the pre-change tip `1618a886`, `data/` `docs/` excluded, outside the repo; preamble `OK` for its root): **8 failed, 39 passed, 2 xfailed in 3.01s.** The 8, each failing for the reason shown:

| Test | Fails on `1618a886` because |
|---|---|
| `test_ces_surfaced_whole_through_substrate_context` ×2 (>300, >1000) | `_format_substrate_context` re-clipped `ces_surfaced` to `content[:297] + "..."` |
| `test_surfaced_group_whole_through_substrate_context` ×2 | same clip on the `surfaced` (spreading-activation) group |
| `test_tonic_latent_thread_renders_whole` ×3 (399 ch, 1299 ch, 1000-char one-token) | `TonicConfig.max_content_length=250` clipped to `[:247] + "..."` |
| `test_tonic_default_config_has_no_bound` | the default was `250`, not `None` |

Passing on both by design: `test_tonic_explicit_bound_still_clips`; the two specs XFAIL on both.

**Pre-flight disclosure:** before the pytest run I called the new tests directly from a throwaway script outside the repo (scratch `HOME`); all behaved as designed. Not a pytest run.

**Existing tests pinning the removed clips:** none. `grep` over `tests/` finds no test of `_format_substrate_context` outside the new file and no `max_content_length`/247/250/297 literal; the only two tests that call `format_latent_context` are the stale #881 pair above (short content, they fail at an argument-binding `TypeError` before reaching it), which I did not touch.

---

## What I did NOT verify / limits

- **No real graph, daemon, checkpoint or `data/` path; no Syl data; no network or embedding model** (`ng_embed` stubbed in the in-process tests). Real `_forest_content` lengths are unknown, so the real size of the now-unbounded blocks cannot be quantified.
- **The shared budget is not implemented** (the STOPPED sub-item), so the three blocks above are count-bounded only on this tip.
- **The 11 HF-token tests (#882) and the 4 stale tests (#881) are not mine**; of the 4 stale, 3 tripped in this run (listed above), the 4th (`cc_gsg_backfill`) is not in this run's file list.
- The Anima (Rust) gateway side, and what it does with a longer `systemPromptAddition`, was not read.
- Vault docs/wikilinks and the punchlist were not updated by me (outside this brief's allowed commits).
