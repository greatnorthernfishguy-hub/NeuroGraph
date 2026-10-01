<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 lane surfacing-whole-812, dispatch #12861) — build-005 return for the #812 N-2 fold
#   What: return for the fold: N-2 restored (monitor-formatter failure reported + pattern twins back), N-3 pinned, N-4/N-5 coverage.
#   Why:  le-043 N-2..N-5 / checker-034 N8..N11; Chief-003 fold approved (row #892). "We fix stuff correctly, not monkey patch or work around."
#   How:  docs-only commit after the code commit 92371a67.
# -------------------
-->

# build-005 — #812 N-2 fold return

Related: [[NeuroGraph]], [[The Laws]] (LAW 3 restore, LAW 4, LAW 5), [[Vendored Files]].
Lane `surfacing-whole-812`, dispatch #12861 (a nudge on the #812 builder thread). Worker seat. **Nothing merged, restarted, wired or connected; no PR.**

| | |
|---|---|
| worktree / branch | `/home/josh/NeuroGraph-worktrees/z12-813-onto-812-20261001` / `cc-laptop-813-onto-812-20261001` (no upstream: `git fetch -q`, never `git pull --rebase`; published commits untouched) |
| pre-fold tip | `55d56b9ade59572e9cbf01e3f55756d6cf7cb76f` |
| **fold code commit** | `92371a67e7f376ad9f3fab0957b0bbdc4dd0c41e` (`cc_ng_organism.py` + `tests/test_cc_pith_off_budget_812.py` only) |

---

## N-2 — what the old guard did, what turn 2 lost, and the fix

**At `e4ebf982`** `cc_assemble_recall` ran `monitor.format_context(monitor_items)` INSIDE the `get_surfaced` try. Two branches:
- `except RuntimeError` (the "dict mutation race"): silent; `monitor_items` and `monitor_node_ids` reset.
- `except Exception as exc`: debug log, **`on_monitor_error(exc)`** (the hemisphere error stat: `cc_ng_host._recall` and the daemon's `_recall` both pass `on_monitor_error=_bump_error`), reset.
Reset meant the monitor stream was empty, so the pattern dedup (`not in monitor_node_ids`) removed nothing: **the pattern twins stayed**.

**At the pre-fold tip** the only `format_context` call was in `_cc_render_unpithed`, after the dedup had already removed the twins, wrapped in a guard that only warned. So a formatter failure (a) bumped no stat and (b) left a node present in both streams in neither (le-043 N-2, "the packet's own comparison point": I understated this in `build-003.md`, which said only that the old code "guarded the same call").

**The fix (smallest that restores both; `cc_ng_organism.py:6118-6130`, plus `pc_all` at `:5970`, `:5981`, `:5986`):** immediately before the single `return _cc_render_unpithed(...)` in `cc_assemble_recall`, call `monitor.format_context(monitor_items)` once, as the old guard did. On failure: one WARNING naming the exception TYPE (never its text, which can carry item text); `on_monitor_error(exc)` unless it is a `RuntimeError` (mirrors the old silent branch exactly); `monitor_items = []`; `pc_results = pc_all`, the pattern list copied just before the display dedup, so the twins come back. ~14 lines, no signature change, no budget change.

**Why this place and not another** (alternatives I rejected):
- *Put the call back inside the `get_surfaced` try:* it would run on the Pith-ON success path too, which today never formats; a formatter failure would start emptying the Pith-ON monitor stream. Not "unchanged".
- *Pass `on_monitor_error`/`pc_all` into `_cc_render_unpithed`:* a signature change, and the renderer formats AFTER the budget, so recovering would mean re-running the budget (a second INFO line for one render).
- *The chosen spot* is reached by exactly the two entries to the renderer (gate OFF, or a Pith failure falling through) and by nothing else, so it covers both with one block.

**Why the Pith-ON path is unchanged:** a Pith-ON success builds its `CacheLine`s from the monitor ITEMS and `return`s inside the Pith block, above `:6118`; the monitor formatter is never called, so there is no formatter failure to report. Pinned: `test_n2_pith_on_success_path_never_formats_so_it_is_unchanged` asserts zero formatter calls and no report. **Documented difference from `e4ebf982`:** there the formatter ran before the gate on every path, so a (practically unreachable) formatter failure also emptied the Pith-ON monitor items; that incidental coupling is not restored. Two other intended differences: (1) a `RuntimeError` is still not reported but now gets the type-only WARNING (the old branch logged nothing); (2) on the un-Pithed path a successful render calls the formatter twice (the guard, then the kept subset), a join over at most a handful of dicts. The renderer's own guard stays as the last resort.

## N-3 — dedupe-before-budget: recorded and PINNED, not changed

The N-2 fix does not resolve it: the display dedup (`:5982`) still runs before the budget. If the budget drops a node's monitor copy (monitor weight 0.6) after its pattern twin was deduped away, the node vanishes whole. `test_n3_pins_dedupe_before_budget_a_node_whose_monitor_copy_is_budget_dropped_can_vanish` pins the CURRENT behaviour (X in both streams; its monitor copy ranks last and is budget-dropped; X appears nowhere; one INFO line). A later fix must flip this test deliberately. No budget redesign was made.

## N-4 / N-5 — coverage added (all in `tests/test_cc_pith_off_budget_812.py`, extended; nothing duplicated)

- **Twin survival** (`test_n2_a_formatter_failure_is_reported_and_the_pattern_twin_survives`, ×2 entries: gate OFF and a Pith failure): formatter raises; the pattern twin renders; `on_monitor_error` called once with the exception; the monitor block is absent; the WARNING carries the type, not the text.
- `test_n2_runtimeerror_stays_silent_as_at_e4ebf982_but_the_twin_still_returns`; `test_n2_the_formatter_failure_takes_the_same_route_as_a_harvest_failure` (both failure points call the same callback once with a `ValueError`).
- **Stat route, through the real wrappers:** `test_n2_the_host_hemisphere_error_stat_is_bumped_by_a_formatter_failure` (real `cc_ng_host._recall`: `_STATE.stats['errors']` +1, twin renders) and `test_n2_the_daemon_hemisphere_error_stat_is_bumped_by_a_formatter_failure` (the daemon's `_recall`, loaded the way `test_cc_recall_unification.py` loads it, skipped if the script is absent).
- `test_n2_the_renderers_own_guard_is_the_last_resort` (a second-call failure loses only the block).
- **Reference form, second shape, through the real Pith-OFF `cc_assemble_recall`:** `test_pith_off_an_over_budget_MONITOR_item_surfaces_as_trees_plus_reference` (the over-budget item arrives through the monitor stream, whole at the source: trees whole, one-line reference, giant text never emitted). The existing pattern-stream test is untouched.
- **N-4 qualifier pinned:** `test_pith_off_an_over_budget_item_with_no_reference_form_is_dropped_loudly_never_cut` ("always keep the top item" holds only while the P417 reference form can be built; an over-budget item whose node is unknown is a never-fit, dropped whole with the INFO line naming it). I first tried to force it with a tiny budget; `cc_l1_budget` clamps to 500-40000, so the reference always fits there; the unknown-node shape is the real one.
- **Second budget-binding shape:** `test_budget_binds_second_shape_many_small_items_under_a_tight_budget` (count-driven: budget 1000, eight 300-char items; strict prefix keeps three, drops five whole, one INFO line `dropping 5 whole items (1500 chars) … kept 3 (900 chars)`, output in rank order). The existing size-driven test is the first shape.

---

## Evidence

**Environment (stated):** my shell exports `CC_PITH_ENABLED=1` and 16 other `CC_PITH_*` variables plus `NG_EMBED_REMOTE` (names only). **All were scrubbed** for every run (`env -u …`), with `HF_HUB_OFFLINE=1`, `PYTHONUSERBASE=/home/josh/.local`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, `EXPECT_NG_ROOT` = the worktree.

**Scratch `HOME` (no model download):** a fresh `mktemp -d` seeded by COPYING, read-only, only the two `models--Snowflake--snowflake-arctic-embed-m-v1.5` directories: the ONNX one from `~/.cache/ng_embed/` and the **tokenizer** one from `~/.cache/huggingface/hub/`. Last time only the ONNX directory was seeded, and the model's separate `Tokenizer.from_pretrained` lookup went to the (empty) hub cache, which is why 4 embedder-dependent tests failed "embedding model unavailable". The hub's `token` file was **not** copied (the scratch tree contains `hub`, no `token`). **`tests/test_cc_recall_unification.py` and the daemon test:** `$HOME/docs/scripts/cc-ng-daemon.py` is a single read-only COPY of the daemon worktree's script (`/home/josh/docs/.claude/worktrees/daemon-recall-756-20260930/scripts/cc-ng-daemon.py`, 152,573 bytes; the primary docs checkout's copy, which I used last time, is the older 142,179-byte file). Its `main()` is behind `if __name__ == '__main__'` and both tests load it under another module name, so nothing starts. No test downloaded a model (the log contains no download line).

**Printed-path preamble (as printed by the session; 8 NG root modules, all under the worktree, `OK`):**
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

**The ONE targeted run** (06:31:01 → 06:31:19 UTC, 15.55s, load 6.3 at start): `tests/test_surfacing_whole.py tests/test_cc_pith_off_budget_812.py tests/test_cc_pith_clip_813.py tests/test_pith_stage2.py tests/test_pith_stage3.py tests/test_pith_stage4.py tests/test_pith_provider_context.py tests/test_cc_recall_unification.py tests/test_cc_recall_dedup.py tests/test_cc_region_confidence.py` → **276 passed, 3 skipped, 2 xfailed, 0 failed, 0 errors.** Skips: the 3 pre-existing `CC_DAEMON_UNDER_TEST` cross-repo-parity skips in `test_pith_provider_context`. XFAILs: the two shared-budget SPECS (the deferred shared budget #883). Per file: `test_cc_pith_clip_813` 97, `test_surfacing_whole` 47, `test_cc_recall_unification` 25 (unchanged, daemon script supplied), `test_cc_region_confidence` 24, `test_pith_provider_context` 22, `test_pith_stage4` 20 (all 20: the 4 that failed last time passed with the complete model seeding), `test_cc_pith_off_budget_812` 19 (7 existing + 12 new), `test_pith_stage2` 9, `test_pith_stage3` 9, `test_cc_recall_dedup` 4.

**The ONE run against the pre-fold tip** (the new test file against a `git archive` of `55d56b9a`, `data/` `docs/` excluded, outside the repo; preamble `OK` for its root): **7 failed, 59 passed, 2 xfailed in 2.47s.** The 7, and why each fails there:

| Test | Fails on `55d56b9a` because |
|---|---|
| `test_n2_a_formatter_failure_is_reported_and_the_pattern_twin_survives[gate_off]` | no `on_monitor_error` call; X's pattern twin was deduped away and the monitor block is lost, so X renders nowhere |
| `…[pith_failed]` | same, via the Pith-failure entry to the renderer |
| `test_n2_runtimeerror_stays_silent_as_at_e4ebf982_but_the_twin_still_returns` | the twin of X is missing |
| `test_n2_the_formatter_failure_takes_the_same_route_as_a_harvest_failure` | the formatter case calls nothing |
| `test_n2_the_renderers_own_guard_is_the_last_resort` | its single formatter call is the first call, which succeeds, so the "second call fails" never happens there |
| `test_n2_the_host_hemisphere_error_stat_is_bumped_by_a_formatter_failure` | `_STATE.stats['errors']` not bumped; twin missing |
| `test_n2_the_daemon_hemisphere_error_stat_is_bumped_by_a_formatter_failure` | same, through the daemon's `_recall` |

**The other 5 new tests pass on BOTH tips, by design** (they pin or cover existing behaviour, so they cannot fail before): `test_n2_pith_on_success_path_never_formats_so_it_is_unchanged` (a pin of "unchanged"), `test_n3_pins_dedupe_before_budget…` (a pin of the current behaviour), `test_budget_binds_second_shape…`, `test_pith_off_an_over_budget_MONITOR_item_surfaces_as_trees_plus_reference`, `test_pith_off_an_over_budget_item_with_no_reference_form_is_dropped_loudly_never_cut` (coverage of existing #813 mechanism). I did not mutation-test them (an extra run was not authorised); that is stated, not claimed.

**Pre-flight disclosure:** before the targeted run I called the new tests directly from a throwaway script outside the repo (a stand-in for `caplog`, scratch `HOME`), which caught one wrong test shape (the never-fit pin: `cc_l1_budget` clamps a tiny budget up, so my first version could not force a never-fit) and one debug script confirmed it; no pytest run was spent on it. Both throwaway `HOME`s and the pre-fold archive copy were removed afterwards.

---

## Not verified / out of scope

- No real graph, daemon, checkpoint or `data/` path; no Syl data; the formatter-failure paths are exercised with fake graphs and a monitor whose `format_context` is made to raise (it is practically unreachable in production: a join over dicts).
- The Pith-ON success path with a REAL formatter failure is not a thing (it never formats); the Pith-ON-then-failed fallback IS covered (`[pith_failed]`).
- **Carried, not touched:** N-1 (#829), the 6 stale tests (#881), the TID-class 2 (#882; note the 4 embedder-dependent stage4 tests passed this time with the complete model seeding, which is evidence about their environment dependency, not a fix), dead constants N-8 (#883), the deferred shared budget #883 (`NG_SURFACE_BUDGET_CHARS`; its two SPEC xfails still XFAIL).
- N-3 (the dedupe-before-budget twin loss) is pinned, not fixed.
- Vault docs/wikilinks and the punchlist were not updated by me (outside this brief's commits).
