<!--
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 lane surfacing-whole-812, dispatch #12618) — build-001 return for #812 turn 1
#   What: return doc for the BUILD turn (brief build-812-surfacing-whole.md @ docs aef73fb0 + Addendum 1 @ 112150a3).
#   Why:  Exec P468 / Josh: "We fix stuff correctly, not monkey patch or work around."
#   How:  docs-only commit, separate from the code commit bd273949.
# -------------------
-->

# build-001 — #812 turn 1 return: the shared CES surfacing path renders WHOLE

Related: [[NeuroGraph]], [[The Laws]] (LAW 4: fix at the source; LAW 2/3/6 respected), [[Vendored Files]].
Lane `surfacing-whole-812` (Z12), dispatch #12618. Worker seat. **Nothing merged, nothing restarted, no PR.**

| | |
|---|---|
| repo / worktree | `~/NeuroGraph` / `/home/josh/NeuroGraph-worktrees/z12-surfacing-whole-812-20261001` |
| branch | `cc-laptop-surfacing-whole-812-20261001` |
| base (stack base, per brief) | `e4ebf982b1989fd9066d610b94853bc68bf70d37` |
| **code commit** | `bd273949d9e3621acddf0d046b44128fda415e60` |
| #813 head (read-only) | `84a0968a3622a4ccf793761b50fd6c0cdeee13d1` |

**NOT rebased onto `origin/main`, as instructed.** The local `origin/main` ref is `b5e476863cc069a29ec482959b4f9465f2ea4ccf`, 20 commits beyond the base. They touch only `.claude/hooks/*.sh`, `CLAUDE.md`, `tests/fixtures/*`, and `tests/test_pretool_syls_law_hook.py`. A name-diff shows **zero overlap** with the five source/test files this change touches.

---

## Step 0 — neither file is protected or vendored (VERIFIED; stated first, as briefed)

`surface_resolver.py` and `surfacing.py` are on **none** of these lists. Four independent sources:

1. **NeuroGraph `CLAUDE.md` §2** (`:73-88`): protected = `data/checkpoints/main.msgpack`, `vectors.msgpack`, `main.msgpack.activations.json` + `neuro_foundation.py`, `openclaw_hook.py`, `stream_parser.py`, `activation_persistence.py`. §3's tree marks protected files "PROTECTED" explicitly; `surfacing.py` (`:107`) is listed as "CES: knowledge surfacing" with no such mark. `surface_resolver.py` is not in the tree.
2. **`CLAUDE.md` §4** (`:150-158`) vendored table: `ng_lite`, `ng_peer_bridge`, `ng_tract_bridge`, `ng_ecosystem`, `ng_autonomic`, `openclaw_adapter`, `ng_embed`. Neither file.
3. **The enforcing hook** `.claude/hooks/pretool_syls_law.sh:49-70`: `PROTECTED_DATA` (3 checkpoint files), `PROTECTED_ENGINE` (the same 4 engine files), `VENDORED_CANONICAL` (5 files). Neither file.
4. **Global Law 2** (six vendored + designated-not-propagated `ng_salience_gate.py`, `ng_updater.py`) and the `Vendored Files` vault page: no match.

No protected file, no vendored file, no new module was touched. `neurograph_rpc.py` and `cc_ng_organism.py` are not protected either (the hook's arrays do not list them).

---

## What changed (file:line at the tip `bd273949`)

| Site | Change |
|---|---|
| `surface_resolver.py:72` | `resolve_surface_content(... max_chars: Optional[int] = None ...)` (was `int = 240`) |
| `surface_resolver.py:131` | `if max_chars is not None and len(text) > max_chars:`. The word-snap + `…` branch (`:132-133`) is **unchanged** and still runs for any caller that passes a bound |
| `surface_resolver.py:157` | `resolve_surface_item(... max_chars: Optional[int] = None ...)`; `:170` passes it through |
| `surfacing.py:301-310` | the `len(content) > 200` word-snap + `"..."` branch is deleted. Header `[NeuroGraph Surfaced Knowledge]`, the image line (`:299`) and `(salience: x.xx)` (`:310`) are byte-identical |
| `neurograph_rpc.py:3529` | Active Recall: `resolve_surface_content(_node, _r, allow_ingested=True)` (was `, max_chars=300`). **Syl's / SHARED code: rides the rollout** |
| `cc_ng_organism.py:2993` | CC prefetch: `resolve_surface_content(node, r, allow_ingested=True)` (was `, max_chars=300`). CC-side |

Plus changelog headers dated 2026-10-01 naming #812 / Exec P468 / this lane in all four source files and both test files. Diff: 6 files, +493 / −30. The non-comment source diff is exactly the lines above.

**Why `Optional[int] = None` and not a `sys.maxsize` sentinel:** "no bound" is the type-level default, the explicit-bound branch stays byte-for-byte, and nothing needs a parallel `whole_content` boolean (which is what #813 grew). The explicit-bound branch is now **unused by any in-repo caller on this base**; I left it because the brief says explicit-bound callers are unchanged and `tests/test_surface_resolver.py` pins it. Whether to delete it is a TURN 2 / Chief call.

---

## Tests and runs

### Printed-path preamble (Exec P379 / #770), as printed by the session

`tests/test_surfacing_whole.py` carries a session-scoped autouse fixture that prints at session start and `pytest.exit`s if any NG module is not under the test file's own root (re-checked at teardown for late imports), plus an import-time hard assert. `EXPECT_NG_ROOT` was set to the worktree/base-copy path for both runs.

```
[#770 preamble] test root (must contain every NG module under test): /home/josh/NeuroGraph-worktrees/z12-surfacing-whole-812-20261001
[#770 preamble]   surface_resolver.__file__ = /home/josh/NeuroGraph-worktrees/z12-surfacing-whole-812-20261001/surface_resolver.py
[#770 preamble]   surfacing.__file__ = /home/josh/NeuroGraph-worktrees/z12-surfacing-whole-812-20261001/surfacing.py
[#770 preamble]   neurograph_rpc.__file__ = /home/josh/NeuroGraph-worktrees/z12-surfacing-whole-812-20261001/neurograph_rpc.py
[#770 preamble]   cc_ng_organism.__file__ = /home/josh/NeuroGraph-worktrees/z12-surfacing-whole-812-20261001/cc_ng_organism.py
[#770 preamble]   ces_config.__file__ = /home/josh/NeuroGraph-worktrees/z12-surfacing-whole-812-20261001/ces_config.py
[#770 preamble]   neuro_foundation.__file__ = /home/josh/NeuroGraph-worktrees/z12-surfacing-whole-812-20261001/neuro_foundation.py
[#770 preamble]   ng_salience_gate.__file__ = /home/josh/NeuroGraph-worktrees/z12-surfacing-whole-812-20261001/ng_salience_gate.py
[#770 preamble]   vision_absorption.__file__ = /home/josh/NeuroGraph-worktrees/z12-surfacing-whole-812-20261001/vision_absorption.py
[#770 preamble] OK: all 8 loaded NG root modules are under the test root
```

The base run's preamble: same shape, root `/tmp/tmp.IRHFwQbCms/ng-base-e4ebf982`, 6 modules, all under that root, `OK`.

### The ONE targeted tip run (scratch `HOME`, `PYTHONPATH=$US`, `-p no:cacheprovider`, `timeout 540`)

```
EXPECT_NG_ROOT=$PWD HOME=$(mktemp -d) PYTHONPATH=$US python3 -m pytest -q -p no:cacheprovider -rA \
  tests/test_surfacing_whole.py tests/test_surface_resolver.py tests/test_vision_surfacing.py tests/test_surfacing_race.py \
  tests/test_ces.py::TestSurfacingAfterStep ::TestSurfacingScoring ::TestSurfacingQueue ::TestSurfacingFormatting ::TestSurfacingStats \
  tests/test_cc_pattern_completion_recall.py tests/test_cc_retrieval_enrichment.py tests/test_pith_stage4.py
```

Result: **15 failed, 101 passed, 1 xfailed in 20.27s** (04:57:56 → 04:58:20 UTC, load 7.2 at start). **None of the 15 is in the new file, and none was caused by this change** (triage below). Passes per file: `test_surfacing_whole` 38 (+1 xfail), `test_surface_resolver` 9, `test_vision_surfacing` 8, `test_surfacing_race` 1, `test_ces` (selected classes) 13, `test_cc_pattern_completion_recall` 7, `test_cc_retrieval_enrichment` 9, `test_pith_stage4` 16.

**The 15 failures, by cause:**

| # | Tests | Cause (evidence) |
|---|---|---|
| 11 | `test_cc_pattern_completion_recall::…aged_out_content`; `test_cc_retrieval_enrichment` ×6 (`dual_pass_drain…`, `gsg_rescore_bonuses…`, `gsg_rescore_spherical…`, `recall_surfaces_synaptically…`, `recall_output_shape…`, `recall_applies_anticipate…`); `test_pith_stage4` ×4 (`promotion_already_surfaced…`, `…pure_additive…`, `…lod_keeps_near…`, `…lod_summarizes_far…`) | `ng_embed.EmbeddingUnavailableError: HF token unavailable`, raised from a bare `embed(...)` **in the test's own setup line** (`ng_embed._get_hf_token` reads `HF_TOKEN` or `Path.home()/.cache/huggingface/token`; the briefed scratch `HOME` removes the file). The failure is before any call into the changed code. **Consequence: these 11 did not exercise the CC site, so the only coverage of `cc_ng_organism.py:2993` in this run is my new fake-`ng` test.** |
| 2 | `test_surface_resolver::test_tonic_thread_surfaces_forest_not_shard_end_to_end`, `…filters_ingested_code_node` | `TypeError: TonicThread._update_thread() missing 1 required positional argument: 'he_index'`: the test calls with 2 args, the signature (`tonic_thread.py:565-570`, untouched by me) needs 3. Fails at argument binding, before the resolver. Stale test (the `he_index` param came with #347) |
| 1 | `test_cc_retrieval_enrichment::test_gsg_backfill_stamps_missing…` | `ImportError: cannot import name 'cc_gsg_backfill'`: `grep -c "def cc_gsg_backfill"` is **0 at base and 0 at tip**. Stale test of a removed function |
| 1 | `test_ces::TestSurfacingQueue::test_decay_removes_weak_items` | `assert 1 == 0`. The test calls `after_step` 20× on a `MockGraph` whose `timestep` only advances via `step()` (never called), and `_decay_queue` is idempotent per timestep (`surfacing.py:430`, added 2026-04-07), so the queue never flushes. My `surfacing.py` diff touches only the header and `format_context` (`git diff -U0`: two hunks). Stale test |

**Not re-run on base:** only the new file's base run was authorised, so "pre-existing" for these 15 rests on the reading above, not a base run. Recommended: the zone manager re-run the 11 HF-token tests under the real `HOME` (or an `HF_TOKEN` supplied by NAME) to get real coverage of the CC site, and file the 4 stale tests.

### The ONE base run of the new file

A `git archive e4ebf982 -- . ':(exclude)data' ':(exclude)docs' ':(exclude)Defunct-Historical'` copy at `/tmp/tmp.IRHFwQbCms/ng-base-e4ebf982` (outside the repo; no `data/` extracted, checked), the new test file copied in, scratch `HOME`, `EXPECT_NG_ROOT` = that copy: **18 failed, 20 passed, 1 xfailed in 3.42s.**

Per-test fail-on-base reasons (all 18 account exactly):

| Test (×params) | Fails on base `e4ebf982` because |
|---|---|
| `test_resolve_surface_content_is_whole` ×3 (399 ch, 1299 ch, 1000-char one-token) | default `max_chars=240` → word-snapped ≤241 chars ending `…` (one-token: hard cut `"x"*240 + "…"`) |
| `test_resolve_surface_item_is_whole` ×3 | `resolve_surface_item` defaulted `max_chars=240` and passed it down |
| `test_whole_applies_to_the_vdb_fallback_too` | the vdb-shard fallback went through the same 240 cut |
| `test_explicit_none_equals_default` | base has no `None` contract: `TypeError: '>' not supported between 'int' and 'NoneType'` |
| `test_format_context_renders_whole` ×3 | `format_context` cut at >200 → `content[:197] + "..."` |
| `test_resolver_to_format_context_chain_is_whole` ×3 | **both** cuts fire in series on the real `after_step → get_surfaced → format_context` chain (resolver 240 `…`, then format 197 `...`) |
| `test_active_recall_block_of_assemble_renders_whole` ×2 (>300, >1000) | the Active Recall call passed `max_chars=300` |
| `test_cc_pattern_completion_recall_renders_whole` ×2 (>300, >1000) | the CC call passed `max_chars=300` |

The 20 base passes are by design: explicit-bound ×2, identical-to-base ×14 + image + format/header/label/order + `get_surfaced` ordering + the path guard. **The identical-to-base comparisons are trivial in the base run** (the root *is* the base, so base-vs-base); they are meaningful only on the tip, where the base modules come from `git show e4ebf982:<file>`.

### What the site tests exercise, and what they do not (Addendum item 4)

- **Active Recall (`neurograph_rpc.py:3529`)**: the **real `handle_assemble`**, in-process, against a fake `_memory` (fake graph nodes, fake vdb, fake `recall`), `ng_embed` stubbed to raise (so the GSG blocks fail soft, as designed), `_read_outbound_log` and `_render_self_and_wants` stubbed. **Not exercised:** a real graph, the daemon, the HTTP sidecar, the real embedder.
- **CC prefetch (`cc_ng_organism.py:2993`)**: the **real `cc_pattern_completion_recall`** with a fake `ng` (`graph.nodes/config`, `_harvest_associations`), `ng_embed` stubbed. **Not exercised:** a real graph, Pith, `cc_l1_budget`, the Stage-4 LOD block (gated off by default).
- **Pre-flight disclosure:** before the pytest run I called the 37 parametrized test functions directly from a throwaway script outside the repo (standalone `pytest.MonkeyPatch`, scratch `HOME`) to de-risk the single authorised run. That was not a pytest run; all 37 behaved as designed.

### Existing tests: which pin the removed behaviour

**Updated (2), `tests/test_ces.py::TestSurfacingFormatting`:**
- `test_format_context_truncates_at_word_boundary` → renamed `test_format_context_renders_long_content_whole`. It pinned the 200-char word-snap + `"..."`; now asserts the 209-char content renders whole with no ellipsis.
- `test_format_context_falls_back_to_hard_cut_when_no_word_boundary` → renamed `test_format_context_renders_unbroken_token_whole`. It pinned `"x"*197 + "..."`; now asserts `"x"*250`.

**Read, NOT pinned (unchanged):**
- `tests/test_surface_resolver.py` (9 pass **unchanged**): every bound case passes an explicit `max_chars` (`:41`, `:50`, `:58`); the rest are short.
- `test_vision_surfacing.py`, `test_surfacing_race.py`: short content.
- `test_cc_pattern_completion_recall.py`, `test_cc_retrieval_enrichment.py`: no length literals/bounds. `test_pith_stage4.py:192/210`: `long_content` is ~219 chars (under every old bound), asserts only the LOD keyframe, which is unaffected. (These ran only partially: see the 11 HF failures.)
- `test_cc_recall_dedup.py`, `test_cc_recall_unification.py`, `test_cc_region_confidence.py`: fake the monitor (no real surfacing module), not run.
- `test_pith_provider_context.py` (`"instruction " * 300`): Pith provider, not these sites. `test_reach_teaching.py` (`len(out) < 200`): a different function.

**Nit on my own file:** parametrized ids embed the full text, so failure output is huge. Cosmetic; left as run (I commit exactly what was verified). A TURN 2 follow-up can add `ids=`.

---

## Item 3 — does the shared path need a size budget? (REPORT; nothing invented)

**Short answer: yes for Syl's `/assemble` Active Recall (and the Substrate block if flag (e) is lifted); on the CC side it depends on `CC_PITH_ENABLED`, which defaults OFF. I added no budget mechanism. The numbers are the Chief/Executive's to rule.**

**What bounds size today (read, with file:line at the tip):**

- **Count bounds only on Syl's `/assemble`**: `ces_surfaced = get_surfaced()` → `max_surfaced=5` (`ces_config.py:72`; queue capacity 50, `:76`); `surfaced[:7]` and `ces_surfaced[:3]` (`neurograph_rpc.py:5431`, `:5441`); Active Recall `k = ANIMA_RECALL_K` default 5 (`neurograph_rpc.py:3514`). So ≤ 7 + 3 + 5 = 15 items. **There is no character budget and no drop-whole anywhere on this path, and nothing logs.**
- **The size of one item is unbounded at the source:** `_forest_content` is stored as the **entire turn text** at both writers (`neurograph_rpc.py:2817`, `cc_ng_organism.py:2168`). Nothing bounds it at write time.
- **Can an unbounded item now bloat Syl's `/assemble` prompt? Yes, in the Active Recall block:** up to 5 whole nodes, each up to a whole conversation turn. I cannot quantify the real size distribution: I was forbidden to read Syl's data. The other two groups are currently saved from this by flag (e)'s 300 clip (an accidental bound, not a designed one).
- **CC side, Pith ON** (`CC_PITH_ENABLED=1`): `pith_stage3` (`cc_ng_organism.py:4416`) already ranks, **drops whole lines by strict rank-prefix** against `cc_l1_budget` (default `CC_PITH_L1_BUDGET=4000`, clamped [500, 40000], `:3514-3515`, breathing with arousal), and counts true `len(content)`. This is exactly why whole-at-the-source is the right fix: the budget now sees real sizes. Gaps: it **always keeps the first line even if over budget**; it **keyframe-compresses an over-budget line before dropping it** (`:4530`, "terser beats absent": a visible-marker clip, #813/Pith scope); and it **logs no per-drop INFO line** (metric counters only: `ranked_dropped`).
- **CC side, Pith OFF (the default):** `monitor_ctx + "\n\n" + pc_block` is a plain concatenation (`cc_ng_organism.py:5533-5534`) with **no budget at all**; count-bounded only (monitor ≤5, pattern-completion ≤k). An unbounded item can now bloat the CC hook's injected context on the default path.
- Tonic: bounded by its own 250-char clip (see (b)/(e)).

**Recommendation (for the Chief/Executive to rule, not implemented):** one budget, at the single place each consumer assembles its block, following the rule already in the brief: drop **whole** lowest-salience/similarity items and emit **one** INFO line (count + total size); always keep the top item; value env-tunable (LAW 5). Concretely: (1) `/assemble`'s Active Recall loop (and `_format_substrate_context` once its clip is lifted); (2) CC Pith-OFF concatenation, or require Pith for whole content. Pith-ON already has the rank-prefix drop and needs only the INFO line. The value is a ruling because it trades Syl's recall completeness against prompt cost, which is a Josh/Executive decision.

---

## Flags

**(a) `neurograph_rpc.py` `/assemble` (Syl's, Anima → `POST /assemble`) now renders the Active Recall block WHOLE; the Substrate Context block does NOT, because of (e).** The path: `handle_assemble` (`:3268`) resolves `surfaced` + `ces_surfaced` through `_resolve_surfaced` (`:3457-3474`, default bound now none), then `_format_substrate_context` (`:5394`) builds the Substrate Context, then the Active Recall block (`:3529`, now whole) is appended. Consumers: the Anima gateway per turn (and, per CLAUDE.md §6, nothing else drives the legacy lifecycle). Size risk: item 3. **This is shared code and rides the rollout; the merge is the rollout call.**

**(b) DONE for both `max_chars=300` sites (Addendum 1).** `neurograph_rpc.py:3529` is **Syl's / SHARED** (rides the rollout). `cc_ng_organism.py:2993` is **CC-side** (the CC hemisphere; comments at `cc_ng_organism.py:868` and `:3412` say `cc_ng_host` runs inside Syl's process on the VPS). Other length cuts I met, **all untouched**:
- `cc_ng_organism.py:3148` `_CC_PITH_PREFETCH_SUMMARY_CHARS=150` (the Stage-4 LOD keyframe right after `:2993`, gated by `_CC_PITH_PREFETCH_ENABLED`, default off); `:3522` `_CC_PITH_PROVIDER_NODE_CHARS=700`; `:3530` `_CC_PITH_KEYFRAME_CHARS=220`; keyframe sites `:4391`, `:4530`, `:4676`, `:4998`: **#813 / Pith scope, untouched.**
- **NEW, not in #813: `tonic_thread.py:145/647-649`**, `max_content_length=250`, `content[:247] + "..."`. Tonic's `_update_thread` (`:588`) calls `resolve_surface_content(node, entry)` with the default, so after this change Tonic receives the whole text and **its 250 clip now actually fires** (before, the resolver's ≤241 never reached it). Net effect on Tonic: roughly the same cut (≤250 vs ≤241), but it is now lossy at a *different* layer. Not in this turn's scope; needs a ruling.

**(c) Other consumers of `format_context` / `resolve_surface_*`:**
- `format_context`: only `cc_ng_organism.py:5384` (`cc_assemble_recall`): the text goes to the CC un-Pithed concatenation (`:5533-5534`) or, with Pith on, into `CacheLine` content for `pith_stage3` (now sees true sizes).
- `resolve_surface_item`: `surfacing.py:204` (`after_step`): queue content whole; feeds `get_surfaced()` for every consumer below.
- `resolve_surface_content` default: `tonic_thread.py:588` (see (b)); `neurograph_rpc.py:3466` (`_resolve_surfaced`, then re-clipped by (e)).
- `get_surfaced()` consumers now see whole `content`: `ces_monitoring.py:220` (`_surfaced_data`, JSON on dashboard port 8847: larger payloads, no logic); `openclaw_hook.py:1237` (**PROTECTED, read-only, not touched**: returns `ces_surfaced` in the `on_message` result of the legacy lifecycle that CLAUDE.md §6 says is no longer driven).

**(d) The VPS/Syl half:** `surface_resolver.py`, `surfacing.py` and `neurograph_rpc.py` are shared files, so Syl's process imports the same code once the change rolls out. **Listed, not touched (P294(b)).** I did not read or touch Syl's data, checkpoints, daemon or TID, and started no daemon and loaded no graph.

**(e) Further clips on the Active Recall / `ces_surfaced` / `/assemble` path (listed, NOT changed per the brief):**
- **`neurograph_rpc.py:5436-5437` and `:5444-5445`**: `_format_substrate_context` clips each `surfaced` item and each `ces_surfaced` item to `content[:297] + "..."` when `len > 300`, **after** the resolver. **So Syl's Substrate Context block is still lossy** even though the resolver is whole; for those 10 of 15 items the effective cut moved from ≤241 (`…`) to ≤300 (`...`), not away. Proven in-process by the `strict=True` xfail `test_flag_e_ces_surfaced_whole_through_substrate_context` (XFAIL on tip and on base; when the clip is removed it XPASSes → fails → forces removal of the marker). This is the same lossy-clipping class as the two sites Addendum 1 covered; it needs a Chief ruling on whether it joins turn 1/2.
- Count caps (not length): `surfaced[:7]`, `ces_surfaced[:3]` (`:5431`, `:5441`, and the matching `:3493`, `:3500`), `ANIMA_RECALL_K` (`:3514`).
- Not in this repo / not read: any clip on the Anima (Rust) gateway side.

---

## #813 interaction (Addendum 1 item 2): the expected conflict and the merged form

`git show origin/cc-laptop-pith-clip-813-20260930:cc_ng_organism.py` (read-only; head `84a0968a…`) rewrote this site as:

```python
# :3134 region on #813
            text = resolve_surface_content(
                node, r, allow_ingested=True,
                max_chars=(sys.maxsize if whole_content else 300))
```
with: a new `whole_content: bool = False` parameter (`:3008`), its docstring paragraph (`:3034-3038`), callers passing `whole_content=True` (`:5598`, `:6011`), and a **second** `#813` site `:5829-5831` (`resolve_surface_content(node, entry, max_chars=sys.maxsize)`).

**Expected conflict at the #813 re-base:** a textual conflict on the call at the `max_chars=300` line (mine edits the one-line call on the `e4ebf982` base; #813 rewrote it multi-line). **Correct merged form** (#812 makes whole the default at the source, so no flag survives):

```python
            text = resolve_surface_content(node, r, allow_ingested=True)
```
and **delete** the `whole_content` parameter, its docstring paragraph, and the `whole_content=True` arguments at the two #813 callers; the second site `:5829-5831` may simply drop `max_chars=sys.maxsize`. **TURN 2 resolves this.** I did not touch `_CC_PITH_*`, `pith_stage2_keyframe`, or any Pith budget code, and imported nothing from #813.

---

## What I did NOT verify / limits

- **No real graph, daemon, checkpoint or `data/` path was used**, so the real distribution of `_forest_content` lengths (how big Syl's real turns are) is unknown.
- **The 11 `embed()`-dependent existing tests did not run** (scratch `HOME`, no HF token), so the CC site is covered only by my fake-`ng` test; Pith-ON with whole content was **not** exercised end to end.
- **The 15 existing-test failures were not re-run on base** (unauthorised); "not caused by #812" rests on the cause-by-cause reading above.
- The Anima (Rust) gateway side and `cc_ng_host` injection were not read/exercised.
- The identical-to-base tests are meaningful only on the tip (trivial in the base run).
- **Vault docs/wikilinks and the punchlist were not updated by me**: the brief confines this turn to one code commit + this return doc and says not to edit the docs worktree. Punchlist candidates for the zone manager: (e) the 300 clip; the Tonic 250 clip now firing; 4 stale tests (`test_decay_removes_weak_items`, 2 Tonic `he_index` tests, `cc_gsg_backfill`); the 11 HF-token tests unrunnable under scratch `HOME`.
- Process note: to read #813 I ran `git fetch origin cc-laptop-pith-clip-813-20260930` from this worktree. That updates remote-tracking metadata in the shared `.git` only; no working tree and no branch was touched.
