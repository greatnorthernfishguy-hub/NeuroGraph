# checker-019 ROLE A (cross-family) — #813 Pith clip removal + character-cap audit

STATUS: COMPLETE

- Seat: checker-019 (cross-family, grok-4.6, `report_only`). ROLE A only; ROLE B not written.
- Lane: `pith-clip-removal-813`. Dispatch #10885. Zone manager Z12 session `52d39aba-db92-4bf2-b3b1-0e4c13f77d8c`.
- Packet: `/home/josh/docs/.claude/worktrees/daemon-recall-756-20260930/handoffs/z12-silent-failure-lanes-20260930/assignments/review-packet-813-pith.md` (docs branch `cc-laptop-daemon-recall-756-20260930`, not edited).
- NG worktree `/home/josh/NeuroGraph-worktrees/z12-pith-clip-813-20260930`, branch `cc-laptop-pith-clip-813-20260930`. `git pull --rebase origin cc-laptop-pith-clip-813-20260930` was already up to date at worker head `9d63846b2e4ef732825165be8f46800af0fe44b8`. This review's stub is `63bc18fd278334b6f29d512525cfd8452bc33339`.
- Worker commits (full hashes from `git rev-parse`): plan+audit `d4bc6158398c4faf4897a39e7fc81d53ea5cb5ac`, build `5a8120a5ce4329a65549ed4d9449f57996832e7c`, return `9d63846b2e4ef732825165be8f46800af0fe44b8`. Base `e4ebf982b1989fd9066d610b94853bc68bf70d37`.
- docs worktree `/home/josh/docs/.claude/worktrees/z12-pith-clip-813-20260930` at `44199c00724c30d4cc78d1620ca0a6c2815b6f79` (preflight `8cba796388550720362c4d9ab850dd63cd2baac5`). Read only; not edited.
- Condensate `master` `408654003778b0f090ac3c2975a78bb15fb3277a` (read-only `git show master:rust_core/src/minitid.rs`). Quest branch `origin/cc-laptop-minitid-card7-quest-removal-20260929` `88ddfecaa3f410f5eb2e96eb386d080e42f0e302` (read-only).
- Authority: report_only. No merge, settle, or dispatch. No graph, msgpack, checkpoint, or tract load. Primary `/home/josh/NeuroGraph` not edited. Real `~/.bashrc` never written. Secrets by NAME only.

## P379 session start

Printed before any NG import, with `PYTHONPATH` and `NG_EMBED_*` unset:

```
python: /usr/bin/python3
sys.path:

  /usr/lib/python312.zip
  /usr/lib/python3.12
  /usr/lib/python3.12/lib-dynload
  /home/josh/.local/lib/python3.12/site-packages
  /usr/local/lib/python3.12/dist-packages
  /usr/lib/python3/dist-packages
NG-related sys.modules: NONE
PYTHONPATH: None
NG_EMBED_*: {}
cwd: /home/josh/NeuroGraph-worktrees/z12-pith-clip-813-20260930
```

The parent shell had `PYTHONPATH=/home/josh/NeuroGraph:` and `NG_EMBED_REMOTE=hf`. Every command that imported or tested NG used `env -u PYTHONPATH -u NG_EMBED_REMOTE` (and `-u NG_EMBED_MODEL -u NG_EMBED_ENDPOINT`). `tests/test_cc_pith_clip_813.py::test_p379_module_under_test_is_the_worktree_copy` passed, so the suite imported this worktree's `cc_ng_organism.py`.

Real `~/.bashrc` sha256 before any work, after tests, after the staged-script rehearsal, and at this write: `72f2e7133cce652ac0a8930ed2bef3eca20c63331d49e65f9ba55470604255a3`.

Stub first-write: commit `63bc18fd278334b6f29d512525cfd8452bc33339` (`STATUS: INCOMPLETE - review in progress`), pushed to `origin cc-laptop-pith-clip-813-20260930`.

---

## Overall verdict

**PASS-WITH-NOTES**

The branch implements the lane's change list: `_pith_node_text` returns whole node text, budget-time shortening is gone, Stage 3 no longer keyframes an over-budget line, the source-label `[:80]` slice is gone, prefetch LOD keyframes are gone, the host contract lists five exports and no longer promises to shorten member prose, and the docs preflight drops `CC_PITH_PROVIDER_NODE_CHARS` from the required set while still tolerating a stale export. Targeted tests were **223 passed, 3 skipped** in 23.76s (the three skips are the parametrized host/daemon parity test, which needs `CC_DAEMON_UNDER_TEST`). Docs preflight tests were **24 passed** in 0.45s. Three reviewer-owned short-node cases vs `git show e4ebf982:cc_ng_organism.py` were byte-identical.

The remaining cuts the worker parked as D1–D7 are real and correctly not decided here. One HIGH note must be folded into D1 before anyone treats "Pith-on is whole" as true of every budgeted Pith path: `cc_assemble_recall` still calls `cc_pattern_completion_recall` with the default `whole_content=False` (300-char snippet + `"…"`) and then feeds those already-cut strings into `pith_stage3`. That is a cut on the Pith-on L1 path, not only the gate-off `## Active Recall` path D1 names.

---

## A1 Whole rendering and the budget rule

**Verdict: PASS-WITH-NOTES**

### Nodes render whole on the provider path

`_pith_node_text` (`cc_ng_organism.py:4695-4707`) returns `_pith_node_raw_text` unchanged. `_CC_PITH_PROVIDER_NODE_CHARS` is absent (`hasattr` is false on the branch module; base still resolves it to 700). `_pith_fit_statement` is deleted (AST: not in `FunctionDef` names). `_pith_fit_connected_line` (`:5010-5022`) copies the line only when `len(_pith_render_connected_line(line)) <= max_chars`; otherwise `None`. `_pith_node_sources` (`:4710-4717`) appends the stripped label with no `[:80]`.

`pith_provider_context` (`:5178-5180`) passes `whole_content=True` into `cc_pattern_completion_recall`, so the provider fallback is not the 300-char snippet.

### Drop-whole rule and the empty-envelope refinement

`_pith_provider_admit` (`:5025-5070`) sorts `(-score, node_id)` and walks that order. A line that does not fit the *remaining* envelope (including the `"\n\n"` separator) is dropped whole. If that line's whole render is `<= budget_chars` (it would have fit an empty envelope) admission **stops**, so a lower-ranked line cannot jump it. If the whole render is `> budget_chars` (it could never fit) the line is **skipped** and admission continues.

That is a real deviation from "strict prefix of every candidate." The ranking guarantee becomes: among assemblies that can fit an empty envelope, keep a strict remaining-envelope prefix; unfittable giants are treated as absent. A lower-ranked small assembly **can** jump a dropped giant. It cannot jump a dropped assembly that would have fit empty.

The refinement is the right product choice for "one giant top-ranked assembly must not blank every other." It is tested:

- `test_admit_is_strict_rank_prefix_and_a_dropped_line_is_never_shortened` — `c` would fit the remainder after `a`; `b` fits empty but not remainder; kept is only `a`.
- `test_admit_skips_an_assembly_that_can_never_fit_and_keeps_the_rest` — giant skipped, `s1` then `s2` kept.
- `test_budget_binding_drops_lowest_ranked_whole_assemblies_with_one_info_line` — lowest two of four dropped whole, one INFO line with count and rendered chars.

### Stage 3 uses a different rule

`pith_stage3` (`:4553-4561`) always keeps the first unpinned line even when `len(content) > budget_chars`, then `break`s at the next overflow. It does **not** skip later giants. `test_stage3_never_returns_an_empty_l1_and_logs_nothing_when_nothing_dropped` keeps a 900-char line on a 500-char budget. That is the pre-existing never-empty-L1 guard, kept on purpose. It is a second budgeted discipline: provider_admit would skip that giant; Stage 3 emits it whole and over budget. See C3.

### INFO line

Provider (`:5065-5069`) and Stage 3 (`:4563-4570`) log one INFO line iff `dropped_count` / `dropped > 0`. `test_no_drop_no_info_line` and the Stage 3 empty-drop case hold. The flood ("fires on every recall where the budget binds") is the stated visibility contract. A rate limit is an operator choice; silence is not.

### Consequence 4 (a whole render that cannot fit)

True on the **provider** path, with a tighter bound than 40,000. `learned_budget = max(0, budget - len(envelope_shell))` (`:5202-5205`). An assembly whose whole render exceeds *that* envelope can never be shown, even if it is under 40,000. If it is the only candidate, state is `empty` + `capacity_empty` (closed, loud) plus the INFO line. Constitutional core exceeding the total budget is `unavailable` (`constitutional_core_exceeds_budget`), not a cut. That is an acceptable price of whole-or-absent, and it is loud.

Stage 3 can still *show* a first line larger than its L1 budget (C3). MiniTID `MAX_PROVIDER_CONTEXT_CHARS` 40,000 is a separate REJECT-LOUDLY ceiling (A3).

### Remaining cuts this item must name

Provider-path node text is whole. These paths still cut or discard content (parked, not fixed):

1. `cc_assemble_recall` `:5392` calls `cc_pattern_completion_recall` with default `whole_content=False` → `resolve_surface_content(..., max_chars=300)` then `"…"`. CacheLines built at `:5439-5449` carry that already-cut `content` into `pith_stage3`. This is Pith-**on** L1, not only gate-off. **C1 / widened D1.**
2. Gate-off `## Active Recall` uses the same 300-char default (D1 as written).
3. `pith_compress_history` `:4422` keeps `keyframe` and discards `_delta` (A2 / D2).
4. D3–D6 (wants, deposit clips, member/overlap drops, `surfacing.py` 200-char `"..."`).

---

## A2 Keyframe carries its delta or does not apply

**Verdict: PASS-WITH-NOTES**

`pith_stage2_keyframe` (`:4134-4173`) documents that `keyframe` + `delta` is lossless and `keyframe` alone is a cut. Production call that still discards the delta: `pith_compress_history` `:4422` (`keyframe, _delta = pith_stage2_keyframe(...)`).

`test_no_keyframe_or_word_cut_call_in_any_budgeted_path` AST-walks `_BUDGETED = (pith_stage3, _pith_node_text, _pith_fit_connected_line, _pith_provider_admit, pith_provider_context, cc_pattern_completion_recall, pith_connected_activation_basins, _pith_node_sources)` and asserts none `Name`-call `pith_stage2_keyframe`, `_pith_cut_at_word_boundary`, or `_pith_fit_statement`. `_pith_fit_statement` is gone. `_pith_cut_at_word_boundary` remains only inside the keyframe function (`:4228`).

Limits of that test (C4): it does not include `cc_assemble_recall` or `pith_compress_history`; it ignores `Attribute` calls. Independent `git grep` of `*.py` at HEAD shows the only non-test, non-docstring call of `pith_stage2_keyframe` is `pith_compress_history`. `cc_assemble_recall` `:5487` calls `pith_stage3` only.

`cc_ng_host._handle_compress_history` (`:974`) is still registered at `_DISPATCH["compress_history"]`. Condensate `master` mentions `compress_history` only in the file header comment (`minitid.rs:12,15`). That is D2: a live socket handler with no live miniTID caller, still a lossy keyframe.

Prefetch LOD keyframe staging is removed (`:3000-3003` comment; `test_promotion_lod_far_content_stays_whole` asserts full `long_content` and no `"⋯[+"`). Dead tunables `CC_PITH_PREFETCH_SUMMARY_CHARS` / `_cc_node_query_distance` remain, as the plan said.

---

## A3 The audit (35 rows)

**Verdict: PASS-WITH-NOTES**

The 35-row table in `plan-001-audit.md` (sha256 `ae9a3f258816c6d34ce29fa9681d6c3e5f41914f3a17748ff1614c87bd88f817`) covers the Pith provider/L1 path, VPS host clips, CES surfacing, docs preflight/daemon, and miniTID. Classifications checked against live source:

| Claim | Source | Holds? |
|---|---|---|
| `MAX_PROVIDER_CONTEXT_CHARS` 40,000 at `:508`; check `:801` (`parse_provider_response` returns `None` if `context_chars > MAX`); check `:1095` (`provider_context_is_usable`) | `git show master:rust_core/src/minitid.rs` | Yes. `:761` then maps `None` to `Err("daemon response was not a fresh provider_context envelope")`. REJECT-LOUDLY. Wording defect (size vs schema) is real. |
| `PITH_NOTICE_WHY_MAX` `:1242`; cut `:1264-1265` `chars().take(200)` + `"..."` | same | Yes. **CUT** of the provider-facing notice. `deposit_text` `:1274` is uncut. Test `:3499-3514` asserts `ends_with("...]")` and `≤ MAX+60`. |
| Tool-tail `bounded_current_episode` 64 KiB, whole-message drop, newest pair kept exact | `:490-512`, `:1000-1065` | Yes. Aggregate `msgs_in`/`msgs_out` at `:1712-1731`. |
| `MAX_QUEST_CHARS` not dead on master | `extract_quest_focus` defined `:849`; called `:1366` and `:1671`; request field `:775` | Yes. Both hosts still forward `quest_focus`: `cc_ng_host.py:1027`, docs `cc-ng-daemon.py:1667`. Organism still concatenates a non-empty `quest_focus` into the recall cue (`:5161-5163`) and refuses `len(quest_focus) > MAX` with `invalid_quest_focus` (`:5147-5148`). REJECT-LOUDLY, live. |
| Card 7 "Quest removed" | `origin/cc-laptop-minitid-card7-quest-removal-20260929` `88ddfec` | Live `extract_quest_focus` / `quest_focus` symbols are gone from that blob (the only `quest_focus` hit is a changelog comment at `:30`). This is the unmerged surface the Exec "Quest removed" line most likely named. D7 should cite it. Removing `MAX_QUEST_CHARS` from NG/docs while master miniTID still sends `quest_focus` would be a host TypeError/guard hole. |

### Caps not in the 35-row table

On the Pith files and the paths they feed, the table already has the live character cuts. Sibling extraction surfaces **not** in the table (outside this lane's edit list, listed so the audit is not mistaken for a repo-wide cap census):

- `neurograph_rpc.py` `handle_assemble` `max_chars=300` and `content[:297]+"..."` (Syl `/assemble`; same family as D6).
- `kiss_filter.py` first-sentence / 60-char summary of old messages (Syl assemble window).
- `tonic_thread.py:649` `content[:max_len-3]+"..."` (Tonic thread display; no `CC_PITH` symbol in that file).

Inside the Pith files, dead `CC_PITH_PREFETCH_SUMMARY_CHARS` (`:3158`) is named in row 11. `_pith_cut_at_word_boundary` remaining as keyframe-internal is row 12. Debug `[:15]/[:48]/[:70]/[:120]` is row 16. I did not find another `[:N]` content cut on the provider/L1 functions that the table omitted.

Row 7's "this branch" text says the 300-char cap is unchanged for "the gate-off L1/`## Active Recall` path." That understates Pith-on `cc_assemble_recall` (C1). The rest of the table's this-branch column matches the code.

---

## A4 Contract + preflight in one change

**Verdict: PASS**

NG `docs/PITH_HOST_CONTRACT.md` (blob sha256 `390b49395753ea8f12ae0b98471b3dec999d4c346949cc95462f6161759bd904`) export block is exactly five names: `ROOTS`, `MEMBERS`, `DEPTH`, `MAX_INSTRUCTION_CHARS`, `MAX_QUEST_CHARS`. `CC_PITH_PROVIDER_NODE_CHARS` is described as gone; a stale export is "ignored and harmless." The "may shorten each member's prose" sentence is absent (`grep` empty). Whole-or-drop + INFO log is at `:183-192`, including the empty-envelope skip.

docs `scripts/cc-ng-service.py` `preflight` required tuple (`:117-122`) is those five Pith provider names plus the pre-existing launch keys. `canonical_exports` still copies any `CC_*` literal, so a stale `NODE_CHARS` export is tolerated. `test_retired_node_chars_export_is_not_required_but_is_tolerated` covers both absences and `=700` present.

Docs tests, from the docs worktree, `HOME` override after computing `site.getusersitepackages()`, `NG_EMBED_*` and `PYTHONPATH` unset, `NG-related sys.modules: NONE` before pytest, service path `.../z12-pith-clip-813-20260930/scripts/cc-ng-service.py`: **24 passed** in 0.45s (`test_cc_ng_service.py` + `test_pith_provider_context_wrapper.py`).

VPS `cc_ng_host.py:509-511` still lists `CC_PITH_PROVIDER_NODE_CHARS` in `PITH_SNAPSHOT_GATE_KEYS` (not edited). `pith_effective_config()` reports `resolved: None` and `authority: "retired (#813: nodes render whole)"` while `env` still shows a stale export if present (`:3839-3847`, `:3888-3897`). The value is telemetry-only. Laptop daemon allow-list `cc-ng-daemon.py:1990` is the same inert leftover. That effect statement in the return is correct.

---

## A5 The staged `.bashrc` script

**Verdict: PASS**

Script: `handoffs/z12-pith-clip-813/returns/bashrc-drop-node-chars.sh` (sha256 `cec7f6e440589645aaa2378df7dbf0f6d49fa65bd93150d6423eaa80f77d66f2`). Removal is by pattern `^export CC_PITH_PROVIDER_NODE_CHARS=`, never a line number. `apply` refuses unless `grep -c` is exactly 1. `verify` runs `bash -n`, prints `CC_PITH_PROVIDER_*` **names** only, requires the five keepers present and `NODE_CHARS` absent. `reverse` byte-exact restores when the post-apply sha still matches; otherwise re-appends the one stored line.

Rehearsal: `cp -p` of the real file to `/tmp/bashrc-813-checker019-copy`, `BASHRC` pointed at the copy. `apply` removed exactly `export CC_PITH_PROVIDER_NODE_CHARS=700` (diff vs the real file was that one line at the old `:291` neighbourhood). `verify` printed the five names, no values. `reverse` restored the copy to sha256 `72f2e7133cce652ac0a8930ed2bef3eca20c63331d49e65f9ba55470604255a3`. Zero-match apply exit 1, two-match apply exit 1, copies unchanged. Real `~/.bashrc` sha256 identical before and after. Temp copies deleted.

S4 ordering is right given P329 (NG first) and the **current** preflight: removing the export before the docs merge is live makes `preflight` raise `missing canonical export`. Leaving the export is always safe (new organism ignores it; new preflight tolerates it). The script belongs after both merges are deployed and before the restart that re-reads the environment.

---

## A6 Golden and tests

**Verdict: PASS**

Worker targeted files, once, `env -u NG_EMBED_REMOTE -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1`, cwd the NG worktree:

`pytest tests/test_cc_pith_clip_813.py tests/test_pith_provider_context.py tests/test_pith_stage{1,2,3,4,5}.py tests/test_pith_l1_provenance.py tests/test_pith_metrics_concurrency.py tests/test_pith_history_metrics.py tests/test_cc_host_pith_telemetry.py tests/test_cc_recall_dedup.py tests/test_cc_recall_unification.py tests/test_cc_region_confidence.py -p no:cacheprovider`

**223 passed, 3 skipped** in 23.76s. The three skips are `test_vps_host_and_laptop_daemon_have_identical_closed_contract[ok|empty|unavailable]` (`CC_DAEMON_UNDER_TEST` unset). Not re-run with the daemon path (packet: once).

`test_short_wellformed_nodes_render_exactly_as_base` compares four fixture scenarios against `tests/fixtures/pith_clip_813_golden_base.json` (`base_commit` starts `e4ebf982`). Generator `tests/fixtures/gen_pith_clip_813_golden.py` loads `git show e4ebf982:cc_ng_organism.py`.

Three reviewer-owned cases vs that same base blob (sha256 `9f9617ada8b5f68800e3e82f1ac67e015c3e049745d8f5b1eb4301887471f5df`), fake graph only, `whole_content` not involved (recall swapped):

| Case | equal | state | context chars | assemblies |
|---|---|---|---|---|
| `short_single` | True | ok/ok | 424/424 | 1/1 |
| `three_tiny_assemblies` | True | ok/ok | 679/679 | 3/3 |
| `source_and_anchor` | True | ok/ok | 461/461 | 1/1 |

Five updated old tests, each with a changelog reason; none weakened to hide a regression:

1. `test_total_context_bound_drops_oversized_connected_line_whole_without_tearing` — used to require shortening into a 700-char envelope; now expects `empty`/`capacity_empty` on 700 and whole render on 40000.
2. `test_overflow_item_is_dropped_whole_not_keyframed` — used to require a Stage 3 keyframe; now `ids == ["top"]`, `compressed_count == 0`.
3. `test_pinned_never_compressed` — pinned still verbatim; the non-pinned overflow is dropped, not keyframed.
4. `test_metrics_snapshot_consistent` — `compressed_count`/`chars_saved` now 0, plus `ranked_kept == 1`.
5. `test_promotion_lod_far_content_stays_whole` — far promoted node keeps full content.

`test_break_when_even_keyframe_overflows` was already a drop case (`compressed_count == 0`); it still holds under whole-or-drop.

---

## A7 D1–D7 and overall verdict

**Verdict: PASS-WITH-NOTES**

| Id | Worker left it undecided? | Correctly identified? | Checker |
|---|---|---|---|
| D1 gate-off `## Active Recall` 300-char snippets | Yes | Incomplete: the same default also feeds **Pith-on** `cc_assemble_recall` L1 (C1). The parked decision is real; the label is too narrow. | note |
| D2 `pith_compress_history` lossy keyframe / retire vs identity | Yes | Yes. Live host handler, no live miniTID caller on master. | agree |
| D3 `_WANT_RE` 600-span silent un-capture + `render_wants` `[:600]` | Yes | Yes. Outside provider_context. Parallel want-repair work exists; this lane must not silently "fix" it. | agree |
| D4 deposit-path clips (`text[:2000]`, host/daemon tool clips) | Yes | Yes. LAW 7 on the deposit path; host/daemon out of this file list. S4 caps precondition is the right routing. | agree |
| D5 silent member/overlap drops (`MEMBERS`/`DEPTH`/0.6 basin) | Yes | Yes. COUNT / DROP-SILENTLY of nodes, not a char cap. Provider INFO does not cover these. | agree |
| D6 `surfacing.py:293-295` 200-char `"..."` | Yes | Yes. Shared with Syl `/assemble`; P329. Used for `monitor_ctx` (gate-off / Pith-failure fallback). `get_surfaced()` itself is uncut; `format_context` cuts. | agree |
| D7 `MAX_QUEST_CHARS` / "Quest removed" | Yes | Yes for master (still live). Cite card 7 `88ddfec` as the unmerged removal (C5). | note |

**Add:** the Stage 3 vs provider_admit rule split (C3) and the learned_budget vs 40,000 wording (C2). Neither needs a worker decision inside this lane if D1 is widened and Consequence 4 is restated.

### Numbered corrections

1. **C1 (HIGH).** Widen D1: `cc_pattern_completion_recall(..., whole_content=False)` at `cc_assemble_recall:5392` still clips to 300 chars with `"…"` **when Pith is on**, then `pith_stage3` whole-or-drops those snippets. Provider_context is whole; L1 assemble is not. Do not advertise "Pith-on renders whole nodes" without this exception.
2. **C2 (LOW).** Consequence 4 is "cannot fit `learned_budget`" (total budget minus the section shell), not "cannot fit 40,000."
3. **C3 (LOW).** Stage 3 still emits the first unpinned line even when it exceeds the L1 budget; provider_admit skips a line that cannot fit an empty envelope. Two budgeted paths, two rules. Document or align later; not a merge blocker for the provider clip.
4. **C4 (LOW).** The AST "no cutter in budgeted path" test does not walk `cc_assemble_recall` and does not see the 300-char `resolve_surface_content` cut. It is true as written and insufficient as a "no cut remains" proof.
5. **C5 (NOTE).** D7 should name Condensate `origin/cc-laptop-minitid-card7-quest-removal-20260929` (`88ddfec`) as the Quest-removed blob. Master `4086540` still extracts and forwards `quest_focus`. Keep `MAX_QUEST_CHARS` until that branch and both hosts move together.

### Not verified

- Live daemon, miniTID PID, `:9090`, `daemon.sock`, tracts, checkpoints, or a running INFO log.
- The three host/daemon parity tests (skipped without `CC_DAEMON_UNDER_TEST`).
- Rust build / `cargo test` (Condensate read-only `git show`).
- VPS host process, `.bashrc` apply on the real file, S4 batch execution.
- Full NG or docs suites.
- Whether a >`learned_budget` assembly exists in the live graph (no graph load).

Nothing in this review writes `~/.bashrc`, opens the live tract, loads a checkpoint, starts or stops a unit, or edits Condensate / the docs worktree / the NG primary checkout.
