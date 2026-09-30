# checker-022 ROLE A — pith-clip-removal-813 TURN-2 + TURN-3 DELTA

STATUS: COMPLETE

- Seat: checker-022 (cross-family, grok-4.6, `report_only`). ROLE A of the ADDENDUM only; ROLE B not written.
- Lane: `pith-clip-removal-813`. Dispatch #11044. Zone manager Z12 session `52d39aba-db92-4bf2-b3b1-0e4c13f77d8c`.
- Packet: `/home/josh/docs/.claude/worktrees/daemon-recall-756-20260930/handoffs/z12-silent-failure-lanes-20260930/assignments/review-packet-813-pith.md` ADDENDUM (TURN-2 + TURN-3 DELTA). Read there; not edited.
- Scope: `git diff bc4ae7a 5a634ca2` (NG) and `git diff 44199c00 fd0ccc9c` (docs). Code le-017 reviewed is `9d63846`; turn-2 head `33d212b8`; revert `82cbbcd`; markings `d2f3b78`; return `5a634ca`. Base `e4ebf982`. Not a wider round.
- NG worktree `/home/josh/NeuroGraph-worktrees/z12-pith-clip-813-20260930`, branch `cc-laptop-pith-clip-813-20260930`. `git pull --rebase origin cc-laptop-pith-clip-813-20260930` was already up to date at worker head `5a634ca2db9ddadcd7c2babec56f18e67caf08b2`. This review's stub is `04ffafb88405b26ec8b4e5aff4b8f49374cdb10f`.
- Full hashes (`git rev-parse`): worker return `5a634ca2db9ddadcd7c2babec56f18e67caf08b2`; le-017 verdict `bc4ae7a9306163888b103df35770b84a4c7b080e`; turn-2 head `33d212b857d5b2e70877c776124679157af76e4d`; revert `82cbbcde620ef79ca53feb30039079b886f860f3`; markings `d2f3b7847e4ee838ba7ccb47a2696ab73d1140ec`; base `e4ebf982b1989fd9066d610b94853bc68bf70d37`.
- docs worktree `/home/josh/docs/.claude/worktrees/z12-pith-clip-813-20260930` at `fd0ccc9c08b965e6b9c68baebe6f78a09ed36931` (revert `336954c3bc44ed48e9c71eed3fc9450dc41395d8`). Read only; not edited.
- Inputs read in full: `returns/build-002.md` (banner: #817/#D9 superseded), `returns/build-003.md`, `reviews/checker-019-813-pith.md`, `reviews/le-017-813-pith.md`. Condensate `/home/josh/Condensate` `master` `408654003778b0f090ac3c2975a78bb15fb3277a` READ-ONLY; not re-audited.
- Authority: report_only. No merge, settle, or dispatch. No graph, checkpoint, or tract load. Primary `/home/josh/NeuroGraph` not edited. Real `~/.bashrc` never written. Secrets by NAME only.

## P379 session start

Printed before any NG import, with `PYTHONPATH` and `NG_EMBED_*` unset, cwd the NG worktree:

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

The parent shell had `PYTHONPATH=/home/josh/NeuroGraph:` and `NG_EMBED_REMOTE=hf`. Every command that imported or tested NG used `env -u PYTHONPATH -u NG_EMBED_REMOTE` (and `-u NG_EMBED_MODEL -u NG_EMBED_ENDPOINT -u NG_EMBED_URL`). `tests/test_cc_pith_clip_813.py::test_p379_module_under_test_is_the_worktree_copy` is in the targeted set and passed, so the suite imported this worktree's `cc_ng_organism.py`. Reviewer golden load printed `HEAD cc_ng_organism.__file__: /home/josh/NeuroGraph-worktrees/z12-pith-clip-813-20260930/cc_ng_organism.py`.

Real `~/.bashrc` sha256 before any work, after the staged-script rehearsal, after tests, and at this write: `72f2e7133cce652ac0a8930ed2bef3eca20c63331d49e65f9ba55470604255a3`.

Stub first-write: commit `04ffafb88405b26ec8b4e5aff4b8f49374cdb10f` (`STATUS: INCOMPLETE - review in progress`), pushed to `origin cc-laptop-pith-clip-813-20260930`.

---

## Overall verdict

**PASS-WITH-NOTES**

C1/F1 HIGH is closed on both Pith-ON streams and the gate-off path: `cc_assemble_recall` passes `whole_content=True` as a literal, the monitor stream is re-resolved whole on a CC-only route, and THE ONE budget rule drops whole items with one INFO line. Stage 3's first-line overrun guard is gone (D8: confirm). #818 names member/depth/overlap/root drops. #819 surfaces an over-budget node as trees + one reference line, with the pre-PASS-2 dependency written in the contract and the organism. The turn-3 revert restores `cc_ng_host.py` byte-identical to base and restores `pith_compress_history` plus both handlers; `test_818_*` / `test_819_*` still pass. F3a/F3b hold on `/tmp` copies. Contract and preflight still list the same five required Pith exports.

Targeted NG tests, once: **261 passed, 3 skipped** in 21.15s. Docs preflight tests: **24 passed** in 0.43s. Three reviewer-owned short-node cases vs `git show e4ebf982:cc_ng_organism.py` were byte-identical.

Notes are documentation drift in `pith_stage3`'s still-stale step-5 docstring, a fail-soft residual on the monitor re-resolve, leftover trees inside the #819 reference form that are not given their own INFO line, and pinned lines still sitting off-budget (identity-by-design, not a cut).

---

## A1 C1/F1 HIGH closed

**Verdict: PASS-WITH-NOTES**

### Whole or absent on the Pith-ON L1 path and gate-off

`cc_assemble_recall` (`cc_ng_organism.py:5784-5785`) calls `cc_pattern_completion_recall(..., whole_content=True)`. AST test `test_cc_assemble_recall_asks_recall_for_whole_content_by_literal_true` requires that literal on every Name-call inside the function. `pith_provider_context` (`:5461-5463`) also passes `True`. The default on `cc_pattern_completion_recall` remains `whole_content: bool = False` (`:2947`) — unused by these production callers; `test_recall_default_still_bounds_the_snippet_and_whole_content_does_not` still pins the default as the 300-char snippet.

Monitor stream: after `monitor.get_surfaced()` (`:5756`), `_cc_monitor_items_whole` (`:5605-5635`) re-resolves each item by `node_id` through `resolve_surface_content(..., max_chars=sys.maxsize)`. Shared `surfacing.py` / `surface_resolver.py` are absent from `git diff e4ebf982 HEAD --stat`. Gate-off and Pith-failure fall through to `_cc_render_unpithed` (`:5914`), which admits whole items under THE ONE rule.

`_pith_node_text` (`:4973-4986`) returns `_pith_node_raw_text` unchanged. There is no `def _pith_fit_connected_line`. `_CC_PITH_PROVIDER_NODE_CHARS` is absent as a live constant; `pith_effective_config` reports `CC_PITH_PROVIDER_NODE_CHARS: None` with authority `"retired (#813: nodes render whole)"` (`:3958-3965`).

Regression: `test_816_long_pattern_and_monitor_items_are_rendered_whole` (Pith-ON and gate-off) keeps a 900-char pattern item and a 900-char monitor item with no `"…"` / `"..."`. `test_816_when_the_budget_binds_whole_items_are_dropped_with_an_info_line` binds the budget and requires the INFO line.

### Grep of changed files (`[:N]`, `max_chars`, `...`)

On `cc_ng_organism.py` at HEAD, digit slices are: deposit `text[:2000]` (`:1217`, D4, not this delta's fix); want-id / tract-name / debug prefixes (`:1262`, `:1653`, `:1844`, `:1855`, `:2047`, `:2618`); `_cc_recall_debug_log` `[:15]` / `[:48]` / `[:70]` / `[:120]` (`:5566-5575`, DIAG, gate off). `_pith_cut_at_word_boundary` still slices to `limit` inside `pith_stage2_keyframe` only. `resolve_surface_content` `max_chars` on the L1/provider paths is `sys.maxsize` or the `whole_content` ternary — AST `test_no_resolve_surface_content_call_passes_a_literal_character_cap` forbids a Constant cap. No `may shorten` in the contract.

Count slices that are not character cuts: `out[:k]` in `cc_pattern_completion_recall` (`:3100`, #818 roots, logged); `_pith_tree_nodes` `out[:limit]` (`:4618`, #819 tree list — see A3).

### Residual (not C1 remaining)

`_cc_monitor_items_whole` fail-soft (`:5631-5634`): if re-resolve raises, the wrapper keeps the monitor's already-cut item and logs at DEBUG. Happy-path tests feed a graph node, so the 240-char fake is replaced. An erroring re-resolve would leak a shared-path snippet. Named as C2 below; it does not reopen F1 on the path the pair required.

---

## A2 THE ONE budget rule

**Verdict: PASS-WITH-NOTES** (D8: confirm the Stage 3 first-line guard stays removed)

`_pith_admit_strict_prefix` (`:4704-4730`) is the shared rule: whole size vs empty envelope → never-fit, skip, continue; else remaining envelope → drop and stop. Separator 2 on the provider path (`:5344-5345`), 0 on Stage 3 (`:4840-4842`) and un-Pithed (`:5692-5693`). `_pith_log_budget_drop` (`:4554-4569`) is called only when `dropped` is non-empty (provider `:5348`, Stage 3 `:4844`, un-Pithed `:5694`). Never-fit ids are named first-time-seen, bounded.

Stage 3 no longer keeps the first unpinned line when it exceeds the budget. Implementation comment (`:4834-4839`) and `test_oversized_top_line_is_skipped_whole_never_emitted_over_budget` (`tests/test_pith_stage3.py:88-95`) require skip + keep the later fitting line. `test_cc_pith_clip_813.py` empty-L1 case at `:587` asserts `out == []` (no silent overrun). That is the right product choice: a silent over-budget emit was the only budgeted path that broke whole-or-absent. An all-never-fit recall is empty unless #819's reference form fits — loudly.

Provider envelope cannot overrun: `len(context) > budget` → closed `unavailable` / `context_bound_failed` (`:5494-5495`). Un-Pithed budget-step failure renders every item whole and unbudgeted with a WARNING (`:5700-5702`); `test_unpithed_renderer_fails_open_loudly_when_its_own_budget_step_breaks` holds.

### Remaining overrun / silence (not D8 undo)

1. **Docstring drift (C1 LOW).** `pith_stage3` step 5 (`:4794-4798`) still says a single line longer than the whole budget is kept if nothing has been added yet. The body contradicts that. Future readers will re-introduce the guard from the docstring.
2. **Pinned lines (C4 NOTE).** Pinned content never consumes budget (`:4780-4781`, `test_pinned_reserved_off_budget_does_not_evict_fitting_unpinned`). A large pin makes L1 exceed `budget_chars` with no drop INFO. Identity/constitutional, not a cut, and not the first-unpinned-line guard D8 named.
3. Prefix drops of assemblies that *would* fit an empty envelope are counted in the INFO line (count + chars); only never-fit ids are named. That matches the stated loudness contract.

---

## A3 #819 over-budget node surfaces

**Verdict: PASS-WITH-NOTES**

Rendering-only: `_pith_reference_lines` / `_pith_reference_items` / basin `_display_text` (`:5144-5149`) copy a new CacheLine or dict; `test_819_the_node_itself_is_never_modified_or_split` asserts metadata equality and no extra nodes. No ingest split, no new node type.

Form: one line from `_pith_whole_node_reference` (`:4620-4632`) plus whole tree texts as relations / following concepts. `test_819_provider_over_budget_node_surfaces_through_its_trees_plus_one_reference_line` keeps trees, drops `GIANT-START`/`GIANT-END`, requires one INFO. L1 and un-Pithed share the form (`test_819_l1_and_unpithed_paths_use_the_same_reference_form`, Pith-ON and gate-off). A fitting node is not referenced (`test_819_a_node_that_fits_is_rendered_whole_not_referenced`). Text-derived anchors of the unshown whole are skipped; metadata anchors remain (`test_819_text_derived_anchors_of_the_unshown_whole_are_not_extracted_but_metadata_ones_are`).

Pre-PASS-2 dependency is stated in the organism block (`:4595-4596`), the contract (`docs/PITH_HOST_CONTRACT.md:213-215`), and `build-002.md` §3. The 2,000-char figure is not a new constant in this delta (`text[:2000]` at `:1217` is the D4 deposit clip). Empirical forest coverage is not-verified (no graph load). The statement as written is honest: the reference says the whole exists; trees are whatever `_tree_concept` neighbours the graph already has.

Residual (C3 NOTE): `_pith_reference_text` (`:4662-4667`) `break`s when further whole trees miss the remaining envelope, and `_pith_tree_nodes(..., _CC_PITH_PROVIDER_MEMBERS)` slices the tree list (`:4618`). Those leftover trees are not a second INFO line. The swap itself is logged by `_pith_log_reference`. Worker residual on `budget − core − 800` vs `learned_budget` still holds: a node in that window can never-fit at admit without trees.

---

## A4 #818 every drop is loud

**Verdict: PASS**

`_pith_log_drop` (`:4572-4587`) no-ops on an empty list, so "nothing dropped" is silent (`test_818_nothing_dropped_logs_nothing`). One INFO line per reason with count, chars, and first-time-seen ids:

| Reason | Site | Test |
|---|---|---|
| `member_limit` | `pith_connected_activation_basins` `:5303-5306` | `test_818_member_limit_drops_are_counted_sized_and_named_once` (second call flood-safe) |
| `depth_limit` | `:5307-5310` | `test_818_depth_limit_drops_the_neighbours_the_walk_declined` |
| `overlap` (≥60%) | `:5311-5312` | `test_818_overlapping_basins_are_dropped_loudly` |
| `roots` (`out[:k]`) | `cc_pattern_completion_recall` `:3100-3103` | `test_818_roots_beyond_k_are_dropped_loudly` |

Declined neighbours count only if they appear in no selected basin (`:5301-5302`). What is selected is unchanged. Stage-1 clutter/dedup remains counted-not-logged (named in `build-002.md` §5; outside this item). Constitutional neighbours skipped before `_display_text` are core, not a learned drop.

---

## A5 THE REVERT (turn 3)

**Verdict: PASS**

`git diff e4ebf982 HEAD --stat -- cc_ng_host.py` is empty. Blob `e3b568c0b5f1f54a14525eb9b5c76e74b7eb2f96` is the same at base and HEAD. Host handler `cc_ng_host.py:974` `_handle_compress_history` and `_DISPATCH` `:1425` `"compress_history"` are present.

`pith_compress_history` is restored at `:4449`. `PithMetrics.history_calls` `:3745` through `history_failures` `:3751`; `_reset_locked` `:3808`; `snapshot` `:3879`; `record_history_compression` `:3833`. The only production call of `pith_stage2_keyframe(` in this file is `:4490` (`keyframe, _delta = ...`) inside `pith_compress_history`. AST `test_complete_caller_set_of_the_cutters_is_empty_outside_the_keyframe_primitive` asserts `keyframe_callers <= {"pith_compress_history"}` and `word_cut_callers <= {"pith_stage2_keyframe"}`, matching the deferred-#817 comment.

History tests vs base: `HEAD:tests/test_cc_host_compress_history.py` blob `b4703f5ed3d6619542bcaaf6e7ebbe8725f856f2` equals `e4ebf982:...`; `HEAD:tests/test_pith_history_metrics.py` blob `4afb0861a5c4fddc0f5dfe6e6f2b5bc5f58cd6f7` equals `e4ebf982:...`.

docs `scripts/cc-ng-daemon.py` blob equals branch base `7cf85149` (`faae31a7228873c3a8052377f9028f9ae23ea22d`); `handle_compress_history` `:1617` and DISPATCH `:1722` are restored. `git diff 44199c00 HEAD --stat -- scripts/` is empty: the only docs delta vs the turn-1 reviewed head is the vault pointer `handoffs/z12-pith-clip-813/plan-001.md`.

Kept turn-2 tests: six `test_818_*` and seven `test_819_*` functions are present and were in the passing targeted run. Nothing kept from turn 2 required the #817 removal: `pith_stage2_keyframe` again has exactly one caller. D9 in `build-002.md` is moot (banner + `build-003.md` §5). Row 10 of the audit stays a CUT and is marked DEFERRED (`plan-001-audit.md` row 10 + §11). Protected/vendored/shared files (`surfacing.py`, `surface_resolver.py`, `neurograph_rpc.py`, `neuro_foundation.py`, `openclaw_hook.py`, `ng_*`, `kiss_filter.py`, `tonic_thread.py`) have empty `git diff e4ebf982 HEAD --stat`.

---

## A6 F3a/F3b staged `.bashrc` script

**Verdict: PASS**

Script: `handoffs/z12-pith-clip-813/returns/bashrc-drop-node-chars.sh` sha256 `47e631b0376d8b9895c0d54bdd67272561cb199e53c59ce57c48d8c4aac86c0f`. Removal is by pattern `^export CC_PITH_PROVIDER_NODE_CHARS=`; `sed -i --follow-symlinks`; `apply` runs `verify` and on failure restores the just-made backup then exits non-zero (`:72-82`). `verify` returns failure explicitly. Apply prints the S4 checklist reminder (the script cannot check that both merges are deployed).

Rehearsal, `BASHRC` pointed at copies under `/tmp/bashrc-813-checker022-*` only; copies deleted afterwards:

- Copy of the real file: apply removed exactly the `CC_PITH_PROVIDER_NODE_CHARS` export (diff vs real, values redacted, was that one line); verify printed the five keeper **names**; reverse restored sha256 `72f2e7133cce652ac0a8930ed2bef3eca20c63331d49e65f9ba55470604255a3`.
- Zero-match apply rc=1, file unchanged; two-match apply rc=1, file unchanged.
- F3a: sole export inside `if true; then … fi` → verify failed (`bash -n` at `fi`), rollback, rc=1, byte-identical to pre-apply.
- F3b: symlink stayed a symlink; target lost then regained the export; reverse left the target byte-exact vs the real file.

Real `~/.bashrc` sha256 before and after: `72f2e7133cce652ac0a8930ed2bef3eca20c63331d49e65f9ba55470604255a3`.

---

## A7 Contract + preflight

**Verdict: PASS**

NG `docs/PITH_HOST_CONTRACT.md` sha256 `ea97b352768c7243c2f33e05888ec8c83c641e78c98e8bfb7abf0e242b86366c`. Export block is five names (`:126-131`). `CC_PITH_PROVIDER_NODE_CHARS` is gone; a stale export is "ignored and harmless." `may shorten` is absent. THE ONE rule, #818 loud drops, #819 reference form with the pre-PASS-2 dependency, and the #817 DEFERRED note on the restored `compress_history` section (`:238-244`) are in the same file.

docs `scripts/cc-ng-service.py` `preflight` required tuple (`:117-122`) is those five Pith names plus the pre-existing launch keys. `NODE_CHARS` is not required. `git rev-parse HEAD:scripts/cc-ng-service.py` = `4ab3882773f47cc2c6eba9212e04f48896697e4a`, identical to `44199c00`. Docs tests from the docs worktree, `HOME` override after `site.getusersitepackages()`, `NG_EMBED_*` unset, NG-related `sys.modules: NONE` before pytest, service path `.../z12-pith-clip-813-20260930/scripts/cc-ng-service.py`: **24 passed** in 0.43s (`test_cc_ng_service.py` + `test_pith_provider_context_wrapper.py`). That is 25 minus the retirement test the revert removed, matching `build-003.md` §3.

The pair still changes together versus base: NG contract (this branch) and docs preflight (`8cba7963`, still on the docs head). Turn 3 did not split them.

---

## A8 Tests

**Verdict: PASS**

Worker targeted files, once, `env -u NG_EMBED_REMOTE -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1`, cwd the NG worktree, `-p no:cacheprovider`:

`tests/test_cc_pith_clip_813.py tests/test_pith_provider_context.py tests/test_pith_stage{1,2,3,4,5}.py tests/test_pith_l1_provenance.py tests/test_pith_metrics_concurrency.py tests/test_cc_host_pith_telemetry.py tests/test_cc_recall_dedup.py tests/test_cc_recall_unification.py tests/test_cc_region_confidence.py tests/test_pith_history_metrics.py tests/test_cc_host_compress_history.py`

**261 passed, 3 skipped** in 21.15s. The three skips are `test_vps_host_and_laptop_daemon_have_identical_closed_contract[ok|empty|unavailable]` (`CC_DAEMON_UNDER_TEST` unset). Not re-run with the daemon path (packet: once; mixing that env poisons later imports — `build-002.md` §8).

Golden vs BASE: fixture `tests/fixtures/pith_clip_813_golden_base.json` `base_commit` starts `e4ebf982`; 4 provider scenarios (`live_rail_placeholder`, `single_chain_short`, `stale_member_alert`, `two_assemblies_short`) and 6 `cc_assemble_recall` scenarios (Pith-ON and gate-off × both / pattern only / monitor only). `test_short_wellformed_nodes_render_exactly_as_base` and the recall golden test are in the passing set.

Three reviewer-owned cases vs `git show e4ebf982:cc_ng_organism.py` (base blob loaded as a separate module; fake graph only; `whole_content` not involved — recall swapped):

| Case | equal | state | context chars | assemblies |
|---|---|---|---|---|
| `short_single` | True | ok/ok | 396/396 | 1/1 |
| `three_tiny_assemblies` | True | ok/ok | 721/721 | 3/3 |
| `source_and_anchor` | True | ok/ok | 430/430 | 1/1 |

Turn-2 updates to old tests carry changelog reasons (`test_pith_stage3.py` oversized-top-line now skip-not-keep; `test_cc_recall_unification.py` / `test_cc_recall_dedup.py` / `test_cc_region_confidence.py` accept `whole_content`; `test_pith_stage2.py`). None of those weaken a regression to hide a cut: the HIGH path has new failing-first tests (`test_816_*`, `test_818_*`, `test_819_*`).

---

## Verdict

| Item | Verdict |
|---|---|
| A1 C1/F1 HIGH closed | PASS-WITH-NOTES |
| A2 THE ONE budget rule / D8 | PASS-WITH-NOTES (confirm: keep the guard removed) |
| A3 #819 | PASS-WITH-NOTES |
| A4 #818 | PASS |
| A5 turn-3 revert | PASS |
| A6 F3a/F3b script | PASS |
| A7 contract + preflight | PASS |
| A8 tests | PASS |
| **Overall** | **PASS-WITH-NOTES** |

D8: Stage 3's first-line overrun guard should stay removed. Restoring it would re-create the only budgeted path that emitted over budget.

---

## Numbered corrections

1. **C1 (LOW).** Update `pith_stage3`'s step-5 docstring (`cc_ng_organism.py:4794-4798`). It still describes "keep the first unpinned line even if it alone exceeds the budget." The body, THE ONE rule block (`:4515-4526`), and `test_oversized_top_line_is_skipped_whole_never_emitted_over_budget` all skip that line. This is documentation, not a merge blocker; leaving it invites a LAW-3 reintroduction of the guard.

2. **C2 (NOTE).** `_cc_monitor_items_whole` (`:5631-5634`) fail-soft keeps the monitor's given content (the shared 240-char snippet) when re-resolve raises, logged at DEBUG. Happy path is whole. An erroring re-resolve is a silent residual cut on that one item. Not F1 remaining; do not "fix" it by editing `surfacing.py`.

3. **C3 (NOTE).** Inside the #819 reference form, leftover trees that miss the remaining envelope (`_pith_reference_text` `:4664-4665`) and trees past `CC_PITH_PROVIDER_MEMBERS` (`_pith_tree_nodes` `:4618`) have no dedicated INFO line. The node swap is loud. Not a node-text cut.

4. **C4 (NOTE).** Pinned Stage 3 lines still do not consume budget, so L1 can exceed `budget_chars` with no drop INFO. Identity-by-design; outside D8. Do not fold pins into THE ONE rule without an Exec call.

No HIGH remaining. No correction required before the delta pair's ROLE B.

---

## Not-verified

- Live daemon, miniTID PID, `:9090`, `daemon.sock`, tracts, checkpoints, or a running INFO log.
- The three host/daemon parity tests (skipped without `CC_DAEMON_UNDER_TEST`; not mixed into the one targeted run).
- Rust build / `cargo test` (Condensate read-only; not this delta).
- VPS host process, `.bashrc` apply on the real file, S4 batch execution.
- Full NG or docs suites.
- Empirical pre-PASS-2 forest coverage of 2,000 chars (no graph load).
- F4/F5/F7 from le-017 (LOD_DIST phantom knob, punchlist filing, `.bashrc` mode 644) — outside this delta's claimed work; `CC_PITH_PREFETCH_LOD_DIST` is still reported as a live resolved setting (`:3962`).
- Whether a >`learned_budget` assembly exists in the live graph.

Nothing in this review writes `~/.bashrc`, opens the live tract, loads a checkpoint, starts or stops a unit, or edits Condensate / the docs worktree / the NG primary checkout.
