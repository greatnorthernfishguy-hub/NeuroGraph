<!--
---- Changelog ----
[2026-09-30] Claude Code (Sonnet 5.5), Z12 worker seat - build-002 return, lane daemon-recall-organism-756b
What: BUILD return for #756 slice B (NeuroGraph ORGANISM reporting callbacks). Docs-only commit; the code
      is the separate commit 42ed7712b43bd6f178b2fa32ceb78590e4ab0db0 on the same branch.
Why: assignment build-002-organism.md (docs worktree daemon-recall-756-20260930, dispatch #12557) including its
     AMENDMENT (Chief-003 decision B: pith_fallback is in this slice's on_degraded scope).
How: read the brief + AMENDMENT, plan-001.md rev 1 (71f40cc2), plan-001-addendum-001-ruling-780.md,
     build-001-addendum-001-drop-pith-wrapper.md, this repo's CLAUDE.md; code at the lane base e4ebf982;
     ONE targeted pytest run under a scratch HOME.
-------------------
-->

# build-002 return: #756 slice B, NeuroGraph organism reporters

Lane `daemon-recall-organism-756b` (a distinct work item under the #756 family), zone manager Z12, worker seat.
Repo `~/NeuroGraph` (worktree `/home/josh/NeuroGraph-worktrees/z12-recall-organism-756b-20260930`).
Related: [[NeuroGraph]], [[The Laws]], [[Duck Ethics]].

## 0. Status

| | |
|---|---|
| Branch | `cc-laptop-recall-organism-756b-20260930` |
| Base (the brief's base, `origin/main` when cut) | `e4ebf982b1989fd9066d610b94853bc68bf70d37` |
| **Code commit** (`cc_ng_organism.py` + `tests/test_cc_recall_reporting.py` ONLY) | `42ed7712b43bd6f178b2fa32ceb78590e4ab0db0` (parent = base) |
| This return | a separate, docs-only commit on top (hash in the closing message; a commit cannot contain its own hash) |
| Targeted run | **46 passed in 3.22 s, exit 0** (21 new + 25 existing unification), ONE run, scratch HOME |
| Rebased onto newer `origin/main`? | **No, as instructed.** `origin/main` is `b5e476863cc069a29ec482959b4f9465f2ea4ccf`, 20 commits past the base; `git diff --stat e4ebf982 origin/main -- cc_ng_organism.py` is **empty** (I re-verified the brief's claim), so those 20 commits do not touch this file. |
| Merge / PR / restart | **None.** Nothing merges now. When it does, **NeuroGraph merges BEFORE the daemon** (a daemon passing the new kwargs to an older organism raises `TypeError`). |
| Files changed vs base | exactly `cc_ng_organism.py`, `tests/test_cc_recall_reporting.py`. Intersection with the protected list (`neuro_foundation.py`, `openclaw_hook.py`, `stream_parser.py`, `activation_persistence.py`), the vendored list (8 files), `data/`, `.claude/`: **none**. No new module. |

## 1. What changed (file:line at the tip `42ed7712`)

Four optional reporters, each **appended last, default `None`**, plus one private guard helper. Nothing else.

| Where | Change |
|---|---|
| `cc_ng_organism.py:6-43` | changelog entry dated 2026-09-30 naming rows #756 / #779 / #780 and this lane (the date is the brief's; the machine's UTC date had already rolled to 2026-10-01 at run time) |
| `:1613` `_cc_report(callback, *args)` | calls an optional reporter and swallows anything it raises (same idiom `cc_assemble_recall` already used for `on_monitor_error`). No logging, no counters (plan section 4: the organism only REPORTS) |
| `:1626-1627`, `:1668` `render_wants` | new `on_error`; `_cc_report(on_error, exc)` in its `except`, **after** the existing `logger.debug`, before `return ""` |
| `:1672`, `:1710` `render_constitutional_core` | same |
| `:2915-2919`, `:3104` `cc_pattern_completion_recall` | new `on_error`; `_cc_report(on_error, exc)` after the existing `logger.debug`, before `return []` (the swallow the brief calls `:3027-3029` at the base) |
| `:5402-5406` `cc_assemble_recall` | new `on_degraded`; docstring `:5439` |
| `:5470-5474` | `except RuntimeError:` becomes `except RuntimeError as exc:`; after the three existing resets, `_cc_report(on_degraded, 'monitor_race', exc)` (base `:5375-5378`: no log, no callback, no counter) |
| `:5493-5498` | the pattern-completion call; only when `on_degraded` is set does it also pass `on_error=<closure that reports 'pattern_completion_failed'>` DOWN into `cc_pattern_completion_recall` (that is where Active Recall actually dies); with `on_degraded` unset the call is `cc_pattern_completion_recall(ng, query, k, state=conv_state)` exactly as at the base |
| `:5507` | outer `except` (base `:5401-5405`): `_cc_report(on_degraded, 'pattern_completion_failed', exc)` after the existing resets |
| `:5625` | the un-Pithed fallback branch (base `:5501-5520`): `_cc_report(on_degraded, 'pith_fallback', exc)` placed **after** the existing `on_pith_failure` block, so the `record_failure()`, the rate-limited WARNING/debug, the `on_pith_failure` call and its own `Pith failure deposit failed` WARNING keep their order and rate (AMENDMENT) |

Six lines were removed in total: the four `def` signatures, `except RuntimeError:`, and the single `cc_pattern_completion_recall(...)` call. Everything else is additions.

**Not touched (as briefed):** `pith_provider_context` (it calls `render_constitutional_core` at `:5171`-region and `cc_pattern_completion_recall` at `:5184`-region at the base with no reporter, so its behaviour is exactly the base's; the sibling silent failure B6 is listed below), `cc_ng_host.py` and everything VPS/Syl, any vendored or protected file, Syl's data/checkpoints/daemon/TID/live tract.

### 1.1 The exact signatures (from `inspect.signature` on the tip, scratch HOME)

```
cc_assemble_recall(ng, query, k, conv_state, commons, allow_pattern_completion=True,
                   on_monitor_error=None, on_pith_failure=None, on_degraded=None) -> str
cc_pattern_completion_recall(ng, query, k=5, threshold=0.4, state=None,
                             preserve_graph_config=False, on_error=None) -> List[Dict]
render_constitutional_core(graph, on_error=None) -> str
render_wants(graph, provenance=('cc_authored', 'cc_emergent'), on_error=None) -> str
```

`on_degraded(code: str, exc: BaseException) -> None`, two positional arguments. `code` is exactly one of the
three strings `'monitor_race'`, `'pattern_completion_failed'`, `'pith_fallback'`.
`on_error(exc) -> None`, one positional argument (the swallowed exception object), same shape as
`on_monitor_error` / `on_pith_failure`.

### 1.2 Call semantics the daemon can rely on

- **Order within one request:** `monitor_race`, then `pattern_completion_failed`, then `pith_fallback` (pipeline order; test `all_three_codes...`). One request can report several.
- **`pith_fallback`:** `on_pith_failure(exc)` is called first (unchanged), then `on_degraded('pith_fallback', exc)` with the **same exception object**. It is reported even if `on_pith_failure` is `None`. It is only reachable when `CC_PITH_ENABLED` is on (default off).
- **Guarded:** a reporter that raises changes nothing: not the returned text, not the existing callbacks, not their order, not the logs or metrics (tests for all three codes and both `on_error` families).
- **Not called for a legitimate empty:** empty query, `ng is None`, an empty graph, `render_wants(None)` do not report (tests).
- **`exc` is the object, not text.** The exception message can carry the prompt, paths or secrets. Log its class at most; never put `str(exc)` on a wire (plan section 3).
- **Caveat for the daemon's wording of `monitor_race`:** the code is set by SITE, not by cause. Any `RuntimeError` raised inside the monitor `try` (`get_surfaced()` or `format_context()`) reports `monitor_race`, which is the documented dict-mutation race but is not proven to be the only source.
- **Possible double report:** `pattern_completion_failed` can arrive twice in one request only if the failure is reported from below AND a later step of the same block raises (the daemon's per-code counters / single-primary-code rule absorb this).
- **`render_constitutional_core(None)`:** by reading, the base raises `AttributeError` inside its `try` for `graph=None` (it has no `None` guard, unlike `render_wants`), so with `on_error` set that case is reported, whereas `render_wants(None)` returns `""` unreported. This asymmetry is the base's and is preserved; the daemon's own `STATE.ng is None` check (plan A9, `ng_unavailable`) comes first. I did not add a test for it.

## 2. What the daemon must pass to use it (a LATER daemon slice; not built here)

```python
cc_assemble_recall(ng, query, k, conv_state, commons,
                   allow_pattern_completion=...,
                   on_monitor_error=<existing>,
                   on_pith_failure=cc_deposit_pith_failure,   # UNCHANGED (decision B; keeps tests/test_cc_recall_unification.py:634 valid)
                   on_degraded=lambda code, exc: ...)          # NEW: record code / log type(exc).__name__
render_constitutional_core(ng.graph, on_error=lambda exc: ...)  # NEW: identity_error core_render_failed
render_wants(ng.graph, on_error=lambda exc: ...)                # NEW: identity_error wants_render_failed
```
`monitor_harvest_failed` stays on the existing `on_monitor_error`; `recall_exception`, `ng_unavailable`,
`organism_unavailable` and the render-`try` split remain daemon-side (plan section 3). Optional for later:
`pith_provider_context` could pass `on_error` to its `cc_pattern_completion_recall` call to fix B6.

## 3. The P379 / #770 preamble and the run

**Command (one run, as the brief specifies; `nice -n 19` and `timeout 540` added as safety wrappers only, output sent to a file):**

```
US=$(python3 -c "import site;print(site.getusersitepackages())")      # computed BEFORE overriding HOME
HOME=<scratch dir> PYTHONPATH=$US PYTHONDONTWRITEBYTECODE=1 nice -n 19 timeout 540 \
  python3 -m pytest -q tests/test_cc_recall_reporting.py tests/test_cc_recall_unification.py -p no:cacheprovider
```
(`PYTHONPATH=$US` also replaces the interactive shell's `PYTHONPATH=/home/josh/NeuroGraph:`, which would otherwise have put the PRIMARY checkout on the path.)

**Mechanism:** `tests/test_cc_recall_reporting.py` prints, to the real stderr (via pytest's capture manager so `-q` still shows it), `cc_ng_organism.__file__` and the resolved path of every NG-looking module in `sys.modules`, tagged `worktree` / `PRIMARY CHECKOUT` / `OUTSIDE WORKTREE` / `external`. It **fails the session** (collection `RuntimeError`, then `pytest.exit(returncode=3)` at session start, then a teardown `pytest.fail` at the end) if `cc_ng_organism` is not this worktree's file, if any module resolves into `/home/josh/NeuroGraph`, or if a name that exists at this worktree's root was loaded from elsewhere. The primary path is taken from the passwd entry, not `~`, so a scratch HOME cannot hide it.

**Verbatim output:**

```
[P379/#770 preamble, session start] worktree root: /home/josh/NeuroGraph-worktrees/z12-recall-organism-756b-20260930
  cc_ng_organism.__file__ = /home/josh/NeuroGraph-worktrees/z12-recall-organism-756b-20260930/cc_ng_organism.py
  primary NG checkout (must NOT be used): /home/josh/NeuroGraph
  NG-looking modules in sys.modules:
    cc_ng_organism                     /home/josh/NeuroGraph-worktrees/z12-recall-organism-756b-20260930/cc_ng_organism.py  [worktree]
    cc_ng_organism_base_e4ebf982       /tmp/cc_ng_organism_base_rrslfgqa/cc_ng_organism.py  [BASE COPY (git show e4ebf982, temp dir; used only for byte-identity)]
    tests                              /home/josh/NeuroGraph-worktrees/z12-recall-organism-756b-20260930/tests/__init__.py  [worktree]
    tests.test_cc_recall_reporting     /home/josh/NeuroGraph-worktrees/z12-recall-organism-756b-20260930/tests/test_cc_recall_reporting.py  [worktree]
    tests.test_cc_recall_unification   /home/josh/NeuroGraph-worktrees/z12-recall-organism-756b-20260930/tests/test_cc_recall_unification.py  [worktree]
  RESULT: PASS -- module under test is this worktree's file; no NG module from the primary checkout
..............................................[P379/#770 preamble, session end (after every collected test, including the unification file)] worktree root: /home/josh/NeuroGraph-worktrees/z12-recall-organism-756b-20260930
  cc_ng_organism.__file__ = /home/josh/NeuroGraph-worktrees/z12-recall-organism-756b-20260930/cc_ng_organism.py
  primary NG checkout (must NOT be used): /home/josh/NeuroGraph
  NG-looking modules in sys.modules:
    cc_ng_daemon_under_test            /tmp/tmp.bujf4jkp4Y/docs/scripts/cc-ng-daemon.py  [external (not an NG-tree file)]
    cc_ng_host                         /home/josh/NeuroGraph-worktrees/z12-recall-organism-756b-20260930/cc_ng_host.py  [worktree]
    cc_ng_organism                     /home/josh/NeuroGraph-worktrees/z12-recall-organism-756b-20260930/cc_ng_organism.py  [worktree]
    cc_ng_organism_base_e4ebf982       /tmp/cc_ng_organism_base_rrslfgqa/cc_ng_organism.py  [BASE COPY (git show e4ebf982, temp dir; used only for byte-identity)]
    ng_tract                           /home/josh/.local/lib/python3.12/site-packages/ng_tract/__init__.py  [external (not an NG-tree file)]
    ng_tract.ng_tract                  /home/josh/.local/lib/python3.12/site-packages/ng_tract/ng_tract.cpython-312-x86_64-linux-gnu.so  [external (not an NG-tree file)]
    surface_resolver                   /home/josh/NeuroGraph-worktrees/z12-recall-organism-756b-20260930/surface_resolver.py  [worktree]
    tests                              /home/josh/NeuroGraph-worktrees/z12-recall-organism-756b-20260930/tests/__init__.py  [worktree]
    tests.test_cc_recall_reporting     /home/josh/NeuroGraph-worktrees/z12-recall-organism-756b-20260930/tests/test_cc_recall_reporting.py  [worktree]
    tests.test_cc_recall_unification   /home/josh/NeuroGraph-worktrees/z12-recall-organism-756b-20260930/tests/test_cc_recall_unification.py  [worktree]
  RESULT: PASS -- module under test is this worktree's file; no NG module from the primary checkout
                           [100%]
46 passed in 3.22 s
```
(exit code 0; the two pytest dots lines are interleaved with the preamble because the preamble bypasses capture.)

**Counts:** `tests/test_cc_recall_reporting.py` has 21 tests; `tests/test_cc_recall_unification.py` has 22 test functions + 3 extra from the one `parametrize` over four Pith failure points = 25; 21 + 25 = 46. `tests/test_cc_recall_unification.py` is **unchanged** vs the base (`git diff --quiet e4ebf982 -- tests/test_cc_recall_unification.py` succeeded).

### 3.1 Which organism does `test_cc_recall_unification.py` actually exercise? (asked for explicitly)

**In this run: this worktree's.** Evidence: the session-end preamble above, taken after the unification file had run, shows `cc_ng_organism`, `cc_ng_host` and `surface_resolver` all resolving into the worktree. Mechanism, from reading the file: line 69 inserts the worktree root at `sys.path[0]` at collection; `import cc_ng_organism` / `import cc_ng_host` inside the tests resolve to it and are cached in `sys.modules`; and the tests that monkeypatch `cc_ng_organism.cc_assemble_recall` patch that one module object, which both wrappers re-import inside `_recall()`. Which tests touch the real organism code: the direct `cc_assemble_recall` tests (Pith gate on/off/fallback/callbacks) and `test_wrappers_bump_error_stat_on_monitor_harvest_failure` (real `cc_assemble_recall` through both wrappers); the wrapper tests that stub `cc_assemble_recall` only prove forwarding.

**Two honest caveats (reasoning, not run):**
1. Line 77 loads the daemon from `~/docs/scripts/cc-ng-daemon.py`, the PRIMARY docs checkout, never the lane's. Under a scratch HOME that path does not exist, so I staged a **byte-identical copy** at the same relative path inside the scratch HOME (sha256 `91e38308f521b3c8cdace41614fcc56cf586abce96175f83734b9e3286711ba9`, identical to `/home/josh/docs/scripts/cc-ng-daemon.py`; primary docs `main` at `391333aaa3633adb944b28d9f6e49d8ecdd5a525` when I read it, read-only). That copy is the OLD daemon: it passes only `on_monitor_error` and `on_pith_failure`, so this run is also an old-daemon x new-organism compatibility check (the default path), and it passed.
2. Under the REAL HOME the daemon's module scope inserts the primary `~/NeuroGraph` at `sys.path[0]` (`cc-ng-daemon.py:540-543`). With the order used here the three NG modules are already in `sys.modules` by then, so it would probably still test the worktree; but any NG module first imported AFTER the daemon loads (for instance a lazy import inside a function) would come from the primary. Under the scratch HOME that insert points at a nonexistent directory, which is why this run is clean. This is the #770 hazard; my preamble protects only sessions that include my file. A repo-wide guard would belong in a `conftest.py`, which I did not add (out of scope).

## 4. Tests (21 new; each "FAILS on base" stated with its mechanism)

Mechanism for every row marked **TypeError**: the new kwarg does not exist at `e4ebf982`, so the call raises `TypeError: ... got an unexpected keyword argument '<kwarg>'`. `test_base_rejects_every_new_kwarg` **proves that in the same run** against the base blob for all four functions (it passed). I did **not** run each reporting test against the base module (that would be a second pytest run, which was not authorized); the claim for those rows rests on that in-run proof plus each test using a new kwarg in its first call.

| Test | Asserts | On base |
|---|---|---|
| `test_pattern_completion_on_error_reports_the_swallowed_exception_and_still_returns_empty` | reporter gets the `RuntimeError`; returns `[]`; the one existing debug record unchanged | TypeError |
| `test_pattern_completion_raising_reporter_does_not_change_the_failsoft_return` | `[]` and same log with a raising reporter | TypeError |
| `test_pattern_completion_does_not_report_a_legitimate_empty` | empty query / `ng None`: no report | TypeError |
| `test_render_constitutional_core_distinguishes_nothing_to_render_from_raised` | raised: reported, `""`, log unchanged; empty graph: `""`, NOT reported; populated: text, not reported | TypeError |
| `test_render_wants_distinguishes_nothing_to_render_from_raised` | same, plus `graph=None` not reported | TypeError |
| `test_render_on_error_raising_reporter_does_not_change_the_return` | `""` / text unchanged | TypeError |
| `test_render_on_error_positional_provenance_still_binds_first` | `render_wants(g, 'cc_authored', on_error=...)` | TypeError |
| `test_assemble_reports_monitor_race` | `('monitor_race', RuntimeError, ...)`; monitor block missing, pattern block intact | TypeError (on base the race is silent) |
| `test_assemble_reports_pattern_completion_failed_from_the_swallow_below` | drives the REAL `cc_pattern_completion_recall`; one `pattern_completion_failed`; monitor block intact; the one existing debug record | TypeError |
| `test_assemble_reports_pattern_completion_failed_from_the_outer_except` | stub raises at the call; one report; existing debug record | TypeError |
| `test_assemble_reports_pith_fallback_and_still_calls_on_pith_failure_first` | all four Pith failure points: order `on_pith_failure` then `pith_fallback`, same exception object, text equals the un-Pithed concat, existing WARNING present | TypeError |
| `test_assemble_pith_fallback_is_reported_even_without_on_pith_failure` | reported with `on_pith_failure=None` | TypeError |
| `test_assemble_all_three_codes_in_one_request_arrive_in_pipeline_order` | `[monitor_race, pattern_completion_failed, pith_fallback]` | TypeError |
| `test_assemble_raising_on_degraded_changes_nothing` | text, log records and `on_pith_failure` calls identical with and without a raising `on_degraded` | TypeError |
| `test_assemble_clean_runs_report_nothing` | healthy request, gate off and gate on (real Pith pipeline): no report | TypeError |
| `test_assemble_pattern_completion_call_shape_is_unchanged_unless_on_degraded_is_set` | unset: call kwargs are exactly `{}` beyond `state`; set: exactly `{'on_error'}` | TypeError (second half) |
| `test_new_kwargs_are_appended_default_none_and_nothing_else_changed` | every existing parameter (name, kind, default) unchanged and in order; the new one last, default `None` | the new parameter is absent |
| `test_base_rejects_every_new_kwarg` | the base blob raises TypeError for each new kwarg | passes on base BY DESIGN (it is the proof) |
| `test_default_path_is_byte_identical_to_the_base_module` | 28 scenarios, base vs new (section 5) | passes on base by construction (guard) |
| `test_default_path_scenarios_exercise_the_paths_they_claim` | non-vacuity: pins the base's own output for the paths that matter | guard |
| `test_module_under_test_is_this_worktrees_file_and_the_base_is_not` | P379 assertion in-test | guard |

Honest note on what is a regression GUARD (passes on base) versus a reporting test (fails on base): the last four rows above (`test_base_rejects_every_new_kwarg`, the two default-path tests, and the P379 test) pass on the base or are about the base itself; every other row fails on base.

## 5. Byte-identity proof (default path vs the base)

The base is loaded from `git show e4ebf982b1989fd9066d610b94853bc68bf70d37:cc_ng_organism.py` into a temp dir and exec'd under its own module name (never from the worktree). The test re-derives the git blob SHA-1 from the bytes and compares it with `git rev-parse e4ebf982:cc_ng_organism.py`, so a truncated or wrong `git show` cannot pass. The same fresh inputs go through the base and the module under test, and each **observation** is compared for equality: the return value (or the exception type and message), **every log record** `(level, message)` on the shared `"cc_ng_organism"` logger, every callback side effect (type and message of what `on_monitor_error` / `on_pith_failure` received), and the `_PITH_METRICS.pith_failures` delta. Each call is made exactly as an existing caller makes it (no new kwarg); the pattern-completion stub in the `cc_assemble_recall` scenarios accepts only `(ng, query, k, state=)`, so an accidental extra kwarg on the default path would raise, be swallowed into an empty block, and show up as a difference.

28 scenarios: `cc_pattern_completion_recall` x4 (empty query, `ng None`, harvest raises, graph without `config`); `render_wants` x6 and `render_constitutional_core` x4 (none/empty/populated/exploding `.nodes`/raises midway on a bad `creation_time` / unsortable `spine_order`); `cc_assemble_recall` x14 (clean, `allow_pattern_completion=False`, monitor `RuntimeError`, monitor `ValueError` with and without a raising `on_monitor_error`, stub raising, the real swallow below, Pith fallback at each of the four failure points with `on_pith_failure`, without it, with a raising one, and the gate-on real-pipeline success). Result: **no mismatch**, plus the non-vacuity test pins that, for example, the base's `monitor_race` path writes **no** log record (the bug) and each Pith failure point bumps `pith_failures` once and writes a WARNING.

Limits, stated plainly: the proof is over these fakes (no real graph was loaded, per the brief), not over every possible input. The success path of the real `cc_pattern_completion_recall` (harvest, GSG re-score, LOD) was not executed in the identity test; by source diff the only edits inside that function are the new parameter, its docstring and one line in the final `except`, so that path cannot differ.

Pre-run exploration (plain Python, not pytest, not the test file; scratch dir, not committed) confirmed three assumptions the tests rest on: the gate-on Pith output is identical across two fresh base loads and the new module (so the success-path comparison is deterministic); the fake reaches the swallow and reports `RuntimeError('INJECTED_HARVEST')`; and no worktree-root filename collides with a loaded module name.

## 6. Collision check (read-only, `git diff origin/main...origin/<branch> -- cc_ng_organism.py`; nothing merged or rebased onto)

| Branch | Hunks (base-relative) | Overlap with my functions |
|---|---|---|
| `z11-card8-quest-removal-20260929` (2 commits: `0ec01c6`, `1950d8d`) | `@@ -3,6 +3,51 @@` **(the changelog header)**, then `-5127,19 +5172,31`, `-5148,10`, `-5165,8`, `-5193,11` (`pith_provider_context`) | **One real, trivial textual conflict the brief's note missed: the changelog header.** That branch inserts its `[2026-09-28]` entry directly under `# ---- Changelog ----`, newest-first; so does mine (I followed the file's newest-first convention). Resolution at merge: keep both entries, mine (2026-09-30) above theirs. The `pith_provider_context` hunks do not overlap any function I touched. |
| `cc-laptop-pith-markers-20260925` (1 WIP commit `b6e9f36`, "STOPPED by Packet 177, not for merge") | `-3243,15`, `-3660,8`, `-4500,6` | none |
| `cc-ng-organism-stale-comment-20260922` (1 commit `a17b1c1`) | `-310,9`, `-452,7`, `-463,12` | none |

## 7. What remains unreportable / listed, not done

- **Daemon-side, later slice:** A1 `_recall` swallow, A2 wire fields, A7 render `try` split (and the `render_wants` raising case that discards an already-computed `self_block`), A9 `ng_unavailable`, `monitor_harvest_failed` (rides the existing `on_monitor_error`).
- **Organism sub-step swallows still `logger.debug` only (context still delivered, one enrichment missing; plan A11):** Pith predictive-promotion per node, LOD query embed, per-node LOD staging, selectivity per-node `firing_rate_ema`, Pith novelty lookup, victim capture, per-line `cc_thermal`. Also **I9**: a failing `_is_identity_protected` lookup inside `_pinned` is treated as not-pinned (Pith gate only) and could let an identity-protected node be budget-evicted. And `_cc_recall_debug_log` (diagnostic, off by default).
- **B6 (listed, not fixed, per the brief):** `pith_provider_context` still reports `state: ok/empty` when Active Recall actually failed; it now could pass `on_error` to fix that, but I was told not to touch it.
- **VPS / Syl's-process half stays JOSH-GATED, listed and untouched (P294(b)):** `cc_ng_host.py:745-778` and `:1141-1149` swallow identically; the new reporters are *available* to that wrapper but wiring them is a separate VPS lane. I did not read or touch Syl's data, checkpoints, daemon or TID.
- **The reason `str(exc)` is never exposed on the wire** is plan section 3's rule; the organism hands over the exception OBJECT and leaves the classification to the daemon.

## 8. Process notes, deviations, observations

- **Gate readings.** The brief recorded 04:18:28 UTC load 2.21. At my first reading load was 7.11 then 7.90 (04:21 UTC), 4.39 immediately before the pytest run (04:30:19 UTC) and 5.32 right after (04:30:40 UTC). `ps` showed other sessions' `opencode` / `codex` processes burning CPU; I cannot tell whether those were worker turns. I ran only the one authorized pytest (3.22 s), a few small `nice -n 19` Python scripts and read-only git/`ls`/`grep`; no full suite, no other NG test file, no graph load, no daemon start, no checkpoint or `data/` path.
- **Deviation 1 (scratch-HOME daemon copy):** described in 3.1; needed because the test loads the daemon through `~`. The brief's command alone would have errored every daemon-using test.
- **Deviation 2:** `nice -n 19`, `timeout 540` and `PYTHONDONTWRITEBYTECODE=1` added to the brief's command (safety/hygiene only; they do not change what runs).
- **No `git pull --rebase`** (the brief forbids rebasing onto the 20 newer commits, and `origin/main` does not touch this file). The code commit's parent is the base.
- **Date.** Local date 2026-09-30 per the brief; the UTC clock had already read 2026-10-01 by the run. Changelog entries use 2026-09-30.
- **Vault docs / wikilinks (global rule: update after a significant code change) NOT done by me.** The vault lives in the docs repo, whose lane worktree I was told not to edit, and this slice is not mergeable yet. Z12 / the docs lane should add the dev-log entry and the NeuroGraph module-page Context Map line when the slice is accepted.
- **No merge, no PR, no restart, no edit in any primary checkout.** Closing check, run after the push: see the closing message.

## 9. Observations for the punchlist (not part of this task, surfaced per the standing rule)

1. **Header-only merge conflict** with `z11-card8-quest-removal-20260929` (section 6): trivial, will occur for whichever merges second.
2. **#770 stays open in the existing tests:** `tests/test_cc_recall_unification.py:77` binds to the primary docs daemon and, under the real HOME, the daemon's `sys.path` insert puts the primary NeuroGraph first. A `conftest.py` guard (module-under-test path assertion) would protect every NG test file, not just mine.
3. **Doc drift:** `CLAUDE.md` section 3 lists `ng_tract.py` in the repo tree ("NOT vendored"), but no `ng_tract.py` exists in either NG tree; `ng_tract` resolves from `~/.local/lib/python3.12/site-packages/ng_tract/` (a compiled extension). Not investigated further.
4. **`monitor_race` is site-coded, not cause-proven** (section 1.2), which matters for the hook text the recall lane will show.
