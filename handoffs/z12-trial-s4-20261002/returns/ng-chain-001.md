<!--
# ---- Changelog ----
# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane ng-trial-chain-s4, dispatches #14575/#14590) — return ng-chain-001 for build NG-TRIAL-CHAIN (S4 minimum step 4; Exec P538 / P555 via Chief-003)
#   What: the NG TRIAL integration branch `cc-laptop-trial-s4-20261002` = NG origin/main b5e47686 + fourteen `cherry-pick -x` (N1-N7 + #905 round 2); proofs; gated test runs; PYTHONHASHSEED 0..31 sweep.
#   Why: the laptop trial runs from a worktree (LAW 5 knob CC_NG_PYTHONPATH); NOT main; the NG merge-hold (P478/P480) STANDS; TRIAL ONLY, never merged by this route; no VPS; NG before the daemon.
#   How: docs-only commit on top of the chain (no code). Nothing pushed, no upstream, no `git pull`.
# -------------------
-->

# NG TRIAL chain — return ng-chain-001

**Branch:** `cc-laptop-trial-s4-20261002` &nbsp; **Worktree:** `/home/josh/NeuroGraph-worktrees/trial-s4-20261002` &nbsp; **Base:** NG `origin/main` = `b5e476863cc069a29ec482959b4f9465f2ea4ccf` (re-read after `git fetch` at start: **did not move**).
**Code tip (the 14th pick, before this docs commit):** `9b647e39926d5c4dc28dcbd789e7eef3128f9819`, tree `a5238ebf26fc5c3334a6c83a4bf092d3e56246ee`. This return is ONE docs-only commit on top; the branch tip moves by that commit.
**Trial only. Never merged. No push, no upstream, no `git pull`, no VPS, no live daemon/graph/checkpoint/tract/socket, no `~/.bashrc` write, no real `systemctl`, nothing deleted.**

## Headline (read this first)

1. **All 14 picks landed, in the brief's order, each `-x`.** 12 are content-equivalent to their originals (non-comment +/- multiset equal). The two hand-resolved commits (N2, N6) differ from their originals in exactly the lines the resolution touches (below). **No protected or vendored file is in the diff.** No STOP condition fired.
2. **Two integration findings the Chief must see (reported, NOT fixed; a fix would edit a test, beyond a cherry-pick chain):**
   - **F-A (#794 test vs D24):** `tests/test_cc_drain_hold_on_failure.py::test_signature_hold_on_failure_is_last_and_defaults_false` pins the EXACT pre-D24 signature `[..., max_entries, hold_on_failure]`. On the integrated tip the signature is `[..., max_entries, batch_nodes, receipt, hold_on_failure]` (hold_on_failure still LAST, default False). The test FAILS **deterministically, 32/32 seeds**; the other 48 of that file pass. Not a code defect; a stale pin. The owed fix is a one-line test update (prefix list + `params[-1] == "hold_on_failure"`).
   - **F-B (#904 test vs #905):** `tests/test_cc_bind_atomic_904.py::test_generate_emergent_want_is_untouched` (docstring: "#905 owns it: its source must be byte-identical to the base") FAILS once #905 part A lands in the same tree. It cannot hold in either merge order. The owed fix is to retire or re-base that pin when #905 and #904 are integrated.
3. **One behavioural observation on the N2 composition (reported, not changed):** when `hold_on_failure=True` breaks the loop, `ended` keeps its initial value `"tract_exhausted"`, so `receipt["reason"]` says `tract_exhausted` for a held (not exhausted) tract. Neither original defined a reason for that case (D24's reasons: size_reached | entries_cap_reached | tract_exhausted | parse_failed | no_batch; #794 knows nothing of the receipt). Adding a `held` reason would be a choice neither made, so I did NOT. Any consumer of `receipt.reason` under the flag (the daemon's phase 2) should be read with this in mind.
4. **Brief-vs-reality notes:** (a) N7 (#918 x3) was expected "clean on N6b" but all three had a comment-only header conflict in `cc_topology_merge.py`; (b) N6b had a comment-only header conflict in `cc_ng_organism.py`; (c) the brief describes N6's D24 side as having "guard/progress parameters" — it does not: on the D24 side `_cc_callosum_consolidate` is still `(graph, idle_steps)`; D24's contribution to that function is the loud `logger.error` in the `except`; the guard/progress signature and loop came in from #905 without conflict. All conflict regions were machine-checked to contain 0 code lines (except the two hand-resolved code hunks).

## The fourteen picks

| # | new hash (trial) | original | non-comment +/- lines new/orig | content-equivalence | range-diff |
|---|---|---|---|---|---|
| N1a | `81807e7add0d` | `38febe6e7819` | 502 / 502 | EQUIV | `!` (message trailer; see note) |
| N1b | `ce7a864064ed` | `24a497dd4521` | 273 / 273 | EQUIV | `!` (message trailer; see note) |
| N1c | `9378b848e204` | `c798a8ed5f03` | 138 / 138 | EQUIV | `!` (message trailer; see note) |
| N2 | `0a6780407345` | `c625623ebcbd` | 415 / 415 | DIFFER (hand-resolved, see below) | `!` (message trailer; see note) |
| N3a | `961deede7ffb` | `895a809a60ab` | 1125 / 1125 | EQUIV | `!` (message trailer; see note) |
| N3b | `c1b689063284` | `12d6241b03fe` | 247 / 247 | EQUIV | `!` (message trailer; see note) |
| N3c | `e92e815f9c22` | `59d332c75459` | 198 / 198 | EQUIV | `!` (message trailer; see note) |
| N4 | `b5e114087b22` | `42ed7712b43b` | 840 / 840 | EQUIV | `!` (message trailer; see note) |
| N5 | `c01316b56030` | `ac5800fd7230` | 493 / 493 | EQUIV | `!` (message trailer; see note) |
| N6 | `aa15336e356e` | `ee94f7d2c516` | 1060 / 1062 | DIFFER (hand-resolved, see below) | `!` (message trailer; see note) |
| N6b | `2f4b928f54f1` | `f685a3a7715d` | 18 / 18 | EQUIV | `!` (message trailer; see note) |
| N7a | `e11da06c09ca` | `86cf5afd9180` | 863 / 863 | EQUIV | `!` (message trailer; see note) |
| N7b | `bd7f73755e59` | `c3c76432e9b2` | 347 / 347 | EQUIV | `!` (message trailer; see note) |
| N7c | `9b647e39926d` | `b1949155b8e2` | 120 / 120 | EQUIV | `!` (message trailer; see note) |

**How to read the table.** `non-comment +/- lines` = the multiset of `+`/`-` lines of `git show -U0` with comment lines (`#...`) removed, per file; script `/tmp/z12-s4/equiv.py` (scratch, not committed). The comment counts for the hand-resolved commits differ by the changelog lines I added (N2: 65 vs 62 = +3; N6: 104 vs 101 = +3). **Range-diff note:** `git range-diff <orig>^! <new>^!` prints `!` on EVERY row, including the clean ones, because the `-x` trailer `(cherry picked from commit ...)` changes the message and rebased context shifts; so `=`/`!` cannot separate real change from trailer. The content-equivalence column is the proof; the two full range-diffs that matter are in the appendices.
**Messages / authors:** verified for all 14: message identical to the original apart from the `-x` trailer, author name/email/date identical, exactly one `(cherry picked from commit <orig>)` trailer naming the right original. (Conflicted picks were committed with `git commit --cleanup=verbatim -F` — see "Process incidents" for why.)
**Header-only conflicts (N3 x3, N4, N6b, N7 x3):** resolved keep-both, HEAD side first, then the picked commit's entry; each region checked to contain only `#`/blank lines (script `resolve_comment_only.py` aborts and writes nothing otherwise). I added NO extra changelog line to these (the brief asks for one only on hand-resolved commits; I read "hand-resolved" as N2 and N6, the two named code resolutions). Say if you want one on each.

## The two hand-resolved commits

### N2 — #794 `c625623e` -> `0a678040` (file: `cc_ng_organism.py`; three conflict regions)
1. **Header (comment-only), `:6-:20`** — kept both entries (D24's `2026-10-01` first, #794's `2026-09-30` below it) plus ONE added changelog entry naming this resolution (`:9-:11` on the N2 commit; on the tip it sits under N6's entry).
2. **Signature, `:2916-:2918` (tip)** — resolved to `batch_nodes: int = None, receipt: dict = None, hold_on_failure: bool = False`. WHY: #794's own text says "ONE new LAST keyword, hold_on_failure=False", D24's two are earlier on the chain, and a keep-both of the two signature lines does not parse (the measured `SyntaxError`). Existing positional callers `(graph, vector_db, state, tract_path, return_consumed, max_entries)` are unaffected; all three new params are keyword-with-default.
3. **Loop body, `~:3085-:3112` (tip; the hold block is at `:3095`, D24's size check at `:3104`)** — resolved to: #794's `hold_reason` / `hold_exc_type` bookkeeping, then `if hold_on_failure and hold_reason is not None:` (warning + `break`), THEN D24's `if size_cap and (len(graph.nodes) - len(ids_before)) >= size_cap:` (`ended = "size_reached"`; `break`), THEN `if max_entries and taken >= max_entries:` (`ended = "entries_cap_reached"`; `break`). WHY this order: it keeps D24's own order (size check before the entries cap, both after the absorb) and puts #794's hold-break first so a failed entry stops the drain before any pacing rule can count it. `safe_offset` bookkeeping and the `if hold_on_failure: consumed_offset = safe_offset` tail are #794's, untouched; `ids_before`, `_fill_receipt`, `_ret`, `size_cap` are D24's, untouched.
4. **Both behaviours intact:** D24's `batch_nodes`/`receipt` pacing (the daemon reads the receipt) and #794's hold (the daemon's #921 clause 1 inspects `hold_on_failure` in the signature — it is present, last, default False). With `hold_on_failure=False` the loop is D24's; with `batch_nodes`/`receipt` unset it is #794's. The interplay case (flag on AND pacing on): a held entry breaks before the size check (the failed entry is not counted toward the size; `receipt.reason` is the observation in Headline 3).

**No behavioural choice neither original made** was needed to resolve it; Headline 3 is the one place the composition is silent, and I left it silent.
**Proofs:** `git range-diff c625623e^! 0a678040^!` (Appendix A): the only non-header inner-diff change vs #794 is the signature's `batch_nodes, receipt` carrier line. **D24 side:** the non-comment lines the resolved commit REMOVES from its parent (the D24 tip `9378b848`) are exactly one: the old signature line `batch_nodes: int = None, receipt: dict = None):` — re-added with the trailing comma and the new keyword. Equivalence vs #794: 415/415 non-comment lines; the 2 `+`/`-` pairs that differ are exactly the signature lines.

### N6 — #905 `ee94f7d2` -> `aa15336e` (file: `cc_ng_organism.py`; two conflict regions)
1. **Header (comment-only)** — kept both sides, HEAD first, plus ONE added changelog entry naming this resolution (`:6-:8` on the tip).
2. **`_cc_callosum_consolidate` `except` block, `:3340-:3350` (tip)** — resolved to D24's loud `logger.error("CC callosum consolidation FAILED after %d of %d step(s) (%s); ...", done, idle_steps, type(exc).__name__)` (`:3346`, replacing the `logger.debug` line, which is exactly D24's change over the base) FOLLOWED BY #905's `_fill(done, failed=True)` (`:3349`) then `return False`. WHY: D24 changed the log line, #905 added the progress fill; they touch adjacent lines of the same `except` and compose by union; the debug line is dropped because D24's loud line IS its replacement. The rest of the function (the `guard=`/`progress=` signature at `:3251`, the per-slice guard loop, `_fill` on every return path) merged without conflict from #905.

**No behavioural choice neither original made.**
**One more (explained) difference from the original, outside the conflict:** the original #905 diff contains `-        done = 0` / `+    done = 0` (moving `done = 0` out of the `try`). D24 had ALREADY made that same move (its loud log needs `done` in scope), so in the integrated tree that pair is a no-op and is absent from the pick: the 1060 vs 1062 non-comment count is exactly that pair. `done = 0` is at function level in the tip (once).
**Proofs:** `git range-diff ee94f7d2^! aa15336e^!` (Appendix B). **D24 side:** the non-comment lines the resolved commit removes from its parent (N5 `c01316b5`) are #905's own: the old signature line and the old docstring tail line `Returns True if the steps ran. Fails soft.` — D24's `logger.error` is NOT removed. Equivalence vs #905: 1060/1062; the only differing `+`/`-` pair is `done = 0`.

## Whole-chain assertions

**`git diff --name-only b5e47686..9b647e39` (the 14 code picks; this docs commit adds only `handoffs/z12-trial-s4-20261002/returns/ng-chain-001.md`):**
```
cc_ng_organism.py
cc_topology_merge.py
tests/test_cc_ack_bound_918.py
tests/test_cc_bind_atomic_904.py
tests/test_cc_drain_hold_on_failure.py
tests/test_cc_drain_pacing.py
tests/test_cc_drain_pacing_seam.py
tests/test_cc_emergent_want_bound_905.py
tests/test_cc_merge_whole_graph_guard.py
tests/test_cc_recall_reporting.py
tests/test_cc_topology_callosum.py
tests/test_cc_topology_capture_423.py
```
**Protected / vendored hits** (`neuro_foundation.py`, `openclaw_hook.py`, `stream_parser.py`, `activation_persistence.py`, `ng_lite.py`, `ng_tract_bridge.py`, `ng_ecosystem.py`, `openclaw_adapter.py`, `ng_autonomic.py`, `ng_embed.py`, `ng_updater.py`, `ng_salience_gate.py`, `data/checkpoints/*`, `.claude/hooks/*`): **NONE.**
**Checksums of the code chain:** commit `9b647e39926d5c4dc28dcbd789e7eef3128f9819`; tree `a5238ebf26fc5c3334a6c83a4bf092d3e56246ee`; **sha256 of `git diff b5e47686..9b647e39` = `43938e2bed124b55bd755b83d0f8f837aa4d0d9a2236bdaf87f2fce997f1995c`**. Every changed `.py` file `ast.parse`s; no conflict markers remain anywhere (`git grep` clean).

## Runs (all gated, spaced 25 s before each, named files only)

**Gate:** `/proc/loadavg` read ONCE per run AFTER the 25 s spacing; refuse (and stop the batch) at 1-min >= 6.0 OR 5-min >= 5.0; never retried; never lowered. **No run was refused** (max START 1-min 4.16 and 5-min 2.71 in the 32-seed sweep; the 10 slice runs started at 1-min <= 1.84, 5-min <= 1.79). **FORBIDDEN and not run:** `tests/test_cc_deposit_step.py`, the NG full suite. **`env -u` form every run used:** `/usr/bin/env -u CC_NG_BATCH_SIZE -u CC_NG_IDLE_STEPS -u CC_NG_DRAIN_HOLD_ON_FAILURE -u NG_EMBED_REMOTE -u CC_NG_IN_TRANSIT_IDS_PATH python3 -m pytest -q -p no:cacheprovider -rs <named files>`. **EFFECTIVE values PRINTED AND SEEN in every record:** all five of those `'<unset>'`; `PYTHONPATH='<unset>'` (and `CC_NG_PYTHONPATH` removed); `HOME='/tmp/z12-s4/home'` (scratch); `PYTHONUSERBASE='/home/josh/.local'` (read-only, so the already-installed `pytest` and `ng_tract` import); `PYTHONDONTWRITEBYTECODE=1`; `PYTHONHASHSEED='<unset>'` for the slice runs and `=<seed>` for the sweep. Only R2 added `Z12_D24_SKIP_UNMERGED=1` (the test's own documented opt-out for "NeuroGraph merged BEFORE the daemon"). Under the scratch `HOME` its default daemon path `~/docs/scripts/cc-ng-daemon.py` expands to `/tmp/z12-s4/home/docs/scripts/cc-ng-daemon.py`, which does not exist, so the four real-loop tests (`:277 :299 :323 :385`) skipped with the reason text "daemon file ... UNREADABLE" (not "no D24"); the real primary daemon has no D24 either (grep count 0), so the outcome would be the same. **No daemon was read or loaded.** The loading tests' preambles print their module paths inside the worktree (`worktree check: PASSED`).

### Slice test runs on the integrated tip `9b647e39`

| run | file(s) | START load 1m / 5m / 15m | result (integrated tip) | original slice return reported |
|---|---|---|---|---|
| R1 N1 D24 pacing | `test_cc_drain_pacing.py` | 1.15 / 1.79 / 3.56 | 23 passed | 23 passed (D24 return build-004) |
| R2 N1 D24 seam (unmerged-daemon opt-out) | `test_cc_drain_pacing_seam.py` | 1.25 / 1.76 / 3.49 | 8 passed, 4 skipped | 10 passed + 2 skipped WITH a D24 daemon copy (build-004); no daemon here |
| R3 N2 #794 hold_on_failure | `test_cc_drain_hold_on_failure.py` | 1.12 / 1.69 / 3.41 | 1 failed, 48 passed | 49 passed (build-794 / #904 build-003) |
| R4 N3 #904 bind atomic | `test_cc_bind_atomic_904.py` | 0.89 / 1.57 / 3.32 | 1 failed, 55 passed, 5 skipped | 56 passed + 5 skipped w/o daemon (#904 build-003) |
| R5 N4 #756b recall reporting | `test_cc_recall_reporting.py` | 0.87 / 1.50 / 3.24 | 21 passed | 21 tests in the file (756b build-002: 46 = 21 + 25 of the unification file, not in this chain) |
| R6 N5/N6/N7 #897 merge guard | `test_cc_merge_whole_graph_guard.py` | 0.65 / 1.40 / 3.15 | 17 passed, 2 skipped | 17 passed, 2 skipped (#905 / #918 returns) |
| R7 N6 #905 emergent want | `test_cc_emergent_want_bound_905.py` | 0.84 / 1.37 / 3.09 | 60 passed | 60 passed (#905 build-002) |
| R8 N6/N7 capture_423 | `test_cc_topology_capture_423.py` | 0.78 / 1.31 / 3.02 | 4 passed | 4 passed (#905 / #918 returns) |
| R9 N7 #918 ack bound | `test_cc_ack_bound_918.py` | 1.65 / 1.45 / 3.01 | 42 passed | 42 passed (#918 build-003) |
| R10 N7 topology_callosum | `test_cc_topology_callosum.py` | 1.84 / 1.52 / 2.98 | 1 failed, 33 passed | 33 passed, 1 failed = the known #902 `poincare_dir` failure (#897 / #918 returns) |

**Where the counts differ from the original returns, and why:**
- **R3 (#794): 48 passed, 1 failed vs 49 passed.** F-A: the exact-signature pin predates D24. Not a defect of the resolution.
- **R4 (#904): 55 passed, 1 failed, 5 skipped vs 56 passed, 5 skipped.** F-B: the #904 "#905 owns it" pin against #905 part A. The 5 skips are the daemon-loop tests (`Z12_904_DAEMON_UNDER_TEST` unset; no daemon in this trial).
- **R2 (D24 seam): 8 passed, 4 skipped vs 10 passed, 2 skipped.** The original had a D24 daemon copy; here there is none by design (NG before the daemon), so the four real-loop tests skip (reason printed: the daemon path is unreadable under the scratch HOME). The original's two real-#905 tests (10 passed + 2 FAILED LOUD on the D24-only tree) are not skipped here: `organism_has_905()` is true on this tip. I did not break the 8 down by name; the skip reasons are in the run record.
- R5: the 21 tests of `test_cc_recall_reporting.py` pass; the 756b return's 46 included the unification file, which is not in this chain.
- R10: the single failure is `test_poincare_dir_is_rederived_locally_not_transmitted` — the known #902 failure both the #897 and #918 returns reported — and nothing else.
- R1, R6, R7, R8, R9: identical to the originals.

### PYTHONHASHSEED 0..31 sweep (N2 + N6 integration points)
One invocation per seed, `tests/test_cc_drain_hold_on_failure.py tests/test_cc_emergent_want_bound_905.py`, each gated and spaced. **All 32 ran, none refused. 32/32 identical: `1 failed, 108 passed`, the failing test being the same one every time (`test_signature_hold_on_failure_is_last_and_defaults_false`, F-A).** 108 + 1 = 49 + 60: no order-dependent or seed-dependent behaviour at the two integration points.

| seed | START 1m / 5m / 15m | result |
|---|---|---|
| 0 | 2.15 / 1.68 / 2.96 | 1 failed, 108 passed |
| 1 | 2.48 / 1.82 / 2.97 | 1 failed, 108 passed |
| 2 | 2.42 / 1.90 / 2.96 | 1 failed, 108 passed |
| 3 | 2.15 / 1.89 / 2.91 | 1 failed, 108 passed |
| 4 | 1.98 / 1.88 / 2.88 | 1 failed, 108 passed |
| 5 | 2.19 / 1.94 / 2.87 | 1 failed, 108 passed |
| 6 | 3.16 / 2.22 / 2.93 | 1 failed, 108 passed |
| 7 | 3.00 / 2.27 / 2.92 | 1 failed, 108 passed |
| 8 | 4.16 / 2.67 / 3.03 | 1 failed, 108 passed |
| 9 | 3.20 / 2.57 / 2.98 | 1 failed, 108 passed |
| 10 | 2.76 / 2.54 / 2.95 | 1 failed, 108 passed |
| 11 | 2.08 / 2.40 / 2.90 | 1 failed, 108 passed |
| 12 | 2.25 / 2.40 / 2.88 | 1 failed, 108 passed |
| 13 | 1.90 / 2.31 / 2.83 | 1 failed, 108 passed |
| 14 | 1.56 / 2.18 / 2.77 | 1 failed, 108 passed |
| 15 | 1.74 / 2.16 / 2.74 | 1 failed, 108 passed |
| 16 | 2.19 / 2.21 / 2.74 | 1 failed, 108 passed |
| 17 | 1.70 / 2.09 / 2.68 | 1 failed, 108 passed |
| 18 | 1.30 / 1.96 / 2.62 | 1 failed, 108 passed |
| 19 | 1.46 / 1.95 / 2.59 | 1 failed, 108 passed |
| 20 | 1.38 / 1.89 / 2.55 | 1 failed, 108 passed |
| 21 | 1.83 / 2.00 / 2.56 | 1 failed, 108 passed |
| 22 | 1.96 / 1.99 / 2.54 | 1 failed, 108 passed |
| 23 | 1.44 / 1.86 / 2.48 | 1 failed, 108 passed |
| 24 | 1.11 / 1.75 / 2.42 | 1 failed, 108 passed |
| 25 | 1.19 / 1.72 / 2.39 | 1 failed, 108 passed |
| 26 | 2.84 / 2.03 / 2.46 | 1 failed, 108 passed |
| 27 | 3.83 / 2.44 / 2.59 | 1 failed, 108 passed |
| 28 | 3.41 / 2.48 / 2.60 | 1 failed, 108 passed |
| 29 | 3.80 / 2.71 / 2.67 | 1 failed, 108 passed |
| 30 | 3.19 / 2.71 / 2.68 | 1 failed, 108 passed |
| 31 | 2.45 / 2.60 / 2.64 | 1 failed, 108 passed |

## Process incidents (disclosed)
1. **Commit-message `#` stripping.** `git cherry-pick --continue` (and any editor-less commit with the default cleanup) strips lines starting with `#`; the #904/#905/#897/#918 subjects start with `#NNN`, so picks 5-7 came out with the wrong subject on the first pass. Caught by comparing every message against its original BEFORE anything else was built on it; I `git reset --hard` my OWN branch to pick 4 (`0a678040`; the discarded tip `e8769f65` is in the reflog) and redid picks 5-9; the first redo used `-c commit.cleanup=whitespace`, which did not take on `--continue`, so the second redo finished each conflicted pick with `git commit --cleanup=verbatim -F <original message + -x trailer>`. Final: 14/14 messages identical to the originals.
2. **Runner defects (mine), no code impact.** Attempt 1: the runner dropped stderr, so R1 (exit 1) showed an empty summary. Attempt 2 (stderr fixed): `No module named pytest`, because my scratch `HOME` hid `~/.local` (START loads 1.33/2.01/3.80 and 1.00/1.83/3.64; both exit 1, neither a refusal, neither a test result). Fixed with `PYTHONUSERBASE`. My first batch was stopped by hand after R1 (a `pkill -f` pattern also matched my own shell and killed it; I then confirmed no process of mine remained). The counted R1 (23 passed) is the third R1 invocation; the two failed ones are logged in `/tmp/z12-s4/runs-attempt1-harness-defect.jsonl` and `runs-attempt2-no-userbase.jsonl` (scratch, not committed).

## What ran / what did not
- **Ran:** 10 slice files (R1-R10) and 32 seeds over 2 files = 42 counted runs, all gated, all spaced (plus the 2 failed-harness R1 invocations above).
- **Did NOT run (and why):** `test_cc_deposit_step.py` and the NG full suite (forbidden, #944); the daemon-dependent tests (no daemon in this trial); the 2 `#897` tests that need `Z12_MERGE_BASE_REF` (unset: not run); any NG test file not named by a slice (`test_cc_callosum_leg1.py`, `test_cc_dual_pass.py`, `test_cc_refeed.py`, `test_cc_capture_mutations_423.py`, `test_cc_want_bounds.py`, ... — the originals ran some of these; the brief asked only for each slice's own files).

## Not verified
- Nothing here exercised a live graph, tract, socket, checkpoint, daemon or the VPS; every test is synthetic/in-process.
- The seed sweep covers the N2 and N6 test files only; the other eight files ran once, `PYTHONHASHSEED` unset.
- The runs used a scratch `HOME` and a read-only `PYTHONUSERBASE`; I do not know which `HOME` the original returns used for every file, so a HOME-sensitive difference cannot be ruled out (none appeared: counts match where the original code is unchanged).
- F-A and F-B were diagnosed from the failing assertions and the test sources; I did not re-run the originals' fail-first or mutation checks on the integrated tip.
- The "no new behaviour" claim for the two resolutions is by construction (a union of non-overlapping edits) plus the slice tests; the interplay case in N2 (`hold_on_failure=True` WITH `batch_nodes`/`receipt` set) is exercised by no test in either original, and none was added.
- The pair LOOK (a separate detached reviewer worktree on the committed hash, limited to the two hand-resolved hunks) has NOT happened; it follows.

## Appendix A — `git range-diff c625623e^! 0a678040^!` (N2 vs #794)
```
1:  c625623 ! 1:  0a67804 drain_ingest_tract: opt-in hold_on_failure=False (#794; Chief-003 ruling / Exec P386)
    @@ Commit message
         Lane ingest-tract-swallow-781. Nothing wired, nothing merged.
     
         Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
    +    (cherry picked from commit c625623ebcbdde894a8f9e36d5013aa76fc3a702)
     
      ## cc_ng_organism.py ##
     @@
      # the callosum, wholeness ring, hyperedge binding and orphan collection (2026-07-31).
      # The wholeness ring ALREADY EXISTS here (Leg 2). Open defect: merge-journal poison-pill.
      # ---- Changelog ----
    ++# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane ng-trial-chain-s4) -- TRIAL-branch hand-resolution of cherry-pick c625623 (#794)
    ++#   onto D24: drain_ingest_tract keeps D24's batch_nodes/receipt AND #794's hold_on_failure (LAST); the loop keeps both rules
    ++#   (the hold break, then D24's size check, then the entries cap); both changelog entries below kept; no new behaviour.
    + # [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane drain-pacing-d24, dispatches #12669/#12731/#12929) — D24 (re-scoped):
    + #   pace the ingest-tract drain by NODES; make _cc_callosum_consolidate's failure loud
    + #   [D24 FOLD, #12929, Exec Packet 482 / #896: NO code change in this file's behaviour -- the KNOWN LIMIT text below is
    +@@
    + #   batch, run by the daemon -- never inside drain_ingest_tract, which still never steps); the newer ruling wins and reconciling the
    + #   two texts is #895. drain_gateway_conduit's inert batch_size/idle_steps arguments coexist with this live path for the other
    + #   drain (LAW 3 shrapnel, #895).
     +# [2026-09-30] Z12 worker (Claude Sonnet 5.5), lane ingest-tract-swallow-781 — #794:
     +#   opt-in hold_on_failure on drain_ingest_tract (Chief-003 ruling / Exec P386)
     +# What: drain_ingest_tract gains ONE new LAST keyword, hold_on_failure=False. With
    @@ cc_ng_organism.py
      # [2026-09-26] Z2 worker (openrouter/deepseek/deepseek-v4.1-flash, OpenCode/T3 Code),
      #   lane z2-ng-recall-passthrough-restore-001 — restore the un-Pithed recall
      #   fallback in cc_assemble_recall (LAW 3, pre-46f9cf8 behavior)
    -@@ cc_ng_organism.py: def _apply_gateway_experience(graph, vector_db, state, entry):
    - 
    +@@ cc_ng_organism.py: def _cc_drain_receipt_write(receipt, **fields) -> None:
      
      def drain_ingest_tract(graph, vector_db, state: dict, tract_path: str = None,
    --                        return_consumed: bool = False, max_entries: int = 0):
    -+                        return_consumed: bool = False, max_entries: int = 0,
    +                         return_consumed: bool = False, max_entries: int = 0,
    +-                        batch_nodes: int = None, receipt: dict = None):
    ++                        batch_nodes: int = None, receipt: dict = None,
     +                        hold_on_failure: bool = False):
          """Drain miniTID's turn-deposit tract file, running each raw experience
          entry through the conversational dual-pass (Task 1). Feeder (miniTID)
    @@ cc_ng_organism.py: def drain_ingest_tract(graph, vector_db, state: dict, tract_p
     +    # flag is off, so the default path is unchanged.
     +    safe_offset = 0
          try:
    -         reader = ng_tract.TractReader(data)
    -         for entry in reader:
    +         if size_cap or receipt is not None:
    +             ids_before = set(graph.nodes)
     @@ cc_ng_organism.py: def drain_ingest_tract(graph, vector_db, state: dict, tract_path: str = None,
                  consumed_offset = reader.position()
                  # Check entry type using ng_tract.ENTRY_EXPERIENCE (the real module constant)
    @@ cc_ng_organism.py: def drain_ingest_tract(graph, vector_db, state: dict, tract_p
     +                    "and everything after it kept in the tract for the next cycle",
     +                    hold_reason, hold_exc_type)
     +                break
    +             # D24: the node-count rule is checked AFTER the atomic dual pass returned, so the turn that crosses
    +             # the size is absorbed whole and only then does the batch end. Unset (0) -> never true.
    +             if size_cap and (len(graph.nodes) - len(ids_before)) >= size_cap:
    +@@ cc_ng_organism.py: def drain_ingest_tract(graph, vector_db, state: dict, tract_path: str = None,
                  if max_entries and taken >= max_entries:
    +                 ended = "entries_cap_reached"
                      break
     +        if hold_on_failure:
     +            # Truncate only what precedes the first failed entry (or the whole
```

## Appendix B — `git range-diff ee94f7d2^! aa15336e^!` (N6 vs #905)
```
1:  ee94f7d ! 1:  aa15336 #905: born-bound emergent want, sweep-eligible guard predicate, redacted id sample, per-slice guard
    @@ Commit message
     
         Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
     
    +    (cherry picked from commit ee94f7d2c516cce40ff81bd32e4febb9781e8e04)
    +
      ## cc_ng_organism.py ##
     @@
      # the callosum, wholeness ring, hyperedge binding and orphan collection (2026-07-31).
      # The wholeness ring ALREADY EXISTS here (Leg 2). Open defect: merge-journal poison-pill.
      # ---- Changelog ----
    ++# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane ng-trial-chain-s4) -- TRIAL-branch hand-resolution of cherry-pick ee94f7d (#905)
    ++#   onto D24+#794..#897: _cc_callosum_consolidate's except block keeps D24's loud logger.error (it replaced the logger.debug line) AND
    ++#   #905's _fill(done, failed=True) before `return False`; the guard/progress signature and loop merged without conflict; no new behaviour.
    + # [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane ng-trial-chain-s4) -- TRIAL-branch hand-resolution of cherry-pick c625623 (#794)
    + #   onto D24: drain_ingest_tract keeps D24's batch_nodes/receipt AND #794's hold_on_failure (LAST); the loop keeps both rules
    + #   (the hold break, then D24's size check, then the entries cap); both changelog entries below kept; no new behaviour.
    +@@
    + #   ROLLOUT: this must merge BEFORE the daemon wiring slice -- a daemon passing the
    + #   new kwargs to an older organism raises TypeError. Tests:
    + #   tests/test_cc_recall_reporting.py.
     +# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane emergent-want-bound-905, dispatch #13138) — #905 parts A and D
     +# What: (A) generate_emergent_want writes the new want BORN BOUND: in the SAME _step_lock block as create_node it
     +#   writes one synapse seed -> want (weight 0.3, the surface_wants pattern) per existing, de-duplicated seed. If NO
    @@ cc_ng_organism.py: def _cc_callosum_consolidate(graph, idle_steps: int) -> bool:
          # Take the lock in SLICES, not for the whole 250 steps. cc_ng_host.py's
          # changelog records real hook timeouts caused by _recall() blocking on a
     @@ cc_ng_organism.py: def _cc_callosum_consolidate(graph, idle_steps: int) -> bool:
    -     # a waiting recall/deposit interleave. Homeostasis does not care whether
    -     # the steps were contiguous; the hooks care a great deal.
    -     slice_n = max(1, int(os.environ.get("CC_CALLOSUM_LOCK_SLICE_STEPS", "25")))
    -+    done = 0
          try:
              lock = getattr(graph, "_concurrent_lock", None)
    --        done = 0
              while done < idle_steps:
     +            if guard is not None:
     +                # Outside every lock this function takes: the guard needs
    @@ cc_ng_organism.py: def _cc_callosum_consolidate(graph, idle_steps: int) -> bool:
     +        _fill(done)
              return True
          except Exception as exc:
    -         logger.debug("CC callosum Leg1 consolidation failed (non-fatal): %s", exc)
    +         # D24 / Exec P476(e) (P370: fixed where the drain's phase 2 reuses it): this used to be logger.debug --
    +@@ cc_ng_organism.py: def _cc_callosum_consolidate(graph, idle_steps: int) -> bool:
    +         logger.error("CC callosum consolidation FAILED after %d of %d step(s) (%s); "
    +                      "this is the CAUSE record -- the caller logs what it did about it",
    +                      done, idle_steps, type(exc).__name__)
     +        _fill(done, failed=True)
              return False
```
