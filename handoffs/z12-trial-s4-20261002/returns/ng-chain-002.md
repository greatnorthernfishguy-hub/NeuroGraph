<!--
# ---- Changelog ----
# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane ng-trial-chain-s4, dispatch #14658) — return ng-chain-002 for build NG-CHAIN N8 (test-only: the two stale cross-change pins F-A, F-B; Chief-003 ruling YES)
#   What: ONE test-only commit N8 on the NG trial branch (bc695eb0 -> N8 -> this docs-only return); F-A updated, F-B RE-BASED (not retired); each pin proven to still bite with a scratch mutant.
#   Why: ng-chain-001 surfaced two stale pins (not code defects); the trial branch must be test-green except the known #902 so the preflight can tell new breakage from stale pins.
#   How: tests only, two files, no production/protected/vendored file; gated, spaced runs; scratch mutants live only under /tmp.
# -------------------
-->

# NG TRIAL chain — return ng-chain-002 (N8)

**Branch** `cc-laptop-trial-s4-20261002` (worktree `/home/josh/NeuroGraph-worktrees/trial-s4-20261002`); history is now `9b647e39` (code tip, 14 picks) -> `bc695eb0` (return 001) -> **`2d4e4467`** (N8, test-only) -> this docs-only return. No upstream, nothing pushed, nothing merged; the NG merge-hold stands. I did not touch any `z12-review-ngchain-*` worktree.

## Headline
1. **N8 = `2d4e4467`, subject `N8: trial integration pin updates (F-A, F-B)`, touching exactly two files, tests only:** `tests/test_cc_drain_hold_on_failure.py` and `tests/test_cc_bind_atomic_904.py`. `git diff --name-only 9b647e39..N8` lists only those two (plus, once this return lands, `handoffs/.../ng-chain-001.md` and `ng-chain-002.md`): **no production file, no protected/vendored file.**
2. **F-A: updated.** **F-B: RE-BASED** (the preferred option), not retired.
3. After N8: `test_cc_drain_hold_on_failure.py` **49 passed** and `test_cc_bind_atomic_904.py` **56 passed + 5 skipped** (the re-based count; the 5 skips are the daemon-loop tests, as before); both together at `PYTHONHASHSEED=0`: 105 passed + 5 skipped (= 49 + 56). The trial branch is therefore green on these two files; the one known #902 failure (`test_poincare_dir_is_rederived_locally_not_transmitted` in `test_cc_topology_callosum.py`) is untouched and still the only known red.
4. **All four mutants fail the intended assertion; both controls pass** (table below). No run was refused (max START 1-min 3.46, 5-min 2.47).

## How I read N8
- **F-A.** The old pin asserted the exact pre-D24 signature list. What it protects is (i) every parameter the #794 era had is still there, in order, (ii) `hold_on_failure` is LAST, (iii) its default is `False`. The new test asserts the prefix `["graph","vector_db","state","tract_path","return_consumed","max_entries","batch_nodes","receipt"]` explicitly (`params[:-1]`), then `params[-1] == "hold_on_failure"`, then `default is False`: the same three protections, true on a tree that also carries D24. (It is deliberately a TRIAL-tree test: on a #794-only tree it would fail, which is the point of N8 being only on the trial branch.)
- **F-B.** The old pin compared `generate_emergent_want` to the PRE-#905 base `c5684334`, impossible once #905 part A is in the tree. I first checked that the integrated function is byte-identical to #905's own (`ee94f7d2`, `f685a3a7` round 2 and the trial pick `2f4b928` all give the identical 10421-char body), so a re-base is meaningful: it now compares against `ee94f7d2:cc_ng_organism.py` via the SAME `git show` mechanism, so what #904 owed (it adds nothing to that function beyond what #905 made) is still pinned. Two small, flagged changes beyond the constant: the 'new' side is now the MODULE UNDER TEST (`Path(org.__file__)`, which is exactly the worktree file in a normal run; the preamble already forces that) so the file's existing `Z12_904_ORG_UNDER_TEST` hook can feed a scratch mutant; and a failed `git show` is now a loud, explained assert instead of an opaque `ValueError`. Retiring was not needed.
- **Does either pin touch set/dict order? No** (a signature list and a string comparison), so no 0..31 sweep: ONE run at seed 0 (P3), as the brief says.
- Changelog headers added to both test files (naming F-A / F-B, the choice, and the dispatch).

## The diff of the two tests (`git show -U2 N8`)
```diff
diff --git a/tests/test_cc_bind_atomic_904.py b/tests/test_cc_bind_atomic_904.py
index dbd1e83..36f7e81 100644
--- a/tests/test_cc_bind_atomic_904.py
+++ b/tests/test_cc_bind_atomic_904.py
@@ -1,3 +1,8 @@
 # ---- Changelog ----
+# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane ng-trial-chain-s4, dispatch #14658) — N8 (TRIAL branch, test-only): F-B, the stale cross-change pin.
+#   test_generate_emergent_want_is_untouched compared generate_emergent_want to the PRE-#905 base (c5684334), which cannot hold once #905 part A is in the
+#   same tree (either merge order). RE-BASED (not retired) to compare against #905's own version (ee94f7d2) through the same `git show` mechanism, so it
+#   still pins that #904 itself adds nothing to that function; the 'new' side is now the module under test (org.__file__), which is the worktree file in
+#   a normal run and lets the existing Z12_904_ORG_UNDER_TEST hook feed a scratch mutant. A failed `git show` is now a loud, explained assert. No other test changed.
 # [2026-10-01] Z12 builder (Claude Sonnet 5.5), lane dual-pass-atomic-904, dispatch #13437 — #904 ROUND 2: le-053 F1-1 (the verdict sentence says what was
 #   KEPT), F1-2 (a probation sweep between the snapshot and the rollback is NOT erased; a key added by the call is deleted), F1-3 (the restore's
@@ -1190,7 +1195,13 @@ def test_want_success_path_is_golden_identical_and_quiet(fn_name, prefix, caplog
 
 def test_generate_emergent_want_is_untouched():
-    """#905 owns it: its source must be byte-identical to the base."""
+    """#905 owns it: #904 must not alter generate_emergent_want. Its source must be byte-identical to #905's.
+
+    N8 (NG trial integration): this used to compare against the PRE-#905 base (c5684334), which cannot hold once #905
+    part A (the born-bound want) is in the same tree, in either merge order. RE-BASED, not retired: the function is now
+    compared against #905's own version (ee94f7d2, read via the same `git show` mechanism), so what #904 owed is still
+    pinned -- #904 itself adds nothing to this function beyond what #905 made. The 'new' side is the MODULE UNDER TEST
+    (org.__file__; the worktree file unless Z12_904_ORG_UNDER_TEST points at a scratch copy)."""
     import subprocess
-    new = (_WORKTREE / 'cc_ng_organism.py').read_text()
+    new = Path(org.__file__).read_text()
 
     def body(src):
@@ -1198,5 +1209,6 @@ def test_generate_emergent_want_is_untouched():
         end = src.index('\ndef ', start + 10)
         return src[start:end]
-    base = subprocess.run(['git', '-C', str(_WORKTREE), 'show', 'c568433434a99154b9f7652b77cbe43752d34324:cc_ng_organism.py'],
-                          capture_output=True, text=True).stdout
-    assert body(new) == body(base)
+    ref = subprocess.run(['git', '-C', str(_WORKTREE), 'show', 'ee94f7d2c516cce40ff81bd32e4febb9781e8e04:cc_ng_organism.py'],
+                         capture_output=True, text=True)
+    assert ref.returncode == 0 and ref.stdout, 'cannot read #905 (ee94f7d2) cc_ng_organism.py: %s' % ref.stderr.strip()
+    assert body(new) == body(ref.stdout)
diff --git a/tests/test_cc_drain_hold_on_failure.py b/tests/test_cc_drain_hold_on_failure.py
index fae50e1..9ba118a 100644
--- a/tests/test_cc_drain_hold_on_failure.py
+++ b/tests/test_cc_drain_hold_on_failure.py
@@ -1,3 +1,7 @@
 # ---- Changelog ----
+# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane ng-trial-chain-s4, dispatch #14658) — N8 (TRIAL branch, test-only): F-A, the stale
+#   cross-change pin. test_signature_hold_on_failure_is_last_and_defaults_false pinned the exact pre-D24 signature; with D24 in the same tree
+#   the signature is [..., max_entries, batch_nodes, receipt, hold_on_failure]. UPDATED to assert the prefix list (now with batch_nodes,
+#   receipt), `params[-1] == "hold_on_failure"` and `default is False`: same protection, true on the integrated tree. No other test changed.
 # [2026-09-30] Z12 worker (Claude Sonnet 5.5), lane ingest-tract-swallow-781 — #794
 # What: tests for drain_ingest_tract(hold_on_failure=...) (Chief-003 ruling / Exec P386):
@@ -212,10 +216,18 @@ def _split(result, rc):
 # ------------------------------------------------- signature / call-shape guards
 def test_signature_hold_on_failure_is_last_and_defaults_false():
-    """FAILS on base: no such parameter (KeyError)."""
-    params = list(inspect.signature(cc_ng_organism.drain_ingest_tract).parameters)
-    assert params == ["graph", "vector_db", "state", "tract_path", "return_consumed",
-                      "max_entries", "hold_on_failure"]
-    assert inspect.signature(cc_ng_organism.drain_ingest_tract).parameters[
-        "hold_on_failure"].default is False
+    """FAILS on base: no such parameter (KeyError).
+
+    N8 (NG trial integration): on a tree that also carries D24 the two D24 keyword
+    parameters `batch_nodes, receipt` sit between `max_entries` and `hold_on_failure`.
+    What this pins is unchanged: every parameter the old pin listed is still there, in
+    order, and `hold_on_failure` is the LAST one and defaults to False. The prefix is
+    therefore asserted explicitly (a dropped or reordered parameter fails), then the
+    last-ness, then the default."""
+    sig = inspect.signature(cc_ng_organism.drain_ingest_tract)
+    params = list(sig.parameters)
+    assert params[:-1] == ["graph", "vector_db", "state", "tract_path", "return_consumed",
+                           "max_entries", "batch_nodes", "receipt"]
+    assert params[-1] == "hold_on_failure"
+    assert sig.parameters["hold_on_failure"].default is False
```

## Runs (gated, 25 s spacing before each, named files only)
**Gate:** `/proc/loadavg` read once per run after the spacing; refuse at 1-min >= 6.0 OR 5-min >= 5.0; no retry. **Forbidden and not run:** `tests/test_cc_deposit_step.py`, the NG full suite. **Env:** every run through `/usr/bin/env -u CC_NG_BATCH_SIZE -u CC_NG_IDLE_STEPS -u CC_NG_DRAIN_HOLD_ON_FAILURE -u NG_EMBED_REMOTE -u CC_NG_IN_TRANSIT_IDS_PATH python3 -m pytest -q -p no:cacheprovider -rs ...`; **EFFECTIVE values PRINTED AND SEEN** in every record: the five all `'<unset>'`, `PYTHONPATH='<unset>'`, `HOME='/tmp/z12-s4/home'` (scratch), `PYTHONUSERBASE='/home/josh/.local'`, `PYTHONHASHSEED='<unset>'` except P3 (`0`); the B-side runs add only `Z12_904_ORG_UNDER_TEST=<scratch path under /tmp/z12-904-scratch>`. The slice runs P1-P3 ran against the COMMITTED N8 tree (a commit is not a run).

| run | scope | START load 1m / 5m / 15m | result |
|---|---|---|---|
| P1 F-A file, committed N8 tip | `test_cc_drain_hold_on_failure.py` | 2.31 / 2.01 / 2.36 | 49 passed |
| P2 F-B file, committed N8 tip | `test_cc_bind_atomic_904.py` | 2.00 / 1.98 / 2.33 | 56 passed, 5 skipped |
| P3 both files, PYTHONHASHSEED=0 | `test_cc_drain_hold_on_failure.py test_cc_bind_atomic_904.py` | 1.73 / 1.91 / 2.30 | 105 passed, 5 skipped |

## Mutant table (each mutation applied to a SCRATCH copy of the committed code; one gated run each)
F-A mutants run `-k test_signature_hold_on_failure_is_last_and_defaults_false` from a scratch repo root under `/tmp/z12-s4/mut-*/` holding the committed test file and a mutated copy of the committed `cc_ng_organism.py` (the module is stdlib-only at import, so the scratch copy is self-contained). F-B runs `-k test_generate_emergent_want_is_untouched` with `Z12_904_ORG_UNDER_TEST` pointing at a mutated copy under `/tmp/z12-904-scratch/` (that file's own allowed scratch root). Each scratch mutant differs from the committed `cc_ng_organism.py` by exactly the stated edit (diffs shown at build time).

| run | mutation of the COMMITTED code (scratch copy only) | START load 1m / 5m / 15m | result | where it failed |
|---|---|---|---|---|
| C-A0 control: unmutated scratch copy, F-A test | none (unmutated scratch copy through the same harness) | 2.70 / 2.17 / 2.37 | 1 passed, 48 deselected | n/a (control: passes) |
| M-A1 | A: `hold_on_failure` no longer last (swapped with `receipt`) | 2.38 / 2.14 / 2.36 | **1 failed, 48 deselected** | prefix assertion, `At index 7 diff: 'hold_on_failure' != 'receipt'` |
| M-A2 | A: `hold_on_failure` default becomes `True` | 2.35 / 2.17 / 2.36 | **1 failed, 48 deselected** | `assert True is False` on `.default` |
| M-A3 | A: `max_entries` (a parameter the old pin listed) removed | 1.72 / 2.04 / 2.31 | **1 failed, 48 deselected** | prefix assertion, `At index 5 diff: 'batch_nodes' != 'max_entries'` |
| C-B0 control: unmutated scratch org, F-B test | none (unmutated scratch copy through the same harness) | 1.75 / 2.01 / 2.29 | 1 passed, 60 deselected | n/a (control: passes) |
| M-B1 | B: `generate_emergent_want` edited in a scratch copy (`weight=0.3` -> `weight=0.31`) | 3.46 / 2.47 / 2.44 | **1 failed, 60 deselected** | `body(new) == body(#905)`, diff shows `- weight=0.3) / + weight=0.31)` |

Controls (`C-A0`, `C-B0`) run the UNMUTATED copy through the identical scratch harness and pass, so a mutant's failure is the mutation, not the harness. **Note on M-A1:** the "not last" mutation is caught by the explicit prefix assertion first (the parameter at index 7 is no longer `receipt`); the separate `params[-1] == "hold_on_failure"` assertion is a second, independent guard that would only fire if the prefix were still right but the last parameter were something else.

## Process notes (disclosed)
- I did not use `pkill`/`killall` at all this round (Chief's rule); no process of mine was started other than the gated runs, each of which ended on its own.
- Scratch files created, all outside the repo: `/tmp/z12-s4/mut-A0-control/`, `mut-A1-not-last/`, `mut-A2-default-true/`, `mut-A3-drop-max_entries/` (a copy of `cc_ng_organism.py` + the test each) and `/tmp/z12-904-scratch/z12-ngchain-n8-mut-B1.py`, `z12-ngchain-n8-ctl-B0.py`. I did not delete anything; `/tmp/z12-904-scratch/` also holds earlier lanes' files, which I did not touch.

## What ran / did not run
- **Ran:** P1, P2, P3 (the two files, once each, plus one seed-0 run of both) and 6 scratch runs (2 controls + 4 mutants) = 9 gated runs.
- **Did NOT run:** `test_cc_deposit_step.py`, the NG full suite, any other NG test file; the other eight slice files were not re-run after N8 (N8 touches no code and no other test, and the 14-pick runs in ng-chain-001 stand for them).

## Not verified
- Only the two N8 files were re-run on the N8 tip. I did not re-run `test_cc_topology_callosum.py` to re-confirm the #902 failure is still the only red; it is unchanged by construction (N8 touches neither it nor any production file) but not re-observed this round.
- The re-based F-B pin proves "#904 adds nothing to `generate_emergent_want` beyond #905's version" only at the level of the function's source text (byte equality of its body); it is silent about behavioural interplay, which the original pin was equally silent about.
- The mutants are four hand-picked mutations; this is not an exhaustive mutation analysis of either pin.
- Nothing ran against any live graph, tract, socket, checkpoint, daemon or the VPS.
