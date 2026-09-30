<!--
# ---- Changelog ----
# [2026-09-30] Claude Sonnet 5.5 (Z12 builder, lane want-hub-engine-d-build-20260930, dispatch #11903) — build-002b
#   What: short return for the Z12-ruled one-line test fix (ADDENDUM 2). NEW file, docs only.
#   Why: build-002 §5.3 proved `test_A_orchestrator_…` computed its reference key after the call removed synapses; Z12 verified and ruled the move.
#   How: hashes from git rev-parse / git hash-object; results from the pytest output of this session. No secret involved.
# -------------------
-->

# build-002b — test A fixed as ruled; `tests/test_want_hub_competition.py`: **27 passed, 0 failed**

Not self-accepted. `neuro_foundation.py`, the engine branch, the driver and every other file: untouched.

## Commit (tests branch `cc-laptop-want-hub-build-20260930`)
- **`909b39f61c7612c081ff8807a4733973b0bb1464`** — ONE file (`git diff --name-only HEAD~1 HEAD` = `tests/test_want_hub_competition.py`), 2 insertions / 1 deletion; **pushed before the run** (`origin/…` == local, verified after `git fetch`). Parent = build-002 (`31b823d11ef3c25c6fabfcccae2dd6334251e741`).
- The diff — the ruled one-line MOVE plus one changelog line:
```
@@ -1,3 +1,4 @@
 # ---- Changelog ----
+# [2026-09-30] Claude Sonnet 5.5 (Z12 builder, dispatch #11903, Z12 ruling ADDENDUM 2) — test_A_orchestrator_…: compute the reference order key BEFORE the call (it read removed synapses); no assertion weakened.
@@ -252,4 +253,5 @@ def test_A_orchestrator_makes_exactly_one_call_and_passes_the_callers_sets(det_i
     g, _ = make()
     s = drv.ref_sets(g, K)
+    ok = drv.ref_order_key(g, s.competing)   # computed BEFORE the call: the call removes synapses, and the reference reads g.synapses[sid]
     calls = []
@@ -267,5 +269,4 @@ …
     assert k["max_removals"] == B_SMALL
-    ok = drv.ref_order_key(g, s.competing)
     assert {sid: tuple(k["order_key"][sid]) for sid in s.competing} == ok
```
No assertion weakened or removed; the comparison is the same `== ok`, now against a key computed from the pre-call state. (My first edit attempt dropped the spy/call lines by mistake; I caught it in the diff review before committing — the committed diff above is what landed.)

## Engine blob check (before the run)
Integration worktree `/home/josh/NeuroGraph-worktrees/z12-want-hub-integ-20260930`: `git hash-object neuro_foundation.py` = **`96d12f507746ce168fffe6feba56e097c8832353`** = the blob of engine commit `8e5785322f910aefaf781fa3525bea2831fedd31` (`git rev-parse 8e578532:neuro_foundation.py`). Test file checked out into it from `909b39f6` (blob `8c1f73567d0c2439be41f07a591f7deb97ffc1a7`, equal to the branch head's). Driver blob `1c144259079d2ea0acddcfb9f2f90bb197903bf5` = tests head's (unchanged, so the golden comparison in build-002 §5.2 was not re-run). Nothing committed or pushed from that worktree (`git status`: two staged files, `HEAD` detached at `95154e33`).

## Result — ONE run, `env -u NG_EMBED_REMOTE -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 python3 -m pytest tests/test_want_hub_competition.py -p no:cacheprovider -v` (load 2.23, MemAvailable ≈ 5.6 GiB before the run)
P379 preamble: `neuro_foundation.__file__=/home/josh/NeuroGraph-worktrees/z12-want-hub-integ-20260930/neuro_foundation.py git_rev=95154e33f299d70e4fdf5945ea1a189bcbd61acc base_rev=e4ebf982b1989fd9066d610b94853bc68bf70d37 new_api_present=True` (`git_rev` is the integration worktree's HEAD, the OLD tests head; the engine file is the staged blob above — same caveat as build-002 §4).

**`============ 27 passed in 18.31s ============`** — P379 + harness self-checks 4, **G** 7, **A** 2 (including `test_A_orchestrator_makes_exactly_one_call_and_passes_the_callers_sets`, now PASS), **K** 8, **R** 6. Nothing else failed; no other run was made.

## Not verified
Unchanged from build-002 §7 (no real graph / PG-1, no native-store performance numbers, full suite not run). This turn added nothing to that list and verified nothing beyond the 27 tests above.

STOP here: the fresh cross-family + law-enforcer delta pair and PG-1 are next and are not mine.
