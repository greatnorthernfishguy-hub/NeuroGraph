# Return — worker-001, ADDENDUM A (lane `commons-test-isolation-738`, rows #738/#742/#754)

Continuation of `handoffs/z11-commons-testiso-20260930/returns/worker-001.md` (now removed from
the tracked tree per this addendum's instruction — see commit `3e04195`). This file is
**UNTRACKED**, per instruction, and stays that way.

Assignment file re-read directly from disk before acting (`handoffs/z11-commons-testiso-20260930/assignments/worker-001.md`) — confirmed ADDENDUM A's text matches what Z11 quoted verbatim before I did anything with it.

## Files changed, with line refs

1. **`tests/conftest.py`** — new file, 148 lines.
   - `pytest_configure` (registers `slow`/`timeout` markers via `addinivalue_line`, then patches `commons.Commons.persist`).
   - `_guarded_persist` closure — the A1 guard itself: raises `RuntimeError` if `os.path.realpath(os.path.expanduser(filepath))` equals the literal `~/NeuroGraph/data/checkpoints/commons.msgpack` (computed once at import, never from `NG_COMMONS_CHECKPOINT_PATH` or `neurograph_rpc._COMMONS_CHECKPOINT_PATH`).
   - `_blocked_persist_attempts` / `reset_blocked_persist_attempts()` / `get_blocked_persist_attempts()` / `real_commons_path()` — the record and its test-facing hooks.
   - `_track_current_test_nodeid` autouse fixture — tags each blocked attempt with the test that caused it.
   - `pytest_sessionfinish` — sets `session.exitstatus = 1` and prints offenders if anything is still recorded at session end.
2. **`tests/test_snn.py`** — `TestLargeGraphStability.test_1k_nodes_10k_steps`, now at line ~205 (was line 193 pre-edit; a 12-line changelog block was inserted above the class). Added `@pytest.mark.slow` and `@pytest.mark.timeout(300)`. Test body and `neuro_foundation.py` untouched — confirmed via `git diff` showing only the two decorator lines + changelog comment added.
3. **`tests/test_commons_checkpoint_isolation_738.py`** — changelog header extended; new `TestRepoWideGuard` class (2 tests) added at end of file; new top-level import `from tests import conftest as _repo_conftest`.
4. **`handoffs/z11-commons-testiso-20260930/returns/worker-001.md`** — removed via `git rm` (commit `3e04195`), per instruction #3. No handoffs/ file committed since.

`git diff --cached --stat` on the addendum commit confirmed exactly the three allowed-by-addendum files (`tests/conftest.py` new, `tests/test_snn.py`, `tests/test_commons_checkpoint_isolation_738.py`) — nothing else touched.

## Commands run, with `NG_EMBED_*` names each ran under

Ambient environment has exactly one `NG_EMBED_*` variable: `NG_EMBED_REMOTE`. Every pytest command below used `env -u NG_EMBED_REMOTE`, so every command ran with **zero `NG_EMBED_*` variables set**. All used `-p no:cacheprovider`.

1. `env -u NG_EMBED_REMOTE python3 -m pytest tests/test_commons_checkpoint_isolation_738.py -p no:cacheprovider -q` (guard file standalone, first pass while writing the new tests) → 6 passed.
2. `env -u NG_EMBED_REMOTE python3 -m pytest tests/test_snn.py::TestLargeGraphStability::test_1k_nodes_10k_steps --collect-only -q -W error::pytest.PytestUnknownMarkWarning` → 1 collected, no warning raised (confirms marker registration works).
3. `env -u NG_EMBED_REMOTE python3 -m pytest tests/test_snn.py::TestLargeGraphStability::test_1k_nodes_10k_steps --timeout=300 --timeout-method=signal -p no:cacheprovider -q` → see #754 timing result below.
4. `NG_COMMONS_CHECKPOINT_PATH=<mktemp -d>/commons.msgpack env -u NG_EMBED_REMOTE python3 -m pytest tests/test_coordinator.py -p no:cacheprovider -q` (guard now in place, var exported) → 28 passed, 1 failed (`TestInitCCHost::test_already_initialized_returns_true`, pre-existing).
5. `env -u NG_EMBED_REMOTE -u NG_COMMONS_CHECKPOINT_PATH python3 -m pytest tests/test_coordinator.py -p no:cacheprovider -q` (guard in place, var unset) → same: 28 passed, 1 failed.
6. Full targeted combined re-run (still within Packet 373 throttle — targeted files only): `env -u NG_EMBED_REMOTE python3 -m pytest tests/test_coordinator.py tests/test_commons_persist_restore.py tests/test_commons_leg2.py tests/test_bootstrap_singleflight_430.py tests/test_cc_commons_persist.py tests/test_commons_bunyan_stage1.py tests/test_commons_experience_deposit.py tests/test_commons_outcome_deposit.py tests/test_commons_checkpoint_isolation_738.py -p no:cacheprovider -q` → **83 passed, 1 failed** (same pre-existing failure). +2 vs. the prior lane's 81/1 total, matching the 2 new `TestRepoWideGuard` tests.
7. Exit-code sanity check: `env -u NG_EMBED_REMOTE python3 -m pytest tests/test_commons_checkpoint_isolation_738.py -p no:cacheprovider -q > /tmp/guard-run.log 2>&1; echo $?` → `PYTEST_EXIT_CODE=0` (6 passed, nothing left blocked — proves the session-fail mechanism does NOT misfire on a clean run).
8. Positive-control of the session-fail mechanism itself (throwaway scratch file `tests/_scratch_sessionfail_proof.py`, written and deleted within this same turn, never committed — see "how I proved it fires" below).

None of these commands wrote to `~/docs` or any auto-memory directory.

## Test counts before/after

Unchanged from the first delivery's measurement (base `e4ebf98`: 28 passed / 1 pre-existing failure on `tests/test_coordinator.py` alone, via the disposable scratch-clone method described in that return before it was removed from the tree). After ADDENDUM A: `test_coordinator.py` alone still 28/1 (both with the var exported and with it unset, guard installed both times). Combined targeted set: 83 passed / 1 failed (up from 81/1 pre-addendum, the +2 being the new guard tests). No new failures anywhere.

## Real-file `stat` + sha256, before/after

Identical at every checkpoint across this entire turn — start, after writing/running the new guard tests, after both `test_coordinator.py` runs, after the combined targeted run, after the #754 slow-test run (5+ minutes of real SNN stepping, unrelated to commons but checked anyway), after the positive-control scratch test, and just before writing this report:

```
Size: 192773
Modify: 2026-09-29 17:38:16.949928157 -0800
sha256: a4947c01f0e94d5eb125fc1a419d0982cb86a88421b838e2606c5ac87b15f319
```

This matches the "current" baseline Z11 quoted in this continuation's instructions exactly. Never touched, never varied, across the whole turn.

## How the session-fail record works, and how I proved it fires

Mechanism: `commons.Commons.persist` is monkeypatched at `pytest_configure` time (before any test runs, so it applies to every test file in `tests/`, closing the exact gap #738 recurred through — a per-file fixture only protects its own file). The patched version resolves the target path and compares it to a **literal**, hardcoded default computed once at conftest import — not read from `neurograph_rpc._COMMONS_CHECKPOINT_PATH` or `NG_COMMONS_CHECKPOINT_PATH`, so this lane's own env-var override (built in the first delivery) cannot be used to defeat the guard. A match raises `RuntimeError` and appends `(current_test_nodeid, resolved_path)` to a module-level list. `pytest_sessionfinish` checks that list at the very end of the run and, if non-empty, prints every offender and sets `session.exitstatus = 1`.

The reason this second layer exists at all: `neurograph_rpc.py`'s `handle_after_turn` wraps the persist call in a bare `except Exception: logger.debug(...)` (originally at `:3796-3800` in the pre-addendum source; still the same shape after this lane's earlier env-var edit). A raise with no session-level backstop would be silently absorbed by that except clause and the test would report green.

I proved this is not just an assertion by reasoning about the code, but an *executed* effect: I wrote a throwaway file `tests/_scratch_sessionfail_proof.py` (never committed — written, run, and `rm`'d within this turn, confirmed gone via `git status -sb` immediately after) containing one test that calls `commons.get_commons().persist(<the real path>)` inside a `try/except Exception: pass` — i.e., it reproduces `handle_after_turn`'s own swallow pattern exactly. Running it alone:

```
$ env -u NG_EMBED_REMOTE python3 -m pytest tests/_scratch_sessionfail_proof.py -p no:cacheprovider -q
.                                                                        [100%]
==============================================================================
#738 REPO-WIDE GUARD: blocked Commons.persist() attempt(s) to the REAL commons checkpoint that were never cleared:
  - 'tests/_scratch_sessionfail_proof.py::test_swallows_the_raise_like_handle_after_turn_does' attempted persist to /home/josh/NeuroGraph/data/checkpoints/commons.msgpack
...
1 passed in 0.29s
PYTEST_EXIT_CODE=1
```

The individual test reports `1 passed` (exactly what a silently-swallowed raise would look like without the session-level check), but the process's own exit code is `1`, not `0` — proving `pytest_sessionfinish` is the thing actually catching it, empirically, not just by code inspection. Real file confirmed untouched immediately after (the guard's raise happens before any write — `Commons.persist` only calls `self._ng.save(filepath)`, i.e. the real write, inside the original unpatched function, which the guard never reaches for the protected path).

The two committed guard tests in `TestRepoWideGuard` (`tests/test_commons_checkpoint_isolation_738.py`) cover the same ground safely within the real session: one proves persist-to-tmp still works unaffected, the other proves persist-to-real raises + is recorded, then explicitly clears the record (`reset_blocked_persist_attempts()`) before the test ends, so exercising the guard doesn't fail the real, committed test session. Both pass; the combined-run exit code was confirmed `0` separately (command 7 above).

## #754 timing result

Run alone, exactly as instructed (`--timeout=300 --timeout-method=signal`, never inside a suite): **did not finish within 300s.** pytest-timeout interrupted it cleanly at 301.27s wall time (`1 failed in 301.27s (0:05:01)`), with a full traceback showing it was mid-`g.step()`, specifically inside `neuro_foundation.py`'s `_cleanup_predictions` → `_on_prediction_error` → `_surprise_exploration` → `_find_synapse`. I did not touch `neuro_foundation.py` (protected) or the test body — this is a finding about the engine for a separate lane, exactly as the addendum anticipated. Before this marker existed the same test had reportedly run >1.5h without completing or being killed; now a full-suite run would stop it at 300s instead of stalling indefinitely.

## Commit / push

```
$ git status -sb && git log --oneline -4
## cc-laptop-commons-testiso-20260930...origin/cc-laptop-commons-testiso-20260930
?? handoffs/
207fc1f tests: repo-wide commons persist guard + #754 slow-test marker (ADDENDUM A)
3e04195 handoffs: remove tracked return file (worker-001, #738/#742)
4e7a26d return: record commit SHA + push confirmation (worker-001, #738/#742)
7950c03 commons: env-drive checkpoint path (LAW 5) + test isolation (#738/#742)
```

Two new commits this turn: `3e04195` (`git rm` of the tracked return file, plain removal, no history rewrite) and `207fc1f` (the ADDENDUM A code). Both pushed with a normal `git push origin cc-laptop-commons-testiso-20260930` (no force). No PR opened, nothing merged. The remaining `?? handoffs/` entry is `handoffs/z11-commons-testiso-20260930/assignments/` (the assignment file itself, present before I started, not mine to commit) plus this untracked report — nothing else.

## What I did not verify

- I did not attempt to determine whether the `commons` module is importable in every environment this repo runs in — only confirmed it imports fine here. The guard's fallback path (print to stderr, guard disabled, `_GUARD_STATUS["reason"]` set) was written but never actually exercised, since `commons` imported successfully every time in this environment. Flagging per the addendum's instruction to say so rather than claim untested coverage.
- I did not re-verify the original lane's negative-control test or the base-revision count measurement from the first delivery — those were not touched this turn and I have no reason to believe they changed, but I did not re-run them.
- I did not investigate the root cause of the #754 stall inside `neuro_foundation.py` beyond what the traceback shows — out of scope, protected file, flagged as a separate-lane finding per instruction.
- I did not determine whether any other org seat's pytest run touched the real file again during this turn (no baseline drift observed at any of my checkpoints, but I only sampled at the points listed above, not continuously).
