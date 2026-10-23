# Assignment worker-002 (CORRECTION) — lane `commons-test-isolation-738` (rows #738/#742/#754; tracked row #999)

- **Zone manager:** Z11 (CC session `c05315b5-4699-4a93-890a-eb9ec3af7952`) · **Reports to:** chief-003 (GO given, including B-5). Same worktree and branch as worker-001: `/home/josh/NeuroGraph-worktrees/z11-commons-testiso-20260930`, `cc-laptop-commons-testiso-20260930`, current head `207fc1f6cdcb7bad552ae22d1cd8fc56f99d1b9e`.
- **Authority:** `worktree_write` only. **NEW commits on top of `207fc1f` only — no amend, no rebase, no force-push, no rewrite.** Plain push. No PR, no merge (P329/P337 hold).
- **Return to:** `handoffs/z11-commons-testiso-20260930/returns/worker-002.md` (untracked, in this worktree), then stop. Do not self-review.
- **Allowed files:** `neurograph_rpc.py`, `tests/conftest.py`, `tests/test_commons_checkpoint_isolation_738.py`. Not `neuro_foundation.py`/`openclaw_hook.py`/`stream_parser.py`/`activation_persistence.py` (PROTECTED), no vendored file, not `commons.py`, not `data/`. (`tests/test_coordinator.py` and `tests/test_snn.py` stay as they are.)

## Why this correction exists

Two independent reviews (cross-family Grok-4.6 and the neurograph-law-enforcer) both returned ACCEPT-WITH-CORRECTIONS on your branch; the zone manager checked each finding against the code. Facts you build on:

- **B-1 (HIGH):** `TestRepoWideGuard::test_persist_to_real_path_raises_and_is_recorded_then_cleared` sets `commons._commons = None` (fixture), so `get_commons()` builds a FRESH EMPTY Commons, then calls `c.persist(REAL_COMMONS_PATH)`. Only the conftest patch stops that write. If the guard is absent or not the outermost wrapper (`--noconftest`; `commons` import failing at `pytest_configure`, where the guard prints and disables itself; another plugin's persist wrapper outermost — the zone manager's own safety plugin did exactly this and the test failed with `assert 0 == 1`), the test OVERWRITES Syl's real shared commons file with an empty Commons and fails only afterwards.
- **B-2:** `reset_blocked_persist_attempts()` (`tests/conftest.py`) `.clear()`s the whole record and the class fixture calls it on every teardown, so a swallowed real-path block recorded earlier in the session by another test is erased and `pytest_sessionfinish` never sees it.
- **B-3:** `pytest_sessionfinish` sets `session.exitstatus = 1` unconditionally, overwriting pytest's 2/3/4 (interrupted / internal error / usage error).
- **B-4:** `os.environ.get("NG_COMMONS_CHECKPOINT_PATH", default)` returns `""` when the variable is set but empty; `os.path.expanduser("")` is `""`; restore logs "starting fresh", `persist("")` fails, and the `logger.debug` swallow hides it — Commons is never saved again.
- **B-5 (Chief-approved):** `handle_after_turn`'s Commons persist (`neurograph_rpc.py` ~`:3811-3812` on this branch) and the clean-exit persist (~`:6002-6003`) swallow failures at `logger.debug`. The test guard's session-end banner only compensates inside tests; in production a real Commons persist failure is invisible at default log level.
- **B-6/B-7:** the `slow` marker text in `tests/conftest.py` says "not selected by default runs" — false (no pytest config in the repo deselects it; the timeout is what bounds the stall); and conftest comments cite the swallow at `~:3796-3800`, which is wrong.

## Required corrections

**B-1 (blocker).** In `TestRepoWideGuard`, before ANY call that persists to the real path, `pytest.skip(<reason>)` unless `getattr(commons_mod.Commons.persist, "_is_738_guard", False)` is true (i.e. THIS repo guard is the outermost wrapper). Fail closed: if you cannot establish that, skip. The remaining assertions (raise matches `#738`, recorded exactly once, real-file fingerprint unchanged) stay. Show BOTH behaviours in your return: with the plugin that makes the conftest guard outermost the test PASSES (`-p z11_noreal_commons_v5`); with the plugin that wraps outermost (`-p z11_noreal_commons`, v4) the test SKIPS with your reason and the real file is untouched. Also make sure the OTHER guard tests in the file cannot reach a real-path persist unguarded.

**B-2.** Replace the wholesale clear with truncation back to the record length captured at fixture setup (keep `reset_blocked_persist_attempts()` for any caller that needs it, but the class fixture must not erase records made before it started). Add a test: a record present before the class's tests start survives the class's teardown.

**B-3.** `pytest_sessionfinish` sets `session.exitstatus = 1` ONLY when the current status is 0 (do not mask 2/3/4; if it is already nonzero leave it, still print the banner). Test it without touching the real path (e.g. drive the hook function directly with a stub session object, or a subprocess pytest run in a tmp dir with a stand-in conftest — your call; no scratch file may be committed).

**B-4.** `_COMMONS_CHECKPOINT_PATH = os.path.expanduser(os.environ.get("NG_COMMONS_CHECKPOINT_PATH") or "<same default>")`. Test: set-but-empty yields the default; unset yields the default; a real value is honoured (subprocess `python -c` evaluating the module-level expression is fine, as the file's existing tests do — never import-and-persist against the real path).

**B-5.** `logger.debug` → `logger.warning` at BOTH Commons-persist swallows (`handle_after_turn` and the clean-exit path), message and `(non-fatal)` wording preserved, nothing else changed in those blocks. Add a source-level pin (AST or `inspect.getsource` on the two functions) that asserts both handlers log at WARNING or higher; show it fails if reverted. Update the changelog block above `_COMMONS_CHECKPOINT_PATH` / add one at each site per repo convention.

**B-6/B-7.** Fix the `slow` marker help text to say what is true (registered for selection with `-m "not slow"`; it is not deselected by default; the 300 s `timeout` marker is what bounds it where pytest-timeout is installed). Replace line-number citations of the swallow in conftest comments/docstring with the function names.

## HARD safety rules

1. The real file `~/NeuroGraph/data/checkpoints/commons.msgpack` (size 192773, mtime `2026-09-29 17:38:16.949928157 -0800`, sha256 `a4947c01f0e94d5eb125fc1a419d0982cb86a88421b838e2606c5ac87b15f319`) must be byte-identical before/after — show stat+sha256. **Every pytest command** is run as `env -u NG_EMBED_REMOTE PYTHONPATH=/tmp/z11_r5 python3 -m pytest ... -p z11_noreal_commons_v5 -p no:cacheprovider` (the v5 plugin file `/tmp/z11_r5/z11_noreal_commons_v5.py` installs its own persist backstop at configure time and prints `[z11 origin]` module-origin lines — **paste them in your return; a run without them does not count**; known hazard #770: `neurograph_rpc.py:735-738` puts the MAIN checkout at `sys.path[0]`). The single deliberate exception is the v4 demonstration in B-1 (`-p z11_noreal_commons`, same PYTHONPATH), which is also guarded.
2. Targeted files only: `tests/test_commons_checkpoint_isolation_738.py`, `tests/test_coordinator.py`, `tests/test_commons_leg2.py`, `tests/test_commons_persist_restore.py`, `tests/test_cc_commons_persist.py`. Never the full suite; never `tests/test_snn.py::TestLargeGraphStability`; never `tests/test_tonic_bridge.py`.
3. Print the `NG_EMBED_*` variable NAMES (never values) each command ran under. No secrets/tokens.
4. Write scope: the three allowed files, the branch, your return file. Nothing in `~/docs`, nothing in any auto-memory directory. Changelog header in every modified file; your own `Co-Authored-By` trailer. No scratch files committed.
5. No live checkpoint reads; no service starts/restarts.

## Acceptance

B-1…B-7 done and tested; B-1 both behaviours shown; baseline preserved (`tests/test_coordinator.py` 28 passed + the one pre-existing failure `TestInitCCHost::test_already_initialized_returns_true`, the isolation file all green or explicitly skipped-with-reason under v4); real commons byte-identical; NEW commit(s) pushed (plain `git push origin cc-laptop-commons-testiso-20260930`); `git status -sb && git log --oneline -5` shown; return lists files changed with line refs, every command with `NG_EMBED_*` names, the `[z11 origin]` lines, counts, and anything unverified or out of scope (list, do not act).
