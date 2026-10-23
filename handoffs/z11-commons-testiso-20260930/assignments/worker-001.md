# Assignment worker-001 — lane `commons-test-isolation-738` (rows #738 / #742)

- **Zone manager:** Z11 (CC session `c05315b5-4699-4a93-890a-eb9ec3af7952`) · **Reports to:** chief-003 · **Ruling:** Executive Packets 370 / 371 (2026-09-30)
- **Worktree / branch:** `/home/josh/NeuroGraph-worktrees/z11-commons-testiso-20260930` · `cc-laptop-commons-testiso-20260930`, cut from NeuroGraph `origin/main` `e4ebf98`
- **Authority:** `worktree_write` only. Commit and push the branch. **Do NOT open a PR. Do NOT merge.** Merges ride the standing P329/P337 rollout hold; nothing here wires into anything live.
- **Return to:** `handoffs/z11-commons-testiso-20260930/returns/worker-001.md` (in this worktree), then stop. Do not self-review or dispatch a checker.

## The defect (confirmed, read-only, Z11 return 010)

`tests/test_coordinator.py::TestHandleAfterTurn::test_advances_graph_timestep` (line 198) writes the REAL file `~/NeuroGraph/data/checkpoints/commons.msgpack`. Mechanism:

1. `neurograph_rpc.py:805` hard-codes `_COMMONS_CHECKPOINT_PATH = os.path.expanduser("~/NeuroGraph/data/checkpoints/commons.msgpack")` (LAW 5 debt).
2. The `live_memory` fixture injects a real `NeuroGraphMemory` on a tmp workspace as `neurograph_rpc._memory`; the test calls `handle_after_turn`.
3. `_last_save_time` is a module global initialised to `0.0` (`:796`), so `time_trigger` (`:3780`) is true on the first afterTurn in a fresh process → `_memory.save()` goes to tmp, then `_get_commons().persist(_COMMONS_CHECKPOINT_PATH)` (`:3798`) goes to the REAL path.
4. `reset_rpc_globals` (`test_coordinator.py:37`) does not reset `_last_save_time`, and nothing in `tests/` references `_COMMONS_CHECKPOINT_PATH`. Any standalone run of `test_coordinator.py` triggers the write. It has fired at least three times today.

The same constant is also read at `:2054-2061` (restore in `_handle_bootstrap_once`) and `:5988` (persist on clean exit in `main`), so a single source change covers all four sites.

## What to build (exactly this, nothing more)

**(a) LAW 5 — env-drive the path, default unchanged.** In `neurograph_rpc.py`, change the constant at `:805` to read env var **`NG_COMMONS_CHECKPOINT_PATH`** (matches the existing `NG_COMMONS_*` names at `:4219-4221`), falling back to the current default string unchanged:
`os.path.expanduser(os.environ.get("NG_COMMONS_CHECKPOINT_PATH", "~/NeuroGraph/data/checkpoints/commons.msgpack"))`.
No `.bashrc` edit (the default is unchanged; the env var is only the override channel). Keep the existing comment block and add a changelog header entry.

**(b) Test isolation in `tests/test_coordinator.py`.** Extend the autouse `reset_rpc_globals` fixture (it currently takes no arguments; add `tmp_path` and `monkeypatch`) so that for every test in the file it:
- points `neurograph_rpc._COMMONS_CHECKPOINT_PATH` at `tmp_path / "commons.msgpack"`;
- saves the original `neurograph_rpc._last_save_time`, sets it to `0.0` for the test, and restores it afterwards;
- saves the original `commons._commons` (the `get_commons()` singleton, `commons.py:114`), sets it to `None` for the test so each test gets a fresh Commons, and restores it afterwards.
Follow the existing save/restore style of that fixture. Add a changelog header entry.

**(c) A regression guard, new file `tests/test_commons_checkpoint_isolation_738.py`** (changelog header required). It must show, without touching the real path:
1. with `NG_COMMONS_CHECKPOINT_PATH` unset, the resolved default equals `os.path.expanduser("~/NeuroGraph/data/checkpoints/commons.msgpack")` (default unchanged);
2. with it set, the override is honored (use a subprocess with a controlled env so the import-time read is really exercised; the module must not start a live graph on import);
3. a `handle_after_turn`-driven persist lands under the redirected tmp path and the real path's `stat` is unchanged. Choose a fake or minimal `_memory` that keeps this test cheap and offline.
State in your return how you know this guard would have caught #738 (a negative control run against a scratch copy outside the repo tree, or an equivalent argument grounded in the code). Do not fake it.

## Allowed files — touch nothing else

`neurograph_rpc.py` (the one constant), `tests/test_coordinator.py` (the fixture), new `tests/test_commons_checkpoint_isolation_738.py`. NOT `neuro_foundation.py` (protected), NOT any vendored file (`ng_lite.py`, `ng_embed.py`, ...), NOT `commons.py`, NOT `data/`.

## HARD safety rules (read before running anything)

1. **Never run `tests/test_coordinator.py` (or any test that reaches `handle_after_turn`) until the fix is in place AND you have first run it with `NG_COMMONS_CHECKPOINT_PATH=<a tmp path>` exported.** The unfixed code rewrites the real file. First run: `NG_COMMONS_CHECKPOINT_PATH=$(mktemp -d)/commons.msgpack env -u NG_EMBED_REMOTE python3 -m pytest tests/test_coordinator.py -p no:cacheprovider -q`. Only after that passes and you have confirmed the fixture redirects on its own, run it a second time with the variable unset.
2. **Real file baseline (recorded by Z11 before dispatch):** `~/NeuroGraph/data/checkpoints/commons.msgpack` size `192776`, mtime `2026-09-29 15:47:53.946341621 -0800`, sha256 `2c910989224f5bed61e39168334510f260fccb2abddc3612755340972929f6f8`. Your return must show `stat` + `sha256sum` before and after your final verification runs, identical. **Do not delete, move, rewrite, or "quarantine" that file** — Chief has logged its removal for a later step; it is not part of this lane.
3. **Do not run the full suite.** It contains `tests/test_snn.py::TestLargeGraphStability::test_1k_nodes_10k_steps`, a PRD §9 stress test that has run >1.5 h without finishing and is unrelated to this lane. Run targeted files only: `test_coordinator.py`, `test_commons_persist_restore.py`, `test_commons_leg2.py`, `test_bootstrap_singleflight_430.py`, `test_cc_commons_persist.py`, `test_commons_bunyan_stage1.py`, `test_commons_experience_deposit.py`, `test_commons_outcome_deposit.py`, and your new file. If you ever need a broader run, add `--timeout=120 --timeout-method=signal --deselect "tests/test_snn.py::TestLargeGraphStability::test_1k_nodes_10k_steps"` and say so.
4. **NG_EMBED_\* env discipline (Packets 353/370, #732).** These runs inherit `NG_EMBED_REMOTE=hf` from the T3 process unless you unset it. Run every pytest command as `env -u NG_EMBED_REMOTE ...` (or set explicitly inside the test), and in your return **print the `NG_EMBED_*` variable NAMES (never values) each command ran under**. Never print secrets or `HF_TOKEN`.
5. **Scope of writes:** the three allowed files, the branch, and your return file only. **Write nothing to the vault (`~/docs`) and nothing to any auto-memory directory.** (The auto-memory store is shared across seats; a worktree_write scope is a code-write scope, not a memory/log scope.)
6. Every modified or new file gets a changelog header (`# ---- Changelog ----` block: date, author, What/Why/How) per repo convention. Attribute commits with your own harness-provided `Co-Authored-By` trailer.

## Acceptance

1. Default path string unchanged when the env var is unset (tested).
2. Env override honored (tested via subprocess).
3. `tests/test_coordinator.py` passes at least as many tests as on the base revision `e4ebf98` (state the base numbers you measured; pre-existing failures on base such as `TestInitCCHost::test_already_initialized_returns_true` are not yours to fix, but you must not add new ones).
4. After all runs with `NG_COMMONS_CHECKPOINT_PATH` unset, the real file is byte-identical to the baseline above (mtime, size, sha256).
5. Branch `cc-laptop-commons-testiso-20260930` committed and pushed (`git push -u origin cc-laptop-commons-testiso-20260930`); `git status -sb && git log -1` shown in the return.

## Return contents (`returns/worker-001.md`)

Files changed with line refs · every command run with the `NG_EMBED_*` names it ran under · test counts before (base) / after · real-file `stat`+sha256 before/after · negative-control evidence for the guard · commit SHA and push confirmation · anything surprising or out of scope you noticed (list it; do not act on it). Be plain about anything you did not verify.

---

## ADDENDUM A (Executive Packet 373 widening, added 2026-09-30 after dispatch; deliver on your next allowed turn, and apply even if you have already built the rest)

The recurrence of the real-file write came from ANOTHER seat's unguarded pytest, so a per-run guard is not enough. The guard must live **in the repo**, so every seat's suite gets it automatically. Two additions to this lane. Allowed files now also include **`tests/conftest.py` (new)** and **`tests/test_snn.py`** (marker only). Everything else in this assignment (safety rules, scope limits, write scope, return contents) is unchanged.

**A1. Repo-wide persist guard — new `tests/conftest.py` (changelog header required).**
- Patch `commons.Commons.persist` (session scope, applied before any test runs) so that a persist whose `os.path.realpath(os.path.expanduser(path))` equals the literal default `~/NeuroGraph/data/checkpoints/commons.msgpack` **raises** (RuntimeError naming #738). Compute the protected path from the literal default, NOT from `NG_COMMONS_CHECKPOINT_PATH`, so it cannot be defeated by the env var. Persists to any other path (tmp paths) behave exactly as before.
- **It must be loud, not just raise.** `handle_after_turn` wraps `persist` in `try/except Exception: logger.debug(...)` (`neurograph_rpc.py:3796-3800`), so a raise alone is swallowed and the test still passes. Record every blocked attempt (node id of the current test) and make the **session fail** at the end (e.g. `pytest_sessionfinish` setting a non-zero exit status and printing the offenders) so a swallowed block cannot hide. The record must be resettable so your own guard test can provoke one and clear it.
- If `commons` is not importable in some environment, say so in your return rather than silently skipping the guard.
- Guard test (in `tests/test_commons_checkpoint_isolation_738.py`): a persist to the real path raises and is recorded; a persist to a tmp path works; the session-fail mechanism is exercised without failing the real session.
- Verify by running `tests/test_coordinator.py` **only after** the guard exists and with `NG_COMMONS_CHECKPOINT_PATH` first pointed at a tmp path, exactly as in the HARD safety rules; then once with it unset. The real file must stay byte-identical (baseline now: size 192773, mtime `2026-09-29 17:38:16.949928157 -0800`; do not touch or quarantine it).

**A2. Fold in row #754 — the stall.** `tests/test_snn.py::TestLargeGraphStability::test_1k_nodes_10k_steps` never completes (>1.5 h) and has no marker or timeout, so any full-suite run silently stalls at ~86%. Mark it `@pytest.mark.slow` and `@pytest.mark.timeout(300)`; register both markers (in `tests/conftest.py` via `pytest_configure`/`addinivalue_line`) so an environment without `pytest-timeout` does not warn or error. Do not change the engine or the test body. Report whether it finishes within 300 s when run alone (run it ONLY with `--timeout=300 --timeout-method=signal`, alone, never inside a suite); if it does not, say so plainly, that is a finding about `neuro_foundation.py` for a separate lane.

**A3. THROTTLE (Packet 373) applies to you.** Only ONE full test suite runs org-wide and Z11 holds that slot: **you run targeted test files only, never a full suite.** Keep runs short.
