# Return — worker-001, lane `commons-test-isolation-738` (rows #738/#742)

Assignment: `handoffs/z11-commons-testiso-20260930/assignments/worker-001.md`. Authority: `worktree_write`. No PR opened, nothing merged, per instructions.

## Surprising / out-of-scope finding (reported, not acted on)

Before touching anything I recorded the real file's state and compared it to the baseline Z11 recorded before dispatch:

- Z11 baseline (pre-dispatch): size `192776`, mtime `2026-09-29 15:47:53.946341621 -0800`, sha256 `2c910989224f5bed61e39168334510f260fccb2abddc3612755340972929f6f8`
- What I observed at session start: size `192773`, mtime `2026-09-29 17:38:16.949928157 -0800`, sha256 `a4947c01f0e94d5eb125fc1a419d0982cb86a88421b838e2606c5ac87b15f319`

These do **not** match. The mtime is ~1h50m after the recorded baseline, consistent with the assignment's own note that the defect "has fired at least three times today" — the bug appears to have fired again between dispatch and the start of this session, most likely from another worktree/session running the still-unfixed `test_coordinator.py` (the hardcoded path is absolute, not worktree-scoped, so any worktree's test run can write the one real file). I did not attempt to restore, revert, or otherwise touch the real file to reconcile this — I don't have the old bytes, and the assignment explicitly forbids rewriting/quarantining it. Instead I used my own session-start snapshot (`192773` / `17:38:16.949928157` / `a4947c01...`) as the reference point for "unchanged by my own work," and that identity holds across every verification run below (see the stat/sha256 blocks). Flagging this for Josh/chief: the drift shows the defect was still live and firing in the window between Z11's baseline capture and dispatch reaching me, from a source outside this lane's scope.

## Files changed

1. **`neurograph_rpc.py`** — `_COMMONS_CHECKPOINT_PATH` constant, now at line 817 (was `:805` pre-edit; the diff added a 12-line changelog block above it). Changed from the bare literal to `os.path.expanduser(os.environ.get("NG_COMMONS_CHECKPOINT_PATH", "~/NeuroGraph/data/checkpoints/commons.msgpack"))`. Default string unchanged. Changelog header added directly above the constant.
2. **`tests/test_coordinator.py`** — `reset_rpc_globals` fixture (now at line 52, was line 38 pre-edit). Added `tmp_path`/`monkeypatch` params; redirects `neurograph_rpc._COMMONS_CHECKPOINT_PATH` to `tmp_path / "commons.msgpack"` via `monkeypatch.setattr` (auto-restored), sets `neurograph_rpc._last_save_time = 0.0` via `monkeypatch.setattr` (auto-restored), and explicitly saves/restores `commons._commons` to `None` for the test (matching the fixture's existing explicit save/restore style for `_memory`/`_tract`/etc.). Changelog header added directly above the fixture.
3. **`tests/test_commons_checkpoint_isolation_738.py`** — new file, 4 tests across 4 classes: default-path-unchanged (subprocess), env-override-honored (subprocess), handle_after_turn-driven persist lands under redirect + real path untouched (in-process, fake `_memory`/`_FakeGraph`, River/tract/embedding helpers monkeypatched to no-ops), and an executed negative control (see below). Changelog header at top of file.

No other files touched. `git diff --stat` confirms exactly these two modified + one new file (plus the pre-existing untracked `handoffs/` dir, which was already present at session start and is where this return lives).

## Commands run, with `NG_EMBED_*` names each ran under

Ambient environment had exactly one `NG_EMBED_*` variable set: `NG_EMBED_REMOTE`. Every pytest invocation below explicitly unset it (`env -u NG_EMBED_REMOTE`), so every pytest run had **zero `NG_EMBED_*` variables set**. The three subprocess-based tests inside the new guard file build their own fully-controlled env dicts (`PATH`, `HOME`, `PYTHONPATH`, plus `NG_COMMONS_CHECKPOINT_PATH` where relevant) that do not inherit `NG_EMBED_*` at all — also zero.

1. First mandated run, fix in place, path exported explicitly:
   `NG_COMMONS_CHECKPOINT_PATH=<mktemp -d>/commons.msgpack env -u NG_EMBED_REMOTE python3 -m pytest tests/test_coordinator.py -p no:cacheprovider -q`
   → 28 passed, 1 failed (`TestInitCCHost::test_already_initialized_returns_true`, pre-existing). Real file unchanged.
2. Second mandated run, var unset, `--basetemp` added only to inspect the redirected file afterward (not part of the normal acceptance run, just fixture verification):
   `env -u NG_EMBED_REMOTE -u NG_COMMONS_CHECKPOINT_PATH python3 -m pytest tests/test_coordinator.py --basetemp=/tmp/pytest-verify-738 -p no:cacheprovider -q`
   → 28 passed, 1 failed (same test). Confirmed `commons.msgpack` written under `test_advances_graph_timestep0/`, `test_recovers_text_from_params0/`, `test_clears_ingest_cache_after0/` inside basetemp — i.e., the fixture redirects on its own, no env var needed. Real file unchanged.
3. New guard file standalone: `env -u NG_EMBED_REMOTE python3 -m pytest tests/test_commons_checkpoint_isolation_738.py -p no:cacheprovider -q` → 4 passed.
4. Base-revision count (see next section) — separate scratch clone, not this worktree.
5. Full targeted combined run (final acceptance run), fix in place, var unset:
   `env -u NG_EMBED_REMOTE python3 -m pytest tests/test_coordinator.py tests/test_commons_persist_restore.py tests/test_commons_leg2.py tests/test_bootstrap_singleflight_430.py tests/test_cc_commons_persist.py tests/test_commons_bunyan_stage1.py tests/test_commons_experience_deposit.py tests/test_commons_outcome_deposit.py tests/test_commons_checkpoint_isolation_738.py -p no:cacheprovider -q`
   → **81 passed, 1 failed** (same `TestInitCCHost::test_already_initialized_returns_true`). No full-suite run, no `test_snn.py` touched.

## Test counts — base vs. after

**Base (`e4ebf98`)**, measured safely: I could not run the unfixed `tests/test_coordinator.py` directly against the real path (the HARD safety rule forbids it, and the pre-fix code has no env-var escape hatch at all). Instead I `git clone`d this worktree into a fully separate scratch directory (`/tmp/ng-base-count-738-*`, outside both this worktree and `~/NeuroGraph`), checked out `e4ebf98` there, and added a **scratch-only, never-committed** `tests/conftest.py` with one autouse fixture that monkeypatches `neurograph_rpc._COMMONS_CHECKPOINT_PATH` to a `tmp_path` location before every test — touching nothing else, so the measured behavior is the real base-revision test bodies, just denied the one path that could hit the real file. Ran there: `env -u NG_EMBED_REMOTE python3 -m pytest tests/test_coordinator.py -p no:cacheprovider -q` →
**28 passed, 1 failed** (`TestInitCCHost::test_already_initialized_returns_true`) — same single pre-existing failure named in the assignment, not mine to fix. Real file confirmed unchanged immediately after (see below). Scratch clone deleted afterward (`rm -rf` of a directory I created under `/tmp`, nothing under version control or the live ecosystem).

**After (this branch)**: 28 passed, 1 failed — identical counts, same one pre-existing failure, no new failures. Combined with the other 6 targeted commons/bootstrap files + the new guard file: 81 passed, 1 failed total.

## Real-file `stat` + `sha256sum`, before/after

Identical at every checkpoint across the whole session (session start, after run 1, after run 2, after the new file's standalone run, after the base-revision scratch run, after the final combined run):

```
  File: /home/josh/NeuroGraph/data/checkpoints/commons.msgpack
  Size: 192773
  Modify: 2026-09-29 17:38:16.949928157 -0800
sha256: a4947c01f0e94d5eb125fc1a419d0982cb86a88421b838e2606c5ac87b15f319
```

This is my own session-start snapshot, **not** the Z11-recorded baseline (`192776` / `15:47:53.946341621` / `2c910989...`) — see the flagged finding above for why those two don't match. What I can attest to directly: nothing I ran during this lane changed the real file by one byte, in either direction, across seven separate pytest invocations plus two ad hoc subprocess checks.

## Negative-control evidence for the guard

Per the assignment: "a negative control run against a scratch copy outside the repo tree, or an equivalent argument grounded in the code... do not fake it." I did the former, actually executed (see `TestNegativeControlAgainstPreFixLine` in the new file) — not a documentation-only stub. It:

1. Writes nothing to disk beyond two one-line `python3 -c` scratch expressions (not full-module imports — the full pre-fix `neurograph_rpc.py` no longer exists anywhere to import, since this branch's copy is already fixed).
2. Runs the **literal pre-fix expression** (`os.path.expanduser("~/NeuroGraph/data/checkpoints/commons.msgpack")`, no env read) and the **literal fixed expression** as two subprocesses, both under an env with `HOME` pointed at a throwaway `tmp_path` scratch directory (so even a full resolution can never land under the real `~/NeuroGraph`) and `NG_COMMONS_CHECKPOINT_PATH` set to a third, distinct redirect target.
3. Asserts the pre-fix expression **ignores** the redirect and resolves under `$HOME` regardless (on a real machine, `$HOME` is the real home directory — this unconditional resolution *is* the #738 mechanism: no test-side fixture, tmp_path, or monkeypatch could ever have redirected it, because the source line never consulted anything but the literal string).
4. Asserts the fixed expression **honors** the redirect.
5. Asserts neither subprocess touched the scratch `$HOME/NeuroGraph` path or the real file (fingerprinted before/after).

This is a real, executed, safe reproduction of the exact causal gap #738 exploited, run in this session (`4 passed` includes this test). I did not fake it with an assertion-only placeholder — an earlier draft of this file did that and I replaced it before finalizing, specifically because it didn't meet the "do not fake it" bar.

## What I did not verify

- I did not attempt to independently confirm *what* wrote the real file between the Z11 baseline capture and my session start (no log/process evidence gathered — out of this lane's scope, flagged above instead).
- I did not run the full test suite (forbidden by the HARD safety rules) and have no data on files outside the targeted list.
- I did not verify behavior under concurrent/multi-process access to the redirected tmp path (not asked for; the existing fixture pattern this mirrors doesn't test that either).

## Commit / push

