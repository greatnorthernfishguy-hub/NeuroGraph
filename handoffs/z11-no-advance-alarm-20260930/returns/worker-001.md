# Return — worker-001 — lane `autonomic-no-advance-alarm-117`

Zone manager: Z11 (`c05315b5-4699-4a93-890a-eb9ec3af7952`) · Reports to: chief-003
Branch: `cc-laptop-no-advance-alarm-20260930`, cut from `origin/main` `e4ebf98`
Status: **implemented, tested, committed, pushed.** Not self-reviewed. No PR opened (per instruction).

## Files changed

- `neurograph_rpc.py`
  - New block inserted after `_LOOP_MUST_PROPAGATE = (...)` (was line 4338, now ~4338–4477 in the new file):
    `_NO_ADVANCE_ALARM_SECS` / `_NO_ADVANCE_REEMIT_SECS` / `_NO_ADVANCE_FAILURE_WARN_EVERY` env-sourced
    constants, `_no_advance_state` module dict, and four helpers:
    `_no_advance_note_step_failure()`, `_no_advance_note_step_success()`, `_no_advance_tick()`,
    `_no_advance_stats_block()`.
  - `_scan_drain_pulse_loop()` (was ~4341–4433, now ~4478–4577): on a successful `graph.step()` calls
    `_no_advance_note_step_success()`; on a caught `BaseException` replaces the old
    `logger.debug("Autonomous substrate step failed: %r", _exc)` with
    `_no_advance_note_step_failure(_exc)`. After the `if not paused:` block (so it also runs on paused
    ticks), added an unconditional block that computes `paused_by` from `sentinel_paused`/`autonomic_paused`
    and calls `_no_advance_tick(_memory.graph.timestep, time.time(), paused, paused_by)`.
  - `handle_stats()` (was ~5314–5323, now ~5468–5480): added `stats["no_advance"] = _no_advance_stats_block()`
    after the existing `module_hooks` assignment. Additive only.
  - Changelog header added on the new block per repo convention.
- `tests/test_no_advance_alarm_117.py` (new, 277 lines, changelog header included).

Nothing else touched. Did not modify `neuro_foundation.py`, any vendored file, `commons.py`,
`openclaw_hook.py`, or anything under `data/`.

## Design choices

- **Threshold default 300s**, from `NG_NO_ADVANCE_ALARM_SECS` (LAW 5, no hardcode). Chosen to exactly match
  `_SAVE_INTERVAL_SECS` (also 300s): the pulse ticks every 2s, so a healthy substrate steps ~150 times within
  one save interval. Going a *whole save interval* without one successful step is already anomalous and
  worth paging on, while still tolerant of a single slow tick or a brief SYMPATHETIC-driven pause. This is
  minutes, not the 17-day / 25h freezes that motivated the lane, so it catches the real failure mode fast
  without being noisy on ordinary jitter. `0` disables (tested).
- **Re-emit interval 900s (15 min)**, from `NG_NO_ADVANCE_REEMIT_SECS` — the assignment's own suggested value.
  Made env-tunable (not required by the assignment, but consistent with LAW 5 and cheap to add) rather than
  hardcoded.
- **Failure-warn cadence**: 1st failure, then every 30th, from `NG_NO_ADVANCE_FAILURE_WARN_EVERY` — again the
  assignment's suggested value, made env-tunable for the same reason.
- `_no_advance_tick(timestep, now, paused, paused_by)` takes no graph reference at all — it's handed a bare
  timestep value, which is what makes "never calls `graph.step()`" a structural guarantee rather than an
  observed behavior. State lives in a module-global dict (`_no_advance_state`), mirroring the file's existing
  pattern for other loop state (e.g. `_tonic_idle_*`, `_last_save_time`).
- Wiring point: the watchdog tick is placed *after* the `if not paused:` block, not inside it, specifically
  so it still runs on paused ticks (per the assignment's explicit requirement). The step-success/failure
  notes stay inside the existing try/except around `graph.step()`, since that's the only place that knows
  the outcome.

## Commands run and NG_EMBED_* names

Every pytest invocation below was run as `env -u NG_EMBED_REMOTE PYTHONPATH=/tmp/z11_r5 python3 -m pytest ... -p z11_noreal_commons -p no:cacheprovider`. The only `NG_EMBED_*` variable present in the environment at any point was **`NG_EMBED_REMOTE`** (unset for every run via `-u`); no other `NG_EMBED_*` names were ever set. No values printed, per instruction.

1. `env | grep -o '^NG_EMBED_[A-Z_]*' | sort -u` → confirmed only `NG_EMBED_REMOTE` present.
2. `pytest tests/test_no_advance_alarm_117.py` (fixed tree) → 12 passed.
3. `git stash push -u -m "z11-no-advance-117-worker-001-check-base-<pid>"` (unique tag, captured SHA immediately via `git stash list --format='%H %gs'`, never bare `git stash`).
4. Extracted the new test file from the stash's untracked-commit (`git show <stash>^3:tests/test_no_advance_alarm_117.py`) into `/tmp/z11_r5/`, copied it into `tests/` on the now-base tree, ran the same pytest command → **12 failed** (all `AttributeError: module 'neurograph_rpc' has no attribute '_no_advance_state'` in `setUp`, since none of the new module state/functions exist on `e4ebf98`).
5. Removed the temp copy, `git stash apply <sha>` (not pop), confirmed `git status -sb` matched pre-stash state, `git stash drop 'stash@{0}'` after re-confirming by tag it was the right entry. Other sessions' stash entries (`cc-laptop-lenia-reconcile-371-...`, etc.) untouched throughout.
6. `pytest tests/test_scan_drain_guard.py tests/test_dream_pulse_wiring.py tests/test_tonic_lifecycle.py tests/test_commons_leg2.py` (fixed tree) → **28 passed**.
7. Repeated the stash-push/measure/stash-apply/drop cycle (steps 3–5's pattern, fresh unique tag) to measure the same four files on the base `e4ebf98` → **28 passed** — identical count, no test in this set changed behavior.
8. `stat` + `sha256sum` on `~/NeuroGraph/data/checkpoints/commons.msgpack` before starting and again after all runs; also checked `/tmp/z11_r5/plugin_trace.log` for `BLOCKED` entries (count: 0 — no test ever attempted to write the real path).

No full-suite run at any point. No `test_coordinator.py`, `test_tonic_bridge.py`, or any `handle_after_turn`-reaching test was run.

## Test counts

| Suite | Base (`e4ebf98`) | Fixed (this branch) |
|---|---|---|
| `tests/test_no_advance_alarm_117.py` (new) | 12 failed (setUp AttributeError — feature doesn't exist) | 12 passed |
| `test_scan_drain_guard.py` + `test_dream_pulse_wiring.py` + `test_tonic_lifecycle.py` + `test_commons_leg2.py` (combined) | 28 passed | 28 passed |

All 12 new tests would have failed on the unfixed base (each test's docstring states this explicitly; verified by actually running them there, not just asserted).

## Real commons file — before/after

Before any work and after all test runs, identical:
- size: `192773`
- mtime: `2026-09-29 17:38:16.949928157 -0800`
- sha256: `a4947c01f0e94d5eb125fc1a419d0982cb86a88421b838e2606c5ac87b15f319`

Matches the assignment's stated baseline exactly. No checkpoint contents were read or printed at any point.

## Commit and push

- Commit: `6d99f097dc76c14dc724dc362ce227a9d90fa108` — "Add no-advance alarm for frozen NG step counter (row #117)"
- `git push -u origin cc-laptop-no-advance-alarm-20260930` — succeeded, new branch on origin, tracking set up. No PR opened, no merge attempted.
- Closing check:
  ```
  ## cc-laptop-no-advance-alarm-20260930...origin/cc-laptop-no-advance-alarm-20260930
  ?? handoffs/
  commit 6d99f097dc76c14dc724dc362ce227a9d90fa108
  ```
  (`handoffs/` is this assignment/return scaffolding itself, pre-existing at session start per the initial `git status`; it is not part of the two allowed source files and was intentionally not committed to the code branch.)

## Surprising / out-of-scope observations (not acted on)

1. **The drain-pulse thread dying would also silence this alarm.** `_no_advance_tick` only runs from inside
   `_scan_drain_pulse_loop`'s own iteration. If the thread itself exits (e.g. `_start_scan_drain_pulse` was
   never called, or some future change lets an exception past the loop's outer `except BaseException`), no
   ticks occur at all, so no ERROR is ever logged and `handle_stats()`'s `no_advance.frozen_secs` simply stops
   updating without `alarm` ever flipping true. The watchdog detects a *live* loop observing a stuck counter;
   it does not detect its own host loop being dead. Whether that gap needs its own alarm (e.g. a last-tick
   heartbeat timestamp checked from outside the thread) is a design question for #117/S7 or a follow-up lane,
   not something I built here — flagging per instruction, not fixing.
2. `_memory is None` gates both the step-success/failure notes and the tick call — before bootstrap, the
   watchdog is inert by design (there's no timestep to watch yet). Not a bug, just noting the boundary.
3. I did not verify this against the live VPS or laptop process in any way — no service was started or
   restarted, per the hard rules. Everything above is from the worktree's static source and pytest only.
4. I added two extra env vars (`NG_NO_ADVANCE_REEMIT_SECS`, `NG_NO_ADVANCE_FAILURE_WARN_EVERY`) beyond the
   one the assignment named (`NG_NO_ADVANCE_ALARM_SECS`). This is an additive, backward-compatible choice
   made for LAW 5 consistency, not a requirement — flagging in case chief-003/the reviewer would rather these
   stayed as hardcoded constants for a "minimal" lane.

## What I did not verify

- I did not run the guard plugin's own test suite or otherwise audit `/tmp/z11_r5/z11_noreal_commons.py`
  beyond reading it once; I'm trusting Z11's setup of it as stated in the assignment.
- I did not check whether any *other* branch/worktree touching `neurograph_rpc.py` concurrently would
  conflict with this diff — this lane was scoped as standalone off `origin/main`, and I did not investigate
  beyond that.
