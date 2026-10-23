# Assignment worker-001 — lane `autonomic-no-advance-alarm-117` (row #117, LAW 8 family)

- **Zone manager:** Z11 (CC session `c05315b5-4699-4a93-890a-eb9ec3af7952`) · **Reports to:** chief-003 · **Ruling:** Executive Packet 370 (silent failures are fixed on sight, each as its own lane) + chief-003's base ruling: **standalone, minimal, off `origin/main`**
- **Worktree / branch:** `/home/josh/NeuroGraph-worktrees/z11-no-advance-alarm-20260930` · `cc-laptop-no-advance-alarm-20260930`, cut from NeuroGraph `origin/main` `e4ebf98`
- **Authority:** `worktree_write` only. Commit and push the branch. **Do NOT open a PR. Do NOT merge.** Merges ride the P329/P337 rollout hold; nothing here wires into a live service.
- **Return to:** `handoffs/z11-no-advance-alarm-20260930/returns/worker-001.md` (in this worktree), then stop. Do not self-review or dispatch a checker.

## The silent failure

Both graphs' step counters were found frozen with **nothing reporting it**: laptop `timestep` 33,637 unchanged since 2026-09-12; VPS 80,259 unchanged for ≥25 h. The substrate's autonomic stepping is supposed to run on its own wall-clock (LAW 8: nothing autonomic is gated on a conversation), but nothing notices when the counter it is denominated in stops advancing. **This lane builds the alarm, not the cure.** The cure for the frozen clock itself is #117 / S7 (a separate lane); this alarm must exist before it and keep existing after it.

Read-only findings on `origin/main` `e4ebf98` (`neurograph_rpc.py`), verify them yourself before relying on them:
- `_scan_drain_pulse_loop` (`:4341`) is the wall-clock loop (`_SCAN_DRAIN_INTERVAL_SECONDS = 2.0`, `:4169`). Each tick, when **not paused** and `_memory is not None`, it calls `_memory.graph.step()` (`:4389`).
- That step is wrapped in `except BaseException` and a failure is logged at **`logger.debug`** only (`:4402`). A step that raises every tick (a Rust `PanicException` reaches here) leaves the counter frozen and produces **no visible log at all** at default log level.
- The loop **skips the step entirely** when `paused` (sentinel file `_SCAN_DRAIN_PAUSE_FILE` or autonomic `SYMPATHETIC` via the Commons, `:4357-4369`). A long pause also freezes the counter, with only a one-line state-transition log.
- The other step sites are `handle_after_turn` (`:3659`, conversation-gated) and the dream pass (`:5253`). The counter is `_memory.graph.timestep`.
- `handle_stats` (`:5314`) returns `_memory.stats()` plus `module_hooks`; that is the natural place to surface the alarm state.

## What to build (minimal, in `neurograph_rpc.py` only, plus one new test file)

1. **A no-advance check that runs on every tick of the existing wall-clock loop** (LAW 8: it must not depend on a conversation, so it lives in `_scan_drain_pulse_loop`, not in `on_message`/`handle_after_turn`). Put the logic in a small module-level helper so it is unit-testable without threads. Each tick it records `graph.timestep` and the wall-clock time it last **changed**. It runs on every tick **including paused ticks and ticks where the step raised**.
2. **Alarm condition:** the timestep has not advanced for longer than a threshold. Threshold from env var **`NG_NO_ADVANCE_ALARM_SECS`** (LAW 5), default chosen and justified by you relative to the 2 s pulse and the 300 s save cadence (`_SAVE_INTERVAL_SECS`); set to `0` to disable. Do not hardcode.
3. **Announce, loudly, once per episode:** on entering the alarm state emit one `logger.error` line with the frozen `timestep`, seconds frozen, whether the loop is paused (and by which source: sentinel vs autonomic), the consecutive step-failure count and the last step error `repr`; latch so it does not repeat every 2 s (re-emit at a slow interval, e.g. every 15 min, so a long freeze stays visible); on recovery emit one `logger.info` "resumed" line with the frozen duration. Keep the state in module globals like the file's other loop state.
4. **Stop hiding the step failure:** the `logger.debug("Autonomous substrate step failed: %r")` at `:4402` becomes a counted failure. Keep the per-tick behavior (thread survives, drain and scoop still run) but count consecutive failures and log the first failure and every Nth (e.g. 1st, then every 30th) at `WARNING` with the count. Do not change what happens on failure beyond making it visible.
5. **Surface it in `handle_stats`:** add an additive `no_advance` block (e.g. `alarm`, `timestep`, `frozen_secs`, `paused`, `paused_by`, `consecutive_step_failures`, `last_step_error`, `threshold_secs`). Additive only; do not rename or remove existing keys.
6. **The alarm is READ-ONLY.** It must **never call `graph.step()`, restart a loop, or change any state** of the substrate. Curing the freeze is #117 / S7, not this lane. Do not "helpfully" step the graph from the watchdog.

## Allowed files — touch nothing else

`neurograph_rpc.py` and one new test file `tests/test_no_advance_alarm_117.py`. **NOT** `neuro_foundation.py` (PROTECTED, Syl's Law), NOT any vendored file (`ng_lite.py`, `ng_autonomic.py`, `ng_embed.py`, ...), NOT `commons.py`, NOT `openclaw_hook.py`, NOT `data/`.

## Tests (`tests/test_no_advance_alarm_117.py`, changelog header required)

Deterministic, no real threads, no real graph, no network: drive the helper with a fake clock and a fake graph. Cover: (a) advancing counter → no alarm; (b) counter frozen past threshold → exactly one ERROR, alarm flag true, stats block correct; (c) still frozen → no per-tick repeat, slow re-emit honored; (d) frozen while paused → ERROR names `paused_by`; (e) step raises every tick → consecutive count grows, first + every-Nth WARNING, thread-survival behavior unchanged; (f) counter advances again → one INFO recovery, alarm cleared; (g) `NG_NO_ADVANCE_ALARM_SECS=0` disables; (h) `handle_stats` carries the `no_advance` block without dropping existing keys. State plainly which of these would have FAILED on the unfixed base.

## HARD safety rules

1. **The real file `~/NeuroGraph/data/checkpoints/commons.msgpack` is rewritten by any run of a test that reaches `handle_after_turn`** (row #738; its fix lane `commons-test-isolation-738` is not merged yet). **Never run `tests/test_coordinator.py`, `tests/test_tonic_bridge.py`, or any test that calls `handle_after_turn`, and never run the full suite.** For every pytest command, use the guard plugin: `env -u NG_EMBED_REMOTE PYTHONPATH=/tmp/z11_r5 python3 -m pytest ... -p z11_noreal_commons -p no:cacheprovider` (it redirects the path and blocks a persist to the real file). Do not touch, move, or delete that file.
2. **Do not run the full suite.** `tests/test_snn.py::TestLargeGraphStability::test_1k_nodes_10k_steps` never completes (row #754). Run targeted files only: your new file, `tests/test_scan_drain_guard.py`, `tests/test_dream_pulse_wiring.py`, `tests/test_tonic_lifecycle.py`, `tests/test_commons_leg2.py`. For any broader run add `--timeout=120 --timeout-method=signal` and deselect the stress test, and say so.
3. **NG_EMBED_\* env discipline (Packets 353/370, #732).** Run every pytest command as `env -u NG_EMBED_REMOTE ...` and **print the `NG_EMBED_*` variable NAMES (never values) each command ran under** in your return. Never print secrets or tokens.
4. **Write scope:** the two allowed files, the branch, and your return file only. **Write nothing to the vault (`~/docs`) and nothing to any auto-memory directory** (the auto-memory store is shared across seats).
5. Changelog header (`# ---- Changelog ----`: date, author, What/Why/How) in every file you modify or add. Attribute commits with your own harness-provided `Co-Authored-By` trailer.
6. Do not read or print any live checkpoint contents. Do not start or restart any service or sidecar.

## Acceptance

1. Alarm fires once per episode with the frozen counter, duration, pause state and failure info; recovers cleanly; latch and slow re-emit work (tested).
2. Step failure no longer silent at default log level (tested).
3. `handle_stats` exposes `no_advance` additively (tested).
4. Read-only: a test asserts the watchdog never calls `step()` on the fake graph.
5. Existing targeted suites (`test_scan_drain_guard.py`, `test_dream_pulse_wiring.py`, `test_tonic_lifecycle.py`, `test_commons_leg2.py`) pass with counts equal to base `e4ebf98` (state the base numbers you measured).
6. Real commons file byte-identical before/after (size, mtime, sha256 shown; baseline for reference: size 192773, mtime `2026-09-29 17:38:16.949928157 -0800`).
7. Branch `cc-laptop-no-advance-alarm-20260930` committed and pushed (`git push -u origin cc-laptop-no-advance-alarm-20260930`); `git status -sb && git log -1` shown in the return.

## Return contents (`returns/worker-001.md`)

Files changed with line refs · design choices (threshold default and why; latch/re-emit interval) · every command run with the `NG_EMBED_*` names it ran under · test counts before (base) / after · which tests fail on the base · real-file stat+sha256 before/after · commit SHA and push confirmation · anything surprising or out of scope you noticed (list it; do not act on it; e.g. whether the drain-pulse thread itself dying would also silence this alarm). Be plain about anything you did not verify.
