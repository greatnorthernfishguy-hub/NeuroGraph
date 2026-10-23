# Return — worker-002 (CORRECTION) — lane `autonomic-no-advance-alarm-117`

Zone manager: Z11 (`c05315b5-4699-4a93-890a-eb9ec3af7952`) · Reports to: chief-003 (GO given)
Branch: `cc-laptop-no-advance-alarm-20260930`, new commit on top of `6d99f097dc76c14dc724dc362ce227a9d90fa108`
Status: **C1-C5 implemented, tested with mutation proof, committed, pushed.** Not self-reviewed. No PR opened, no merge.

## Files changed (line refs are post-correction line numbers in the pushed file)

- `neurograph_rpc.py`
  - `:4341-4580` — full no-advance block rewritten: changelog corrected to state the TRUE coverage (was false: claimed to cover "the autonomous step() call raises every tick", which per `neuro_foundation.py:2184-2185` cannot be detected by a raw-counter watchdog since `timestep += 1` is `Graph.step()`'s first statement under its lock). New `_no_advance_state` keys: `last_autonomous_success_ts`, `kind`, `last_tick_ts`. `_no_advance_note_step_failure` (`:4382-4404`): C3(c) zero/negative-safe warn-every (no bare modulo), C3(b) wrapped in its own `try/except Exception`. `_no_advance_note_step_success` (`:4406-4424`): new injectable `now` param, sets `last_autonomous_success_ts`, same exception-safety wrap. `_no_advance_tick` (`:4426-4492`): C1 dual-clock rewrite — tracks raw-counter staleness and autonomous-success staleness independently, `kind` names which, recovery requires both clear, bootstraps `last_autonomous_success_ts` on first tick; C2 sets `last_tick_ts` unconditionally (even when disabled); C3(c) `_NO_ADVANCE_REEMIT_SECS<=0` → never re-emit. `_no_advance_stats_block` (`:4494-4514`): additive `kind`, `autonomous_stale_secs`, `last_tick_ts`, `tick_age_secs` (None, not 0, when never ticked).
  - `_scan_drain_pulse_loop` (`:4517-...`): single `_tick_now = time.time()` read per iteration (was two separate reads), reused for the success-note, the tick, and the auto-save check — this is what makes the loop deterministically fake-clock-testable (C4). `_no_advance_note_step_success(now=_tick_now)` now passed explicitly. The tick call site (`_paused_by` computation + `_memory.graph.timestep` read + `_no_advance_tick(...)`) is now wrapped in its own `try/except Exception` logging WARNING (C3a) so a watchdog bug can never reach the outer loop guard and skip the auto-save below it.
  - `handle_stats()` unchanged beyond what `_no_advance_stats_block()` now additively returns (no new code needed there — same `stats["no_advance"] = _no_advance_stats_block()` line as worker-001).
- `tests/test_no_advance_alarm_117.py` — substantially rewritten (26 tests, up from 12). Updated changelog documents which old tests (`a`, `f`) encoded the old counter-only semantics and why they changed (see docstrings). Added: `TestNoAdvanceWatchdogLiveness` (C2), zero/negative-tunable unit tests (C3c), exception-safety unit tests (C3b, via an `_ExplodingDict` that raises on write), a dual-clock unit test proving the C1 bug directly, and a new `TestRealLoopWiring` class (C4) that drives the REAL `_scan_drain_pulse_loop()` with `_FakeShutdown`/`_FakeGraph`/`_FakeMemory`/`_make_clock` fakes — 7 tests covering (i)-(vi) per the assignment.

Nothing else touched. `neuro_foundation.py` was read-only consulted to verify the reviews' claim (confirmed, see below) and never modified.

## Verified facts from the correction brief

- `neuro_foundation.py:2184-2185`: confirmed by direct read — `with self._step_lock: self.timestep += 1` is literally the first two lines of `Graph.step()`'s body. A step raising anywhere after this still leaves the counter advanced.
- Reproduced this exactly in `_FakeGraph.step()` (test file): `self.timestep += 1` unconditionally, THEN raises on a scripted "raise" outcome — `test_vi_step_raising_every_tick_with_counter_advancing_still_alarms` proves the old counter-only alarm would have stayed silent here forever, and the corrected one alarms via `kind="no_successful_autonomous_step"`.
- The "conversational turn advances the same counter" scenario is proven at the wiring level by `test_vi_conversation_style_external_timestep_advance_without_autonomous_success_alarms`: the loop runs fully PAUSED (autonomous step never attempted), yet an externally-injected `graph.timestep += 1` (simulating `handle_after_turn`) between pulses still produces an alarm, because the autonomous-success clock — untouched by that external advance — is what trips it.

## Design choices

- **Two independent clocks, not one.** `last_change_ts` (raw counter, any driver) and `last_autonomous_success_ts` (set ONLY by `_no_advance_note_step_success`, i.e. only by this loop's own autonomous call site — `handle_after_turn`'s call does not feed it). Alarm = either stale; recovery = both clear. This is the direct fix for the reviews' core finding.
- **`last_autonomous_success_ts` bootstraps on the first tick, not on the first success** — mirrors the original `last_change_ts` bootstrap (process start isn't "infinitely stale"), per the assignment's explicit instruction.
- **Single `time.time()` read per loop iteration** (new in this pass): originally there were two separate `time.time()` calls (one for the tick, one for auto-save). Unifying them into one `_tick_now` was necessary for C4's real-loop tests to be deterministic, and is also more correct in its own right (all three consumers — success-note, tick, auto-save — now agree on "what time is it this iteration").
- **Fake-clock design for C4**: a fixed-length `side_effect` list does NOT work here — `logging.LogRecord.__init__` calls `time.time()` for every emitted log record, so the number of real `time.time()` calls per iteration depends on how many WARNING/ERROR/INFO lines that iteration happens to emit (verified empirically: a first-failure WARNING consumes one extra call). I switched to `_make_clock()`, a stateful zero-arg callable returning a value the test advances explicitly via `.set()`, driven from `_FakeShutdown`'s new `on_check` hook (fires once per while-condition check, i.e. once per "about to start iteration N"). This decouples "how many log lines fired" from "what time it is" and is the only part of this correction's test design I had to iterate on after first observing a `StopIteration` failure.
- **C3(c) semantics**: `<=0` chosen (not `==0`) for both tunables, matching the assignment's literal wording and covering a stray negative env value the same way as zero.

## Commands run and NG_EMBED_* names / `[z11 origin]` lines

Every command: `env -u NG_EMBED_REMOTE PYTHONPATH=/tmp/z11_r5 python3 -m pytest ... -p z11_noreal_commons_v5 -p no:cacheprovider`. The only `NG_EMBED_*` name ever present in the environment was `NG_EMBED_REMOTE` (unset via `-u` every time); no other `NG_EMBED_*` names existed.

1. `tests/test_no_advance_alarm_117.py` alone (first pass, before the clock fix) → 4 failed (StopIteration in `test_iii`/`test_v`/`test_vi`×2) — diagnosed and fixed via `_make_clock`, see Design choices.
2. `tests/test_no_advance_alarm_117.py` alone (after the fix) → **26 passed**. `[z11 origin]`:
   ```
   [z11 origin] cwd/worktree = /home/josh/NeuroGraph-worktrees/z11-no-advance-alarm-20260930/
   [z11 origin]  neurograph_rpc       WORKTREE       .../neurograph_rpc.py
   [z11 origin]  commons              WORKTREE       .../commons.py
   [z11 origin]  neuro_foundation     (not imported yet)
   [z11 origin] neurograph_rpc.__file__ = .../neurograph_rpc.py
   [z11 origin] END: ... | repo modules loaded from MAIN checkout: ng_salience_gate
   ```
3. Four neighbor files together → **28 passed**. Same origin lines, this time `neuro_foundation WORKTREE` too (imported this run). Same `ng_salience_gate` MAIN-CHECKOUT note (see Out of scope below).
4. All five targeted files together → **54 passed** (26+28, no cross-pollution).
5. Mutation proof (below) — 4 separate runs, one per mutation, each printing its own `[z11 origin] neurograph_rpc.__file__` confirming the mutant copy, not the worktree, was exercised.

## Test counts

| Suite | Count |
|---|---|
| `tests/test_no_advance_alarm_117.py` | 26 passed (up from worker-001's 12; 14 new) |
| 4 neighbor files combined | 28 passed (unchanged from worker-001's measured base) |
| All 5 together | 54 passed |

I did not re-measure a "pre-worker-001 base" count for this pass — that was already established in worker-001's return (12 tests, all AttributeError on `e4ebf98`) and remains valid; this correction only changes behavior worker-001 introduced, not anything in `e4ebf98` itself.

## Mutation proof (C4, required)

Four copies made under `/tmp/z11_117_mut/mut{1,2,3,4}/`, each containing the corrected `neurograph_rpc.py` with exactly ONE edit applied (never the worktree file), plus an unmodified copy of `commons.py` (needed only so the guard plugin's configure-time `import commons` resolves — `commons.py` itself was not part of any mutation) and the test file. Each run used `PYTHONPATH=/tmp/z11_r5:/tmp/z11_117_mut/mutN` so `commons` resolves from the mutation dir and `neurograph_rpc` resolves from the mutant copy (confirmed by the printed `neurograph_rpc.__file__` each time).

| # | Mutation | Failing test(s) |
|---|---|---|
| 1 | Removed the `_no_advance_tick(...)` call (replaced with `pass`) | `test_iii_tick_runs_from_real_loop_and_errors_after_threshold_while_paused`, `test_iv_tick_raising_does_not_skip_auto_save_and_warns`, `test_vi_conversation_style_external_timestep_advance_without_autonomous_success_alarms`, `test_vi_step_raising_every_tick_with_counter_advancing_still_alarms` (4 failed, 22 passed) |
| 2 | Removed the `_no_advance_note_step_success(now=_tick_now)` call | `test_ii_succeeding_step_through_real_loop_resets_and_sets_autonomous_ts` (1 failed, 25 passed) |
| 3 | Removed the `_no_advance_note_step_failure(_exc)` call | `test_i_failing_step_through_real_loop_increments_failure_count`, `test_v_zero_tunables_do_not_raise_or_skip_auto_save` (2 failed, 24 passed) |
| 4 | Removed the `try/except Exception` containment around the tick call site | `test_iv_tick_raising_does_not_skip_auto_save_and_warns` (1 failed, 25 passed) |

All four mutations are killed by at least one test — no mutation survived. `[z11 origin] neurograph_rpc.__file__` for each run pointed at `/tmp/z11_117_mut/mutN/neurograph_rpc.py`, confirming the mutant (not the worktree copy) was actually exercised.

## Real commons file — before/after

Unchanged across the entire correction turn (implementation + both full test-suite runs + all four mutation runs):
- size: `192773`
- mtime: `2026-09-29 17:38:16.949928157 -0800`
- sha256: `a4947c01f0e94d5eb125fc1a419d0982cb86a88421b838e2606c5ac87b15f319`

Matches the assignment's reference baseline exactly, identical to worker-001's measurement.

## Commit and push

- Commit: `018df3b` — "Correct no-advance alarm: dual-clock (counter + autonomous success), liveness, exception-safety (row #117)", on top of `6d99f09` (no amend, no rebase).
- `git push origin cc-laptop-no-advance-alarm-20260930` — plain push, fast-forward, succeeded.
- Closing check:
  ```
  ## cc-laptop-no-advance-alarm-20260930...origin/cc-laptop-no-advance-alarm-20260930
  ?? handoffs/
  018df3b Correct no-advance alarm: dual-clock (counter + autonomous success), liveness, exception-safety (row #117)
  6d99f09 Add no-advance alarm for frozen NG step counter (row #117)
  e4ebf98 gitignore: cover .venv/ alongside venv/
  fa93843 Merge pull request #58 from greatnorthernfishguy-hub/cc-laptop-ng-recall-passthrough-restore-20260926
  8c93149 docs: CC Stop door / one-step-per-turn operator note
  ```
  (`handoffs/` is the assignment/return scaffolding itself, not part of the two allowed source files.)

## Out of scope / noticed, not acted on

1. **`ng_salience_gate` resolved from the MAIN checkout, not this worktree, in every run** (the `[z11 origin]` END line flagged it every time, in both my runs and the neighbor-file runs). I did not investigate why `ng_salience_gate` specifically gets imported during this test run or whether its main-checkout version differs meaningfully from the worktree's — it is unrelated to anything the no-advance alarm touches, and per the correction's own scope I did not act on it. Flagging per the hazard #770 instruction to report discrepancies.
2. **The drain-pulse thread itself dying would still silence the ENTIRE watchdog**, carried over from worker-001's original return and still true after this correction: if `_scan_drain_pulse_loop`'s thread exits entirely, no ticks occur, so `last_tick_ts` stops advancing (C2's `tick_age_secs` in `handle_stats()` would correctly show this growing, once someone queries stats) but no NEW alarm transition (entry, re-emit, or recovery) would ever fire again, since `_no_advance_tick` simply never runs. C2 makes this observable via polling `handle_stats()`; it does not make it self-alerting. This is explicitly out of scope per the assignment (C2's own text: "the independent watcher is a separate row, NOT this turn").
3. I did not re-run `test_coordinator.py`, `test_tonic_bridge.py`, or anything touching `handle_after_turn` at any point, consistent with the hard safety rules.
4. I did not verify against the live VPS or laptop process — no service was started or restarted.
5. One thing I want to be plain about not having independently re-verified: the correction brief's claim that "`NG_NO_ADVANCE_REEMIT_SECS=0` produces an ERROR every 2s" on the ORIGINAL (worker-001) code — I did not go back and empirically reproduce that specific failure mode on the pre-correction code (I verified the ROOT CAUSE is real by reading worker-001's `elif (now - st["last_emit_ts"]) >= _NO_ADVANCE_REEMIT_SECS:` with `_NO_ADVANCE_REEMIT_SECS=0`, which is indeed always true once alarmed — logically confirmed, not empirically reproduced on the old code). Similarly for the `NG_NO_ADVANCE_FAILURE_WARN_EVERY=0` ZeroDivisionError claim — logically confirmed by reading `n % 0` in the old code, not empirically reproduced by running the unfixed version with that env var, since doing so would have meant running a version of the file I was actively replacing rather than a separate verification step. Both are now covered by passing tests against the CORRECTED code (confirming the fix works); I did not spend time additionally proving the bugs on the unfixed code beyond source inspection.
