# tests/test_no_advance_alarm_117.py
# ---- Changelog ----
# [2026-09-29] Claude Code (Sonnet 5.5) — no-advance alarm coverage (row #117)
# [2026-10-01] Claude Code (Sonnet 5.5) — CORRECTION (worker-002, dispatch #15002)
# What: Unit tests for the no-advance watchdog in neurograph_rpc.py, updated for
#       the C1-C5 correction: dual-clock alarm (raw counter OR autonomous-step
#       success, independently tracked), watchdog liveness (C2), exception-safety
#       and zero/negative-tunable semantics (C3), and — new in this pass — real
#       wiring tests that drive the ACTUAL _scan_drain_pulse_loop (C4).
# Why:  Two independent reviews (Grok-4.6 + neurograph-law-enforcer) found the
#       first cut's raw-counter-only alarm structurally blind to the exact bug it
#       was built to catch: neuro_foundation.py's Graph.step() increments
#       `timestep` as its FIRST statement, so a step failing partway still
#       advances the counter, and handle_after_turn() advances the SAME counter
#       on every conversational turn — so an autonomous step stuck or paused
#       while a conversation continues read as perfectly healthy. Tests a/b/f
#       encoded that old, incomplete semantics and are UPDATED below (not
#       deleted) to require a real autonomous success, not just a moving
#       counter; see each docstring for which and why.
# How:  Deterministic fake clock passed as `now` (no time.sleep, no real
#       threads). The new TestRealLoopWiring class drives the REAL
#       _scan_drain_pulse_loop() function synchronously for a scripted number of
#       iterations via a fake _scan_drain_shutdown (is_set()/wait() stubs) and a
#       fake _memory/graph, with time.time() patched and every non-watchdog
#       side-effecting call (_drain_scan_dir, _drain_peer_tracts,
#       _run_commons_enhance_scoop, _deposit_topology_to_river,
#       _deposit_substrate_metrics, the `commons` and `tid_peninsula_commons`
#       modules) stubbed out. Never reaches handle_after_turn; never touches a
#       real graph, real files, or the real Commons. Module globals are
#       saved/restored per test, mirroring tests/test_tonic_lifecycle.py's
#       pattern for this file's other loop-state globals.
# -------------------
import inspect
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

_NG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _NG_DIR not in sys.path:
    sys.path.insert(0, _NG_DIR)

import neurograph_rpc as rpc


def _fresh_state():
    return {
        "last_timestep": None,
        "last_change_ts": None,
        "last_autonomous_success_ts": None,
        "alarm": False,
        "kind": None,
        "last_emit_ts": 0.0,
        "last_tick_ts": None,
        "paused": False,
        "paused_by": None,
        "consecutive_step_failures": 0,
        "last_step_error": None,
    }


class _NoAdvanceTestBase(unittest.TestCase):
    """Common setUp/tearDown: isolate the watchdog's module-global state."""

    def setUp(self):
        self._saved_state = dict(rpc._no_advance_state)
        self._saved_threshold = rpc._NO_ADVANCE_ALARM_SECS
        self._saved_reemit = rpc._NO_ADVANCE_REEMIT_SECS
        self._saved_warn_every = rpc._NO_ADVANCE_FAILURE_WARN_EVERY
        self._saved_memory = rpc._memory
        rpc._no_advance_state.clear()
        rpc._no_advance_state.update(_fresh_state())
        rpc._NO_ADVANCE_ALARM_SECS = 60.0
        rpc._NO_ADVANCE_REEMIT_SECS = 900.0
        rpc._NO_ADVANCE_FAILURE_WARN_EVERY = 30

    def tearDown(self):
        rpc._no_advance_state.clear()
        rpc._no_advance_state.update(self._saved_state)
        rpc._NO_ADVANCE_ALARM_SECS = self._saved_threshold
        rpc._NO_ADVANCE_REEMIT_SECS = self._saved_reemit
        rpc._NO_ADVANCE_FAILURE_WARN_EVERY = self._saved_warn_every
        rpc._memory = self._saved_memory


class TestNoAdvanceTickAlarmLifecycle(_NoAdvanceTestBase):
    def test_a_advancing_counter_with_real_successes_never_alarms(self):
        """(a) UPDATED for C1: the old version only advanced the raw counter and
        asserted no alarm — under the corrected dual-clock semantics that is no
        longer sufficient (a counter can advance from a conversational step while
        the autonomous step is dead). A genuinely healthy process both advances
        the counter AND succeeds its autonomous step every tick, so this test now
        records a success at each tick before ticking."""
        for i, now in enumerate([0.0, 30.0, 65.0, 130.0, 200.0]):
            rpc._no_advance_note_step_success(now=now)
            with self.assertNoLogs(rpc.logger, level="ERROR"):
                rpc._no_advance_tick(1000 + i, now, paused=False, paused_by=None)
        self.assertFalse(rpc._no_advance_state["alarm"])
        self.assertIsNone(rpc._no_advance_state["kind"])

    def test_counter_advancing_without_autonomous_success_still_alarms(self):
        """NEW for C1 — this is the exact bug the correction fixes. The raw
        counter advances every tick (as it would from a conversational
        handle_after_turn step) but _no_advance_note_step_success is never
        called (the autonomous step is dead/paused). The old counter-only
        watchdog read this as healthy forever; the corrected one must alarm via
        kind="no_successful_autonomous_step"."""
        for i, now in enumerate([0.0, 30.0]):
            with self.assertNoLogs(rpc.logger, level="ERROR"):
                rpc._no_advance_tick(2000 + i, now, paused=False, paused_by=None)
        with self.assertLogs(rpc.logger, level="ERROR") as cm:
            rpc._no_advance_tick(2002, 61.0, paused=False, paused_by=None)  # counter STILL advancing
        self.assertEqual(len(cm.records), 1)
        self.assertIn("no_successful_autonomous_step", cm.records[0].getMessage())
        self.assertTrue(rpc._no_advance_state["alarm"])
        self.assertEqual(rpc._no_advance_state["kind"], "no_successful_autonomous_step")

    def test_b_frozen_past_threshold_fires_exactly_one_error_and_stats_correct(self):
        """(b) FAILS on the pre-worker-001 base: _no_advance_tick/_no_advance_state/
        _no_advance_stats_block do not exist there — AttributeError."""
        rpc._no_advance_tick(5000, 0.0, paused=False, paused_by=None)  # seed
        with self.assertNoLogs(rpc.logger, level="ERROR"):
            rpc._no_advance_tick(5000, 30.0, paused=False, paused_by=None)  # 30s < 60s threshold
        with self.assertLogs(rpc.logger, level="ERROR") as cm:
            rpc._no_advance_tick(5000, 61.0, paused=False, paused_by=None)  # crosses 60s
        self.assertEqual(len(cm.records), 1)
        self.assertTrue(rpc._no_advance_state["alarm"])
        # No success was ever recorded, so both clocks are stale together here.
        self.assertEqual(
            rpc._no_advance_state["kind"], "counter_frozen+no_successful_autonomous_step"
        )

        block = rpc._no_advance_stats_block()
        self.assertTrue(block["alarm"])
        self.assertEqual(block["timestep"], 5000)
        self.assertGreaterEqual(block["frozen_secs"], 0.0)
        self.assertGreaterEqual(block["autonomous_stale_secs"], 0.0)
        self.assertFalse(block["paused"])
        self.assertIsNone(block["paused_by"])
        self.assertEqual(block["consecutive_step_failures"], 0)
        self.assertIsNone(block["last_step_error"])
        self.assertEqual(block["threshold_secs"], 60.0)
        self.assertEqual(block["last_tick_ts"], 61.0)
        self.assertIsNotNone(block["tick_age_secs"])  # real-clock-derived; just must exist

    def test_c_still_frozen_does_not_repeat_per_tick_but_honors_slow_reemit(self):
        """(c) FAILS on the pre-worker-001 base: no latch exists, so there is
        nothing to assert "does not repeat" or "re-emits" about."""
        rpc._NO_ADVANCE_REEMIT_SECS = 100.0
        rpc._no_advance_tick(7, 0.0, paused=False, paused_by=None)
        with self.assertLogs(rpc.logger, level="ERROR") as cm:
            rpc._no_advance_tick(7, 61.0, paused=False, paused_by=None)  # alarm entry
        self.assertEqual(len(cm.records), 1)
        with self.assertNoLogs(rpc.logger, level="ERROR"):
            rpc._no_advance_tick(7, 65.0, paused=False, paused_by=None)   # still frozen, no repeat
            rpc._no_advance_tick(7, 100.0, paused=False, paused_by=None)  # still under reemit gap
        with self.assertLogs(rpc.logger, level="ERROR") as cm2:
            rpc._no_advance_tick(7, 162.0, paused=False, paused_by=None)  # >=100s since last emit
        self.assertEqual(len(cm2.records), 1)

    def test_reemit_zero_or_negative_means_never_reemit(self):
        """NEW — C3(c): NG_NO_ADVANCE_REEMIT_SECS<=0 must mean one ERROR per
        episode, not the base's bug (an ERROR every 2s, since the un-guarded
        `>=` against 0 is always true once alarmed)."""
        rpc._NO_ADVANCE_REEMIT_SECS = 0.0
        rpc._no_advance_tick(21, 0.0, paused=False, paused_by=None)
        with self.assertLogs(rpc.logger, level="ERROR") as cm:
            rpc._no_advance_tick(21, 61.0, paused=False, paused_by=None)
        self.assertEqual(len(cm.records), 1)
        with self.assertNoLogs(rpc.logger, level="ERROR"):
            rpc._no_advance_tick(21, 1_000_000.0, paused=False, paused_by=None)  # huge gap
        self.assertTrue(rpc._no_advance_state["alarm"])

    def test_d_frozen_while_paused_names_paused_by(self):
        """(d) FAILS on the pre-worker-001 base: paused_by is not tracked or
        logged anywhere there. The paused-tick ERROR is intentionally KEPT by
        C1 — a full-interval pause IS a frozen autonomic clock."""
        rpc._no_advance_tick(9, 0.0, paused=True, paused_by="sentinel")
        with self.assertLogs(rpc.logger, level="ERROR") as cm:
            rpc._no_advance_tick(9, 61.0, paused=True, paused_by="sentinel")
        self.assertIn("paused_by=sentinel", cm.records[0].getMessage())
        self.assertEqual(rpc._no_advance_state["paused_by"], "sentinel")

        block = rpc._no_advance_stats_block()
        self.assertTrue(block["paused"])
        self.assertEqual(block["paused_by"], "sentinel")

    def test_f_recovery_requires_both_conditions_to_clear(self):
        """(f) UPDATED for C1: the old version advanced the counter once and
        asserted immediate recovery. Under the corrected semantics a bare
        counter advance (e.g. from a conversational turn) must NOT by itself
        clear an alarm whose autonomous-step clock is still stale — recovery
        requires an actual _no_advance_note_step_success(). This is the direct
        behavioral proof of the C1 fix, not just the absence of a false
        positive."""
        rpc._no_advance_tick(11, 0.0, paused=False, paused_by=None)
        with self.assertLogs(rpc.logger, level="ERROR"):
            rpc._no_advance_tick(11, 61.0, paused=False, paused_by=None)
        self.assertTrue(rpc._no_advance_state["alarm"])

        # Counter advances (as handle_after_turn would do) but no autonomous
        # success is recorded -- must stay latched, now purely on the
        # autonomous-step condition, and must NOT emit a premature recovery.
        with self.assertNoLogs(rpc.logger, level="INFO"):
            rpc._no_advance_tick(12, 65.0, paused=False, paused_by=None)
        self.assertTrue(rpc._no_advance_state["alarm"])
        self.assertEqual(rpc._no_advance_state["kind"], "no_successful_autonomous_step")

        # Now a genuine autonomous success lands -- both conditions clear together.
        rpc._no_advance_note_step_success(now=66.0)
        with self.assertLogs(rpc.logger, level="INFO") as cm:
            rpc._no_advance_tick(13, 66.0, paused=False, paused_by=None)
        info_records = [r for r in cm.records if r.levelname == "INFO"]
        self.assertEqual(len(info_records), 1)
        self.assertIn("resumed", info_records[0].getMessage().lower())
        self.assertFalse(rpc._no_advance_state["alarm"])
        self.assertIsNone(rpc._no_advance_state["kind"])

    def test_g_zero_threshold_disables(self):
        """(g) FAILS on the pre-worker-001 base: there is no threshold, env var,
        or disable path there at all. Also confirms C2: liveness bookkeeping
        (last_tick_ts) still runs even when the alarm itself is disabled."""
        rpc._NO_ADVANCE_ALARM_SECS = 0.0
        rpc._no_advance_tick(13, 0.0, paused=False, paused_by=None)
        with self.assertNoLogs(rpc.logger, level="ERROR"):
            rpc._no_advance_tick(13, 999999.0, paused=False, paused_by=None)
        self.assertFalse(rpc._no_advance_state["alarm"])
        self.assertIsNone(rpc._no_advance_state["kind"])
        self.assertEqual(rpc._no_advance_state["last_tick_ts"], 999999.0)

    def test_threshold_env_var_name_is_law5_compliant(self):
        """The threshold constant's source line actually reads NG_NO_ADVANCE_ALARM_SECS
        from the environment (LAW 5: env vars are the source of truth), not a bare
        hardcoded literal. Checked by source inspection rather than reloading the
        module in-process, to avoid re-executing top-level module code (and its
        singleton globals) mid-suite for a one-line assertion."""
        src = inspect.getsource(rpc)
        self.assertIn('os.environ.get("NG_NO_ADVANCE_ALARM_SECS"', src)


class TestNoAdvanceStepFailureVisibility(_NoAdvanceTestBase):
    def test_e_step_raises_every_tick_counts_and_warns_first_and_every_nth(self):
        """(e) FAILS on the pre-worker-001 base: the old code logged the failure
        at logger.debug only (silent at default level) and kept no counter."""
        rpc._NO_ADVANCE_FAILURE_WARN_EVERY = 3
        exc = ValueError("boom")
        should_warn_at = {1, 3, 6}  # 1st, then every 3rd (n=3, n=6)
        for n in range(1, 8):
            if n in should_warn_at:
                with self.assertLogs(rpc.logger, level="WARNING") as cm:
                    rpc._no_advance_note_step_failure(exc)
                self.assertEqual(len(cm.records), 1)
                self.assertIn(str(n), cm.records[0].getMessage())
            else:
                with self.assertNoLogs(rpc.logger, level="WARNING"):
                    rpc._no_advance_note_step_failure(exc)
        self.assertEqual(rpc._no_advance_state["consecutive_step_failures"], 7)
        self.assertEqual(rpc._no_advance_state["last_step_error"], repr(exc))

    def test_failure_warn_every_zero_or_negative_means_warn_first_only_no_zerodiv(self):
        """NEW — C3(c): NG_NO_ADVANCE_FAILURE_WARN_EVERY<=0 must not
        ZeroDivisionError (the base bug: a bare `n % 0` inside the step()'s
        except handler, which would escape to the loop's outer guard and
        silently skip the scoop/tick/auto-save for that tick on every single
        failing tick) and must warn on the first failure only thereafter."""
        rpc._NO_ADVANCE_FAILURE_WARN_EVERY = 0
        with self.assertLogs(rpc.logger, level="WARNING") as cm:
            rpc._no_advance_note_step_failure(ValueError("1"))  # must not raise
        self.assertEqual(len(cm.records), 1)
        with self.assertNoLogs(rpc.logger, level="WARNING"):
            rpc._no_advance_note_step_failure(ValueError("2"))
            rpc._no_advance_note_step_failure(ValueError("3"))
        self.assertEqual(rpc._no_advance_state["consecutive_step_failures"], 3)

        rpc._no_advance_note_step_success()
        rpc._NO_ADVANCE_FAILURE_WARN_EVERY = -5
        with self.assertLogs(rpc.logger, level="WARNING") as cm2:
            rpc._no_advance_note_step_failure(ValueError("x"))  # negative must not raise either
        self.assertEqual(len(cm2.records), 1)

    def test_step_success_resets_consecutive_count_and_last_error(self):
        rpc._no_advance_note_step_failure(ValueError("x"))
        rpc._no_advance_note_step_failure(ValueError("x"))
        self.assertEqual(rpc._no_advance_state["consecutive_step_failures"], 2)
        rpc._no_advance_note_step_success()
        self.assertEqual(rpc._no_advance_state["consecutive_step_failures"], 0)
        self.assertIsNone(rpc._no_advance_state["last_step_error"])

    def test_step_success_now_is_injectable_and_sets_autonomous_clock(self):
        """NEW — C1: _no_advance_note_step_success must accept an injectable
        `now` (tests use a fake clock) and record it as the autonomous-step
        success time."""
        rpc._no_advance_note_step_success(now=12345.0)
        self.assertEqual(rpc._no_advance_state["last_autonomous_success_ts"], 12345.0)

    def test_note_step_failure_never_raises_even_if_state_write_fails(self):
        """NEW — C3(b): _no_advance_note_step_failure runs inside the step()'s
        own except handler and must never raise into it, regardless of cause."""
        class _ExplodingDict(dict):
            def __setitem__(self, key, value):
                raise RuntimeError("simulated internal failure")

        saved = rpc._no_advance_state
        rpc._no_advance_state = _ExplodingDict(saved)
        try:
            with self.assertLogs(rpc.logger, level="WARNING"):
                rpc._no_advance_note_step_failure(ValueError("x"))  # must not raise
        finally:
            rpc._no_advance_state = saved

    def test_note_step_success_never_raises_even_if_state_write_fails(self):
        """NEW — C3(b): same exception-safety guarantee for the success path —
        a raise here would otherwise be caught by the step()'s own try's
        `except BaseException` and mis-logged as a step failure."""
        class _ExplodingDict(dict):
            def __setitem__(self, key, value):
                raise RuntimeError("simulated internal failure")

        saved = rpc._no_advance_state
        rpc._no_advance_state = _ExplodingDict(saved)
        try:
            with self.assertLogs(rpc.logger, level="WARNING"):
                rpc._no_advance_note_step_success(now=1.0)  # must not raise
        finally:
            rpc._no_advance_state = saved

    def test_e_thread_survival_behavior_unchanged(self):
        """Mirrors tests/test_scan_drain_guard.py's guard shape to confirm the
        edit that replaced logger.debug with _no_advance_note_step_failure()
        did not touch the BaseException/_LOOP_MUST_PROPAGATE guard around it: an
        ordinary step failure (or a Rust-panic-shaped BaseException) is still
        caught and the thread survives, while a real shutdown signal still
        propagates and kills it."""
        class _FakePanic(BaseException):
            pass

        def guarded_body(raiser):
            try:
                raiser()
            except BaseException as exc:  # noqa: BLE001 - mirrors the widened guard
                if isinstance(exc, rpc._LOOP_MUST_PROPAGATE):
                    raise
                rpc._no_advance_note_step_failure(exc)
                return type(exc).__name__
            return None

        def boom_panic():
            raise _FakePanic("Already borrowed: PyBorrowMutError")

        def boom_ordinary():
            raise ValueError("ordinary")

        with self.assertLogs(rpc.logger, level="WARNING"):
            self.assertEqual(guarded_body(boom_panic), "_FakePanic")
        with self.assertNoLogs(rpc.logger, level="WARNING"):
            # already at consecutive=2, not the 1st and not a multiple of 30 -> no re-warn
            self.assertEqual(guarded_body(boom_ordinary), "ValueError")
        self.assertEqual(rpc._no_advance_state["consecutive_step_failures"], 2)

        for sig in (KeyboardInterrupt, SystemExit, GeneratorExit):
            def boom_signal(s=sig):
                raise s()
            with self.assertRaises(sig):
                guarded_body(boom_signal)

    def test_watchdog_is_structurally_read_only_never_touches_step(self):
        """(4) Read-only guarantee, checked structurally rather than by absence-of-
        evidence in one test run: none of these functions' compiled bytecode
        references a name called "step" anywhere (co_names covers every global
        lookup and attribute access the function body performs) — so there is no
        code path by which any of them could call graph.step(), regardless of what
        arguments they are given. _no_advance_tick's signature also carries no
        graph reference at all (a bare timestep value), which is why this holds."""
        for fn in (
            rpc._no_advance_tick,
            rpc._no_advance_note_step_failure,
            rpc._no_advance_note_step_success,
            rpc._no_advance_stats_block,
        ):
            self.assertNotIn("step", fn.__code__.co_names)

        sig = inspect.signature(rpc._no_advance_tick)
        self.assertEqual(list(sig.parameters), ["timestep", "now", "paused", "paused_by"])


class TestNoAdvanceWatchdogLiveness(_NoAdvanceTestBase):
    def test_c2_stats_shows_last_tick_ts_and_tick_age_secs(self):
        """NEW — C2: a dead or never-started watchdog must be visible in
        handle_stats()'s existing endpoint, which previously read as a clean
        "timestep=None, frozen_secs=0.0, alarm=False" (looks healthy, isn't)."""
        # Never ticked: tick_age_secs must be None, not 0 -- "never observed" is
        # a distinct, more alarming state than "just observed, all clear".
        block = rpc._no_advance_stats_block()
        self.assertIsNone(block["last_tick_ts"])
        self.assertIsNone(block["tick_age_secs"])

        rpc._no_advance_tick(50, 100.0, paused=False, paused_by=None)
        block2 = rpc._no_advance_stats_block()
        self.assertEqual(block2["last_tick_ts"], 100.0)
        self.assertIsNotNone(block2["tick_age_secs"])
        self.assertGreaterEqual(block2["tick_age_secs"], 0.0)


class TestHandleStatsAdditive(_NoAdvanceTestBase):
    def test_h_handle_stats_carries_no_advance_without_dropping_existing_keys(self):
        """(h) FAILS on the pre-worker-001 base: handle_stats() has no
        "no_advance" key there at all. Key list UPDATED for C1/C2's additive
        "kind", "autonomous_stale_secs", "last_tick_ts", "tick_age_secs"."""
        fake_memory = types.SimpleNamespace(
            stats=lambda: {"nodes": 42, "synapses": 7},
            graph=types.SimpleNamespace(timestep=100),
        )
        rpc._memory = fake_memory
        result = rpc.handle_stats({})

        # Existing keys untouched (additive-only).
        self.assertEqual(result["nodes"], 42)
        self.assertEqual(result["synapses"], 7)
        self.assertIn("module_hooks", result)

        self.assertIn("no_advance", result)
        block = result["no_advance"]
        for key in (
            "alarm", "kind", "timestep", "frozen_secs", "autonomous_stale_secs",
            "paused", "paused_by", "consecutive_step_failures", "last_step_error",
            "threshold_secs", "last_tick_ts", "tick_age_secs",
        ):
            self.assertIn(key, block)


# ---------------------------------------------------------------------------
# C4 — real wiring tests: drive the ACTUAL _scan_drain_pulse_loop synchronously.
# ---------------------------------------------------------------------------

class _FakeShutdown:
    """Stands in for _scan_drain_shutdown: is_set() returns False for exactly
    `iterations` calls, then True (stopping the real while-loop deterministically
    after a known number of passes). wait() returns immediately -- no real sleep.
    `on_check`, if given, is invoked on every is_set() call (including the final
    True one) so a test can inject state changes "between pulses" (e.g. a
    conversational step landing, as handle_after_turn would do)."""

    def __init__(self, iterations, on_check=None):
        self._remaining = iterations
        self.wait_calls = 0
        self._on_check = on_check

    def is_set(self):
        if self._on_check is not None:
            self._on_check()
        if self._remaining > 0:
            self._remaining -= 1
            return False
        return True

    def wait(self, timeout=None):
        self.wait_calls += 1
        return False


def _make_clock(start=0.0):
    """A time.time() stand-in whose value the TEST advances explicitly via
    .set(), rather than one driven by call count. A fixed-length side_effect
    list is NOT safe here: logging.LogRecord.__init__ calls time.time() too
    (once per emitted log record, for the record's timestamp), so the number of
    real time.time() calls per loop iteration varies with how many WARNING/
    ERROR/INFO lines that iteration happens to emit -- confirmed empirically
    (an iteration with a first-failure WARNING consumes one extra call beyond
    the loop's own single _tick_now read). This clock returns the current value
    for ANY number of calls, decoupling "how many log lines fired" from "what
    time it is", and tests drive it forward via _FakeShutdown's on_check hook.
    """
    state = {"now": start}

    def _time():
        return state["now"]

    _time.set = lambda v: state.__setitem__("now", v)
    return _time


class _FakeGraph:
    """Settable timestep; step() outcomes are scripted, one popped per call
    ("ok" default if the script is exhausted). Mirrors neuro_foundation.py's
    Graph.step(), which does `self.timestep += 1` as its FIRST statement under
    its lock: timestep is incremented UNCONDITIONALLY, before a "raise" outcome
    is applied -- this is the exact fact C1 corrects the watchdog for, so the
    fake must reproduce it faithfully rather than assume a step only advances
    the counter on success."""

    def __init__(self, timestep=0, outcomes=None):
        self.timestep = timestep
        self._outcomes = list(outcomes) if outcomes else []
        self.step_calls = 0

    def step(self):
        self.step_calls += 1
        outcome = self._outcomes.pop(0) if self._outcomes else "ok"
        self.timestep += 1
        if outcome == "raise":
            raise RuntimeError("fake autonomous step failure")
        return types.SimpleNamespace(timestep=self.timestep)


class _FakeMemory:
    def __init__(self, graph):
        self.graph = graph
        self.save_calls = 0

    def save(self):
        self.save_calls += 1


class _RealLoopTestBase(unittest.TestCase):
    """Drives the REAL _scan_drain_pulse_loop for a scripted number of
    iterations with every non-watchdog side effect stubbed out. Never reaches
    handle_after_turn; never touches a real graph, real files, or the real
    Commons/tid_peninsula_commons modules (both are replaced in sys.modules for
    the duration of the test)."""

    def setUp(self):
        self._saved_memory = rpc._memory
        self._saved_shutdown = rpc._scan_drain_shutdown
        self._saved_pause_file = rpc._SCAN_DRAIN_PAUSE_FILE
        self._saved_drain_scan_dir = rpc._drain_scan_dir
        self._saved_drain_peer_tracts = rpc._drain_peer_tracts
        self._saved_scoop = rpc._run_commons_enhance_scoop
        self._saved_deposit_topo = rpc._deposit_topology_to_river
        self._saved_deposit_metrics = rpc._deposit_substrate_metrics
        self._saved_last_save_time = rpc._last_save_time
        self._saved_tick = rpc._no_advance_tick
        self._saved_state = dict(rpc._no_advance_state)
        self._saved_threshold = rpc._NO_ADVANCE_ALARM_SECS
        self._saved_reemit = rpc._NO_ADVANCE_REEMIT_SECS
        self._saved_warn_every = rpc._NO_ADVANCE_FAILURE_WARN_EVERY
        self._saved_commons_mod = sys.modules.get("commons")
        self._saved_tid_mod = sys.modules.get("tid_peninsula_commons")

        sys.modules["commons"] = types.SimpleNamespace(get_commons=lambda: None)
        sys.modules["tid_peninsula_commons"] = types.SimpleNamespace(
            tid_peninsula_push_enhanced=lambda: None
        )
        rpc._SCAN_DRAIN_PAUSE_FILE = "/tmp/z11_117_nonexistent_pause_sentinel"
        rpc._drain_scan_dir = lambda: None
        rpc._drain_peer_tracts = lambda: None
        rpc._run_commons_enhance_scoop = lambda: None
        rpc._deposit_topology_to_river = lambda step_result: None
        rpc._deposit_substrate_metrics = lambda step_result, to_jsonl=True: None
        rpc._no_advance_state.clear()
        rpc._no_advance_state.update(_fresh_state())
        rpc._NO_ADVANCE_ALARM_SECS = 60.0
        rpc._NO_ADVANCE_REEMIT_SECS = 900.0
        rpc._NO_ADVANCE_FAILURE_WARN_EVERY = 30
        rpc._last_save_time = 0.0

    def tearDown(self):
        rpc._memory = self._saved_memory
        rpc._scan_drain_shutdown = self._saved_shutdown
        rpc._SCAN_DRAIN_PAUSE_FILE = self._saved_pause_file
        rpc._drain_scan_dir = self._saved_drain_scan_dir
        rpc._drain_peer_tracts = self._saved_drain_peer_tracts
        rpc._run_commons_enhance_scoop = self._saved_scoop
        rpc._deposit_topology_to_river = self._saved_deposit_topo
        rpc._deposit_substrate_metrics = self._saved_deposit_metrics
        rpc._last_save_time = self._saved_last_save_time
        rpc._no_advance_tick = self._saved_tick
        rpc._no_advance_state.clear()
        rpc._no_advance_state.update(self._saved_state)
        rpc._NO_ADVANCE_ALARM_SECS = self._saved_threshold
        rpc._NO_ADVANCE_REEMIT_SECS = self._saved_reemit
        rpc._NO_ADVANCE_FAILURE_WARN_EVERY = self._saved_warn_every
        if self._saved_commons_mod is not None:
            sys.modules["commons"] = self._saved_commons_mod
        else:
            sys.modules.pop("commons", None)
        if self._saved_tid_mod is not None:
            sys.modules["tid_peninsula_commons"] = self._saved_tid_mod
        else:
            sys.modules.pop("tid_peninsula_commons", None)


class TestRealLoopWiring(_RealLoopTestBase):
    def test_i_failing_step_through_real_loop_increments_failure_count(self):
        """(i) Drives the REAL loop for one iteration with a step that raises."""
        graph = _FakeGraph(timestep=5, outcomes=["raise"])
        rpc._memory = _FakeMemory(graph)
        rpc._scan_drain_shutdown = _FakeShutdown(1)
        with mock.patch.object(rpc.time, "time", return_value=1000.0):
            rpc._scan_drain_pulse_loop()
        self.assertEqual(graph.step_calls, 1)
        self.assertEqual(graph.timestep, 6)  # incremented despite the raise (neuro_foundation.py fidelity)
        self.assertEqual(rpc._no_advance_state["consecutive_step_failures"], 1)
        self.assertIsNotNone(rpc._no_advance_state["last_step_error"])

    def test_ii_succeeding_step_through_real_loop_resets_and_sets_autonomous_ts(self):
        """(ii) A failure then a success, through the real loop: the success
        resets the failure counter AND stamps the autonomous-success clock with
        the real call site's `now` (proving the single-clock-per-iteration wiring
        feeds _no_advance_note_step_success correctly)."""
        graph = _FakeGraph(timestep=5, outcomes=["raise", "ok"])
        rpc._memory = _FakeMemory(graph)
        rpc._no_advance_state["consecutive_step_failures"] = 3  # pretend prior failures
        clock = _make_clock(1000.0)
        calls = {"n": 0}

        def _on_check():
            calls["n"] += 1
            if calls["n"] == 2:
                clock.set(1002.0)

        rpc._scan_drain_shutdown = _FakeShutdown(2, on_check=_on_check)
        with mock.patch.object(rpc.time, "time", side_effect=clock):
            rpc._scan_drain_pulse_loop()
        self.assertEqual(rpc._no_advance_state["consecutive_step_failures"], 0)
        self.assertEqual(rpc._no_advance_state["last_autonomous_success_ts"], 1002.0)

    def test_iii_tick_runs_from_real_loop_and_errors_after_threshold_while_paused(self):
        """(iii) Covers both "the tick runs from the real loop and an ERROR
        appears after the fake clock passes the threshold" and "including on a
        PAUSED iteration" in one proof: while paused, graph.step() is never
        called by the real loop's own control flow, so the raw counter is
        genuinely frozen (not simulated) and the watchdog must still observe
        and alarm on it."""
        pause_file = tempfile.NamedTemporaryFile(delete=False)
        pause_file.close()
        try:
            rpc._SCAN_DRAIN_PAUSE_FILE = pause_file.name  # sentinel EXISTS -> paused=True
            graph = _FakeGraph(timestep=42)
            rpc._memory = _FakeMemory(graph)
            clock = _make_clock(0.0)
            calls = {"n": 0}

            def _on_check():
                calls["n"] += 1
                if calls["n"] == 2:
                    clock.set(61.0)

            rpc._scan_drain_shutdown = _FakeShutdown(2, on_check=_on_check)
            with mock.patch.object(rpc.time, "time", side_effect=clock):
                rpc._scan_drain_pulse_loop()
            self.assertEqual(graph.step_calls, 0)   # never stepped: genuinely paused
            self.assertEqual(graph.timestep, 42)    # genuinely frozen
            self.assertTrue(rpc._no_advance_state["alarm"])
            self.assertTrue(rpc._no_advance_state["paused"])
            self.assertEqual(rpc._no_advance_state["paused_by"], "sentinel")
        finally:
            os.unlink(pause_file.name)

    def test_iv_tick_raising_does_not_skip_auto_save_and_warns(self):
        """(iv) C3(a) proof through the real loop: patch _no_advance_tick to
        raise and confirm the auto-save still runs (the save interval has
        elapsed) and a WARNING is logged -- the watchdog's own try/except (added
        around the real call site in _scan_drain_pulse_loop) must contain the
        failure rather than let it starve auto-save."""
        graph = _FakeGraph(timestep=5, outcomes=["ok"])
        memory = _FakeMemory(graph)
        rpc._memory = memory
        rpc._scan_drain_shutdown = _FakeShutdown(1)
        rpc._last_save_time = 0.0  # force the save interval to have elapsed
        rpc._no_advance_tick = mock.Mock(side_effect=RuntimeError("tick boom"))
        with mock.patch.object(rpc.time, "time", return_value=10_000.0):
            with self.assertLogs(rpc.logger, level="WARNING") as cm:
                rpc._scan_drain_pulse_loop()  # must not raise
        self.assertTrue(
            any("watchdog tick failed" in r.getMessage().lower() for r in cm.records)
        )
        self.assertEqual(memory.save_calls, 1)  # auto-save was NOT starved

    def test_v_zero_tunables_do_not_raise_or_skip_auto_save(self):
        """(v) C3(c) proof through the real loop: both zero-valued tunables must
        not raise (no ZeroDivisionError from the base bug) and must not starve
        auto-save, across repeatedly failing ticks."""
        rpc._NO_ADVANCE_FAILURE_WARN_EVERY = 0
        rpc._NO_ADVANCE_REEMIT_SECS = 0.0
        graph = _FakeGraph(timestep=5, outcomes=["raise", "raise"])
        memory = _FakeMemory(graph)
        rpc._memory = memory
        clock = _make_clock(1000.0)
        calls = {"n": 0}

        def _on_check():
            calls["n"] += 1
            if calls["n"] == 2:
                clock.set(1361.0)

        rpc._scan_drain_shutdown = _FakeShutdown(2, on_check=_on_check)
        rpc._last_save_time = 0.0
        with mock.patch.object(rpc.time, "time", side_effect=clock):
            rpc._scan_drain_pulse_loop()  # must not raise
        self.assertEqual(rpc._no_advance_state["consecutive_step_failures"], 2)
        self.assertGreaterEqual(memory.save_calls, 1)

    def test_vi_step_raising_every_tick_with_counter_advancing_still_alarms(self):
        """(vi, first half) The exact C1 bug, proven through the real loop:
        _FakeGraph mirrors neuro_foundation.py by incrementing `timestep` BEFORE
        a step can raise, so the raw counter advances every tick even though
        every autonomous step fails. The old counter-only watchdog would read
        this as healthy forever; the corrected one must alarm via the
        autonomous-step clock."""
        graph = _FakeGraph(timestep=5, outcomes=["raise", "raise", "raise"])
        rpc._memory = _FakeMemory(graph)
        clock = _make_clock(0.0)
        calls = {"n": 0}

        def _on_check():
            calls["n"] += 1
            if calls["n"] == 2:
                clock.set(30.0)
            elif calls["n"] == 3:
                clock.set(61.0)

        rpc._scan_drain_shutdown = _FakeShutdown(3, on_check=_on_check)
        with mock.patch.object(rpc.time, "time", side_effect=clock):
            rpc._scan_drain_pulse_loop()
        self.assertEqual(graph.timestep, 8)  # counter DID advance, every tick
        self.assertTrue(rpc._no_advance_state["alarm"])
        self.assertIn("no_successful_autonomous_step", rpc._no_advance_state["kind"])

    def test_vi_conversation_style_external_timestep_advance_without_autonomous_success_alarms(self):
        """(vi, second half) Simulates handle_after_turn advancing the SAME
        raw counter from a conversational turn while the autonomous step is
        paused (never even attempted, let alone succeeded) -- the exact LAW 8
        scenario this alarm exists for. The "conversational" advance is injected
        between pulse iterations via the fake shutdown's on_check hook."""
        pause_file = tempfile.NamedTemporaryFile(delete=False)
        pause_file.close()
        try:
            rpc._SCAN_DRAIN_PAUSE_FILE = pause_file.name  # paused: autonomous step never attempted
            graph = _FakeGraph(timestep=100)
            rpc._memory = _FakeMemory(graph)
            clock = _make_clock(0.0)
            calls = {"n": 0}

            def _on_check():
                calls["n"] += 1
                if calls["n"] == 2:
                    clock.set(61.0)
                    graph.timestep += 1  # simulates handle_after_turn's conversational step()

            rpc._scan_drain_shutdown = _FakeShutdown(2, on_check=_on_check)
            with mock.patch.object(rpc.time, "time", side_effect=clock):
                rpc._scan_drain_pulse_loop()
            self.assertEqual(graph.step_calls, 0)   # autonomous step never attempted (paused)
            self.assertEqual(graph.timestep, 101)   # yet the raw counter DID move
            self.assertTrue(rpc._no_advance_state["alarm"])
            self.assertEqual(rpc._no_advance_state["kind"], "no_successful_autonomous_step")
        finally:
            os.unlink(pause_file.name)


if __name__ == "__main__":
    unittest.main()
