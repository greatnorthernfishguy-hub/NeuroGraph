# tests/test_no_advance_alarm_117.py
# ---- Changelog ----
# [2026-09-29] Claude Code (Sonnet 5.5) — no-advance alarm coverage (row #117)
# What: Unit tests for the no-advance watchdog added to neurograph_rpc.py:
#       _no_advance_tick, _no_advance_note_step_failure/_success, and the
#       additive handle_stats() "no_advance" block.
# Why:  Executive Packet 370 row #117 — both graphs' step counters were found
#       frozen with nothing reporting it. These tests pin the alarm-entry /
#       latch / slow-reemit / recovery / disable behavior and the previously-
#       silent step-failure visibility, and lock down that the watchdog is
#       read-only (its tick function is never handed a graph and never calls
#       step()).
# How:  Deterministic fake clock passed as `now` (no time.sleep, no threads,
#       no real graph). Module globals (_no_advance_state and the three env-
#       sourced tunables) are saved/restored per test, mirroring the pattern
#       tests/test_tonic_lifecycle.py already uses for this file's other
#       loop-state globals (_TONIC_IDLE_SECS et al).
# -------------------
import inspect
import os
import sys
import types
import unittest

_NG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _NG_DIR not in sys.path:
    sys.path.insert(0, _NG_DIR)

import neurograph_rpc as rpc


def _fresh_state():
    return {
        "last_timestep": None,
        "last_change_ts": None,
        "alarm": False,
        "last_emit_ts": 0.0,
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
    def test_a_advancing_counter_never_alarms(self):
        """(a) Would PASS on the unfixed base too (nothing existed to alarm) —
        pins the non-alarming path so a future change can't make it noisy."""
        for i, now in enumerate([0.0, 30.0, 65.0, 130.0, 200.0]):
            with self.assertNoLogs(rpc.logger, level="ERROR"):
                rpc._no_advance_tick(1000 + i, now, paused=False, paused_by=None)
        self.assertFalse(rpc._no_advance_state["alarm"])

    def test_b_frozen_past_threshold_fires_exactly_one_error_and_stats_correct(self):
        """(b) FAILS on the unfixed base: _no_advance_tick/_no_advance_state/
        _no_advance_stats_block do not exist there — AttributeError."""
        rpc._no_advance_tick(5000, 0.0, paused=False, paused_by=None)  # seed
        with self.assertNoLogs(rpc.logger, level="ERROR"):
            rpc._no_advance_tick(5000, 30.0, paused=False, paused_by=None)  # 30s < 60s threshold
        with self.assertLogs(rpc.logger, level="ERROR") as cm:
            rpc._no_advance_tick(5000, 61.0, paused=False, paused_by=None)  # crosses 60s
        self.assertEqual(len(cm.records), 1)
        self.assertTrue(rpc._no_advance_state["alarm"])

        block = rpc._no_advance_stats_block()
        self.assertTrue(block["alarm"])
        self.assertEqual(block["timestep"], 5000)
        self.assertGreaterEqual(block["frozen_secs"], 0.0)
        self.assertFalse(block["paused"])
        self.assertIsNone(block["paused_by"])
        self.assertEqual(block["consecutive_step_failures"], 0)
        self.assertIsNone(block["last_step_error"])
        self.assertEqual(block["threshold_secs"], 60.0)

    def test_c_still_frozen_does_not_repeat_per_tick_but_honors_slow_reemit(self):
        """(c) FAILS on the unfixed base: no latch exists, so there is nothing
        to assert "does not repeat" or "re-emits" about."""
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

    def test_d_frozen_while_paused_names_paused_by(self):
        """(d) FAILS on the unfixed base: paused_by is not tracked or logged
        anywhere there."""
        rpc._no_advance_tick(9, 0.0, paused=True, paused_by="sentinel")
        with self.assertLogs(rpc.logger, level="ERROR") as cm:
            rpc._no_advance_tick(9, 61.0, paused=True, paused_by="sentinel")
        self.assertIn("paused_by=sentinel", cm.records[0].getMessage())
        self.assertEqual(rpc._no_advance_state["paused_by"], "sentinel")

        block = rpc._no_advance_stats_block()
        self.assertTrue(block["paused"])
        self.assertEqual(block["paused_by"], "sentinel")

    def test_f_counter_advances_again_emits_one_recovery_info_and_clears_alarm(self):
        """(f) FAILS on the unfixed base: there is no alarm state to clear and
        no recovery log to emit."""
        rpc._no_advance_tick(11, 0.0, paused=False, paused_by=None)
        with self.assertLogs(rpc.logger, level="ERROR"):
            rpc._no_advance_tick(11, 61.0, paused=False, paused_by=None)
        self.assertTrue(rpc._no_advance_state["alarm"])

        with self.assertLogs(rpc.logger, level="INFO") as cm:
            rpc._no_advance_tick(12, 65.0, paused=False, paused_by=None)
        info_records = [r for r in cm.records if r.name == rpc.logger.name and r.levelname == "INFO"]
        self.assertEqual(len(info_records), 1)
        self.assertIn("resumed", info_records[0].getMessage().lower())
        self.assertFalse(rpc._no_advance_state["alarm"])

    def test_g_zero_threshold_disables(self):
        """(g) FAILS on the unfixed base: there is no threshold, env var, or
        disable path there at all."""
        rpc._NO_ADVANCE_ALARM_SECS = 0.0
        rpc._no_advance_tick(13, 0.0, paused=False, paused_by=None)
        with self.assertNoLogs(rpc.logger, level="ERROR"):
            rpc._no_advance_tick(13, 999999.0, paused=False, paused_by=None)
        self.assertFalse(rpc._no_advance_state["alarm"])

    def test_threshold_env_var_name_is_law5_compliant(self):
        """The threshold constant's source line actually reads NG_NO_ADVANCE_ALARM_SECS
        from the environment (LAW 5: env vars are the source of truth), not a bare
        hardcoded literal. Checked by source inspection rather than reloading the
        module in-process, to avoid re-executing top-level module code (and its
        singleton globals) mid-suite for a one-line assertion."""
        src = inspect.getsource(rpc)
        self.assertIn(
            'os.environ.get("NG_NO_ADVANCE_ALARM_SECS"',
            src,
        )


class TestNoAdvanceStepFailureVisibility(_NoAdvanceTestBase):
    def test_e_step_raises_every_tick_counts_and_warns_first_and_every_nth(self):
        """(e) FAILS on the unfixed base: the old code logged the failure at
        logger.debug only (silent at default level) and kept no counter."""
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

    def test_step_success_resets_consecutive_count_and_last_error(self):
        rpc._no_advance_note_step_failure(ValueError("x"))
        rpc._no_advance_note_step_failure(ValueError("x"))
        self.assertEqual(rpc._no_advance_state["consecutive_step_failures"], 2)
        rpc._no_advance_note_step_success()
        self.assertEqual(rpc._no_advance_state["consecutive_step_failures"], 0)
        self.assertIsNone(rpc._no_advance_state["last_step_error"])

    def test_e_thread_survival_behavior_unchanged(self):
        """Mirrors tests/test_scan_drain_guard.py's guard shape to confirm the edit
        that replaced logger.debug with _no_advance_note_step_failure() did not
        touch the BaseException/_LOOP_MUST_PROPAGATE guard around it: an ordinary
        step failure (or a Rust-panic-shaped BaseException) is still caught and the
        thread survives, while a real shutdown signal still propagates and kills it.
        """
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


class TestHandleStatsAdditive(_NoAdvanceTestBase):
    def test_h_handle_stats_carries_no_advance_without_dropping_existing_keys(self):
        """(h) FAILS on the unfixed base: handle_stats() has no "no_advance" key
        there at all, and _no_advance_stats_block doesn't exist to build one."""
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
            "alarm", "timestep", "frozen_secs", "paused", "paused_by",
            "consecutive_step_failures", "last_step_error", "threshold_secs",
        ):
            self.assertIn(key, block)


if __name__ == "__main__":
    unittest.main()
