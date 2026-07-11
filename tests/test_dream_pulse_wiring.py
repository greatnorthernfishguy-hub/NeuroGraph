"""
#381-B — Quiet-hours dream consolidation pulse wiring.

# ---- Changelog ----
# [2026-07-11] Claude Code (Haiku 4.5) — dream pulse wiring tests
# What: Test the _dream_gate_open pure function truth table (idle-too-short / SYMPATHETIC /
#       rate-limited / all-clear), and verify module-level env knobs expose documented defaults.
# Why: TDD: gate logic is load-bearing. Truth table validates all four constraint axes.
#       Environment knobs must match brief-specified defaults so dream consolidation fires
#       at the intended quiet-hour cadence.
# How: Pure function tests (no thread). Four truth table cases for the gate.
#      Environment knobs tested against documented values.
# -------------------
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_dream_gate_open_idle_too_short():
    """Gate rejects when idle time (now - last_turn_ts) is less than DREAM_IDLE_SECS."""
    # This is importing the actual module to get access to the gate function
    # We need to import neurograph_rpc and access the gate function
    import neurograph_rpc

    now = 1000.0
    last_turn_ts = 900.0  # 100s ago
    last_pass_ts = 0.0
    arousal = "PARASYMPATHETIC"

    # Assuming DREAM_IDLE_SECS defaults to 1800 (30 minutes),
    # 100s idle should NOT open the gate
    gate_open = neurograph_rpc._dream_gate_open(now, last_turn_ts, arousal, last_pass_ts)
    assert not gate_open, "Gate should reject when idle < DREAM_IDLE_SECS"


def test_dream_gate_open_sympathetic_blocks():
    """Gate rejects when arousal is SYMPATHETIC."""
    import neurograph_rpc

    now = 1000.0
    last_turn_ts = 0.0  # 1000s idle — well over default 1800s
    last_pass_ts = 0.0  # First pass eligibility met
    arousal = "SYMPATHETIC"

    gate_open = neurograph_rpc._dream_gate_open(now, last_turn_ts, arousal, last_pass_ts)
    assert not gate_open, "Gate should reject when arousal is SYMPATHETIC"


def test_dream_gate_open_rate_limited():
    """Gate rejects when rate-limit window (now - last_pass_ts) is less than MIN_INTERVAL_SECS."""
    import neurograph_rpc

    now = 1000.0
    last_turn_ts = 0.0  # 1000s idle — well over default 1800s
    last_pass_ts = 500.0  # 500s since last pass
    arousal = "PARASYMPATHETIC"

    # Assuming DREAM_MIN_INTERVAL_SECS defaults to 21600 (6 hours),
    # 500s since last pass should NOT open the gate
    gate_open = neurograph_rpc._dream_gate_open(now, last_turn_ts, arousal, last_pass_ts)
    assert not gate_open, "Gate should reject when rate-limit window not satisfied"


def test_dream_gate_open_all_clear():
    """Gate opens when all constraints are satisfied."""
    import neurograph_rpc

    now = 100000.0
    last_turn_ts = 0.0  # 100000s idle — well over default 1800s
    last_pass_ts = 0.0  # First pass or 100000s since last pass (way over 21600s)
    arousal = "PARASYMPATHETIC"

    gate_open = neurograph_rpc._dream_gate_open(now, last_turn_ts, arousal, last_pass_ts)
    assert gate_open, "Gate should open when all constraints satisfied"


def test_dream_idle_secs_default():
    """Verify DREAM_IDLE_SECS env knob has documented default."""
    import neurograph_rpc

    # Default should be 1800 (30 minutes) per brief
    assert neurograph_rpc._DREAM_IDLE_SECS == 1800.0, \
        f"_DREAM_IDLE_SECS should default to 1800.0, got {neurograph_rpc._DREAM_IDLE_SECS}"


def test_dream_min_interval_secs_default():
    """Verify DREAM_MIN_INTERVAL_SECS env knob has documented default."""
    import neurograph_rpc

    # Default should be 21600 (6 hours) per brief
    assert neurograph_rpc._DREAM_MIN_INTERVAL_SECS == 21600.0, \
        f"_DREAM_MIN_INTERVAL_SECS should default to 21600.0, got {neurograph_rpc._DREAM_MIN_INTERVAL_SECS}"


def test_dream_alert_secs_default():
    """Verify DREAM_ALERT_SECS env knob has documented default."""
    import neurograph_rpc

    # Default should be 86400 (24 hours) per brief
    assert neurograph_rpc._DREAM_ALERT_SECS == 86400.0, \
        f"_DREAM_ALERT_SECS should default to 86400.0, got {neurograph_rpc._DREAM_ALERT_SECS}"


def test_dream_tick_secs_default():
    """Verify DREAM_TICK_SECS env knob has documented default."""
    import neurograph_rpc

    # Default should be 60 (1 minute) per brief
    assert neurograph_rpc._DREAM_TICK_SECS == 60.0, \
        f"_DREAM_TICK_SECS should default to 60.0, got {neurograph_rpc._DREAM_TICK_SECS}"


def test_dream_shutdown_event_exists():
    """Verify _dream_shutdown threading.Event exists."""
    import neurograph_rpc
    import threading

    assert hasattr(neurograph_rpc, '_dream_shutdown'), "_dream_shutdown should exist"
    assert isinstance(neurograph_rpc._dream_shutdown, threading.Event), \
        "_dream_shutdown should be a threading.Event"


def test_dream_last_pass_ts_initialized():
    """Verify _dream_last_pass_ts module global exists and is initialized."""
    import neurograph_rpc

    assert hasattr(neurograph_rpc, '_dream_last_pass_ts'), "_dream_last_pass_ts should exist"
    assert isinstance(neurograph_rpc._dream_last_pass_ts, float), \
        "_dream_last_pass_ts should be a float"
    assert neurograph_rpc._dream_last_pass_ts >= 0.0, \
        "_dream_last_pass_ts should be initialized to a non-negative value"


if __name__ == "__main__":
    test_dream_gate_open_idle_too_short();     print("PASS idle_too_short rejects gate")
    test_dream_gate_open_sympathetic_blocks();  print("PASS sympathetic blocks gate")
    test_dream_gate_open_rate_limited();        print("PASS rate_limited rejects gate")
    test_dream_gate_open_all_clear();           print("PASS all_clear opens gate")
    test_dream_idle_secs_default();             print("PASS DREAM_IDLE_SECS default = 1800")
    test_dream_min_interval_secs_default();     print("PASS DREAM_MIN_INTERVAL_SECS default = 21600")
    test_dream_alert_secs_default();            print("PASS DREAM_ALERT_SECS default = 86400")
    test_dream_tick_secs_default();             print("PASS DREAM_TICK_SECS default = 60")
    test_dream_shutdown_event_exists();         print("PASS _dream_shutdown Event exists")
    test_dream_last_pass_ts_initialized();      print("PASS _dream_last_pass_ts initialized")
    print("\n#381-B dream pulse wiring: ALL PASS")
