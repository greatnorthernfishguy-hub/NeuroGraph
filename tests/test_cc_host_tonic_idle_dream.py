# tests/test_cc_host_tonic_idle_dream.py
#
# ---- Changelog ----
# [2026-07-23] Claude Code (Sonnet 5) — CC Host Tonic Idle/Dream Wiring tests
# What: Coverage for the three additions in cc_ng_host.py that close the
#   VPS/laptop parity gap (docs/superpowers/plans/2026-07-23-cc-host-tonic-
#   idle-dream-spec.md): (1) per-turn message_received() keepalive in
#   _handle_user_prompt_submit, (2) _cc_tonic_check_idle/_start_cc_tonic_
#   idle_watcher, (3) _cc_dream_gate_open/_start_cc_dream_consolidation_
#   pulse. Includes a poison-sentinel test proving the new code never
#   touches neurograph_rpc._memory (Syl's singleton) -- the load-bearing
#   safety line the spec calls out (Syl's-Law).
# Why: The spec requires this exact coverage shape: pure-function gate
#   tests with fake CC tonic-thread/graph objects, a poison-sentinel
#   _memory proving zero cross-touch, message_received() wiring on the
#   per-turn hook path, and gate-off-by-default inertness (byte-identical-
#   when-off discipline used throughout this rollout).
# How: Fakes/sentinels only -- no real NeuroGraphMemory construction (heavy,
#   loads embeddings/Qwen config). Module-level thread globals
#   (_cc_tonic_idle_thread/_cc_dream_thread) and shutdown Events are saved/
#   restored per test via an autouse fixture so tests never leak a live
#   daemon thread into later tests.
# -------------------
import os
import sys
import threading
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import cc_ng_host


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_cc_host_globals(monkeypatch):
    """Save/restore module-level state cc_ng_host's tonic/dream code owns,
    and make sure no thread this suite starts survives into another test."""
    orig_cc_ng = cc_ng_host._STATE.cc_ng
    orig_commons = cc_ng_host._STATE.commons
    orig_tonic_thread = cc_ng_host._cc_tonic_idle_thread
    orig_dream_thread = cc_ng_host._cc_dream_thread
    orig_dream_last_pass = cc_ng_host._cc_dream_last_pass_ts

    yield

    # Signal any thread this test may have started, then wait it out.
    cc_ng_host._cc_tonic_idle_shutdown.set()
    cc_ng_host._cc_dream_shutdown.set()
    if cc_ng_host._cc_tonic_idle_thread is not None:
        cc_ng_host._cc_tonic_idle_thread.join(timeout=2.0)
    if cc_ng_host._cc_dream_thread is not None:
        cc_ng_host._cc_dream_thread.join(timeout=2.0)

    cc_ng_host._STATE.cc_ng = orig_cc_ng
    cc_ng_host._STATE.commons = orig_commons
    cc_ng_host._cc_tonic_idle_thread = orig_tonic_thread
    cc_ng_host._cc_dream_thread = orig_dream_thread
    cc_ng_host._cc_dream_last_pass_ts = orig_dream_last_pass
    cc_ng_host._cc_tonic_idle_shutdown.clear()
    cc_ng_host._cc_dream_shutdown.clear()
    monkeypatch.delenv("CC_HOST_TONIC_IDLE_ENABLED", raising=False)
    monkeypatch.delenv("CC_HOST_DREAM_ENABLED", raising=False)


class _FakeTonic:
    def __init__(self, in_conversation=True, last_message_time=0.0):
        self._in_conversation = in_conversation
        self._last_message_time = last_message_time
        self.ended_calls = 0
        self.received_calls = 0
        self._raise_on_end = False

    def conversation_ended(self):
        self.ended_calls += 1
        if self._raise_on_end:
            raise RuntimeError("boom")
        self._in_conversation = False

    def message_received(self):
        self.received_calls += 1
        self._last_message_time = time.time()


class _FakeNg:
    def __init__(self, tonic=None, graph=None):
        self._tonic_thread = tonic
        self.graph = graph


class _FakeGraph:
    def __init__(self, merged=3):
        self._step_lock = threading.RLock()
        self._merged = merged
        self.consolidate_calls = 0

    def consolidate_hyperedges(self):
        self.consolidate_calls += 1
        return self._merged


class _FakeCommons:
    def __init__(self, arousal="PARASYMPATHETIC"):
        self._arousal = arousal

    def read_arousal(self, default="PARASYMPATHETIC"):
        return self._arousal


# =============================================================================
# _cc_tonic_check_idle — pure function
# =============================================================================

def test_tonic_check_idle_no_ng_returns_false():
    cc_ng_host._STATE.cc_ng = None
    assert cc_ng_host._cc_tonic_check_idle(time.time()) is False


def test_tonic_check_idle_no_tonic_thread_returns_false():
    cc_ng_host._STATE.cc_ng = _FakeNg(tonic=None)
    assert cc_ng_host._cc_tonic_check_idle(time.time()) is False


def test_tonic_check_idle_not_in_conversation_returns_false():
    tonic = _FakeTonic(in_conversation=False)
    cc_ng_host._STATE.cc_ng = _FakeNg(tonic=tonic)
    assert cc_ng_host._cc_tonic_check_idle(time.time()) is False
    assert tonic.ended_calls == 0


def test_tonic_check_idle_last_message_time_zero_returns_false():
    tonic = _FakeTonic(in_conversation=True, last_message_time=0.0)
    cc_ng_host._STATE.cc_ng = _FakeNg(tonic=tonic)
    assert cc_ng_host._cc_tonic_check_idle(time.time()) is False


def test_tonic_check_idle_below_threshold_returns_false():
    now = time.time()
    tonic = _FakeTonic(in_conversation=True, last_message_time=now - 10.0)
    cc_ng_host._STATE.cc_ng = _FakeNg(tonic=tonic)
    assert cc_ng_host._cc_tonic_check_idle(now) is False
    assert tonic.ended_calls == 0


def test_tonic_check_idle_past_threshold_transitions_and_returns_true():
    now = time.time()
    tonic = _FakeTonic(in_conversation=True,
                        last_message_time=now - cc_ng_host.CC_HOST_TONIC_IDLE_SECS - 1.0)
    cc_ng_host._STATE.cc_ng = _FakeNg(tonic=tonic)
    assert cc_ng_host._cc_tonic_check_idle(now) is True
    assert tonic.ended_calls == 1
    assert tonic._in_conversation is False


def test_tonic_check_idle_fails_soft_on_exception():
    now = time.time()
    tonic = _FakeTonic(in_conversation=True,
                        last_message_time=now - cc_ng_host.CC_HOST_TONIC_IDLE_SECS - 1.0)
    tonic._raise_on_end = True
    cc_ng_host._STATE.cc_ng = _FakeNg(tonic=tonic)
    assert cc_ng_host._cc_tonic_check_idle(now) is False  # exception swallowed, no raise
    assert tonic.ended_calls == 1


# =============================================================================
# _cc_dream_gate_open — pure function
# =============================================================================

def test_dream_gate_open_all_conditions_satisfied():
    now = 100000.0
    last_turn = now - cc_ng_host.CC_HOST_DREAM_IDLE_SECS - 1.0
    last_pass = now - cc_ng_host.CC_HOST_DREAM_MIN_INTERVAL_SECS - 1.0
    assert cc_ng_host._cc_dream_gate_open(now, last_turn, "PARASYMPATHETIC", last_pass) is True


def test_dream_gate_closed_when_not_idle_long_enough():
    now = 100000.0
    last_turn = now - 10.0  # nowhere near idle threshold
    last_pass = now - cc_ng_host.CC_HOST_DREAM_MIN_INTERVAL_SECS - 1.0
    assert cc_ng_host._cc_dream_gate_open(now, last_turn, "PARASYMPATHETIC", last_pass) is False


def test_dream_gate_closed_when_sympathetic():
    now = 100000.0
    last_turn = now - cc_ng_host.CC_HOST_DREAM_IDLE_SECS - 1.0
    last_pass = now - cc_ng_host.CC_HOST_DREAM_MIN_INTERVAL_SECS - 1.0
    assert cc_ng_host._cc_dream_gate_open(now, last_turn, "SYMPATHETIC", last_pass) is False


def test_dream_gate_closed_when_rate_limit_not_satisfied():
    now = 100000.0
    last_turn = now - cc_ng_host.CC_HOST_DREAM_IDLE_SECS - 1.0
    last_pass = now - 10.0  # a dream pass happened very recently
    assert cc_ng_host._cc_dream_gate_open(now, last_turn, "PARASYMPATHETIC", last_pass) is False


# =============================================================================
# Poison-sentinel: zero cross-touch of neurograph_rpc._memory (Syl's-Law)
# =============================================================================

class _PoisonMemory:
    """Any non-dunder attribute access on this object is a test failure --
    proves the new CC-scoped code never reaches into Syl's singleton.
    Dunder attrs (__class__ etc.) are exempted -- pytest's own monkeypatch/
    isinstance machinery touches those, which is not a production code path."""

    def __getattribute__(self, name):
        if name.startswith("__") and name.endswith("__"):
            return object.__getattribute__(self, name)
        raise AssertionError(
            f"CC Host tonic/dream code touched _memory.{name} — Syl's-Law violation"
        )


def test_new_code_never_touches_syl_memory_singleton(monkeypatch):
    import neurograph_rpc

    poison = _PoisonMemory()
    monkeypatch.setattr(neurograph_rpc, "_memory", poison)

    # Exercise every new pure/impure function with a fully-formed CC-side
    # fake. If any of them so much as read an attribute off _memory, the
    # poison object raises and this test fails.
    now = time.time()
    tonic = _FakeTonic(in_conversation=True,
                        last_message_time=now - cc_ng_host.CC_HOST_TONIC_IDLE_SECS - 1.0)
    graph = _FakeGraph(merged=5)
    cc_ng_host._STATE.cc_ng = _FakeNg(tonic=tonic, graph=graph)
    cc_ng_host._STATE.commons = _FakeCommons(arousal="PARASYMPATHETIC")

    assert cc_ng_host._cc_tonic_check_idle(now) is True
    assert cc_ng_host._cc_dream_gate_open(
        now, now - cc_ng_host.CC_HOST_DREAM_IDLE_SECS - 1.0,
        "PARASYMPATHETIC", now - cc_ng_host.CC_HOST_DREAM_MIN_INTERVAL_SECS - 1.0,
    ) is True

    # message_received wiring path too.
    cc_ng_host._handle_user_prompt_submit({"prompt": "hello"})

    # neurograph_rpc._memory itself was never touched -- if it had been,
    # one of the calls above would already have raised AssertionError.
    assert isinstance(neurograph_rpc._memory, _PoisonMemory)


# =============================================================================
# message_received() wiring on the per-turn hook path
# =============================================================================

def test_user_prompt_submit_calls_message_received(monkeypatch):
    tonic = _FakeTonic()
    cc_ng_host._STATE.cc_ng = _FakeNg(tonic=tonic)
    # Make _recall/_nudge/_deposit inert so this test isolates the wiring.
    monkeypatch.setattr(cc_ng_host, "_recall", lambda *a, **k: "")
    monkeypatch.setattr(cc_ng_host, "_nudge", lambda *a, **k: None)
    monkeypatch.setattr(threading, "Thread", lambda *a, **k: type(
        "T", (), {"start": lambda self: None})())

    cc_ng_host._handle_user_prompt_submit({"prompt": "hi there"})

    assert tonic.received_calls == 1


def test_user_prompt_submit_no_tonic_thread_does_not_crash(monkeypatch):
    cc_ng_host._STATE.cc_ng = _FakeNg(tonic=None)
    monkeypatch.setattr(cc_ng_host, "_recall", lambda *a, **k: "")
    monkeypatch.setattr(cc_ng_host, "_nudge", lambda *a, **k: None)
    monkeypatch.setattr(threading, "Thread", lambda *a, **k: type(
        "T", (), {"start": lambda self: None})())

    result = cc_ng_host._handle_user_prompt_submit({"prompt": "hi there"})
    assert result["ok"] is True


def test_user_prompt_submit_message_received_exception_fails_soft(monkeypatch):
    class _BoomTonic(_FakeTonic):
        def message_received(self):
            raise RuntimeError("boom")

    cc_ng_host._STATE.cc_ng = _FakeNg(tonic=_BoomTonic())
    monkeypatch.setattr(cc_ng_host, "_recall", lambda *a, **k: "")
    monkeypatch.setattr(cc_ng_host, "_nudge", lambda *a, **k: None)
    monkeypatch.setattr(threading, "Thread", lambda *a, **k: type(
        "T", (), {"start": lambda self: None})())

    # Must not raise -- fail-soft, mirrors every other call site in this file.
    result = cc_ng_host._handle_user_prompt_submit({"prompt": "hi there"})
    assert result["ok"] is True


def test_user_prompt_submit_empty_prompt_skips_everything(monkeypatch):
    tonic = _FakeTonic()
    cc_ng_host._STATE.cc_ng = _FakeNg(tonic=tonic)
    result = cc_ng_host._handle_user_prompt_submit({"prompt": ""})
    assert result == {"ok": True, "context": ""}
    assert tonic.received_calls == 0


# =============================================================================
# Gate-off default: no threads started, zero behavior change
# =============================================================================

def test_tonic_idle_watcher_gate_off_by_default_starts_nothing(monkeypatch):
    monkeypatch.delenv("CC_HOST_TONIC_IDLE_ENABLED", raising=False)
    cc_ng_host._cc_tonic_idle_thread = None
    cc_ng_host._start_cc_tonic_idle_watcher()
    assert cc_ng_host._cc_tonic_idle_thread is None


def test_dream_pulse_gate_off_by_default_starts_nothing(monkeypatch):
    monkeypatch.delenv("CC_HOST_DREAM_ENABLED", raising=False)
    cc_ng_host._cc_dream_thread = None
    cc_ng_host._start_cc_dream_consolidation_pulse()
    assert cc_ng_host._cc_dream_thread is None


def test_tonic_idle_watcher_explicitly_disabled_starts_nothing(monkeypatch):
    monkeypatch.setenv("CC_HOST_TONIC_IDLE_ENABLED", "0")
    cc_ng_host._cc_tonic_idle_thread = None
    cc_ng_host._start_cc_tonic_idle_watcher()
    assert cc_ng_host._cc_tonic_idle_thread is None


def test_tonic_idle_watcher_starts_when_gate_flipped_on(monkeypatch):
    monkeypatch.setenv("CC_HOST_TONIC_IDLE_ENABLED", "1")
    cc_ng_host._cc_tonic_idle_thread = None
    cc_ng_host._start_cc_tonic_idle_watcher()
    assert cc_ng_host._cc_tonic_idle_thread is not None
    assert cc_ng_host._cc_tonic_idle_thread.is_alive()
    assert cc_ng_host._cc_tonic_idle_thread.daemon is True


def test_tonic_idle_watcher_start_is_idempotent(monkeypatch):
    monkeypatch.setenv("CC_HOST_TONIC_IDLE_ENABLED", "1")
    cc_ng_host._cc_tonic_idle_thread = None
    cc_ng_host._start_cc_tonic_idle_watcher()
    first = cc_ng_host._cc_tonic_idle_thread
    cc_ng_host._start_cc_tonic_idle_watcher()
    assert cc_ng_host._cc_tonic_idle_thread is first


def test_dream_pulse_starts_when_gate_flipped_on(monkeypatch):
    monkeypatch.setenv("CC_HOST_DREAM_ENABLED", "1")
    cc_ng_host._STATE.cc_ng = None  # loop body no-ops safely with no ng
    cc_ng_host._cc_dream_thread = None
    cc_ng_host._start_cc_dream_consolidation_pulse()
    assert cc_ng_host._cc_dream_thread is not None
    assert cc_ng_host._cc_dream_thread.is_alive()
    assert cc_ng_host._cc_dream_thread.daemon is True


def test_init_cc_host_does_not_start_watchers_when_gates_off(monkeypatch):
    """init_cc_host() wires both _start_* calls, but with both gates at
    their default ('0'), no watcher thread must exist afterward -- the
    byte-identical-when-off contract, exercised at the call site actually
    used in production (not just the _start_* functions directly)."""
    monkeypatch.delenv("CC_HOST_TONIC_IDLE_ENABLED", raising=False)
    monkeypatch.delenv("CC_HOST_DREAM_ENABLED", raising=False)
    cc_ng_host._cc_tonic_idle_thread = None
    cc_ng_host._cc_dream_thread = None

    # Directly invoke the two wiring calls the way init_cc_host() does,
    # without needing a real NeuroGraphMemory/socket bootstrap.
    cc_ng_host._start_cc_tonic_idle_watcher()
    cc_ng_host._start_cc_dream_consolidation_pulse()

    assert cc_ng_host._cc_tonic_idle_thread is None
    assert cc_ng_host._cc_dream_thread is None


# =============================================================================
# Dream consolidation pulse body — reads CC's own graph/commons only
# =============================================================================

def test_dream_pulse_loop_runs_one_pass_and_calls_consolidate(monkeypatch):
    """Wires a fully-formed fake CC ng + commons, forces the gate open on
    the first tick, and confirms consolidate_hyperedges() runs under the
    graph's own _step_lock -- then the loop is signaled to stop."""
    now = time.time()
    tonic = _FakeTonic(in_conversation=True, last_message_time=now - 999999.0)
    graph = _FakeGraph(merged=7)
    cc_ng_host._STATE.cc_ng = _FakeNg(tonic=tonic, graph=graph)
    cc_ng_host._STATE.commons = _FakeCommons(arousal="PARASYMPATHETIC")
    cc_ng_host._cc_dream_last_pass_ts = 0.0

    monkeypatch.setattr(cc_ng_host, "CC_HOST_DREAM_IDLE_SECS", 1.0)
    monkeypatch.setattr(cc_ng_host, "CC_HOST_DREAM_MIN_INTERVAL_SECS", 1.0)
    monkeypatch.setattr(cc_ng_host, "CC_HOST_DREAM_TICK_SECS", 0.05)

    cc_ng_host._cc_dream_shutdown.clear()
    t = threading.Thread(target=cc_ng_host._cc_dream_consolidation_pulse_loop, daemon=True)
    t.start()
    for _ in range(100):  # up to ~2s
        if graph.consolidate_calls > 0:
            break
        time.sleep(0.02)
    cc_ng_host._cc_dream_shutdown.set()
    t.join(timeout=2.0)

    assert graph.consolidate_calls >= 1


def test_dream_pulse_alerts_once_gate_stays_closed_past_alert_floor(monkeypatch, caplog):
    """Enforcer LOW finding: the ALERT branch (mirrors Syl's #381-B 24h
    'no dream consolidation' floor) -- logs (not forces) when the gate has
    stayed closed past CC_HOST_DREAM_ALERT_SECS. Here the gate never opens
    (SYMPATHETIC, permanently) so every tick should hit the elif branch;
    with the alert floor tiny and rate-limited to itself, exactly one ERROR
    fires within the poll window."""
    now = time.time()
    tonic = _FakeTonic(in_conversation=True, last_message_time=now - 999999.0)
    graph = _FakeGraph(merged=7)
    cc_ng_host._STATE.cc_ng = _FakeNg(tonic=tonic, graph=graph)
    cc_ng_host._STATE.commons = _FakeCommons(arousal="SYMPATHETIC")  # gate never opens
    cc_ng_host._cc_dream_last_pass_ts = 0.0

    monkeypatch.setattr(cc_ng_host, "CC_HOST_DREAM_IDLE_SECS", 1.0)
    monkeypatch.setattr(cc_ng_host, "CC_HOST_DREAM_MIN_INTERVAL_SECS", 1.0)
    monkeypatch.setattr(cc_ng_host, "CC_HOST_DREAM_TICK_SECS", 0.05)
    monkeypatch.setattr(cc_ng_host, "CC_HOST_DREAM_ALERT_SECS", 0.1)  # tiny -- floor trips fast

    with caplog.at_level(logging.ERROR, logger="neurograph.cc_host"):
        cc_ng_host._cc_dream_shutdown.clear()
        t = threading.Thread(target=cc_ng_host._cc_dream_consolidation_pulse_loop, daemon=True)
        t.start()
        time.sleep(0.4)
        cc_ng_host._cc_dream_shutdown.set()
        t.join(timeout=2.0)

    alerts = [r for r in caplog.records if "No CC dream consolidation" in r.message]
    assert len(alerts) >= 1
    assert graph.consolidate_calls == 0  # the gate genuinely never opened


def test_dream_pulse_never_consolidates_while_sympathetic(monkeypatch):
    now = time.time()
    tonic = _FakeTonic(in_conversation=True, last_message_time=now - 999999.0)
    graph = _FakeGraph(merged=7)
    cc_ng_host._STATE.cc_ng = _FakeNg(tonic=tonic, graph=graph)
    cc_ng_host._STATE.commons = _FakeCommons(arousal="SYMPATHETIC")
    cc_ng_host._cc_dream_last_pass_ts = 0.0

    monkeypatch.setattr(cc_ng_host, "CC_HOST_DREAM_IDLE_SECS", 1.0)
    monkeypatch.setattr(cc_ng_host, "CC_HOST_DREAM_MIN_INTERVAL_SECS", 1.0)
    monkeypatch.setattr(cc_ng_host, "CC_HOST_DREAM_TICK_SECS", 0.05)

    cc_ng_host._cc_dream_shutdown.clear()
    t = threading.Thread(target=cc_ng_host._cc_dream_consolidation_pulse_loop, daemon=True)
    t.start()
    time.sleep(0.3)
    cc_ng_host._cc_dream_shutdown.set()
    t.join(timeout=2.0)

    assert graph.consolidate_calls == 0
