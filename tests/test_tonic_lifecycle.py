# tests/test_tonic_lifecycle.py
# Tests for the Tonic conversation<->latent lifecycle restore in neurograph_rpc.py.
# Never instantiates NeuroGraphMemory against live checkpoints.
#
# ---- Changelog ----
# [2026-06-07] CC (Opus 4.8) — Tonic lifecycle restore coverage
# What: Unit tests for _tonic_check_idle() — the idle->latent transition that
#       restores conversation_ended on Anima's 2-verb HTTP surface.
# Why:  The bug was _in_conversation pinned True forever (conversation_ended
#       lived only in the never-called handle_dispose). These lock in the fix.
# How:  Fake _tonic_thread + monkeypatched module globals; pure logic, no daemon.
# -------------------

import sys
import os
import time
import types
import unittest

_NG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _NG_DIR not in sys.path:
    sys.path.insert(0, _NG_DIR)

import neurograph_rpc as rpc


def _fake_thread(in_conversation=True, last_message_time=100.0):
    calls = {"ended": 0}
    t = types.SimpleNamespace()
    t._in_conversation = in_conversation
    t._last_message_time = last_message_time

    def conversation_ended():
        t._in_conversation = False
        calls["ended"] += 1

    t.conversation_ended = conversation_ended
    return t, calls


class TestTonicCheckIdle(unittest.TestCase):
    def setUp(self):
        self._saved_mem = rpc._memory
        self._saved_idle = rpc._TONIC_IDLE_SECS
        rpc._TONIC_IDLE_SECS = 90.0

    def tearDown(self):
        rpc._memory = self._saved_mem
        rpc._TONIC_IDLE_SECS = self._saved_idle

    def test_transitions_to_latent_after_threshold(self):
        t, calls = _fake_thread(in_conversation=True, last_message_time=100.0)
        rpc._memory = types.SimpleNamespace(_tonic_thread=t)
        self.assertTrue(rpc._tonic_check_idle(300.0))   # 200s quiet >= 90
        self.assertFalse(t._in_conversation)
        self.assertEqual(calls["ended"], 1)

    def test_noop_within_threshold(self):
        t, calls = _fake_thread(in_conversation=True, last_message_time=100.0)
        rpc._memory = types.SimpleNamespace(_tonic_thread=t)
        self.assertFalse(rpc._tonic_check_idle(150.0))  # only 50s quiet
        self.assertTrue(t._in_conversation)
        self.assertEqual(calls["ended"], 0)

    def test_noop_when_already_latent(self):
        t, calls = _fake_thread(in_conversation=False, last_message_time=100.0)
        rpc._memory = types.SimpleNamespace(_tonic_thread=t)
        self.assertFalse(rpc._tonic_check_idle(300.0))
        self.assertEqual(calls["ended"], 0)

    def test_safe_when_no_memory(self):
        rpc._memory = None
        self.assertFalse(rpc._tonic_check_idle(300.0))

    def test_safe_when_no_last_message_time(self):
        t, calls = _fake_thread(in_conversation=True, last_message_time=0.0)
        rpc._memory = types.SimpleNamespace(_tonic_thread=t)
        self.assertFalse(rpc._tonic_check_idle(300.0))  # never stamped -> no false drop
        self.assertEqual(calls["ended"], 0)


if __name__ == "__main__":
    unittest.main()
