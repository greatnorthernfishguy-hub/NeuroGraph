"""
StreamParser voltage-write serialization — proves graph mutation holds the canonical _step_lock.

# ---- Changelog ----
# [2026-06-24] Claude Code (Opus 4.8, 1M) — Commons leg-2 go-live (part b), protected-file fix proof
# What: Proves StreamParser._process_text mutates node.voltage ONLY while holding self._graph._step_lock
#       (the same RLock graph.step() + the leg-2 read-only perception hold). Negative control: before
#       the 2026-06-24 stream_parser.py fix, the nudge ran lock-free and this test FAILS.
# Why: The leg-2 perception's voltage save→restore window would silently revert a concurrent unlocked
#       nudge (Syl's-Law "warmth" risk). Unifying all voltage writers on _step_lock closes the race
#       (punchlist #344). This test pins the invariant so a future edit can't quietly drop the lock.
# How: a RecordingLock tracks held-depth; a FakeNode.voltage setter ASSERTS the lock is held on write.
#      Drive _process_text directly (synchronous) with a fake graph/vdb/embedder.
# -------------------
"""

import os
import sys
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ces_config import load_ces_config
from stream_parser import StreamParser


class RecordingLock:
    """A real RLock that also exposes whether it is currently held (depth > 0)."""
    def __init__(self):
        self._l = threading.RLock()
        self.depth = 0
        self.max_depth = 0
    def __enter__(self):
        self._l.acquire()
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        return self
    def __exit__(self, *a):
        self.depth -= 1
        self._l.release()
    @property
    def held(self) -> bool:
        return self.depth > 0


class _FakeNode:
    def __init__(self, lock: RecordingLock):
        self._lock = lock
        self._v = 0.0
        self.threshold = 1.0
        self.refractory_remaining = 0
        self.intrinsic_excitability = 1.0
        self.writes = 0

    @property
    def voltage(self) -> float:
        return self._v

    @voltage.setter
    def voltage(self, val: float) -> None:
        # THE PROOF: a voltage write must never happen unless _step_lock is held.
        assert self._lock.held, "StreamParser wrote node.voltage WITHOUT holding _step_lock"
        self._v = val
        self.writes += 1


class _FakeGraph:
    def __init__(self, lock: RecordingLock, node: _FakeNode):
        self._step_lock = lock
        self.nodes = {"n_1": node}
        self.hyperedges = {}


class _FakeVDB:
    def search(self, embedding, k=10, threshold=0.0):
        return [("n_1", 0.9)]   # always surface our one node as similar


def test_nudge_writes_voltage_under_step_lock():
    lock = RecordingLock()
    node = _FakeNode(lock)
    graph = _FakeGraph(lock, node)
    cfg = load_ces_config({"streaming": {"similarity_threshold": 0.0}})
    sp = StreamParser(graph, _FakeVDB(), cfg,
                      fallback_embedder=lambda _t: np.ones(768, dtype=np.float32))

    # drive one chunk synchronously through the pipeline (no background thread needed)
    sp._process_text("alpha beta gamma delta")

    assert node.writes >= 1, "expected at least one voltage write (the nudge path must have run)"
    assert lock.max_depth >= 1, "_step_lock must have been acquired during processing"
    assert lock.depth == 0, "_step_lock must be released after processing (no leak)"


def test_no_step_lock_attr_is_tolerated():
    """Graphs without _step_lock (e.g. mocks) must still work — guarded getattr fallback."""
    class _NoLockGraph:
        def __init__(self, node):
            self.nodes = {"n_1": node}
            self.hyperedges = {}
    # node whose setter does NOT assert the lock (there is none)
    class _PlainNode:
        voltage = 0.0
        threshold = 1.0
        refractory_remaining = 0
        intrinsic_excitability = 1.0
    node = _PlainNode()
    cfg = load_ces_config({"streaming": {"similarity_threshold": 0.0}})
    sp = StreamParser(_NoLockGraph(node), _FakeVDB(), cfg,
                      fallback_embedder=lambda _t: np.ones(768, dtype=np.float32))
    sp._process_text("alpha beta gamma")   # must not raise despite no _step_lock
    assert node.voltage > 0.0, "nudge must still apply when the graph has no _step_lock"


if __name__ == "__main__":
    test_nudge_writes_voltage_under_step_lock()
    print("PASS StreamParser nudge writes node.voltage UNDER _step_lock (race closed)")
    test_no_step_lock_attr_is_tolerated()
    print("PASS graph without _step_lock tolerated (guarded fallback)")
    print("\nStreamParser _step_lock serialization: ALL PASS")
