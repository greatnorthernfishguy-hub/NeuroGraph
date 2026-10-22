"""#1160 — the unified Tonic releases _step_lock between its real steps (one step() per hold).

Design: dev-log 2026-10-21_tonic-unified-real-step §5 / §7 (#1160: waves made one drive hold the lock 10-25 s);
Josh 2026-10-22: "Let's go ahead with your recommendations, DudeMan. Good plan."

# ---- Changelog ----
# [2026-10-22] Claude Code (Opus 5.5, laptop; lane tonic-lock-split) — initial
# What: one hold per real step, the prime in the same hold as the step that consumes it; a waiting thread gets in
#       between the Tonic's steps (not behind the whole drive); an interleaving stepper loses / doubles no delayed spike
#       and the clock counts every step once; the yield key reads live; single-threaded the result is identical to the
#       one-hold drive of 7a70748 (scenario digest); off (key absent / False) never reaches this code.
# Why:  the #1160 build brief's correctness list. How: sandbox Graphs only (never a live singleton, never a checkpoint).
# -------------------
"""
import os
import subprocess
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from test_tonic_unified import _graph, _warm, _thread, _engine  # noqa: E402
from tonic_thread import tonic_unified_drive, new_unified_stats  # noqa: E402


class _CountingLock:
    """Wraps the graph's RLock: counts outermost acquisitions (holds) so a test can see which hold a call ran in."""

    def __init__(self, inner):
        self._inner = inner
        self.holds = 0
        self._depth = threading.local()

    def acquire(self, *a, **k):
        ok = self._inner.acquire(*a, **k)
        if ok:
            d = getattr(self._depth, "d", 0)
            if d == 0 and threading.current_thread() is threading.main_thread():
                self.holds += 1
            self._depth.d = d + 1
        return ok

    def release(self):
        self._depth.d = getattr(self._depth, "d", 1) - 1
        self._inner.release()

    __enter__ = acquire

    def __exit__(self, *exc):
        self.release()
        return False


def _instrument(g):
    lock = _CountingLock(g._step_lock)
    g._step_lock = lock
    calls = []
    real_step, real_stim = g.step, g.stimulate

    def step(*a, **k):
        calls.append(("step", lock.holds))
        return real_step(*a, **k)

    def stim(nid, cur):
        calls.append(("stim", lock.holds))
        return real_stim(nid, cur)

    g.step, g.stimulate = step, stim
    return lock, calls


def test_one_step_per_hold_and_the_prime_rides_the_first_step():
    g, ids = _graph()
    lock, calls = _instrument(g)
    stats = new_unified_stats()
    tonic_unified_drive(g, [ids[0], ids[1]], [1.0, 0.8], 3, stats)
    holds_of = [h for kind, h in calls if kind == "step"]
    stim_holds = {h for kind, h in calls if kind == "stim"}
    assert holds_of == [1, 2, 3], "each step() in its own hold"
    assert stim_holds == {1}, "the seeds are stimulated in the same hold as the step that consumes them"
    assert stats["holds"] == 3 and stats["steps"] == 3 and stats["drives"] == 1


def test_zero_steps_still_primes_in_one_hold():
    g, ids = _graph()
    lock, calls = _instrument(g)
    stats = new_unified_stats()
    tonic_unified_drive(g, [ids[2]], [1.0], 0, stats)
    assert [k for k, _ in calls] == ["stim"]
    assert stats["holds"] == 1 and stats["steps"] == 0


def _slow_steps(g, seconds):
    real = g.step

    def slow(*a, **k):
        time.sleep(seconds)
        return real(*a, **k)

    g.step = slow


def test_a_waiter_gets_in_between_the_tonic_steps():
    """A reader that asks for the lock while a 4-step drive runs waits at most about one step, not the whole drive."""
    g, ids = _graph()
    step_s = 0.15
    _slow_steps(g, step_s)
    got = {}

    def reader():
        time.sleep(step_s * 0.5)              # arrives during the first step
        t0 = time.perf_counter()
        with g._step_lock:
            got["wait"] = time.perf_counter() - t0
            got["ts"] = g.timestep

    ts0 = g.timestep
    th = threading.Thread(target=reader)
    th.start()
    tonic_unified_drive(g, [ids[0]], [1.0], 4)
    th.join()
    assert got["wait"] < 2 * step_s, f"waited {got['wait']:.3f}s: behind more than one step"
    assert got["ts"] - ts0 < 4, "the reader got in before the drive finished"


def test_interleaved_stepper_no_lost_or_doubled_spike_and_clock_counts_once():
    g, ids = _graph(n=40, p=0.15)
    ts0 = g.timestep
    n_drain, n_tonic, steps_per = 150, 40, 3
    errors = []
    stats = new_unified_stats()
    pops = []
    real_step = g.step

    def step(*a, **k):                         # records which timestep each step() consumed (every thread)
        with g._step_lock:
            pops.append(g.timestep + 1)
            return real_step(*a, **k)

    g.step = step

    def drain():
        try:
            for _ in range(n_drain):
                g.step()
                time.sleep(0.0005)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    th = threading.Thread(target=drain)
    th.start()
    for i in range(n_tonic):
        tonic_unified_drive(g, [ids[i % 40]], [3.0], steps_per, stats)
    th.join()
    assert not errors
    assert g.timestep == ts0 + n_drain + n_tonic * steps_per
    assert sorted(pops) == list(range(ts0 + 1, g.timestep + 1)), "every tick stepped exactly once"
    assert not [k for k in g._delay_buffer if k <= g.timestep], "no delayed spike stranded behind the clock"
    assert stats["steps"] == n_tonic * steps_per
    assert stats["timestep_advance"] == n_tonic * steps_per
    assert stats["interleaved_steps"] > 0, "the drain actually got in between the Tonic's holds"


def test_yield_key_read_live_and_zero_allowed():
    g, ids = _graph(extra={"tonic_unified_lock_yield_ms": 0})
    sleeps = []
    import tonic_thread as tt
    real_sleep = tt.time.sleep
    try:
        tt.time.sleep = lambda s: sleeps.append(s)
        tonic_unified_drive(g, [ids[0]], [1.0], 3)
        assert sleeps == []
        g.config["tonic_unified_lock_yield_ms"] = 2.0
        tonic_unified_drive(g, [ids[0]], [1.0], 3)
        assert sleeps == [0.002, 0.002]
        g.config["tonic_unified_lock_yield_ms"] = "junk"
        tonic_unified_drive(g, [ids[0]], [1.0], 2)
        assert sleeps[-1] == 0.001, "malformed -> the default"
    finally:
        tt.time.sleep = real_sleep


def test_engine_tick_still_five_steps_and_two_drives_plus_whisper():
    g, ids = _graph()
    _warm(g, ids)
    g.nodes[ids[0]].metadata = {"constitutional": True}
    t = _thread(g)
    eng = _engine(g, t, [(ids[1], 0.9)])
    ts0 = g.timestep
    eng._generate_latent_token_inner()
    assert g.timestep - ts0 == 5
    u = t.status["unified"]
    assert u["timestep_advance"] == 5 and u["holds"] == 5 and u["drives"] == 3


# --------------------------------------------------------------------------- single-threaded == the one-hold drive

UNIFIED_TREE = os.environ.get("TONIC_UNIFIED_TREE", os.path.expanduser("~/worktrees/ng-tonic-unified-20261021"))


def _scenario(tree, mode):
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=tree, HF_HUB_OFFLINE="1")
    out = subprocess.run([sys.executable, os.path.join(HERE, "tonic_unified_scenario.py"), mode],
                         cwd="/tmp", env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.skipif(not os.path.isdir(UNIFIED_TREE), reason="unified tree (7a70748) not present")
def test_single_threaded_identical_to_the_one_hold_drive():
    for mode in ("absent", "false", "on"):
        assert _scenario(ROOT, mode) == _scenario(UNIFIED_TREE, mode), mode
