# ---- Changelog ----
# [2026-10-23] Claude Code (Opus 5.5, laptop) — #1176: the drain's head trim never loses a concurrent miniTID append
# What: (1) a CONCURRENT WRITER: separate processes append frames exactly the way miniTID does (open by path with
#   O_WRONLY|O_CREAT|O_APPEND, flock LOCK_EX, write, unlock: Condensate rust_core/vendor/ng_tract/src/write.rs) while
#   the drain absorbs and trims in a loop; every frame written ends up absorbed exactly once or still in the tract.
#   (2) the trim waits for the tract lock (a writer holding it delays the trim; its append survives). (3) a file that no
#   longer starts with the consumed prefix is left untouched.
# Why: #1176 (dev-log 2026-10-23_turn-duplication-1173 §5; Exec P295 found the same window): the old trim re-read the
#   tract and rewrote it with open(path, "wb"); an append in between was erased (raw experience lost, LAW 7).
# How: real ng_tract frames on tmp files, a fake dual pass (instant success); nothing under ~/.claude.
# -------------------
import fcntl
import multiprocessing as mp
import os
import sys
import threading
import time
import types

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import cc_ng_organism
import ng_tract
from test_cc_poison_retry_920 import env, _frame, _write, _read, _frames_of  # noqa: F401 (fixture)


def _minitid_append(path, data):
    """The #1173 miniTID append_frame_locked, step for step: open by path (O_APPEND), flock(LOCK_EX), and ONLY if the
    path is still the inode we locked (the drain's trim renames a new tract in place) write, else unlock, close and
    re-open (Condensate rust_core/src/minitid.rs)."""
    for _ in range(64):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                try:
                    same = os.stat(path).st_ino == os.fstat(fd).st_ino
                except FileNotFoundError:
                    same = False
                if not same:
                    continue
                view = memoryview(data)
                while view:
                    n = os.write(fd, view)
                    view = view[n:]
                return
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    raise AssertionError("append could not settle on the current inode")


def _writer(path, frames_path, start_evt, n_frames, tag):
    with open(frames_path, "rb") as f:
        frame_bytes = f.read()
    frames = _frames_of(frame_bytes)
    start_evt.wait()
    for i in range(n_frames):
        _minitid_append(path, frames[i])
        if i % 7 == 0:
            time.sleep(0.0005)


def _drain_all(path, absorbed):
    return cc_ng_organism.drain_ingest_tract(types.SimpleNamespace(nodes={}), None, {"last_forest_id": None},
                                             tract_path=path, return_consumed=True)


def test_concurrent_writers_lose_nothing(env, monkeypatch):
    absorbed = []
    monkeypatch.setattr(cc_ng_organism, "run_conversational_dual_pass",
                        lambda g, v, text, emb, state, prepared=None: (absorbed.append(text), True)[1])
    tract = _write(env.tmp / "turns.tract", [])
    n_per, writers = 400, 2
    # pre-build each writer's frames (distinct texts) with the real ng_tract writer
    frame_files = []
    for w in range(writers):
        fp = str(env.tmp / ("w%d.frames" % w))
        with open(fp, "wb") as f:
            for i in range(n_per):
                f.write(_frame(env.tmp, "writer %d turn %d %s" % (w, i, "x" * (i % 50)), 10_000 + w * n_per + i))
        frame_files.append(fp)
    ctx = mp.get_context("fork")
    evt = ctx.Event()
    procs = [ctx.Process(target=_writer, args=(tract, frame_files[w], evt, n_per, w)) for w in range(writers)]
    for p in procs:
        p.start()
    evt.set()
    deadline = time.time() + 120
    while any(p.is_alive() for p in procs) and time.time() < deadline:
        _drain_all(tract, absorbed)
    for p in procs:
        p.join(5)
        assert p.exitcode == 0
    for _ in range(3):
        _drain_all(tract, absorbed)
    left = [e.content for e in ng_tract.TractReader(_read(tract))]
    seen = absorbed + left
    expected = ["writer %d turn %d %s" % (w, i, "x" * (i % 50)) for w in range(writers) for i in range(n_per)]
    assert len(seen) == len(expected), "lost %d frame(s)" % (len(expected) - len(seen))
    assert sorted(seen) == sorted(expected)                       # every frame exactly once


def test_trim_waits_for_the_tract_lock_and_keeps_the_held_append(env, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, "run_conversational_dual_pass",
                        lambda g, v, text, emb, state, prepared=None: True)
    a, b = _frame(env.tmp, "turn a", 1), _frame(env.tmp, "turn b", 2)
    late = _frame(env.tmp, "late append", 3)
    tract = _write(env.tmp / "turns.tract", [a, b])
    fd = os.open(tract, os.O_WRONLY | os.O_APPEND)
    fcntl.flock(fd, fcntl.LOCK_EX)                                # "miniTID" is mid-append
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", _drain_all(tract, None)))
    t.start()
    time.sleep(0.3)
    assert t.is_alive(), "the trim must wait for the tract lock"
    assert _read(tract) == a + b
    os.write(fd, late)                                            # the append lands while the drain waits
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    t.join(10)
    assert not t.is_alive()
    assert out["r"] == (2, a + b)
    assert _read(tract) == late                                   # the append survived the trim


def test_trim_leaves_a_rewritten_file_untouched(env):
    a, b, c = (_frame(env.tmp, "t%d" % i, i) for i in range(3))
    path = _write(env.tmp / "turns.tract", [b, c])
    assert cc_ng_organism._drain_trim_head_locked(path, a) == b""
    assert _read(path) == b + c
    assert cc_ng_organism._drain_trim_head_locked(path, b) == b
    assert _read(path) == c


def test_trim_fsyncs(env, monkeypatch):
    synced = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])
    a, b = _frame(env.tmp, "x", 1), _frame(env.tmp, "y", 2)
    path = _write(env.tmp / "turns.tract", [a, b])
    cc_ng_organism._drain_trim_head_locked(path, a)
    assert synced and _read(path) == b


def test_trim_is_a_rename_and_never_leaves_a_half_file(env, monkeypatch):
    """The remainder is written to <tract>.trim-tmp and renamed in: a crash before the rename leaves the WHOLE old tract
    (absorbed again next pass, as before), never a half-written one; the temp file is fsynced first."""
    a, b = _frame(env.tmp, "x", 1), _frame(env.tmp, "y", 2)
    path = _write(env.tmp / "turns.tract", [a, b])
    ino = os.stat(path).st_ino

    def boom(src, dst):
        raise OSError("crash before the rename")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        cc_ng_organism._drain_trim_head_locked(path, a)
    assert _read(path) == a + b and os.stat(path).st_ino == ino          # untouched
    monkeypatch.undo()
    assert cc_ng_organism._drain_trim_head_locked(path, a) == a
    assert _read(path) == b and os.stat(path).st_ino != ino                # replaced, not rewritten in place


def test_trim_keeps_the_tract_mode(env):
    a, b = _frame(env.tmp, "x", 1), _frame(env.tmp, "y", 2)
    path = _write(env.tmp / "turns.tract", [a, b])
    os.chmod(path, 0o600)
    cc_ng_organism._drain_trim_head_locked(path, a)
    assert os.stat(path).st_mode & 0o777 == 0o600
