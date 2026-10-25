# ---- Changelog ----
# [2026-10-25] Claude Code (Opus 5.5; LANE B syl-stage1, #920 adapted) — RetryQueue quarantines instead of dropping
# What: an item reaching max_attempts lands in <path>.quarantine.msgpack (length-prefixed msgpack) and leaves the
#       queue; successes leave without a quarantine frame; a failed quarantine append keeps the item queued.
# Why:  never silently lose one of Syl's turn memories (laptop #920 policy).
# How:  tmp_path queue; monkeypatched open() for the append-failure case.
# -------------------
import builtins
import os
import struct
import sys

import msgpack

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_retry_queue import RetryQueue  # noqa: E402


def _frames(path):
    raw = open(path, "rb").read()
    out, off = [], 0
    while off + 4 <= len(raw):
        (n,) = struct.unpack(">I", raw[off:off + 4])
        out.append(msgpack.unpackb(raw[off + 4:off + 4 + n], raw=False))
        off += 4 + n
    return out


def test_item_at_max_attempts_is_quarantined_not_lost(tmp_path):
    q = RetryQueue(str(tmp_path / "q.msgpack"), max_attempts=2)
    q.enqueue("a", "turn a")
    q.enqueue("b", "turn b")

    def attempt(item):
        if item["target_id"] == "b":
            return True
        raise RuntimeError("cannot import name 'embed'")

    assert q.drain(attempt) == 1 and q.pending_count() == 1
    assert not os.path.exists(q.quarantine_path)
    assert q.drain(attempt) == 0 and q.pending_count() == 0
    (frame,) = _frames(q.quarantine_path)
    assert frame["target_id"] == "a" and frame["content"] == "turn a" and frame["attempts"] == 2
    assert "cannot import name" in frame["last_error"]


def test_failed_quarantine_keeps_item(tmp_path, monkeypatch):
    q = RetryQueue(str(tmp_path / "q.msgpack"), max_attempts=1)
    q.enqueue("a", "turn a")
    real_open = builtins.open

    def guarded(path, mode="r", *a, **k):
        if str(path).endswith(".quarantine.msgpack"):
            raise OSError("disk full")
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded)
    q.drain(lambda item: False)
    monkeypatch.setattr(builtins, "open", real_open)
    assert q.pending_count() == 1
    assert RetryQueue(str(tmp_path / "q.msgpack")).pending_count() == 1
