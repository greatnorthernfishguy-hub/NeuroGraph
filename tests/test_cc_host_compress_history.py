# ---- Changelog ----
# [2026-09-12] Codex — VPS host parity coverage for miniTID's Pith peninsula.
# What: prove compress_history is dispatched, uses CC's graph under its lock,
#   forwards the optional budget, preserves positional output, and fails soft.
# Why: the missing hosted-socket verb made miniTID silently use faux truncation.
# How: isolated sentinels only; no live graph, checkpoint, model, or metrics.
# -------------------
"""Contract tests for the hosted CC compress_history socket handler."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cc_ng_host
import cc_ng_organism


class _RecordingLock:
    def __init__(self):
        self.depth = 0
        self.entries = 0

    def __enter__(self):
        self.depth += 1
        self.entries += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.depth -= 1


class _Graph:
    def __init__(self):
        self._concurrent_lock = _RecordingLock()


class _NG:
    def __init__(self):
        self.graph = _Graph()


def test_compress_history_is_registered_for_socket_dispatch():
    assert cc_ng_host._DISPATCH["compress_history"] is cc_ng_host._handle_compress_history


def test_compress_history_uses_cc_graph_lock_and_forwards_budget(monkeypatch):
    ng = _NG()
    turns = ["older first turn", "older second turn"]
    seen = {}

    monkeypatch.setattr(cc_ng_host._STATE, "cc_ng", ng)

    def fake_compress(got_turns, graph, per_turn_chars=None):
        seen["turns"] = got_turns
        seen["graph"] = graph
        seen["per_turn_chars"] = per_turn_chars
        seen["lock_depth"] = graph._concurrent_lock.depth
        return ["first pith", "second pith"]

    monkeypatch.setattr(cc_ng_organism, "pith_compress_history", fake_compress)

    result = cc_ng_host._handle_compress_history(
        {"turns": turns, "per_turn_chars": 321}
    )

    assert result == {"ok": True, "compressed": ["first pith", "second pith"]}
    assert seen == {
        "turns": turns,
        "graph": ng.graph,
        "per_turn_chars": 321,
        "lock_depth": 1,
    }
    assert ng.graph._concurrent_lock.entries == 1
    assert ng.graph._concurrent_lock.depth == 0


def test_compress_history_empty_input_is_a_noop(monkeypatch):
    called = []
    monkeypatch.setattr(
        cc_ng_organism,
        "pith_compress_history",
        lambda *args, **kwargs: called.append(True),
    )

    assert cc_ng_host._handle_compress_history({"turns": []}) == {
        "ok": True,
        "compressed": [],
    }
    assert not called


def test_compress_history_returns_original_turns_on_failure(monkeypatch):
    turns = ["do not lose this", "or this"]
    monkeypatch.setattr(cc_ng_host._STATE, "cc_ng", _NG())

    def fail(*args, **kwargs):
        raise RuntimeError("compressor unavailable")

    monkeypatch.setattr(cc_ng_organism, "pith_compress_history", fail)

    assert cc_ng_host._handle_compress_history({"turns": turns}) == {
        "ok": True,
        "compressed": turns,
    }


def test_compress_history_can_run_during_graphless_startup(monkeypatch):
    turns = ["older turn"]
    seen = {}
    monkeypatch.setattr(cc_ng_host._STATE, "cc_ng", None)

    def fake_compress(got_turns, graph, per_turn_chars=None):
        seen["graph"] = graph
        return ["compressed"]

    monkeypatch.setattr(cc_ng_organism, "pith_compress_history", fake_compress)

    assert cc_ng_host._handle_compress_history({"turns": turns}) == {
        "ok": True,
        "compressed": ["compressed"],
    }
    assert seen["graph"] is None
