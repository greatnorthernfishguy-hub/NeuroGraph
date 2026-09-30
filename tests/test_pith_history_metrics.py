# ---- Changelog ----
# [2026-09-12] Codex — outbound Pith history telemetry acceptance coverage.
# What: exercise success, fail-soft, reset, config, and concurrent exact counts.
# Why: the old inbound counters could not prove miniTID history compression ran.
# How: real PithMetrics and canonical compressor with pure deterministic stubs.
# -------------------
"""Telemetry for the canonical outbound miniTID history-compression path."""

import threading

import pytest

import cc_ng_organism as cc


@pytest.fixture(autouse=True)
def _reset_metrics():
    cc._PITH_METRICS.reset()
    yield
    cc._PITH_METRICS.reset()


def test_history_compression_records_result(monkeypatch):
    monkeypatch.setattr(cc, "cc_thermal", lambda graph, node_id: 0.0)
    monkeypatch.setattr(
        cc, "pith_stage2_keyframe",
        lambda text, max_chars: (text[:max_chars], text[max_chars:]),
    )
    turns = ["a" * 300, "short"]

    out = cc.pith_compress_history(turns, graph=object(), per_turn_chars=100)
    snap = cc._PITH_METRICS.snapshot()

    assert out == ["a" * 100, "short"]
    assert snap["history_calls"] == 1
    assert snap["history_turns_in"] == 2
    assert snap["history_turns_compressed"] == 1
    assert snap["history_chars_in"] == 305
    assert snap["history_chars_out"] == 105
    assert snap["history_chars_saved"] == 200
    assert snap["history_failures"] == 0


def test_history_failure_is_visible_and_preserves_turn(monkeypatch):
    monkeypatch.setattr(cc, "cc_thermal", lambda graph, node_id: 0.0)
    monkeypatch.setattr(
        cc, "pith_stage2_keyframe",
        lambda text, max_chars: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    turns = ["the original turn"]

    assert cc.pith_compress_history(turns, graph=object()) == turns
    snap = cc._PITH_METRICS.snapshot()
    assert snap["history_calls"] == 1
    assert snap["history_turns_compressed"] == 0
    assert snap["history_failures"] == 1
    assert snap["pith_failures"] == 1
    assert snap["history_chars_in"] == snap["history_chars_out"]


def test_falsy_non_string_turn_is_recorded_as_a_failure():
    turns = [None, {}, [], b""]

    assert cc.pith_compress_history(turns, graph=object()) == turns
    snap = cc._PITH_METRICS.snapshot()
    assert snap["history_calls"] == 1
    assert snap["history_turns_in"] == len(turns)
    assert snap["history_failures"] == len(turns)
    assert snap["pith_failures"] == len(turns)
    assert snap["history_chars_in"] == 0
    assert snap["history_chars_out"] == 0


def test_history_metrics_reset_together(monkeypatch):
    monkeypatch.setattr(cc, "cc_thermal", lambda graph, node_id: 0.0)
    monkeypatch.setattr(cc, "pith_stage2_keyframe", lambda text, max_chars: (text, ""))
    cc.pith_compress_history(["unchanged"], graph=object())

    cc._PITH_METRICS.reset()

    snap = cc._PITH_METRICS.snapshot()
    for key in (
        "history_calls", "history_turns_in", "history_turns_compressed",
        "history_chars_in", "history_chars_out", "history_chars_saved",
        "history_failures",
    ):
        assert snap[key] == 0


def test_history_budget_is_in_effective_config():
    config = cc.pith_effective_config()
    assert config["resolved"]["CC_PITH_KEYFRAME_CHARS"] == cc._CC_PITH_KEYFRAME_CHARS
    assert config["authority"]["CC_PITH_KEYFRAME_CHARS"] == "cc_ng_organism"


def test_concurrent_history_calls_do_not_lose_results(monkeypatch):
    monkeypatch.setattr(cc, "cc_thermal", lambda graph, node_id: 0.0)
    monkeypatch.setattr(
        cc, "pith_stage2_keyframe",
        lambda text, max_chars: (text[:max_chars], text[max_chars:]),
    )
    workers, calls_per_worker = 8, 40

    def worker():
        for _ in range(calls_per_worker):
            cc.pith_compress_history(["x" * 100], graph=object(), per_turn_chars=60)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not any(thread.is_alive() for thread in threads)
    total = workers * calls_per_worker
    snap = cc._PITH_METRICS.snapshot()
    assert snap["history_calls"] == total
    assert snap["history_turns_in"] == total
    assert snap["history_turns_compressed"] == total
    assert snap["history_chars_saved"] == total * 40


def test_history_snapshot_never_observes_half_a_result(monkeypatch):
    monkeypatch.setattr(cc, "cc_thermal", lambda graph, node_id: 0.0)
    monkeypatch.setattr(
        cc, "pith_stage2_keyframe",
        lambda text, max_chars: (text[:max_chars], text[max_chars:]),
    )
    stop = threading.Event()
    violations = []

    def writer():
        while not stop.is_set():
            cc.pith_compress_history(["x" * 100], graph=object(), per_turn_chars=60)

    def resetter():
        while not stop.is_set():
            cc._PITH_METRICS.reset()

    def reader():
        while not stop.is_set():
            snap = cc._PITH_METRICS.snapshot()
            calls = snap["history_calls"]
            expected = (calls, calls, 100 * calls, 60 * calls, 40 * calls, 0)
            observed = (
                snap["history_turns_in"], snap["history_turns_compressed"],
                snap["history_chars_in"], snap["history_chars_out"],
                snap["history_chars_saved"], snap["history_failures"],
            )
            if observed != expected:
                violations.append((calls, observed))
                stop.set()

    threads = [threading.Thread(target=writer) for _ in range(3)]
    threads += [threading.Thread(target=resetter), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    threading.Event().wait(1.0)
    stop.set()
    for thread in threads:
        thread.join(timeout=30)

    assert not any(thread.is_alive() for thread in threads)
    assert not violations
