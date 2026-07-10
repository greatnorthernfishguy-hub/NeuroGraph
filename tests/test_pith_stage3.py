# tests/test_pith_stage3.py
#
# ---- Changelog ----
# [2026-07-09] Claude Code (Sonnet 5) — Pith Stage 3 tests (unified rank + char budget)
# What: Direct, ungated tests of pith_stage3() + CacheLine's `stream` field --
#   cross-stream ranking (per-stream normalization + weighting), budget-bounded
#   drop, pinned-lines-reserved-off-budget, oversized-top-line-kept,
#   degenerate empty/all-equal inputs, and _PITH_METRICS bookkeeping
#   (ranked_in/ranked_kept/ranked_dropped/budget_chars_used).
# Why: docs/superpowers/plans/2026-07-08-pith-extraction-pipeline*.md, Stage 3
#   increment. pith_stage3 is a pure function (no I/O, no graph, no embed) --
#   these tests exercise it directly rather than through the gated _recall()
#   wiring in cc-ng-daemon.py.
# How: Plain CacheLine construction (with `stream=`) + pith_stage3() calls;
#   _PITH_METRICS is reset() in an autouse fixture before/after each test so
#   counter assertions are isolated, same pattern as test_pith_stage1.py.
# -------------------
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from cc_ng_organism import CacheLine, pith_stage3, _PITH_METRICS


@pytest.fixture(autouse=True)
def _reset_metrics():
    _PITH_METRICS.reset()
    yield
    _PITH_METRICS.reset()


def test_cross_stream_ranking_pattern_beats_monitor_under_default_weights():
    # Default weights: pattern (CC_PITH_W_RELEVANCE=1.0) > monitor
    # (CC_PITH_W_RECENCY=0.6). A high-raw-score pattern line must rank above
    # a low-raw-score monitor line -- AND above a monitor line that is
    # top-of-its-own-stream (norm=1.0), since 1.0*1.0 > 0.6*1.0.
    lines = [
        CacheLine.from_surfaced("mon_top", "monitor top of its own stream", score=1.7, stream="monitor"),
        CacheLine.from_surfaced("pat_low", "pattern low of its own stream", score=10.0, stream="pattern"),
        CacheLine.from_surfaced("pat_top", "pattern top of its own stream", score=180.0, stream="pattern"),
    ]
    out = pith_stage3(lines)
    ids = [l.node_id for l in out]
    assert ids.index("pat_top") < ids.index("mon_top") < ids.index("pat_low")


def test_budget_drop_lowest_ranked_dropped_and_metrics_reflect_it():
    # Three unpinned lines, each 10 chars, all pattern stream so norm alone
    # ranks by raw score. Budget of 15 chars fits exactly one line.
    lines = [
        CacheLine.from_surfaced("low", "aaaaaaaaaa", score=1.0, stream="pattern"),
        CacheLine.from_surfaced("mid", "bbbbbbbbbb", score=5.0, stream="pattern"),
        CacheLine.from_surfaced("high", "cccccccccc", score=10.0, stream="pattern"),
    ]
    out = pith_stage3(lines, budget_chars=15)
    ids = [l.node_id for l in out]
    assert ids == ["high"]
    assert sum(len(l.content) for l in out) <= 15
    snap = _PITH_METRICS.snapshot()
    assert snap["ranked_in"] == 3
    assert snap["ranked_kept"] == 1
    assert snap["ranked_dropped"] == 2


def test_pinned_reserved_off_budget_does_not_evict_fitting_unpinned():
    # A pinned line with content far larger than the budget must ALWAYS be
    # in the output, and must NOT count against the budget -- an unpinned
    # line that fits the budget on its own must still survive alongside it.
    pinned = CacheLine.from_surfaced("pin", "x" * 5000, score=0.1, pinned=True, stream="monitor")
    fits = CacheLine.from_surfaced("fits", "y" * 100, score=10.0, stream="pattern")
    out = pith_stage3([pinned, fits], budget_chars=200)
    ids = {l.node_id for l in out}
    assert ids == {"pin", "fits"}
    # pinned line first, original relative order preserved.
    assert out[0].node_id == "pin"


def test_oversized_top_line_kept_alone_no_empty_l1():
    # A single unpinned line longer than the whole budget must still be
    # emitted (never an empty L1 just because the top item is large), and
    # fill must stop after it (no other lines follow).
    big = CacheLine.from_surfaced("big", "z" * 500, score=10.0, stream="pattern")
    small = CacheLine.from_surfaced("small", "w" * 10, score=1.0, stream="pattern")
    out = pith_stage3([big, small], budget_chars=200)
    ids = [l.node_id for l in out]
    assert ids == ["big"]


def test_empty_input_returns_empty_list():
    assert pith_stage3([]) == []


def test_all_equal_scores_in_stream_no_div_by_zero_stable_order_preserved():
    lines = [
        CacheLine.from_surfaced("a", "content a", score=5.0, stream="pattern"),
        CacheLine.from_surfaced("b", "content b", score=5.0, stream="pattern"),
        CacheLine.from_surfaced("c", "content c", score=5.0, stream="pattern"),
    ]
    out = pith_stage3(lines)
    ids = [l.node_id for l in out]
    assert ids == ["a", "b", "c"]  # all normalize to 1.0 -> stable, input order


def test_metrics_snapshot_consistent_with_run():
    lines = [
        CacheLine.from_surfaced("keep1", "a" * 50, score=10.0, stream="pattern"),
        CacheLine.from_surfaced("keep2", "b" * 50, score=1.7, stream="monitor"),
        CacheLine.from_surfaced("drop1", "c" * 50, score=1.0, stream="pattern"),
    ]
    out = pith_stage3(lines, budget_chars=120)
    snap = _PITH_METRICS.snapshot()
    assert snap["ranked_in"] == 3
    assert snap["ranked_kept"] == len(out)
    assert snap["ranked_kept"] + snap["ranked_dropped"] == 3
    assert snap["budget_chars_used"] == sum(len(l.content) for l in out)


def test_unknown_stream_fails_open_to_weight_one():
    # An unrecognized stream name must not be zeroed out -- it stays in
    # contention with weight 1.0, same as "pattern"/"recall" default.
    lines = [
        CacheLine.from_surfaced("unk", "unknown stream content", score=5.0, stream="mystery"),
        CacheLine.from_surfaced("mon", "monitor content", score=1.7, stream="monitor"),
    ]
    out = pith_stage3(lines)
    ids = [l.node_id for l in out]
    # "unk" is top-of-its-own-stream (norm=1.0) at weight 1.0 (unified=1.0),
    # beating "mon" (norm=1.0, weight 0.6, unified=0.6).
    assert ids == ["unk", "mon"]


def test_stream_field_defaults_to_recall_and_from_surfaced_accepts_override():
    default_cl = CacheLine.from_surfaced("n1", "content", score=1.0)
    assert default_cl.stream == "recall"
    tagged_cl = CacheLine.from_surfaced("n2", "content", score=1.0, stream="pattern")
    assert tagged_cl.stream == "pattern"
