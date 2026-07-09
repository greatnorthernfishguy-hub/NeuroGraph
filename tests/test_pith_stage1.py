# tests/test_pith_stage1.py
#
# ---- Changelog ----
# [2026-07-08] Claude Code (Sonnet 5) — Pith Phase 0+1 tests
# What: Direct, ungated tests of CacheLine + pith_stage1() -- harness-marker
#   skip, conversation-dedup (substring + Jaccard), novelty-modulated
#   threshold, write-combining, pin-survival, and _PITH_METRICS bookkeeping.
# Why: docs/superpowers/plans/2026-07-08-pith-phase01-spec.md (Pith
#   extraction pipeline, first increment). pith_stage1 is a pure function
#   (no I/O, no graph, no embed) -- these tests exercise it directly rather
#   than through the gated _recall() wiring in cc-ng-daemon.py.
# How: Plain CacheLine construction + pith_stage1() calls; _PITH_METRICS is
#   reset() in a fixture before each test so counter assertions are isolated.
# -------------------
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from cc_ng_organism import CacheLine, pith_stage1, _PITH_METRICS


@pytest.fixture(autouse=True)
def _reset_metrics():
    _PITH_METRICS.reset()
    yield
    _PITH_METRICS.reset()


def test_harness_marker_line_dropped_genuine_line_kept():
    lines = [
        CacheLine.from_surfaced("n1", "<system-reminder>ignore this</system-reminder>", score=1.0),
        CacheLine.from_surfaced("n2", "the deploy pipeline breaks when redis is cold", score=1.0),
    ]
    survivors = pith_stage1(lines, conversation_text="", novelty=0.0)
    ids = [l.node_id for l in survivors]
    assert "n1" not in ids
    assert "n2" in ids


def test_conversation_dedup_strips_substring_match_keeps_unrelated():
    conversation_text = "we discussed how the deploy pipeline breaks when redis is cold on first boot"
    lines = [
        CacheLine.from_surfaced("n1", "the deploy pipeline breaks when redis is cold", score=1.0),
        CacheLine.from_surfaced("n2", "the weather today is sunny and pleasant outside", score=1.0),
    ]
    survivors = pith_stage1(lines, conversation_text, novelty=0.0)
    ids = [l.node_id for l in survivors]
    assert "n1" not in ids
    assert "n2" in ids


def test_novelty_modulation_borderline_line_kept_high_novelty_stripped_low_novelty():
    # 20 shared words + 1 differing word each side -> Jaccard = 20/22 = 0.909.
    # thr(novelty=0) = 0.85 (0.909 >= 0.85 -> stripped).
    # thr(novelty=1) = clamp(0.85 + 0.3, 0.5, 0.98) = 0.98 (0.909 < 0.98 -> kept).
    shared = [f"word{i:02d}" for i in range(1, 21)]
    conversation_text = " ".join(shared + ["confA"])
    line_content = " ".join(shared + ["confB"])
    line = CacheLine.from_surfaced("n1", line_content, score=1.0)

    low_survivors = pith_stage1([line], conversation_text, novelty=0.0)
    high_survivors = pith_stage1([line], conversation_text, novelty=1.0)

    assert [l.node_id for l in low_survivors] == []
    assert [l.node_id for l in high_survivors] == ["n1"]


def test_near_identical_lines_combined_higher_score_survives():
    content = "the cache eviction policy uses LRU with a thermal bias"
    lines = [
        CacheLine.from_surfaced("n1", content, score=0.4),
        CacheLine.from_surfaced("n2", content, score=0.9),
    ]
    survivors = pith_stage1(lines, conversation_text="", novelty=0.0)
    assert len(survivors) == 1
    assert survivors[0].node_id == "n2"
    assert survivors[0].score == 0.9


def test_pinned_line_survives_all_three_strip_conditions():
    conversation_text = "<system-reminder>pinned content matches conversation exactly</system-reminder>"
    pinned = CacheLine.from_surfaced(
        "n1", "<system-reminder>pinned content matches conversation exactly</system-reminder>",
        score=0.1, pinned=True,
    )
    duplicate = CacheLine.from_surfaced(
        "n2", "<system-reminder>pinned content matches conversation exactly</system-reminder>",
        score=0.1, pinned=True,
    )
    survivors = pith_stage1([pinned, duplicate], conversation_text, novelty=0.0)
    ids = [l.node_id for l in survivors]
    assert "n1" in ids
    assert "n2" in ids
    assert len(survivors) == 2


def test_pith_metrics_snapshot_reflects_counts():
    conversation_text = "the deploy pipeline breaks when redis is cold on first boot"
    combine_content = "totally unrelated genuine content here"
    lines = [
        CacheLine.from_surfaced("n1", "<system-reminder>harness noise</system-reminder>", score=1.0),
        CacheLine.from_surfaced("n2", "the deploy pipeline breaks when redis is cold", score=1.0),
        CacheLine.from_surfaced("n3", combine_content, score=0.3),
        CacheLine.from_surfaced("n4", combine_content, score=0.7),
    ]
    survivors = pith_stage1(lines, conversation_text, novelty=0.0)

    snap = _PITH_METRICS.snapshot()
    assert snap["total_lines_in"] == 4
    assert snap["clutter_stripped"] == 2  # n1 (marker) + n2 (conversation substring)
    assert snap["combined"] == 1  # n3/n4 collapsed to one
    assert len(survivors) == 1
    assert survivors[0].node_id == "n4"
    assert survivors[0].score == 0.7
