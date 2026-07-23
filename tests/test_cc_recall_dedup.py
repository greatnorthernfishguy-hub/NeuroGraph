# tests/test_cc_recall_dedup.py
#
# ---- Changelog ----
# [2026-07-07] Claude Code (Sonnet 5) — Cross-block dedup coverage for _recall()
# What: Exercises cc_ng_host._recall() to confirm a node that is both
#   recently-fired (SurfacingMonitor) and semantically matching the query
#   (Active Recall / cc_pattern_completion_recall) is rendered exactly once
#   in the combined injected context, not once per block.
# Why: Ground-truth read of both mirrored _recall() implementations
#   (cc_ng_host.py, docs/scripts/cc-ng-daemon.py) found no filtering between
#   the two node_id-keyed result sets before concatenation — confirmed live
#   by this session's own hook output repeating the same content twice.
#   See /home/josh/docs/superpowers/plans/2026-07-07-cc-surfacing-truncation-dedup-devlog.md
# How: Fakes _surfacing_monitor (get_surfaced/format_context) and
#   monkeypatches cc_ng_organism.cc_pattern_completion_recall so _recall()'s
#   own dedup logic runs for real against controlled, overlapping node_ids.
# -------------------
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


class _FakeMonitor:
    def __init__(self, items):
        self._items = items

    def get_surfaced(self):
        return self._items

    def format_context(self, surfaced_items=None):
        items = self._items if surfaced_items is None else surfaced_items
        if not items:
            return ""
        lines = ["[NeuroGraph Surfaced Knowledge]"]
        for it in items:
            lines.append(f"- {it['content']} (salience: {it['score']:.2f})")
        return "\n".join(lines)


@pytest.fixture
def cc_ng_state(monkeypatch):
    import cc_ng_host
    import cc_ng_organism

    monkeypatch.setattr(cc_ng_host._STATE, "cc_ng", type("NG", (), {"_surfacing_monitor": None})())
    monkeypatch.setattr(cc_ng_host._STATE, "conv_state", {})
    # This suite exercises the pre-Pith monitor_ctx/pc_block dedup logic in
    # cc_assemble_recall() specifically, regardless of the ambient
    # CC_PITH_ENABLED env var (set process-wide in this environment's
    # ~/.bashrc) -- force the gate off so these tests are deterministic.
    # See tests/test_pith_stage5.py for the same module-constant-patch pattern.
    monkeypatch.setattr(cc_ng_organism, "_CC_PITH_ENABLED", False)
    return cc_ng_host._STATE


def test_recall_dedups_node_shared_across_both_blocks(monkeypatch, cc_ng_state):
    """A node_id present in both SurfacingMonitor's queue and the
    pattern-completion results must render only once in the combined output."""
    import cc_ng_host
    import cc_ng_organism

    monitor = _FakeMonitor([{"node_id": "n1", "content": "shared content", "score": 1.2}])
    cc_ng_state.cc_ng._surfacing_monitor = monitor

    def fake_pc(ng, query, k, state=None):
        return [
            {"node_id": "n1", "score": 0.9, "content": "shared content"},
            {"node_id": "n2", "score": 0.8, "content": "genuinely new content"},
        ]

    monkeypatch.setattr(cc_ng_organism, "cc_pattern_completion_recall", fake_pc)

    result = cc_ng_host._recall("some query", k=5)

    assert result.count("shared content") == 1
    assert "genuinely new content" in result
    assert "## Active Recall" in result
    assert "[NeuroGraph Surfaced Knowledge]" in result


def test_recall_renders_both_blocks_when_no_overlap(monkeypatch, cc_ng_state):
    """No shared node_id between the two blocks -- both render in full, nothing dropped."""
    import cc_ng_host
    import cc_ng_organism

    monitor = _FakeMonitor([{"node_id": "n1", "content": "recency content", "score": 1.0}])
    cc_ng_state.cc_ng._surfacing_monitor = monitor

    def fake_pc(ng, query, k, state=None):
        return [{"node_id": "n2", "score": 0.7, "content": "semantic content"}]

    monkeypatch.setattr(cc_ng_organism, "cc_pattern_completion_recall", fake_pc)

    result = cc_ng_host._recall("some query", k=5)

    assert "recency content" in result
    assert "semantic content" in result


def test_recall_returns_monitor_only_when_pattern_completion_disabled(monkeypatch, cc_ng_state):
    """allow_pattern_completion=False skips Active Recall entirely -- used by
    _handle_pre_tool_use() when gate_pattern_completion() already covered this
    file_path recently."""
    import cc_ng_host
    import cc_ng_organism

    monitor = _FakeMonitor([{"node_id": "n1", "content": "recency content", "score": 1.0}])
    cc_ng_state.cc_ng._surfacing_monitor = monitor

    called = []

    def fake_pc(ng, query, k, state=None):
        called.append(True)
        return [{"node_id": "n2", "score": 0.7, "content": "should not appear"}]

    monkeypatch.setattr(cc_ng_organism, "cc_pattern_completion_recall", fake_pc)

    result = cc_ng_host._recall("some query", k=5, allow_pattern_completion=False)

    assert not called
    assert "recency content" in result
    assert "## Active Recall" not in result
    assert "should not appear" not in result


def test_recall_returns_empty_string_when_no_ng(monkeypatch):
    """No cc_ng wired up yet (daemon still starting) -- fail soft, empty string."""
    import cc_ng_host

    monkeypatch.setattr(cc_ng_host._STATE, "cc_ng", None)
    assert cc_ng_host._recall("some query", k=5) == ""
