#!/usr/bin/env python3
# ---- Changelog ----
# [2026-09-24] Claude Sonnet 5 (Claude Code, groupb-kiss-shared-graduation-001) — Revision 1
# What: add test_effective_threshold_magnitude_at_extremes (pins the shift's
#   magnitude, not just direction) and test_floor_above_base_does_not_break_
#   neutral_equals_base (FLOOR > base still yields base at neutral).
# Why: ZM Revision 1 review, charter §3 -- a mutation halving the span factor
#   (`* 2.0` -> `* 1.0`) passed all 8 existing tests; nothing pinned the
#   magnitude. A FLOOR set above base was also an unguarded edge.
# How: same style as the existing (a)/(f) tests -- _CapturingVectorDB plus a
#   monkeypatched cc_region_confidence, asserting the exact numeric threshold.
# [2026-09-24] Claude Sonnet 5 (Claude Code, groupb-kiss-shared-graduation-001) — new file
# What: tests for the KISS half of Shared Graduation (COMB-04) -- the region-
#   confidence-shifted redundancy threshold in _cc_kiss_find_redundant_node.
# Why: assignment groupb-kiss-shared-graduation-001.md Scope §3 (a)-(f); the LE's
#   four binding conditions (live/pure query, one uniform function, reinforce
#   never drop, novel input still passes) each need a receipt.
# How: unit-level tests against _cc_kiss_find_redundant_node / run_conversational_
#   dual_pass with a minimal graph double (mirrors test_cc_region_confidence.py's
#   MockGraph/MockVectorDB style) and the real universal_ingestor.SimpleVectorDB
#   (the actual class NeuroGraphMemory uses) for genuine cosine-threshold
#   filtering, so the threshold shift is exercised through real search behavior
#   rather than asserted by construction.
# -------------------
"""Tests for COMB-04 Shared Graduation, KISS half: region confidence shifts
_cc_kiss_find_redundant_node's redundancy threshold, mirroring cc_l1_budget's
existing use of the same cc_region_confidence() query on the Pith side.
"""

import os
import sys
import threading

import numpy as np
import pytest
from unittest.mock import Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cc_ng_organism as cc
from universal_ingestor import SimpleVectorDB


# ---------------------------------------------------------------------------
# Minimal graph double -- just enough surface for _cc_kiss_find_redundant_node,
# _cc_kiss_reinforce_node, _cc_bind_conversational_topology and
# _cc_deposit_memory_node to run without crashing. cc_anticipate and the
# synapse/hyperedge creation calls fail soft (broad try/except) on a graph
# this thin, which is exactly the behavior being relied on here -- none of
# these tests exercise topology creation, only the threshold/redundancy path.
# ---------------------------------------------------------------------------

class _MiniNode:
    def __init__(self, node_id, metadata=None):
        self.node_id = node_id
        self.metadata = dict(metadata or {})
        self.threshold = 1.0
        self.intrinsic_excitability = 1.0
        self.spike_history = []


class _MiniGraph:
    def __init__(self):
        self.nodes = {}
        self.synapses = {}
        self._outgoing = {}
        self._step_lock = threading.RLock()
        self.config = {"default_threshold": 1.0}

    def create_node(self, node_id, metadata=None):
        node = _MiniNode(node_id, metadata)
        self.nodes[node_id] = node
        return node

    def create_synapse(self, *args, **kwargs):
        pass

    def create_hyperedge(self, *args, **kwargs):
        pass


class _CapturingVectorDB:
    """Records the threshold (and k) _cc_kiss_find_redundant_node passes to
    search(), and always reports no hits -- isolates the threshold math from
    any actual similarity filtering."""

    def __init__(self):
        self.calls = []

    def search(self, embedding, k=5, threshold=None):
        self.calls.append({"k": k, "threshold": threshold})
        return []


def _similar_vector(base, sim):
    """A 2D unit vector at cosine similarity `sim` to unit vector `base` =
    [1, 0]. Used so a MockVectorDB-free, real SimpleVectorDB search produces
    a genuine, known cosine similarity."""
    assert base == pytest.approx([1.0, 0.0])
    return np.array([sim, (1.0 - sim ** 2) ** 0.5], dtype=np.float32)


# ============================================================================
# (a) Uniformity: effective threshold is monotonic in confidence, equals the
#     base threshold exactly at neutral (0.5).
# ============================================================================

def test_effective_threshold_monotonic_and_equals_base_at_neutral(monkeypatch):
    monkeypatch.setattr(cc, "_CC_KISS_REGION_CONFIDENCE_ENABLED", True)
    # Small enough span that no point in this sweep hits the [floor, 1.0]
    # clamp -- this test is about the raw shape of the function, not the
    # clamp (that's (f)'s job).
    monkeypatch.setattr(cc, "_CC_KISS_REGION_CONFIDENCE_SPAN", 0.05)
    monkeypatch.setattr(cc, "_CC_KISS_REGION_CONFIDENCE_FLOOR", 0.0)

    thresholds = []
    for confidence in (0.0, 0.25, 0.5, 0.75, 1.0):
        monkeypatch.setattr(cc, "cc_region_confidence", lambda g, v, e, c=confidence: c)
        vdb = _CapturingVectorDB()
        cc._cc_kiss_find_redundant_node(_MiniGraph(), vdb, np.array([1.0, 0.0]))
        thresholds.append(vdb.calls[0]["threshold"])

    # Equals the base exactly at neutral confidence (also cc_region_confidence's
    # own fail-soft value).
    assert thresholds[2] == pytest.approx(cc._CC_KISS_REDUNDANCY_THRESHOLD)
    # Strictly decreasing as confidence rises: higher confidence -> lower
    # (more permissive) threshold -> collapses more readily. One function,
    # no jumps, no categories.
    assert all(thresholds[i] > thresholds[i + 1] for i in range(len(thresholds) - 1))


def test_effective_threshold_magnitude_at_extremes(monkeypatch):
    """Revision 1, item 1 (ZM): pins the *magnitude* of the shift, not just its
    direction/monotonicity -- a mutation that scales the span factor (e.g.
    `* 1.0` instead of `* 2.0`) changes these exact values and must fail this
    test. Floor is set low enough (0.0) that it never binds, isolating the
    raw formula from the clamp (the clamp itself is (f)'s job)."""
    monkeypatch.setattr(cc, "_CC_KISS_REGION_CONFIDENCE_ENABLED", True)
    monkeypatch.setattr(cc, "_CC_KISS_REGION_CONFIDENCE_SPAN", 0.1)
    monkeypatch.setattr(cc, "_CC_KISS_REGION_CONFIDENCE_FLOOR", 0.0)
    base = cc._CC_KISS_REDUNDANCY_THRESHOLD

    monkeypatch.setattr(cc, "cc_region_confidence", lambda g, v, e: 1.0)
    vdb = _CapturingVectorDB()
    cc._cc_kiss_find_redundant_node(_MiniGraph(), vdb, np.array([1.0, 0.0]))
    assert vdb.calls[0]["threshold"] == pytest.approx(base - 0.1)

    monkeypatch.setattr(cc, "cc_region_confidence", lambda g, v, e: 0.0)
    vdb = _CapturingVectorDB()
    cc._cc_kiss_find_redundant_node(_MiniGraph(), vdb, np.array([1.0, 0.0]))
    assert vdb.calls[0]["threshold"] == pytest.approx(min(1.0, base + 0.1))


def test_kiss_flag_on_but_pith_flag_off_yields_neutral_confidence_and_base_threshold(monkeypatch):
    """How the two flags interact: CC_KISS_REGION_CONFIDENCE_ENABLED only gates
    whether the KISS path calls cc_region_confidence at all. cc_region_confidence's
    own CC_PITH_REGION_CONFIDENCE_ENABLED gate is independent -- with THAT one
    off, cc_region_confidence fail-softs to neutral (0.5) regardless of who
    calls it, so the effective threshold still lands on the base threshold.
    Exercises the real cc_region_confidence (not mocked)."""
    monkeypatch.setattr(cc, "_CC_KISS_REGION_CONFIDENCE_ENABLED", True)
    monkeypatch.setattr(cc, "_CC_PITH_REGION_CONFIDENCE_ENABLED", False)

    vdb = _CapturingVectorDB()
    cc._cc_kiss_find_redundant_node(_MiniGraph(), vdb, np.array([1.0, 0.0]))
    assert vdb.calls[0]["threshold"] == pytest.approx(cc._CC_KISS_REDUNDANCY_THRESHOLD)


# ============================================================================
# (b) Reinforce, never drop: at high confidence a near-duplicate reinforces
#     the existing node -- node count unchanged, the turn still returns True.
# ============================================================================

def test_reinforce_not_drop_at_high_confidence(monkeypatch):
    graph = _MiniGraph()
    vector_db = SimpleVectorDB()
    existing = graph.create_node(
        "existing:conv", metadata={"cc": True, "creation_mode": "conversational"})
    vector_db.insert(id="existing:conv", embedding=np.array([1.0, 0.0], dtype=np.float32),
                      content="", metadata=existing.metadata)

    # cosine similarity ~0.90 to the stored node: BELOW the base threshold
    # (0.95, would not match today) but ABOVE the confidence-shifted effective
    # threshold at confidence=1.0 (default span=0.1/floor=0.85 -> 0.85). The
    # match only happens because of the shift.
    query_vec = _similar_vector([1.0, 0.0], 0.9)

    monkeypatch.setattr(cc, "_CC_KISS_REGION_CONFIDENCE_ENABLED", True)
    monkeypatch.setattr(cc, "cc_region_confidence", lambda g, v, e: 1.0)

    state = {"last_forest_id": None}
    result = cc.run_conversational_dual_pass(
        graph, vector_db, "near duplicate turn", query_vec, state)

    assert result is True
    assert len(graph.nodes) == 1  # no new node -- collapsed into the existing one
    assert existing.metadata.get("kiss_reinforcement_count") == 1
    assert state["last_forest_id"] == "existing:conv"


def test_reinforce_at_high_confidence_does_not_match_below_floor(monkeypatch):
    """Sanity converse of the above: the same ~0.90-similarity query does NOT
    match with the feature off (confirms the prior test's premise -- the base
    threshold of 0.95 really does exclude a 0.90 similarity)."""
    graph = _MiniGraph()
    vector_db = SimpleVectorDB()
    existing = graph.create_node(
        "existing:conv", metadata={"cc": True, "creation_mode": "conversational"})
    vector_db.insert(id="existing:conv", embedding=np.array([1.0, 0.0], dtype=np.float32),
                      content="", metadata=existing.metadata)
    query_vec = _similar_vector([1.0, 0.0], 0.9)

    monkeypatch.setattr(cc, "_CC_KISS_REGION_CONFIDENCE_ENABLED", False)
    result = cc._cc_kiss_find_redundant_node(graph, vector_db, query_vec)
    assert result is None


# ============================================================================
# (c) Novel input in a high-confidence region still deposits fresh.
# ============================================================================

def test_novel_input_in_high_confidence_region_deposits_fresh(monkeypatch):
    import ng_embed

    graph = _MiniGraph()
    vector_db = SimpleVectorDB()
    existing = graph.create_node(
        "existing:conv", metadata={"cc": True, "creation_mode": "conversational"})
    vector_db.insert(id="existing:conv", embedding=np.array([1.0, 0.0], dtype=np.float32),
                      content="", metadata=existing.metadata)

    # Orthogonal query: cosine similarity 0.0, nowhere near the confidence-
    # shifted effective threshold (floor 0.85 at confidence=1.0).
    novel_vec = np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(cc, "_CC_KISS_REGION_CONFIDENCE_ENABLED", True)
    monkeypatch.setattr(cc, "cc_region_confidence", lambda g, v, e: 1.0)

    class _FakeEmbedSingleton:
        def dual_record_outcome(self, ecosystem, content, embedding, target_id,
                                 success, strength=1.0, metadata=None):
            # Mirrors the one call this test needs from the real
            # NGEmbed.dual_record_outcome: deposit the forest via the eco
            # adapter. Tree-concept extraction is out of scope here.
            return ecosystem.record_outcome(embedding, target_id, success, strength, metadata)

    monkeypatch.setattr(ng_embed.NGEmbed, "get_instance",
                         classmethod(lambda cls, config=None: _FakeEmbedSingleton()))

    state = {"last_forest_id": None}
    result = cc.run_conversational_dual_pass(
        graph, vector_db, "a genuinely novel turn", novel_vec, state)

    assert result is True
    assert len(graph.nodes) == 2  # fresh node deposited alongside the existing one
    assert existing.metadata.get("kiss_reinforcement_count") is None  # not reinforced
    new_ids = [nid for nid in graph.nodes if nid != "existing:conv"]
    assert len(new_ids) == 1
    assert graph.nodes[new_ids[0]].metadata.get("creation_mode") == "conversational"


# ============================================================================
# (d) Flag off: cc_region_confidence is never called; the search threshold
#     equals _CC_KISS_REDUNDANCY_THRESHOLD.
# ============================================================================

def test_flag_off_no_confidence_call_and_threshold_equals_base(monkeypatch):
    monkeypatch.setattr(cc, "_CC_KISS_REGION_CONFIDENCE_ENABLED", False)
    mock_confidence = Mock(
        side_effect=AssertionError("cc_region_confidence must not be called when the flag is off"))
    monkeypatch.setattr(cc, "cc_region_confidence", mock_confidence)

    vdb = _CapturingVectorDB()
    result = cc._cc_kiss_find_redundant_node(_MiniGraph(), vdb, np.array([1.0, 0.0]))

    mock_confidence.assert_not_called()
    assert result is None
    assert vdb.calls == [{"k": 5, "threshold": cc._CC_KISS_REDUNDANCY_THRESHOLD}]


# ============================================================================
# (e) Read-only: graph and vector_db state are unchanged by the threshold
#     computation itself.
# ============================================================================

def test_threshold_computation_is_read_only(monkeypatch):
    monkeypatch.setattr(cc, "_CC_KISS_REGION_CONFIDENCE_ENABLED", True)
    monkeypatch.setattr(cc, "cc_region_confidence", lambda g, v, e: 0.75)

    graph = _MiniGraph()
    existing = graph.create_node("n1", metadata={"cc": True, "creation_mode": "conversational"})
    vector_db = SimpleVectorDB()
    vector_db.insert(id="n1", embedding=np.array([1.0, 0.0], dtype=np.float32),
                      content="", metadata=existing.metadata)

    nodes_before = dict(graph.nodes)
    node_meta_before = dict(existing.metadata)
    vdb_ids_before = set(vector_db.embeddings.keys())
    vdb_embeddings_before = {k: v.copy() for k, v in vector_db.embeddings.items()}
    vdb_metadata_before = {k: dict(v) for k, v in vector_db.metadata.items()}

    # Far enough away that nothing matches -- confirms the call is a pure read.
    novel = np.array([0.0, 1.0], dtype=np.float32)
    result = cc._cc_kiss_find_redundant_node(graph, vector_db, novel)

    assert result is None
    assert graph.nodes == nodes_before
    assert existing.metadata == node_meta_before
    assert set(vector_db.embeddings.keys()) == vdb_ids_before
    for node_id, emb in vdb_embeddings_before.items():
        assert np.array_equal(vector_db.embeddings[node_id], emb)
    assert vector_db.metadata == vdb_metadata_before


# ============================================================================
# (f) The clamp holds at confidence 0.0 and 1.0.
# ============================================================================

def test_floor_above_base_does_not_break_neutral_equals_base(monkeypatch):
    """Revision 1, item 2 (ZM): CC_KISS_REGION_CONFIDENCE_FLOOR set above
    _CC_KISS_REDUNDANCY_THRESHOLD must not make the neutral-confidence
    threshold silently diverge from base -- the clamp caps the floor it uses
    at base, so 'effective == base at neutral' holds for any FLOOR setting,
    not just the sane ones."""
    monkeypatch.setattr(cc, "_CC_KISS_REGION_CONFIDENCE_ENABLED", True)
    monkeypatch.setattr(cc, "_CC_KISS_REGION_CONFIDENCE_FLOOR", 1.0)  # > base (0.95)
    monkeypatch.setattr(cc, "cc_region_confidence",
                         lambda g, v, e: cc._CC_PITH_REGION_CONFIDENCE_NEUTRAL)

    vdb = _CapturingVectorDB()
    cc._cc_kiss_find_redundant_node(_MiniGraph(), vdb, np.array([1.0, 0.0]))
    assert vdb.calls[0]["threshold"] == pytest.approx(cc._CC_KISS_REDUNDANCY_THRESHOLD)


def test_clamp_holds_at_confidence_extremes(monkeypatch):
    monkeypatch.setattr(cc, "_CC_KISS_REGION_CONFIDENCE_ENABLED", True)
    monkeypatch.setattr(cc, "_CC_KISS_REGION_CONFIDENCE_SPAN", 0.3)
    monkeypatch.setattr(cc, "_CC_KISS_REGION_CONFIDENCE_FLOOR", 0.85)
    # base (_CC_KISS_REDUNDANCY_THRESHOLD) default is 0.95. Unclamped this
    # span would give 0.65 at confidence=1.0 and 1.25 at confidence=0.0 --
    # both outside [0.85, 1.0], so this exercises the clamp actually engaging
    # at both ends, not just landing on the boundary by coincidence.
    for confidence, expected in ((1.0, 0.85), (0.0, 1.0)):
        monkeypatch.setattr(cc, "cc_region_confidence", lambda g, v, e, c=confidence: c)
        vdb = _CapturingVectorDB()
        cc._cc_kiss_find_redundant_node(_MiniGraph(), vdb, np.array([1.0, 0.0]))
        assert vdb.calls[0]["threshold"] == pytest.approx(expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
