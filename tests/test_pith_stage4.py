# ---- Changelog ----
# [2026-07-22] Claude Code (Sonnet 5) — Pith Stage 4 phase 5a unit tests (#55)
# What: first tests for Stage 4 predictive promotion — the promotion-at-recall
#   extension to cc_pattern_completion_recall()'s anticipatory bonus block
#   (4a), proximity-keyed LOD staging via _cc_node_query_distance (4c), and
#   the promoted_predicted/prefetch_hits PithMetrics counters.
# Why: docs/superpowers/plans/2026-07-22-pith-stage4-spec.md sec 5a/6 — the
#   spec's own def-of-done demands dedicated tests (prior "Stage 4 built"
#   claims were false; this build must not repeat that).
# How: real NeuroGraphMemory fixture (same pattern as
#   test_cc_retrieval_enrichment.py) — promotion needs a real harvest to prove
#   a node the query-driven harvest genuinely misses. Predicted nodes are kept
#   OUT of the vdb and unwired from any fired seed so harvest structurally
#   cannot find them; they get content via node.metadata['_forest_content']
#   directly (resolve_surface_content's substrate-first path) instead of a
#   vdb insert. LOD tests stamp 'poincare_dir' directly (like the existing
#   GSG rescore tests' _mk_gsg_node helper) for deterministic near/far control.
# -------------------
"""Unit tests for Pith Stage 4 phase 5a — predictive promotion (#55)."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import tempfile, shutil
import numpy as np
import pytest

import cc_ng_organism as cc
from cc_ng_organism import cc_pattern_completion_recall, _cc_node_query_distance


@pytest.fixture
def cc_ng():
    from openclaw_hook import NeuroGraphMemory
    workspace = tempfile.mkdtemp(prefix='pith_stage4_test_')
    ng = NeuroGraphMemory(workspace_dir=workspace,
                          config={"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}})
    yield ng
    shutil.rmtree(workspace, ignore_errors=True)


@pytest.fixture(autouse=True)
def _isolate_metrics():
    cc._PITH_METRICS.reset()
    yield
    cc._PITH_METRICS.reset()


def _make_predicted_node(g, nid, content="the substrate remembers what the harvest cannot reach"):
    """A node deliberately kept OUT of the vdb and unwired from any seed --
    structurally unreachable by ng._harvest_associations() -- with resolvable
    content stamped directly (substrate-first path resolve_surface_content
    reads first)."""
    g.create_node(node_id=nid)
    g.nodes[nid].metadata["_forest_content"] = content
    return g.nodes[nid]


class FakeNode:
    def __init__(self, metadata=None, diffpc_layer=0, manifold_type="hyperbolic"):
        self.metadata = metadata or {}
        self.diffpc_layer = diffpc_layer
        self.manifold_type = manifold_type


# ---- gate default / byte-identical off ----

def test_prefetch_gate_defaults_off():
    assert cc._CC_PITH_PREFETCH_ENABLED is False


def test_promotion_noop_when_gated_off(cc_ng, monkeypatch):
    monkeypatch.setattr(cc, "_CC_PITH_PREFETCH_ENABLED", False)
    g = cc_ng.graph
    _make_predicted_node(g, "predicted")
    state = {"primed_nodes": {"predicted": (1.0, time.time() + 60)}}
    results = cc_pattern_completion_recall(cc_ng, "anything at all", k=5, threshold=0.3, state=state)
    ids = [r["node_id"] for r in results]
    assert "predicted" not in ids
    assert cc._PITH_METRICS.promoted_predicted == 0


# ---- promotion at recall (4a) ----

def test_promotion_injects_unsurfaced_primed_node(cc_ng, monkeypatch):
    monkeypatch.setattr(cc, "_CC_PITH_PREFETCH_ENABLED", True)
    g = cc_ng.graph
    _make_predicted_node(g, "predicted")
    state = {"primed_nodes": {"predicted": (5.0, time.time() + 60)}}
    results = cc_pattern_completion_recall(cc_ng, "query with no relation to it", k=5,
                                            threshold=0.3, state=state)
    ids = [r["node_id"] for r in results]
    assert "predicted" in ids, (
        "a live primed node the query harvest cannot reach must be promoted "
        "as a candidate -- got %r" % ids)
    assert cc._PITH_METRICS.promoted_predicted == 1
    assert cc._PITH_METRICS.prefetch_hits == 1


def test_promotion_expired_primed_node_not_injected(cc_ng, monkeypatch):
    monkeypatch.setattr(cc, "_CC_PITH_PREFETCH_ENABLED", True)
    g = cc_ng.graph
    _make_predicted_node(g, "stale")
    state = {"primed_nodes": {"stale": (5.0, time.time() - 1.0)}}  # already expired
    results = cc_pattern_completion_recall(cc_ng, "irrelevant query", k=5, threshold=0.3, state=state)
    ids = [r["node_id"] for r in results]
    assert "stale" not in ids
    assert cc._PITH_METRICS.promoted_predicted == 0


def test_promotion_dangling_node_id_skipped_without_crash(cc_ng, monkeypatch):
    monkeypatch.setattr(cc, "_CC_PITH_PREFETCH_ENABLED", True)
    state = {"primed_nodes": {"ghost-not-in-graph": (5.0, time.time() + 60)}}
    results = cc_pattern_completion_recall(cc_ng, "irrelevant query", k=5, threshold=0.3, state=state)
    ids = [r["node_id"] for r in results]
    assert "ghost-not-in-graph" not in ids
    assert cc._PITH_METRICS.promoted_predicted == 0


def test_promotion_already_surfaced_node_not_double_counted(cc_ng, monkeypatch):
    """A primed node the harvest ALSO found gets only the existing bonus
    treatment (#256), not a second promoted-candidate injection."""
    from ng_embed import embed
    monkeypatch.setattr(cc, "_CC_PITH_PREFETCH_ENABLED", True)
    g, vdb = cc_ng.graph, cc_ng.vector_db
    g.create_node(node_id="both")
    vdb.insert(id="both", embedding=embed("daemon restart procedure"), content="daemon restart procedure")
    g.nodes["both"].threshold = 0.1
    state = {"primed_nodes": {"both": (5.0, time.time() + 60)}}
    results = cc_pattern_completion_recall(cc_ng, "daemon restart procedure", k=5, threshold=0.2, state=state)
    ids = [r["node_id"] for r in results]
    assert ids.count("both") <= 1
    assert cc._PITH_METRICS.promoted_predicted == 0


def test_promotion_is_pure_additive_not_a_hard_override(cc_ng, monkeypatch):
    """A weak (bonus-only) promoted prediction still loses to a strong,
    genuinely query-relevant harvest hit -- promotion adds a candidate, it
    does not override the rank."""
    from ng_embed import embed
    monkeypatch.setattr(cc, "_CC_PITH_PREFETCH_ENABLED", True)
    g, vdb = cc_ng.graph, cc_ng.vector_db
    query = "the lenia distance cache rebuild on the vps"
    g.create_node(node_id="seed")
    vdb.insert(id="seed", embedding=embed(query), content="lenia distance cache rebuild")
    g.nodes["seed"].threshold = 0.1
    _make_predicted_node(g, "predicted")
    state = {"primed_nodes": {"predicted": (5.0, time.time() + 60)}}
    results = cc_pattern_completion_recall(cc_ng, query, k=5, threshold=0.3, state=state)
    ids = [r["node_id"] for r in results]
    assert ids[0] == "seed", "the strong harvest hit must still rank first -- got %r" % ids


# ---- proximity-keyed LOD (4c) ----

def test_node_query_distance_none_without_stamp():
    node = FakeNode(metadata={})
    assert _cc_node_query_distance(node, np.zeros(4)) is None


def test_node_query_distance_hyperbolic_zero_for_identical_direction():
    qdir = np.array([0.3, 0.1, 0.0, 0.0])
    node = FakeNode(metadata={"poincare_dir": qdir.tolist()})
    assert _cc_node_query_distance(node, qdir) == pytest.approx(0.0, abs=1e-6)


def test_node_query_distance_spherical_branch():
    qdir = np.array([1.0, 0.0])
    node = FakeNode(metadata={"poincare_dir": [0.0, 1.0]}, manifold_type="spherical")
    dist = _cc_node_query_distance(node, qdir)
    assert dist == pytest.approx(np.pi / 2, abs=1e-4)   # orthogonal -> quarter turn


def test_promotion_lod_keeps_near_content_full(cc_ng, monkeypatch):
    from ng_embed import embed
    from cc_ng_organism import _cc_embed_to_poincare_dir
    monkeypatch.setattr(cc, "_CC_PITH_PREFETCH_ENABLED", True)
    monkeypatch.setattr(cc, "_CC_PITH_PREFETCH_LOD_DIST", 0.05)  # tight, deterministic threshold
    g = cc_ng.graph
    query = "checkpoint save cadence on the vps"
    qdir = _cc_embed_to_poincare_dir(embed(query))
    long_content = ("the checkpoint save cadence changed after the rebuild. " * 4).strip()
    assert len(long_content) > cc._CC_PITH_PREFETCH_SUMMARY_CHARS
    _make_predicted_node(g, "near", content=long_content)
    g.nodes["near"].metadata["poincare_dir"] = qdir.tolist()   # identical direction -> distance 0
    state = {"primed_nodes": {"near": (5.0, time.time() + 60)}}
    results = cc_pattern_completion_recall(cc_ng, query, k=5, threshold=0.3, state=state)
    item = next(r for r in results if r["node_id"] == "near")
    assert "⋯[+" not in item["content"], "near prediction must stage at full resolution"


def test_promotion_lod_summarizes_far_content(cc_ng, monkeypatch):
    from ng_embed import embed
    from cc_ng_organism import _cc_embed_to_poincare_dir
    monkeypatch.setattr(cc, "_CC_PITH_PREFETCH_ENABLED", True)
    monkeypatch.setattr(cc, "_CC_PITH_PREFETCH_LOD_DIST", 0.05)  # tight, deterministic threshold
    g = cc_ng.graph
    query = "checkpoint save cadence on the vps"
    qdir = _cc_embed_to_poincare_dir(embed(query))
    long_content = ("the checkpoint save cadence changed after the rebuild. " * 4).strip()
    assert len(long_content) > cc._CC_PITH_PREFETCH_SUMMARY_CHARS
    _make_predicted_node(g, "far", content=long_content)
    g.nodes["far"].metadata["poincare_dir"] = (-qdir).tolist()   # opposite direction -> distance > 0
    state = {"primed_nodes": {"far": (5.0, time.time() + 60)}}
    results = cc_pattern_completion_recall(cc_ng, query, k=5, threshold=0.3, state=state)
    item = next(r for r in results if r["node_id"] == "far")
    assert "⋯[+" in item["content"], "far prediction must be staged as a keyframe summary"
    assert len(item["content"]) < len(long_content)


def test_promotion_lod_no_stamp_stays_full(cc_ng, monkeypatch):
    """No poincare_dir stamp -> _cc_node_query_distance returns None -> can't
    tell how far it is -> don't downgrade (fail-soft-toward-full, not toward
    dropping information)."""
    monkeypatch.setattr(cc, "_CC_PITH_PREFETCH_ENABLED", True)
    monkeypatch.setattr(cc, "_CC_PITH_PREFETCH_LOD_DIST", 0.05)
    g = cc_ng.graph
    long_content = ("no gsg stamp on this node at all, ever, so distance is unknown. " * 3).strip()
    _make_predicted_node(g, "unstamped", content=long_content)
    state = {"primed_nodes": {"unstamped": (5.0, time.time() + 60)}}
    results = cc_pattern_completion_recall(cc_ng, "some query", k=5, threshold=0.3, state=state)
    item = next(r for r in results if r["node_id"] == "unstamped")
    assert "⋯[+" not in item["content"]


# ---- metrics (§13.3) ----

def test_metrics_reset_zeroes_new_counters():
    cc._PITH_METRICS.promoted_predicted = 3
    cc._PITH_METRICS.prefetch_hits = 2
    cc._PITH_METRICS.reset()
    assert cc._PITH_METRICS.promoted_predicted == 0
    assert cc._PITH_METRICS.prefetch_hits == 0


def test_metrics_snapshot_includes_new_fields():
    snap = cc._PITH_METRICS.snapshot()
    assert "promoted_predicted" in snap
    assert "prefetch_hits" in snap


def test_metrics_promoted_predicted_and_prefetch_hits_increment_together(cc_ng, monkeypatch):
    monkeypatch.setattr(cc, "_CC_PITH_PREFETCH_ENABLED", True)
    g = cc_ng.graph
    _make_predicted_node(g, "predicted")
    state = {"primed_nodes": {"predicted": (5.0, time.time() + 60)}}
    cc_pattern_completion_recall(cc_ng, "unrelated query text", k=5, threshold=0.3, state=state)
    assert cc._PITH_METRICS.promoted_predicted == 1
    assert cc._PITH_METRICS.prefetch_hits == 1


def test_metrics_untouched_when_gate_off(cc_ng, monkeypatch):
    monkeypatch.setattr(cc, "_CC_PITH_PREFETCH_ENABLED", False)
    g = cc_ng.graph
    _make_predicted_node(g, "predicted")
    state = {"primed_nodes": {"predicted": (5.0, time.time() + 60)}}
    cc_pattern_completion_recall(cc_ng, "unrelated query text", k=5, threshold=0.3, state=state)
    assert cc._PITH_METRICS.promoted_predicted == 0
    assert cc._PITH_METRICS.prefetch_hits == 0


# ---- fail-soft ----

def test_recall_still_fails_soft_with_promotion_enabled(monkeypatch):
    monkeypatch.setattr(cc, "_CC_PITH_PREFETCH_ENABLED", True)
    assert cc_pattern_completion_recall(None, "anything") == []
    class BrokenNg:
        graph = None
    assert cc_pattern_completion_recall(BrokenNg(), "anything", state={"primed_nodes": {}}) == []


# ---- phase 5b (2026-09-05): seed helper + prefetch_surfaced counter ----

def test_prefetch_seed_filters_expired_and_never_raises():
    now = time.time()
    st = {"primed_nodes": {"live": (0.7, now + 60), "dead": (0.9, now - 1), "half": (0.35, now + 60)}}
    seed = cc.pith_prefetch_seed(st)
    assert seed == {"live": 1.0, "half": 0.5}      # normalised by the live max; expired dropped
    assert cc.pith_prefetch_seed(None) == {}
    assert cc.pith_prefetch_seed({"primed_nodes": "garbage"}) == {}


def test_prefetch_surfaced_counts_harvest_found_primed_node(cc_ng, monkeypatch):
    """A live primed node the harvest surfaces on its own is the true prefetch hit."""
    monkeypatch.setattr(cc, "_CC_PITH_PREFETCH_ENABLED", False)  # no injection help
    g = cc_ng.graph
    nid = "seedme"
    g.create_node(node_id=nid)
    g.nodes[nid].metadata["_forest_content"] = "seedme content the query matches"
    state = {"primed_nodes": {nid: (1.0, time.time() + 60)}}
    # Force the harvest to surface it by stubbing the association harvest.
    monkeypatch.setattr(cc_ng, "_harvest_associations",
                        lambda q, novelty=None: [{"node_id": nid, "strength": 0.5}])
    cc_pattern_completion_recall(cc_ng, "seedme content", k=5, threshold=0.0, state=state)
    assert cc._PITH_METRICS.prefetch_surfaced == 1
    assert cc._PITH_METRICS.promoted_predicted == 0
    assert "prefetch_surfaced" in cc._PITH_METRICS.snapshot()
