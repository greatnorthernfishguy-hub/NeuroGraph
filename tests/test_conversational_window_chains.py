# ---- Changelog ----
# [2026-09-26] openrouter/deepseek/deepseek-v4.1-flash (OpenCode harness on T3 Code),
#   lane z2-dualpass-reconcile-20260926 — P264(2) follow-up (Addendum 1).
# What: the cc_mem fixture no longer calls
#   monkeypatch.setattr(org, "_CC_KISS_GATE_ENABLED", False); with the lane-B
#   Delta Gate removal there is no such symbol to disable.
# Why: main's d9106a3 (#523) deleted _CC_KISS_GATE_ENABLED; this file was based
#   pre-lane-B and its cc_mem tests errored at fixture setup after the merge.
#   LAW 3 — the Delta Gate stays removed; the symbol is never re-added.
# How: delete that one setattr line. No symbol re-added, no raising=False.
# -------------------
# [2026-09-22] Grok 4.6 — punchlist-001 B1: CC organism window chains
# What: Mirror the RPC 5a/5b assertions on cc_ng_organism.run_conversational_dual_pass.
# Why:  CC's parameterized copy of the conversational path must grow the same
#       graph-only window topology. Forest-only is not an outcome.
# How:  FakeGraph + SimpleVectorDB + _step_lock; same patches.
# [2026-09-21] Grok 4.6 — Lane 3 §7 polychrony window chains (graph-only)
# What: Long-turn window nodes live in the SNN and stay out of recall vdb;
#       delay-chained in order with the #257 sampler; forest-linked both
#       ways. Short turns create no window nodes.
# Why:  Plan Task 10 / spec §7 — "We have polychrony, we use polychrony."
# How:  Patch embed_windows + _extract_concepts=[] so forest lands via the
#       real eco; assert forest in vdb, windows in graph, windows absent
#       from vector_db.all_ids().
# -------------------

import hashlib
import os
import sys

import numpy as np
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_memory_phase1 import _FakeGraph


@pytest.fixture
def rpc_mem(monkeypatch):
    import neurograph_rpc as rpc
    from universal_ingestor import SimpleVectorDB
    old, old_last = rpc._memory, getattr(rpc, "_last_conv_forest_id", None)
    mem = type("M", (), {})()
    mem.vector_db = SimpleVectorDB()
    mem.graph = _FakeGraph()
    rpc._memory = mem
    rpc._last_conv_forest_id = None
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    yield rpc, mem
    rpc._memory = old
    rpc._last_conv_forest_id = old_last


def _tid(text):
    return "conv::" + hashlib.sha1(text.encode()).hexdigest()


def test_long_turn_window_nodes_in_graph_absent_from_vdb(rpc_mem):
    rpc, mem = rpc_mem
    from ng_embed import EmbedWindow, WindowedEmbedding
    text = "long-turn-placeholder"
    forest_vec = np.arange(768, dtype=np.float32) + 1.0
    w1 = EmbedWindow(text="w1", embedding=np.ones(768, np.float32), token_count=512)
    w2 = EmbedWindow(text="w2", embedding=np.ones(768, np.float32) * 2, token_count=200)
    we = WindowedEmbedding(pooled=np.ones(768, np.float32) * 9, windows=(w1, w2), token_count=712)

    with patch("ng_embed.NGEmbed._extract_concepts", return_value=[]), \
         patch("ng_embed.NGEmbed.embed_windows", return_value=we):
        ok = rpc._run_conversational_dual_pass(text, forest_vec)
    assert ok is True
    fid = _tid(text)
    wids = [f"{fid}::window::0", f"{fid}::window::1"]
    for wid in wids:
        assert wid in mem.graph.nodes, f"{wid} must be an SNN node"
        assert wid not in mem.vector_db.all_ids(), f"{wid} must NOT enter recall"
        assert mem.graph.nodes[wid].metadata.get("_window") is True
        assert mem.graph.nodes[wid].metadata.get("_forest_id") == fid
        assert "poincare_dir" in mem.graph.nodes[wid].metadata
    assert mem.graph.nodes[wids[0]].metadata.get("_window_index") == 0
    assert mem.graph.nodes[wids[1]].metadata.get("_window_index") == 1
    pairs = [(a, b, d) for (a, b, _w, d) in mem.graph.synapses]
    assert any(a == wids[0] and b == wids[1] and d >= 2 for a, b, d in pairs)
    weights = {(a, b): w for (a, b, w, _d) in mem.graph.synapses}
    for wid in wids:
        assert weights.get((fid, wid)) == 0.2
        assert weights.get((wid, fid)) == 0.15
    assert fid in mem.graph.nodes
    assert fid in mem.vector_db.all_ids()
    stored = mem.vector_db.get(fid)["embedding"]
    expected = forest_vec / np.linalg.norm(forest_vec)
    np.testing.assert_allclose(stored, expected, rtol=1e-5, atol=1e-5)


def test_short_turn_creates_no_window_nodes(rpc_mem):
    rpc, mem = rpc_mem
    from ng_embed import WindowedEmbedding
    we = WindowedEmbedding(pooled=np.ones(768, np.float32), windows=(), token_count=12)
    text = "short"
    with patch("ng_embed.NGEmbed._extract_concepts", return_value=[]), \
         patch("ng_embed.NGEmbed.embed_windows", return_value=we):
        ok = rpc._run_conversational_dual_pass(text, np.ones(768, np.float32))
    assert ok is True
    fid = _tid(text)
    assert fid in mem.graph.nodes
    assert fid in mem.vector_db.all_ids()
    assert not any("::window::" in k for k in mem.graph.nodes)
    assert mem.graph.synapses == []


@pytest.fixture
def cc_mem(monkeypatch):
    import threading
    import cc_ng_organism as org
    from universal_ingestor import SimpleVectorDB
    g = _FakeGraph()
    g._step_lock = threading.RLock()
    vdb = SimpleVectorDB()
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    yield org, g, vdb


def _cc_tid(text):
    return "cc:conv::" + hashlib.sha1(text.encode()).hexdigest()


def test_cc_long_turn_window_nodes_in_graph_absent_from_vdb(cc_mem):
    org, graph, vdb = cc_mem
    from ng_embed import EmbedWindow, WindowedEmbedding
    text = "long-turn-placeholder"
    forest_vec = np.arange(768, dtype=np.float32) + 1.0
    w1 = EmbedWindow(text="w1", embedding=np.ones(768, np.float32), token_count=512)
    w2 = EmbedWindow(text="w2", embedding=np.ones(768, np.float32) * 2, token_count=200)
    we = WindowedEmbedding(pooled=np.ones(768, np.float32) * 9, windows=(w1, w2), token_count=712)

    with patch("ng_embed.NGEmbed._extract_concepts", return_value=[]), \
         patch("ng_embed.NGEmbed.embed_windows", return_value=we):
        ok = org.run_conversational_dual_pass(graph, vdb, text, forest_vec, {"last_forest_id": None})
    assert ok is True
    fid = _cc_tid(text)
    wids = [f"{fid}::window::0", f"{fid}::window::1"]
    for wid in wids:
        assert wid in graph.nodes, f"{wid} must be an SNN node"
        assert wid not in vdb.all_ids(), f"{wid} must NOT enter recall"
        assert graph.nodes[wid].metadata.get("_window") is True
        assert graph.nodes[wid].metadata.get("_forest_id") == fid
        assert "poincare_dir" in graph.nodes[wid].metadata
    assert graph.nodes[wids[0]].metadata.get("_window_index") == 0
    assert graph.nodes[wids[1]].metadata.get("_window_index") == 1
    pairs = [(a, b, d) for (a, b, _w, d) in graph.synapses]
    assert any(a == wids[0] and b == wids[1] and d >= 2 for a, b, d in pairs)
    weights = {(a, b): w for (a, b, w, _d) in graph.synapses}
    for wid in wids:
        assert weights.get((fid, wid)) == 0.2
        assert weights.get((wid, fid)) == 0.15
    assert fid in graph.nodes
    assert fid in vdb.all_ids()
    stored = vdb.get(fid)["embedding"]
    expected = forest_vec / np.linalg.norm(forest_vec)
    np.testing.assert_allclose(stored, expected, rtol=1e-5, atol=1e-5)


def test_cc_short_turn_creates_no_window_nodes(cc_mem):
    org, graph, vdb = cc_mem
    from ng_embed import WindowedEmbedding
    we = WindowedEmbedding(pooled=np.ones(768, np.float32), windows=(), token_count=12)
    text = "short"
    with patch("ng_embed.NGEmbed._extract_concepts", return_value=[]), \
         patch("ng_embed.NGEmbed.embed_windows", return_value=we):
        ok = org.run_conversational_dual_pass(
            graph, vdb, text, np.ones(768, np.float32), {"last_forest_id": None},
        )
    assert ok is True
    fid = _cc_tid(text)
    assert fid in graph.nodes
    assert fid in vdb.all_ids()
    assert not any("::window::" in k for k in graph.nodes)
    assert graph.synapses == []
