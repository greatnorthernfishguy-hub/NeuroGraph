# ---- Changelog ----
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
