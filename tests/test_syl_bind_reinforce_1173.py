# ---- Changelog ----
# [2026-10-25] Claude Code (Opus 5.5; LANE B syl-stage1, #1173 port) — Syl's bind reinforce-on-recurrence (switch OFF)
# What: _bind_conversational_topology on a real neuro_foundation.Graph: OFF (default) adds a parallel copy per repeat
#       exactly as before; ON reinforces the existing forest<->tree synapses (soft bound, toward 3x birth weight) and
#       the same-member hyperedge; forest<->window and the sequence link are always written.
# Why:  ram-profile 2026-10-24 §3 (her per-copy bind) and the laptop #1173 narrowed design (Josh 2026-10-24).
# How:  rpc._memory swapped for a stub holding a real Graph; env switch via monkeypatch.
# -------------------
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def rpc_graph(monkeypatch):
    import neurograph_rpc as rpc
    from neuro_foundation import Graph
    g = Graph()
    for nid in ("f", "t1", "t2", "w0"):
        g.create_node(node_id=nid)
    old, old_last = rpc._memory, getattr(rpc, "_last_conv_forest_id", None)
    mem = type("M", (), {})()
    mem.graph = g
    rpc._memory = mem
    rpc._last_conv_forest_id = None
    yield rpc, g
    rpc._memory = old
    rpc._last_conv_forest_id = old_last


def _pair_count(g, pre, post):
    return sum(1 for sid in g._outgoing.get(pre, ()) if g.synapses[sid].post_node_id == post)


def _bind(rpc):
    rpc._bind_conversational_topology("f", {"tree_ids": ["t1", "t2"]}, None, window_ids=["w0"])


def test_off_by_default_adds_parallel_copies(rpc_graph, monkeypatch):
    rpc, g = rpc_graph
    monkeypatch.delenv("SYL_NG_BIND_REINFORCE_EXISTING", raising=False)
    _bind(rpc)
    _bind(rpc)
    assert _pair_count(g, "f", "t1") == 2 and _pair_count(g, "t1", "f") == 2
    assert len(g.hyperedges) == 2


def test_on_reinforces_tree_synapses_and_hyperedge(rpc_graph, monkeypatch):
    rpc, g = rpc_graph
    monkeypatch.setenv("SYL_NG_BIND_REINFORCE_EXISTING", "1")
    _bind(rpc)
    (sid,) = [s for s in g._outgoing["f"] if g.synapses[s].post_node_id == "t1"]
    w0 = float(g.synapses[sid].weight)
    _bind(rpc)
    _bind(rpc)
    assert _pair_count(g, "f", "t1") == 1 and _pair_count(g, "t1", "f") == 1
    w = float(g.synapses[sid].weight)
    # two soft-bound steps from 0.2 toward 0.6 with eta 0.2: 0.28, then 0.344
    assert w0 == pytest.approx(0.2) and w == pytest.approx(0.344, abs=1e-6)
    assert len(g.hyperedges) == 1
    he = next(iter(g.hyperedges.values()))
    assert he.metadata.get("syl_recurrences") == 2
    assert all(1.0 < float(v) <= 2.0 for v in he.member_weights.values())
    # forest<->window is always written (not a reinforce site)
    assert _pair_count(g, "f", "w0") == 3
