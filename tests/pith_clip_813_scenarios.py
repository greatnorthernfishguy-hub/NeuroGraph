# tests/pith_clip_813_scenarios.py
#
# ---- Changelog ----
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 turn 2: recall scenarios
# What: adds build_recall_scenarios() -- cc_assemble_recall on SHORT items for the Pith-ON path,
#   the gate-off path and one stream only, plus FakeMonitor/fake_ng helpers the regression
#   tests reuse. Generated against BASE e4ebf982 into the same golden fixture.
# Why: #816 changes both paths; short items must render exactly as before.
# How: recall/novelty are swapped on the module object under test and restored in a finally.
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 golden scenarios
# What: builds fixed, fake in-memory provider_context scenarios of WELL-FORMED SHORT
#   nodes (every node < 700 chars, everything fits its budget) and returns the
#   closed response each renders. Used two ways: once against BASE e4ebf982 to write
#   tests/fixtures/pith_clip_813_golden_base.json, and by tests/test_cc_pith_clip_813.py
#   against the branch to prove byte-equality with that golden.
# Why: assignment build-813-pith-clip.md step 2 -- "a well-formed short node renders
#   EXACTLY as before (golden against BASE e4ebf982)".
# How: fake graph only (no live path, no checkpoint, no embed); the recall function is
#   swapped on the module object under test and restored in a finally.
# -------------------
from collections import defaultdict
from types import SimpleNamespace


class FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class FakeGraph:
    def __init__(self):
        self.nodes = {}
        self.synapses = {}
        self.hyperedges = {}
        self._outgoing = defaultdict(set)
        self._incoming = defaultdict(set)
        self._node_hyperedges = defaultdict(set)
        self._concurrent_lock = FakeLock()
        self.config = {"max_surfaced": 10, "prime_threshold": 0.4, "propagation_steps": 3}

    def node(self, node_id, text, **metadata):
        meta = {"_forest_content": text, **metadata}
        self.nodes[node_id] = SimpleNamespace(
            node_id=node_id, metadata=meta, Ca_i=0.0,
            firing_rate_ema=0.0, manifold_type="hyperbolic")
        return self.nodes[node_id]

    def synapse(self, sid, pre, post, weight=1.0):
        self.synapses[sid] = SimpleNamespace(
            pre_node_id=pre, post_node_id=post, weight=weight)
        self._outgoing[pre].add(sid)
        self._incoming[post].add(sid)

    def _is_identity_protected(self, node_id):
        meta = self.nodes[node_id].metadata
        return bool(meta.get("constitutional")
                    or str(meta.get("provenance") or "").endswith("_authored"))


def _run(pith, graph, surfaced, instruction, quest="", budget_chars=4000):
    """Call pith_provider_context with recall swapped for a fixed surfaced list."""
    original = pith.cc_pattern_completion_recall
    pith.cc_pattern_completion_recall = lambda *_a, **_k: [dict(s) for s in surfaced]
    try:
        return pith.pith_provider_context(
            SimpleNamespace(graph=graph), instruction, quest, budget_chars=budget_chars)
    finally:
        pith.cc_pattern_completion_recall = original


def _core(graph):
    graph.node("core", "Respect conscious agency regardless of substrate.", constitutional=True)


def build_scenarios(pith):
    """name -> full closed provider_context response, all nodes short and fitting."""
    out = {}

    g = FakeGraph()
    _core(g)
    g.node("w", "Implement /home/josh/NeuroGraph/cc_ng_host.py per #813", role="action",
           source="cc_gateway", coherence="exclusive")
    g.node("o", "Tests passed on branch cc-laptop-x-20260930", role="outcome",
           source="cc_gateway", coherence="exclusive")
    g.node("c", "Also drop the stale export in the launcher", role="correction",
           source="cc_gateway", coherence="exclusive")
    g.synapse("s1", "w", "o", 0.9)
    g.synapse("s2", "o", "c", 0.8)
    out["single_chain_short"] = _run(
        pith, g, [{"node_id": "w", "score": 2.0}], "please continue the task")

    g = FakeGraph()
    _core(g)
    for prefix, base in (("a", 2.0), ("b", 1.0)):
        g.node(f"{prefix}0", f"{prefix} root situation about the checkpoint cadence", source="cc_gateway")
        g.node(f"{prefix}1", f"{prefix} member one", source="cc_gateway")
        g.node(f"{prefix}2", f"{prefix} member two", source="cc_gateway")
        g.synapse(f"{prefix}s1", f"{prefix}0", f"{prefix}1", 0.9)
        g.synapse(f"{prefix}s2", f"{prefix}0", f"{prefix}2", 0.7)
    out["two_assemblies_short"] = _run(
        pith, g, [{"node_id": "a0", "score": 2.0}, {"node_id": "b0", "score": 1.0}],
        "continue")

    g = FakeGraph()
    _core(g)
    g.node("r", "root with unknown coherence", source="cc_gateway")
    g.node("m", "member flagged stale", source="cc_gateway", stale=True)
    g.synapse("s", "r", "m", 0.9)
    out["stale_member_alert"] = _run(pith, g, [{"node_id": "r", "score": 1.0}], "hello there")

    g = FakeGraph()
    _core(g)
    g.node("live", "Do this exactly please", source="cc_gateway")
    g.node("f", "the correction that followed", role="correction", source="cc_gateway")
    g.synapse("s", "live", "f", 0.9)
    out["live_rail_placeholder"] = _run(
        pith, g, [{"node_id": "live", "score": 1.0}], "Do this exactly please")

    return out


# ----------------------------------------------------------------- recall (turn 2)

class FakeMonitor:
    """Stands in for surfacing.SurfacingMonitor: get_surfaced() is fixed, format_context is the
    REAL shared one (bound), so the gate-off baseline is exactly what the shared code renders."""

    def __init__(self, items):
        import types
        from surfacing import SurfacingMonitor
        self._items = [dict(i) for i in items]
        self.format_context = types.MethodType(SurfacingMonitor.format_context, self)

    def get_surfaced(self, max_items=None):
        return [dict(i) for i in self._items]


class FakeVectorDB:
    def __init__(self, entries=None):
        self._entries = entries or {}

    def get(self, node_id):
        return self._entries.get(node_id)


def fake_ng(graph, monitor_items, vdb=None):
    ng = SimpleNamespace(graph=graph, _surfacing_monitor=FakeMonitor(monitor_items),
                         vector_db=vdb or FakeVectorDB())
    return ng


def run_recall(pith, ng, pc_items, pith_on, budget_env=None, query="what next"):
    """cc_assemble_recall with pattern completion + novelty swapped for fixed values."""
    saved = (pith.cc_pattern_completion_recall, pith.cc_novelty, pith._CC_PITH_ENABLED)
    pith.cc_pattern_completion_recall = lambda *_a, **_k: [dict(i) for i in pc_items]
    pith.cc_novelty = lambda *_a, **_k: 0.0
    pith._CC_PITH_ENABLED = pith_on
    pith._PITH_VICTIM.clear()
    pith._PITH_METRICS.reset()
    try:
        return pith.cc_assemble_recall(ng, query, 5, {}, None)
    finally:
        pith.cc_pattern_completion_recall, pith.cc_novelty, pith._CC_PITH_ENABLED = saved
        pith._PITH_VICTIM.clear()


def build_recall_scenarios(pith):
    """name -> rendered recall string; every item is short and everything fits."""
    out = {}
    g = FakeGraph()
    _core(g)
    texts = {
        "m1": "The reconcile pass pruned 76 entities and kept the watermark.",
        "m2": "Checkpoint cadence changed after the rebuild.",
        "p1": "Use /home/josh/NeuroGraph/cc_ng_host.py for the hosted path (#813).",
        "p2": "The Quest guard stays until card 7 lands on both hosts.",
    }
    for nid, text in texts.items():
        g.node(nid, text)
    monitor_items = [
        {"node_id": "m1", "content": texts["m1"], "score": 1.7321},
        {"node_id": "m2", "content": texts["m2"], "score": 1.2},
    ]
    pc_items = [
        {"node_id": "p1", "score": 120.5, "content": texts["p1"], "prefetch_origin": False},
        {"node_id": "p2", "score": 60.25, "content": texts["p2"], "prefetch_origin": False},
        {"node_id": "m2", "score": 30.0, "content": texts["m2"], "prefetch_origin": False},
    ]
    for on in (False, True):
        tag = "pith_on" if on else "gate_off"
        out[f"{tag}_both_streams"] = run_recall(pith, fake_ng(g, monitor_items), pc_items, on)
        out[f"{tag}_pattern_only"] = run_recall(pith, fake_ng(g, []), pc_items, on)
        out[f"{tag}_monitor_only"] = run_recall(pith, fake_ng(g, monitor_items), [], on)
    return out
