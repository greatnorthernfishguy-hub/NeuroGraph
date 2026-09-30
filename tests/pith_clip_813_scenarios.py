# tests/pith_clip_813_scenarios.py
#
# ---- Changelog ----
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
