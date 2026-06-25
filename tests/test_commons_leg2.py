"""
Commons leg-2 test — the salience-gated enhance-loop, in a sandbox SNN.

# ---- Changelog ----
# [2026-06-10] Claude Code (Opus 4.8, 1M) — Commons Pool leg 2 (substrate-as-protocol Phase 7)
# What: Proves the §6 verification criteria for the enhance-loop (commons-leg2-design.md),
#       against a SANDBOX neuro_foundation Graph (the real SNN, separate instance) + a fresh
#       Commons — never the live NeuroGraphMemory singleton (Syl's §0 discipline).
# Why: Leg 2 = NG buckets the Commons -> SNN-enhances the SALIENT (salience-gated) -> returns
#       the enhancement to the Commons, with NG's substrate never written by Commons traffic.
#       This proves the mechanism + safety invariants; the FELT test is leg 3 / go-live.
# How: sandbox Graph + Commons + CommonsEnhancer. Crafted high/low-novelty deposits via
#       controlled cosine similarity. Asserts: §6.1 salience gates, §6.2 enhancement lands on
#       the content-node (leg-1 carries), §6.3 bounded (rate cap), §6.4 fail-fresh, §6.5 return
#       scope bounded, §6.6 no live touch, + Syl's §6 addition: transient cleanup (node-count
#       returns to baseline).
# -------------------
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commons as commons_mod
from commons_enhance import CommonsEnhancer
from neuro_foundation import Graph


def _emb(seed: int, dim: int = 768) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _emb_like(base: np.ndarray, sim: float, seed: int) -> np.ndarray:
    """Unit embedding with cos(result, base) ~= sim — to control novelty deterministically."""
    rng = np.random.RandomState(seed)
    r = rng.randn(len(base)).astype(np.float32)
    base_u = base / (np.linalg.norm(base) + 1e-8)
    r = r - np.dot(r, base_u) * base_u          # orthogonalize r to base
    r = r / (np.linalg.norm(r) + 1e-8)
    v = sim * base_u + np.sqrt(max(0.0, 1.0 - sim * sim)) * r
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def _fresh_sandbox():
    """A separate SNN Graph + a fresh Commons — NOT the live singleton (§0/§6.6)."""
    graph = Graph()                              # real SNN, sandbox instance
    medium = commons_mod.Commons()               # fresh bare-NG-Lite (bypass the process singleton)
    return graph, medium


def test_salience_gates():
    """§6.1 — high-novelty deposit gets enhanced; low-novelty stays fresh."""
    graph, medium = _fresh_sandbox()
    enh = CommonsEnhancer(medium, graph)
    K = _emb(1)
    enh.seed_knowledge(K, "knowledge:K")          # NG already knows K

    novel = _emb_like(K, 0.40, seed=2)            # novelty ~0.60 (>=0.50) AND related (binds K)
    routine = _emb_like(K, 0.92, seed=3)          # novelty ~0.08 (<0.50) — NG basically knows it

    stats = enh.enhance_pulse([(novel, "novel"), (routine, "routine")])
    assert stats["enhanced"] == 1, f"only the novel deposit should enhance; {stats}"
    assert stats["gated_fresh"] == 1, f"the routine deposit should stay fresh; {stats}"
    assert "novel" in stats["enhancements"], "the novel content should have an enhancement"


def test_enhancement_lands_on_content_node():
    """§6.2 — the enhancement is bucketable by the content's own address (leg-1 property carries)."""
    graph, medium = _fresh_sandbox()
    enh = CommonsEnhancer(medium, graph)
    K = _emb(10)
    enh.seed_knowledge(K, "knowledge:K")
    novel = _emb_like(K, 0.42, seed=11)
    enh.enhance_pulse([(novel, "needle")])
    targets = [t for (t, _c, _r) in medium.bucket(novel, top_k=10)]
    assert any(t == "enhanced:needle" for t in targets), (
        f"NG's enhancement must surface to a bucket on the content's own node; got {targets!r}"
    )


def test_rate_cap_bounds_enhancement():
    """§6.3 — the hard cap bounds enhances per pulse even when everything is salient."""
    graph, medium = _fresh_sandbox()
    enh = CommonsEnhancer(medium, graph, max_enhances=8)
    K = _emb(20)
    enh.seed_knowledge(K, "knowledge:K")
    # 12 distinct salient deposits (each ~0.40 sim to K -> novelty ~0.60, all gate-in)
    deposits = [(_emb_like(K, 0.40, seed=100 + i), f"item_{i}") for i in range(12)]
    stats = enh.enhance_pulse(deposits)
    assert stats["enhanced"] == 8, f"rate cap must hold at 8; {stats}"
    assert stats["gated_cap"] == 4, f"the 4 over-cap deposits must be gated_cap; {stats}"


def test_fail_fresh_on_gate_error():
    """§6.4 — a novelty-gate error enhances NOTHING (fails toward the recoverable side)."""
    graph, medium = _fresh_sandbox()

    def _boom(_embedding):
        raise RuntimeError("simulated detect_novelty failure")

    enh = CommonsEnhancer(medium, graph, novelty_fn=_boom)
    enh.seed_knowledge(_emb(30), "knowledge:K")
    stats = enh.enhance_pulse([(_emb(31), "a"), (_emb(32), "b")])
    assert stats["enhanced"] == 0, f"gate error must enhance nothing; {stats}"
    assert stats["gated_error"] == 2, f"both deposits must fail-fresh; {stats}"


def test_return_scope_bounded():
    """§6.5 — the returned enhancement is the content's neighborhood, not the whole graph."""
    graph, medium = _fresh_sandbox()
    enh = CommonsEnhancer(medium, graph)
    K = _emb(40)
    # seed a substrate of 20 knowledge items so "whole graph" is clearly larger than 1-hop
    for i in range(20):
        enh.seed_knowledge(_emb_like(K, 0.35 + 0.02 * i, seed=200 + i), f"k_{i}")
    novel = _emb_like(K, 0.45, seed=300)
    stats = enh.enhance_pulse([(novel, "probe")])
    e = stats["enhancements"]["probe"]
    scope = len(set(e["associations"]))
    assert scope <= 3, f"return scope must be the evoked neighborhood (<=top-3 primed), not graph-wide; got {scope}"
    assert scope < stats["baseline_nodes"], "enhancement must be smaller than the substrate"


def test_substrate_read_only():
    """§6 (Syl's addition), strengthened for the PERCEPTION mechanism — the enhance is READ-ONLY:
    after the pulse, NOT ONLY node-count but synapse-count is unchanged. The old create-transient+
    step() is gone; prime_and_propagate(write_mode=False) writes NOTHING to NG's substrate."""
    graph, medium = _fresh_sandbox()
    enh = CommonsEnhancer(medium, graph)
    K = _emb(50)
    for i in range(5):
        enh.seed_knowledge(_emb_like(K, 0.3 + 0.05 * i, seed=400 + i), f"k_{i}")
    base_nodes = len(graph.nodes)
    base_syn = len(graph.synapses)
    deposits = [(_emb_like(K, 0.45, seed=500 + i), f"d_{i}") for i in range(4)]
    stats = enh.enhance_pulse(deposits)
    assert stats["enhanced"] >= 1, "expected at least one enhance to exercise the perception path"
    assert len(graph.nodes) == base_nodes, f"perception created nodes: {len(graph.nodes)} != {base_nodes}"
    assert len(graph.synapses) == base_syn, f"perception created synapses: {len(graph.synapses)} != {base_syn}"
    assert stats["final_nodes"] == stats["baseline_nodes"] == base_nodes


def test_no_live_singleton():
    """§6.6 — the sandbox never constructs the live NeuroGraphMemory singleton."""
    # Structural: this test imports only Graph + Commons + CommonsEnhancer. Assert no live
    # NeuroGraphMemory instance was created as a side effect.
    import neurograph_rpc  # noqa: F401  (importing the module must not spin up a live graph)
    inst = getattr(neurograph_rpc, "_memory", None)
    assert inst is None, "leg-2 sandbox must not instantiate the live NeuroGraphMemory"


def test_live_resolvers_seam():
    """leg-2 part b — the SAME enhancer runs through INJECTED (vector_db-style) resolvers.

    Live, NG injects seed_fn/novelty_fn/assoc_fn backed by NeuroGraphMemory.vector_db instead of
    the sandbox _knowledge map. This proves the injection seam: enhance fires through the injected
    resolvers, returns content-addresses, and STILL writes nothing to the substrate (read-only).
    """
    graph, medium = _fresh_sandbox()
    K = _emb(60)
    # create real SNN nodes (the live substrate already has these); capture node_ids + a fake vdb.
    nid_a = graph.create_node(metadata={}).node_id
    nid_b = graph.create_node(metadata={}).node_id
    fake_vdb = {nid_a: ("alpha content", _emb_like(K, 0.5, seed=61)),
                nid_b: ("beta content", _emb_like(K, 0.45, seed=62))}

    def seed_fn(_emb_in):                       # live: vector_db.search → [(node_id, content)]
        return [(nid_a, fake_vdb[nid_a][0]), (nid_b, fake_vdb[nid_b][0])]

    def novelty_fn(_emb_in):                    # force salient so the enhance path runs
        return 0.99

    def assoc_fn(node_id):                      # live: fired node → its raw content
        e = fake_vdb.get(node_id)
        return e[0] if e else None

    enh = CommonsEnhancer(medium, graph, seed_fn=seed_fn, novelty_fn=novelty_fn, assoc_fn=assoc_fn)
    base_nodes, base_syn = len(graph.nodes), len(graph.synapses)
    stats = enh.enhance_pulse([(_emb_like(K, 0.4, seed=63), "live_probe")])

    assert stats["enhanced"] == 1, f"injected-resolver enhance should run; {stats}"
    assert len(graph.nodes) == base_nodes and len(graph.synapses) == base_syn, "live seam stayed read-only"
    targets = [t for (t, _c, _r) in medium.bucket(_emb_like(K, 0.4, seed=63), top_k=10)]
    assert any(t == "enhanced:live_probe" for t in targets), "salt returns to the depositor's node"


if __name__ == "__main__":
    test_salience_gates();                      print("PASS §6.1 salience gates (novel enhanced, routine fresh)")
    test_live_resolvers_seam();                 print("PASS part-b: injected vector_db-style resolvers drive enhance, read-only")
    test_enhancement_lands_on_content_node();   print("PASS §6.2 enhancement buckets on the content's own node (leg-1 carries)")
    test_rate_cap_bounds_enhancement();         print("PASS §6.3 rate cap bounds enhances at 8")
    test_fail_fresh_on_gate_error();            print("PASS §6.4 fail-fresh: gate error enhances nothing")
    test_return_scope_bounded();                print("PASS §6.5 return scope is 1-hop, not graph-wide")
    test_substrate_read_only();                 print("PASS §6+  substrate READ-ONLY: node + synapse counts unchanged (perception writes nothing)")
    test_no_live_singleton();                   print("PASS §6.6 no live NeuroGraphMemory touch")
    print("\nCommons leg-2: ALL PASS — salience-gated enhance-loop proven in sandbox; substrate untouched")
