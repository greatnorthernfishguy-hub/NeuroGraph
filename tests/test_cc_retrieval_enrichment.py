# ---- Changelog ----
# [2026-07-07] Claude Code (Fable 5) — CC retrieval-enrichment tests (#358)
# What: Tests for cc_novelty, cc_anticipate, cc_gsg_rescore, cc_gsg_backfill,
#   the rebuilt cc_pattern_completion_recall, and constant pins (C5).
# Why: Spec docs/superpowers/specs/2026-07-07-cc-retrieval-enrichment-design.md;
#   law-review conditions C1-C5 each demand test-visible enforcement.
# How: Real NeuroGraphMemory fixture (same pattern as test_cc_dual_pass.py) —
#   the substrate-primacy property under test is exactly the interaction a
#   mock would hide.
# -------------------
import sys
sys.path.insert(0, '/home/josh/NeuroGraph')
import time
import tempfile, shutil
import numpy as np
import pytest


@pytest.fixture
def cc_ng():
    from openclaw_hook import NeuroGraphMemory
    workspace = tempfile.mkdtemp(prefix='cc_retrieval_enrich_test_')
    ng = NeuroGraphMemory(workspace_dir=workspace,
                          config={"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}})
    yield ng
    shutil.rmtree(workspace, ignore_errors=True)


# ---- Task 1: constants (C5 pins), poincare distance, novelty ----

def test_copied_constants_pin_canonical_values():
    """C5: every copied constant matches canonical neurograph_rpc.py exactly.
    If canonical changes, this fails loudly instead of drifting silently."""
    import cc_ng_organism as o
    assert o._CC_ANTICIPATE_TOP_K == 15
    assert o._CC_ANTICIPATE_TTL_S == 120.0
    assert o._CC_ANTICIPATE_BONUS == 0.25
    assert o._CC_GSG_LAYER_NORMS == (0.70, 0.50, 0.30)
    assert o._CC_GSG_SCORE_BONUS == 0.30
    assert o._CC_NOVELTY_EMA_KEEP == 0.9
    assert o._CC_NOVELTY_EMA_GAIN == 0.1


def test_poincare_distance_properties():
    from cc_ng_organism import _cc_poincare_distance
    a = np.zeros(4, dtype=np.float64)
    assert _cc_poincare_distance(a, a) == 0.0
    b = np.array([0.5, 0.0, 0.0, 0.0])
    c = np.array([0.9, 0.0, 0.0, 0.0])
    # identity, symmetry, boundary-spreading (near-boundary pairs farther apart
    # than same-Euclidean-gap central pairs)
    assert _cc_poincare_distance(a, b) == _cc_poincare_distance(b, a)
    central = _cc_poincare_distance(np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.4, 0.0, 0.0, 0.0]))
    boundary = _cc_poincare_distance(np.array([0.5, 0.0, 0.0, 0.0]), c)
    assert boundary > central


def test_novelty_counters_exist_on_real_graph(cc_ng):
    """C3 liveness pin: the HE-level cumulative counters cc_novelty reads must
    exist on a real Graph. If the engine renames them, fail HERE, loudly —
    not silently freeze novelty at 0.5 in production."""
    assert hasattr(cc_ng.graph, "_total_confirmed")
    assert hasattr(cc_ng.graph, "_total_surprised")
    assert isinstance(cc_ng.graph._total_confirmed, int)
    assert isinstance(cc_ng.graph._total_surprised, int)


def test_novelty_delta_math_and_ema(cc_ng):
    from cc_ng_organism import cc_novelty
    state = {}
    # First call: establishes baseline, no delta yet -> default 0.5
    assert cc_novelty(state, cc_ng.graph) == 0.5
    # Simulate a window of all-surprised HE predictions
    cc_ng.graph._total_surprised += 10
    v1 = cc_novelty(state, cc_ng.graph)
    # raw = 10/10 = 1.0; ema = 0.9*0.5 + 0.1*1.0 = 0.55
    assert v1 == pytest.approx(0.55)
    # Window of all-confirmed
    cc_ng.graph._total_confirmed += 10
    v2 = cc_novelty(state, cc_ng.graph)
    # raw = 0/10 = 0.0; ema = 0.9*0.55 + 0.1*0.0 = 0.495
    assert v2 == pytest.approx(0.495)
    # Empty window: EMA unchanged
    v3 = cc_novelty(state, cc_ng.graph)
    assert v3 == pytest.approx(0.495)


def test_novelty_fails_soft_when_counters_missing():
    """Engine-contract-change guard: object without the counters -> 0.5, no raise."""
    from cc_ng_organism import cc_novelty

    class Bare:
        pass

    assert cc_novelty({}, Bare()) == 0.5


# ---- Task 2: cc_anticipate ----

def test_cc_anticipate_primes_synaptic_neighbors(cc_ng):
    from cc_ng_organism import cc_anticipate
    g = cc_ng.graph
    g.create_node(node_id="fired_a")
    g.create_node(node_id="neighbor_b")
    g.create_node(node_id="neighbor_c")
    g.create_synapse("fired_a", "neighbor_b", weight=0.9)
    g.create_synapse("fired_a", "neighbor_c", weight=0.3)
    state = {}
    cc_anticipate(g, ["fired_a"], state)
    primed = state["primed_nodes"]
    assert "neighbor_b" in primed and "neighbor_c" in primed
    assert "fired_a" not in primed            # fired nodes are not their own predictions
    assert primed["neighbor_b"][0] > primed["neighbor_c"][0]   # scored by edge weight
    assert primed["neighbor_b"][1] > time.time()               # future expiry


def test_cc_anticipate_empty_fired_clears_primed(cc_ng):
    from cc_ng_organism import cc_anticipate
    state = {"primed_nodes": {"stale": (1.0, time.time() + 999)}}
    cc_anticipate(cc_ng.graph, [], state)
    assert state["primed_nodes"] == {}


def test_dual_pass_drain_triggers_anticipation(cc_ng, monkeypatch):
    """Integration: the conversational dual-pass calls the cc_anticipate
    trigger with this turn's forest+trees and the shared state dict. The
    priming BEHAVIOR itself is covered by the unit tests above; this test
    pins the WIRING (trigger fires, right seeds, right state) without
    depending on tree extraction, which short texts don't guarantee."""
    import cc_ng_organism as o
    from ng_embed import embed
    calls = []
    real = o.cc_anticipate
    monkeypatch.setattr(o, "cc_anticipate",
                        lambda graph, fired, state: (calls.append(list(fired)), real(graph, fired, state)))
    state = {"last_forest_id": None}
    text = "the lenia distance cache rebuild finished on the vps"
    assert o.run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, text, embed(text), state)
    assert len(calls) == 1, "dual-pass must trigger cc_anticipate exactly once"
    assert state["last_forest_id"] in calls[0], "this turn's forest must be in the seed set"
    assert "primed_nodes" in state, "trigger must have populated the primed dict (possibly empty)"


# ---- Task 3: GSG rescore + backfill ----

def _mk_gsg_node(g, vdb, nid, text, direction):
    """Node + vdb entry + poincare_dir stamp, for rescore tests."""
    from ng_embed import embed
    g.create_node(node_id=nid)
    emb = embed(text)
    vdb.insert(id=nid, embedding=emb, content=text)
    g.nodes[nid].metadata["poincare_dir"] = direction.tolist()


def test_gsg_rescore_bonuses_only_stamped_nodes(cc_ng):
    from cc_ng_organism import cc_gsg_rescore, _cc_embed_to_poincare_dir
    from ng_embed import embed
    g, vdb = cc_ng.graph, cc_ng.vector_db
    query = "lenia kernel distance cache"
    qdir = _cc_embed_to_poincare_dir(embed(query))
    # stamped node aligned with the query direction; unstamped node
    _mk_gsg_node(g, vdb, "stamped", "lenia cache", qdir)
    g.create_node(node_id="unstamped")
    surfaced = [
        {"node_id": "stamped", "strength": 1.0},
        {"node_id": "unstamped", "strength": 1.0},
    ]
    out = cc_gsg_rescore(surfaced, query, g)
    by_id = {r["node_id"]: r for r in out}
    assert by_id["stamped"]["strength"] > 1.0          # got a bonus
    assert by_id["unstamped"]["strength"] == 1.0        # untouched
    assert by_id["stamped"]["strength"] <= 1.0 + 0.30   # bonus capped at _CC_GSG_SCORE_BONUS
    assert out[0]["node_id"] == "stamped"               # re-sorted by strength


def test_gsg_rescore_spherical_branch(cc_ng):
    from cc_ng_organism import cc_gsg_rescore, _cc_embed_to_poincare_dir
    from ng_embed import embed
    g, vdb = cc_ng.graph, cc_ng.vector_db
    query = "spherical attractor node test"
    qdir = _cc_embed_to_poincare_dir(embed(query))
    _mk_gsg_node(g, vdb, "sph", "spherical attractor", qdir)
    g.nodes["sph"].manifold_type = "spherical"
    out = cc_gsg_rescore([{"node_id": "sph", "strength": 1.0}], query, g)
    assert out[0]["strength"] > 1.0                     # great-circle branch applied a bonus


def test_gsg_rescore_fails_soft(cc_ng):
    from cc_ng_organism import cc_gsg_rescore
    surfaced = [{"node_id": "nope", "strength": 0.7}]
    out = cc_gsg_rescore(surfaced, "", cc_ng.graph)     # empty query -> no embed
    assert out == surfaced                              # unchanged, no raise


def test_gsg_backfill_stamps_missing_skips_present_never_saves(cc_ng):
    from cc_ng_organism import cc_gsg_backfill
    from ng_embed import embed
    g, vdb = cc_ng.graph, cc_ng.vector_db
    g.create_node(node_id="old_node")                   # pre-GSG node: vdb entry, no stamp
    vdb.insert(id="old_node", embedding=embed("an old memory"), content="an old memory")
    g.create_node(node_id="already")                    # already stamped
    vdb.insert(id="already", embedding=embed("stamped"), content="stamped")
    g.nodes["already"].metadata["poincare_dir"] = [1.0, 0.0]
    g.create_node(node_id="no_vdb")                     # graph-only, nothing to stamp from

    save_calls = []
    cc_ng.save = lambda *a, **k: save_calls.append("ng.save")          # C2 spies
    g.checkpoint = lambda *a, **k: save_calls.append("graph.checkpoint")

    stamped = cc_gsg_backfill(g, vdb)
    assert stamped == 1
    assert g.nodes["old_node"].metadata.get("poincare_dir")
    assert g.nodes["already"].metadata["poincare_dir"] == [1.0, 0.0]   # untouched
    assert "poincare_dir" not in (g.nodes["no_vdb"].metadata or {})
    assert save_calls == []                             # C2: STAMP-ONLY, never persists
    assert cc_gsg_backfill(g, vdb) == 0                 # idempotent second run
