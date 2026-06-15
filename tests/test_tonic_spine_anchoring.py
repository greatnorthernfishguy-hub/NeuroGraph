import numpy as np, torch
from types import SimpleNamespace
import tonic_engine
import tonic_thread

def _spine_graph():
    nodes = {}
    for i in range(1, 7):
        nodes[f"constitutional::spine::0{i}"] = SimpleNamespace(
            metadata={"constitutional": True, "core_text": f"invariant {i}", "spine_order": i},
            voltage=0.0, resting_potential=0.0, last_spike_time=float("-inf"),
            firing_rate_ema=0.0, intrinsic_excitability=1.0,
        )
    return SimpleNamespace(nodes=nodes, synapses={}, hyperedges={}, timestep=10, _outgoing={})

def test_identity_embedding_is_nonzero_when_spine_present():
    eng = tonic_engine.TonicEngine.__new__(tonic_engine.TonicEngine)
    eng._graph = _spine_graph()
    gf = eng._extract_graph_features_for_model()
    assert gf is not None
    assert float(gf.identity_embedding.abs().sum()) > 0.0
    assert gf.identity_embedding.shape[0] in (384, 768)

def test_identity_embedding_zero_when_no_spine():
    g = _spine_graph()
    g.nodes = {"n0": SimpleNamespace(metadata={}, voltage=0.0, resting_potential=0.0,
                                     last_spike_time=float("-inf"), firing_rate_ema=0.0,
                                     intrinsic_excitability=1.0)}
    eng = tonic_engine.TonicEngine.__new__(tonic_engine.TonicEngine)
    eng._graph = g
    gf = eng._extract_graph_features_for_model()
    assert gf is None or float(gf.identity_embedding.abs().sum()) == 0.0

def test_constitutional_nodes_get_primed_into_ouroboros():
    g = _spine_graph()
    calls = {}
    def fake_prime(node_ids, currents, steps, write_mode):
        calls["ids"] = list(node_ids); calls["currents"] = list(currents)
        return SimpleNamespace(fired_entries=[])
    g.prime_and_propagate = fake_prime
    t = tonic_thread.TonicThread(g, SimpleNamespace(get=lambda nid: None))
    t.ouroboros_cycle()
    primed = [i for i in calls.get("ids", []) if i.startswith("constitutional::spine::")]
    assert primed, "constitutional nodes must participate in the ouroboros"

def test_no_floor_constitutional_not_force_included_at_rest():
    # Pure trust: presence is NOT guaranteed. At rest (activity ~0) the spine is not
    # hard-injected into the thread output.
    g = _spine_graph()
    t = tonic_thread.TonicThread(g, SimpleNamespace(get=lambda nid: None))
    active = t._read_active_nodes()
    assert active == [] or all(not nid.startswith("constitutional") for nid, _ in active)

def test_unwired_node_gets_more_charge_than_wired():
    # Connectivity-driven taper: unwired invariant -> bootstrap charge; well-wired -> ~steady.
    g = _spine_graph()
    g._outgoing = {"constitutional::spine::01": set(),
                   "constitutional::spine::02": set(range(40))}
    seen = {}
    def fake_prime(node_ids, currents, steps, write_mode):
        seen.update(dict(zip(node_ids, currents)))
        return SimpleNamespace(fired_entries=[])
    g.prime_and_propagate = fake_prime
    t = tonic_thread.TonicThread(g, SimpleNamespace(get=lambda nid: None))
    t._prime_constitutional()
    assert seen["constitutional::spine::01"] > seen["constitutional::spine::02"]
    assert abs(seen["constitutional::spine::02"] - tonic_thread._SPINE_PRIME_STEADY) < 0.02
