import numpy as np, torch
from types import SimpleNamespace
import tonic_engine

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
