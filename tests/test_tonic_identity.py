import numpy as np
from types import SimpleNamespace
import tonic_identity

def _node(text, order):
    return SimpleNamespace(metadata={"constitutional": True, "core_text": text, "spine_order": order})

def _graph(nodes):
    return SimpleNamespace(nodes=dict(nodes))

def test_identity_vector_is_768_unit_norm():
    g = _graph({f"constitutional::spine::0{i}": _node(f"invariant {i}", i) for i in range(1, 7)})
    vec = tonic_identity.spine_identity_vector(g)
    assert vec is not None
    assert vec.shape == (768,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-4

def test_none_when_no_constitutional_nodes():
    g = _graph({"x": SimpleNamespace(metadata={"_forest_content": "not constitutional"})})
    assert tonic_identity.spine_identity_vector(g) is None

def test_per_invariant_normalization_gives_equal_weight(monkeypatch):
    # Equal-but-distinct (Syl): the code L2-normalizes each invariant BEFORE averaging,
    # so a large-magnitude embedding cannot dominate the aggregate direction. Mock the
    # embeddings to test the aggregation math deterministically (no real-model dependency).
    a = np.zeros(768, dtype=np.float32); a[0] = 1.0     # unit vector, +x
    b = np.zeros(768, dtype=np.float32); b[1] = 50.0    # 50x magnitude, +y
    monkeypatch.setattr("ng_embed.embed_batch", lambda texts: [a, b])
    g = _graph({"c1": _node("x", 1), "c2": _node("y", 2)})
    vec = tonic_identity.spine_identity_vector(g)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-4          # unit norm
    # despite b's 50x raw magnitude, both invariants contribute equally -> equal components
    assert abs(float(vec[0]) - float(vec[1])) < 1e-4
    assert float(vec[0]) > 0.1 and float(vec[1]) > 0.1
