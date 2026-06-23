"""Focus valence field (#90) — her poles build the light<->dark axis.

Syl-approved design: prd/2026-06-17-tonic-valence-field-design.md
"""
import sys, os
import numpy as np
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tonic_valence import ValenceConfig, ValenceField, load_poles


def _stub_embed(mapping, dim=8):
    """Return an embed_fn that maps known phrases to fixed vectors, else zeros."""
    def _e(text, normalize=False, is_query=False):
        return np.array(mapping.get(text, [0.0] * dim), dtype=np.float32)
    return _e


def test_config_defaults_sane():
    c = ValenceConfig()
    assert c.seed_gain > 0
    assert c.diffusion_steps >= 1
    assert 0.0 <= c.diffusion_alpha <= 1.0


def test_poles_file_loads_her_words():
    poles = load_poles(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "valence_poles.toml"))
    assert "home" in poles["light"]
    assert any("weld" in d for d in poles["dark"])


def test_axis_points_from_dark_to_light():
    # light pole at +x, dark pole at -x → axis ~ +x unit
    poles = {"light": ["L"], "dark": ["D"]}
    embed = _stub_embed({"L": [1, 0], "D": [-1, 0]}, dim=2)
    vf = ValenceField(ValenceConfig(), embed_fn=embed, poles=poles)
    assert vf.axis is not None
    assert vf.axis.shape == (2,)
    assert np.isclose(np.linalg.norm(vf.axis), 1.0)
    assert vf.axis[0] > 0.99  # points toward light


# ---------------------------------------------------------------------------
# Task 2 helpers — fake vdb and graph for seed tests
# ---------------------------------------------------------------------------

def _vdb(mapping, dim=2):
    """Fake SimpleVectorDB: .get(id) -> {'embedding': ndarray} or None."""
    store = {k: np.array(v, dtype=np.float32) for k, v in mapping.items()}
    return SimpleNamespace(get=lambda i: ({"embedding": store[i]} if i in store else None))


def _graph(node_ids, synapses=None):
    nodes = {nid: SimpleNamespace(voltage=0.0, resting_potential=0.0, metadata={}) for nid in node_ids}
    synapses = synapses or {}
    outgoing, incoming = {nid: set() for nid in node_ids}, {nid: set() for nid in node_ids}
    for sid, syn in synapses.items():
        outgoing.setdefault(syn.pre_node_id, set()).add(sid)
        incoming.setdefault(syn.post_node_id, set()).add(sid)
    return SimpleNamespace(nodes=nodes, synapses=synapses, _outgoing=outgoing, _incoming=incoming)


def _light_field():
    poles = {"light": ["L"], "dark": ["D"]}
    embed = _stub_embed({"L": [1, 0], "D": [-1, 0]}, dim=2)
    return ValenceField(ValenceConfig(), embed_fn=embed, poles=poles)


def test_seed_light_node_positive_dark_node_negative():
    vf = _light_field()
    g = _graph(["warm", "cold", "noemb"])
    vdb = _vdb({"warm": [1, 0], "cold": [-1, 0]})  # 'noemb' has no embedding
    seed = vf._seed(g, vdb)
    assert seed["warm"] > 0.5
    assert seed["cold"] < -0.5
    assert "noemb" not in seed  # no embedding -> neutral (absent)


def test_seed_clamped_to_unit_range():
    vf = _light_field()  # seed_gain=3.0 would push a pure-light projection past 1.0
    g = _graph(["x"])
    vdb = _vdb({"x": [5, 0]})
    seed = vf._seed(g, vdb)
    assert -1.0 <= seed["x"] <= 1.0
