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


# ---------------------------------------------------------------------------
# Task 3 helpers — synapse factory for diffusion tests
# ---------------------------------------------------------------------------

def _syn(pre, post, weight, inhibitory=False):
    st = SimpleNamespace(name=("INHIBITORY" if inhibitory else "EXCITATORY"))
    return SimpleNamespace(pre_node_id=pre, post_node_id=post, weight=weight, synapse_type=st)


def test_neutral_node_wired_to_light_picks_up_warmth():
    vf = _light_field()
    # 'mid' has no embedding (no seed) but is wired strongly to light-seeded 'warm'
    syn = {"s1": _syn("warm", "mid", 2.0)}
    g = _graph(["warm", "mid"], synapses=syn)
    seed = {"warm": 0.9}  # mid absent
    field = vf._diffuse(g, seed)
    assert field["mid"] > 0.1  # warmth flowed across her synapse
    assert field["mid"] < field["warm"]  # but it's a glow, not the source


def test_inhibitory_synapse_flips_sign_of_influence():
    vf = _light_field()
    syn = {"s1": _syn("warm", "mid", 2.0, inhibitory=True)}
    g = _graph(["warm", "mid"], synapses=syn)
    field = vf._diffuse(g, {"warm": 0.9})
    assert field["mid"] < 0.0  # inhibitory tie to a light node reads as shadow


def test_diffuse_does_not_mutate_graph():
    vf = _light_field()
    syn = {"s1": _syn("warm", "mid", 2.0)}
    g = _graph(["warm", "mid"], synapses=syn)
    before = {nid: nd.voltage for nid, nd in g.nodes.items()}
    weights_before = {sid: s.weight for sid, s in g.synapses.items()}
    vf._diffuse(g, {"warm": 0.9})
    assert {nid: nd.voltage for nid, nd in g.nodes.items()} == before
    assert {sid: s.weight for sid, s in g.synapses.items()} == weights_before
