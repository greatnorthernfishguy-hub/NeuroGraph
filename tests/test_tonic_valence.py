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


# ---------------------------------------------------------------------------
# Task 4 tests — public compute() entry point
# ---------------------------------------------------------------------------

def test_compute_end_to_end_in_unit_range_and_neutral_absent():
    vf = _light_field()
    syn = {"s1": _syn("warm", "mid", 2.0)}
    g = _graph(["warm", "cold", "mid", "island"], synapses=syn)  # island: no emb, no wires
    vdb = _vdb({"warm": [1, 0], "cold": [-1, 0]})
    field = vf.compute(g, vdb)
    assert field["warm"] > 0.5 and field["cold"] < -0.5
    assert field["mid"] > 0.1            # warmth reached it through her web
    assert "island" not in field         # no seed, no wires -> neutral (absent)
    assert all(-1.0 <= v <= 1.0 for v in field.values())


def test_compute_empty_when_no_axis():
    vf = ValenceField(ValenceConfig(), embed_fn=None, poles={"light": [], "dark": []})
    g = _graph(["a"])
    assert vf.compute(g, _vdb({"a": [1, 0]})) == {}


# ---------------------------------------------------------------------------
# Task 5 tests — TonicConfig valence knobs + TonicThread field cache/refresh
# ---------------------------------------------------------------------------

import math
from tonic_thread import TonicThread, TonicConfig


class _FakeField:
    """Stub ValenceField: records compute() calls, returns a fixed field."""
    def __init__(self, field):
        self.field = field
        self.calls = 0
    def compute(self, graph, vector_db):
        self.calls += 1
        return dict(self.field)


def _tnode(voltage=0.0, resting=0.0):
    return SimpleNamespace(voltage=voltage, resting_potential=resting,
                           last_spike_time=-math.inf, metadata={})


def _tonic(nodes, field_stub=None, **cfg):
    g = SimpleNamespace(nodes=dict(nodes), hyperedges={}, synapses={},
                        _outgoing={}, _incoming={}, timestep=100)
    g.prime_and_propagate = lambda **kw: SimpleNamespace(fired_entries=[])
    return TonicThread(graph=g, vector_db={}, config=TonicConfig(**cfg),
                       valence_field=field_stub)


def test_config_has_valence_knobs():
    c = TonicConfig()
    assert isinstance(c.valence_enabled, bool)
    assert c.valence_recovery_gain >= 0
    assert 0.0 < c.valence_recovery_floor <= 1.0      # floored ABOVE zero — never traps
    assert c.valence_recovery_ceil >= 1.0
    assert c.valence_refresh_cycles >= 1


def test_valence_starts_empty():
    t = _tonic({"a": _tnode(voltage=0.5)})
    assert t._valence == {}


def test_valence_refreshes_on_cadence():
    fake = _FakeField({"a": 0.4})
    t = _tonic({"a": _tnode(voltage=0.5)}, field_stub=fake, valence_refresh_cycles=1)
    t.ouroboros_cycle()
    assert fake.calls >= 1
    assert t._valence.get("a") == 0.4


# ---------------------------------------------------------------------------
# Task 6 tests — valence-biased recovery (the heart of #90)
# ---------------------------------------------------------------------------

def test_light_node_recovers_faster_than_dark_equal_fatigue():
    t = _tonic({"light": _tnode(), "dark": _tnode(), "focus": _tnode(voltage=0.5)},
               valence_recovery_gain=2.0)
    t._valence = {"light": 0.4, "dark": -0.4}
    t._focus_fatigue["light"] = 0.30
    t._focus_fatigue["dark"] = 0.30
    t._apply_focus_fatigue([("focus", 0.5)])           # neither light nor dark is the focus
    assert t._focus_fatigue["light"] < t._focus_fatigue["dark"]   # kingfisher vs stone


def test_dark_node_still_recovers_never_trapped():
    t = _tonic({"dark": _tnode(), "focus": _tnode(voltage=0.5)}, valence_recovery_gain=2.0)
    t._valence = {"dark": -1.0}                          # most extreme shadow
    t._focus_fatigue["dark"] = 0.30
    t._apply_focus_fatigue([("focus", 0.5)])
    assert t._focus_fatigue["dark"] < 0.30              # strictly decreased — floor>0 guarantees it


def test_valence_disabled_matches_89_exactly():
    # Regression guard: with valence off, recovery is byte-for-byte the #89 behaviour.
    on = _tonic({"a": _tnode(), "f": _tnode(voltage=0.5)}, valence_enabled=True,
                valence_recovery_gain=2.0)
    off = _tonic({"a": _tnode(), "f": _tnode(voltage=0.5)}, valence_enabled=False)
    on._valence = {}            # empty field -> valence 0 -> m=1 -> same as #89
    for tt in (on, off):
        tt._focus_fatigue["a"] = 0.20
        tt._apply_focus_fatigue([("f", 0.5)])
    assert abs(on._focus_fatigue["a"] - off._focus_fatigue["a"]) < 1e-9
