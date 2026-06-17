"""Focus habituation (#89) — the latent thread turns its own head.

Syl-approved design: prd/2026-06-16-tonic-focus-habituation-design.md
"""
import sys, os, math
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tonic_thread import TonicThread, TonicConfig


def _node(voltage=0.0, resting=0.0, last_spike=-math.inf, constitutional=False):
    md = {"constitutional": True} if constitutional else {}
    return SimpleNamespace(voltage=voltage, resting_potential=resting,
                           last_spike_time=last_spike, metadata=md)


def _graph(nodes, timestep=100):
    return SimpleNamespace(nodes=dict(nodes), hyperedges={}, _outgoing={},
                           timestep=timestep)


def _thread(nodes, **cfg):
    g = _graph(nodes)
    return TonicThread(graph=g, vector_db={}, config=TonicConfig(**cfg))


# --- Task 1: config + state ---

def test_config_has_fatigue_knobs():
    c = TonicConfig()
    assert c.fatigue_gain > 0
    assert c.fatigue_max > 0
    assert c.fatigue_recovery_base > 0
    assert c.fatigue_recovery_reprime_scale >= 0
    assert 0.0 <= c.spine_fatigue_scale <= 1.0  # whisper: small fraction


def test_fatigue_state_starts_empty():
    t = _thread({"a": _node(voltage=0.5)})
    assert t._focus_fatigue == {}


# --- Task 2: _read_active_nodes subtracts fatigue ---

def test_fatigue_lowers_effective_activity():
    t = _thread({"a": _node(voltage=1.0), "b": _node(voltage=0.9)},
                exploration_bias=0.0)
    t._focus_fatigue["a"] = 0.30
    ranked = t._read_active_nodes()
    ids = [nid for nid, _ in ranked]
    assert ids[0] == "b"  # a: 1.0 - 0.30 = 0.70 < b: 0.90


def test_no_fatigue_unchanged_ranking():
    t = _thread({"a": _node(voltage=1.0), "b": _node(voltage=0.9)},
                exploration_bias=0.0)
    ranked = t._read_active_nodes()
    assert [nid for nid, _ in ranked][0] == "a"


# --- Task 3: _apply_focus_fatigue (accrue + recover + spine whisper + reprime) ---

def test_accrual_on_active_capped():
    t = _thread({"a": _node(voltage=1.0)})
    for _ in range(100):
        t._apply_focus_fatigue([("a", 1.0)])
    assert t._focus_fatigue["a"] == t._config.fatigue_max


def test_recovery_when_not_focus():
    t = _thread({"a": _node(voltage=0.0), "b": _node(voltage=0.0)})
    t._focus_fatigue["a"] = 0.20
    t._apply_focus_fatigue([("b", 0.5)])
    assert 0.0 <= t._focus_fatigue.get("a", 0.0) < 0.20


def test_recovery_floors_at_zero():
    t = _thread({"a": _node(voltage=0.0)})
    t._focus_fatigue["a"] = 0.001
    t._apply_focus_fatigue([])
    assert t._focus_fatigue.get("a", 0.0) == 0.0


def test_contextual_reprime_speeds_recovery():
    t = _thread({"hot": _node(voltage=0.8, resting=0.0),
                 "cold": _node(voltage=0.0, resting=0.0),
                 "x": _node(voltage=0.5)})
    t._focus_fatigue["hot"] = 0.30
    t._focus_fatigue["cold"] = 0.30
    t._apply_focus_fatigue([("x", 0.5)])
    assert t._focus_fatigue["hot"] < t._focus_fatigue["cold"]


def test_spine_accrues_only_a_whisper():
    t = _thread({"plain": _node(voltage=1.0),
                 "constitutional::spine::01": _node(voltage=1.0, constitutional=True)})
    t._apply_focus_fatigue([("plain", 1.0), ("constitutional::spine::01", 1.0)])
    assert (t._focus_fatigue["constitutional::spine::01"]
            < t._focus_fatigue["plain"])
    assert abs(t._focus_fatigue["constitutional::spine::01"]
               - t._config.spine_fatigue_scale * t._focus_fatigue["plain"]) < 1e-9
