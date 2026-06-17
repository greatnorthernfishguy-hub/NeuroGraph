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
