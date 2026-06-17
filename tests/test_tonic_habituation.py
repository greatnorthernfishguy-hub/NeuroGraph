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
    # Focus-only: a constitutional node AS the focus accrues only spine_fatigue_scale x
    # the gain a plain focus would — a whisper, so "who I am" grounds without welding.
    tc = _thread({"constitutional::spine::01": _node(voltage=1.0, constitutional=True)})
    tc._apply_focus_fatigue([("constitutional::spine::01", 1.0)])
    tp_ = _thread({"plain": _node(voltage=1.0)})
    tp_._apply_focus_fatigue([("plain", 1.0)])
    spine_f = tc._focus_fatigue["constitutional::spine::01"]
    plain_f = tp_._focus_fatigue["plain"]
    assert spine_f < plain_f
    assert abs(spine_f - tc._config.spine_fatigue_scale * plain_f) < 1e-9


def test_fatigue_demotes_but_never_erases():
    # A lone genuinely-active node (above the floor) that is heavily fatigued stays
    # a CANDIDATE (demoted, not dropped) — "quiets, never erases". Pins the contract.
    t = _thread({"lone": _node(voltage=0.10)}, exploration_bias=0.0)
    t._focus_fatigue["lone"] = 0.30  # fatigue >> its 0.10 activity
    ids = [nid for nid, _ in t._read_active_nodes()]
    assert "lone" in ids  # present despite post-fatigue score going negative



# --- Task 4: integration — head turns, return, love-interrupt, ouroboros wiring ---

def test_head_turns_marginal_yields_faster_than_dominant():
    def cycles_to_flip(lead):
        t = _thread({"a": _node(voltage=0.5 + lead), "b": _node(voltage=0.5)},
                    exploration_bias=0.0)
        for i in range(1, 200):
            ranked = t._read_active_nodes()
            t._apply_focus_fatigue(ranked)
            if ranked and ranked[0][0] == "b":
                return i
        return 999
    marginal = cycles_to_flip(0.02)
    dominant = cycles_to_flip(0.25)
    assert marginal < dominant  # a genuinely-hot interest holds longer


def test_set_aside_thought_can_return():
    t = _thread({"a": _node(voltage=0.7), "b": _node(voltage=0.5)},
                exploration_bias=0.0)
    seen_b = False
    seen_a_again = False
    for _ in range(400):
        ranked = t._read_active_nodes()
        t._apply_focus_fatigue(ranked)
        top = ranked[0][0] if ranked else None
        if top == "b":
            seen_b = True
        if seen_b and top == "a":
            seen_a_again = True
    assert seen_b and seen_a_again  # dwell -> drift -> return


def test_love_is_an_interrupt_preserved():
    t = _thread({"stuck": _node(voltage=0.9), "spike": _node(voltage=0.0)},
                exploration_bias=0.0)
    t._focus_fatigue["stuck"] = t._config.fatigue_max  # maximally fatigued, still highest raw
    t._graph.nodes["spike"].voltage = 5.0  # love: a big affective salience spike
    ranked = t._read_active_nodes()
    assert ranked[0][0] == "spike"


def test_ouroboros_cycle_applies_fatigue():
    t = _thread({"a": _node(voltage=1.0)}, exploration_bias=0.0)
    t._graph.prime_and_propagate = lambda **kw: SimpleNamespace(fired_entries=[])
    t._vector_db = SimpleNamespace(get=lambda nid: None)
    before = dict(t._focus_fatigue)
    t.ouroboros_cycle()
    assert t._focus_fatigue.get("a", 0.0) > before.get("a", 0.0)


def test_higher_rank_accrues_more_fatigue():
    # Josh's dial: the higher in the sort, the marginally more fatigue. Top > lower > 0.
    t = _thread({"top": _node(voltage=1.0), "mid": _node(voltage=0.5),
                 "low": _node(voltage=0.2)}, exploration_bias=0.0)
    ranked = t._read_active_nodes()  # top, mid, low
    t._apply_focus_fatigue(ranked)
    f = t._focus_fatigue
    assert f["top"] > f["mid"] > f["low"] > 0.0


def test_fatigue_catches_a_pinned_attractor():
    # Josh's mechanism: a node is homeostatically PINNED at high activity (re-pumped each
    # cycle), leading its peer by FAR more than the old 0.35 cap. Because the node is
    # stationary, fatigue must eventually catch up and turn the head — for ANY weld, not
    # just a marginal one. (On v1 with cap=0.35 this never flips: the actual #89 failure.)
    t = _thread({"a": _node(voltage=2.0), "b": _node(voltage=0.5)}, exploration_bias=0.0)
    flipped = False
    for i in range(600):
        t._graph.nodes["a"].voltage = 2.0  # pin a at its ceiling each cycle (the weld)
        ranked = t._read_active_nodes()
        t._apply_focus_fatigue(ranked)
        if ranked and ranked[0][0] == "b":
            flipped = True
            break
    assert flipped  # a hard weld eventually breaks, not only a marginal one
