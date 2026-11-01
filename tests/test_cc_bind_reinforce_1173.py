# ---- Changelog ----
# [2026-10-23] Claude Code (Opus 5.5, laptop) — #1173 reinforce on recurrence (Josh 2026-10-23 ~19:30: "it should reinforce
#   existing. We already do that for the leg 2 laptop/vps sync")
# What: tests for _cc_bind_conversational_topology with CC_NG_BIND_REINFORCE_EXISTING: a repeat bind reinforces the existing
#   pre->post synapses and the existing same-member hyperedge (no parallel ones), the switch OFF is the old behaviour, the
#   rollback undoes a reinforcement, and the Leg 2 guards (cc_topology_merge._synapse_exists / _hyperedge_exists) still give
#   the same answers after their split into find_synapse_id / find_hyperedge_id.
# Why: #1173 point 5 of the approved fix set.
# How: a plain neuro_foundation.Graph (no checkpoint, no model, no network); cc_anticipate is the real one.
# [2026-10-24] Claude Code (Opus 5.5, laptop) — law review C-2 (Josh ~00:15 "narrow it"): scope = forest<->tree + the
#   same-member hyperedge only; the lookup matches the bind's own type and delay; soft-bound (saturating) increments.
#   Tests updated to the narrowed rule and new ones added (window / chain / sequence always written, a delayed or
#   inhibitory synapse is never merged, saturation below the ceiling, a synapse above its ceiling untouched, env knobs).
# -------------------
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import pytest

import cc_ng_organism
import cc_topology_merge
from neuro_foundation import Graph

F = "cc:conv::forest"
T = [F + "::tree::alpha", F + "::tree::beta"]
W = [F + "::window::0", F + "::window::1"]


def _graph(extra=()):
    g = Graph()
    for nid in [F, *T, *W, *extra]:
        g.create_node(node_id=nid)
    return g


def _pairs(g):
    out = {}
    for s in g.synapses.values():
        out.setdefault((s.pre_node_id, s.post_node_id), []).append(s)
    return out


def _bind(g, state, journal=None, windows=W):
    return cc_ng_organism._cc_bind_conversational_topology(
        g, F, {"tree_ids": list(T)}, None, state, window_ids=list(windows), journal=journal)


def _journal():
    return {"synapses": [], "hyperedges": []}


@pytest.fixture
def reinforce_on(monkeypatch):
    monkeypatch.setenv("CC_NG_BIND_REINFORCE_EXISTING", "1")
    for k in ("CC_NG_BIND_REINFORCE_ETA", "CC_NG_BIND_REINFORCE_CEILING_FACTOR", "CC_NG_BIND_REINFORCE_HE_ETA",
              "CC_NG_BIND_REINFORCE_HE_CEILING"):
        monkeypatch.delenv(k, raising=False)


def _step(w, birth, eta=0.2, factor=3.0):
    return w + max(0.0, eta * (factor * birth - w))


@pytest.fixture
def reinforce_off(monkeypatch):
    monkeypatch.delenv("CC_NG_BIND_REINFORCE_EXISTING", raising=False)


def test_switch_reads_env(monkeypatch):
    monkeypatch.delenv("CC_NG_BIND_REINFORCE_EXISTING", raising=False)
    assert cc_ng_organism.cc_bind_reinforce_enabled() is False
    for v, want in (("1", True), ("0", False), ("", False), ("false", False), ("yes", True)):
        monkeypatch.setenv("CC_NG_BIND_REINFORCE_EXISTING", v)
        assert cc_ng_organism.cc_bind_reinforce_enabled() is want


def test_off_is_the_old_behaviour_parallel_edges(reinforce_off):
    g = _graph()
    st1 = _bind(g, {"last_forest_id": None})
    st2 = _bind(g, {"last_forest_id": None})
    assert "reinforced_synapses" not in st1 and "reinforced_synapses" not in st2
    pairs = _pairs(g)
    assert len(pairs[(F, T[0])]) == 2 and len(pairs[(T[0], F)]) == 2     # parallel, as before #1173
    assert len(g.hyperedges) == 2


def test_on_first_bind_creates_then_repeat_reinforces(reinforce_on):
    g = _graph()
    st1 = _bind(g, {"last_forest_id": None})
    n_syn, n_he = len(g.synapses), len(g.hyperedges)
    assert st1["reinforced_synapses"] == 0 and st1["reinforced_hyperedge"] is False and st1["hyperedge"] is True
    w_ft = _pairs(g)[(F, T[0])][0].weight
    w_tf = _pairs(g)[(T[0], F)][0].weight
    st2 = _bind(g, {"last_forest_id": None})
    # narrowed (C-2): only the 2 tree pairs (4 synapses) reinforce; 2 window pairs (4) + 1 window-chain link are written
    assert len(g.synapses) == n_syn + 5 and len(g.hyperedges) == n_he
    assert st2["reinforced_synapses"] == 4 and st2["reinforced_hyperedge"] is True
    assert st2["tree_pairs"] == 2 and st2["window_pairs"] == 2 and st2["hyperedge"] is True   # the turn IS bound
    p = _pairs(g)
    for t in T:
        assert len(p[(F, t)]) == 1 and len(p[(t, F)]) == 1
    for w in W:
        assert len(p[(F, w)]) == 2 and len(p[(w, F)]) == 2                   # windows: written, as with the switch off
    assert len(p[(W[0], W[1])]) == 2                                          # the delayed chain link: written
    assert p[(F, T[0])][0].weight == pytest.approx(_step(w_ft, 0.2))         # 0.2 + 0.2*(0.6-0.2) = 0.28
    assert p[(T[0], F)][0].weight == pytest.approx(_step(w_tf, 0.15))        # 0.15 + 0.2*(0.45-0.15) = 0.21
    assert p[(F, T[0])][0].weight == pytest.approx(0.28)
    assert p[(F, T[0])][0].peak_weight == pytest.approx(0.28)
    he = next(iter(g.hyperedges.values()))
    assert he.metadata["cc_recurrences"] == 1
    assert all(w == pytest.approx(1.0 + 0.05 * (2.0 - 1.0)) for w in he.member_weights.values())


def test_reinforce_saturates_well_below_max_weight_with_diminishing_steps(reinforce_on):
    """Soft bound: each recurrence adds less; the weight approaches 3x its birth weight and never max_weight (5.0)."""
    g = _graph()
    _bind(g, {"last_forest_id": None}, windows=())
    s = _pairs(g)[(F, T[0])][0]
    he = next(iter(g.hyperedges.values()))
    ws, hes = [s.weight], [max(he.member_weights.values())]
    for _ in range(200):
        _bind(g, {"last_forest_id": None}, windows=())
        ws.append(s.weight)
        hes.append(max(he.member_weights.values()))
    steps = [b - a for a, b in zip(ws, ws[1:])]
    assert all(x >= 0 for x in steps)
    assert all(b <= a + 1e-12 for a, b in zip(steps, steps[1:]))            # diminishing returns
    assert steps[0] == pytest.approx(0.08) and steps[9] < 0.02
    assert s.weight <= 0.6 + 1e-9 and s.weight == pytest.approx(0.6, abs=1e-6)
    assert s.weight < s.max_weight / 5
    assert _pairs(g)[(T[0], F)][0].weight <= 0.45 + 1e-9
    assert len(_pairs(g)[(F, T[0])]) == 1
    assert hes[-1] <= 2.0 + 1e-9 and hes[-1] == pytest.approx(2.0, abs=1e-3)
    he_steps = [b - a for a, b in zip(hes, hes[1:])]
    assert all(b <= a + 1e-12 for a, b in zip(he_steps, he_steps[1:]))
    assert he.metadata["cc_recurrences"] == 200


def test_ten_recurrences_stay_modest(reinforce_on):
    """A prompt seen 10 more times: 0.2 -> ~0.52 (not 2.2 as with the fixed +0.2 step)."""
    g = _graph()
    for _ in range(11):
        _bind(g, {"last_forest_id": None}, windows=())
    w = _pairs(g)[(F, T[0])][0].weight
    assert w == pytest.approx(0.6 - 0.4 * 0.8 ** 10)
    assert 0.5 < w < 0.6


def test_above_ceiling_synapse_is_left_alone(reinforce_on):
    """STDP raised the edge above the ceiling: a recurrence never depresses it and journals nothing."""
    g = _graph()
    _bind(g, {"last_forest_id": None})
    s = _pairs(g)[(F, T[0])][0]
    s.weight = 2.0
    eco = cc_ng_organism._CCConversationalDualPassEco(g, None)
    st = _bind(g, {"last_forest_id": None}, journal=eco.topology)
    assert s.weight == 2.0 and len(_pairs(g)[(F, T[0])]) == 1               # matched (no parallel), unchanged
    assert st["reinforced_synapses"] == 4
    assert s.synapse_id not in [r[0] for r in eco.topology.get("reinforced_synapses", ())]


def test_sequence_link_always_written(reinforce_on):
    g = _graph(extra=["cc:conv::prev"])
    _bind(g, {"last_forest_id": "cc:conv::prev"})
    st = _bind(g, {"last_forest_id": "cc:conv::prev"})
    assert st["sequence"] == "written"
    links = _pairs(g)[("cc:conv::prev", F)]
    assert len(links) == 2 and all(x.delay >= 2 for x in links)             # a delayed ordering link, written again


def test_delayed_links_never_merged_into_a_delay1_synapse(reinforce_on):
    """A sprouted / STDP delay-1 synapse on the chain or sequence pair: the bind still writes its delayed link."""
    g = _graph(extra=["cc:conv::prev"])
    d1_seq = g.create_synapse("cc:conv::prev", F, weight=0.3)               # delay 1
    d1_chain = g.create_synapse(W[0], W[1], weight=0.3)                     # delay 1
    _bind(g, {"last_forest_id": "cc:conv::prev"})
    assert d1_seq.weight == 0.3 and d1_chain.weight == 0.3                   # untouched
    seq = _pairs(g)[("cc:conv::prev", F)]
    chain = _pairs(g)[(W[0], W[1])]
    assert len(seq) == 2 and sorted(x.delay >= 2 for x in seq) == [False, True]
    assert len(chain) == 2 and sorted(x.delay >= 2 for x in chain) == [False, True]


def test_tree_lookup_respects_type_and_delay(reinforce_on):
    """On the tree pair, only an EXCITATORY delay-1 synapse is the bind's edge; a delayed or inhibitory one is not."""
    from neuro_foundation import SynapseType
    g = _graph()
    delayed = g.create_synapse(F, T[0], weight=0.3, delay=3)
    inhib = g.create_synapse(T[0], F, weight=0.3, synapse_type=SynapseType.INHIBITORY)
    st = _bind(g, {"last_forest_id": None}, windows=())
    assert delayed.weight == 0.3 and inhib.weight == 0.3                     # never reinforced
    assert len(_pairs(g)[(F, T[0])]) == 2 and len(_pairs(g)[(T[0], F)]) == 2 # the bind's own edges written beside them
    assert st["reinforced_synapses"] == 0                                     # nothing of the bind's own kind existed
    st2 = _bind(g, {"last_forest_id": None}, windows=())
    assert st2["reinforced_synapses"] == 4 and delayed.weight == 0.3 and inhib.weight == 0.3
    assert len(_pairs(g)[(F, T[0])]) == 2


def test_env_knobs(reinforce_on, monkeypatch):
    p = cc_ng_organism.cc_bind_reinforce_params(None)
    assert p == {"eta": 0.2, "ceiling_factor": 3.0, "he_eta": 0.05, "he_ceiling": 2.0}
    monkeypatch.setenv("CC_NG_BIND_REINFORCE_ETA", "0.5")
    monkeypatch.setenv("CC_NG_BIND_REINFORCE_CEILING_FACTOR", "2")
    monkeypatch.setenv("CC_NG_BIND_REINFORCE_HE_ETA", "junk")
    monkeypatch.setenv("CC_NG_BIND_REINFORCE_HE_CEILING", "99")
    p = cc_ng_organism.cc_bind_reinforce_params(None)
    assert p == {"eta": 0.5, "ceiling_factor": 2.0, "he_eta": 0.05, "he_ceiling": 5.0}   # clamped to the engine cap
    monkeypatch.setenv("CC_NG_BIND_REINFORCE_ETA", "nan")
    assert cc_ng_organism.cc_bind_reinforce_params(None)["eta"] == 0.2
    g = _graph()
    _bind(g, {"last_forest_id": None}, windows=())
    _bind(g, {"last_forest_id": None}, windows=())
    assert _pairs(g)[(F, T[0])][0].weight == pytest.approx(0.2 + 0.2 * (0.4 - 0.2))   # factor 2 -> ceiling 0.4


def test_different_member_set_gets_its_own_hyperedge(reinforce_on):
    g = _graph(extra=[F + "::tree::gamma"])
    _bind(g, {"last_forest_id": None})
    st = cc_ng_organism._cc_bind_conversational_topology(
        g, F, {"tree_ids": [T[0], F + "::tree::gamma"]}, None, {"last_forest_id": None}, window_ids=list(W))
    assert st["reinforced_hyperedge"] is False and len(g.hyperedges) == 2    # a genuinely different set: a new edge
    assert len(_pairs(g)[(F, T[0])]) == 1                                     # the shared pair is still reinforced


def test_already_parallel_edges_reinforce_the_smallest_id(reinforce_off, monkeypatch):
    g = _graph()
    _bind(g, {"last_forest_id": None})
    _bind(g, {"last_forest_id": None})                                        # two parallel sets from before #1173
    monkeypatch.setenv("CC_NG_BIND_REINFORCE_EXISTING", "1")
    before = {s.synapse_id: s.weight for s in _pairs(g)[(F, T[0])]}
    _bind(g, {"last_forest_id": None})
    after = {s.synapse_id: s.weight for s in _pairs(g)[(F, T[0])]}
    assert set(after) == set(before)                                          # still two, none added
    sid = min(before)
    assert after[sid] == pytest.approx(_step(before[sid], 0.2))
    assert [after[k] for k in before if k != sid] == [before[k] for k in before if k != sid]


def test_rollback_undoes_a_reinforcement_exactly(reinforce_on):
    g = _graph()
    _bind(g, {"last_forest_id": None})
    snap_syn = {sid: (s.weight, s.peak_weight) for sid, s in g.synapses.items()}
    he = next(iter(g.hyperedges.values()))
    snap_he = (dict(he.member_weights), dict(he.metadata))
    eco = cc_ng_organism._CCConversationalDualPassEco(g, None)
    _bind(g, {"last_forest_id": None}, journal=eco.topology)
    assert len(eco.topology["reinforced_synapses"]) == 4 and len(eco.topology["reinforced_hyperedges"]) == 1
    assert eco.rollback() is True
    assert {sid: (s.weight, s.peak_weight) for sid, s in g.synapses.items()} == snap_syn
    assert (dict(he.member_weights), dict(he.metadata)) == snap_he


def test_rollback_leaves_a_later_change_by_another_writer(reinforce_on):
    g = _graph()
    _bind(g, {"last_forest_id": None})
    eco = cc_ng_organism._CCConversationalDualPassEco(g, None)
    _bind(g, {"last_forest_id": None}, journal=eco.topology)
    s = _pairs(g)[(F, T[0])][0]
    s.weight = 3.21                                                           # STDP moved it after the reinforcement
    eco.rollback()
    assert s.weight == 3.21


def test_failure_after_a_reinforcement_rolls_back_through_the_dual_pass_path(reinforce_on, monkeypatch):
    """A bind that raises after it reinforced: the reinforcement is journalled before the raise, so the dual pass's
    rollback restores the edge (the #904 all-or-nothing rule holds for reinforcement too)."""
    g = _graph()
    _bind(g, {"last_forest_id": None})
    snap = {sid: s.weight for sid, s in g.synapses.items()}
    eco = cc_ng_organism._CCConversationalDualPassEco(g, None)
    real = cc_ng_organism._cc_reinforce_hyperedge

    def boom(graph, hid, journal, *a):
        raise RuntimeError("injected")
    monkeypatch.setattr(cc_ng_organism, "_cc_reinforce_hyperedge", boom)
    with pytest.raises(RuntimeError):
        _bind(g, {"last_forest_id": None}, journal=eco.topology)
    assert eco.topology["site"] == "hyperedge"
    assert eco.rollback() is True
    assert {sid: s.weight for sid, s in g.synapses.items()} == snap
    monkeypatch.setattr(cc_ng_organism, "_cc_reinforce_hyperedge", real)


def test_dirty_marks_for_incremental_save(reinforce_on):
    g = _graph()
    _bind(g, {"last_forest_id": None})
    g._dirty_synapses.clear()
    g._dirty_hyperedges.clear()
    _bind(g, {"last_forest_id": None})
    assert len(g._dirty_synapses) == 9 and len(g._dirty_hyperedges) == 1     # 4 reinforced + 5 written (windows, chain)


def test_leg2_guards_unchanged_after_the_split():
    g = _graph()
    assert cc_topology_merge._synapse_exists(g, F, T[0]) is False
    assert cc_topology_merge.find_synapse_id(g, F, T[0]) is None
    s = g.create_synapse(F, T[0], weight=0.2)
    assert cc_topology_merge._synapse_exists(g, F, T[0]) is True
    assert cc_topology_merge.find_synapse_id(g, F, T[0]) == s.synapse_id
    assert cc_topology_merge._synapse_exists(g, T[0], F) is False             # direction matters
    m = {F, T[0], T[1]}
    assert cc_topology_merge._hyperedge_exists(g, m) is False
    he = g.create_hyperedge(member_node_ids=m)
    assert cc_topology_merge._hyperedge_exists(g, m) is True
    assert cc_topology_merge.find_hyperedge_id(g, m) == he.hyperedge_id
    assert cc_topology_merge._hyperedge_exists(g, {F, T[0]}) is False         # a subset is not the same set
    assert cc_topology_merge.find_hyperedge_id(g, set()) is None
    # match= (2026-10-24): None is the Leg 2 answer; a predicate narrows it
    d3 = g.create_synapse(T[1], F, weight=0.2, delay=3)
    assert cc_topology_merge.find_synapse_id(g, T[1], F) == d3.synapse_id
    assert cc_topology_merge.find_synapse_id(g, T[1], F, match=cc_ng_organism._cc_is_bind_tree_synapse) is None
    assert cc_topology_merge.find_synapse_id(g, F, T[0], match=cc_ng_organism._cc_is_bind_tree_synapse) == s.synapse_id


def test_state_counters(reinforce_on, monkeypatch):
    """run_conversational_dual_pass adds the bind's reinforcement counts to the caller's state."""
    g = _graph()
    state = {"last_forest_id": None}
    monkeypatch.setattr(cc_ng_organism, "_cc_bind_conversational_topology",
                        lambda *a, **k: {"trees": 0, "reinforced_synapses": 4, "reinforced_hyperedge": True})

    class _NDR:
        @staticmethod
        def get_instance():
            return _NDR()

        def dual_record_outcome(self, **kw):
            return {"tree_ids": []}
    import types
    fake = types.ModuleType("ng_embed")
    fake.NGEmbed = _NDR
    monkeypatch.setitem(sys.modules, "ng_embed", fake)
    ok = cc_ng_organism.run_conversational_dual_pass(g, None, "text", [0.0], state,
                                                     prepared={"ndr": {}, "windows": ()})
    assert ok is True
    assert state["bind_reinforced_synapses_total"] == 4 and state["bind_reinforced_hyperedges_total"] == 1


def test_identity_protection_never_read_by_the_reinforcing_bind(reinforce_on, monkeypatch):
    """Packet 153-155 / #523: identity protection stays out of deposit; the narrowed, soft-bound reinforcement never
    consults it (a recurrence strengthens uniformly, whatever the node)."""
    import neuro_foundation
    calls = []
    real = neuro_foundation.Graph._is_identity_protected

    def spy(self, *a, **k):
        calls.append(a)
        return real(self, *a, **k)
    monkeypatch.setattr(neuro_foundation.Graph, "_is_identity_protected", spy)
    g = _graph()
    for _ in range(3):
        _bind(g, {"last_forest_id": None})
    assert calls == []
