# ---- Changelog ----
# [2026-10-23] Claude Code (Opus 5.5, laptop) — #1173 reinforce on recurrence (Josh 2026-10-23 ~19:30: "it should reinforce
#   existing. We already do that for the leg 2 laptop/vps sync")
# What: tests for _cc_bind_conversational_topology with CC_NG_BIND_REINFORCE_EXISTING: a repeat bind reinforces the existing
#   pre->post synapses and the existing same-member hyperedge (no parallel ones), the switch OFF is the old behaviour, the
#   rollback undoes a reinforcement, and the Leg 2 guards (cc_topology_merge._synapse_exists / _hyperedge_exists) still give
#   the same answers after their split into find_synapse_id / find_hyperedge_id.
# Why: #1173 point 5 of the approved fix set.
# How: a plain neuro_foundation.Graph (no checkpoint, no model, no network); cc_anticipate is the real one.
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
    assert len(g.synapses) == n_syn and len(g.hyperedges) == n_he            # nothing parallel
    # 2 tree pairs + 2 window pairs + 1 window-chain link, all reinforced
    assert st2["reinforced_synapses"] == 9 and st2["reinforced_hyperedge"] is True
    assert st2["tree_pairs"] == 2 and st2["window_pairs"] == 2 and st2["hyperedge"] is True   # the turn IS bound
    p = _pairs(g)
    assert all(len(v) == 1 for v in p.values())
    assert p[(F, T[0])][0].weight == pytest.approx(w_ft + 0.2)               # the weight a new synapse would have had
    assert p[(T[0], F)][0].weight == pytest.approx(w_tf + 0.15)
    assert p[(F, T[0])][0].peak_weight == pytest.approx(w_ft + 0.2)
    he = next(iter(g.hyperedges.values()))
    assert he.metadata["cc_recurrences"] == 1
    assert all(w == pytest.approx(1.0 + 0.05) for w in he.member_weights.values())


def test_reinforce_caps_at_max_weight(reinforce_on):
    g = _graph()
    _bind(g, {"last_forest_id": None})
    for _ in range(200):
        _bind(g, {"last_forest_id": None})
    s = _pairs(g)[(F, T[0])]
    assert len(s) == 1 and s[0].weight == pytest.approx(s[0].max_weight)
    he = next(iter(g.hyperedges.values()))
    assert max(he.member_weights.values()) <= 5.0
    assert he.metadata["cc_recurrences"] == 200


def test_sequence_link_reinforced_on_repeat_pair(reinforce_on):
    g = _graph(extra=["cc:conv::prev"])
    _bind(g, {"last_forest_id": "cc:conv::prev"})
    st = _bind(g, {"last_forest_id": "cc:conv::prev"})
    assert st["sequence"] == "written"
    assert len(_pairs(g)[("cc:conv::prev", F)]) == 1


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
    assert after[sid] == pytest.approx(before[sid] + 0.2)
    assert [after[k] for k in before if k != sid] == [before[k] for k in before if k != sid]


def test_rollback_undoes_a_reinforcement_exactly(reinforce_on):
    g = _graph()
    _bind(g, {"last_forest_id": None})
    snap_syn = {sid: (s.weight, s.peak_weight) for sid, s in g.synapses.items()}
    he = next(iter(g.hyperedges.values()))
    snap_he = (dict(he.member_weights), dict(he.metadata))
    eco = cc_ng_organism._CCConversationalDualPassEco(g, None)
    _bind(g, {"last_forest_id": None}, journal=eco.topology)
    assert len(eco.topology["reinforced_synapses"]) == 9 and len(eco.topology["reinforced_hyperedges"]) == 1
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

    def boom(graph, hid, journal):
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
    assert len(g._dirty_synapses) == 9 and len(g._dirty_hyperedges) == 1


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
