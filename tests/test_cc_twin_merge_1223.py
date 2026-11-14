# ---- Changelog ----
# [2026-11-14] DudeMan CC (Claude Fable 5.2, laptop) -- punch #1223: merge of the bind's stacked forest<->tree copies
# What: cc_ng_organism.cc_merge_bind_tree_twins on graphs the real bind stacked (switch off, the same turn bound again).
# Why:  Josh 2026-11-14 "Just do choice 3"; the merge must leave what #1173's reinforce-on-recurrence would have.
# How:  the bind itself builds the copies; the checks are one edge per pair, the weight rule, nothing else touched,
#       the archive, the dry run, idempotence, and a later recurrence with the switch on.
# -------------------
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import cc_ng_organism as O
from tests import test_cc_bind_reinforce_1173 as B

F, T, W = B.F, B.T, B.W


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    for k in ("CC_NG_BIND_REINFORCE_EXISTING", "CC_NG_BIND_REINFORCE_ETA", "CC_NG_BIND_REINFORCE_CEILING_FACTOR"):
        monkeypatch.delenv(k, raising=False)


def _stacked(n):
    g = B._graph()
    state = {}
    for _ in range(n):
        B._bind(g, state)
    return g


def _state(g, skip=()):
    return sorted((s.synapse_id, s.pre_node_id, s.post_node_id, s.weight, s.peak_weight, s.delay)
                  for s in g.synapses.values() if s.synapse_id not in skip)


def _soft(w, n, birth):
    for _ in range(n - 1):
        w = w + max(0.0, 0.2 * (3.0 * birth - w))
    return w


@pytest.mark.parametrize("n", [2, 3, 7])
def test_merge_leaves_one_edge_per_tree_pair_with_the_soft_bound_weight(n, tmp_path):
    g = _stacked(n)
    pairs = B._pairs(g)
    assert all(len(pairs[(F, t)]) == n and len(pairs[(t, F)]) == n for t in T)
    tree_ids = {s.synapse_id for (a, b), v in pairs.items() if "::tree::" in a + b for s in v}
    others = _state(g, skip=tree_ids)
    arch = tmp_path / "twins.jsonl"
    r = O.cc_merge_bind_tree_twins(g, str(arch))
    assert r["pairs"] == 4 and r["surplus"] == r["removed"] == 4 * (n - 1) and r["skipped"] == 0 and r["largest"] == n
    after = B._pairs(g)
    for t in T:
        (down,), (up,) = after[(F, t)], after[(t, F)]
        assert down.weight == pytest.approx(_soft(0.2, n, 0.2)) and up.weight == pytest.approx(_soft(0.15, n, 0.15))
        assert down.peak_weight >= down.weight and up.peak_weight >= up.weight
        assert down.synapse_id == min(s.synapse_id for s in pairs[(F, t)])       # the one a recurrence reinforces
    assert _state(g, skip=tree_ids) == others                                     # windows, chain: untouched
    assert all(len(after[(F, w)]) == n for w in W)                                # window copies are out of scope
    rows = [json.loads(x) for x in arch.read_text().splitlines()]
    assert len(rows) == 4 and sum(len(x["copies"]) for x in rows) == 4 * n
    assert all(len(x["remove"]) == n - 1 and x["copies"][0]["synapse_id"] == x["keep"] for x in rows)


def test_the_highest_weight_of_a_group_is_kept_and_a_weight_above_the_ceiling_is_not_lowered(tmp_path):
    g = _stacked(3)
    pairs = B._pairs(g)
    a = sorted(pairs[(F, T[0])], key=lambda s: s.synapse_id)
    a[2].weight = 0.5                                   # a later copy learned more: below the 0.6 ceiling
    b = sorted(pairs[(F, T[1])], key=lambda s: s.synapse_id)
    b[1].weight = 2.0                                   # above the ceiling
    b[1].peak_weight = 2.5
    O.cc_merge_bind_tree_twins(g, str(tmp_path / "a.jsonl"))
    after = B._pairs(g)
    assert after[(F, T[0])][0].weight == pytest.approx(_soft(0.5, 3, 0.2))
    assert after[(F, T[1])][0].weight == 2.0 and after[(F, T[1])][0].peak_weight == 2.5


def test_dry_run_without_an_archive_changes_nothing_and_a_second_merge_is_a_no_op(tmp_path):
    g = _stacked(4)
    before = _state(g)
    r = O.cc_merge_bind_tree_twins(g)
    assert r["pairs"] == 4 and r["surplus"] == 12 and r["removed"] == 0 and _state(g) == before
    O.cc_merge_bind_tree_twins(g, str(tmp_path / "a.jsonl"))
    once = _state(g)
    r2 = O.cc_merge_bind_tree_twins(g, str(tmp_path / "b.jsonl"))
    assert r2["pairs"] == 0 and r2["removed"] == 0 and _state(g) == once and not (tmp_path / "b.jsonl").exists()


def test_a_delayed_synapse_on_the_pair_is_a_different_edge_and_stays(tmp_path):
    g = _stacked(2)
    d = g.create_synapse(F, T[0], weight=0.4, delay=3)
    O.cc_merge_bind_tree_twins(g, str(tmp_path / "a.jsonl"))
    on_pair = B._pairs(g)[(F, T[0])]
    assert len(on_pair) == 2 and d.synapse_id in {s.synapse_id for s in on_pair}


def test_indices_and_confirmation_history_follow_the_removal(tmp_path):
    g = _stacked(3)
    gone = sorted(s.synapse_id for s in B._pairs(g)[(F, T[0])])[1:]
    O.cc_merge_bind_tree_twins(g, str(tmp_path / "a.jsonl"))
    assert not set(gone) & set(g.synapses.keys())
    assert not set(gone) & set(g._outgoing.get(F, set())) and not set(gone) & set(g._incoming.get(T[0], set()))
    assert not set(gone) & set(g._synapse_confirmation_history.keys())
    g.step()


def test_after_the_merge_a_recurrence_with_the_switch_on_reinforces_the_one_edge(tmp_path, monkeypatch):
    g = _stacked(3)
    O.cc_merge_bind_tree_twins(g, str(tmp_path / "a.jsonl"))
    w = B._pairs(g)[(F, T[0])][0].weight
    monkeypatch.setenv("CC_NG_BIND_REINFORCE_EXISTING", "1")
    st = B._bind(g, {})
    (one,) = B._pairs(g)[(F, T[0])]
    assert st["reinforced_synapses"] == 4 and one.weight == pytest.approx(w + 0.2 * (0.6 - w))
