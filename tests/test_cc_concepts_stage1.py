# ---- Changelog ----
# [2026-11-10] DudeMan CC (Claude Opus 5.5, laptop; Grok review Minor 2) — + concepts on with a wheel lacking ConceptIndex:
#   ERROR once, recognition off, turns land, no concept founded, backlog idle.
# [2026-10-31] DudeMan CC (Claude Opus 5.5, laptop; lane concepts-stage1, #1203; Josh 2026-10-31 "You are a go")
# What: tests for concept formation stage 1 — the switch and parameters, the concept index, recognition at encoding,
#   sleep-time recognition of the existing past.
# Why: docs/superpowers/plans/2026-10-31-concept-formation-stage1.md (spec 2026-10-31-concept-formation-and-recall-design).
# How: plain Graph + SimpleVectorDB with hand-made vectors, as test_cc_bind_reinforce_1173.py builds its graph.
# -------------------
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import pytest

import cc_ng_organism

_CONCEPT_ENV = ("CC_NG_CONCEPTS", "CC_NG_CONCEPT_MATCH", "CC_NG_CONCEPT_DRIFT", "CC_NG_CONCEPT_ETA",
                "CC_NG_CONCEPT_CEILING", "CC_NG_CONCEPT_BIRTH_UP", "CC_NG_CONCEPT_BIRTH_DOWN",
                "CC_NG_CONCEPT_REACTIVATE", "CC_NG_CONCEPT_BACKLOG_PER_SLEEP", "CC_NG_CONCEPT_BACKLOG_BATCH")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _CONCEPT_ENV:
        monkeypatch.delenv(k, raising=False)


# ---- Task 2: switch and parameters ----

def test_concepts_off_by_default():
    assert cc_ng_organism.cc_concepts_enabled() is False


def test_concepts_on(monkeypatch):
    monkeypatch.setenv("CC_NG_CONCEPTS", "1")
    assert cc_ng_organism.cc_concepts_enabled() is True


def test_params_defaults():
    p = cc_ng_organism.cc_concept_params()
    assert p == {"match": 0.60, "drift": 0.05, "eta": 0.2, "ceiling": 1.0, "birth_up": 0.2, "birth_down": 0.15,
                 "reactivate": 1.1, "backlog": 200, "batch": 10}
    assert isinstance(p["backlog"], int) and isinstance(p["batch"], int)


def test_params_clamped(monkeypatch):
    monkeypatch.setenv("CC_NG_CONCEPT_MATCH", "2")
    monkeypatch.setenv("CC_NG_CONCEPT_DRIFT", "abc")
    monkeypatch.setenv("CC_NG_CONCEPT_BACKLOG_PER_SLEEP", "-5")
    monkeypatch.setenv("CC_NG_CONCEPT_BACKLOG_BATCH", "5000")
    monkeypatch.setenv("CC_NG_CONCEPT_BIRTH_UP", "0")
    p = cc_ng_organism.cc_concept_params()
    assert p["match"] == 0.95
    assert p["drift"] == 0.05
    assert p["backlog"] == 0
    assert p["batch"] == 1000
    assert p["birth_up"] == 0.01


# ---- Task 3: the concept index and nearest-concept lookup ----

import numpy as np
from universal_ingestor import SimpleVectorDB


def _unit(*xs):
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def zero_centre(monkeypatch):
    monkeypatch.setattr(cc_ng_organism, "_cc_concept_centre", lambda dim: np.zeros(dim, dtype=np.float32))


def _concept_row(vdb, cid, vec, label="c"):
    vdb.insert(cid, np.asarray(vec, dtype=np.float32), label, {"cc": True, "_cc_concept": True})


def test_empty_index_returns_no_match(zero_centre):
    assert cc_ng_organism._cc_concept_nearest(SimpleVectorDB(), _unit(1, 0, 0), 0.6) is None


def test_match_above_threshold(zero_centre):
    vdb = SimpleVectorDB()
    _concept_row(vdb, "cc:concept::a", _unit(1, 0, 0))
    _concept_row(vdb, "cc:concept::b", _unit(0, 1, 0))
    vdb.insert("cc:conv::x::tree::q", _unit(1, 0.05, 0), "not a concept", {"cc": True})
    hit = cc_ng_organism._cc_concept_nearest(vdb, _unit(1, 0.1, 0), 0.6)
    assert hit is not None and hit[0] == "cc:concept::a" and hit[1] > 0.9


def test_below_threshold_none(zero_centre):
    vdb = SimpleVectorDB()
    _concept_row(vdb, "cc:concept::a", _unit(1, 0, 0))
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(1, 1.5, 0), 0.6) is None


def test_centred_not_raw(monkeypatch):
    centre = np.asarray([0.6, 0.6, 0.0], dtype=np.float32)
    monkeypatch.setattr(cc_ng_organism, "_cc_concept_centre", lambda dim: centre)
    vdb = SimpleVectorDB()
    _concept_row(vdb, "cc:concept::a", _unit(1, 0.2, 0))
    q = _unit(0.2, 1, 0)
    assert float(np.dot(_unit(1, 0.2, 0), q)) >= 0.35       # raw cosine is sizeable ...
    assert cc_ng_organism._cc_concept_nearest(vdb, q, 0.35) is None   # ... centred it is not a match


def test_dim_mismatch_skipped(zero_centre):
    vdb = SimpleVectorDB()
    _concept_row(vdb, "cc:concept::a", _unit(1, 0, 0))
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(1, 0, 0, 0), 0.6) is None


def test_index_rebuilds_after_vdb_change(zero_centre):
    vdb = SimpleVectorDB()
    _concept_row(vdb, "cc:concept::a", _unit(1, 0, 0))
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(1, 0, 0), 0.6)[0] == "cc:concept::a"
    _concept_row(vdb, "cc:concept::a", _unit(0, 0, 1))      # same object, same count, new vector
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(1, 0, 0), 0.6) is None
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(0, 0, 1), 0.6)[0] == "cc:concept::a"


def test_index_after_load(zero_centre, tmp_path):
    other = SimpleVectorDB()
    _concept_row(other, "cc:concept::z", _unit(0, 1, 0))
    path = str(tmp_path / "v.msgpack")
    other.save(path)
    vdb = SimpleVectorDB()
    _concept_row(vdb, "cc:concept::a", _unit(1, 0, 0))
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(1, 0, 0), 0.6)[0] == "cc:concept::a"
    vdb.load(path)                                          # same object, same concept count, different rows
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(1, 0, 0), 0.6) is None
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(0, 1, 0), 0.6)[0] == "cc:concept::z"


# ---- Task 4: recognition at encoding ----

import copy
import hashlib
import types

from neuro_foundation import Graph

VEC = {"lenia": _unit(1, 0, 0, 0), "lenia2": _unit(1, 0.08, 0, 0), "commons": _unit(0, 1, 0, 0),
       "embed": _unit(0, 0, 1, 0)}


def _fake_embedder(monkeypatch, concepts_by_text):
    """ng_embed stand-in: dual_record_outcome deposits the forest and one tree per concept through the dual pass's own
    ecosystem adapter (so the journal sees them), exactly as the real one does, and returns their ids."""
    class _NDR:
        @staticmethod
        def get_instance():
            return _NDR()

        def embed_windows(self, text):
            return types.SimpleNamespace(windows=())

        def dual_record_outcome(self, ecosystem, content, embedding, target_id, metadata, **kw):
            ecosystem.deposit(target_id, embedding, content, dict(metadata), kind="forest")
            ids = []
            for name in concepts_by_text[content]:
                tid = f"{target_id}::tree::{name}"
                ecosystem.deposit(tid, VEC[name], name, {**metadata, "_tree_concept": True, "_concept": name},
                                  kind="tree")
                ids.append(tid)
            return {"tree_ids": ids}
    fake = types.ModuleType("ng_embed")
    fake.NGEmbed = _NDR
    monkeypatch.setitem(sys.modules, "ng_embed", fake)


def _turn(g, vdb, state, text):
    return cc_ng_organism.run_conversational_dual_pass(g, vdb, text, _unit(0, 0, 0, 1), state)


def _tree(text, name):
    return "cc:conv::" + hashlib.sha1(text.encode()).hexdigest() + "::tree::" + name


def _concepts(g):
    return [n for n, node in g.nodes.items() if node.metadata.get("_cc_concept")]


def _pairs(g):
    out = {}
    for s in g.synapses.values():
        out.setdefault((s.pre_node_id, s.post_node_id), []).append(s)
    return out


@pytest.fixture
def concepts_on(monkeypatch, zero_centre):
    monkeypatch.setenv("CC_NG_CONCEPTS", "1")


def test_off_is_byte_identical(monkeypatch, zero_centre):
    _fake_embedder(monkeypatch, {"a lenia turn": ["lenia"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    assert _turn(g, vdb, st, "a lenia turn") is True
    assert _concepts(g) == []
    assert all("_cc_concept_id" not in n.metadata for n in g.nodes.values())
    assert len(g.synapses) == 2          # forest<->tree only


def test_without_the_native_index_concepts_log_error_once_and_stay_off(monkeypatch, concepts_on, caplog):
    """[2026-11-10] Grok review Minor 2: a wheel with no ConceptIndex -> ERROR once, recognition off (as S10 / S13 /
    S18), the turn still lands and no concept is founded; the sleep backlog does nothing either."""
    import logging
    import ng_tract
    monkeypatch.delattr(ng_tract, "ConceptIndex")
    monkeypatch.setattr(cc_ng_organism, "_CC_CONCEPTS_NATIVE_ERR", [False])
    _fake_embedder(monkeypatch, {"proto hits lenia": ["lenia"], "back to lenia later": ["lenia2"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    with caplog.at_level(logging.ERROR, logger="cc_ng_organism"):
        assert _turn(g, vdb, st, "proto hits lenia") is True
        assert _turn(g, vdb, st, "back to lenia later") is True
        out = cc_ng_organism.cc_concept_recognize_backlog(g, vdb, 100, 10)
    assert _concepts(g) == []
    assert all("_cc_concept_id" not in n.metadata for n in g.nodes.values())
    assert out["matched"] == out["created"] == out["failed"] == 0
    errs = [r for r in caplog.records if r.levelno == logging.ERROR and "ConceptIndex" in r.getMessage()]
    assert len(errs) == 1


def test_same_concept_two_turns_one_node(monkeypatch, concepts_on):
    _fake_embedder(monkeypatch, {"proto hits lenia": ["lenia"], "back to lenia later": ["lenia2"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    assert _turn(g, vdb, st, "proto hits lenia") and _turn(g, vdb, st, "back to lenia later")
    cs = _concepts(g)
    assert len(cs) == 1
    cid = cs[0]
    for t in (_tree("proto hits lenia", "lenia"), _tree("back to lenia later", "lenia2")):
        assert (t, cid) in _pairs(g) and (cid, t) in _pairs(g)
        assert g.nodes[t].metadata["_cc_concept_id"] == cid
    assert g.nodes[cid].metadata["_cc_recognitions"] == 2
    assert st["concepts_matched_total"] == 1 and st["concepts_created_total"] == 1


def test_different_concept_two_nodes(monkeypatch, concepts_on):
    _fake_embedder(monkeypatch, {"lenia turn": ["lenia"], "commons turn": ["commons"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    assert _turn(g, vdb, st, "lenia turn") and _turn(g, vdb, st, "commons turn")
    assert len(_concepts(g)) == 2


def test_episode_unchanged(monkeypatch, concepts_on):
    _fake_embedder(monkeypatch, {"first lenia": ["lenia"], "second lenia": ["lenia2"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    _turn(g, vdb, st, "first lenia")
    t1 = _tree("first lenia", "lenia")
    before_meta = {k: copy.deepcopy(v) for k, v in g.nodes[t1].metadata.items() if k != "_cc_concept_id"}
    before_row = vdb.get(t1)
    before_vec, before_content = np.array(before_row["embedding"]), before_row["content"]
    _turn(g, vdb, st, "second lenia")
    after_meta = {k: v for k, v in g.nodes[t1].metadata.items() if k != "_cc_concept_id"}
    assert after_meta == before_meta
    row = vdb.get(t1)
    assert np.array_equal(row["embedding"], before_vec) and row["content"] == before_content


def test_prototype_drifts(monkeypatch, concepts_on):
    _fake_embedder(monkeypatch, {"one": ["lenia"], "two": ["lenia2"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    _turn(g, vdb, st, "one")
    _turn(g, vdb, st, "two")
    cid = _concepts(g)[0]
    want = 0.95 * VEC["lenia"] + 0.05 * VEC["lenia2"]
    want = want / np.linalg.norm(want)
    assert np.allclose(vdb.get(cid)["embedding"], want, atol=1e-6)


def test_two_trees_same_turn_same_concept(monkeypatch, concepts_on):
    _fake_embedder(monkeypatch, {"lenia and lenia": ["lenia", "lenia2"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    assert _turn(g, vdb, st, "lenia and lenia")
    assert len(_concepts(g)) == 1
    assert all(len(v) == 1 for v in _pairs(g).values())


def test_tree_without_vector_is_skipped(monkeypatch, concepts_on):
    _fake_embedder(monkeypatch, {"a turn": ["lenia"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    real_get = vdb.get
    monkeypatch.setattr(vdb, "get", lambda i: None if "::tree::" in i else real_get(i))
    assert _turn(g, vdb, st, "a turn") is True
    assert _concepts(g) == [] and st["concepts_skipped_total"] == 1
    assert g.nodes[_tree("a turn", "lenia")].metadata.get("_cc_concept_skip")


def test_no_hyperedge_created(monkeypatch, concepts_on):
    _fake_embedder(monkeypatch, {"one": ["lenia"], "two": ["lenia2"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    _turn(g, vdb, st, "one")
    n_he = len(g.hyperedges)
    monkeypatch.delenv("CC_NG_CONCEPTS")
    _turn(g, vdb, st, "two")
    off_he = len(g.hyperedges) - n_he
    monkeypatch.setenv("CC_NG_CONCEPTS", "1")
    g2, vdb2, st2 = Graph(), SimpleVectorDB(), {}
    _turn(g2, vdb2, st2, "one")
    _turn(g2, vdb2, st2, "two")
    assert len(g2.hyperedges) == n_he + off_he
    assert all(not any(m.startswith("cc:concept::") for m in he.member_nodes) for he in g2.hyperedges.values())


def _state_of(g, vdb):
    nodes = {n: (copy.deepcopy(dict(node.metadata)), float(node.voltage)) for n, node in g.nodes.items()}
    syns = {s: (x.pre_node_id, x.post_node_id, float(x.weight), float(x.peak_weight)) for s, x in g.synapses.items()}
    rows = {i: (np.array(vdb.embeddings[i]), vdb.content[i], copy.deepcopy(vdb.metadata[i])) for i in vdb.embeddings}
    return nodes, syns, rows


def _same_state(a, b):
    (na, sa, ra), (nb, sb, rb) = a, b
    assert na.keys() == nb.keys() and sa == sb and ra.keys() == rb.keys()
    for k in na:
        assert na[k] == nb[k], k
    for k in ra:
        assert np.allclose(ra[k][0], rb[k][0]) and ra[k][1] == rb[k][1] and ra[k][2] == rb[k][2], k


@pytest.mark.parametrize("fail_after", ["node", "syn_pair", "reinforce", "proto", "meta", "volt"])
def test_rollback_restores_concepts(monkeypatch, concepts_on, fail_after):
    third_concept = "embed" if fail_after == "node" else "lenia"     # "node": the third turn founds a NEW concept
    _fake_embedder(monkeypatch, {"one": ["lenia"], "two": ["lenia2"], "three": [third_concept]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    _turn(g, vdb, st, "one")
    _turn(g, vdb, st, "two")        # the third turn matches an EXISTING concept (or founds one, for "node")
    for _ in range(5):              # the concept cools between experiences, so the third turn reactivates it
        g.step()
    before = _state_of(g, vdb)
    real = cc_ng_organism._cc_concept_journal_add
    seen = []

    def boom(journal, entry):
        real(journal, entry)
        seen.append(entry[0])
        if entry[0] == fail_after:
            raise RuntimeError("injected after " + fail_after)
    monkeypatch.setattr(cc_ng_organism, "_cc_concept_journal_add", boom)
    third = "three"
    ok = _turn(g, vdb, st, third)
    if fail_after not in seen:
        pytest.skip(f"no {fail_after} write on this path")
    assert ok is False
    after = _state_of(g, vdb)
    tree3 = _tree(third, third_concept)
    assert tree3 not in g.nodes
    _same_state(before, after)
    assert cc_ng_organism._cc_concept_nearest(vdb, VEC["lenia"], 0.6)[0] == _concepts(g)[0]


def test_concept_fires_with_turn(monkeypatch, concepts_on):
    _fake_embedder(monkeypatch, {"one": ["lenia"], "two": ["lenia2"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    _turn(g, vdb, st, "one")
    _turn(g, vdb, st, "two")
    cid = _concepts(g)[0]
    res = g.step()
    assert cid in set(getattr(res, "fired_node_ids", []) or [])


def test_label_is_most_central(monkeypatch, concepts_on):
    VEC["lenia3"] = _unit(1, -0.3, 0, 0)
    _fake_embedder(monkeypatch, {"a": ["lenia3"], "b": ["lenia"], "c": ["lenia2"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    for t in ("a", "b", "c"):
        _turn(g, vdb, st, t)
    cid = _concepts(g)[0]
    proto = vdb.get(cid)["embedding"]
    best = max(("lenia3", "lenia", "lenia2"), key=lambda n: float(np.dot(VEC[n], proto)))
    assert g.nodes[cid].metadata["_label"] == best


def test_poincare_follows_prototype(monkeypatch, concepts_on):
    from neuro_foundation import poincare_dir_array
    _fake_embedder(monkeypatch, {"one": ["lenia"], "two": ["lenia2"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    _turn(g, vdb, st, "one")
    _turn(g, vdb, st, "two")
    cid = _concepts(g)[0]
    pd = poincare_dir_array(g.nodes[cid].metadata)
    v = np.asarray(vdb.get(cid)["embedding"])
    assert np.allclose(pd, v / np.linalg.norm(v), atol=1e-5)


def test_mark_repair(monkeypatch, concepts_on):
    _fake_embedder(monkeypatch, {"one": ["lenia"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    _turn(g, vdb, st, "one")
    t, cid = _tree("one", "lenia"), _concepts(g)[0]
    sid = _pairs(g)[(cid, t)][0].synapse_id
    g.remove_synapse(sid)
    n = g.nodes[cid].metadata["_cc_recognitions"]
    assert cc_ng_organism._cc_concept_recognize_tree(g, vdb, t, cc_ng_organism.cc_concept_params(), []) == "repaired"
    assert (cid, t) in _pairs(g) and g.nodes[cid].metadata["_cc_recognitions"] == n


def test_checkpoint_round_trip(monkeypatch, concepts_on, tmp_path):
    _fake_embedder(monkeypatch, {"one": ["lenia"], "two": ["lenia2"], "three": ["lenia"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    _turn(g, vdb, st, "one")
    _turn(g, vdb, st, "two")
    cid = _concepts(g)[0]
    gp, vp = str(tmp_path / "g.msgpack"), str(tmp_path / "v.msgpack")
    g.checkpoint(gp)
    vdb.save(vp)
    g2, vdb2 = Graph(), SimpleVectorDB()
    g2.restore(gp)
    vdb2.load(vp)
    m = g2.nodes[cid].metadata
    assert m["_cc_concept"] and m["_cc_recognitions"] == 2 and m["_label"] and "poincare_dir" in m
    assert m is not vdb2.metadata[cid]
    assert g2.nodes[_tree("one", "lenia")].metadata["_cc_concept_id"] == cid
    assert np.allclose(vdb2.get(cid)["embedding"], vdb.get(cid)["embedding"])
    assert _turn(g2, vdb2, {}, "three")
    assert _concepts(g2) == [cid] and g2.nodes[cid].metadata["_cc_recognitions"] == 3


# ---- Task 5: sleep-time recognition of the existing past ----

import time


def _past_tree(g, vdb, tid, vec, ctime, concept="x"):
    n = g.create_node(node_id=tid, metadata={"_tree_concept": True, "_concept": concept, "cc": True})
    n.creation_time = float(ctime)
    vdb.insert(tid, np.asarray(vec, dtype=np.float32), concept, {"cc": True, "_tree_concept": True})
    return n


def _backlog(g, vdb, budget=200, batch=25):
    return cc_ng_organism.cc_concept_recognize_backlog(g, vdb, budget, batch)


def test_backlog_oldest_first_and_budget(concepts_on):
    g, vdb = Graph(), SimpleVectorDB()
    for i, t in enumerate([50, 10, 40, 20, 30]):
        _past_tree(g, vdb, f"cc:conv::{i}::tree::x", _unit(1, 0.01 * i, 0, 0), t)
    out = _backlog(g, vdb, budget=2)
    marked = sorted(n for n, node in g.nodes.items() if node.metadata.get("_cc_concept_id"))
    assert marked == ["cc:conv::1::tree::x", "cc:conv::3::tree::x"]      # creation_time 10 and 20
    assert out["created"] + out["matched"] == 2 and out["remaining"] == 3


def test_backlog_idempotent(concepts_on):
    g, vdb = Graph(), SimpleVectorDB()
    for i in range(3):
        _past_tree(g, vdb, f"cc:conv::{i}::tree::x", _unit(1, 0.01 * i, 0, 0), i)
    _backlog(g, vdb)
    n_syn, n_nodes = len(g.synapses), len(g.nodes)
    out = _backlog(g, vdb)
    assert out["created"] == out["matched"] == out["repaired"] == 0
    assert (len(g.synapses), len(g.nodes)) == (n_syn, n_nodes)


def test_backlog_off_noop(zero_centre):
    g, vdb = Graph(), SimpleVectorDB()
    _past_tree(g, vdb, "cc:conv::0::tree::x", _unit(1, 0, 0, 0), 0)
    out = _backlog(g, vdb)
    assert out["created"] == 0 and _concepts(g) == []


def test_backlog_tree_failure_rolls_back_only_that_tree(monkeypatch, concepts_on):
    g, vdb = Graph(), SimpleVectorDB()
    a = _past_tree(g, vdb, "cc:conv::a::tree::x", _unit(1, 0, 0, 0), 1)
    _past_tree(g, vdb, "cc:conv::b::tree::x", _unit(1, 0.05, 0, 0), 2)
    _backlog(g, vdb)
    cid = a.metadata["_cc_concept_id"]
    # tree b loses its concept->tree link: the next backlog REPAIRS it, which REINFORCES the surviving tree->concept link
    sid = _pairs(g)[(cid, "cc:conv::b::tree::x")][0].synapse_id
    g.remove_synapse(sid)
    up = _pairs(g)[("cc:conv::b::tree::x", cid)][0]
    _past_tree(g, vdb, "cc:conv::c::tree::x", _unit(0, 1, 0, 0), 3)   # an unrelated tree after it
    before = _state_of(g, vdb)
    real = cc_ng_organism._cc_concept_journal_add

    def boom(journal, entry):
        real(journal, entry)
        if entry[0] == "reinforce":
            raise RuntimeError("injected after reinforce")
    monkeypatch.setattr(cc_ng_organism, "_cc_concept_journal_add", boom)
    out = _backlog(g, vdb)
    assert out["failed"] == 1
    assert float(up.weight) == before[1][up.synapse_id][2]           # the reinforcement was undone
    assert (cid, "cc:conv::b::tree::x") not in _pairs(g)             # tree b: nothing left of its attempt
    assert g.nodes["cc:conv::c::tree::x"].metadata.get("_cc_concept_id")   # tree c still recognised


def test_backlog_vectorless_not_retried(concepts_on):
    g, vdb = Graph(), SimpleVectorDB()
    n = g.create_node(node_id="cc:conv::v::tree::x", metadata={"_tree_concept": True, "_concept": "x"})
    n.creation_time = 0.0
    assert _backlog(g, vdb)["skipped"] == 1
    assert n.metadata.get("_cc_concept_skip") == "no_vector"
    assert _backlog(g, vdb)["skipped"] == 0 and _backlog(g, vdb)["remaining"] == 0


class _CountingLock:
    def __init__(self, inner):
        self.inner, self.enters = inner, 0

    def __enter__(self):
        self.inner.acquire()
        self.enters += 1
        return self

    def __exit__(self, *a):
        self.inner.release()

    def acquire(self, *a, **k):
        return self.inner.acquire(*a, **k)

    def release(self):
        return self.inner.release()


def test_backlog_releases_lock(concepts_on):
    g, vdb = Graph(), SimpleVectorDB()
    for i in range(5):
        _past_tree(g, vdb, f"cc:conv::{i}::tree::x", _unit(1, 0, 0.3 * i, 0), i)
    lock = _CountingLock(g._step_lock)
    g._step_lock = lock
    _backlog(g, vdb, budget=5, batch=2)
    assert lock.enters >= 3                        # 2 + 2 + 1 trees: at least three separate holds


def test_backlog_lock_hold_bounded(monkeypatch, concepts_on):
    rng = np.random.default_rng(7)
    g, vdb = Graph(), SimpleVectorDB()
    for i in range(30000 - 5000):
        g.create_node(node_id=f"n{i}")
    for i in range(3000):                              # existing concepts
        v = rng.standard_normal(768).astype(np.float32)
        g.create_node(node_id=f"cc:concept::{i:016d}", metadata={"_cc_concept": True, "_label": "c",
                                                                 "_cc_recognitions": 1})
        vdb.insert(f"cc:concept::{i:016d}", v, "c", {"cc": True, "_cc_concept": True})
    for i in range(2000):                              # the past to recognise
        _past_tree(g, vdb, f"cc:conv::{i}::tree::x", rng.standard_normal(768), i)
    t0 = time.monotonic()
    out = _backlog(g, vdb, budget=2000, batch=10)
    assert out["max_hold_s"] < 1.0, out
    assert out["created"] + out["matched"] == 2000
    print("backlog 2000 trees:", round(time.monotonic() - t0, 1), "s, max hold", round(out["max_hold_s"], 3), "s")


# ---- Final-review fixes (GPT-6.1 Sol, 2026-10-31) ----

def test_create_failure_after_mutation_rolls_back(monkeypatch, concepts_on):
    """F1: a failure INSIDE create_node / create_synapse after they mutated leaves nothing behind."""
    _fake_embedder(monkeypatch, {"one": ["lenia"]})
    for target in ("create_node", "create_synapse"):
        g, vdb, st = Graph(), SimpleVectorDB(), {}
        real = getattr(g, target)
        calls = {"n": 0}

        def wrapped(*a, _real=real, _t=target, **k):
            out = _real(*a, **k)
            nid = k.get("node_id") or (a[0] if a else "")
            if (_t == "create_node" and str(nid).startswith("cc:concept::")) or \
               (_t == "create_synapse" and any(str(x).startswith("cc:concept::") for x in a[:2])):
                raise RuntimeError("fail after mutation in " + _t)
            return out
        monkeypatch.setattr(g, target, wrapped)
        assert _turn(g, vdb, st, "one") is False
        assert _concepts(g) == [] and not any(i.startswith("cc:concept::") for i in vdb.embeddings), target
        assert not any(s.pre_node_id.startswith("cc:concept::") or s.post_node_id.startswith("cc:concept::")
                       for s in g.synapses.values()), target


def test_backlog_batch_bounds_examined_trees(concepts_on):
    """F2: already-recognised trees count toward the batch, and the completeness check stays cheap on a huge concept."""
    g, vdb = Graph(), SimpleVectorDB()
    for i in range(2500):
        _past_tree(g, vdb, f"cc:conv::{i}::tree::x", _unit(1, 0, 0, 0), i)
    _backlog(g, vdb, budget=100000, batch=500)          # all 2,500 share one concept now
    lock = _CountingLock(g._step_lock)
    g._step_lock = lock
    out = _backlog(g, vdb, budget=1, batch=1)
    assert out["max_hold_s"] < 0.5, out
    assert lock.enters >= 2500 // 1 // 2               # many short holds, not one long one


def test_index_sees_concepts_added_by_load(zero_centre, tmp_path):
    """F3: an empty index must not stay empty after concepts arrive in the same store (load / merge)."""
    other = SimpleVectorDB()
    _concept_row(other, "cc:concept::z", _unit(0, 1, 0))
    path = str(tmp_path / "v.msgpack")
    other.save(path)
    vdb = SimpleVectorDB()
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(0, 1, 0), 0.6) is None
    vdb.load(path)
    hit = cc_ng_organism._cc_concept_nearest(vdb, _unit(0, 1, 0), 0.6)
    assert hit is not None and hit[0] == "cc:concept::z"
    _concept_row(vdb, "cc:concept::y", _unit(1, 0, 0))   # an outside writer adds a concept row
    assert cc_ng_organism._cc_concept_nearest(vdb, _unit(1, 0, 0), 0.6)[0] == "cc:concept::y"


def test_exact_repeat_is_recurrence(monkeypatch, concepts_on):
    monkeypatch.setenv("CC_NG_BIND_REINFORCE_EXISTING", "1")
    _fake_embedder(monkeypatch, {"same words": ["lenia"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    _turn(g, vdb, st, "same words")
    cid = _concepts(g)[0]
    g.nodes[cid].voltage = 0.0
    t = _tree("same words", "lenia")
    w_before = float(_pairs(g)[(t, cid)][0].weight)
    assert _turn(g, vdb, st, "same words")
    assert g.nodes[cid].metadata["_cc_recognitions"] == 2
    assert float(_pairs(g)[(t, cid)][0].weight) > w_before
    assert g.nodes[cid].voltage > 0.0
    assert st.get("concepts_recurred_total") == 1


def test_cold_concept_fires_with_default_reactivation(monkeypatch, concepts_on):
    """F5: with DEFAULT settings, a cold matched concept fires on the step after the turn (intervening steps between)."""
    _fake_embedder(monkeypatch, {"one": ["lenia"], "two": ["lenia2"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    _turn(g, vdb, st, "one")
    for _ in range(10):
        g.step()
    cid = _concepts(g)[0]
    assert g.nodes[cid].voltage < g.nodes[cid].threshold
    _turn(g, vdb, st, "two")
    fired = set(g.step().fired_node_ids or [])
    assert cid in fired


def test_episode_vector_row_metadata_unchanged(monkeypatch, concepts_on):
    """F10: recognition must not change the episode's vector-row metadata (deposits alias node and row metadata)."""
    _fake_embedder(monkeypatch, {"aliased": ["lenia"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    monkeypatch.delenv("CC_NG_CONCEPTS")
    _turn(g, vdb, st, "aliased")
    t = _tree("aliased", "lenia")
    g.nodes[t].metadata = vdb.metadata[t]              # the live deposit path's aliasing
    row_before = copy.deepcopy(vdb.metadata[t])
    monkeypatch.setenv("CC_NG_CONCEPTS", "1")
    cc_ng_organism.cc_concept_recognize_backlog(g, vdb, 10, 10)
    assert g.nodes[t].metadata.get("_cc_concept_id")
    assert vdb.metadata[t] == row_before


def test_turn_path_hold_recorded(monkeypatch, concepts_on):
    """Law Enforcer condition 1: the encoding-time lock hold is measured (read by the copy-run census)."""
    _fake_embedder(monkeypatch, {"one": ["lenia"]})
    g, vdb, st = Graph(), SimpleVectorDB(), {}
    _turn(g, vdb, st, "one")
    last, mx = g.__dict__["_cc_concept_hold"]
    assert 0.0 <= last <= mx
