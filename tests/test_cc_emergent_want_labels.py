#!/usr/bin/env python3
# ---- Changelog ----
# [2026-10-04] Claude (lane emergent-want-labels) — an emergent want's text is the nodes' own words, never their ids
# What: against a REAL neuro_foundation.Graph with real-shaped CC ids (cc:conv::<40hex> forests, ...::tree::<concept>
#   trees), prove generate_emergent_want's want_text carries no `cc:conv::` id or 40-hex hash, carries the tree concept /
#   the start of the forest turn, keeps the ids in the want node's METADATA, mints nothing when no referenced node has
#   words (debug reason=no_readable_text), and leaves the dedup identity (concept_key / want_id) as it was.
# Why: want nodes now surface as their want_text (P408, cb6ad82); id-built text meant nothing there. The Tonic-bridge
#   spec (docs/superpowers/specs/2026-05-15-syl-tonic-bridge.md) intended {label}→{label}.
# How: real Graph + real SimpleVectorDB; prime_and_propagate stubbed only where a test needs a concept to resolve.
# -------------------
import hashlib
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKTREE))

from neuro_foundation import Graph, Prediction  # noqa: E402
from universal_ingestor import SimpleVectorDB  # noqa: E402
import cc_ng_organism as cno  # noqa: E402

H1 = "213a97a5feab413e5586d8be509e5c9277d86273"
H2 = "2c1ed0db9f0a4c11b2e3d4f5a6b7c8d9e0f1a2b3"
FOREST1 = "cc:conv::" + H1
TREE1 = FOREST1 + "::tree::Python"
FOREST2 = "cc:conv::" + H2
TREE2 = FOREST2 + "::tree::recall budget"
TURN1 = "We should look at how the Python ingest path paces itself under load."
TURN2 = "Josh asked why the recall budget drops whole items instead of trimming them."
ID_RE = re.compile(r"cc:conv::|[0-9a-f]{40}")
DIM = 8


def _graph():
    g = Graph()
    g.create_node(node_id=FOREST1, metadata={"_forest_content": TURN1, "creation_mode": "conversational"})
    g.create_node(node_id=TREE1, metadata={"_tree_concept": True, "_concept": "Python", "_forest_content": TURN1,
                                           "_forest_target_id": FOREST1})
    g.create_node(node_id=FOREST2, metadata={"_forest_content": TURN2, "creation_mode": "conversational"})
    g.create_node(node_id=TREE2, metadata={"_tree_concept": True, "_concept": "recall budget", "_forest_content": TURN2,
                                           "_forest_target_id": FOREST2})
    return g


def _predict(g, *pairs, conf=0.9):
    for i, (src, tgt) in enumerate(pairs):
        g.active_predictions["p%d" % i] = Prediction(source_node_id=src, target_node_id=tgt, confidence=conf - 0.01 * i)


def _resolve_concept_to(g, nid):
    """Make the attractor fire `nid` and give it the only vdb embedding, so the concept resolves to it."""
    vdb = SimpleVectorDB()
    vdb.insert(id=nid, embedding=np.eye(DIM, dtype=np.float32)[0], content="shard", metadata={})
    g.prime_and_propagate = lambda **kw: SimpleNamespace(fired_entries=[SimpleNamespace(node_id=nid)])
    return vdb


def test_want_text_is_built_from_labels_never_ids_and_ids_stay_in_metadata():
    g = _graph()
    vdb = _resolve_concept_to(g, TREE1)
    _predict(g, (TREE1, FOREST2), (TREE2, FOREST1))
    res = cno.generate_emergent_want(g, vdb)
    assert res is not None
    text = res["text"]
    assert not ID_RE.search(text), text
    assert text.startswith("tonic-triggered: Python -- open questions: ")
    assert "Python→" + TURN2 in text and "recall budget→" + TURN1 in text
    meta = g.nodes[res["id"]].metadata
    assert meta["want_text"] == text
    assert meta["emergent_concept_node_id"] == TREE1
    assert meta["emergent_seed_ids"] == [TREE1, TREE2]
    assert meta["emergent_open_question_ids"] == [[TREE1, FOREST2], [TREE2, FOREST1]]
    # dedup identity unchanged: the concept_key is still the structural key (no `label` -> the node id)
    assert meta["concept_key"] == "tonic-concept::" + TREE1
    assert res["id"] == "cc:want::" + hashlib.sha1(("tonic-concept::" + TREE1).encode()).hexdigest()[:16]
    assert meta["provenance"] == "cc_emergent" and meta["creation_mode"] == "emergent"


def test_before_after_the_old_id_text_is_gone():
    """The exact shape reported live: tree seed -> forest target, label-less branch."""
    g = _graph()
    _predict(g, (TREE1, FOREST2))
    res = cno.generate_emergent_want(g, None)
    old = "tonic-triggered: (unknown) -- open questions: %s→%s" % (TREE1, FOREST2)
    assert res["text"] == "tonic-triggered -- open questions: Python→" + TURN2
    assert res["text"] != old
    # the label-less identity is unchanged: the same id the old text hashed to
    assert res["id"] == "cc:want::" + hashlib.sha1(old.encode()).hexdigest()[:16]


def test_long_forest_turn_is_bounded_to_its_start_with_visible_elision():
    g = _graph()
    long_turn = "word " * 200
    g.nodes[FOREST2].metadata["_forest_content"] = long_turn
    _predict(g, (TREE1, FOREST2))
    text = cno.generate_emergent_want(g, None)["text"]
    q = text.split("open questions: ", 1)[1]
    assert q.startswith("Python→word word") and q.endswith("…")
    assert len(q) <= len("Python→") + cno._CC_EMERGENT_WANT_LABEL_CHARS + 1


def test_a_node_without_words_is_left_out():
    g = _graph()
    g.create_node(node_id="cc:conv::" + "a" * 40, metadata={})            # no words at all
    _predict(g, (TREE1, "cc:conv::" + "a" * 40), (TREE2, "cc:conv::missing"))
    text = cno.generate_emergent_want(g, None)["text"]
    assert text == "tonic-triggered -- open questions: Python, recall budget"
    assert not ID_RE.search(text)


def test_a_label_that_is_just_the_id_and_a_tree_below_the_floor_are_not_text():
    g = Graph()
    g.create_node(node_id="s1", metadata={"label": "s1"})
    g.create_node(node_id=TREE1, metadata={"_tree_concept": True, "_concept": "the", "_forest_content": TURN1})
    _predict(g, ("s1", TREE1))
    assert cno.generate_emergent_want(g, None) is None                   # a tree never borrows its parent turn


def test_an_emergent_want_seed_contributes_no_nested_text():
    g = _graph()
    g.create_node(node_id="cc:want::0123456789abcdef", metadata={
        "kind": "want", "want_state": "open", "creation_mode": "emergent", "provenance": "cc_emergent",
        "want_text": "tonic-triggered: Python -- open questions: x"})
    _predict(g, ("cc:want::0123456789abcdef", TREE2))
    text = cno.generate_emergent_want(g, None)["text"]
    assert text == "tonic-triggered -- open questions: recall budget"


def test_nothing_readable_mints_no_want_and_logs_at_debug(caplog):
    g = Graph()
    g.create_node(node_id=FOREST1, metadata={})
    g.create_node(node_id=FOREST2, metadata={"label": FOREST2})
    _predict(g, (FOREST1, FOREST2))
    n_before = len(g.nodes)
    with caplog.at_level(logging.DEBUG, logger="cc_ng_organism"):
        assert cno.generate_emergent_want(g, None) is None
    assert len(g.nodes) == n_before and not g.synapses
    assert not any(n.startswith("cc:want::") for n in g.nodes)
    recs = [r for r in caplog.records if r.name == "cc_ng_organism" and "no_readable_text" in r.getMessage()]
    assert len(recs) == 1 and recs[0].levelno == logging.DEBUG
    assert not ID_RE.search(recs[0].getMessage())


def test_reinforcement_heals_an_old_id_built_text_and_refreshes_provenance():
    g = _graph()
    vdb = _resolve_concept_to(g, TREE1)
    wid = "cc:want::" + hashlib.sha1(("tonic-concept::" + TREE1).encode()).hexdigest()[:16]
    g.create_node(node_id=wid, metadata={
        "kind": "want", "want_state": "open", "provenance": "cc_emergent", "creation_mode": "emergent",
        "concept_key": "tonic-concept::" + TREE1,
        "want_text": "tonic-triggered: %s -- open questions: %s→%s" % (TREE1, TREE1, FOREST2)})
    _predict(g, (TREE1, FOREST2))
    res = cno.generate_emergent_want(g, vdb)
    assert res["reinforced"] is True and res["id"] == wid
    meta = g.nodes[wid].metadata
    assert meta["kiss_reinforcement_count"] == 1
    assert meta["want_text"] == "tonic-triggered: Python -- open questions: Python→" + TURN2
    assert meta["emergent_open_question_ids"] == [[TREE1, FOREST2]]


def test_the_minted_want_surfaces_as_readable_text():
    """End to end with the P408 resolver: what reaches awareness has no ids."""
    from surface_resolver import resolve_surface_content
    g = _graph()
    _predict(g, (TREE1, FOREST2))
    res = cno.generate_emergent_want(g, None)
    shown = resolve_surface_content(g.nodes[res["id"]], None)
    assert shown == res["text"] and not ID_RE.search(shown)
