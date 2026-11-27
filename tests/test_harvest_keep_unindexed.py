# ---- Changelog ----
# [2026-11-27] DudeMan CC (Claude Fable 5.2, laptop; quest SQ11, punch #1254c) -- CREATE
# What: pins _harvest_associations(keep_unindexed_override=): by default a fired node the vector DB has no row for
#       is dropped (as before); with True it is kept with empty content, empty metadata and "unindexed": True; an
#       indexed node's item is the same either way; and the CC organism passes the keyword only when
#       CC_RECALL_KEEP_UNINDEXED is on, and then recall returns the node's OWN text.
# Why:  the vector DB only seeds; what surfaces is what fires. Windows, small tree concepts and wants have no row.
# How:  the real NeuroGraphMemory (as test_harvest_orphan_seeds does): a seed with a vector row, and a neighbour with
#       NO vector row joined to it by one strong synapse, both with a low threshold so the sweep fires them.
# -------------------
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import tempfile

import pytest

QUERY = "the lenia distance cache rebuild finished"
WINDOW_TEXT = "the second half of a long turn about the distance cache"


@pytest.fixture
def cc_ng():
    from openclaw_hook import NeuroGraphMemory
    from ng_embed import embed
    workspace = tempfile.mkdtemp(prefix='harvest_unindexed_test_')
    ng = NeuroGraphMemory(workspace_dir=workspace,
                          config={"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}})
    ng.graph.create_node(node_id="seed")
    ng.graph.nodes["seed"].threshold = 0.1
    ng.vector_db.insert(id="seed", embedding=embed("lenia cache rebuild status"),
                        content="lenia cache rebuild status")
    # The neighbour: a graph node with its own text and NO vector row (a window node's shape).
    ng.graph.create_node(node_id="window", metadata={"_forest_content": WINDOW_TEXT, "_window": True})
    ng.graph.nodes["window"].threshold = 0.1
    ng.graph.create_synapse("seed", "window", weight=5.0)
    assert ng.vector_db.get("window") is None
    yield ng
    shutil.rmtree(workspace, ignore_errors=True)


def _by_id(items):
    return {i["node_id"]: i for i in items}


def test_default_drops_a_fired_node_without_a_vector_row(cc_ng):
    got = _by_id(cc_ng._harvest_associations(QUERY))
    assert "seed" in got and "window" not in got
    assert "unindexed" not in got["seed"]


def test_keyword_keeps_it_with_empty_content_and_a_mark(cc_ng):
    got = _by_id(cc_ng._harvest_associations(QUERY, keep_unindexed_override=True))
    assert "seed" in got and "window" in got, sorted(got)
    w = got["window"]
    assert w["content"] == "" and w["metadata"] == {} and w["unindexed"] is True
    assert set(w) == {"node_id", "content", "metadata", "latency", "strength", "was_predicted", "unindexed"}


def test_an_indexed_node_is_the_same_item_either_way(cc_ng):
    off = _by_id(cc_ng._harvest_associations(QUERY))["seed"]
    on = _by_id(cc_ng._harvest_associations(QUERY, keep_unindexed_override=True))["seed"]
    assert on == off and on["content"] == "lenia cache rebuild status"


def test_false_and_none_are_the_default(cc_ng):
    base = cc_ng._harvest_associations(QUERY)
    assert cc_ng._harvest_associations(QUERY, keep_unindexed_override=False) == base
    assert cc_ng._harvest_associations(QUERY, keep_unindexed_override=None) == base


def test_the_cut_still_applies(cc_ng):
    got = cc_ng._harvest_associations(QUERY, keep_unindexed_override=True, max_surfaced_override=1)
    assert len(got) <= 5          # the harvest's own floor (max(5, ...)) is unchanged


def test_organism_passes_the_keyword_only_when_the_switch_is_on(monkeypatch):
    import cc_ng_organism as org
    monkeypatch.setattr(org, "_CC_RECALL_KEEP_UNINDEXED", False)
    assert org._cc_unindexed_kwargs() == {}
    monkeypatch.setattr(org, "_CC_RECALL_KEEP_UNINDEXED", True)
    assert org._cc_unindexed_kwargs() == {"keep_unindexed_override": True}


@pytest.mark.parametrize("on", [False, True])
def test_recall_returns_the_nodes_own_text_when_on(cc_ng, monkeypatch, on):
    import cc_ng_organism as org
    monkeypatch.setattr(org, "_CC_RECALL_KEEP_UNINDEXED", on)
    out = org.cc_pattern_completion_recall(cc_ng, QUERY, k=5)
    texts = {r["node_id"]: r["content"] for r in out}
    assert "seed" in texts
    if on:
        assert texts.get("window") == WINDOW_TEXT
    else:
        assert "window" not in texts
