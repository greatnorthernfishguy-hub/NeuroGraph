# ---- Changelog ----
# [2026-11-28] DudeMan CC (Claude Fable 5.2, laptop; quest SQ11, punch #1254c) -- CREATE
# What: pins CC_RECALL_FOLD_WINDOWS end to end on the real NeuroGraphMemory with the native stores: the enabler
#       (off / no graph / a store without the setting raises / on); with the setting off the harvest drops a fired
#       window whose turn did not fire (today's behaviour); with it on and evoked the harvest returns that TURN with
#       the turn's own text; without evoked nothing changes.
# Why:  a window node has no vector row by design (#295); the fold is in ng_tract.recall_sweep, no Python on the path.
# How:  one turn with a vector row and a high threshold (never fires), its window without a vector row, a cue node
#       wired to the window. Skipped on a wheel without NodeStore.recall_fold_windows or without the native recall.
# -------------------
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import tempfile

import pytest

import cc_ng_organism as org

CUE = "the lenia distance cache rebuild finished"
TURN = "turn-1"
WIN = TURN + "::window::0"


@pytest.fixture
def ng():
    from openclaw_hook import NeuroGraphMemory
    from ng_embed import embed
    ws = tempfile.mkdtemp(prefix='fold_windows_test_')
    m = NeuroGraphMemory(workspace_dir=ws, config={"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}})
    if not hasattr(type(m.graph.nodes), "recall_fold_windows"):
        shutil.rmtree(ws, ignore_errors=True)
        pytest.skip("node store without recall_fold_windows")
    g = m.graph
    g.create_node(node_id="cue", metadata={"_forest_content": CUE})
    g.nodes["cue"].threshold = 0.1
    m.vector_db.insert(id="cue", embedding=embed(CUE), content=CUE)
    g.create_node(node_id=TURN, metadata={"_forest_content": "what the turn said"})
    g.nodes[TURN].threshold = 50.0                                   # the turn itself never fires
    m.vector_db.insert(id=TURN, embedding=embed("zebra crossing paint"), content="what the turn said")
    g.create_node(node_id=WIN, metadata={})                          # no vector row: a window
    g.nodes[WIN].threshold = 0.1
    g.create_synapse("cue", WIN, weight=5.0)
    yield m
    shutil.rmtree(ws, ignore_errors=True)


def _harvest(ng, **kw):
    return [r["node_id"] for r in ng._harvest_associations(CUE, **kw)]


def test_enabler(ng, monkeypatch):
    monkeypatch.setattr(org, "RECALL_FOLD_WINDOWS", False)
    assert org.cc_enable_recall_fold_windows(ng.graph) is False and ng.graph.nodes.recall_fold_windows is False
    monkeypatch.setattr(org, "RECALL_FOLD_WINDOWS", True)
    assert org.cc_enable_recall_fold_windows(None) is False

    class G:
        nodes = {}
    with pytest.raises(RuntimeError):
        org.cc_enable_recall_fold_windows(G())
    assert org.cc_enable_recall_fold_windows(ng.graph) is True and ng.graph.nodes.recall_fold_windows is True


def test_off_the_fired_window_is_dropped_and_on_its_turn_is_returned(ng):
    fired = {e.node_id for e in ng.graph.prime_and_propagate(["cue"], [1.0], steps=3).fired_entries}
    assert WIN in fired and TURN not in fired
    if not getattr(ng.graph.prime_and_propagate(["cue"], [1.0], steps=3, evoked=True), "evoked", False):
        pytest.skip("the sweep did not run natively with evoked (no native recall in this configuration)")
    assert TURN not in _harvest(ng, evoked_override=True) and WIN not in _harvest(ng, evoked_override=True)
    ng.graph.nodes.recall_fold_windows = True
    out = ng._harvest_associations(CUE, evoked_override=True)
    row = next(r for r in out if r["node_id"] == TURN)
    assert row["content"] == "what the turn said" and row.get("evoked") is True
    assert WIN not in [r["node_id"] for r in out]
    # a sweep that is not evoked is not folded: every other reader sees what fired
    assert TURN not in _harvest(ng)
    assert WIN in {e.node_id for e in ng.graph.prime_and_propagate(["cue"], [1.0], steps=3).fired_entries}
