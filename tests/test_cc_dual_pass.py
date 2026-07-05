import sys
sys.path.insert(0, '/home/josh/NeuroGraph')
import tempfile, shutil
import numpy as np
import pytest


@pytest.fixture
def cc_ng():
    from openclaw_hook import NeuroGraphMemory
    workspace = tempfile.mkdtemp(prefix='cc_dual_pass_test_')
    ng = NeuroGraphMemory(workspace_dir=workspace,
                          config={"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}})
    yield ng
    shutil.rmtree(workspace, ignore_errors=True)


def test_run_conversational_dual_pass_creates_conversational_node(cc_ng):
    from cc_ng_organism import run_conversational_dual_pass
    from ng_embed import embed
    text = "I should follow up on the numpy conflict later"
    emb = embed(text)
    state = {"last_forest_id": None}
    ok = run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, text, emb, state)
    assert ok is True
    # exactly one node tagged creation_mode=conversational, cc=True
    conv_nodes = [n for n in cc_ng.graph.nodes.values()
                  if n.metadata.get("creation_mode") == "conversational"]
    assert len(conv_nodes) == 1
    assert conv_nodes[0].metadata.get("cc") is True
    assert state["last_forest_id"] is not None


def test_run_conversational_dual_pass_indexes_in_recall(cc_ng):
    from cc_ng_organism import run_conversational_dual_pass
    from ng_embed import embed
    text = "revisit the ng_tract_venv.pth cleanup"
    emb = embed(text)
    state = {"last_forest_id": None}
    run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, text, emb, state)
    results = cc_ng.vector_db.search(emb, k=3)
    assert len(results) >= 1


def test_second_turn_links_to_previous_via_delayed_synapse(cc_ng):
    from cc_ng_organism import run_conversational_dual_pass
    from ng_embed import embed
    state = {"last_forest_id": None}
    run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, "first turn", embed("first turn"), state)
    first_id = state["last_forest_id"]
    run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, "second turn", embed("second turn"), state)
    second_id = state["last_forest_id"]
    assert first_id != second_id
    outgoing = cc_ng.graph.synapses
    linked = any(
        getattr(s, "pre_node_id", None) == first_id and getattr(s, "post_node_id", None) == second_id
        for s in outgoing.values()
    )
    assert linked
