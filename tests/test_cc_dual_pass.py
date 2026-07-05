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


def test_probation_graduation(cc_ng):
    from cc_ng_organism import run_conversational_dual_pass, cc_update_probation, _CC_CONV_PROBATION_PERIOD
    from ng_embed import embed
    # Deposit a conversational node with probation
    state = {"last_forest_id": None}
    text = "probation test node"
    ok = run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, text, embed(text), state)
    assert ok is True
    node_id = state["last_forest_id"]
    assert node_id is not None

    # Node should initially be in probation
    node = cc_ng.graph.nodes[node_id]
    assert node.metadata.get("probation_remaining") == _CC_CONV_PROBATION_PERIOD
    assert node.metadata.get("graduated") is not True
    initial_excitability = node.intrinsic_excitability
    assert initial_excitability < 1.0  # should be dampened

    # Call cc_update_probation repeatedly until graduation
    for i in range(_CC_CONV_PROBATION_PERIOD):
        graduated = cc_update_probation(cc_ng.graph)
        if i < _CC_CONV_PROBATION_PERIOD - 1:
            # Should not graduate yet
            assert node.metadata.get("probation_remaining") == _CC_CONV_PROBATION_PERIOD - i - 1
            assert node.metadata.get("graduated") is not True
        else:
            # Final call should graduate
            assert node_id in graduated
            assert node.metadata.get("graduated") is True
            assert node.metadata.get("probation_remaining") <= 0

    # After graduation, node should be at full excitability
    assert node.intrinsic_excitability == 1.0
    base_threshold = cc_ng.graph.config.get("default_threshold", 1.0)
    assert node.threshold == base_threshold


def test_drain_ingest_tract_absorbs_experience_entries(cc_ng, tmp_path):
    """Test that drain_ingest_tract reads and absorbs experience entries from a tract file.

    Note: ng_tract v0.1.0 doesn't provide deposit_experience() yet (miniTID Tasks 4-7).
    We mock the TractReader to return experience entries for testing.
    """
    from cc_ng_organism import drain_ingest_tract
    from unittest.mock import Mock, patch, MagicMock
    import sys

    tract_path = str(tmp_path / "turns.tract")

    # Create mock experience entries
    ENTRY_EXPERIENCE = 0

    class MockExperienceEntry:
        def __init__(self, content, source):
            self.entry_type = ENTRY_EXPERIENCE
            self.source = source
            self.content_type = "text"
            self.content = content

    # Write a placeholder file so drain_ingest_tract knows the file exists
    with open(tract_path, 'wb') as f:
        f.write(b"placeholder")

    # Mock TractReader to return our experience entries
    mock_entries = [
        MockExperienceEntry("the user asked about numpy conflicts", "cc_gateway"),
        MockExperienceEntry("I should look into the ng_tract_venv.pth file", "cc_gateway"),
    ]

    # Patch ng_tract in the context where it's imported (in drain_ingest_tract)
    with patch('ng_tract.TractReader') as mock_reader_class:
        mock_reader_class.return_value = iter(mock_entries)
        state = {"last_forest_id": None}
        absorbed = drain_ingest_tract(cc_ng.graph, cc_ng.vector_db, state, tract_path=tract_path)
        assert absorbed == 2

    conv_nodes = [n for n in cc_ng.graph.nodes.values()
                  if n.metadata.get("creation_mode") == "conversational"]
    assert len(conv_nodes) == 2


def test_drain_ingest_tract_is_idempotent_on_empty_file(cc_ng, tmp_path):
    from cc_ng_organism import drain_ingest_tract
    tract_path = str(tmp_path / "nonexistent.tract")
    state = {"last_forest_id": None}
    absorbed = drain_ingest_tract(cc_ng.graph, cc_ng.vector_db, state, tract_path=tract_path)
    assert absorbed == 0
