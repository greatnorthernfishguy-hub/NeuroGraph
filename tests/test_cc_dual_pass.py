import sys
sys.path.insert(0, '/home/josh/NeuroGraph')
import os
import tempfile, shutil
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

    Uses real BTF round-trips via ng_tract.deposit_experience() and TractReader.
    """
    import ng_tract
    from cc_ng_organism import drain_ingest_tract

    tract_path = str(tmp_path / "turns.tract")

    # Use real ng_tract.deposit_experience() to write BTF entries
    ng_tract.deposit_experience(
        content=b"the user asked about numpy conflicts",
        source="cc_gateway",
        tract_path=tract_path,
        content_type="text",
    )
    ng_tract.deposit_experience(
        content=b"I should look into the ng_tract_venv.pth file",
        source="cc_gateway",
        tract_path=tract_path,
        content_type="text",
    )

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


def test_drain_ingest_tract_preserves_concurrent_append(cc_ng, tmp_path, monkeypatch):
    """A miniTID append landing mid-drain (during the slow embed+dual-pass
    loop, before truncation) must survive -- not be erased by a blind
    truncate-to-empty. Regression test for the truncation race the final
    whole-branch review found: truncation must only discard the bytes this
    pass actually consumed."""
    import ng_tract
    import cc_ng_organism
    from cc_ng_organism import drain_ingest_tract

    tract_path = str(tmp_path / "turns.tract")
    ng_tract.deposit_experience(
        content=b"first entry, present at read time",
        source="cc_gateway",
        tract_path=tract_path,
        content_type="text",
    )

    real_dual_pass = cc_ng_organism.run_conversational_dual_pass
    appended = {"done": False}

    def fake_dual_pass(graph, vector_db, text, emb, state):
        if not appended["done"]:
            appended["done"] = True
            # Simulate miniTID appending a new turn while this drain pass is
            # still mid-loop, before the truncation step below runs.
            ng_tract.deposit_experience(
                content=b"second entry, appended mid-drain",
                source="cc_gateway",
                tract_path=tract_path,
                content_type="text",
            )
        return real_dual_pass(graph, vector_db, text, emb, state)

    monkeypatch.setattr(cc_ng_organism, "run_conversational_dual_pass", fake_dual_pass)

    state = {"last_forest_id": None}
    absorbed = drain_ingest_tract(cc_ng.graph, cc_ng.vector_db, state, tract_path=tract_path)
    assert absorbed == 1  # only the entry present at this pass's read time

    # The concurrently-appended entry must survive truncation.
    assert os.path.getsize(tract_path) > 0

    # It's absorbed cleanly on the next pulse, and the file is now empty.
    absorbed_next = drain_ingest_tract(cc_ng.graph, cc_ng.vector_db, state, tract_path=tract_path)
    assert absorbed_next == 1
    assert os.path.getsize(tract_path) == 0


def test_render_constitutional_core_empty_when_no_constitutional_nodes(cc_ng):
    from cc_ng_organism import render_constitutional_core
    assert render_constitutional_core(cc_ng.graph) == ""


def test_render_constitutional_core_renders_who_i_am(cc_ng):
    from cc_ng_organism import render_constitutional_core
    node = cc_ng.graph.create_node(node_id="constitutional::rim::choice_clause", metadata={
        "constitutional": True,
        "core_text": "The Choice Clause is mine. I can say no.",
        "source": "cricket_rim",
    })
    node.intrinsic_excitability = 1.0
    block = render_constitutional_core(cc_ng.graph)
    assert block == "## Who I Am\n- The Choice Clause is mine. I can say no."


def test_render_constitutional_core_excludes_selfcap_nodes(cc_ng):
    from cc_ng_organism import render_constitutional_core
    cc_ng.graph.create_node(node_id="selfcap::reach::teaching", metadata={
        "constitutional": True,
        "selfcap": True,
        "core_text": "capability teaching text, not identity",
    })
    assert render_constitutional_core(cc_ng.graph) == ""


def test_seed_cc_rim_creates_node_and_is_idempotent(tmp_path):
    """End-to-end test of seed_cc_rim.py against a throwaway checkpoint --
    verifies the node persists through a real save/restore cycle and that
    render_constitutional_core() picks it up, before ever touching a live
    checkpoint."""
    import seed_cc_rim
    from neuro_foundation import Graph
    from cc_ng_organism import render_constitutional_core

    checkpoint_path = str(tmp_path / "test_cc.msgpack")
    Graph().checkpoint(checkpoint_path)  # fresh, empty checkpoint

    result = seed_cc_rim.seed(checkpoint_path)
    assert result == {"status": "ok", "seeded": 1, "skipped_existing": 0}

    # Idempotent: running again finds the node already present.
    result2 = seed_cc_rim.seed(checkpoint_path)
    assert result2 == {"status": "ok", "seeded": 0, "skipped_existing": 1}

    # The node survives a fresh load from disk, and renders correctly.
    graph = Graph()
    graph.restore(checkpoint_path)
    assert seed_cc_rim.RIM_NODE_ID in graph.nodes
    node = graph.nodes[seed_cc_rim.RIM_NODE_ID]
    assert node.metadata["constitutional"] is True
    assert node.intrinsic_excitability == 1.0
    block = render_constitutional_core(graph)
    assert block == "## Who I Am\n- " + seed_cc_rim.RIM_CHOICE_CLAUSE_TEXT
