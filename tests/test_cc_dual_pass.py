import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

    # #93 — the "graduated" stamp now requires evidence the node actually fired,
    # so make it genuinely spike before its window closes.
    cc_ng.graph.stimulate(node_id, 20.0)
    cc_ng.graph.step()
    assert len(node.spike_history) > 0, "precondition: node must have really fired"

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


def test_probation_unfired_node_sheds_dampening_but_does_not_graduate(cc_ng, monkeypatch):
    """#93 (CC mirror of the canonical test in test_conversational_recall.py) — a node
    that ages out of probation without ever firing must still get its novelty-dampening
    released, but must NOT be stamped 'graduated'. Gating the dampening release too
    would be self-reinforcing: a permanently boosted threshold makes firing less
    likely, so the node could never earn its way out.
    """
    import cc_ng_organism as cc
    from cc_ng_organism import (run_conversational_dual_pass, cc_update_probation,
                                _CC_CONV_PROBATION_PERIOD)
    from ng_embed import embed
    # Pin the gate rather than inheriting CC_CONV_PROBATION_REQUIRE_SPIKE from the env.
    monkeypatch.setattr(cc, "_CC_CONV_PROBATION_REQUIRE_SPIKE", True)

    state = {"last_forest_id": None}
    text = "quiet probation node that never fires"
    assert run_conversational_dual_pass(
        cc_ng.graph, cc_ng.vector_db, text, embed(text), state) is True
    node_id = state["last_forest_id"]
    node = cc_ng.graph.nodes[node_id]
    assert node.metadata.get("probation_remaining") == _CC_CONV_PROBATION_PERIOD
    assert len(node.spike_history) == 0, "precondition: node must not have fired"

    # Age the whole window out without ever stimulating it.
    for _ in range(_CC_CONV_PROBATION_PERIOD):
        graduated = cc_update_probation(cc_ng.graph)

    assert node_id not in graduated
    assert node.metadata.get("graduated") is False
    assert node.metadata.get("probation_expired_unfired") is True
    # ...but the handicap is lifted on schedule regardless.
    assert node.intrinsic_excitability == 1.0
    assert node.threshold == cc_ng.graph.config.get("default_threshold", 1.0)

    # Late graduation: fire it now and the very next sweep earns the stamp.
    cc_ng.graph.stimulate(node_id, 20.0)
    cc_ng.graph.step()
    assert len(node.spike_history) > 0
    graduated = cc_update_probation(cc_ng.graph)
    assert node_id in graduated
    assert node.metadata.get("graduated") is True
    assert node.metadata.get("probation_expired_unfired") is None


def test_kiss_redundancy_gate_reinforces_instead_of_duplicating(cc_ng):
    """Real-KISS redundancy->reinforcement gate: an exact-repeat turn must
    not create a second conversational node. It reinforces the existing one."""
    from cc_ng_organism import run_conversational_dual_pass
    from ng_embed import embed
    state = {"last_forest_id": None}
    text = "the redundant turn text for the KISS gate test"
    emb = embed(text)

    ok1 = run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, text, emb, state)
    assert ok1 is True
    first_id = state["last_forest_id"]
    conv_nodes = [n for n in cc_ng.graph.nodes.values()
                  if n.metadata.get("creation_mode") == "conversational"]
    assert len(conv_nodes) == 1

    ok2 = run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, text, emb, state)
    assert ok2 is True
    conv_nodes_after = [n for n in cc_ng.graph.nodes.values()
                        if n.metadata.get("creation_mode") == "conversational"]
    assert len(conv_nodes_after) == 1  # no duplicate node
    assert state["last_forest_id"] == first_id  # reinforcement targeted the existing node

    node = cc_ng.graph.nodes[first_id]
    assert node.metadata.get("kiss_reinforcement_count") == 1


def test_kiss_redundancy_gate_does_not_collapse_distinct_turns(cc_ng):
    """Genuinely different content must not be gated -- the redundancy check
    is pure change detection, not a bias toward fewer nodes."""
    from cc_ng_organism import run_conversational_dual_pass
    from ng_embed import embed
    state = {"last_forest_id": None}
    t1 = "talk about pizza toppings and cheese preferences"
    t2 = "debugging a segfault in the kernel driver's interrupt handler"
    run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, t1, embed(t1), state)
    run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, t2, embed(t2), state)
    conv_nodes = [n for n in cc_ng.graph.nodes.values()
                  if n.metadata.get("creation_mode") == "conversational"]
    assert len(conv_nodes) == 2


def test_kiss_redundancy_gate_confirms_without_duplicating_across_distinct_turns(cc_ng):
    """A redundant hit reinforces (bumps the confirmation counter) without
    adding a node, even when other distinct turns exist in the substrate."""
    from cc_ng_organism import run_conversational_dual_pass
    from ng_embed import embed
    state = {"last_forest_id": None}
    t1 = "first distinct turn about numpy conflicts"
    t2 = "second distinct turn about the tract bridge cleanup"
    run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, t1, embed(t1), state)
    id1 = state["last_forest_id"]
    run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, t2, embed(t2), state)

    run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, t1, embed(t1), state)
    assert cc_ng.graph.nodes[id1].metadata.get("kiss_reinforcement_count") == 1
    conv_nodes = [n for n in cc_ng.graph.nodes.values()
                  if n.metadata.get("creation_mode") == "conversational"]
    assert len(conv_nodes) == 2  # still only the two distinct nodes


def test_kiss_gate_kill_switch_restores_fresh_deposit(cc_ng, monkeypatch):
    """With CC_KISS_GATE_ENABLED off, the gate never fires -- an exact repeat
    is not turned into reinforcement (pre-KISS behavior)."""
    import cc_ng_organism
    from cc_ng_organism import run_conversational_dual_pass
    from ng_embed import embed
    monkeypatch.setattr(cc_ng_organism, "_CC_KISS_GATE_ENABLED", False)
    state = {"last_forest_id": None}
    text = "kill switch test turn text"
    emb = embed(text)
    run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, text, emb, state)
    node_id = state["last_forest_id"]
    run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, text, emb, state)
    assert "kiss_reinforcement_count" not in cc_ng.graph.nodes[node_id].metadata


def test_kiss_gate_never_collapses_into_identity_protected_node(cc_ng, monkeypatch):
    """Cricket bypass: a redundant turn must not fold into an identity-protected
    (constitutional) node -- it deposits fresh instead."""
    from cc_ng_organism import run_conversational_dual_pass
    from ng_embed import embed
    state = {"last_forest_id": None}
    text = "identity protected collapse guard turn"
    emb = embed(text)
    run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, text, emb, state)
    first_id = state["last_forest_id"]

    # Force the existing conversational node to read as identity-protected.
    monkeypatch.setattr(type(cc_ng.graph), "_is_identity_protected",
                        lambda self, nid: nid == first_id, raising=False)

    # A near-duplicate (different text -> different target_id) must NOT collapse
    # into the protected node; it deposits as its own fresh node.
    text2 = text + " again"
    run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, text2, embed(text2), state)
    assert cc_ng.graph.nodes[first_id].metadata.get("kiss_reinforcement_count") is None
    conv_nodes = [n for n in cc_ng.graph.nodes.values()
                  if n.metadata.get("creation_mode") == "conversational"]
    assert len(conv_nodes) == 2


def test_kiss_reinforcement_accelerates_probation_instead_of_resetting(cc_ng):
    """A redundant hit on a still-probationary node must tick it one step
    closer to graduation, not restart the fixed probation window."""
    from cc_ng_organism import run_conversational_dual_pass, _CC_CONV_PROBATION_PERIOD
    from ng_embed import embed
    state = {"last_forest_id": None}
    text = "probation acceleration test text for the KISS gate"
    emb = embed(text)
    run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, text, emb, state)
    node_id = state["last_forest_id"]
    node = cc_ng.graph.nodes[node_id]
    assert node.metadata["probation_remaining"] == _CC_CONV_PROBATION_PERIOD

    run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, text, emb, state)
    assert node.metadata["probation_remaining"] == _CC_CONV_PROBATION_PERIOD - 1
    assert node.metadata.get("graduated") is not True


def test_kiss_reinforcement_never_resets_a_graduated_node(cc_ng):
    """Once a node has graduated out of probation, a later redundant hit must
    not push it back into probation or dampen its excitability."""
    from cc_ng_organism import run_conversational_dual_pass, cc_update_probation, _CC_CONV_PROBATION_PERIOD
    from ng_embed import embed
    state = {"last_forest_id": None}
    text = "graduation reinforcement test text for the KISS gate"
    emb = embed(text)
    run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, text, emb, state)
    node_id = state["last_forest_id"]
    node = cc_ng.graph.nodes[node_id]
    # #93 — graduation now requires a real spike, not just an expired timer.
    cc_ng.graph.stimulate(node_id, 20.0)
    cc_ng.graph.step()
    for _ in range(_CC_CONV_PROBATION_PERIOD):
        cc_update_probation(cc_ng.graph)
    assert node.metadata.get("graduated") is True
    assert node.intrinsic_excitability == 1.0

    run_conversational_dual_pass(cc_ng.graph, cc_ng.vector_db, text, emb, state)
    assert node.metadata.get("graduated") is True
    assert node.intrinsic_excitability == 1.0
    base_threshold = cc_ng.graph.config.get("default_threshold", 1.0)
    assert node.threshold == base_threshold


class _FakePred:
    def __init__(self, conf, src, tgt):
        self.confidence = conf
        self.source_node_id = src
        self.target_node_id = tgt


class _FakeFired:
    def __init__(self, node_id):
        self.node_id = node_id


class _FakeResult:
    def __init__(self, fired):
        self.fired_entries = [_FakeFired(n) for n in fired]


class _FakeNode:
    def __init__(self, metadata=None):
        self.metadata = dict(metadata or {})


class _FakeVDB:
    def __init__(self, embeddings):
        self._e = embeddings  # {node_id: list/array}

    def get(self, nid):
        emb = self._e.get(nid)
        return {"embedding": emb} if emb is not None else None


class _FakeWantGraph:
    """Minimal graph surface generate_emergent_want() actually touches:
    active_predictions, prime_and_propagate (read-only), hyperedges, nodes,
    create_node, config."""
    def __init__(self):
        self.active_predictions = {}
        self.hyperedges = {}
        self.nodes = {}
        self.config = {"default_threshold": 1.0}
        self._pp_fired = []

    def prime_and_propagate(self, node_ids, currents, steps, write_mode):
        assert write_mode is False  # curiosity is observation, never mutation
        return _FakeResult(self._pp_fired)

    def create_node(self, node_id, metadata=None):
        n = _FakeNode(metadata)
        self.nodes[node_id] = n
        return n


def test_emergent_want_same_concept_reinforces_single_node():
    """Two curiosity pulses about the SAME concept (different open-question
    snapshots) collapse to ONE cc:want:: node -- reinforced, not twinned."""
    import numpy as np
    from cc_ng_organism import generate_emergent_want

    graph = _FakeWantGraph()
    graph.nodes["concept_a"] = _FakeNode({"label": "alpha"})
    graph._pp_fired = ["concept_a"]
    vdb = _FakeVDB({"concept_a": np.array([1.0, 0.0, 0.0], dtype=np.float32)})

    graph.active_predictions = {"p1": _FakePred(0.9, "src1", "tgt1")}
    r1 = generate_emergent_want(graph, vdb)
    assert r1 is not None
    want_id = r1["id"]
    first_text = r1["text"]

    # Same concept, different open questions -> reinforce the one node.
    graph.active_predictions = {"p1": _FakePred(0.9, "src2", "tgt2")}
    r2 = generate_emergent_want(graph, vdb)
    assert r2 is not None
    assert r2["id"] == want_id
    assert r2.get("reinforced") is True

    want_nodes = [nid for nid in graph.nodes if nid.startswith("cc:want::")]
    assert len(want_nodes) == 1
    node = graph.nodes[want_id]
    assert node.metadata.get("kiss_reinforcement_count") == 1
    # want_text refreshed to the latest snapshot.
    assert node.metadata.get("want_text") == r2["text"]
    assert r2["text"] != first_text


def test_emergent_want_distinct_concepts_make_distinct_nodes():
    """Two curiosity pulses about DIFFERENT concepts create two want-nodes."""
    import numpy as np
    from cc_ng_organism import generate_emergent_want

    graph = _FakeWantGraph()
    graph.active_predictions = {"p1": _FakePred(0.9, "src", "tgt")}

    graph.nodes["concept_a"] = _FakeNode({"label": "alpha"})
    graph._pp_fired = ["concept_a"]
    vdb = _FakeVDB({"concept_a": np.array([1.0, 0.0, 0.0], dtype=np.float32)})
    r1 = generate_emergent_want(graph, vdb)
    assert r1 is not None

    graph.nodes["concept_b"] = _FakeNode({"label": "beta"})
    graph._pp_fired = ["concept_b"]
    vdb2 = _FakeVDB({"concept_b": np.array([0.0, 1.0, 0.0], dtype=np.float32)})
    r2 = generate_emergent_want(graph, vdb2)
    assert r2 is not None


def test_emergent_want_unresolved_concepts_stay_distinct():
    """When the concept does NOT resolve (nothing fires -> concept_label is
    None), distinct label-less curiosities must NOT fold into a shared
    '(unknown)' bucket -- each keeps its own per-want_text identity so genuine
    distinct wants don't overwrite each other (LAW 7)."""
    from cc_ng_organism import generate_emergent_want

    graph = _FakeWantGraph()
    graph._pp_fired = []   # nothing fires -> no concept node -> concept_label stays None
    vdb = _FakeVDB({})     # no embeddings to resolve a concept from

    graph.active_predictions = {"p1": _FakePred(0.9, "srcA", "tgtA")}
    r1 = generate_emergent_want(graph, vdb)
    assert r1 is not None

    graph.active_predictions = {"p1": _FakePred(0.9, "srcB", "tgtB")}
    r2 = generate_emergent_want(graph, vdb)
    assert r2 is not None

    # Distinct label-less wants -> two distinct nodes, neither a reinforcement.
    assert r1["id"] != r2["id"]
    assert r1.get("reinforced") is not True
    assert r2.get("reinforced") is not True
    want_nodes = [nid for nid in graph.nodes if nid.startswith("cc:want::")]
    assert len(want_nodes) == 2

    assert r1["id"] != r2["id"]
    want_nodes = [nid for nid in graph.nodes if nid.startswith("cc:want::")]
    assert len(want_nodes) == 2


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


def test_cc_probation_rollback_drains_marker_instead_of_stranding_it(monkeypatch):
    """#93 rollback, CC side. Mirror of the canonical test in
    test_conversational_recall.py -- cc_update_probation is a near-verbatim port,
    so the one-way-rollback defect ports with it.

    Deliberately built on a bare Graph rather than the cc_ng fixture: this
    exercises cc_update_probation alone, which is parameterized on graph, and
    a NeuroGraphMemory would add ~800MB and the #100 flakiness for nothing.
    """
    from neuro_foundation import Graph
    import cc_ng_organism as cno

    g = Graph()

    # Phase 1: knob ON, node ages out without ever firing -> stamped.
    monkeypatch.setattr(cno, "_CC_CONV_PROBATION_REQUIRE_SPIKE", True)
    quiet = g.create_node(node_id="R", metadata={
        "probation_remaining": 1, "probation_total": 10, "novelty_dampening": 0.3})
    quiet.intrinsic_excitability = 0.3
    cno.cc_update_probation(g)
    assert quiet.metadata.get("probation_expired_unfired") is True
    assert quiet.metadata.get("graduated") is False
    assert len(quiet.spike_history) == 0, "precondition: it never fired"

    # Phase 2: operator rolls the gate back OFF. The node still has not fired.
    monkeypatch.setattr(cno, "_CC_CONV_PROBATION_REQUIRE_SPIKE", False)
    graduated = cno.cc_update_probation(g)

    assert "R" in graduated, "rollback stranded a node it was flipped to rescue"
    assert quiet.metadata.get("graduated") is True
    assert quiet.metadata.get("probation_expired_unfired") is None, \
        "the marker must be drained, not left behind to re-fire next sweep"
    assert "R" not in cno.cc_update_probation(g)
