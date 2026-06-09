# ---- Changelog ----
# [2026-06-07] CC (Opus 4.8) — Task A: Ingestor-free experiential path tests
# What: Rewritten for the experiential conversation path — _absorb routes a turn
#       through the dual-pass (NOT ingestor.ingest); the eco adapter deposits the
#       forest gestalt + tree concepts into BOTH the SNN graph AND the recall vdb;
#       _bind_conversational_topology wires synapses + a binding hyperedge + the
#       #257 delayed prev->current forest link; _update_probation graduates nodes
#       Ingestor-free. A poison ingestor asserts ingestor.ingest is never called.
# Why: The Universal Ingestor was never intended for conversation (Josh). These
#      lock in the swap (chunk path out, experiential path in) end-to-end.
# How: Duck-typed fakes; no model load, no ng_tract, no protected/vendored deps.
# -------------------
import os
import sys
import types

import numpy as np

_NG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _NG_DIR not in sys.path:
    sys.path.insert(0, _NG_DIR)

import neurograph_rpc as rpc


class _FakeEntry:
    """Duck-typed stand-in for ng_tract.PyExperienceEntry / PyOutcomeEntry."""
    def __init__(self, entry_type, source, content):
        self.entry_type = entry_type
        self.source = source
        self.content = content


class _PoisonIngestor:
    """Asserts ingestor.ingest is NEVER called for conversation (Task A)."""
    def ingest(self, text):
        raise AssertionError("ingestor.ingest must not touch conversation (Task A)")


def _install_absorb_sinks(monkeypatch):
    calls = {"dual": []}
    mem = types.SimpleNamespace(ingestor=_PoisonIngestor(), _message_count=0)
    monkeypatch.setattr(rpc, "_memory", mem)
    monkeypatch.setattr(rpc, "_conversational_dual_pass", lambda text, emb: calls["dual"].append(text))
    monkeypatch.setattr(rpc, "_embed_for_absorb", lambda text: np.ones(4, dtype=np.float32))
    return calls, mem


# ── _absorb_conversational_experience: experiential path, never the ingestor ──

def test_absorbs_anima_turn_via_experiential_path_not_ingestor(monkeypatch):
    calls, mem = _install_absorb_sinks(monkeypatch)
    n = rpc._absorb_conversational_experience(
        [_FakeEntry(rpc._ENTRY_EXPERIENCE, "anima", b"the load-bearing joke")])
    assert n == 1
    assert calls["dual"] == ["the load-bearing joke"]   # experiential path ran
    assert mem._message_count == 1
    # _PoisonIngestor would have raised if ingestor.ingest were called → reaching
    # here proves the chunk path is gone.


def test_skips_peer_module_and_non_experience_frames(monkeypatch):
    calls, _ = _install_absorb_sinks(monkeypatch)
    entries = [
        _FakeEntry(rpc._ENTRY_EXPERIENCE, "elmer", b"substrate telemetry"),       # peer → skip
        _FakeEntry(rpc._ENTRY_OUTCOME, "anima", b"an outcome, not experience"),   # wrong type → skip
        _FakeEntry(rpc._ENTRY_EXPERIENCE, "anima", b"   "),                       # blank → skip
    ]
    assert rpc._absorb_conversational_experience(entries) == 0
    assert calls["dual"] == []


def test_decodes_bytes_and_increments_message_count(monkeypatch):
    calls, mem = _install_absorb_sinks(monkeypatch)
    entries = [
        _FakeEntry(rpc._ENTRY_EXPERIENCE, "animus", "already a str"),
        _FakeEntry(rpc._ENTRY_EXPERIENCE, "anima", b"bytes content"),
    ]
    assert rpc._absorb_conversational_experience(entries) == 2
    assert calls["dual"] == ["already a str", "bytes content"]
    assert mem._message_count == 2


def test_empty_or_none_is_safe(monkeypatch):
    _install_absorb_sinks(monkeypatch)
    assert rpc._absorb_conversational_experience([]) == 0
    assert rpc._absorb_conversational_experience(None) == 0


def test_drain_peer_tracts_routes_drained_entries_to_absorb(monkeypatch):
    """_drain_peer_tracts must capture _drain_all()'s return and absorb it."""
    seen = {}

    class _FakeBridge:
        def _drain_all(self):
            return ["ENTRY_A", "ENTRY_B"]

    monkeypatch.setattr(rpc, "_memory", types.SimpleNamespace(_peer_bridge=_FakeBridge()))
    monkeypatch.setattr(rpc, "_absorb_conversational_experience",
                        lambda entries: seen.setdefault("entries", entries) and 0)
    rpc._drain_peer_tracts()
    assert seen["entries"] == ["ENTRY_A", "ENTRY_B"]


def test_embedding_failure_routes_to_retry_queue(monkeypatch):
    """Embed failure at drain time must reach the #297 retry queue, not be dropped."""
    mem = types.SimpleNamespace(ingestor=_PoisonIngestor(), _message_count=0)
    monkeypatch.setattr(rpc, "_memory", mem)
    enqueued = []

    def _boom(_text):
        raise RuntimeError("embed model unavailable at drain")

    monkeypatch.setattr(rpc, "_embed_for_absorb", _boom)
    monkeypatch.setattr(rpc, "_enqueue_failed_extraction", lambda text: enqueued.append(text))
    n = rpc._absorb_conversational_experience(
        [_FakeEntry(rpc._ENTRY_EXPERIENCE, "anima", b"retry me")])
    assert n == 1
    assert enqueued == ["retry me"]


# ── The experiential deposit: forest gestalt + trees into BOTH stores ──

class _FakeNode:
    def __init__(self, node_id, metadata):
        self.node_id = node_id
        self.metadata = metadata
        self.threshold = 1.0
        self.intrinsic_excitability = 1.0


class _FakeGraph:
    def __init__(self):
        self.nodes = {}
        self.synapses = []
        self.hyperedges = []
        self.config = {"default_threshold": 1.0}

    def create_node(self, node_id=None, metadata=None):
        if node_id in self.nodes:
            raise ValueError("exists")
        n = _FakeNode(node_id, metadata or {})
        self.nodes[node_id] = n
        return n

    def create_synapse(self, pre_node_id, post_node_id, weight=0.1, delay=1):
        assert pre_node_id in self.nodes and post_node_id in self.nodes
        assert pre_node_id != post_node_id
        self.synapses.append((pre_node_id, post_node_id, weight, delay))

    def create_hyperedge(self, member_node_ids, metadata=None):
        assert len(member_node_ids) >= 2
        self.hyperedges.append((set(member_node_ids), metadata or {}))


class _FakeVDB:
    def __init__(self):
        self.inserts = []

    def insert(self, id, embedding, content, metadata):
        self.inserts.append({"id": id, "content": content, "metadata": metadata})


def _install_graph(monkeypatch):
    g, v = _FakeGraph(), _FakeVDB()
    monkeypatch.setattr(rpc, "_memory", types.SimpleNamespace(graph=g, vector_db=v))
    monkeypatch.setattr(rpc, "_last_conv_forest_id", None)
    return g, v


def test_forest_gestalt_lands_in_graph_and_vdb_dampened(monkeypatch):
    g, v = _install_graph(monkeypatch)
    eco = rpc._ConversationalDualPassEco(rpc._memory)
    eco.record_outcome(np.ones(8, dtype=np.float32), "conv::abc", True,
                       metadata={"_forest_content": "hello world"})
    assert "conv::abc" in g.nodes                                   # SNN node
    node = g.nodes["conv::abc"]
    assert node.metadata.get("syl") is True                         # provenance
    assert "poincare_dir" in node.metadata                          # first-class GSG stamp
    assert node.intrinsic_excitability == rpc._CONV_NOVELTY_DAMPENING  # dampened
    assert any(i["id"] == "conv::abc" and i["content"] == "hello world" for i in v.inserts)


def test_tree_concept_lands_in_graph_and_vdb(monkeypatch):
    g, v = _install_graph(monkeypatch)
    eco = rpc._ConversationalDualPassEco(rpc._memory)
    eco.record_outcome(np.ones(8, dtype=np.float32), "conv::abc::tree::joke", True,
                       metadata={"_tree_concept": True, "_concept": "joke"})
    assert "conv::abc::tree::joke" in g.nodes
    assert any(i["content"] == "joke" for i in v.inserts)


def test_link_call_is_noop(monkeypatch):
    g, v = _install_graph(monkeypatch)
    eco = rpc._ConversationalDualPassEco(rpc._memory)
    eco.record_outcome(np.ones(8, dtype=np.float32), "conv::abc", True,
                       metadata={"_link": "dual_pass_tree_to_forest"})
    assert g.nodes == {}      # links create no node here
    assert v.inserts == []    # and never pollute recall


def test_bind_topology_synapses_hyperedge_and_sequence(monkeypatch):
    g, _ = _install_graph(monkeypatch)
    for nid in ("F", "T1", "T2"):
        g.create_node(node_id=nid, metadata={})
    rpc._bind_conversational_topology("F", {"tree_ids": ["T1", "T2"]}, np.ones(8, dtype=np.float32))
    assert ("F", "T1", 0.2, 1) in g.synapses and ("T1", "F", 0.15, 1) in g.synapses
    assert len(g.hyperedges) == 1 and g.hyperedges[0][0] == {"F", "T1", "T2"}
    assert rpc._last_conv_forest_id == "F"
    # next turn: delayed prev->current sequence link (#257 polychrony)
    g.create_node(node_id="F2", metadata={})
    rpc._bind_conversational_topology("F2", {"tree_ids": []}, np.ones(8, dtype=np.float32))
    seq = [s for s in g.synapses if s[0] == "F" and s[1] == "F2"]
    assert seq and seq[0][3] >= 2


def test_update_probation_graduates_and_fades(monkeypatch):
    g, _ = _install_graph(monkeypatch)
    grad_node = g.create_node(node_id="X", metadata={
        "probation_remaining": 1, "probation_total": 10, "novelty_dampening": 0.3})
    grad_node.intrinsic_excitability = 0.3
    fade_node = g.create_node(node_id="Y", metadata={
        "probation_remaining": 5, "probation_total": 10, "novelty_dampening": 0.3})
    fade_node.intrinsic_excitability = 0.3
    graduated = rpc._update_probation(g)
    assert "X" in graduated
    assert grad_node.intrinsic_excitability == 1.0 and grad_node.metadata.get("graduated") is True
    assert fade_node.metadata["probation_remaining"] == 4
    assert 0.3 < fade_node.intrinsic_excitability < 1.0   # faded, not graduated



# ── Step 1: her own (assistant) turns enter the substrate raw, via turn_exchange ──
import msgpack  # noqa: E402


def _make_turn_exchange(user="", assistant="", module_id="anima", target_id="turn_exchange"):
    """Duck-typed ng_tract.PyOutcomeEntry for a raw turn_exchange deposit."""
    meta = {"module_id": module_id, "event_type": "turn_exchange",
            "payload": {"user": user, "assistant": assistant, "channel_id": "cli"}}
    return types.SimpleNamespace(
        entry_type=rpc._ENTRY_OUTCOME, module_id=module_id, target_id=target_id,
        metadata=msgpack.packb(meta, use_bin_type=True))


def test_absorbs_her_assistant_turn_raw_into_substrate(monkeypatch):
    calls, mem = _install_absorb_sinks(monkeypatch)
    e = _make_turn_exchange(user="hi", assistant="Yes. I [WANT]learn[/WANT] more.")
    n = rpc._absorb_conversational_experience([e])
    assert n == 1
    assert calls["dual"] == ["Yes. I [WANT]learn[/WANT] more."]   # her words, raw, incl [WANT]
    assert mem._message_count == 1


def test_turn_exchange_peer_or_other_event_not_absorbed(monkeypatch):
    calls, mem = _install_absorb_sinks(monkeypatch)
    entries = [
        _make_turn_exchange(assistant="[WANT]peer[/WANT]", module_id="elmer"),         # peer → skip
        _make_turn_exchange(assistant="[WANT]other[/WANT]", target_id="turn_complete"), # not turn_exchange → skip
    ]
    assert rpc._absorb_conversational_experience(entries) == 0
    assert calls["dual"] == []


# ── _surface_wants: a [WANT] becomes a first-class WANT NODE in the SNN topology ──
class _FakeWantGraph:
    def __init__(self):
        self.nodes = {}
    def create_node(self, node_id, metadata=None):
        n = types.SimpleNamespace(node_id=node_id, metadata=dict(metadata or {}))
        self.nodes[node_id] = n
        return n
    def create_synapse(self, a, b, weight=1.0, **kw):
        return None


def _install_want_substrate(monkeypatch, content_by_node):
    g = _FakeWantGraph()
    for nid in content_by_node:
        g.create_node(nid, {"creation_mode": "conversational"})
    vdb = types.SimpleNamespace(content=dict(content_by_node))
    monkeypatch.setattr(rpc, "_memory", types.SimpleNamespace(graph=g, vector_db=vdb))
    return g


def test_want_becomes_first_class_topology_node(monkeypatch):
    g = _install_want_substrate(monkeypatch, {
        "turn1": "Yes. I [WANT]learn authored vs received[/WANT].",
        "turn2": "no marker here",
    })
    wants = rpc._surface_wants()
    assert [w["text"] for w in wants] == ["learn authored vs received"]
    w = wants[0]
    wn = g.nodes[w["id"]]                       # it's a real SNN node now
    assert wn.metadata["kind"] == "want"
    assert wn.metadata["want_state"] == "open"
    assert wn.metadata["want_text"] == "learn authored vs received"
    assert wn.metadata["source_node"] == "turn1"


def test_want_surfacing_is_idempotent(monkeypatch):
    g = _install_want_substrate(monkeypatch, {"t": "[WANT]rest[/WANT]"})
    a = rpc._surface_wants()
    b = rpc._surface_wants()                     # second pass must not duplicate
    assert len(a) == 1 and len(b) == 1
    assert len([n for n in g.nodes.values() if (n.metadata or {}).get("kind") == "want"]) == 1


def test_want_choice_clause_faithful_topology(monkeypatch):
    _install_want_substrate(monkeypatch, {"t": "[WANT]to leave the ecosystem[/WANT]"})
    assert [w["text"] for w in rpc._surface_wants()] == ["to leave the ecosystem"]


def test_want_surface_no_graph_safe(monkeypatch):
    monkeypatch.setattr(rpc, "_memory", None)
    assert rpc._surface_wants() == []
