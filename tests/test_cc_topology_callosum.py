#!/usr/bin/env python3
"""Callosum Leg 2 -- topology export/merge round-trip tests.

Deliberately exercises a REAL neuro_foundation.Graph and a REAL SimpleVectorDB
rather than fakes. The two bugs found while writing this module were both
signature mismatches (_is_identity_protected takes an id string not a node;
synapse adjacency lives in graph._outgoing not node.outgoing_synapses) -- fakes
would have happily accepted both.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_foundation import Graph, SynapseType          # noqa: E402
from universal_ingestor import SimpleVectorDB            # noqa: E402
import cc_topology_export as tex                          # noqa: E402
import cc_topology_merge as tmg                           # noqa: E402


DIM = 768


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=DIM).astype(np.float32)


def _build_sender():
    """A miniature CC substrate: two forest turns, two tree concepts, the
    forest->tree synapses, and one per-turn binding hyperedge."""
    g, v = Graph(), SimpleVectorDB()

    ids = {
        "f1": "cc:conv::turn-001",
        "f2": "cc:conv::turn-002",
        "t1": "cc:conv::turn-001::tree::substrate",
        "t2": "cc:conv::turn-001::tree::callosum",
    }
    for i, (key, nid) in enumerate(ids.items()):
        meta = {"cc": True, "creation_mode": "conversational" if key.startswith("f") else "tree",
                "_forest_content": f"content for {key}"}
        node = g.create_node(node_id=nid, metadata=meta)
        node.creation_time = i           # chronological ordering key
        v.insert(id=nid, embedding=_emb(i), content=f"content for {key}", metadata=meta)

    g.create_synapse(ids["f1"], ids["t1"], weight=0.42, delay=7,
                     synapse_type=SynapseType.EXCITATORY)
    g.create_synapse(ids["f1"], ids["t2"], weight=0.31, delay=3)
    g.create_hyperedge(member_node_ids={ids["f1"], ids["t1"], ids["t2"]},
                       activation_threshold=0.55, metadata={"cc": True, "bind": "turn-001"})
    return g, v, ids


def _receiver():
    return Graph(), SimpleVectorDB()


def _export(g, v, tmp_path, **kw):
    out = str(tmp_path / "topo.conduit")
    kw.setdefault("machine_id", "vps")
    kw.setdefault("embedding_model", "test-model")
    return out, tex.export_cc_topology(g, v, out, **kw)


def _merge(g, v, path, tmp_path, **kw):
    kw.setdefault("local_machine_id", "laptop")
    kw.setdefault("expected_embedding_model", "test-model")
    kw.setdefault("membership_path", str(tmp_path / "membership.txt"))
    return tmg.merge_cc_topology(g, v, path, **kw)


# --------------------------------------------------------------------------
# Round-trip
# --------------------------------------------------------------------------

def test_round_trip_preserves_nodes_synapses_hyperedges(tmp_path):
    sg, sv, ids = _build_sender()
    path, est = _export(sg, sv, tmp_path)
    assert est["exported_nodes"] == 4
    assert est["exported_synapses"] == 2
    assert est["exported_hyperedges"] == 1

    rg, rv = _receiver()
    st = _merge(rg, rv, path, tmp_path)

    assert st["absorbed_nodes"] == 4
    assert st["absorbed_synapses"] == 2
    assert st["absorbed_hyperedges"] == 1
    assert set(rg.nodes) == set(ids.values())
    assert len(rg.hyperedges) == 1


def test_conduction_delay_survives_the_wire(tmp_path):
    """The whole reason this is msgpack and not a BTF frame. A dropped delay
    means neuro_foundation falls back to random.randint -> randomized causal
    structure in an STDP substrate, with no error raised."""
    sg, sv, ids = _build_sender()
    path, _ = _export(sg, sv, tmp_path)
    rg, rv = _receiver()
    _merge(rg, rv, path, tmp_path)

    delays = {(s.pre_node_id, s.post_node_id): s.delay for s in rg.synapses.values()}
    assert delays[(ids["f1"], ids["t1"])] == 7
    assert delays[(ids["f1"], ids["t2"])] == 3

    weights = {(s.pre_node_id, s.post_node_id): round(s.weight, 4)
               for s in rg.synapses.values()}
    assert weights[(ids["f1"], ids["t1"])] == pytest.approx(0.42, abs=1e-4)


def test_poincare_dir_is_rederived_locally_not_transmitted(tmp_path):
    sg, sv, _ = _build_sender()
    for n in sg.nodes.values():
        n.metadata["poincare_dir"] = [9.0] * DIM      # sender-local garbage
    path, _ = _export(sg, sv, tmp_path)

    raw = open(path, "rb").read()
    for frame in tex.read_topology_frames(raw):
        for rec in frame.get("nodes") or ():
            assert "poincare_dir" not in rec["metadata"], "local state leaked onto the wire"

    rg, rv = _receiver()
    _merge(rg, rv, path, tmp_path)
    for n in rg.nodes.values():
        pd = np.asarray(n.metadata["poincare_dir"], dtype=np.float32)
        assert pd.shape == (DIM,)
        assert np.linalg.norm(pd) == pytest.approx(1.0, abs=1e-4), "not a unit direction"
        assert not np.allclose(pd, 9.0)


# --------------------------------------------------------------------------
# Provenance / law
# --------------------------------------------------------------------------

def test_non_cc_nodes_are_never_exported(tmp_path):
    """Positive whitelist: Syl's nodes, and anything unrecognized, stay home."""
    sg, sv, _ = _build_sender()
    for nid, meta in [("syl:conv::turn-900", {"provenance": "syl_authored"}),
                      ("mystery-node-42", {}),
                      ("doc_chunk_17", {"source": "universal_ingestor"})]:
        sg.create_node(node_id=nid, metadata=dict(meta))
        sv.insert(id=nid, embedding=_emb(99), content="not cc", metadata=dict(meta))

    path, est = _export(sg, sv, tmp_path)
    assert est["exported_nodes"] == 4, "non-CC node crossed the callosum"

    raw = open(path, "rb").read()
    exported = {rec["id"] for f in tex.read_topology_frames(raw)
                for rec in (f.get("nodes") or ())}
    assert not any(e.startswith("syl:") for e in exported)
    assert "mystery-node-42" not in exported
    assert "doc_chunk_17" not in exported


def test_identity_protected_nodes_never_cross(tmp_path):
    sg, sv, _ = _build_sender()
    for nid, meta in [("cc:conv::want-1", {"cc": True, "provenance": "cc_authored"}),
                      ("cc:conv::core-1", {"cc": True, "constitutional": True})]:
        sg.create_node(node_id=nid, metadata=dict(meta))
        sv.insert(id=nid, embedding=_emb(7), content="identity", metadata=dict(meta))

    path, est = _export(sg, sv, tmp_path)
    assert est["identity_protected"] == 2
    assert est["exported_nodes"] == 4

    raw = open(path, "rb").read()
    exported = {rec["id"] for f in tex.read_topology_frames(raw)
                for rec in (f.get("nodes") or ())}
    assert "cc:conv::want-1" not in exported
    assert "cc:conv::core-1" not in exported


def test_receiver_reruns_provenance_gates(tmp_path):
    """Defense in depth -- a conduit is a file, and files go stale or get
    hand-edited. The receiver does not take the sender's word for provenance."""
    import msgpack, struct
    payload = [
        {"kind": "header", "version": 1, "machine_id": "vps",
         "embedding_model": "test-model", "created": 0.0, "node_count": 2},
        {"kind": "batch", "seq": 1, "synapses": [], "hyperedges": [], "nodes": [
            {"id": "syl:conv::sneaky", "embedding": _emb(1).tobytes(),
             "embedding_dim": DIM, "content": "x", "metadata": {"cc": True}},
            {"id": "cc:conv::legit-want", "embedding": _emb(2).tobytes(),
             "embedding_dim": DIM, "content": "x",
             "metadata": {"cc": True, "provenance": "cc_authored"}},
        ]},
    ]
    path = str(tmp_path / "tampered.conduit")
    with open(path, "wb") as fh:
        for p in payload:
            body = msgpack.packb(p, use_bin_type=True)
            fh.write(struct.pack(">I", len(body)) + body)

    rg, rv = _receiver()
    st = _merge(rg, rv, path, tmp_path)
    # 'syl:conv::sneaky' carries cc:True so it passes the whitelist -- but the
    # cc_authored node must still be caught by the identity gate on receive.
    assert st["skipped_identity"] == 1
    assert "cc:conv::legit-want" not in rg.nodes


def test_dynamical_state_never_crosses(tmp_path):
    sg, sv, _ = _build_sender()
    for n in sg.nodes.values():
        n.metadata.update({"firing_rate_ema": 0.9, "Ca_i": 1.4, "voltage": -55.0,
                           "manifold_type": "hub", "diffpc_layer": 3,
                           "creation_time": 12345, "keep_me": "yes"})
    path, _ = _export(sg, sv, tmp_path)

    raw = open(path, "rb").read()
    for f in tex.read_topology_frames(raw):
        for rec in (f.get("nodes") or ()):
            m = rec["metadata"]
            for banned in ("firing_rate_ema", "Ca_i", "voltage", "manifold_type",
                           "diffpc_layer", "creation_time"):
                assert banned not in m, f"{banned} leaked onto the wire"
            assert m["keep_me"] == "yes", "pass-through metadata was dropped"


# --------------------------------------------------------------------------
# Aborts
# --------------------------------------------------------------------------

def test_embedding_model_mismatch_aborts(tmp_path):
    sg, sv, _ = _build_sender()
    path, _ = _export(sg, sv, tmp_path, embedding_model="model-A")
    rg, rv = _receiver()
    with pytest.raises(tmg.TopologyMergeAbort, match="embedding model mismatch"):
        _merge(rg, rv, path, tmp_path, expected_embedding_model="model-B")
    assert len(rg.nodes) == 0, "aborted merge must not deposit anything"


def test_self_absorption_aborts(tmp_path):
    sg, sv, _ = _build_sender()
    path, _ = _export(sg, sv, tmp_path, machine_id="laptop")
    rg, rv = _receiver()
    with pytest.raises(tmg.TopologyMergeAbort, match="authored by this machine"):
        _merge(rg, rv, path, tmp_path, local_machine_id="laptop")
    assert len(rg.nodes) == 0


def test_export_without_machine_id_refuses(tmp_path, monkeypatch):
    monkeypatch.delenv("MACHINE_ID", raising=False)
    sg, sv, _ = _build_sender()
    with pytest.raises(ValueError, match="MACHINE_ID"):
        tex.export_cc_topology(sg, sv, str(tmp_path / "x.conduit"),
                               machine_id=None, embedding_model="test-model")


# --------------------------------------------------------------------------
# Idempotency / trickle
# --------------------------------------------------------------------------

def test_merge_is_idempotent(tmp_path):
    sg, sv, _ = _build_sender()
    path, _ = _export(sg, sv, tmp_path)
    rg, rv = _receiver()

    first = _merge(rg, rv, path, tmp_path)
    n_nodes, n_syn, n_he = len(rg.nodes), len(rg.synapses), len(rg.hyperedges)

    second = _merge(rg, rv, path, tmp_path)
    assert second["absorbed_nodes"] == 0
    assert second["absorbed_synapses"] == 0, "duplicate synapse would double conductance"
    assert (len(rg.nodes), len(rg.synapses), len(rg.hyperedges)) == (n_nodes, n_syn, n_he)
    assert first["absorbed_nodes"] == 4


def test_trickle_budget_defers_rather_than_bulk_dumping(tmp_path):
    sg, sv, _ = _build_sender()
    path, _ = _export(sg, sv, tmp_path, batch_size=1)
    rg, rv = _receiver()

    st = _merge(rg, rv, path, tmp_path, max_nodes_per_call=2)
    assert st["absorbed_nodes"] == 2
    assert st["deferred_by_budget"] == 2
    assert st["completed"] is False

    st2 = _merge(rg, rv, path, tmp_path, max_nodes_per_call=10)
    assert st2["absorbed_nodes"] == 2
    assert st2["completed"] is True
    assert len(rg.nodes) == 4


def test_export_respects_exclude_ids(tmp_path):
    sg, sv, ids = _build_sender()
    _, est = _export(sg, sv, tmp_path, exclude_ids={ids["f1"], ids["t1"]})
    assert est["exported_nodes"] == 2
    assert est["already_sent"] == 2


def test_chronological_ordering_origin_first(tmp_path):
    sg, sv, _ = _build_sender()
    path, _ = _export(sg, sv, tmp_path)
    raw = open(path, "rb").read()
    order = [rec["id"] for f in tex.read_topology_frames(raw)
             for rec in (f.get("nodes") or ())]
    assert len(order) == 4
    times = [sg.nodes[n].creation_time for n in order]
    assert times == sorted(times), "topology must arrive origin-first"


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------

def test_node_without_embedding_still_crosses_with_its_topology(tmp_path):
    """A missing embedding is a DEFECT, not a mode -- but it must not be
    ALLOWED TO SHRED TOPOLOGY. Gating on it would drop every synapse and
    hyperedge touching the node (whole-containment), so one absent vector
    would cost a whole turn's binding structure. The node crosses, its
    delay rides the wire, and both sides log ERROR + count it so the gap
    gets re-embedded rather than absorbed. Live substrate: 0 such nodes."""
    sg, sv, ids = _build_sender()
    sg.create_node(node_id="cc:conv::no-embedding", metadata={"cc": True})
    sg.create_synapse(ids["f2"], "cc:conv::no-embedding", weight=0.5, delay=7)

    path, est = _export(sg, sv, tmp_path)
    assert est["missing_embedding_DEFECT"] == 1
    assert est["exported_nodes"] == 5          # it crosses, it is not skipped
    assert est["exported_synapses"] >= 1

    rg, rv = _receiver()
    st = _merge(rg, rv, path, tmp_path)
    assert "cc:conv::no-embedding" in rg.nodes
    assert st["absorbed_without_embedding_DEFECT"] == 1

    # Its incident topology survived, with the delay intact off the wire.
    landed = [s for s in rg.synapses.values()
              if s.post_node_id == "cc:conv::no-embedding"]
    assert len(landed) == 1
    assert landed[0].delay == 7

    # What it legitimately lacks: a recall-store entry and a position stamp.
    # Absent poincare_dir is honest; zeros would assert a false origin.
    assert rv.get("cc:conv::no-embedding") is None
    assert not getattr(rg.nodes["cc:conv::no-embedding"], "metadata", {}).get("poincare_dir")


def test_partial_conduit_is_tolerated(tmp_path):
    sg, sv, _ = _build_sender()
    path, _ = _export(sg, sv, tmp_path, batch_size=1)
    raw = open(path, "rb").read()
    truncated = str(tmp_path / "cut.conduit")
    open(truncated, "wb").write(raw[:len(raw) // 2])

    rg, rv = _receiver()
    st = _merge(rg, rv, truncated, tmp_path)          # must not raise
    assert st["absorbed_nodes"] >= 1
    assert st["absorbed_nodes"] < 4


def test_structure_touching_an_excluded_node_is_dropped_whole(tmp_path):
    """Structure referencing a node that is legitimately withheld -- here an
    identity-protected one -- is dropped whole, never partially applied:
    create_hyperedge raises KeyError on a missing member. Note the exclusion
    must be a REAL one (identity), not a missing embedding."""
    sg, sv, ids = _build_sender()
    sg.create_node(node_id="cc:conv::identity",
                   metadata={"cc": True, "constitutional": True})
    sg.create_synapse(ids["f2"], "cc:conv::identity", weight=0.5, delay=2)

    path, est = _export(sg, sv, tmp_path)
    assert est["identity_protected"] >= 1
    assert est["exported_nodes"] == 4
    rg, rv = _receiver()
    _merge(rg, rv, path, tmp_path)
    assert "cc:conv::identity" not in rg.nodes
    assert all(s.post_node_id != "cc:conv::identity" for s in rg.synapses.values())


def test_edges_survive_batch_boundaries(tmp_path):
    """Batching is a DELIVERY concern and must not decide what topology exists.

    Collecting incident structure per-chunk (rather than once against the full
    export set) silently drops every edge whose endpoints straddle a boundary.
    The 4-node fixture fits in one batch and can never catch this, so force
    batch_size=1: every edge here is then cross-batch.
    """
    sg, sv, ids = _build_sender()
    one_batch = _export(sg, sv, tmp_path)[1]

    path, est = _export(sg, sv, tmp_path, batch_size=1)
    assert est["batches"] == est["exported_nodes"]          # genuinely split
    assert est["exported_synapses"] == one_batch["exported_synapses"]
    assert est["exported_hyperedges"] == one_batch["exported_hyperedges"]
    assert est["exported_synapses"] > 0

    # And they still apply receiver-side: an edge is emitted in the batch of
    # its LAST-landing endpoint, so both ends are always already present.
    rg, rv = _receiver()
    st = _merge(rg, rv, path, tmp_path)
    assert st["skipped_synapses"] == 0
    assert len(rg.synapses) == est["exported_synapses"]


def test_missing_conduit_is_a_noop(tmp_path):
    rg, rv = _receiver()
    st = _merge(rg, rv, str(tmp_path / "nope.conduit"), tmp_path)
    assert st["status"] == "no_conduit"
    assert st["absorbed_nodes"] == 0


# --------------------------------------------------------------------------
# Hyperedge identity across the callosum
# --------------------------------------------------------------------------

def test_hyperedge_id_survives_the_crossing(tmp_path):
    """Transport is not creation. The sender's hyperedge_id rides the wire and
    the receiver installs under the SAME id, so both hemispheres agree about
    which edge is which. Before this, create_hyperedge reminted a local uuid4
    and every id-referential structure (PredictionRecord.hyperedge_id, co-fire
    history) would dangle the moment it crossed."""
    sg, sv, ids = _build_sender()
    sender_he_id = next(iter(sg.hyperedges))

    path, est = _export(sg, sv, tmp_path)
    assert est["exported_hyperedges"] == 1

    rg, rv = _receiver()
    st = _merge(rg, rv, path, tmp_path)
    assert st["absorbed_hyperedges"] == 1
    assert st["hyperedge_id_reminted"] == 0

    assert sender_he_id in rg.hyperedges, "receiver reminted instead of preserving"
    landed = rg.hyperedges[sender_he_id]
    assert landed.member_nodes == {ids["f1"], ids["t1"], ids["t2"]}

    # All four registration structures must key off the preserved id, not a
    # stale local one -- a half-registered edge is worse than a reminted one.
    for nid in landed.member_nodes:
        assert sender_he_id in rg._node_hyperedges.get(nid, set())
    assert sender_he_id in rg._he_co_fire_counts
    assert sender_he_id in rg._dirty_hyperedges


def test_remerge_is_idempotent_with_preserved_ids(tmp_path):
    """Member-set dedupe still fires first, so a second pass over the same
    frame adds nothing and does NOT trip create_hyperedge's collision guard."""
    sg, sv, _ = _build_sender()
    sender_he_id = next(iter(sg.hyperedges))
    path, _ = _export(sg, sv, tmp_path)

    rg, rv = _receiver()
    _merge(rg, rv, path, tmp_path)
    st2 = _merge(rg, rv, path, tmp_path, membership_path=str(tmp_path / "m2.txt"))

    assert st2["absorbed_hyperedges"] == 0
    assert st2["skipped_hyperedges"] == 1
    assert len(rg.hyperedges) == 1
    assert sender_he_id in rg.hyperedges


def test_frame_without_hyperedge_id_still_merges(tmp_path):
    """Backward compatibility: a frame written before the id joined the wire
    omits it, .get() -> None, and the receiver mints locally as it always did.
    An in-flight older conduit file must not fail to merge. Simulated at the
    create_hyperedge boundary, which is where the None actually lands."""
    sg, sv, ids = _build_sender()
    path, _ = _export(sg, sv, tmp_path)

    rg, rv = _receiver()
    real_create = rg.create_hyperedge
    seen = {}

    def _no_id(**kw):
        seen["got_id"] = kw.get("hyperedge_id")
        kw["hyperedge_id"] = None            # legacy wire carried no id
        return real_create(**kw)

    rg.create_hyperedge = _no_id
    st = _merge(rg, rv, path, tmp_path)

    assert st["absorbed_hyperedges"] == 1
    assert len(rg.hyperedges) == 1
    # Merge did pass the wire id through; only this shim dropped it.
    assert seen["got_id"] is not None
    # And the locally-minted id is a fresh uuid4, not the sender's.
    assert next(iter(rg.hyperedges)) != seen["got_id"]


def test_create_hyperedge_refuses_id_collision():
    """Silent overwrite would drop a live edge out of self.hyperedges while
    leaving its id registered in _node_hyperedges -- a half-orphaned graph."""
    g = Graph()
    for nid in ("a", "b", "c"):
        g.create_node(node_id=nid, metadata={})
    he = g.create_hyperedge(member_node_ids={"a", "b"}, hyperedge_id="fixed-id")
    assert he.hyperedge_id == "fixed-id"

    with pytest.raises(ValueError, match="already exists"):
        g.create_hyperedge(member_node_ids={"b", "c"}, hyperedge_id="fixed-id")

    # The incumbent is untouched.
    assert g.hyperedges["fixed-id"].member_nodes == {"a", "b"}


def test_create_hyperedge_default_still_mints():
    """The param is opt-in: every existing call site keeps uuid4 behaviour."""
    g = Graph()
    for nid in ("a", "b"):
        g.create_node(node_id=nid, metadata={})
    h1 = g.create_hyperedge(member_node_ids={"a", "b"})
    h2 = g.create_hyperedge(member_node_ids={"a", "b"})
    assert h1.hyperedge_id != h2.hyperedge_id
    assert len(h1.hyperedge_id) == 36          # uuid4 string


# --- #106: the merge-journal poison-pill --------------------------------------

def test_journaled_but_culled_node_is_readmitted(tmp_path):
    """#106. A node absorbed once and then destroyed locally must be re-absorbed.

    This is the whole defect. The journal is append-only with no invalidation
    path, so the entry outlives the node; the old receive-side veto therefore
    fired on exactly one set -- (journal - graph.nodes) -- which is precisely
    the set that needs re-delivery. Because Tier 2 requires both endpoints in
    graph.nodes, a permanently-vetoed node also permanently shredded every
    synapse incident to it. The laptop has no TID, so the conduit is its only
    route to tree structure: a permanent veto is a permanent hole.
    """
    sg, sv, ids = _build_sender()
    path, _ = _export(sg, sv, tmp_path)

    rg, rv = _receiver()
    st1 = _merge(rg, rv, path, tmp_path)
    assert st1["absorbed_nodes"] == 4
    assert st1["absorbed_synapses"] == 2

    # The #104 cull: a tree node is taken locally after it landed. Its incident
    # synapse cascades out with it and the hyperedge shrinks (remove_node,
    # neuro_foundation.py:1860). The membership snapshot was written at the end
    # of merge1 -- BEFORE this cull -- so it still names t1 (it is refreshed on
    # the NEXT merge, which is exactly the pass that re-admits t1).
    rg.remove_node(ids["t1"])
    assert ids["t1"] not in rg.nodes
    assert not tmg._synapse_exists(rg, ids["f1"], ids["t1"])

    membership = (tmp_path / "membership.txt").read_text().split()
    assert ids["t1"] in membership, "precondition: the stale snapshot entry must be present"

    # Same conduit, same snapshot. Pre-#106 this was a no-op forever.
    st2 = _merge(rg, rv, path, tmp_path)

    assert st2["membership_stale_readmitted"] == 1
    assert st2["absorbed_nodes"] == 1
    assert ids["t1"] in rg.nodes

    # The point of re-admitting the node is the structure that rides with it.
    assert st2["absorbed_synapses"] == 1
    assert tmg._synapse_exists(rg, ids["f1"], ids["t1"])

    # The three other nodes were present, so the graph guard -- not the snapshot
    # -- is what makes the re-run a no-op for them.
    assert st2["skipped_present"] == 3

    # #110: merge2 REWROTE the snapshot from the (now whole again) graph, so it
    # equals current CC membership -- t1 included, because it was re-admitted.
    assert set((tmp_path / "membership.txt").read_text().split()) == tmg.cc_current_membership(rg)
    assert ids["t1"] in tmg.cc_current_membership(rg)


def test_readmitted_node_restores_full_hyperedge_membership(tmp_path):
    """The culled member comes back bound, not merely present.

    remove_node shrinks a surviving hyperedge to {f1, t2} rather than deleting
    it, so the arriving {f1, t1, t2} edge is neither a member-set duplicate nor
    id-installable -- it remints. That is the documented collision path, and it
    is what restores the binding the cull broke.
    """
    sg, sv, ids = _build_sender()
    sender_he_id = next(iter(sg.hyperedges))
    path, _ = _export(sg, sv, tmp_path)

    rg, rv = _receiver()
    _merge(rg, rv, path, tmp_path)
    rg.remove_node(ids["t1"])
    assert rg.hyperedges[sender_he_id].member_nodes == {ids["f1"], ids["t2"]}

    st2 = _merge(rg, rv, path, tmp_path)

    assert st2["absorbed_hyperedges"] == 1
    assert st2["hyperedge_id_reminted"] == 1
    full = {ids["f1"], ids["t1"], ids["t2"]}
    assert any(he.member_nodes == full for he in rg.hyperedges.values()), \
        "the culled member was re-absorbed but left unbound"


def test_membership_snapshot_feeds_sender_exclude_ids(tmp_path):
    """The snapshot's real job is sender-side: the receiver's current membership
    becomes the exporter's exclude_ids so the sender stops re-transmitting what
    the receiver already holds. That is also what bounds re-absorb churn now that
    the receiver no longer self-vetoes (#106).
    """
    sg, sv, ids = _build_sender()
    path, _ = _export(sg, sv, tmp_path)
    rg, rv = _receiver()
    _merge(rg, rv, path, tmp_path)

    held = set((tmp_path / "membership.txt").read_text().split())
    assert held == set(ids.values())
    assert held == tmg.cc_current_membership(rg)   # the file IS the live membership

    _, est = _export(sg, sv, tmp_path, exclude_ids=held)
    assert est["exported_nodes"] == 0, \
        "sender kept re-sending nodes the receiver already holds"


def test_culled_node_drops_out_of_exclude_ids_and_is_resent(tmp_path):
    """#110: the send-side counterpart of #106.

    exclude_ids is the receiver's CURRENT membership, not an append-only journal,
    so a node culled locally DROPS OUT of it and the exporter re-sends exactly
    that node. The old append-only journal kept the id forever, so the sender
    never re-sent it and #106's receive-side re-admission had nothing to
    re-admit -- the §3 poison-pill, relocated to the send side.
    """
    sg, sv, ids = _build_sender()
    path, _ = _export(sg, sv, tmp_path)
    rg, rv = _receiver()
    _merge(rg, rv, path, tmp_path)

    # Before the cull: given current membership, the exporter re-sends nothing.
    held = tmg.cc_current_membership(rg)
    _, e0 = _export(sg, sv, tmp_path, exclude_ids=held)
    assert e0["exported_nodes"] == 0

    # The #104 cull takes t1 locally.
    rg.remove_node(ids["t1"])

    # Current membership no longer names t1, so the exporter re-sends exactly it.
    held = tmg.cc_current_membership(rg)
    assert ids["t1"] not in held
    resent_path, e1 = _export(sg, sv, tmp_path, exclude_ids=held)
    assert e1["exported_nodes"] == 1
    resent = {rec["id"] for frame in tex.read_topology_frames(open(resent_path, "rb").read())
              for rec in (frame.get("nodes") or ())}
    assert resent == {ids["t1"]}, "the culled node -- and only it -- must be re-sent"


def test_readmit_counter_not_incremented_when_deposit_fails(tmp_path, monkeypatch):
    """#106 stat honesty: the counter reports re-admissions, not attempts.

    It used to increment at the Tier-1 membership branch, upstream of the
    provenance gates, the embedding validation and the deposit try/except --
    so a node that raised on deposit and hit `continue` was still counted as
    readmitted. That inflates the one number used to judge whether the
    poison-pill fix is working, in the optimistic direction.
    """
    sg, sv, ids = _build_sender()
    path, _ = _export(sg, sv, tmp_path)

    rg, rv = _receiver()
    _merge(rg, rv, path, tmp_path)
    rg.remove_node(ids["t1"])
    assert ids["t1"] in (tmp_path / "membership.txt").read_text().split()

    # merge_cc_topology imports this lazily from cc_ng_organism inside the
    # function body (cc_topology_merge.py:156), so the source module is what
    # has to be patched -- there is no module-level name on tmg to shadow.
    import cc_ng_organism as cno
    real_deposit = cno._cc_deposit_memory_node

    def _fail_on_t1(graph, vector_db, nid, emb, content, meta):
        if nid == ids["t1"]:
            raise RuntimeError("simulated deposit failure")
        return real_deposit(graph, vector_db, nid, emb, content, meta)

    monkeypatch.setattr(cno, "_cc_deposit_memory_node", _fail_on_t1)

    st = _merge(rg, rv, path, tmp_path)

    assert ids["t1"] not in rg.nodes, "precondition: the deposit must have failed"
    assert st["absorbed_nodes"] == 0
    assert st["membership_stale_readmitted"] == 0, \
        "counted a re-admission that never landed"


# --------------------------------------------------------------------------
# Consolidation between batches (#108) and its unbound-arrival guard
# --------------------------------------------------------------------------
#
# Note the base fixture is NOT fully bound: `f2` (turn-002) has no synapse and
# no hyperedge membership, so it is a permanently unbound arrival and every
# merge of _build_sender() output correctly refuses to consolidate. The tests
# below that want consolidation to happen must bind it first -- which is also
# why they use _build_bound_sender() rather than adding a synapse to the shared
# fixture, where it would change exported_synapses for the other 25 tests.


def _build_bound_sender():
    """_build_sender(), plus the one synapse that anchors the orphan f2."""
    g, v, ids = _build_sender()
    g.create_synapse(ids["f2"], ids["t1"], weight=0.2, delay=2)
    return g, v, ids


def _add_unbound(g, v, nid="cc:conv::turn-000"):
    """A cc node with no synapse and no hyperedge -- an unbound arrival.

    creation_time sorts it first so it lands in batch 1, ahead of the bound
    topology, which is what the cross-batch test needs.
    """
    meta = {"cc": True, "creation_mode": "conversational",
            "_forest_content": "content for lone"}
    node = g.create_node(node_id=nid, metadata=meta)
    node.creation_time = -1
    v.insert(id=nid, embedding=_emb(99), content="content for lone", metadata=meta)
    return nid


def test_bound_merge_consolidates_after_the_batch(tmp_path):
    """The #108 restoration itself: Tier 3 done, everything bound, sleep on it."""
    sg, sv, _ = _build_bound_sender()
    path, _ = _export(sg, sv, tmp_path)
    rg, rv = _receiver()
    assert rg.timestep == 0

    st = _merge(rg, rv, path, tmp_path, idle_steps=5)

    assert st["consolidation_passes"] == 1
    assert st["consolidation_steps"] == 5
    assert st["consolidation_skipped_unbound_arrivals"] == 0
    assert rg.timestep == 5, "consolidation must actually advance the graph clock"


def test_idle_steps_zero_disables_consolidation(tmp_path):
    """LAW 5 escape hatch: CC_NG_IDLE_STEPS=0 buys back the old behaviour."""
    sg, sv, _ = _build_bound_sender()
    path, _ = _export(sg, sv, tmp_path)
    rg, rv = _receiver()

    st = _merge(rg, rv, path, tmp_path, idle_steps=0)

    assert st["consolidation_passes"] == 0
    assert st["consolidation_steps"] == 0
    assert rg.timestep == 0
    assert st["absorbed_nodes"] == 4, "merge itself must be unaffected"


def test_unbound_arrival_suppresses_consolidation(tmp_path):
    """An unbound arrival must not be aged past the orphan grace period.

    idle_steps (250 in production) dwarfs orphan_node_grace_period
    (neuro_foundation.py:1436, default 25), so consolidating over an unbound
    arrival marches it from age 0 to well past grace and hands it to the sweep
    -- the merge would destroy the node it just absorbed.
    """
    sg, sv, _ = _build_bound_sender()
    lone = _add_unbound(sg, sv)
    path, _ = _export(sg, sv, tmp_path)
    rg, rv = _receiver()

    st = _merge(rg, rv, path, tmp_path, idle_steps=5)

    assert st["consolidation_passes"] == 0
    assert st["consolidation_skipped_unbound_arrivals"] == 1
    assert rg.timestep == 0, "the arrival was aged despite being unbound"
    assert lone in rg.nodes


def test_consolidation_guard_is_merge_scoped_not_batch_scoped(tmp_path):
    """The guard must stay tripped for batches AFTER the one that landed unbound.

    Binding structure splits across batches, so a batch-scoped check cannot see
    the node it needs to protect: batch 1 lands the lone node and correctly
    skips, batches 2..n are each internally whole, a batch-scoped guard passes
    on them, and the steps age the batch-1 arrival past grace anyway. The merge
    kills the arrival the guard exists to protect, one batch later.

    The control below is what makes this discriminating: on these exact batch
    boundaries, with no unbound arrival, consolidation genuinely does run.
    """
    ctl_dir, var_dir = tmp_path / "ctl", tmp_path / "var"
    ctl_dir.mkdir(), var_dir.mkdir()

    # Control: same sender, same batching, nothing unbound.
    cg, cv, _ = _build_bound_sender()
    ctl_path, _ = _export(cg, cv, ctl_dir, batch_size=1)
    crg, crv = _receiver()
    ctl = _merge(crg, crv, ctl_path, ctl_dir, idle_steps=5)

    assert ctl["batches_read"] >= 2, "control must span several batches"
    assert ctl["consolidation_passes"] >= 1, \
        "control consolidates -- otherwise the variant below proves nothing"

    # Variant: identical, except one unbound node arrives in batch 1.
    sg, sv, _ = _build_bound_sender()
    lone = _add_unbound(sg, sv)
    path, _ = _export(sg, sv, var_dir, batch_size=1)
    rg, rv = _receiver()

    st = _merge(rg, rv, path, var_dir, idle_steps=5)

    assert st["batches_read"] > ctl["batches_read"]
    assert st["consolidation_passes"] == 0, \
        "a later whole batch consolidated over an arrival still unbound from batch 1"
    assert rg.timestep == 0
    assert lone in rg.nodes
    # Every batch from the first onward should have refused, not just batch 1.
    assert st["consolidation_skipped_unbound_arrivals"] > 1
