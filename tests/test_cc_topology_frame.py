#!/usr/bin/env python3
"""Callosum Leg 2 §10.4-A -- cursor-based single-frame export + ratchet round-trip.

Sibling of test_cc_topology_callosum.py (the whole-graph exporter/merge suite),
scoped to export_cc_topology_frame() -- the PACED sender the leg2-tick driver
drives. Same discipline: a REAL neuro_foundation.Graph + REAL SimpleVectorDB, no
fakes (the signature mismatches that bit the sibling suite -- _is_identity_protected
takes an id string; adjacency lives in graph._outgoing -- a fake would hide).

Covers the plan's §Tests list: husk-drop, chronological order, exclude_ids,
whole-HE-never-split, overflow-keeps-HE-whole, oversized_he_at_source alarm,
structural-anchor-per-node (§8.13-G survival invariant), edges-to-acked still
ship, exhausted/convergence, only-frame-payloads-built (RAM bound via an embed
spy), atomic write, and the frame->merge->membership->next-frame ratchet.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402

from neuro_foundation import Graph                         # noqa: E402
from universal_ingestor import SimpleVectorDB              # noqa: E402
import cc_topology_export as tex                            # noqa: E402
import cc_topology_merge as tmg                             # noqa: E402


DIM = 768


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=DIM).astype(np.float32)


def _cc(g, v, nid, ct, content=None):
    """Add one connected-eligible CC node (mirrors _build_sender's provenance)."""
    meta = {"cc": True, "creation_mode": "conversational",
            "_forest_content": content or f"content for {nid}"}
    node = g.create_node(node_id=nid, metadata=meta)
    node.creation_time = ct                       # chronological ordering key
    v.insert(id=nid, embedding=_emb(ct), content=content or f"content for {nid}",
             metadata=meta)
    return node


def _run_frame(g, v, tmp_path, **kw):
    """Export one frame with test defaults; the resource gate is skipped so a
    loaded CI box never turns a correctness test into a flaky deferral."""
    out = str(tmp_path / "frame.conduit")
    kw.setdefault("machine_id", "vps")
    kw.setdefault("embedding_model", "test-model")
    kw.setdefault("skip_resource_gate", True)
    stats = tex.export_cc_topology_frame(g, v, out, **kw)
    return out, stats


def _decode(out):
    raw = open(out, "rb").read()
    frames = list(tex.read_topology_frames(raw))
    header = next(f for f in frames if f.get("kind") == "header")
    batches = [f for f in frames if f.get("kind") == "batch"]
    nodes = [n for b in batches for n in b["nodes"]]
    syns = [s for b in batches for s in b["synapses"]]
    hes = [h for b in batches for h in b["hyperedges"]]
    return header, nodes, syns, hes


def _merge(g, v, path, membership_path, idle_steps=0):
    return tmg.merge_cc_topology(
        g, v, path,
        local_machine_id="laptop",
        expected_embedding_model="test-model",
        membership_path=membership_path,
        idle_steps=idle_steps,
    )


def _connected_cc_ids(g):
    conn = set()
    for s in g.synapses.values():
        conn.add(s.pre_node_id)
        conn.add(s.post_node_id)
    for he in g.hyperedges.values():
        members = (getattr(he, "member_nodes", None)
                   or getattr(he, "member_node_ids", None) or [])
        conn |= set(members)
    return {n for n in conn
            if tex.is_cc_provenance(n, getattr(g.nodes.get(n), "metadata", {}) or {})}


def _assert_every_node_anchored(out, exclude):
    """§8.13-G: every node written ships with >=1 STRUCTURAL in-frame anchor --
    a synapse to a present (in-frame or acked) node, or membership in a whole
    hyperedge closed within (frame | acked). A husk-by-transport would violate
    this; the exporter must never produce one."""
    _, nodes, syns, hes = _decode(out)
    frame_ids = {n["id"] for n in nodes}
    present = frame_ids | set(exclude)
    anchored = set()
    for s in syns:
        if s["pre"] in frame_ids and s["post"] in present:
            anchored.add(s["pre"])
        if s["post"] in frame_ids and s["pre"] in present:
            anchored.add(s["post"])
    for h in hes:
        members = set(h["members"])
        if members <= present:
            anchored |= (members & frame_ids)
    assert frame_ids <= anchored, f"unanchored nodes shipped: {frame_ids - anchored}"


# --------------------------------------------------------------------------
# Husk drop + candidate selection
# --------------------------------------------------------------------------

def test_degree_zero_husk_is_never_shipped(tmp_path):
    g, v = Graph(), SimpleVectorDB()
    _cc(g, v, "cc:conv::turn-001", 0)
    _cc(g, v, "cc:conv::turn-002", 1)
    g.create_synapse("cc:conv::turn-001", "cc:conv::turn-002", weight=0.4, delay=2)
    _cc(g, v, "cc:conv::husk", 2)                 # connected to nothing

    out, stats = _run_frame(g, v, tmp_path, frame_size=25)
    assert "cc:conv::husk" not in stats["frame_node_ids"]
    assert set(stats["frame_node_ids"]) == {"cc:conv::turn-001", "cc:conv::turn-002"}
    _assert_every_node_anchored(out, exclude=set())


def test_non_cc_and_identity_protected_are_not_candidates(tmp_path):
    g, v = Graph(), SimpleVectorDB()
    _cc(g, v, "cc:conv::a", 0)
    _cc(g, v, "cc:conv::b", 1)
    g.create_synapse("cc:conv::a", "cc:conv::b", weight=0.4, delay=1)
    # A non-CC node wired to a CC one must still never cross.
    g.create_node(node_id="plain::x", metadata={"creation_mode": "conversational"})
    g.create_synapse("cc:conv::b", "plain::x", weight=0.4, delay=1)

    out, stats = _run_frame(g, v, tmp_path, frame_size=25)
    assert "plain::x" not in stats["frame_node_ids"]
    assert set(stats["frame_node_ids"]) == {"cc:conv::a", "cc:conv::b"}


def test_chronological_order_origin_first(tmp_path):
    g, v = Graph(), SimpleVectorDB()
    # creation_time assigned in REVERSE of id order to prove the sort is by ct.
    _cc(g, v, "cc:conv::d", 3)
    _cc(g, v, "cc:conv::c", 2)
    _cc(g, v, "cc:conv::b", 1)
    _cc(g, v, "cc:conv::a", 0)
    g.create_synapse("cc:conv::a", "cc:conv::b", weight=0.3, delay=1)
    g.create_synapse("cc:conv::b", "cc:conv::c", weight=0.3, delay=1)
    g.create_synapse("cc:conv::c", "cc:conv::d", weight=0.3, delay=1)

    out, stats = _run_frame(g, v, tmp_path, frame_size=25)
    assert stats["frame_node_ids"][0] == "cc:conv::a"          # earliest ct first


# --------------------------------------------------------------------------
# exclude_ids (membership-as-ack) + edges to acked nodes
# --------------------------------------------------------------------------

def test_exclude_ids_are_never_reshipped(tmp_path):
    g, v = Graph(), SimpleVectorDB()
    _cc(g, v, "cc:conv::a", 0)
    _cc(g, v, "cc:conv::b", 1)
    _cc(g, v, "cc:conv::c", 2)
    g.create_synapse("cc:conv::a", "cc:conv::b", weight=0.3, delay=1)
    g.create_synapse("cc:conv::b", "cc:conv::c", weight=0.3, delay=1)

    out, stats = _run_frame(g, v, tmp_path, frame_size=25,
                            exclude_ids={"cc:conv::a"})
    assert "cc:conv::a" not in stats["frame_node_ids"]
    assert set(stats["frame_node_ids"]) == {"cc:conv::b", "cc:conv::c"}


def test_synapse_to_an_acked_node_anchors_and_ships(tmp_path):
    """A node whose ONLY anchor is a synapse to an already-acked node must ship,
    and that cross-boundary synapse must ride along (edges-to-acked survive)."""
    g, v = Graph(), SimpleVectorDB()
    _cc(g, v, "cc:conv::acked", 0)
    _cc(g, v, "cc:conv::new", 1)
    g.create_synapse("cc:conv::acked", "cc:conv::new", weight=0.7, delay=5)

    out, stats = _run_frame(g, v, tmp_path, frame_size=25,
                            exclude_ids={"cc:conv::acked"})
    assert stats["frame_node_ids"] == ["cc:conv::new"]
    _, _, syns, _ = _decode(out)
    pairs = {(s["pre"], s["post"]) for s in syns}
    assert ("cc:conv::acked", "cc:conv::new") in pairs
    _assert_every_node_anchored(out, exclude={"cc:conv::acked"})


# --------------------------------------------------------------------------
# Whole-hyperedge invariant: overflow keeps it whole; oversized alarms
# --------------------------------------------------------------------------

def test_whole_hyperedge_ships_whole_past_frame_size(tmp_path):
    """A 5-member HE with frame_size=3 (hard cap 3*3=9): the HE crosses WHOLE in
    one frame rather than being split at the frame boundary (§8.14 forbids
    splitting in transport)."""
    g, v = Graph(), SimpleVectorDB()
    members = [f"cc:conv::m{i}" for i in range(5)]
    for i, nid in enumerate(members):
        _cc(g, v, nid, i)
    g.create_hyperedge(member_node_ids=set(members), activation_threshold=0.5,
                       metadata={"cc": True})

    out, stats = _run_frame(g, v, tmp_path, frame_size=3, overflow_factor=3)
    assert stats["exported_nodes"] == 5                       # whole HE, > frame_size
    _, _, _, hes = _decode(out)
    assert len(hes) == 1
    assert set(hes[0]["members"]) == set(members)
    assert stats["oversized_he_at_source"] == 0


def test_oversized_hyperedge_alarms_and_is_skipped_but_frame_proceeds(tmp_path):
    """An HE larger than the hard cap is an oversized source-side blob (§8.14):
    it raises oversized_he_at_source and is skipped WHOLE (never truncated), while
    an independent connected pair in the same graph still ships."""
    g, v = Graph(), SimpleVectorDB()
    big = [f"cc:conv::big{i}" for i in range(10)]     # 10 > hard cap (2*3=6)
    for i, nid in enumerate(big):
        _cc(g, v, nid, i)
    g.create_hyperedge(member_node_ids=set(big), activation_threshold=0.5,
                       metadata={"cc": True})
    # An unrelated shippable pair (later creation_time).
    _cc(g, v, "cc:conv::ok1", 100)
    _cc(g, v, "cc:conv::ok2", 101)
    g.create_synapse("cc:conv::ok1", "cc:conv::ok2", weight=0.4, delay=1)

    out, stats = _run_frame(g, v, tmp_path, frame_size=2, overflow_factor=3)
    assert stats["oversized_he_at_source"] == 1              # alarmed once per HE
    assert not any(nid in stats["frame_node_ids"] for nid in big)
    assert set(stats["frame_node_ids"]) == {"cc:conv::ok1", "cc:conv::ok2"}
    _assert_every_node_anchored(out, exclude=set())


# --------------------------------------------------------------------------
# Exhaustion + RAM bound + atomic write
# --------------------------------------------------------------------------

def test_exhausted_when_everything_is_acked(tmp_path):
    g, v = Graph(), SimpleVectorDB()
    _cc(g, v, "cc:conv::a", 0)
    _cc(g, v, "cc:conv::b", 1)
    g.create_synapse("cc:conv::a", "cc:conv::b", weight=0.3, delay=1)

    out, stats = _run_frame(g, v, tmp_path, frame_size=25,
                            exclude_ids={"cc:conv::a", "cc:conv::b"})
    assert stats["exhausted"] is True
    assert stats["exported_nodes"] == 0
    assert stats["candidates"] == 0


def test_only_frame_payloads_are_built(tmp_path, monkeypatch):
    """RAM discipline: payloads (embeddings) are materialized ONLY for the frame's
    nodes, never the whole graph -- the win over collect_cc_topology. Spy the
    embed fetch and assert it fires exactly once per shipped node."""
    g, v = Graph(), SimpleVectorDB()
    ids = [f"cc:conv::n{i}" for i in range(10)]
    for i, nid in enumerate(ids):
        _cc(g, v, nid, i)
    for a, b in zip(ids, ids[1:]):
        g.create_synapse(a, b, weight=0.3, delay=1)

    calls = []
    real = tex._embedding_for
    monkeypatch.setattr(tex, "_embedding_for",
                        lambda vdb, nid: calls.append(nid) or real(vdb, nid))

    out, stats = _run_frame(g, v, tmp_path, frame_size=3)
    assert len(calls) == stats["exported_nodes"]
    assert len(calls) < len(ids)                              # NOT the whole graph


def test_write_is_atomic_no_partial_left(tmp_path):
    g, v = Graph(), SimpleVectorDB()
    _cc(g, v, "cc:conv::a", 0)
    _cc(g, v, "cc:conv::b", 1)
    g.create_synapse("cc:conv::a", "cc:conv::b", weight=0.3, delay=1)

    out, stats = _run_frame(g, v, tmp_path, frame_size=25)
    assert os.path.exists(out)
    assert not os.path.exists(out + ".partial")
    header, nodes, _, _ = _decode(out)                       # decodes cleanly
    assert header["node_count"] == len(nodes) == 2


def test_frame_export_without_machine_id_refuses(tmp_path, monkeypatch):
    """Same refusal the whole-graph exporter makes: a missing hemisphere id is
    silent one-way data loss that looks like success."""
    monkeypatch.delenv("MACHINE_ID", raising=False)
    g, v = Graph(), SimpleVectorDB()
    _cc(g, v, "cc:conv::a", 0)
    _cc(g, v, "cc:conv::b", 1)
    g.create_synapse("cc:conv::a", "cc:conv::b", weight=0.3, delay=1)
    with pytest.raises(ValueError):
        tex.export_cc_topology_frame(g, v, str(tmp_path / "f.conduit"),
                                     machine_id=None, skip_resource_gate=True)


# --------------------------------------------------------------------------
# The ratchet: frame -> merge -> membership -> next frame -> convergence
# --------------------------------------------------------------------------

def _big_sender():
    g, v = Graph(), SimpleVectorDB()
    _cc(g, v, "cc:conv::t1", 0)
    _cc(g, v, "cc:conv::t1::a", 1)
    _cc(g, v, "cc:conv::t1::b", 2)
    g.create_synapse("cc:conv::t1", "cc:conv::t1::a", weight=0.4, delay=2)
    g.create_synapse("cc:conv::t1", "cc:conv::t1::b", weight=0.3, delay=1)
    g.create_hyperedge(
        member_node_ids={"cc:conv::t1", "cc:conv::t1::a", "cc:conv::t1::b"},
        activation_threshold=0.5, metadata={"cc": True})
    _cc(g, v, "cc:conv::t2", 3)
    _cc(g, v, "cc:conv::t2::a", 4)
    _cc(g, v, "cc:conv::t2::b", 5)
    g.create_synapse("cc:conv::t2", "cc:conv::t2::a", weight=0.5, delay=2)
    g.create_synapse("cc:conv::t1::a", "cc:conv::t2", weight=0.2, delay=4)   # bridge
    g.create_hyperedge(
        member_node_ids={"cc:conv::t2", "cc:conv::t2::a", "cc:conv::t2::b"},
        activation_threshold=0.5, metadata={"cc": True})
    return g, v


def test_ratchet_converges_and_binds_every_node(tmp_path):
    sg, sv = _big_sender()
    rg, rv = Graph(), SimpleVectorDB()
    membership = str(tmp_path / "laptop_cc_membership.json")

    exclude = set()
    frames = 0
    for _ in range(50):                       # generous bound; must converge well inside
        out, st = _run_frame(sg, sv, tmp_path, frame_size=2, exclude_ids=exclude)
        if st.get("exhausted"):
            break
        assert st["exported_nodes"] > 0, "cursor stalled -- candidates but no progress"
        frames += 1
        _assert_every_node_anchored(out, exclude)

        # HE never split across frames: any shipped HE is closed within (frame|acked).
        _, nodes, _, hes = _decode(out)
        frame_ids = {n["id"] for n in nodes}
        for h in hes:
            assert set(h["members"]) <= (frame_ids | exclude), \
                "hyperedge split across the frame boundary"

        mst = _merge(rg, rv, out, membership)
        assert mst.get("consolidation_skipped_unbound_arrivals", 0) == 0, \
            "a frame stranded an unbound arrival (whole-containment gap)"
        exclude = tmg._load_membership(membership)
    else:
        pytest.fail("ratchet did not converge within the iteration bound")

    assert frames >= 2, "fixture should need multiple frames to be a real ratchet"
    connected = _connected_cc_ids(sg)
    assert connected.issubset(set(rg.nodes)), \
        f"receiver missing: {connected - set(rg.nodes)}"
    # Both binding hyperedges made it across whole.
    assert len(rg.hyperedges) == 2


def test_remerging_the_same_frame_absorbs_nothing_new(tmp_path):
    sg, sv = _big_sender()
    rg, rv = Graph(), SimpleVectorDB()
    membership = str(tmp_path / "laptop_cc_membership.json")

    out, _ = _run_frame(sg, sv, tmp_path, frame_size=25, exclude_ids=set())
    first = _merge(rg, rv, out, membership)
    assert first["absorbed_nodes"] > 0
    again = _merge(rg, rv, out, membership)
    assert again["absorbed_nodes"] == 0            # idempotent (skipped_present)


def test_membership_ack_advances_the_cursor(tmp_path):
    """After a merge, the receiver's written membership must exclude exactly what
    it now holds, so the next export ships strictly new nodes."""
    sg, sv = _big_sender()
    rg, rv = Graph(), SimpleVectorDB()
    membership = str(tmp_path / "laptop_cc_membership.json")

    out1, st1 = _run_frame(sg, sv, tmp_path, frame_size=2, exclude_ids=set())
    _merge(rg, rv, out1, membership)
    ack = tmg._load_membership(membership)
    assert set(st1["frame_node_ids"]).issubset(ack)

    out2, st2 = _run_frame(sg, sv, tmp_path, frame_size=2, exclude_ids=ack)
    assert not (set(st2["frame_node_ids"]) & set(st1["frame_node_ids"])), \
        "second frame re-shipped acked nodes"
