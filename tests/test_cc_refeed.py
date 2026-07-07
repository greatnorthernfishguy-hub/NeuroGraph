# ---- Changelog ----
# [2026-07-07] Claude Code (Fable 5) — Refeed feeder tests
# What: Tests for cc_refeed.py -- orphan collection (floor/journal/live-node/
#   already-embodied exclusions), journal roundtrip + resume, load-pause logic,
#   and the golden end-to-end: feeder writes -> real drain_ingest_tract absorbs
#   -> conversational node exists in the graph.
# Why: Josh's refeed requirements (recoverable, load-aware) each need a
#   test-visible pin; the end-to-end proves the front-door contract (ng_tract
#   frame format + source filter) against the REAL drain, not a mock.
# How: msgpack checkpoint fixtures built from real Graph/SimpleVectorDB saves;
#   real NeuroGraphMemory for the absorption test (same cc_ng pattern as
#   test_cc_dual_pass.py).
# -------------------
import sys
sys.path.insert(0, '/home/josh/NeuroGraph')
import os
import tempfile, shutil
import pytest

from cc_refeed import (content_hash, passes_floor, load_journal, append_journal,
                       collect_orphans, feed_batch, should_pause_for_load)


@pytest.fixture
def cc_ng():
    from openclaw_hook import NeuroGraphMemory
    workspace = tempfile.mkdtemp(prefix='cc_refeed_test_')
    ng = NeuroGraphMemory(workspace_dir=workspace,
                          config={"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}})
    yield ng
    shutil.rmtree(workspace, ignore_errors=True)


@pytest.fixture
def checkpoint_pair(tmp_path):
    """Real msgpack checkpoints: 1 live node, plus vdb entries in every skip
    category and 2 genuine refeed candidates."""
    from neuro_foundation import Graph
    from universal_ingestor import SimpleVectorDB
    import numpy as np

    g = Graph()
    vdb = SimpleVectorDB()
    live_text = "a memory whose node is still alive in the graph today"
    g.create_node(node_id="live_node")
    vdb.insert(id="live_node", embedding=np.random.rand(8), content=live_text)

    cand1 = "the lenia distance cache rebuild took eight hours on the vps graph"
    cand2 = "josh corrected the vdb-primacy framing: the substrate is the memory"
    embodied = "this orphan was already re-embodied by a previous refeed run"
    g.create_node(node_id="cc:conv::" + content_hash(embodied))
    for nid, text in [("orphan_1", cand1), ("orphan_2", cand2),
                      ("orphan_embodied", embodied),
                      ("orphan_short", "too short"),
                      ("orphan_tool", "tool:Read file:/x result: junk that must never re-enter"),
                      ("orphan_degenerate", "3f2a-99d1 7bc0-11aa 55ef-20cd 90ab-44ff 12cd-88ee 34ab-77cc")]:
        vdb.insert(id=nid, embedding=np.random.rand(8), content=text)

    main_path = str(tmp_path / "main.msgpack")
    vectors_path = str(tmp_path / "vectors.msgpack")
    g.checkpoint(main_path)
    vdb.save(vectors_path)
    return main_path, vectors_path, cand1, cand2


def test_passes_floor():
    assert passes_floor("a genuinely substantive ecosystem memory about the lenia cache")
    assert not passes_floor("short")
    assert not passes_floor("tool:Edit file:/x " + "y" * 60)
    assert not passes_floor("bash:ls " + "z" * 60)
    assert not passes_floor("3f2a-99d1 " * 12)          # degenerate: <30% letters
    assert not passes_floor("")


def test_collect_orphans_applies_every_exclusion(checkpoint_pair):
    main_path, vectors_path, cand1, cand2 = checkpoint_pair
    got = collect_orphans(main_path, vectors_path, journal=set())
    contents = [c for _, c in got]
    assert contents == [cand1, cand2], (
        "exactly the two candidates, in vdb order -- live/embodied/short/tool/"
        "degenerate all excluded; got %r" % contents)


def test_collect_orphans_respects_journal(checkpoint_pair):
    main_path, vectors_path, cand1, cand2 = checkpoint_pair
    got = collect_orphans(main_path, vectors_path, journal={content_hash(cand1)})
    assert [c for _, c in got] == [cand2]


def test_journal_roundtrip_and_resume(tmp_path):
    jp = str(tmp_path / "refeed_journal.txt")
    assert load_journal(jp) == set()
    append_journal(jp, ["aaa", "bbb"])
    append_journal(jp, ["ccc"])
    assert load_journal(jp) == {"aaa", "bbb", "ccc"}


def test_should_pause_for_load_thresholds(monkeypatch):
    monkeypatch.setattr(os, "getloadavg", lambda: (100.0, 0, 0))
    assert should_pause_for_load(ceiling=0.75) is True
    monkeypatch.setattr(os, "getloadavg", lambda: (0.0, 0, 0))
    assert should_pause_for_load(ceiling=0.75) is False


def test_feed_batch_absorbed_by_real_drain(cc_ng, tmp_path):
    """The golden end-to-end: feeder frames -> REAL drain_ingest_tract ->
    conversational nodes exist. Proves format + source-filter compatibility
    against the actual production consumer, no mocks."""
    from cc_ng_organism import drain_ingest_tract
    tract_path = str(tmp_path / "turns.tract")
    batch = [(content_hash(t), t) for t in (
        "the ghost commits were condensate cc with compaction amnesia",
        "the harvest orphan seed fix healed syl and cc at once",
    )]
    assert feed_batch(batch, tract_path) == 2

    state = {"last_forest_id": None}
    absorbed = drain_ingest_tract(cc_ng.graph, cc_ng.vector_db, state, tract_path=tract_path)
    assert absorbed == 2
    conv = [n for n in cc_ng.graph.nodes.values()
            if n.metadata.get("creation_mode") == "conversational"]
    assert len(conv) == 2
    # Backstop contract: the re-embodied node ids ARE cc:conv::<sha1> -- what
    # collect_orphans uses to skip already-fed content after journal loss.
    for h, _t in batch:
        assert ("cc:conv::" + h) in cc_ng.graph.nodes
