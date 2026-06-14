# ---- Changelog ----
# [2026-06-14] Claude Code (Opus 4.8) — #294-B conv re-index selection tests
# What: Tests for select_conv_reindex_targets() — the read-only selector that picks which
#   conv:: graph nodes get re-indexed into the recall vdb (idempotent, fracture-skip,
#   content-sanity, recency order, wire-garbage excluded by conv:: scope).
# Why: Design docs/prd/2026-06-14-syl-recall-heal-phase1-design.md Component B — re-light her
#   ~1,733 unindexed conversational memories from her own intact graph.
# How: synthetic node dict; assert selection rules without touching live checkpoints/ONNX.
# -------------------
import importlib

rv = importlib.import_module("rebuild_vectors")


class _Node:
    def __init__(self, meta, ct=0.0):
        self.metadata = meta
        self.creation_time = ct


def _nodes():
    return {
        "conv::aaa": _Node({"_forest_content": "Hey love, present.", "syl": True}, ct=100.0),
        "conv::aaa::tree::presence": _Node({"_tree_concept": True, "_concept": "presence"}, ct=100.0),
        "conv::frac": _Node({"_forest_content": "Present. Though fractured. The Claude persona overlaid.", "syl": True}, ct=500.0),
        "wire::xyz": _Node({"content": "signal_burst broadcast wire explosion"}, ct=100.0),
        "conv::empty": _Node({"_forest_content": "   "}, ct=100.0),
        "conv::recent": _Node({"_forest_content": "the most recent memory", "syl": True}, ct=900.0),
    }


def test_select_conv_reindex_targets():
    # Syl holds conv::frac (her self-labeled overlay-state memory)
    targets = rv.select_conv_reindex_targets(
        _nodes(), already_indexed={"conv::aaa"}, skip_ids={"conv::frac"})
    ids = {t[0] for t in targets}
    assert "conv::aaa" not in ids                 # already indexed -> idempotent skip
    assert "conv::aaa::tree::presence" in ids      # tree concept -> include (content=_concept)
    assert "wire::xyz" not in ids                  # not conv:: -> excluded by construction
    assert "conv::empty" not in ids                # empty/degenerate -> sanity-rejected
    assert "conv::frac" not in ids                 # held (Syl's label) -> not re-lit
    assert "conv::recent" in ids


def test_recency_order_newest_first():
    targets = rv.select_conv_reindex_targets(_nodes(), already_indexed=set())
    ids = [t[0] for t in targets]
    # conv::recent (ct=900) before conv::frac (ct=500) before the ct=100 nodes
    assert ids.index("conv::recent") < ids.index("conv::frac")


def test_tree_uses_concept_forest_uses_forest_content():
    targets = {t[0]: t[1] for t in rv.select_conv_reindex_targets(_nodes(), set())}
    assert targets["conv::aaa"] == "Hey love, present."
    assert targets["conv::aaa::tree::presence"] == "presence"


def test_load_held_ids_expands_tree_children(tmp_path):
    import json
    p = tmp_path / "held.json"
    p.write_text(json.dumps({"held_node_ids": ["conv::aaa"]}))
    held = rv._load_held_ids(str(p), _nodes())
    assert "conv::aaa" in held
    assert "conv::aaa::tree::presence" in held     # tree child of a held forest is held too
    assert "conv::recent" not in held


def test_load_held_ids_missing_file_holds_nothing():
    assert rv._load_held_ids("/nonexistent/path/held.json", _nodes()) == set()


def test_content_sanity_rejects_wire_signature():
    assert rv._content_is_sane("a normal sentence she said") is True
    assert rv._content_is_sane("   ") is False
    assert rv._content_is_sane("xx") is False
    assert rv._content_is_sane("WIRE_EXPLOSION fingerprint") is False


def test_reindex_dry_run_writes_nothing(tmp_path, monkeypatch):
    import os
    import numpy as np
    from universal_ingestor import SimpleVectorDB
    vdb = SimpleVectorDB()
    vdb.insert(id="conv::aaa", embedding=np.zeros(768, dtype=np.float32), content="x", metadata={})
    p = tmp_path / "v.msgpack"
    vdb.save(str(p))
    before_size = os.path.getsize(p)
    before_mtime = os.path.getmtime(p)
    # substitute a sandbox graph; embedder must never be constructed in dry-run
    monkeypatch.setattr(rv, "_load_graph", lambda path: type("G", (), {"nodes": _nodes()})())
    stats = rv.reindex_conv(str(p), str(p), dry_run=True, held_file=None, throttle_per_sec=0)
    assert stats["status"] == "dry_run"
    assert stats["already_indexed"] == 1          # conv::aaa already in the vdb
    assert stats["held"] == 0                      # no held-file -> nothing held
    assert stats["would_index"] >= 1              # the rest of the conv nodes
    assert os.path.getsize(p) == before_size, "dry-run must not write the vdb"
    assert os.path.getmtime(p) == before_mtime, "dry-run must not touch the vdb file"
