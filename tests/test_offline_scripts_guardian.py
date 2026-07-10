# ---- Changelog ----
# [2026-07-11] Claude Code (Fable 5) — #379: offline scripts write atomically + refresh manifest
# What: rebaseline and cleanup apply-paths produce a valid checkpoint, a manifest with
#   post-pass counts + offline_pass stamp, and leave no tmp litter.
# Why: #379 — offline in-place writes tore/mutated the newest guardian generation and
#   left stale manifests that could false-trip the SaveGate.
# How: real Graph + SimpleVectorDB on tmp paths (no-mocks convention).
# -------------------
"""Tests for #379 — offline surgery scripts honor the checkpoint guardian."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkpoint_guardian import read_manifest
from neuro_foundation import Graph
from universal_ingestor import SimpleVectorDB


def _seed_checkpoint(tmp_path, n=6):
    g = Graph()
    for i in range(n):
        g.create_node(node_id=f"n{i}")
        g.nodes[f"n{i}"].threshold = 1.2  # untuned-era, above 0.85
    main = str(tmp_path / "main.msgpack")
    g.checkpoint(main)
    return main


def test_rebaseline_apply_writes_manifest_atomically(tmp_path):
    from cc_threshold_rebaseline import rebaseline
    main = _seed_checkpoint(tmp_path)
    out = rebaseline(main, apply=True)
    assert out["status"] == "ok" and out["changed"] == 6
    m = read_manifest(main)
    assert m is not None and m["nodes"] == 6
    assert m["offline_pass"] == "cc_threshold_rebaseline"
    # checkpoint still loadable, thresholds applied
    g2 = Graph(); g2.restore(main)
    assert all(nd.threshold <= 0.86 for nd in g2.nodes.values())
    assert not [f for f in os.listdir(tmp_path) if ".tmp-" in f], "no tmp litter"


def test_cleanup_apply_writes_both_atomically_and_refreshes_manifest(tmp_path):
    from cleanup_cc_tool_noise import cleanup
    main = _seed_checkpoint(tmp_path)
    vectors = str(tmp_path / "vectors.msgpack")
    vdb = SimpleVectorDB()
    import numpy as np
    vdb.insert("n0", np.ones(8, dtype=np.float32), content="genuine memory", metadata={})
    vdb.save(vectors)
    out = cleanup(main, vectors, apply=True)
    assert out["status"] == "ok"
    m = read_manifest(main)
    assert m is not None and m["offline_pass"] == "cleanup_cc_tool_noise"
    assert m["nodes"] == 6 and m["vdb_count"] == 1
    g2 = Graph(); g2.restore(main)
    assert len(g2.nodes) == 6
    v2 = SimpleVectorDB(); v2.load(vectors)
    assert v2.count() == 1
    assert not [f for f in os.listdir(tmp_path) if ".tmp-" in f], "no tmp litter"
