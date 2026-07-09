# ---- Changelog ----
# [2026-07-09] Claude Code (Fable 5 design / Haiku implementation) — #373 integration tests
# What: real NeuroGraphMemory boots against tmp workspaces: happy path (atomic save +
#   manifest + generation), the 2026-07-08 incident replay (corrupt checkpoint ->
#   provisional -> save quarantined, primary bytes untouched), collapsed-state ratio
#   refusal, and guardian-absent parity (monkeypatched import failure -> today's
#   exact behavior).
# Why: #373's crown jewel — the test that proves the incident that cost ~1800 nodes
#   can never repeat.
# How: NeuroGraphMemory(workspace_dir=tmp, tonic+peer_bridge disabled) — the
#   established real-boot fixture pattern (see tests/test_harvest_orphan_seeds.py).
# -------------------
"""Integration tests: checkpoint_guardian wired into NeuroGraphMemory (#373)."""

import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

TEST_CONFIG = {"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}}


def _boot(workspace):
    from openclaw_hook import NeuroGraphMemory
    return NeuroGraphMemory(workspace_dir=workspace, config=TEST_CONFIG)


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.fixture
def ws():
    d = tempfile.mkdtemp(prefix="guardian_integ_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _grow(ng, n):
    for i in range(n):
        ng.graph.create_node(node_id=f"guardian-test-concept-{i}")


def test_happy_path_save_writes_manifest_and_generation(ws):
    ng = _boot(ws)
    _grow(ng, 120)  # above the gate floor
    path = ng.save()
    ckpt_dir = os.path.join(ws, "checkpoints")
    assert path == os.path.join(ckpt_dir, "main.msgpack")
    manifest = json.load(open(os.path.join(ckpt_dir, "main.msgpack.manifest.json")))
    assert manifest["nodes"] == len(ng.graph.nodes)
    assert manifest["vdb_count"] == ng.vector_db.count()
    gens = os.listdir(os.path.join(ckpt_dir, "generations"))
    assert len(gens) == 1
    gen_files = os.listdir(os.path.join(ckpt_dir, "generations", gens[0]))
    assert "main.msgpack" in gen_files and "vectors.msgpack" in gen_files


def test_incident_20260708_replay_provisional_quarantine(ws, caplog):
    """THE test: a boot against a corrupt checkpoint must never clobber it."""
    ng = _boot(ws)
    _grow(ng, 150)
    primary = ng.save()
    good_hash = _sha(primary)
    manifest_path = primary + ".manifest.json"
    manifest_hash = _sha(manifest_path)

    # Corrupt the checkpoint the way a hard crash does: truncate mid-file.
    good = open(primary, "rb").read()
    open(primary, "wb").write(good[: len(good) // 3])
    corrupt_hash = _sha(primary)

    with caplog.at_level(logging.ERROR):
        ng2 = _boot(ws)  # restore raises inside -> warning -> empty graph
    assert ng2._save_gate is not None and ng2._save_gate.provisional
    assert any("PROVISIONAL" in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.ERROR):
        out = ng2.save()  # the write that destroyed ~1800 nodes on 2026-07-08
    # Primary and manifest untouched; refused state quarantined; loud.
    assert _sha(primary) == corrupt_hash
    assert _sha(manifest_path) == manifest_hash
    assert os.sep + "quarantine" + os.sep in out
    assert os.path.exists(out)
    assert any("REFUSED" in r.getMessage() for r in caplog.records)


def test_collapsed_state_ratio_refusal(ws, caplog):
    ng = _boot(ws)
    _grow(ng, 200)
    primary = ng.save()
    good_hash = _sha(primary)

    # Fresh process, restore SUCCEEDS, but state then collapses in RAM
    # (simulates the empty-daemon-that-somehow-restored-nothing class):
    ng2 = _boot(ws)
    assert not ng2._save_gate.provisional
    for nid in list(ng2.graph.nodes.keys()):
        if not ng2.graph._is_identity_protected(nid):
            ng2.graph.remove_node(nid)
    with caplog.at_level(logging.ERROR):
        out = ng2.save()
    assert _sha(primary) == good_hash
    assert os.sep + "quarantine" + os.sep in out


def test_second_save_after_growth_rotates_second_generation(ws):
    ng = _boot(ws)
    _grow(ng, 120)
    ng.save()
    for i in range(10):
        ng.graph.create_node(node_id=f"guardian-test-concept-second-{i}")
    ng.save()
    gens = os.listdir(os.path.join(ws, "checkpoints", "generations"))
    assert len(gens) == 2


def test_guardian_absent_parity(ws, monkeypatch):
    """With checkpoint_guardian unimportable, save() behaves exactly as today:
    plain in-place checkpoint write, no manifest, no generations, no gate."""
    import openclaw_hook as oh
    monkeypatch.setattr(oh, "_GUARDIAN_AVAILABLE", False)
    ng = oh.NeuroGraphMemory(workspace_dir=ws, config=TEST_CONFIG)
    assert ng._save_gate is None
    _grow(ng, 120)
    path = ng.save()
    ckpt_dir = os.path.join(ws, "checkpoints")
    assert path == os.path.join(ckpt_dir, "main.msgpack")
    assert not os.path.exists(os.path.join(ckpt_dir, "main.msgpack.manifest.json"))
    assert not os.path.exists(os.path.join(ckpt_dir, "generations"))
