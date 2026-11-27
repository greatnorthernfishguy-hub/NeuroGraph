# ---- Changelog ----
# [2026-11-27] DudeMan CC (Claude Fable 5.2, laptop; quest SQ11, punch #1201) -- CREATE
# What: pins the vector DB's two-stage capture: capture_state() == materialize_capture(capture_refs()) and packs to
#       the same bytes; a row inserted or deleted after capture_refs() is not in the materialized capture; with
#       two_stage_capture set, the checkpoint capture runs only capture_refs() while the graph's step lock is held
#       and materializes after it, and the captured set (vectors, counts) equals the one-stage capture; flag off =
#       capture_state() inside the lock as before; the organism's enabler needs CC_NG_VDB_META_SLIM.
# Why:  the vector DB's whole capture held the step lock 4.4 s per save on the live daemon; recall waits on that lock.
# How:  the real SimpleVectorDB and the real NeuroGraphMemory._capture_checkpoint_state; the store's two stage
#       methods are wrapped to record whether the step lock is owned when each runs.
# -------------------
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import tempfile

import msgpack
import numpy as np
import pytest

from universal_ingestor import SimpleVectorDB


def _vdb(n=6):
    v = SimpleVectorDB()
    rng = np.random.default_rng(7)
    for i in range(n):
        v.insert(id=f"n{i}", embedding=rng.standard_normal(16).astype(np.float32),
                 content=f"text {i}", metadata={"creation_mode": "conversational", "k": [i]})
    return v


def test_capture_state_is_the_two_stages_and_packs_to_the_same_bytes():
    v = _vdb()
    one = v.capture_state(detach=True)
    two = v.materialize_capture(v.capture_refs(), detach=True)
    assert msgpack.packb(one, use_bin_type=True) == msgpack.packb(two, use_bin_type=True)
    assert one["count"] == 6 and list(one) == ["version", "count", "entries"]


def test_the_instant_is_capture_refs_not_materialize():
    v = _vdb()
    before = v.capture_state(detach=True)
    refs = v.capture_refs()
    v.insert(id="late", embedding=np.ones(16, dtype=np.float32), content="late", metadata={})
    v.insert(id="n0", embedding=np.zeros(16, dtype=np.float32), content="replaced", metadata={"x": 1})
    v.delete("n1")
    got = v.materialize_capture(refs, detach=True)
    assert msgpack.packb(got, use_bin_type=True) == msgpack.packb(before, use_bin_type=True)


def test_materialize_detaches_metadata():
    v = _vdb()
    got = v.materialize_capture(v.capture_refs(), detach=True)
    assert got["entries"]["n2"]["metadata"] is not v.metadata["n2"]
    assert got["entries"]["n2"]["metadata"]["k"] is not v.metadata["n2"]["k"]


@pytest.fixture
def ng():
    from openclaw_hook import NeuroGraphMemory
    from ng_embed import embed
    ws = tempfile.mkdtemp(prefix='vdb_two_stage_test_')
    m = NeuroGraphMemory(workspace_dir=ws, config={"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}})
    for i in range(4):
        m.graph.create_node(node_id=f"n{i}")
        m.vector_db.insert(id=f"n{i}", embedding=embed(f"a small text number {i}"), content=f"text {i}",
                           metadata={"creation_mode": "conversational"})
    yield m
    shutil.rmtree(ws, ignore_errors=True)


def _spy(ng, monkeypatch):
    """Record (stage name, step lock owned?) for each vector DB capture call."""
    calls = []
    owned = ng.graph._step_lock._is_owned
    v = ng.vector_db
    for name in ("capture_state", "capture_refs", "materialize_capture"):
        real = getattr(v, name)

        def wrap(*a, _real=real, _name=name, **k):
            calls.append((_name, bool(owned())))
            return _real(*a, **k)
        monkeypatch.setattr(v, name, wrap)
    return calls


def test_flag_off_captures_inside_the_lock_as_before(ng, monkeypatch):
    assert getattr(ng.vector_db, "two_stage_capture", False) is False
    calls = _spy(ng, monkeypatch)
    cap = ng._capture_checkpoint_state()
    assert calls[0] == ("capture_state", True)
    assert cap["counts"]["vdb_count"] == cap["vectors"]["count"] == 4


def test_flag_on_takes_only_the_references_inside_the_lock(ng, monkeypatch):
    base = ng._capture_checkpoint_state()
    ng.vector_db.two_stage_capture = True
    calls = _spy(ng, monkeypatch)
    cap = ng._capture_checkpoint_state()
    assert calls == [("capture_refs", True), ("materialize_capture", False)]
    assert not ng.graph._step_lock._is_owned()
    assert msgpack.packb(cap["vectors"], use_bin_type=True) == msgpack.packb(base["vectors"], use_bin_type=True)
    assert cap["counts"] == base["counts"] and set(cap) == set(base)


def test_flag_on_a_failed_materialize_is_reported_as_the_vectors_component(ng, monkeypatch):
    ng.vector_db.two_stage_capture = True

    def boom(*a, **k):
        raise ValueError("no")
    monkeypatch.setattr(ng.vector_db, "materialize_capture", boom)
    with pytest.raises(RuntimeError) as e:
        ng._capture_checkpoint_state()
    assert e.value.checkpoint_component == "vectors" and not ng.graph._step_lock._is_owned()


def test_flag_on_a_full_save_writes_the_same_vector_file(ng):
    ng.save()
    one = open(ng._vector_db_path, "rb").read()
    ng.vector_db.two_stage_capture = True
    ng.save()
    assert open(ng._vector_db_path, "rb").read() == one


def test_enabler_needs_the_env_and_the_slim_metadata_switch(monkeypatch):
    import cc_ng_organism as org
    v = _vdb(2)
    monkeypatch.setattr(org, "VDB_CAPTURE_SPLIT", False)
    assert org.cc_enable_vdb_two_stage_capture(v) is False and v.two_stage_capture is False
    monkeypatch.setattr(org, "VDB_CAPTURE_SPLIT", True)
    monkeypatch.setattr(org, "VDB_META_SLIM", False)
    with pytest.raises(RuntimeError):
        org.cc_enable_vdb_two_stage_capture(v)
    assert v.two_stage_capture is False
    monkeypatch.setattr(org, "VDB_META_SLIM", True)
    assert org.cc_enable_vdb_two_stage_capture(v) is True and v.two_stage_capture is True
    assert org.cc_enable_vdb_two_stage_capture(None) is False


def test_a_misconfigured_capture_switch_does_not_block_the_native_scan(monkeypatch):
    import cc_ng_organism as org
    v = _vdb(2)
    monkeypatch.setattr(org, "VDB_CAPTURE_SPLIT", True)
    monkeypatch.setattr(org, "VDB_META_SLIM", False)
    monkeypatch.setattr(org, "VDB_NATIVE_SCAN", False)
    assert org.cc_enable_native_vdb_scan(v) is False        # no raise
    assert v.two_stage_capture is False
