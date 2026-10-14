# ---- Changelog ----
# [2026-10-13] Claude Code (Opus 5.5, lane nglite-rustcore) — #1096 / #1098: the Commons on both NG-Lite cores
# What: The Commons verbs that used to read NGLite internals (bucket_recent incl. metadata / embedding modes, the
#   metric / error / wire recency windows, arousal, the autonomic drop on restore) behave the same on the Python
#   core and on the Rust NGLiteCore (NG_LITE_RUST_CORE=1); persist() writes msgpack (#1098) and restore() converts
#   the old JSON-in-commons.msgpack once.
# Why: Josh 2026-10-13 rulings on #1096 — consumers read through NGLite's public methods so the Rust core can hold
#   the state; saves are msgpack, never JSON.
# How: each test runs per core (the Rust leg is skipped when the installed ng_tract has no _NGLiteCore).
# -------------------
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commons as commons_mod  # noqa: E402

try:
    import ng_tract
    _HAS_CORE = hasattr(ng_tract, "_NGLiteCore")
except ImportError:  # pragma: no cover
    ng_tract = None
    _HAS_CORE = False


@pytest.fixture(params=["python", "rust"])
def core(request, monkeypatch):
    if request.param == "rust":
        if not _HAS_CORE:
            pytest.skip("ng_tract._NGLiteCore not installed")
        monkeypatch.setenv("NG_LITE_RUST_CORE", "1")
        monkeypatch.setattr(ng_tract, "NGLiteCore", ng_tract._NGLiteCore, raising=False)
    else:
        monkeypatch.delenv("NG_LITE_RUST_CORE", raising=False)
    return request.param


def _emb(seed, dim=768):
    r = np.random.RandomState(seed)
    v = r.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _new(core):
    c = commons_mod.Commons()
    assert (c._ng._core is not None) == (core == "rust")
    return c


def _targets(c, prefix):
    return [s.target_id for s in c._ng.list_synapses(target_prefix=prefix, with_metadata=False)]


def test_bucket_recent_rows_metadata_and_embedding(core):
    c = _new(core)
    for i in range(30):
        c.deposit(_emb(i), f"experience:{i}", metadata={"user_text": f"u{i}"})
    rows = c.bucket_recent(limit=10)
    assert [r[0] for r in rows] == [f"experience:{i}" for i in range(29, 19, -1)]
    rows4 = c.bucket_recent(limit=5, with_metadata=True)
    assert [r[3] for r in rows4] == [{"user_text": f"u{i}"} for i in range(29, 24, -1)]
    rows5 = c.bucket_recent(limit=3, with_embedding=True)
    for r, i in zip(rows5, (29, 28, 27)):
        assert len(r) == 5 and r[4] is not None
        assert float(np.dot(r[4], _emb(i))) > 0.999  # the deposit's own (normalized) vector
    # watermark: only deposits newer than `since`
    since = c._ng.get_synapse(c._ng.list_synapses(target_id="experience:24")[0].source_id,
                              "experience:24").last_updated
    assert [r[0] for r in c.bucket_recent(limit=50, since=since)] == [f"experience:{i}" for i in range(29, 24, -1)]
    # suppression still honoured
    c.suppress("experience:29", "soft")
    assert c.bucket_recent(limit=1)[0][0] == "experience:28"


def test_metric_error_wire_windows(core):
    c = _new(core)
    for i in range(commons_mod._METRICS_KEEP_PER_KIND + 15):
        c.deposit(_emb(1000 + i), f"metrics:elmer:cpu:{i}")
    for i in range(commons_mod._ERRORS_KEEP_PER_KIND + 5):
        c.deposit(_emb(5000 + i), f"error:elmer:ValueError:{i}")
    for i in range(commons_mod._WIRE_KEEP_PER_DIR + 7):
        c.deposit(_emb(9000 + i), f"wire:tid.http.outbound:h{i}")
    c.deposit(_emb(1), "experience:keep")
    m = _targets(c, "metrics:elmer:cpu:")
    assert len(m) == commons_mod._METRICS_KEEP_PER_KIND
    assert min(int(t.rsplit(":", 1)[1]) for t in m) == 15  # the oldest went
    assert len(_targets(c, "error:elmer:ValueError:")) == commons_mod._ERRORS_KEEP_PER_KIND
    assert len(_targets(c, "wire:tid.http.outbound:")) == commons_mod._WIRE_KEEP_PER_DIR
    assert _targets(c, "experience:") == ["experience:keep"]


def test_arousal_and_autonomic_drop_on_restore(core, tmp_path):
    c = _new(core)
    c.deposit(_emb(7), "autonomic:arousal", metadata={"state": "SYMPATHETIC", "threat_level": "high"})
    c.deposit(_emb(8), "experience:x", metadata={"t": 1})
    assert c.read_arousal() == "SYMPATHETIC"
    assert c.arousal()["threat_level"] == "high"
    p = str(tmp_path / "commons.msgpack")
    c.persist(p)
    with open(p, "rb") as f:
        assert f.read(1) != b"{"  # #1098: real msgpack now
    c2 = _new(core)
    c2.restore(p)
    assert _targets(c2, "autonomic:") == []          # fresh-assess on restart (#328)
    assert c2.read_arousal() == "PARASYMPATHETIC"
    assert _targets(c2, "experience:") == ["experience:x"]


def test_restore_converts_the_old_json_commons_once(core, tmp_path):
    # what the old Python core wrote into commons.msgpack: JSON (#1098)
    src = commons_mod.Commons()
    for i in range(12):
        src.deposit(_emb(200 + i), f"experience:{i}", metadata={"i": i})
    legacy = src._ng.export_state()
    p = tmp_path / "commons.msgpack"
    p.write_text(json.dumps(legacy, indent=2))
    c = _new(core)
    c.restore(str(p))
    with open(p, "rb") as f:
        assert f.read(1) != b"{"
    assert (tmp_path / "commons.msgpack.legacy-json").exists()
    assert [r[0] for r in c.bucket_recent(limit=3)] == ["experience:11", "experience:10", "experience:9"]
    c2 = _new(core)
    c2.restore(str(p))  # second restore: msgpack, nothing to convert
    assert c2._ng.synapse_count() == c._ng.synapse_count() == 12


def test_bucket_is_read_only_on_rust(core):
    c = _new(core)
    for i in range(10):
        c.deposit(_emb(300 + i), f"experience:{i}")
    n0 = c._ng.node_count()
    for i in range(10):
        c.bucket(_emb(900 + i), top_k=3)  # unknown queries
    if core == "rust":
        assert c._ng.node_count() == n0  # quirk fix (c): a query writes nothing
    else:
        assert c._ng.node_count() >= n0  # the Python core still creates nodes on a query (unchanged)
