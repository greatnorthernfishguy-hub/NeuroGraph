# ---- Changelog ----
# [2026-10-14] Claude Code (Opus 5.5, lane dim-guard) — #1097: the dimension guard, on both NG-Lite cores
# What: Proves that a wrong embedding dimension fails LOUD (EmbeddingDimensionError / VectorDimensionError,
#   naming the file and both dimensions, logged at ERROR once) everywhere NG-Lite and SimpleVectorDB take
#   vectors in: inputs, the constitutional rim, loaded states (msgpack, legacy JSON, JSON-in-msgpack) and
#   vector-store files — and that correct-dim data is unaffected. Pins ng_lite's default embedding_dim to
#   ng_embed's EMBEDDING_DIM so the two canonical files cannot drift apart again.
# Why: #1097 (Elmer's 384-d rim swallowed silently for weeks); Josh 2026-10-14 — make it structural.
# How: pytest, both cores via the NG_LITE_RUST_CORE fixture pattern of test_commons_rustcore_1096.py.
# -------------------
import json
import logging
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ng_lite  # noqa: E402
from ng_lite import EmbeddingDimensionError, NGLite  # noqa: E402

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
    ng_lite._DIM_ERRORS_LOGGED.clear()
    return request.param


def _emb(seed, dim=768):
    v = np.random.RandomState(seed).randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _rim(dim, n=3):
    return [{"category": f"cat{i}", "description": f"rim {i}", "embedding": _emb(100 + i, dim).tolist()}
            for i in range(n)]


def _ng(core, **cfg):
    ng = NGLite(module_id="dimtest", config=cfg or None)
    assert (ng._core is not None) == (core == "rust")
    return ng


# ---------------------------------------------------------------- the single source of truth

def test_default_dim_is_ng_embeds():
    import ng_embed
    assert ng_lite.DEFAULT_CONFIG["embedding_dim"] == ng_embed.EMBEDDING_DIM == 768
    assert ng_embed.MODEL_NAME == ng_embed._DEFAULT_CONFIG["model_id"]


# ---------------------------------------------------------------- inputs

@pytest.mark.parametrize("call", ["record_outcome", "get_recommendations", "detect_novelty", "find_or_create_node"])
def test_wrong_dim_input_raises_and_logs_once(core, call, caplog):
    ng = _ng(core)
    ng.record_outcome(_emb(1), "t", True)
    bad = _emb(2, 384)
    fn = {
        "record_outcome": lambda: ng.record_outcome(bad, "t", True),
        "get_recommendations": lambda: ng.get_recommendations(bad),
        "detect_novelty": lambda: ng.detect_novelty(bad),
        "find_or_create_node": lambda: ng.find_or_create_node(bad),
    }[call]
    with caplog.at_level(logging.ERROR, logger="ng_lite"):
        for _ in range(3):
            with pytest.raises(EmbeddingDimensionError) as ei:
                fn()
    msg = str(ei.value)
    assert "384-d" in msg and "768-d" in msg and call in msg and "dimtest" in msg
    assert isinstance(ei.value, ValueError)  # existing handlers still catch it
    errors = [r for r in caplog.records if r.levelno == logging.ERROR and "384-d" in r.getMessage()]
    assert len(errors) == 1  # loud once, not a log flood
    assert ng.node_count() == 1  # nothing was created by the refused calls


def test_correct_dim_unaffected(core):
    ng = _ng(core)
    for i in range(20):
        ng.record_outcome(_emb(i), f"t{i % 3}", i % 2 == 0)
    assert ng.get_recommendations(_emb(4))
    assert 0.0 <= ng.detect_novelty(_emb(999)) <= 1.0


# ---------------------------------------------------------------- the rim

def test_wrong_dim_rim_refused_at_construction(core, caplog):
    with caplog.at_level(logging.ERROR, logger="ng_lite"):
        with pytest.raises(EmbeddingDimensionError) as ei:
            _ng(core, constitutional_embeddings=_rim(384))
    msg = str(ei.value)
    assert "constitutional rim" in msg and "384-d" in msg and "768-d" in msg and "3 of 3 entries" in msg
    assert "'cat0'" in msg
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_correct_rim_seeds(core):
    ng = _ng(core, constitutional_embeddings=_rim(768))
    assert ng.node_count() == 3


# ---------------------------------------------------------------- loaded states

def _state_384(tmp_path):
    """A real NG-Lite state whose vectors are 384-d (written by an NG-Lite configured for 384)."""
    old = NGLite(module_id="dimtest", config={"embedding_dim": 384, "receptor_layer_enabled": False})
    for i in range(5):
        old.record_outcome(_emb(i, 384), "t", True)
    return old


def test_wrong_dim_msgpack_state_refused_and_not_loaded(core, tmp_path, monkeypatch):
    monkeypatch.delenv("NG_LITE_RUST_CORE", raising=False)
    path = _state_384(tmp_path).save(str(tmp_path / "s.msgpack"))
    if core == "rust":
        monkeypatch.setenv("NG_LITE_RUST_CORE", "1")
    ng = _ng(core)
    with pytest.raises(EmbeddingDimensionError) as ei:
        ng.load(path)
    msg = str(ei.value)
    assert path in msg and "384-d x5" in msg and "768-d" in msg
    assert ng.node_count() == 0  # nothing half-loaded


def test_wrong_dim_legacy_json_refused_and_kept(core, tmp_path, monkeypatch):
    monkeypatch.delenv("NG_LITE_RUST_CORE", raising=False)
    st = _state_384(tmp_path).export_state()
    legacy = tmp_path / "s.json"
    legacy.write_text(json.dumps(st))
    if core == "rust":
        monkeypatch.setenv("NG_LITE_RUST_CORE", "1")
    ng = _ng(core)
    with pytest.raises(EmbeddingDimensionError) as ei:
        ng.load(str(legacy))
    assert str(legacy) in str(ei.value)
    assert legacy.exists() and not (tmp_path / "s.msgpack").exists()  # no migration of a refused state


def test_json_inside_msgpack_refused(core, tmp_path, monkeypatch):
    monkeypatch.delenv("NG_LITE_RUST_CORE", raising=False)
    st = _state_384(tmp_path).export_state()
    p = tmp_path / "commons.msgpack"
    p.write_text(json.dumps(st))
    if core == "rust":
        monkeypatch.setenv("NG_LITE_RUST_CORE", "1")
    with pytest.raises(EmbeddingDimensionError):
        _ng(core).load(str(p))
    assert not (tmp_path / "commons.msgpack.legacy-json").exists()


def test_saved_config_cannot_override_embedding_dim(core, tmp_path, monkeypatch):
    """A state whose config claims 384 but whose vectors are 768 loads; the instance stays 768."""
    monkeypatch.delenv("NG_LITE_RUST_CORE", raising=False)
    src = NGLite(module_id="dimtest")
    for i in range(4):
        src.record_outcome(_emb(i), "t", True)
    st = src.export_state()
    st["config"]["embedding_dim"] = 384
    p = tmp_path / "s.json"
    p.write_text(json.dumps(st))
    if core == "rust":
        monkeypatch.setenv("NG_LITE_RUST_CORE", "1")
    ng = _ng(core)
    assert ng.load(str(p))
    assert ng.embedding_dim == 768 and ng.config["embedding_dim"] == 768
    assert ng.node_count() == 4
    with pytest.raises(EmbeddingDimensionError):
        ng.record_outcome(_emb(9, 384), "t", True)


def test_wrong_dim_prototypes_refused(core, tmp_path, monkeypatch):
    monkeypatch.delenv("NG_LITE_RUST_CORE", raising=False)
    src = NGLite(module_id="dimtest")
    src.record_outcome(_emb(1), "t", True)
    st = src.export_state()
    st["receptor_layer"] = {"prototypes": [_emb(i, 384).tolist() for i in range(4)],
                            "prototype_counts": [0, 0, 0, 0], "input_count": 300}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(st))
    if core == "rust":
        monkeypatch.setenv("NG_LITE_RUST_CORE", "1")
    with pytest.raises(EmbeddingDimensionError) as ei:
        _ng(core).load(str(p))
    assert "receptor prototypes by dim: 384-d x4" in str(ei.value)


def test_saved_rim_checked_only_when_constructor_has_none(core, tmp_path, monkeypatch):
    monkeypatch.delenv("NG_LITE_RUST_CORE", raising=False)
    src = NGLite(module_id="dimtest")
    src.record_outcome(_emb(1), "t", True)
    st = src.export_state()
    st["config"]["constitutional_embeddings"] = _rim(384)
    p = tmp_path / "s.json"
    p.write_text(json.dumps(st))
    if core == "rust":
        monkeypatch.setenv("NG_LITE_RUST_CORE", "1")
    with pytest.raises(EmbeddingDimensionError) as ei:
        _ng(core).load(str(p))
    assert "saved config" in str(ei.value)


# ---------------------------------------------------------------- SimpleVectorDB

def test_vdb_load_refuses_wrong_and_mixed_dims(tmp_path):
    from universal_ingestor import SimpleVectorDB, VectorDimensionError
    src = SimpleVectorDB()
    for i in range(4):
        src.insert(f"a{i}", _emb(i, 384), f"c{i}")
    p = str(tmp_path / "v.msgpack")
    src.save(p)
    live = SimpleVectorDB(dim=768)
    live.insert("keep", _emb(1), "kept")
    with pytest.raises(VectorDimensionError) as ei:
        live.load(p)
    assert p in str(ei.value) and "384-d x4" in str(ei.value) and "768-d" in str(ei.value)
    assert live.count() == 1 and live.get_content("keep") == "kept"  # the live store was not replaced
    # mixed dims in one file are refused even with no explicit dim
    src.embeddings["b"] = _emb(9, 768)
    src.content["b"] = ""
    src.metadata["b"] = {}
    src.save(p)
    with pytest.raises(VectorDimensionError) as ei:
        SimpleVectorDB().load(p)
    assert "mixed" in str(ei.value)


def test_vdb_insert_and_search_refuse_other_dims(tmp_path):
    from universal_ingestor import SimpleVectorDB, VectorDimensionError
    db = SimpleVectorDB()
    db.insert("x", _emb(1), "x")
    with pytest.raises(VectorDimensionError):
        db.insert("y", _emb(2, 384), "y")
    with pytest.raises(VectorDimensionError):
        db.search(_emb(3, 384))
    assert db.search(_emb(1), k=1, threshold=0.5)[0][0] == "x"
    p = str(tmp_path / "ok.msgpack")
    db.save(p)
    again = SimpleVectorDB(dim=768)
    assert again.load(p) == 1
