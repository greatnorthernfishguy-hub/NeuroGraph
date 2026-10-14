# ---- Changelog ----
# [2026-10-14] Claude Code (Opus 5.5, lane mean-centre) — new file: #1107 mean-centred NG-Lite
# What: Pins the shipped centre (version, sha256, dim) to ng_embed; proves NG-Lite centres every comparison (node
#   match, novelty, receptor snap) with stored vectors kept raw; the rim threshold and its priority; the
#   embedding_space tag in saves and the load checks (another model refused, an off-manifold untagged state refused, a
#   raw-era untagged state loads with its space thresholds reset, a tagged state keeps tuned values); an off-manifold
#   rim refused; the repo's rim file is Snowflake (on the manifold). Both cores where ng_tract._NGLiteCore exists.
# Why: Josh 2026-10-14 "Mean center" (#1107); DIM_GUARD.md §4 / §11.
# How: synthetic vectors on the model's manifold (ng_embed's centre + a spread of |d| ~ 0.6, like real Snowflake
#   vectors: cosine to the centre direction ~0.8) — no model download needed.
# -------------------
import hashlib
import json
import logging
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ng_embed  # noqa: E402
import ng_lite  # noqa: E402
from ng_lite import EmbeddingSpaceError, NGLite  # noqa: E402

try:
    import ng_tract
    _HAS_CORE = hasattr(ng_tract, "_NGLiteCore") and hasattr(ng_tract._NGLiteCore, "set_space")
except ImportError:  # pragma: no cover
    ng_tract = None
    _HAS_CORE = False

MU = np.asarray(ng_embed.embedding_centre(), np.float32)
D = MU.shape[0]


@pytest.fixture(params=["python", "rust"])
def core(request, monkeypatch):
    if request.param == "rust":
        if not _HAS_CORE:
            pytest.skip("ng_tract._NGLiteCore with set_space (#1107) not installed")
        monkeypatch.setenv("NG_LITE_RUST_CORE", "1")
        monkeypatch.setattr(ng_tract, "NGLiteCore", ng_tract._NGLiteCore, raising=False)
    else:
        monkeypatch.delenv("NG_LITE_RUST_CORE", raising=False)
    return request.param


def snow(rng, base=None, spread=0.6):
    b = MU if base is None else base
    return (b + rng.standard_normal(D).astype(np.float32) * np.float32(spread / math.sqrt(D))).astype(np.float32)


def centred_cos(a, b):
    ca = a / np.linalg.norm(a) - MU
    cb = b / np.linalg.norm(b) - MU
    return float(ca @ cb / np.linalg.norm(ca) / np.linalg.norm(cb))


def at_centred_cos(rng, ref, target):
    r = ref / np.linalg.norm(ref)
    rc = r - MU
    rc /= np.linalg.norm(rc)
    o = rng.standard_normal(D).astype(np.float32)
    o -= (o @ rc) * rc
    o /= np.linalg.norm(o)
    c = target * rc + math.sqrt(1 - target * target) * o
    return (c * np.float32(np.linalg.norm(r - MU)) + MU).astype(np.float32)


def rim_entry(v, d="rim"):
    return {"category": "c", "description": d, "embedding": [float(x) for x in v]}


# ---------------------------------------------------------------- the artifact

def test_shipped_centre_is_the_versioned_artifact():
    raw = MU.astype("<f4").tobytes()
    assert MU.shape == (ng_embed.EMBEDDING_DIM,)
    assert hashlib.sha256(raw).hexdigest() == ng_embed.CENTRE_SHA256
    assert ng_embed.CENTRE_VERSION.startswith("snowflake-arctic-embed-m-v1.5/")
    assert not ng_embed.embedding_centre().flags.writeable
    assert 0.79 < float(np.linalg.norm(MU)) < 0.81


def test_ng_lite_resolves_the_space(core):
    ng = NGLite(module_id="mc")
    assert ng._model == ng_embed.MODEL_NAME and ng._centre_version == ng_embed.CENTRE_VERSION
    assert ng.get_stats()["embedding_space"] == {"model": ng_embed.MODEL_NAME, "dim": D,
                                                 "centre_version": ng_embed.CENTRE_VERSION}
    assert NGLite(module_id="mc", config={"mean_centre": False})._centre is None
    assert NGLite(module_id="mc", config={"embedding_dim": 8})._centre is None


def test_thresholds_and_bounds():
    c = ng_lite.DEFAULT_CONFIG
    assert c["novelty_threshold"] == 0.40 and c["constitutional_threshold"] == 0.55
    assert c["receptor_prototype_threshold"] == 0.75 and c["mean_centre"] is True
    assert NGLite.TUNABLE_PARAMS["novelty_threshold"] == (0.20, 0.60)
    assert "constitutional_threshold" not in NGLite.TUNABLE_PARAMS  # the rim is not runtime-tunable


# ---------------------------------------------------------------- centred matching

def test_unrelated_model_vectors_no_longer_collapse(core):
    """Raw cosine between two unrelated on-manifold vectors is ~0.6 (the old snap matched); centred it is ~0."""
    rng = np.random.default_rng(1)
    vs = [snow(rng) for _ in range(40)]
    raw = [float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)) for a, b in zip(vs, vs[1:])]
    assert min(raw) > 0.5
    ng = NGLite(module_id="mc", config={"receptor_layer_enabled": False})
    ids = {ng.find_or_create_node(v).node_id for v in vs}
    assert len(ids) == 40 and ng.node_count() == 40
    old = NGLite(module_id="mc", config={"receptor_layer_enabled": False, "mean_centre": False,
                                         "novelty_threshold": 0.7})
    assert len({old.find_or_create_node(v).node_id for v in vs}) == 1  # the raw-era collapse, for contrast


def test_match_at_threshold_and_raw_storage(core):
    rng = np.random.default_rng(2)
    base = snow(rng)
    ng = NGLite(module_id="mc", config={"receptor_layer_enabled": False})
    n0 = ng.find_or_create_node(base)
    thr = 1.0 - ng.config["novelty_threshold"]
    above = at_centred_cos(rng, base, thr + 0.05)
    below = at_centred_cos(rng, base, thr - 0.05)
    assert ng.find_or_create_node(above).node_id == n0.node_id
    assert ng.find_or_create_node(below).node_id != n0.node_id
    stored = ng.get_node(n0.node_id).embedding
    assert np.allclose(stored, base / np.linalg.norm(base), atol=1e-6)  # raw (normalised), not centred


def test_novelty_is_one_minus_best_centred_cosine(core):
    rng = np.random.default_rng(3)
    ng = NGLite(module_id="mc", config={"receptor_layer_enabled": False})
    stored = [snow(rng) for _ in range(10)]
    for v in stored:
        ng.find_or_create_node(v)
    q = snow(rng)
    want = 1.0 - max(0.0, max(centred_cos(q, s) for s in stored))
    assert abs(ng.detect_novelty(q) - want) < 1e-5


# ---------------------------------------------------------------- the rim

def test_rim_threshold_and_priority(core):
    rng = np.random.default_rng(4)
    rv = snow(rng)
    thr = ng_lite.DEFAULT_CONFIG["constitutional_threshold"]
    ng = NGLite(module_id="mc", config={"receptor_layer_enabled": False,
                                        "constitutional_embeddings": [rim_entry(rv)]})
    assert ng.find_or_create_node(at_centred_cos(rng, rv, thr + 0.03)).constitutional is True
    near = ng.find_or_create_node(at_centred_cos(rng, rv, thr - 0.03))
    assert near.constitutional is False
    r = ng.record_outcome(at_centred_cos(rng, rv, 0.9), "x", True)
    assert r.get("constitutional") is True  # caught at 0.9 even with an ordinary node nearby
    assert ng.get_recommendations(at_centred_cos(rng, rv, 0.9)) == []


def test_off_manifold_rim_is_refused_loud(core, caplog):
    rng = np.random.default_rng(5)
    iso = rng.standard_normal(D).astype(np.float32)
    with caplog.at_level(logging.ERROR, logger="ng_lite"):
        with pytest.raises(EmbeddingSpaceError, match="constitutional rim"):
            NGLite(module_id="mc_rim", config={"constitutional_embeddings": [rim_entry(iso)]})
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_repo_rim_file_is_the_snowflake_rim_on_the_manifold(core):
    path = os.path.join(ROOT, "data", "constitutional_embeddings.json")
    data = json.load(open(path))
    assert data["model"] == ng_embed.MODEL_NAME and data["embedding_dim"] == D
    assert data["reembedded_from"]["model"] == "BAAI/bge-base-en-v1.5"
    ng = NGLite(module_id="mc", config={"constitutional_embeddings": data["embeddings"]})
    assert sum(1 for n in ng.list_nodes(with_embedding=False) if n.constitutional) == 22
    for e in data["embeddings"]:  # each entry is caught by itself
        assert ng.find_or_create_node(np.asarray(e["embedding"], np.float32)).constitutional


# ---------------------------------------------------------------- the space tag

def _state_with(ng):
    return ng.export_state()


def test_space_tag_and_load_checks(core, tmp_path, caplog):
    rng = np.random.default_rng(6)
    src = NGLite(module_id="mc", config={"receptor_layer_enabled": False})
    for _ in range(12):
        src.record_outcome(snow(rng), "t", True)
    st = src.export_state()
    assert st["embedding_space"]["model"] == ng_embed.MODEL_NAME
    # another model: refused, nothing loaded
    bad = json.loads(json.dumps(st))
    bad["embedding_space"]["model"] = "BAAI/bge-base-en-v1.5"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    ng = NGLite(module_id="mc")
    with pytest.raises(EmbeddingSpaceError, match="records model"):
        ng.load(str(p))
    assert ng.node_count() == 0
    # untagged + isotropic noise (hash-fallback-like): refused
    noise = json.loads(json.dumps(st))
    noise.pop("embedding_space")
    for nd in noise["nodes"].values():
        nd["embedding"] = [float(x) for x in rng.standard_normal(D)]
    p = tmp_path / "noise.json"
    p.write_text(json.dumps(noise))
    with pytest.raises(EmbeddingSpaceError, match="untagged"):
        NGLite(module_id="mc").load(str(p))
    # untagged raw-era state on the manifold: loads, thresholds follow this instance, the save is tagged
    old = json.loads(json.dumps(st))
    old.pop("embedding_space")
    old["config"]["novelty_threshold"] = 0.7
    p = tmp_path / "old.json"
    p.write_text(json.dumps(old))
    ng = NGLite(module_id="mc")
    with caplog.at_level(logging.WARNING, logger="ng_lite"):
        assert ng.load(str(p))
    assert ng.config["novelty_threshold"] == ng_lite.DEFAULT_CONFIG["novelty_threshold"]
    assert any("space thresholds reset" in r.getMessage() for r in caplog.records)
    out = ng.save(str(tmp_path / "old2.msgpack"))
    ng2 = NGLite(module_id="mc")
    assert ng2.load(out) and ng2.export_state()["embedding_space"]["centre_version"] == ng_embed.CENTRE_VERSION
    # a tagged state keeps a tuned value
    tuned = json.loads(json.dumps(st))
    tuned["config"]["novelty_threshold"] = 0.5
    p = tmp_path / "tuned.json"
    p.write_text(json.dumps(tuned))
    ng3 = NGLite(module_id="mc")
    assert ng3.load(str(p)) and ng3.config["novelty_threshold"] == 0.5
