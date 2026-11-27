# ---- Changelog ----
# [2026-11-27] DudeMan CC (Claude Fable 5.2, laptop; quest SQ11) -- CREATE
# What: pins CC_RECALL_CUE_NG_EMBED and _harvest_associations(query_vec_override=): the cue vector from ng_embed
#       directly equals the Ingestor embedder's embed_text for the same text; with the switch on, recall (hook
#       path and prepared path) never calls the Ingestor's embedder and returns the same results as with it off;
#       with it off the Ingestor's embedder is called as before; an ng_embed failure falls back to the old path;
#       the harvest keyword None is the old behaviour.
# Why:  the Universal Ingestor is the document path; recall's cue must not be embedded through it (Josh 2026-11-27).
# How:  the real NeuroGraphMemory; the Ingestor's embedder is wrapped to count calls, or poisoned to raise.
# -------------------
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import tempfile

import numpy as np
import pytest

import cc_ng_organism as org

QUERY = "the lenia distance cache rebuild finished"


@pytest.fixture
def ng():
    from openclaw_hook import NeuroGraphMemory
    from ng_embed import embed
    ws = tempfile.mkdtemp(prefix='cue_ng_embed_test_')
    m = NeuroGraphMemory(workspace_dir=ws, config={"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}})
    for nid, text in (("a", "lenia cache rebuild status"), ("b", "the distance cache was rebuilt overnight")):
        m.graph.create_node(node_id=nid, metadata={"_forest_content": text})
        m.graph.nodes[nid].threshold = 0.1
        m.vector_db.insert(id=nid, embedding=embed(text), content=text)
    m.graph.create_synapse("a", "b", weight=5.0)
    yield m
    shutil.rmtree(ws, ignore_errors=True)


def _count_ingestor_embeds(ng, monkeypatch):
    calls = []
    real = ng.ingestor.embedder.embed_text
    monkeypatch.setattr(ng.ingestor.embedder, "embed_text", lambda t: (calls.append(t), real(t))[1])
    return calls


def _poison_ingestor_embedder(ng, monkeypatch):
    def boom(_t):
        raise AssertionError("the Ingestor's embedder was called on the recall path")
    monkeypatch.setattr(ng.ingestor.embedder, "embed_text", boom)


def _ids(out):
    return [(r["node_id"], r["content"]) for r in out]


def test_the_direct_cue_vector_is_the_ingestor_embedders_vector(ng):
    a = np.asarray(ng.ingestor.embedder.embed_text(QUERY), dtype=np.float32)
    b = org._cc_cue_vec(QUERY)
    assert b.dtype == np.float32 and a.shape == b.shape
    assert np.allclose(a, b, atol=1e-6) and abs(float(np.linalg.norm(b)) - 1.0) < 1e-5


def test_harvest_keyword_none_is_the_old_call(ng, monkeypatch):
    calls = _count_ingestor_embeds(ng, monkeypatch)
    base = ng._harvest_associations(QUERY)
    assert calls == [QUERY] and base
    assert ng._harvest_associations(QUERY, query_vec_override=None) == base and len(calls) == 2


def test_harvest_with_the_vector_does_not_call_the_ingestor_embedder(ng, monkeypatch):
    base = ng._harvest_associations(QUERY)
    _poison_ingestor_embedder(ng, monkeypatch)
    got = ng._harvest_associations(QUERY, query_vec_override=org._cc_cue_vec(QUERY))
    assert [g["node_id"] for g in got] == [b["node_id"] for b in base]


def test_off_recall_embeds_through_the_ingestor_as_before(ng, monkeypatch):
    monkeypatch.setattr(org, "RECALL_CUE_NG_EMBED", False)
    calls = _count_ingestor_embeds(ng, monkeypatch)
    assert org.cc_pattern_completion_recall(ng, QUERY, k=5)
    assert QUERY in calls


def test_on_hook_recall_never_calls_the_ingestor_embedder_and_matches_off(ng, monkeypatch):
    monkeypatch.setattr(org, "RECALL_CUE_NG_EMBED", False)
    off = _ids(org.cc_pattern_completion_recall(ng, QUERY, k=5))
    monkeypatch.setattr(org, "RECALL_CUE_NG_EMBED", True)
    _poison_ingestor_embedder(ng, monkeypatch)
    on = _ids(org.cc_pattern_completion_recall(ng, QUERY, k=5))
    assert on == off and on


def test_on_prepared_recall_never_calls_the_ingestor_embedder(ng, monkeypatch):
    monkeypatch.setattr(org, "RECALL_CUE_NG_EMBED", False)
    off = _ids(org.cc_pattern_completion_recall(ng, QUERY, k=5, prepared=org.cc_recall_prepare(ng, QUERY)))
    monkeypatch.setattr(org, "RECALL_CUE_NG_EMBED", True)
    _poison_ingestor_embedder(ng, monkeypatch)
    prepared = org.cc_recall_prepare(ng, QUERY)
    assert prepared.cue_direct is True and prepared.harvest_status == "warm" and prepared.gsg_emb is prepared.harvest_vec
    on = _ids(org.cc_pattern_completion_recall(ng, QUERY, k=5, prepared=prepared))
    assert on == off and on


def test_on_embeds_the_cue_once_per_recall(ng, monkeypatch):
    import ng_embed
    monkeypatch.setattr(org, "RECALL_CUE_NG_EMBED", True)
    n = []
    real = ng_embed.embed
    monkeypatch.setattr(ng_embed, "embed", lambda t, **k: (n.append(t), real(t, **k))[1])
    assert org.cc_pattern_completion_recall(ng, QUERY, k=5)
    assert n == [QUERY]
    n.clear()
    assert org.cc_pattern_completion_recall(ng, QUERY, k=5, prepared=org.cc_recall_prepare(ng, QUERY))
    assert n == [QUERY]


def test_on_an_ng_embed_failure_falls_back_to_the_old_path(ng, monkeypatch):
    monkeypatch.setattr(org, "RECALL_CUE_NG_EMBED", False)
    off = _ids(org.cc_pattern_completion_recall(ng, QUERY, k=5))
    monkeypatch.setattr(org, "RECALL_CUE_NG_EMBED", True)

    def boom(_q):
        raise RuntimeError("model down")
    monkeypatch.setattr(org, "_cc_cue_vec", boom)
    calls = _count_ingestor_embeds(ng, monkeypatch)
    assert _ids(org.cc_pattern_completion_recall(ng, QUERY, k=5)) == off and QUERY in calls
    prepared = org.cc_recall_prepare(ng, QUERY)
    assert prepared.cue_direct is False and prepared.harvest_status == "warm"
