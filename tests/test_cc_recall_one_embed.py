# ---- Changelog ----
# [2026-11-27] DudeMan CC (Claude Fable 5.2, laptop; quest SQ11, punch #1201) -- CREATE
# What: cc_recall_prepare with CC_RECALL_ONE_EMBED: one model run when the harvest embedder is ng_embed-backed, the
#       old two runs otherwise; the re-score direction is the same to rounding.
# Why:  the same query was embedded twice per recall.
# How:  a stand-in harvest embedder and a counted ng_embed.embed; no model is loaded.
# -------------------
import types

import numpy as np

import cc_ng_organism as org
import ng_embed


class _Emb:
    def __init__(self, backed=True, available=True, fail=False):
        self._ng_embed = object() if backed else None
        self._model_available = available
        self.fail = fail
        self.calls = 0

    def embed_text(self, text):
        self.calls += 1
        if self.fail:
            raise RuntimeError("no")
        v = np.arange(1, 9, dtype=np.float32)
        return v / np.linalg.norm(v)


def _ng(emb):
    return types.SimpleNamespace(ingestor=types.SimpleNamespace(embedder=emb))


def _count(monkeypatch):
    n = [0]

    def embed(text, **k):
        n[0] += 1
        return np.arange(1, 9, dtype=np.float32) * 3.0          # the raw (un-normalised) vector
    monkeypatch.setattr(ng_embed, "embed", embed)
    return n


def test_off_runs_both_as_before(monkeypatch):
    n = _count(monkeypatch)
    monkeypatch.setattr(org, "RECALL_ONE_EMBED", False)
    e = _Emb()
    p = org.cc_recall_prepare(_ng(e), "q")
    assert e.calls == 1 and n[0] == 1 and p.gsg_emb is not p.harvest_vec


def test_on_runs_the_model_once_and_the_direction_is_the_same(monkeypatch):
    n = _count(monkeypatch)
    monkeypatch.setattr(org, "RECALL_ONE_EMBED", True)
    e = _Emb()
    p = org.cc_recall_prepare(_ng(e), "q")
    assert e.calls == 1 and n[0] == 0 and p.gsg_emb is p.harvest_vec and not p.gsg_failed
    raw = np.arange(1, 9, dtype=np.float32) * 3.0
    assert np.allclose(org._cc_embed_to_poincare_dir(p.gsg_emb), org._cc_embed_to_poincare_dir(raw), atol=1e-6)


def test_on_but_another_embedder_or_a_failure_takes_the_old_path(monkeypatch):
    monkeypatch.setattr(org, "RECALL_ONE_EMBED", True)
    for e in (_Emb(backed=False), _Emb(available=False), _Emb(fail=True)):
        n = _count(monkeypatch)
        p = org.cc_recall_prepare(_ng(e), "q")
        assert n[0] == 1 and p.gsg_emb is not None and p.gsg_emb is not p.harvest_vec


def test_the_helper_for_the_path_without_a_prepared_half(monkeypatch):
    monkeypatch.setattr(org, "RECALL_ONE_EMBED", False)
    assert org._cc_one_embed_vec(_ng(_Emb()), "q") is None
    monkeypatch.setattr(org, "RECALL_ONE_EMBED", True)
    assert org._cc_one_embed_vec(_ng(_Emb()), "q") is not None
    for e in (_Emb(backed=False), _Emb(available=False), _Emb(fail=True)):
        assert org._cc_one_embed_vec(_ng(e), "q") is None
    assert org._cc_one_embed_vec(types.SimpleNamespace(), "q") is None


def test_the_rescore_takes_the_vector_and_does_not_embed(monkeypatch):
    graph = types.SimpleNamespace(nodes={})
    n = _count(monkeypatch)
    org.cc_gsg_rescore([{"node_id": "x"}], "q", graph, query_emb=np.ones(8, dtype=np.float32))
    assert n[0] == 0                                           # the vector was used
    org.cc_gsg_rescore([{"node_id": "x"}], "q", graph, query_emb=None)
    assert n[0] == 1                                           # without it the re-score embeds, as before
    org.cc_gsg_rescore([], "q", graph, query_emb=None)
    assert n[0] == 1                                           # nothing to re-score: no embedding


def _hook_ng(monkeypatch):
    import test_cc_recall_evoked as ev
    ng = ev._build_ng()
    emb = _Emb()
    emb.embed_text = lambda text, _e=emb: (setattr(_e, "calls", _e.calls + 1), np.array([1.0, 0.0], dtype=np.float32))[1]
    ng.ingestor = types.SimpleNamespace(embedder=emb)
    return ng, emb, ev.CUE_TEXT


def test_the_hook_path_embeds_once_with_the_switch_on_and_twice_off(monkeypatch):
    for on, want in ((False, 1), (True, 0)):
        n = _count(monkeypatch)
        monkeypatch.setattr(org, "RECALL_ONE_EMBED", on)
        ng, emb, cue = _hook_ng(monkeypatch)
        out = org.cc_pattern_completion_recall(ng, cue, k=4)       # no prepared half: the hook path
        assert out, "the stand-in mind must surface something for this test to mean anything"
        assert n[0] == want


def test_an_empty_recall_costs_no_extra_embedding(monkeypatch):
    n = _count(monkeypatch)
    monkeypatch.setattr(org, "RECALL_ONE_EMBED", True)
    ng, emb, cue = _hook_ng(monkeypatch)
    ng.vector_db.search = lambda *a, **k: []                      # no seed: nothing surfaces
    before = emb.calls
    assert org.cc_pattern_completion_recall(ng, cue, k=4) == []
    assert n[0] == 0 and emb.calls - before == 1                  # the harvest's own embed only
