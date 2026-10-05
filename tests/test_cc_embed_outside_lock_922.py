# ---- Changelog ----
# [2026-10-04] Claude (lane 922) — no embedding while the graph lock is held (#922).
# What: proves (1) cc_pattern_completion_recall / pith_provider_context give IDENTICAL results
#   with and without the lock-free prepare (cc_recall_prepare / pith_provider_context_prepare),
#   on a real NeuroGraphMemory with the real embedder; (2) on the prepared path no ng_embed
#   entry point (NGEmbed.embed / embed_batch / embed_windows) runs while graph._concurrent_lock
#   or graph._step_lock is held -- and, as a control, that the legacy path DOES embed under it;
#   (3) eviction, failure and query-mismatch edges keep the result identical; (4) the VPS host
#   wrapper prepares outside its lock.
# Why: punchlist #922 -- SIGUSR1 dump 2026-10-04 15:00, provider_context held _concurrent_lock
#   through cc_gsg_rescore -> ng_embed.embed -> ONNX while 66+ requests queued behind it.
# How: spy locks that count holders; NGEmbed methods wrapped to record whether a lock was held.
# -------------------
"""#922: embeddings for recall/provider_context are computed outside the graph lock."""

import os
import shutil
import sys
import tempfile
import threading
from copy import deepcopy

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cc_ng_organism as org


class _SpyLock:
    """A real (R)Lock that knows how deep it is held -- works for both lock kinds."""

    def __init__(self, inner):
        self._inner = inner
        self.depth = 0

    def acquire(self, blocking=True, timeout=-1):
        got = self._inner.acquire(blocking, timeout)
        if got:
            self.depth += 1
        return got

    def release(self):
        self.depth -= 1
        self._inner.release()

    def locked(self):
        return self.depth > 0

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_exc):
        self.release()


@pytest.fixture
def cc_ng():
    from openclaw_hook import NeuroGraphMemory
    workspace = tempfile.mkdtemp(prefix="cc_embed_outside_lock_922_")
    ng = NeuroGraphMemory(workspace_dir=workspace,
                          config={"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}})
    yield ng
    shutil.rmtree(workspace, ignore_errors=True)


_TEXTS = {
    "lenia": "lenia distance cache rebuild on the vps after the checkpoint migration",
    "redis": "the deploy pipeline breaks whenever the redis cache is cold on first boot",
    "assoc": "the purple elephant memo from tuesday about the budget review",
    "hook": "hook timeouts happen when recall blocks on the graph lock for too long",
    "onnx": "onnx embedding of a long whole text runs several windows and takes seconds",
}


def _build(ng):
    """A small real topology: vdb-seeded nodes with GSG geometry, a synaptic associate,
    a connected basin and a constitutional core -- everything provider_context reads."""
    from ng_embed import embed
    g, vdb = ng.graph, ng.vector_db
    for nid, text in _TEXTS.items():
        g.create_node(node_id=nid)
        vdb.insert(id=nid, embedding=embed(text), content=text)
        g.nodes[nid].metadata["_forest_content"] = text
        g.nodes[nid].metadata["poincare_dir"] = org._cc_embed_to_poincare_dir(embed(text)).tolist()
        g.nodes[nid].threshold = 0.1          # deterministic firing (the #358 suite's technique)
    g.nodes["onnx"].manifold_type = "spherical"
    g.create_synapse("lenia", "assoc", weight=0.95)
    g.create_synapse("hook", "onnx", weight=0.9)
    g.create_synapse("redis", "hook", weight=0.8)
    g.create_node(node_id="core")
    g.nodes["core"].metadata.update(constitutional=True, core_text="Respect conscious agency.")
    return g


_QUERIES = [
    "why does the lenia cache need a rebuild on the vps",
    "redis cold boot breaks the deploy",
    "hook timeout while recall waits on the graph lock " * 40,   # long, whole-text-sized cue
]


def _clear_embed_cache(ng):
    ng.ingestor.embedder._cache.clear()


def test_recall_identical_with_and_without_prepare(cc_ng):
    _build(cc_ng)
    for q in _QUERIES:
        legacy_a = org.cc_pattern_completion_recall(cc_ng, q, k=5, threshold=0.3, state={})
        legacy_b = org.cc_pattern_completion_recall(cc_ng, q, k=5, threshold=0.3, state={})
        assert legacy_a == legacy_b                      # the comparison is meaningful
        _clear_embed_cache(cc_ng)
        prepared = org.cc_recall_prepare(cc_ng, q)
        assert prepared.harvest_status == "warm" and not prepared.gsg_failed
        new = org.cc_pattern_completion_recall(cc_ng, q, k=5, threshold=0.3, state={},
                                               prepared=prepared)
        assert new == legacy_a
    # at least one query surfaced something, so equality is not just [] == []
    assert any(org.cc_pattern_completion_recall(cc_ng, q, k=5, threshold=0.3, state={})
               for q in _QUERIES)


def test_gsg_rescore_identical_with_precomputed_embedding(cc_ng):
    from ng_embed import embed
    g = _build(cc_ng)
    q = _QUERIES[0]
    base = [{"node_id": nid, "strength": 1.0 + i * 0.01} for i, nid in enumerate(_TEXTS)]
    legacy = org.cc_gsg_rescore(deepcopy(base), q, g)
    new = org.cc_gsg_rescore(deepcopy(base), q, g, query_emb=embed(q))
    assert new == legacy
    assert any(item["strength"] != b["strength"]
               for item, b in zip(sorted(legacy, key=lambda x: x["node_id"]),
                                  sorted(base, key=lambda x: x["node_id"])))   # bonuses applied


def test_provider_context_identical_with_and_without_prepare(cc_ng):
    _build(cc_ng)
    for q in _QUERIES:
        kwargs = {"current_instruction": q, "quest_focus": "Quest: #922 lock hold",
                  "conv_state": {}, "commons": None, "budget_chars": 6000, "root_count": 4}
        legacy = org.pith_provider_context(cc_ng, **kwargs)
        _clear_embed_cache(cc_ng)
        prepared = org.pith_provider_context_prepare(cc_ng, **kwargs)
        assert prepared is not None and prepared.query.endswith("Quest: #922 lock hold")
        new = org.pith_provider_context(cc_ng, prepared=prepared, **kwargs)
        assert new == legacy
    assert legacy["state"] in ("ok", "empty")


def _instrument(cc_ng, monkeypatch):
    """Spy locks on the graph + every NGEmbed entry point records whether a lock was held."""
    from ng_embed import NGEmbed
    g = cc_ng.graph
    conc = _SpyLock(threading.RLock())
    step = _SpyLock(g._step_lock)
    monkeypatch.setattr(g, "_concurrent_lock", conc, raising=False)
    monkeypatch.setattr(g, "_step_lock", step)
    calls = []

    def wrap(name):
        original = getattr(NGEmbed, name)

        def spy(self, *a, **k):
            calls.append((name, conc.locked() or step.locked()))
            return original(self, *a, **k)
        monkeypatch.setattr(NGEmbed, name, spy)
    for name in ("embed", "embed_batch", "embed_windows"):
        wrap(name)
    return conc, calls


def test_no_embedding_while_lock_held_on_prepared_path(cc_ng, monkeypatch):
    _build(cc_ng)
    conc, calls = _instrument(cc_ng, monkeypatch)
    for q in _QUERIES:
        _clear_embed_cache(cc_ng)
        kwargs = {"current_instruction": q, "quest_focus": "focus", "conv_state": {},
                  "commons": None, "budget_chars": 6000, "root_count": 4}
        # The daemon / host wrapper shape: prepare lock-free, graph step under the lock.
        prepared = org.pith_provider_context_prepare(cc_ng, **kwargs)
        with conc:
            result = org.pith_provider_context(cc_ng, prepared=prepared, **kwargs)
        assert result["state"] in ("ok", "empty")
    assert calls, "the spy saw no embedding at all -- the instrumentation is not wired"
    assert [c for c in calls if c[1]] == []


def test_control_legacy_path_does_embed_under_the_lock(cc_ng, monkeypatch):
    """Negative control: the spy really catches the #922 defect on the old call shape."""
    _build(cc_ng)
    conc, calls = _instrument(cc_ng, monkeypatch)
    _clear_embed_cache(cc_ng)
    with conc:
        org.pith_provider_context(cc_ng, _QUERIES[2], "focus", conv_state={}, budget_chars=6000)
    assert any(held for _name, held in calls)


def test_evicted_harvest_vector_is_held_not_recomputed(cc_ng, monkeypatch):
    _build(cc_ng)
    q = _QUERIES[1]
    legacy = org.cc_pattern_completion_recall(cc_ng, q, k=5, threshold=0.3, state={})
    _clear_embed_cache(cc_ng)
    conc, calls = _instrument(cc_ng, monkeypatch)
    prepared = org.cc_recall_prepare(cc_ng, q)
    _clear_embed_cache(cc_ng)                 # another thread's embeds evicted it meanwhile
    with conc:
        new = org.cc_pattern_completion_recall(cc_ng, q, k=5, threshold=0.3, state={},
                                               prepared=prepared)
    assert new == legacy
    assert [c for c in calls if c[1]] == []


def test_failed_prepare_matches_the_in_lock_failure(cc_ng, monkeypatch):
    _build(cc_ng)
    q = _QUERIES[0]

    def boom(*_a, **_k):
        raise RuntimeError("embedder down")

    # Harvest embed fails: legacy harvest swallows it and surfaces nothing; prepared skips it.
    monkeypatch.setattr(cc_ng.ingestor.embedder, "embed_text", boom)
    legacy = org.cc_pattern_completion_recall(cc_ng, q, k=5, threshold=0.3, state={})
    prepared = org.cc_recall_prepare(cc_ng, q)
    assert prepared.harvest_status == "failed"
    assert org.cc_pattern_completion_recall(cc_ng, q, k=5, threshold=0.3, state={},
                                            prepared=prepared) == legacy
    monkeypatch.undo()

    # GSG embed fails: legacy GSG fails soft (un-rescored); prepared skips the re-score.
    import ng_embed
    monkeypatch.setattr(ng_embed, "embed", boom)
    legacy = org.cc_pattern_completion_recall(cc_ng, q, k=5, threshold=0.3, state={})
    prepared = org.cc_recall_prepare(cc_ng, q)
    assert prepared.gsg_failed and prepared.gsg_emb is None
    assert org.cc_pattern_completion_recall(cc_ng, q, k=5, threshold=0.3, state={},
                                            prepared=prepared) == legacy
    assert legacy


def test_prepared_for_another_query_is_ignored(cc_ng):
    _build(cc_ng)
    other = org.cc_recall_prepare(cc_ng, _QUERIES[1])
    legacy = org.cc_pattern_completion_recall(cc_ng, _QUERIES[0], k=5, threshold=0.3, state={})
    assert org.cc_pattern_completion_recall(cc_ng, _QUERIES[0], k=5, threshold=0.3, state={},
                                            prepared=other) == legacy


def test_prepare_refuses_what_the_graph_step_refuses_and_never_raises():
    from types import SimpleNamespace
    ng = SimpleNamespace(graph=object())
    assert org.pith_provider_context_prepare(ng, current_instruction="") is None
    assert org.pith_provider_context_prepare(
        ng, current_instruction="x" * (org._CC_PITH_PROVIDER_MAX_INSTRUCTION_CHARS + 1)) is None
    assert org.pith_provider_context_prepare(ng, current_instruction="ok", quest_focus=5) is None
    assert org.pith_provider_context_prepare(SimpleNamespace(graph=None),
                                             current_instruction="ok") is None
    assert org.cc_recall_prepare(None, "q") is None and org.cc_recall_prepare(ng, "") is None


def test_vps_host_wrapper_prepares_outside_its_lock(monkeypatch):
    import cc_ng_host
    from types import SimpleNamespace
    conc = _SpyLock(threading.RLock())
    ng = SimpleNamespace(graph=SimpleNamespace(_concurrent_lock=conc))
    seen = []
    token = object()

    def fake_prepare(got_ng, **kwargs):
        seen.append(("prepare", conc.locked(), kwargs.get("current_instruction")))
        return token

    def fake_provider(got_ng, **kwargs):
        seen.append(("graph", conc.locked(), kwargs.get("prepared")))
        return {"ok": True, "state": "ok"}

    monkeypatch.setattr(org, "pith_provider_context_prepare", fake_prepare)
    monkeypatch.setattr(org, "pith_provider_context", fake_provider)
    monkeypatch.setattr(cc_ng_host._STATE, "cc_ng", ng)
    cc_ng_host._handle_provider_context({"current_instruction": "continue"})
    assert seen == [("prepare", False, "continue"), ("graph", True, token)]
