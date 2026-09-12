"""#426 host startup uses CC state and never requests a private transformer.

Changelog: 2026-09-11 Codex — lightweight startup/failure fixtures only.
"""
import sys
from types import SimpleNamespace
import cc_ng_host as host


def test_host_starts_shared_only_engine_without_session(monkeypatch):
    created = []
    class Engine:
        def __init__(self, graph, vector_db, tonic, *, require_shared_body):
            assert require_shared_body is True
            self.args = graph, vector_db, tonic
            self.started = False
            created.append(self)
        def set_prefetch_seed(self, fn):
            self.seed = fn
        def start(self):
            self.started = True
        def stop(self):
            self.started = False
    monkeypatch.setitem(sys.modules, "tonic_engine", SimpleNamespace(TonicEngine=Engine))
    state = {"last_forest_id": "cc-only"}
    monkeypatch.setattr(host._STATE, "conv_state", state)
    monkeypatch.setitem(sys.modules, "cc_ng_organism", SimpleNamespace(pith_prefetch_seed=lambda s: s))
    tt = SimpleNamespace(_latent_engine=None)
    tt.set_latent_engine = lambda e: setattr(tt, "_latent_engine", e)
    ng = SimpleNamespace(graph=object(), vector_db=object(), _tonic_thread=tt)
    assert host._start_cc_tonic_engine(ng)
    assert created[0].args == (ng.graph, ng.vector_db, tt)
    assert tt._latent_engine.started
    assert tt._latent_engine.seed() is state
    assert host._start_cc_tonic_engine(ng)
    assert len(created) == 1


def test_start_failure_does_not_publish_partial_engine(monkeypatch):
    stopped = []
    class Engine:
        def __init__(self, *a, **kw):
            pass
        def set_prefetch_seed(self, fn):
            pass
        def start(self):
            raise RuntimeError("thread start failed")
        def stop(self):
            stopped.append(True)
    monkeypatch.setitem(sys.modules, "tonic_engine", SimpleNamespace(TonicEngine=Engine))
    monkeypatch.setitem(sys.modules, "cc_ng_organism", SimpleNamespace(pith_prefetch_seed=lambda s: {}))
    tt = SimpleNamespace(_latent_engine=None)
    tt.set_latent_engine = lambda e: setattr(tt, "_latent_engine", e)
    ng = SimpleNamespace(graph=object(), vector_db=object(), _tonic_thread=tt)
    assert not host._start_cc_tonic_engine(ng)
    assert tt._latent_engine is None
    assert stopped == [True]
