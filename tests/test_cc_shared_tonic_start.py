"""#426 host startup uses CC state and never requests a private transformer.

Changelog:
2026-09-11 Codex — lightweight startup/failure fixtures only.
2026-09-23 Claude (z8-cc-tonic-relief) — updated fake Engine to accept the
  config= kwarg _start_cc_tonic_engine now passes (cc_ng_host.py:1674), and
  added coverage for the CC_TONIC_PROPAGATION_STEPS/CC_TONIC_NODE_SAMPLE_BUDGET
  env-var throttle (cc_ng_host.py:1650-1651) added for Executive Packet 067.
  What: config kwarg assertions + env-var-to-EngineConfig plumbing test.
  Why: prior fixtures predate the throttle and would mask a regression in it.
  How: read config.propagation_steps/node_sample_budget off the fake engine's
  captured config object; reload cc_ng_host under monkeypatched os.environ to
  exercise the module-level env reads.
"""
import importlib
import sys
from types import SimpleNamespace
import cc_ng_host as host


def test_host_starts_shared_only_engine_without_session(monkeypatch):
    created = []
    class Engine:
        def __init__(self, graph, vector_db, tonic, *, config=None, require_shared_body):
            assert require_shared_body is True
            self.args = graph, vector_db, tonic
            self.config = config
            self.started = False
            created.append(self)
        def set_prefetch_seed(self, fn):
            self.seed = fn
        def start(self):
            self.started = True
        def stop(self):
            self.started = False
    monkeypatch.setitem(sys.modules, "tonic_engine", SimpleNamespace(TonicEngine=Engine, EngineConfig=lambda **kw: SimpleNamespace(**kw)))
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
    monkeypatch.setitem(sys.modules, "tonic_engine", SimpleNamespace(TonicEngine=Engine, EngineConfig=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setitem(sys.modules, "cc_ng_organism", SimpleNamespace(pith_prefetch_seed=lambda s: {}))
    tt = SimpleNamespace(_latent_engine=None)
    tt.set_latent_engine = lambda e: setattr(tt, "_latent_engine", e)
    ng = SimpleNamespace(graph=object(), vector_db=object(), _tonic_thread=tt)
    assert not host._start_cc_tonic_engine(ng)
    assert tt._latent_engine is None
    assert stopped == [True]


def test_tonic_env_vars_default_to_prior_hardcoded_values(monkeypatch):
    """No env vars set -> byte-identical to pre-throttle behavior (propagation_steps=2, node_sample_budget=5000)."""
    monkeypatch.delenv("CC_TONIC_PROPAGATION_STEPS", raising=False)
    monkeypatch.delenv("CC_TONIC_NODE_SAMPLE_BUDGET", raising=False)
    reloaded = importlib.reload(host)
    try:
        assert reloaded.CC_TONIC_PROPAGATION_STEPS == 2
        assert reloaded.CC_TONIC_NODE_SAMPLE_BUDGET == 5000
    finally:
        importlib.reload(host)


def test_tonic_env_vars_flow_into_engine_config(monkeypatch):
    monkeypatch.setenv("CC_TONIC_PROPAGATION_STEPS", "1")
    monkeypatch.setenv("CC_TONIC_NODE_SAMPLE_BUDGET", "1500")
    reloaded = importlib.reload(host)
    try:
        created = []
        class Engine:
            def __init__(self, graph, vector_db, tonic, *, config=None, require_shared_body):
                self.config = config
                created.append(self)
            def set_prefetch_seed(self, fn):
                pass
            def start(self):
                pass
            def stop(self):
                pass
        monkeypatch.setitem(sys.modules, "tonic_engine", SimpleNamespace(TonicEngine=Engine, EngineConfig=lambda **kw: SimpleNamespace(**kw)))
        monkeypatch.setattr(reloaded._STATE, "conv_state", {})
        monkeypatch.setitem(sys.modules, "cc_ng_organism", SimpleNamespace(pith_prefetch_seed=lambda s: s))
        tt = SimpleNamespace(_latent_engine=None)
        tt.set_latent_engine = lambda e: setattr(tt, "_latent_engine", e)
        ng = SimpleNamespace(graph=object(), vector_db=object(), _tonic_thread=tt)
        assert reloaded._start_cc_tonic_engine(ng)
        assert created[0].config.propagation_steps == 1
        assert created[0].config.node_sample_budget == 1500
    finally:
        importlib.reload(host)


def test_tonic_env_vars_clamp_to_minimum_one(monkeypatch):
    monkeypatch.setenv("CC_TONIC_PROPAGATION_STEPS", "0")
    monkeypatch.setenv("CC_TONIC_NODE_SAMPLE_BUDGET", "-5")
    reloaded = importlib.reload(host)
    try:
        assert reloaded.CC_TONIC_PROPAGATION_STEPS == 1
        assert reloaded.CC_TONIC_NODE_SAMPLE_BUDGET == 1
    finally:
        importlib.reload(host)
