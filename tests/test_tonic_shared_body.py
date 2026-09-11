# ---- Changelog ----
# [2026-09-11] Claude Code (DudeMan CC, Fable 5.1) — shared-body-only attachment tests
# What: TonicEngine(require_shared_body=True) — no private load at construction (with or
#   without a body in hand), lazy wrapper build on first offer, retry after a missing
#   checkpoint and after a loader failure, repeated attach/revoke/re-attach on the SAME
#   wrapper, identity-verified success, and the forward-path recheck racing a revoke.
# Why: the mode's whole point is a negative — "no second transformer body in this process"
#   (~/docs/memory/laptop_home_cc-rpc-hosts-cc-ng-shared-transformer.md). A negative needs a
#   test that would SEE the private load happen; asserting the happy path would not.
# How: surgery.tonic_brain is stubbed into sys.modules with a fake load_tonic_brain that
#   records every call, so nothing here imports torch, transformers, or a checkpoint. The
#   fake raises if called with transformer_body=None — that is the own-copy branch, and it
#   fails the test loudly rather than silently costing 2GB in production.
# -------------------
"""Unit tests for TonicEngine shared-body-only attachment (require_shared_body)."""
import contextlib
import os
import sys
import threading
import types
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from neuro_foundation import Graph
import tonic_engine as te
from tonic_engine import TonicEngine, EngineConfig


# ---------------------------------------------------------------------------
# Stubs — no torch, no transformers, no checkpoint on disk
# ---------------------------------------------------------------------------

class _FakeBody:
    """Stand-in for ProtoUniBrain's transformer body. Identity is what matters."""
    def __init__(self, tag="proto"):
        self.tag = tag


class _FakePart:
    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self


class _FakeBrain:
    """Stand-in for TonicBrain: a body reference plus our own encoder/decoder."""
    def __init__(self, body):
        self.body = body
        self.encoder = _FakePart()
        self.decoder = _FakePart()
        self.eval_calls = 0
        self.forward_calls = 0

    def eval(self):
        # Recording only — the engine must NOT call this (it would recurse into
        # the borrowed body and flip proto's training flag).
        self.eval_calls += 1
        return self

    def __call__(self, features):
        self.forward_calls += 1
        return {"activations": [0.9, 0.8], "exploration": 0.0}


class _FakeGraphFeatures:
    """Stand-in for surgery.tonic_brain.GraphFeatures — only needs to be importable."""
    pass


class _Loader:
    """Records every load_tonic_brain call so a private body load cannot hide."""
    def __init__(self, fail_with=None):
        self.calls = []          # list of transformer_body arguments
        self.fail_with = fail_with
        self.brains = []

    def __call__(self, path, model_name="Qwen/Qwen2.5-0.5B", transformer_body=None):
        self.calls.append(transformer_body)
        if transformer_body is None:
            # This is load_tonic_brain's from_pretrained branch — the ~2GB own-copy
            # allocation this mode exists to prevent. Never acceptable here.
            raise AssertionError("loader called with transformer_body=None (private body load)")
        if self.fail_with is not None:
            raise self.fail_with
        brain = _FakeBrain(transformer_body)
        self.brains.append(brain)
        return brain


@contextlib.contextmanager
def _no_grad():
    yield


@pytest.fixture
def loader(monkeypatch):
    """Install a stub surgery.tonic_brain and pretend the checkpoint exists.

    Also stubs `torch` in sys.modules. `_model_inference` does `import torch` on its
    own first line and returns the heuristic on ImportError — without this stub the
    forward-path tests below would pass by never reaching the code they name.
    """
    ldr = _Loader()
    mod = types.ModuleType("surgery.tonic_brain")
    mod.load_tonic_brain = ldr
    mod.GraphFeatures = _FakeGraphFeatures
    pkg = sys.modules.get("surgery") or types.ModuleType("surgery")
    monkeypatch.setitem(sys.modules, "surgery", pkg)
    monkeypatch.setitem(sys.modules, "surgery.tonic_brain", mod)
    monkeypatch.setattr(pkg, "tonic_brain", mod, raising=False)

    torch_stub = types.ModuleType("torch")
    torch_stub.no_grad = _no_grad
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    monkeypatch.setattr(te, "_TORCH_AVAILABLE", True)
    monkeypatch.setattr(te.os.path, "exists", lambda p: True)
    return ldr


def _graph():
    g = Graph()
    for nid in ("A", "B", "C"):
        g.create_node(node_id=nid)
    return g


def _engine(**kw):
    return TonicEngine(_graph(), None, None, config=EngineConfig(), **kw)


def test_shared_only_engine_waits_without_heuristic_or_graph_writes(loader, monkeypatch):
    engine = _engine(require_shared_body=True)
    monkeypatch.setattr(engine, "_heuristic_inference", lambda _: pytest.fail("heuristic ran"))
    monkeypatch.setattr(engine._graph, "prime_and_propagate", lambda **_: pytest.fail("graph write"))
    for _ in range(3):
        assert engine._generate_latent_token_inner()["waiting_for_shared_body"] is True
    assert engine._tokens_generated == 0
    assert engine.status["using_heuristic"] is False
    assert engine.status["heuristic_allowed"] is False
    assert engine.status["waiting_for_shared_body"] is True
    assert loader.calls == []


def test_direct_heuristic_is_prohibited_for_shared_required_mode(loader):
    engine = _engine(require_shared_body=True)
    assert engine._heuristic_inference(None) == []  # Does not even inspect features.


def test_default_consumer_retains_fallback(loader, monkeypatch):
    engine = _engine()
    monkeypatch.setattr(engine, "_heuristic_inference", lambda _: [("A", 1.0)])
    assert engine._fallback_inference({}) == [("A", 1.0)]
    assert engine.status["heuristic_allowed"] is True


def test_missing_model_features_never_falls_back_on_vps(loader, monkeypatch):
    engine = _engine(require_shared_body=True)
    assert engine.offer_shared_body(_FakeBody())
    monkeypatch.setattr(engine, "_extract_graph_features_for_model", lambda: None)
    monkeypatch.setattr(engine, "_heuristic_inference", lambda _: pytest.fail("heuristic ran"))
    assert engine._model_inference({}) == []


# ---------------------------------------------------------------------------
# No private load, ever
# ---------------------------------------------------------------------------

def test_construction_loads_nothing(loader):
    """require_shared_body=True: construction must not touch the loader at all."""
    eng = _engine(require_shared_body=True)
    assert loader.calls == []
    assert eng._model is None
    assert eng._shared_body is None
    assert eng._use_heuristic is True
    assert eng.status["require_shared_body"] is True
    assert eng.status["shared_body_attached"] is False


def test_construction_with_body_attaches_without_private_load(loader):
    """A body handed in at construction goes through the attach path, not the loader's
    own-copy branch."""
    body = _FakeBody()
    eng = _engine(require_shared_body=True, transformer_body=body)
    assert loader.calls == [body]
    assert eng._model.body is body
    assert eng._use_heuristic is False


def test_try_load_model_is_never_called(monkeypatch, loader):
    """The own-copy init path is unreachable in this mode."""
    called = []
    monkeypatch.setattr(TonicEngine, "_try_load_model",
                        lambda self: called.append(True))
    _engine(require_shared_body=True)
    _engine(require_shared_body=True, transformer_body=_FakeBody())
    assert called == []


def test_default_mode_still_uses_try_load_model(monkeypatch, loader):
    """Default consumers keep the pre-existing init path untouched."""
    called = []
    monkeypatch.setattr(TonicEngine, "_try_load_model",
                        lambda self: called.append(True))
    _engine()
    assert called == [True]


def test_default_mode_offer_without_wrapper_still_refuses(loader):
    """Prior contract for default consumers: no wrapper -> offer refused, no lazy build.

    Default mode DOES call the loader at construction (the own-copy path this mode
    exists to avoid), so the baseline here is not empty — what must stay empty is
    everything the offer itself triggers.
    """
    eng = _engine()
    eng._model = None
    before = len(loader.calls)
    assert eng.offer_shared_body(_FakeBody()) is False
    assert loader.calls[before:] == []


# ---------------------------------------------------------------------------
# Failure is retryable
# ---------------------------------------------------------------------------

def test_missing_checkpoint_refuses_and_retries(monkeypatch, loader):
    """No checkpoint -> False, no loader call, no model; a later offer retries."""
    eng = _engine(require_shared_body=True)
    monkeypatch.setattr(te.os.path, "exists", lambda p: False)
    body = _FakeBody()
    assert eng.offer_shared_body(body) is False
    assert loader.calls == []
    assert eng._model is None
    assert eng._use_heuristic is True

    monkeypatch.setattr(te.os.path, "exists", lambda p: True)
    assert eng.offer_shared_body(body) is True
    assert eng._model.body is body
    assert eng._use_heuristic is False


def test_loader_failure_refuses_and_retries(loader):
    """A load error leaves no half-attached state and does not poison later offers."""
    eng = _engine(require_shared_body=True)
    loader.fail_with = RuntimeError("corrupt checkpoint")
    body = _FakeBody()
    assert eng.offer_shared_body(body) is False
    assert eng._model is None
    assert eng._shared_body is None
    assert eng._use_heuristic is True

    loader.fail_with = None
    assert eng.offer_shared_body(body) is True
    assert eng._model.body is body
    assert len(loader.calls) == 2


def test_none_body_offer_is_refused(loader):
    """A None offer must never reach the loader — that is the own-copy branch."""
    eng = _engine(require_shared_body=True)
    assert eng.offer_shared_body(None) is False
    assert loader.calls == []
    assert eng._use_heuristic is True


# ---------------------------------------------------------------------------
# Attach / revoke / re-attach
# ---------------------------------------------------------------------------

def test_late_attach_builds_wrapper_once(loader):
    """The wrapper is built on the first successful offer and reused thereafter."""
    eng = _engine(require_shared_body=True)
    body = _FakeBody()
    assert eng.offer_shared_body(body) is True
    wrapper = eng._model
    assert loader.calls == [body]

    assert eng.offer_shared_body(body) is True
    assert eng._model is wrapper          # same wrapper
    assert loader.calls == [body]         # no second build


def test_revoke_keeps_wrapper_and_reoffer_rejoins_it(loader):
    """Shed drops only the body; re-offer rejoins the SAME encoder/decoder."""
    eng = _engine(require_shared_body=True)
    body = _FakeBody("first")
    assert eng.offer_shared_body(body) is True
    wrapper = eng._model

    assert eng.revoke_shared_body() is True
    assert eng._model is wrapper
    assert eng._model.body is None
    assert eng._shared_body is None
    assert eng._use_heuristic is True
    assert eng.status["shared_body_attached"] is False

    body2 = _FakeBody("reloaded")
    assert eng.offer_shared_body(body2) is True
    assert eng._model is wrapper          # rejoined, not rebuilt
    assert eng._model.body is body2
    assert eng._use_heuristic is False
    assert loader.calls == [body]         # still exactly one build


def test_repeated_attach_revoke_cycles(loader):
    """Many cycles never build a second wrapper and never call the loader again."""
    eng = _engine(require_shared_body=True)
    bodies = [_FakeBody(f"b{i}") for i in range(4)]
    assert eng.offer_shared_body(bodies[0]) is True
    wrapper = eng._model
    for body in bodies[1:]:
        assert eng.revoke_shared_body() is True
        assert eng.offer_shared_body(body) is True
        assert eng._model is wrapper
        assert eng._model.body is body
    assert loader.calls == [bodies[0]]


def test_revoke_when_never_attached_is_false(loader):
    eng = _engine(require_shared_body=True)
    assert eng.revoke_shared_body() is False


def test_success_requires_the_exact_offered_body(loader, monkeypatch):
    """If installation does not land the offered object, report failure, not success."""
    eng = _engine(require_shared_body=True)
    assert eng.offer_shared_body(_FakeBody("first")) is True

    class _Stubborn(_FakeBrain):
        def __setattr__(self, name, value):
            if name == "body" and getattr(self, "_locked", False):
                return  # silently refuse the swap
            object.__setattr__(self, name, value)

    wrapper = eng._model
    wrapper.__class__ = _Stubborn
    wrapper._locked = True
    assert eng.offer_shared_body(_FakeBody("second")) is False
    assert eng._use_heuristic is True
    assert eng._shared_body is None
    assert eng._model is None
    replacement = _FakeBody("retry")
    assert eng.offer_shared_body(replacement) is True
    assert eng._shared_body is replacement


def test_wrapper_eval_does_not_touch_the_borrowed_body(loader):
    """eval() is applied to our encoder/decoder only — the body is proto's state."""
    eng = _engine(require_shared_body=True)
    assert eng.offer_shared_body(_FakeBody()) is True
    assert eng._model.encoder.eval_calls == 1
    assert eng._model.decoder.eval_calls == 1
    assert eng._model.eval_calls == 0


# ---------------------------------------------------------------------------
# Lock discipline / forward race
# ---------------------------------------------------------------------------

def test_offer_and_revoke_take_the_body_lock(loader):
    """Both mutators serialize against inference on the BrainSwitcher lock."""
    class _CountingLock:
        def __init__(self):
            self.inner = threading.Lock()
            self.acquires = 0

        def __enter__(self):
            self.acquires += 1
            return self.inner.__enter__()

        def __exit__(self, *a):
            return self.inner.__exit__(*a)

    eng = _engine(require_shared_body=True)
    lock = _CountingLock()
    eng.set_body_lock(lock)
    assert eng.offer_shared_body(_FakeBody()) is True
    assert lock.acquires == 1
    assert eng.revoke_shared_body() is True
    assert lock.acquires == 2


def test_monitor_offer_defers_without_waiting_for_inference(loader):
    engine = _engine(require_shared_body=True)
    lock = threading.Lock()
    engine.set_body_lock(lock)
    body = _FakeBody()
    with lock:
        assert engine.offer_shared_body(body, blocking=False) is False
        assert engine._shared_body is None
    assert engine.offer_shared_body(body, blocking=False) is True
    assert engine._shared_body is body


def test_shared_only_attachment_requires_declared_lock_file(loader, tmp_path):
    engine = _engine(require_shared_body=True)
    engine.set_body_lock(threading.Lock())
    lock_file = tmp_path / "body.lock"
    engine.set_lock_file(str(lock_file))
    body = _FakeBody()
    assert engine.offer_shared_body(body, blocking=False) is False
    assert engine._shared_body is None
    lock_file.touch()
    assert engine.offer_shared_body(body, blocking=False) is True


def test_forward_rechecks_body_inside_the_lock(loader, monkeypatch):
    """A revoke landing between the outer mode test and the forward must wait without
    heuristic execution or calling a wrapper whose body is gone."""
    eng = _engine(require_shared_body=True)
    assert eng.offer_shared_body(_FakeBody()) is True
    wrapper = eng._model

    monkeypatch.setattr(eng, "_extract_graph_features_for_model", lambda: object())
    heuristic_calls = []
    monkeypatch.setattr(eng, "_heuristic_inference",
                        lambda features: heuristic_calls.append(True) or [("A", 1.0)])

    # The revoke happens while the "caller" still believes transformer mode is on:
    # simulated by shedding after the outer check and before _model_inference runs.
    eng.revoke_shared_body()
    out = eng._model_inference({"thread_nodes": [], "active_nodes": [], "recent_spikes": []})

    assert heuristic_calls == []
    assert wrapper.forward_calls == 0
    assert out == []


def test_forward_runs_when_body_is_attached(loader, monkeypatch):
    """Control for the test above: with a body present the forward actually happens."""
    eng = _engine(require_shared_body=True)
    assert eng.offer_shared_body(_FakeBody()) is True
    monkeypatch.setattr(eng, "_extract_graph_features_for_model", lambda: object())
    monkeypatch.setattr(eng, "_get_activation_candidates",
                        lambda features: [("A", 1.0), ("B", 1.0)])
    out = eng._model_inference({"thread_nodes": [], "active_nodes": [], "recent_spikes": []})
    assert eng._model.forward_calls == 1
    assert dict(out).keys() == {"A", "B"}


def test_forward_does_not_reenter_the_body_lock(loader, monkeypatch):
    """_body_lock is a plain non-reentrant Lock; the in-lock path must not take it twice
    (and a required-sharing consumer must wait without heuristic execution)."""
    eng = _engine(require_shared_body=True)
    assert eng.offer_shared_body(_FakeBody()) is True
    eng.set_body_lock(threading.Lock())   # deadlocks on re-entry
    eng.revoke_shared_body()

    monkeypatch.setattr(eng, "_extract_graph_features_for_model", lambda: object())
    monkeypatch.setattr(eng, "_heuristic_inference", lambda features: [("A", 1.0)])

    done = []
    t = threading.Thread(
        target=lambda: done.append(
            eng._model_inference({"thread_nodes": [], "active_nodes": [], "recent_spikes": []})
        ),
        daemon=True,
    )
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), "forward path deadlocked on the body lock"
    assert done == [[]]
