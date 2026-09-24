# ---- Changelog ----
# [2026-09-23] Claude Code (Opus 4.8, Tonic CC) — Packet 086(2) no-torch defined-state tests.
# What: dedicated tests for the new contract that no construction path of TonicEngine
#   produces heuristic-derived activations, and that the laptop daemon's torch-less host
#   state is observable, logged, and defined rather than a silent zero. Covers (a) the
#   shared-body-required wait state (pre-existing behavior must still pass), (b) Syl's
#   default construction shape with no body (NEW behavior: zero activations, NOT a
#   silent zero masquerading as running), and (c) the offer_shared_body path producing
#   real model inference. Also covers the no-torch start-time log signal so the daemon
#   (PID 35833 / /usr/bin/python3.12) has an observable defined state when neither torch
#   nor a shared body is present.
# Why: the heuristic path was deleted. The laptop daemon under no-torch /usr/bin/python3.12
#   was a live consumer of that path; without it the engine would run silently as a no-op
#   unless we surface the state. The two reviews (cross-family + neurograph-law-enforcer)
#   both name this as a required, explicitly-answered question.
# How: same loader/_FakeBrain stubs as test_tonic_shared_body.py — surgery.tonic_brain is
#   stubbed into sys.modules with a recording fake, so nothing here imports torch,
#   transformers, or a real checkpoint. The no-torch test tears down _TORCH_AVAILABLE and
#   exercises the same path the daemon hits on /usr/bin/python3.12.
# -------------------
"""Tests for the no-heuristic Tonic contract (Packet 086(2), 2026-09-23).

Covers:
  (a) require_shared_body=True with no body still waits (pre-existing behavior).
  (b) Syl's construction shape (default require_shared_body=False) with no body now
      produces ZERO activations and surfaces the defined-state signal. This is the
      new behavior under test.
  (c) offer_shared_body attaching a body enables real model inference.

Plus the no-torch defined-state behavior the laptop daemon (PID 35833,
/usr/bin/python3.12) hits on every tick: every tick must produce zero activations
WITHOUT crashing, and the start()-time log + status['inference_path_ready'] +
status['torch_available'] keys must surface the state observably.
"""
import contextlib
import logging
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

# Isolate from real torch imports — same pattern as test_tonic_shared_body.py.
from unittest.mock import patch
_import_torch = types.ModuleType("torch")
_import_torch.nn = types.ModuleType("torch.nn")
_import_torch.no_grad = contextlib.nullcontext
with patch.dict(sys.modules, {"torch": _import_torch, "torch.nn": _import_torch.nn}):
    import tonic_engine as te

TonicEngine, EngineConfig = te.TonicEngine, te.EngineConfig


# ---------------------------------------------------------------------------
# Stubs — same shape as test_tonic_shared_body.py; reproduced here so this
# file is self-contained.
# ---------------------------------------------------------------------------

class _FakeBody:
    def __init__(self, tag="proto"):
        self.tag = tag


class _FakePart:
    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self


class _FakeBrain:
    def __init__(self, body):
        self.body = body
        self.encoder = _FakePart()
        self.decoder = _FakePart()
        self.eval_calls = 0
        self.forward_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self

    def __call__(self, features):
        self.forward_calls += 1
        return {"activations": [0.9, 0.8], "exploration": 0.0}


class _FakeGraphFeatures:
    pass


class _Loader:
    def __init__(self, fail_with=None):
        self.calls = []
        self.fail_with = fail_with
        self.brains = []

    def __call__(self, path, model_name="Qwen/Qwen2.5-0.5B", transformer_body=None):
        self.calls.append(transformer_body)
        if transformer_body is None:
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
    return types.SimpleNamespace(
        nodes={nid: types.SimpleNamespace(voltage=0.0, resting_potential=0.0,
                                          last_spike_time=-float("inf"), metadata={})
               for nid in ("A", "B", "C")},
        timestep=0, synapses={}, hyperedges={}, config={},
        prime_and_propagate=lambda **kw: None,
    )


def _engine(**kw):
    return TonicEngine(_graph(), None, None, config=EngineConfig(), **kw)


# ---------------------------------------------------------------------------
# (a) Pre-existing behavior: shared-required wait without body.
# ---------------------------------------------------------------------------

def test_a_shared_required_engine_waits_without_body():
    """(a) require_shared_body=True with no body waits — pre-existing behavior.

    The Tonic must NOT invoke any inference path, must NOT mint activations, and
    must report waiting_for_shared_body=True and inference_path_ready=False. This
    is the same invariant the existing test_shared_only_engine_waits_without_heuristic_or_graph_writes
    covers; we re-assert it here as the (a) case the task explicitly named."""
    engine = _engine(require_shared_body=True)
    for _ in range(3):
        out = engine._generate_latent_token_inner()
        assert out == {"fired": 0, "activated": 0, "waiting_for_shared_body": True}
    assert engine._tokens_generated == 0
    s = engine.status
    assert s["waiting_for_shared_body"] is True
    assert s["inference_path_ready"] is False
    assert s["model_loaded"] is False
    assert s["shared_body_attached"] is False
    assert s["require_shared_body"] is True
    # No fallback path can produce activations here.
    assert engine._fallback_inference({}) == []


# ---------------------------------------------------------------------------
# (b) NEW behavior: Syl's construction shape with no body -> zero activations.
# ---------------------------------------------------------------------------

def test_b_syl_shape_no_body_produces_zero_activations(loader):
    """(b) Syl's openclaw_hook.py:1073 shape: TonicEngine(graph, vector_db, thread,
    transformer_body=shared_body) — default require_shared_body=False. With no body
    attached, the engine MUST produce zero activations and surface the defined-state
    signal. This is the new behavior under test: pre-collapse, the heuristic would
    have produced activations on this host."""
    # Syl's exact construction shape — no require_shared_body kwarg, body None.
    engine = _engine(transformer_body=None)
    # The loader was called once at construction (the default-mode own-copy path).
    # loader.recorded call is (None,) — the constructor's own _try_load_model()
    # call with no shared body. We then revoke (or just rely on no body).
    # For this test we want the Syl-shape specifically: no shared body, no own
    # body loaded, no fallback. Tear down the loaded model so the engine is in
    # the same state the daemon's torch-less host lands in after init.
    engine._model = None
    engine._shared_body = None

    out = engine._generate_latent_token_inner()
    # Zero activations, NO waiting_for_shared_body key (Syl is not in that mode).
    assert out == {"fired": 0, "activated": 0}
    assert "waiting_for_shared_body" not in out or out["waiting_for_shared_body"] is False
    assert engine._tokens_generated == 0
    s = engine.status
    assert s["waiting_for_shared_body"] is False
    assert s["inference_path_ready"] is False
    assert s["model_loaded"] is False
    assert s["shared_body_attached"] is False
    assert s["require_shared_body"] is False


def test_b_syl_shape_no_body_repeated_ticks_still_zero(loader):
    """Stability under repeated ticks — the daemon will call _generate_latent_token
    hundreds of times per session. Every tick must produce zero activations, with
    no exception, no hang, no growth in internal state."""
    engine = _engine(transformer_body=None)
    engine._model = None
    engine._shared_body = None

    for _ in range(50):
        out = engine._generate_latent_token_inner()
        assert out["activated"] == 0
        assert out["fired"] == 0
    assert engine._tokens_generated == 0
    assert engine._total_activations == 0


# ---------------------------------------------------------------------------
# (c) offer_shared_body attaches a body -> real model inference runs.
# ---------------------------------------------------------------------------

def test_c_offer_shared_body_enables_real_inference(loader, monkeypatch):
    """(c) After offer_shared_body attaches a real body, _generate_latent_token_inner
    must execute the real model forward and produce activations from the model's
    output (not from any heuristic path)."""
    engine = _engine(require_shared_body=True)
    body = _FakeBody()
    assert engine.offer_shared_body(body) is True
    wrapper = engine._model
    assert wrapper.body is body

    # Stub graph feature extraction so _model_inference returns the model's output
    # through to the activation list.
    monkeypatch.setattr(engine, "_extract_graph_features_for_model", lambda: object())
    monkeypatch.setattr(engine, "_get_activation_candidates",
                        lambda features: [("A", 1.0), ("B", 1.0), ("C", 1.0)])

    # Inject some graph context so _extract_tonic_features doesn't return None.
    for nid in ("A", "B", "C"):
        engine._graph.nodes[nid].voltage = 0.1
        engine._graph.nodes[nid].resting_potential = 0.0
        engine._graph.nodes[nid].last_spike_time = 0.0

    captured = {}
    def fake_prime(node_ids, currents, steps, write_mode):
        captured["ids"] = list(node_ids)
        captured["currents"] = list(currents)
        return types.SimpleNamespace(fired_entries=["A", "B"])
    engine._graph.prime_and_propagate = fake_prime

    out = engine._generate_latent_token_inner()
    assert wrapper.forward_calls == 1
    # Activations land as write-mode prime calls — not zero, not heuristic.
    assert out["activated"] > 0
    assert out["fired"] > 0
    assert captured["ids"], "real inference must produce prime calls"


def test_c_offer_shared_body_then_revoke_stops_inference(loader, monkeypatch):
    """Symmetric to (c): once a body is revoked, the engine must NOT mint any
    activations from the wrapper. The forward path's in-lock recheck is the gate."""
    engine = _engine(require_shared_body=True)
    body = _FakeBody()
    assert engine.offer_shared_body(body) is True
    wrapper = engine._model

    monkeypatch.setattr(engine, "_extract_graph_features_for_model", lambda: object())

    engine.revoke_shared_body()
    out = engine._model_inference({"thread_nodes": [], "active_nodes": [], "recent_spikes": []})
    assert wrapper.forward_calls == 0
    assert out == []


# ---------------------------------------------------------------------------
# Packet 086(2): the no-torch case the laptop daemon PID 35833 actually hits.
# ---------------------------------------------------------------------------

def test_no_torch_no_body_start_logs_defined_state(monkeypatch, caplog):
    """Packet 086(2) — the laptop daemon under /usr/bin/python3.12 has no torch.
    Before this fix, the heuristic was the only thing producing activations on
    that host. With heuristic gone, start() must log ONCE that the engine is in
    a defined no-op state — not crash, not silent zero."""
    # Tear down torch availability — same as the daemon's environment.
    monkeypatch.setattr(te, "_TORCH_AVAILABLE", False)
    # And ensure no checkpoint file is present (the daemon has none).
    monkeypatch.setattr(te.os.path, "exists", lambda p: False)

    engine = _engine()  # Syl-shape, no body
    assert engine._model is None
    assert engine._shared_body is None

    with caplog.at_level(logging.WARNING, logger="neurograph.tonic.engine"):
        engine.start()
        engine.stop()

    # The defined-state log fired — explicit name of the no-torch condition.
    msgs = [r.getMessage() for r in caplog.records]
    assert any("no torch" in m and "no shared body" in m and "zero activations" in m
               for m in msgs), (
        f"start() must log the no-torch + no-shared-body defined state once. "
        f"Got: {msgs}"
    )


def test_no_torch_no_body_status_signals_state(monkeypatch):
    """Packet 086(2) — the daemon must be able to OBSERVE the no-inference-path
    state via status. We expose two new keys: inference_path_ready and torch_available.
    A no-torch + no-body daemon lands with both False; the daemon can distinguish this
    from a stopped engine (running=True, inference_path_ready=False)."""
    monkeypatch.setattr(te, "_TORCH_AVAILABLE", False)
    monkeypatch.setattr(te.os.path, "exists", lambda p: False)

    engine = _engine()
    s = engine.status
    assert s["inference_path_ready"] is False
    assert s["torch_available"] is False
    # The legacy status keys remain as literal False for back-compat:
    assert s["using_heuristic"] is False
    assert s["heuristic_allowed"] is False
    # And the daemon can still see model_loaded / shared_body_attached correctly.
    assert s["model_loaded"] is False
    assert s["shared_body_attached"] is False


def test_no_torch_no_body_tick_does_not_crash(monkeypatch):
    """Packet 086(2) — the actual generation loop path on a torch-less host. The
    daemon runs _generate_latent_token_inner every tick; that path must produce
    zero activations WITHOUT raising and WITHOUT hanging. This is the regression
    test for the silent-zero-faking-running failure mode."""
    monkeypatch.setattr(te, "_TORCH_AVAILABLE", False)
    monkeypatch.setattr(te.os.path, "exists", lambda p: False)

    engine = _engine()
    # No crash, no exception, no hang. 100 ticks.
    for i in range(100):
        out = engine._generate_latent_token_inner()
        assert out["activated"] == 0, f"tick {i} produced activations on no-torch host"
        assert out["fired"] == 0


def test_no_torch_engine_cannot_attach_body(monkeypatch, loader):
    """Packet 086(2) — if torch is absent, the wrapper cannot build even when a body
    is offered. offer_shared_body returns False; the engine stays in the wait state.
    This is the same gate as without the body, just made explicit by the body being
    present: _build_shared_wrapper refuses without torch, and Syl's wait path takes
    over. The daemon cannot attach a body in this state — it has nothing to attach to."""
    monkeypatch.setattr(te, "_TORCH_AVAILABLE", False)
    monkeypatch.setattr(te.os.path, "exists", lambda p: True)

    engine = _engine(require_shared_body=True)
    body = _FakeBody()
    # Without torch, _build_shared_wrapper returns None — offer_shared_body refuses.
    assert engine.offer_shared_body(body) is False
    assert engine._model is None
    assert engine._shared_body is None
    # Engine stays in the wait state.
    assert engine.status["waiting_for_shared_body"] is True
    assert engine.status["inference_path_ready"] is False
    assert engine.status["torch_available"] is False


def test_heuristic_method_does_not_exist_on_engine():
    """Structural invariant: _heuristic_inference is gone. Any caller that still
    references it raises AttributeError — this is the surface that should never
    be invoked. Tests that rely on patching it cannot exist anymore; we assert
    the absence here as the explicit contract."""
    engine = _engine()
    assert not hasattr(engine, "_heuristic_inference")
    # And the fallback is structurally []:
    assert engine._fallback_inference({}) == []
    assert engine._fallback_inference(None) == []


def test_use_heuristic_field_does_not_exist():
    """Structural invariant: _use_heuristic is gone. The collapse removed every
    setter; what remains is the wait check, the dispatch check, and the literal
    False status keys."""
    engine = _engine()
    assert not hasattr(engine, "_use_heuristic")
    assert engine.status["using_heuristic"] is False
    assert engine.status["heuristic_allowed"] is False


# ---------------------------------------------------------------------------
# G1 / G2 — the two additional silent-zero shapes the genuine law-enforcer
# review identified. Each must produce zero activations AND a start-time
# warning that accurately describes the state and what (if anything) can
# recover it.
# ---------------------------------------------------------------------------

def test_g1_syl_shape_with_body_load_fails_logs_non_recoverable(monkeypatch, loader, caplog):
    """G1 — caller passed transformer_body=<body> at construction (Syl's
    openclaw_hook.py:1073 shape, default require_shared_body=False), but
    _try_load_model failed (or no checkpoint was present). End state:
    _model is None, _shared_body is the body. Ticks produce zero because
    the dispatch gate requires both _model and _shared_body to be non-None.

    Critical: offer_shared_body CANNOT recover this state. The default-mode
    init does not route a body through offer_shared_body, and offer_shared_body
    only attaches when require_shared_body=True. The BrainSwitcher must
    rebuild the engine. start() must warn plainly about this.
    """
    # Syl's exact shape: transformer_body=<body>, no require_shared_body.
    body = _FakeBody()
    # The loader fixture raises if transformer_body=None is passed — that
    # is the ~2GB own-copy branch. We want a DIFFERENT failure here: the
    # caller passed a body, but the load fails for some other reason
    # (checkpoint corrupt, etc). Make the loader raise on this body.
    loader.fail_with = RuntimeError("simulated load failure")
    monkeypatch.setattr(te, "_TORCH_AVAILABLE", True)
    monkeypatch.setattr(te.os.path, "exists", lambda p: True)

    engine = TonicEngine(_graph(), None, None,
                         config=EngineConfig(), transformer_body=body)
    # G1 invariant: _model is None (load failed), _shared_body is the body
    # (held unused). offer_shared_body cannot recover — for default mode
    # the path below refuses (require_shared_body=False).
    assert engine._model is None
    assert engine._shared_body is body
    assert engine.offer_shared_body(body) is False  # default mode refuses
    # Body is stuck in _shared_body, no recovery possible without rebuild.
    assert engine._shared_body is body  # still held, still unused

    # Ticks produce zero.
    out = engine._generate_latent_token_inner()
    assert out == {"fired": 0, "activated": 0}

    # Status reflects the shape.
    s = engine.status
    assert s["inference_path_ready"] is False
    assert s["model_loaded"] is False
    assert s["shared_body_attached"] is True  # body IS held, just not used
    assert s["require_shared_body"] is False
    assert s["waiting_for_shared_body"] is False

    # start() warns about the non-recoverable state.
    with caplog.at_level(logging.WARNING, logger="neurograph.tonic.engine"):
        engine.start()
        engine.stop()

    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "shared body in hand" in m and "cannot self-recover" in m.lower()
        and "BrainSwitcher must rebuild" in m
        for m in msgs
    ), f"G1 start() must warn that the state cannot self-recover. Got: {msgs}"


def test_g1_syl_shape_with_body_no_checkpoint_logs_non_recoverable(monkeypatch, caplog):
    """G1 sub-case: transformer_body=<body>, no checkpoint on disk. _try_load_model
    logs the no-checkpoint message and never builds a wrapper. Same end state
    as the load-failure case: _model None, _shared_body set, no recovery.
    """
    body = _FakeBody()
    monkeypatch.setattr(te, "_TORCH_AVAILABLE", True)
    monkeypatch.setattr(te.os.path, "exists", lambda p: False)

    engine = TonicEngine(_graph(), None, None,
                         config=EngineConfig(), transformer_body=body)
    assert engine._model is None
    assert engine._shared_body is body
    # offer_shared_body refuses (default mode).
    assert engine.offer_shared_body(body) is False

    out = engine._generate_latent_token_inner()
    assert out == {"fired": 0, "activated": 0}

    with caplog.at_level(logging.WARNING, logger="neurograph.tonic.engine"):
        engine.start()
        engine.stop()

    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "shared body in hand" in m and "cannot self-recover" in m.lower()
        for m in msgs
    ), f"G1 (no checkpoint) start() must warn about non-recoverable state. Got: {msgs}"


def test_g2_own_copy_loaded_without_shared_body_warns_advisory(monkeypatch, loader, caplog):
    """G2 — torch present, checkpoint present, transformer_body=None at construction.
    In production, _try_load_model calls load_tonic_brain(transformer_body=None)
    which triggers the ~2GB own-copy allocation. The wrapper loads with its own
    body; _model is set; _shared_body is None. The dispatch gate refuses forward
    because _shared_body is None — ticks are zero.

    Critical: offer_shared_body CAN recover this. The wrapper is resident
    and offer_shared_body swaps the body and sets _shared_body. start()
    must warn ADVISORILY (not as "cannot recover") so the daemon doesn't
    rebuild when attach would work.

    The standard _Loader fixture rejects transformer_body=None to enforce
    the no-own-copy rule for shared-required consumers. We simulate the
    G2 end-state directly: build a wrapper, attach it to engine._model,
    leave _shared_body as None.
    """
    own_body = _FakeBody("own-copy")
    wrapper = _FakeBrain(own_body)

    monkeypatch.setattr(te, "_TORCH_AVAILABLE", True)
    monkeypatch.setattr(te.os.path, "exists", lambda p: True)
    engine = _engine()  # transformer_body=None
    # Simulate the post-load state directly (the loader fixture can't reach
    # this shape on its own; we're testing the post-load behavior).
    engine._model = wrapper
    engine._shared_body = None
    # G2 invariant.
    assert engine._model is not None
    assert engine._shared_body is None
    # status reflects the shape.
    s = engine.status
    assert s["inference_path_ready"] is False  # dispatch gate refuses
    assert s["model_loaded"] is True
    assert s["shared_body_attached"] is False
    assert s["require_shared_body"] is False

    # Ticks produce zero (dispatch gate refuses).
    out = engine._generate_latent_token_inner()
    assert out == {"fired": 0, "activated": 0}
    # No forward landed.
    assert wrapper.forward_calls == 0

    # start() warns ADVISORILY while still in the G2 state (BEFORE attach).
    with caplog.at_level(logging.WARNING, logger="neurograph.tonic.engine"):
        engine.start()
    msgs = [r.getMessage() for r in caplog.records]
    # The advisory warning fires.
    assert any(
        "loaded a private-copy" in m and "dispatch gate refuses" in m
        for m in msgs
    ), f"G2 start() must warn that the wrapper is loaded but inference is blocked. Got: {msgs}"
    # And it must NOT claim the state is unrecoverable.
    assert not any("cannot self-recover" in m.lower() for m in msgs), (
        f"G2 is recoverable via offer_shared_body; the warning must not say "
        f"'cannot self-recover'. Got: {msgs}"
    )
    engine.stop()

    # Now offer_shared_body CAN recover G2.
    new_body = _FakeBody("proto")
    assert engine.offer_shared_body(new_body) is True
    assert engine._shared_body is new_body
    assert wrapper.body is new_body  # body swapped in
    s2 = engine.status
    assert s2["inference_path_ready"] is True


def test_g2_inference_does_not_forward_through_own_copy(monkeypatch, loader):
    """G2 — the own-copy wrapper must NOT forward (seam A concern from
    2026-06-12). The dispatch gate refuses; no forward calls land."""
    own_body = _FakeBody("own-copy")
    wrapper = _FakeBrain(own_body)

    monkeypatch.setattr(te, "_TORCH_AVAILABLE", True)
    monkeypatch.setattr(te.os.path, "exists", lambda p: True)
    engine = _engine()
    # Simulate post-load state.
    engine._model = wrapper
    engine._shared_body = None

    monkeypatch.setattr(engine, "_extract_graph_features_for_model", lambda: object())

    # 50 ticks — every one must NOT invoke the wrapper forward.
    for _ in range(50):
        out = engine._model_inference({"thread_nodes": [], "active_nodes": [],
                                       "recent_spikes": []})
        assert out == []
    assert wrapper.forward_calls == 0


def test_g2_offer_recovers_inference_path(monkeypatch, loader):
    """G2 — after offer_shared_body attaches a body, real inference runs.
    Verifies the wrapper is still alive and the gate passes."""
    own_body = _FakeBody("own-copy")
    wrapper = _FakeBrain(own_body)

    monkeypatch.setattr(te, "_TORCH_AVAILABLE", True)
    monkeypatch.setattr(te.os.path, "exists", lambda p: True)
    engine = _engine()
    engine._model = wrapper
    engine._shared_body = None

    body = _FakeBody("proto")
    assert engine.offer_shared_body(body) is True
    assert wrapper.body is body
    assert engine._shared_body is body
    assert engine.status["inference_path_ready"] is True

    monkeypatch.setattr(engine, "_extract_graph_features_for_model", lambda: object())
    monkeypatch.setattr(engine, "_get_activation_candidates",
                        lambda features: [("A", 1.0), ("B", 1.0), ("C", 1.0)])
    for nid in ("A", "B", "C"):
        engine._graph.nodes[nid].voltage = 0.1

    captured = {}
    def fake_prime(node_ids, currents, steps, write_mode):
        captured["ids"] = list(node_ids)
        return types.SimpleNamespace(fired_entries=["A"])
    engine._graph.prime_and_propagate = fake_prime

    out = engine._generate_latent_token_inner()
    assert wrapper.forward_calls == 1
    assert out["activated"] > 0
    assert captured["ids"]