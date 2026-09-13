# ---- Changelog ----
# [2026-09-13] Grok Build (grok-4.6) — bounded per-stage Tonic timing tests
# What: Isolated fakes prove status last/EMA schema stays constant-size; candidate
#   and model-tensor feature extraction, lock wait, and transformer forward are
#   measured separately without body-lock re-entry;
#   waiting-for-shared-body / empty-graph early returns zero last-samples; adaptive
#   cadence still uses total tick work; timing/logging failures cannot stop a tick
#   or change activations.
# Why: ASSIGNMENT.md for cc-tonic-stage-timing-20260913. Live substrate and real
#   models are out of scope.
# How: surgery.tonic_brain / torch stubbed as in test_tonic_shared_body.py. No
#   Graph() from checkpoints, no tonic_brain.pt, no daemon thread start except a
#   single _generation_loop pass that self-stops on wait().
# -------------------
"""Unit tests for TonicEngine per-stage latent-token timing."""
import contextlib
import inspect
import logging
import os
import sys
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch

_import_torch = types.ModuleType("torch")
_import_torch.nn = types.ModuleType("torch.nn")
_import_torch.no_grad = contextlib.nullcontext
with patch.dict(sys.modules, {"torch": _import_torch, "torch.nn": _import_torch.nn}):
    import tonic_engine as te
TonicEngine, EngineConfig = te.TonicEngine, te.EngineConfig


STAGE_NAMES = (
    "feature_extract",
    "model_feature_extract",
    "body_lock_wait",
    "transformer_forward",
    "propagate",
    "ouroboros",
    "latent",
    "autostep",
)
STAGE_STATUS_KEYS = tuple(
    f"{kind}_{name}_ms" for name in STAGE_NAMES for kind in ("last", "ema")
)


class _FakeBody:
    def __init__(self, tag="proto"):
        self.tag = tag


class _FakePart:
    def eval(self):
        return self


class _FakeBrain:
    def __init__(self, body, forward_s=0.0):
        self.body = body
        self.encoder = _FakePart()
        self.decoder = _FakePart()
        self.forward_calls = 0
        self.forward_s = forward_s
        self.lock_held_during_forward = []

    def __call__(self, features):
        self.forward_calls += 1
        if self.forward_s:
            time.sleep(self.forward_s)
        return {"activations": [0.9, 0.8], "exploration": 0.0}


class _FakeGraphFeatures:
    pass


class _Loader:
    def __init__(self):
        self.calls = []
        self.brains = []

    def __call__(self, path, model_name="Qwen/Qwen2.5-0.5B", transformer_body=None):
        self.calls.append(transformer_body)
        if transformer_body is None:
            raise AssertionError("loader called with transformer_body=None (private body load)")
        brain = _FakeBrain(transformer_body)
        self.brains.append(brain)
        return brain


class _CountingLock:
    """Non-reentrant lock that counts enter/acquire. Re-entry deadlocks."""

    def __init__(self):
        self._lock = threading.Lock()
        self.enter_count = 0
        self.acquire_count = 0

    def acquire(self, blocking=True, timeout=-1):
        self.acquire_count += 1
        if timeout == -1:
            return self._lock.acquire(blocking)
        return self._lock.acquire(blocking, timeout)

    def release(self):
        return self._lock.release()

    def __enter__(self):
        self.enter_count += 1
        return self._lock.__enter__()

    def __exit__(self, *a):
        return self._lock.__exit__(*a)


class _FakeThread:
    def __init__(self, cycle_s=0.0):
        self.thread = []
        self.cycles = 0
        self.cycle_s = cycle_s

    def ouroboros_cycle(self):
        if self.cycle_s:
            time.sleep(self.cycle_s)
        self.cycles += 1


def _node():
    return types.SimpleNamespace(
        voltage=0.0, resting_potential=0.0, last_spike_time=-float("inf"), metadata={}
    )


def _graph(prime_s=0.0):
    def prime(**kw):
        if prime_s:
            time.sleep(prime_s)
        return types.SimpleNamespace(fired_entries=["A"])

    return types.SimpleNamespace(
        nodes={nid: _node() for nid in ("A", "B", "C")},
        timestep=0,
        synapses={},
        hyperedges={},
        config={},
        _outgoing={},
        _incoming={},
        active_predictions={},
        prime_and_propagate=prime,
        _step_lock=contextlib.nullcontext(),
        step=lambda: None,
    )


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


def _engine(loader, **kw):
    thread = kw.pop("tonic_thread", _FakeThread())
    graph = kw.pop("graph", None) or _graph()
    return TonicEngine(graph, None, thread, config=EngineConfig(), **kw)


def _attach_ready(engine, loader, forward_s=0.0):
    body = _FakeBody()
    assert engine.offer_shared_body(body) is True
    brain = _FakeBrain(body, forward_s=forward_s)
    engine._model = brain
    return brain


def _ready_inference(engine, monkeypatch):
    monkeypatch.setattr(engine, "_extract_graph_features_for_model", lambda: object())
    monkeypatch.setattr(
        engine, "_get_activation_candidates", lambda features: [("A", 1.0), ("B", 1.0)]
    )


def test_status_exposes_bounded_stage_telemetry(loader, monkeypatch):
    engine = _engine(loader, require_shared_body=True)
    _attach_ready(engine, loader, forward_s=0.02)
    _ready_inference(engine, monkeypatch)
    orig = te._extract_tonic_features

    def slow_extract(*a, **k):
        time.sleep(0.02)
        return orig(*a, **k)

    monkeypatch.setattr(te, "_extract_tonic_features", slow_extract)

    def slow_model_extract():
        time.sleep(0.02)
        return object()

    monkeypatch.setattr(engine, "_extract_graph_features_for_model", slow_model_extract)
    engine._graph.prime_and_propagate = _graph(prime_s=0.02).prime_and_propagate
    engine._tonic_thread.cycle_s = 0.02

    keys_first = None
    for _ in range(5):
        out = engine._generate_latent_token()
        assert out["activated"] == 2
        st = engine.status
        if keys_first is None:
            keys_first = set(st)
        else:
            assert set(st) == keys_first
        for key in STAGE_STATUS_KEYS:
            assert key in st
            val = st[key]
            assert isinstance(val, (int, float))
            assert val == round(float(val), 2)
        assert len(engine._stage_last_ms) == len(STAGE_NAMES)
        assert len(engine._stage_ema_ms) == len(STAGE_NAMES)
        assert not any(isinstance(v, (list, dict)) for v in engine._stage_last_ms.values())

    st = engine.status
    assert st["last_feature_extract_ms"] >= 15
    assert st["last_model_feature_extract_ms"] >= 15
    assert st["last_transformer_forward_ms"] >= 15
    assert st["last_propagate_ms"] >= 15
    assert st["last_ouroboros_ms"] >= 15
    assert st["last_latent_ms"] >= st["last_feature_extract_ms"]
    assert st["ema_feature_extract_ms"] > 0
    assert st["ema_model_feature_extract_ms"] > 0
    assert st["ema_transformer_forward_ms"] > 0
    assert engine._tonic_thread.cycles == 5


def test_lock_wait_and_forward_are_separate_without_reentry(loader, monkeypatch):
    engine = _engine(loader, require_shared_body=True)
    brain = _attach_ready(engine, loader, forward_s=0.09)
    _ready_inference(engine, monkeypatch)
    lock = _CountingLock()
    engine.set_body_lock(lock)

    hold_s = 0.07
    held = threading.Event()
    release = threading.Event()

    def holder():
        with lock:
            held.set()
            release.wait(timeout=5)

    ht = threading.Thread(target=holder, daemon=True)
    ht.start()
    assert held.wait(timeout=2)

    done = []

    def run():
        done.append(engine._generate_latent_token())

    wt = threading.Thread(target=run, daemon=True)
    wt.start()
    deadline = time.time() + 2
    while lock.enter_count < 2 and time.time() < deadline:
        time.sleep(0.001)
    assert lock.enter_count >= 2, "inference never entered the body lock"
    time.sleep(hold_s)
    release.set()
    wt.join(timeout=5)
    ht.join(timeout=2)
    assert not wt.is_alive(), "latent path deadlocked on the body lock"
    assert done and done[0]["activated"] == 2
    assert brain.forward_calls == 1
    assert lock.enter_count == 2  # holder + one inference enter; re-entry would hang
    assert lock.acquire_count == 0  # blocking path uses __enter__, not acquire()

    wait_ms = engine.status["last_body_lock_wait_ms"]
    fwd_ms = engine.status["last_transformer_forward_ms"]
    combined = (hold_s + 0.09) * 1000.0
    assert wait_ms >= 40
    assert fwd_ms >= 60
    assert wait_ms < combined * 0.85
    assert fwd_ms < combined * 0.85


def test_waiting_for_shared_body_clears_prior_stage_samples(loader, monkeypatch):
    engine = _engine(loader, require_shared_body=True)
    _attach_ready(engine, loader, forward_s=0.03)
    _ready_inference(engine, monkeypatch)
    out = engine._generate_latent_token()
    assert out["activated"] == 2
    assert engine.status["last_transformer_forward_ms"] >= 20
    prior_ema = engine.status["ema_transformer_forward_ms"]
    assert prior_ema > 0

    engine.revoke_shared_body()
    waiting = engine._generate_latent_token()
    assert waiting["waiting_for_shared_body"] is True
    st = engine.status
    assert st["last_feature_extract_ms"] == 0.0
    assert st["last_model_feature_extract_ms"] == 0.0
    assert st["last_body_lock_wait_ms"] == 0.0
    assert st["last_transformer_forward_ms"] == 0.0
    assert st["last_propagate_ms"] == 0.0
    assert st["last_ouroboros_ms"] == 0.0
    assert st["last_autostep_ms"] == 0.0
    assert st["ema_transformer_forward_ms"] == round(prior_ema, 2)
    assert waiting["fired"] == 0
    assert waiting["activated"] == 0


def test_empty_graph_early_return_zeros_unrun_stages(loader, monkeypatch):
    engine = _engine(loader, require_shared_body=True)
    _attach_ready(engine, loader, forward_s=0.03)
    _ready_inference(engine, monkeypatch)
    engine._generate_latent_token()
    assert engine.status["last_propagate_ms"] >= 0
    assert engine.status["last_transformer_forward_ms"] >= 20

    engine._graph.nodes = {}
    out = engine._generate_latent_token()
    assert out == {"fired": 0, "activated": 0}
    st = engine.status
    assert st["last_body_lock_wait_ms"] == 0.0
    assert st["last_model_feature_extract_ms"] == 0.0
    assert st["last_transformer_forward_ms"] == 0.0
    assert st["last_propagate_ms"] == 0.0
    assert st["last_ouroboros_ms"] == 0.0
    assert st["ema_transformer_forward_ms"] > 0


def test_adaptive_cadence_uses_total_tick_work(loader, monkeypatch):
    engine = _engine(loader, require_shared_body=True)
    monkeypatch.setattr(te, "_CC_NG_AUTOSTEP", False)
    engine._config.latent_interval = 0.05
    engine._config.conversation_interval = 0.05
    engine._config.adaptive_cadence = True
    engine._config.latent_interval_max = 10.0
    engine._in_conversation = False
    engine._ema_tick_ms = 0.0

    work_s = 0.12
    waits = []

    def fake_generate():
        time.sleep(work_s)
        return {"fired": 0, "activated": 0}

    def fake_wait(timeout=None):
        waits.append(timeout)
        engine._shutdown_event.set()
        return True

    engine._generate_latent_token = fake_generate
    engine._shutdown_event.wait = fake_wait
    engine._generation_loop()

    assert len(waits) == 1
    # elapsed ~120ms > 50% of 50ms base -> wait = ema*2 ≈ 0.24s, not the 0.05s base.
    assert waits[0] > engine._config.latent_interval
    assert waits[0] == pytest.approx(work_s * 2.0, rel=0.45, abs=0.08)

    src = inspect.getsource(te.TonicEngine._generation_loop)
    assert "target_wait = (self._ema_tick_ms / 1000.0) * 2.0" in src
    cadence_block = src.split("if self._config.adaptive_cadence")[1].split("self._current_interval")[0]
    assert "_stage_" not in cadence_block
    assert "ema_latent" not in cadence_block


def test_timing_and_logging_failures_cannot_stop_or_change_output(loader, monkeypatch, caplog):
    engine = _engine(loader, require_shared_body=True)
    brain = _attach_ready(engine, loader)
    _ready_inference(engine, monkeypatch)
    expected = engine._generate_latent_token()
    assert expected["activated"] == 2
    forwards_before = brain.forward_calls

    class BoomDict(dict):
        def __setitem__(self, k, v):
            raise RuntimeError("timing store failed")

        def __getitem__(self, k):
            raise RuntimeError("timing store failed")

        def get(self, k, default=None):
            raise RuntimeError("timing store failed")

        def __iter__(self):
            raise RuntimeError("timing store failed")

    engine._stage_last_ms = BoomDict()
    engine._stage_ema_ms = BoomDict()
    out = engine._generate_latent_token()
    assert out == expected
    assert brain.forward_calls == forwards_before + 1
    st = engine.status
    assert st["tokens_generated"] == 2
    for key in STAGE_STATUS_KEYS:
        assert key in st
        assert st[key] == 0.0

    monkeypatch.setattr(te, "_CC_NG_AUTOSTEP", False)
    engine._config.tick_budget_seconds = 0.0

    def boom_warning(*a, **k):
        raise RuntimeError("log failed")

    monkeypatch.setattr(te.logger, "warning", boom_warning)
    waits = []

    def fake_wait(timeout=None):
        waits.append(timeout)
        engine._shutdown_event.set()
        return True

    engine._shutdown_event.wait = fake_wait
    engine._generation_loop()
    assert len(waits) == 1
    assert brain.forward_calls == forwards_before + 2


def test_over_budget_log_includes_stages_only_when_over(loader, monkeypatch, caplog):
    engine = _engine(loader, require_shared_body=True)
    _attach_ready(engine, loader)
    _ready_inference(engine, monkeypatch)
    monkeypatch.setattr(te, "_CC_NG_AUTOSTEP", False)
    engine._config.tick_budget_seconds = 30.0
    waits = []

    def one_wait(timeout=None):
        waits.append(timeout)
        engine._shutdown_event.set()
        return True

    engine._shutdown_event.wait = one_wait
    with caplog.at_level(logging.WARNING, logger="neurograph.tonic.engine"):
        engine._generation_loop()
    assert not any("Tonic tick over budget" in r.message for r in caplog.records)

    engine._shutdown_event.clear()
    waits.clear()
    engine._config.tick_budget_seconds = 0.0
    engine._shutdown_event.wait = one_wait
    with caplog.at_level(logging.WARNING, logger="neurograph.tonic.engine"):
        engine._generation_loop()
    over = [r.message for r in caplog.records if "Tonic tick over budget" in r.message]
    assert len(over) == 1
    msg = over[0]
    for name in (
        "feature_extract",
        "model_feature_extract",
        "body_lock_wait",
        "transformer_forward",
        "propagate",
        "ouroboros",
        "latent",
        "autostep",
    ):
        assert name in msg


def test_graph_try_lock_stays_nonblocking_and_body_lock_order_unchanged():
    gen_src = inspect.getsource(te.TonicEngine._generate_latent_token)
    assert "lock.acquire(blocking=False)" in gen_src
    assert gen_src.count(".acquire(") == 1
    ctx_src = inspect.getsource(te.TonicEngine._body_lock_context)
    assert ctx_src.index("if self._body_lock is not None") < ctx_src.index("if self._lock_file_path is not None")
    model_src = inspect.getsource(te.TonicEngine._model_inference)
    assert "with self._body_lock_context():" in model_src
    assert model_src.count("_body_lock_context(") == 1


def test_autostep_timing_recorded_only_when_it_runs(loader, monkeypatch):
    engine = _engine(loader, require_shared_body=True)
    _attach_ready(engine, loader)
    _ready_inference(engine, monkeypatch)
    monkeypatch.setattr(te, "_CC_NG_AUTOSTEP", True)
    monkeypatch.setattr(te, "_CC_NG_AUTOSTEP_MIN_INTERVAL", 0.0)
    engine._last_autostep = 0.0
    stepped = []

    def slow_step():
        time.sleep(0.04)
        stepped.append(1)

    engine._graph.step = slow_step
    waits = []

    def one_wait(timeout=None):
        waits.append(timeout)
        engine._shutdown_event.set()
        return True

    engine._shutdown_event.wait = one_wait
    engine._generation_loop()
    assert stepped == [1]
    assert engine.status["last_autostep_ms"] >= 30
    assert engine.status["ema_autostep_ms"] >= 30
