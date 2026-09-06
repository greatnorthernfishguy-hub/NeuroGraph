# ---- Changelog ----
# [2026-09-05] Claude Code (DudeMan CC, Fable 5.1) — Pith Stage 4 phase 5b unit tests (#55)
# What: TonicEngine._merge_prefetch_seeds — gate-off byte-identical, no-seed no-op,
#   score-scaled current, cap, dedup against model/heuristic picks, missing-node
#   skip, raising-seed fail-soft, status counter.
# Why: spec sec 6 def-of-done demands dedicated tests; the dict-cache take 1 was
#   reverted and this is the tick-riding replacement (see tonic_engine.py changelog).
# How: real Graph via the same helper shape as test_tonic_write_mode.py; the engine
#   is built with tonic_thread=None and never started -- the merge is a pure list
#   transform, so no tick, no thread, no model.
# -------------------
"""Unit tests for Pith Stage 4 phase 5b — Markov prefetch on the Tonic tick (#55)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from neuro_foundation import Graph
import tonic_engine as te
from tonic_engine import TonicEngine, EngineConfig


def _graph():
    g = Graph()
    for nid in ("A", "B", "P1", "P2", "P3"):
        g.create_node(node_id=nid)
    return g


@pytest.fixture
def engine():
    return TonicEngine(_graph(), None, None, config=EngineConfig(activation_strength=1.0))


BASE = {"A": 0.9, "B": 0.4}


def _merge(engine, base):
    """Run the in-place merge on a copy of `base` and return the resulting dict."""
    seen = dict(base)
    engine._merge_prefetch_seeds(seen)
    return seen


def test_gate_default_parses_off(monkeypatch):
    """The CODE default, not the process env: an unset var parses to off. Written
    this way so satisfying LAW 5 by exporting the var in ~/.bashrc cannot turn
    this test red (law-enforcer 2026-09-05)."""
    import importlib
    monkeypatch.delenv("CC_PITH_PREFETCH_WARM_ENABLED", raising=False)
    importlib.reload(te)
    try:
        assert te._CC_PITH_PREFETCH_WARM_ENABLED is False
        assert te._CC_PITH_PREFETCH_REPEATS == 1
    finally:
        importlib.reload(te)


def test_off_is_byte_identical(engine, monkeypatch):
    monkeypatch.setattr(te, "_CC_PITH_PREFETCH_WARM_ENABLED", False)
    engine.set_prefetch_seed(lambda: {"P1": 1.0})
    assert _merge(engine, BASE) == BASE
    assert engine._prefetch_seeded == 0


def test_no_seed_source_is_noop(engine, monkeypatch):
    monkeypatch.setattr(te, "_CC_PITH_PREFETCH_WARM_ENABLED", True)
    assert _merge(engine, BASE) == BASE


def test_seeds_added_with_scaled_current(engine, monkeypatch):
    monkeypatch.setattr(te, "_CC_PITH_PREFETCH_WARM_ENABLED", True)
    monkeypatch.setattr(te, "_CC_PITH_PREFETCH_CURRENT_SCALE", 0.5)
    engine.set_prefetch_seed(lambda: {"P1": 1.0, "P2": 0.5})
    out = _merge(engine, BASE)
    assert out["A"] == 0.9 and out["B"] == 0.4
    assert out["P1"] == pytest.approx(0.5)
    assert out["P2"] == pytest.approx(0.25)
    assert engine._prefetch_seeded == 2
    assert engine.status["prefetch_seeded"] == 2


def test_score_capped_at_one(engine, monkeypatch):
    monkeypatch.setattr(te, "_CC_PITH_PREFETCH_WARM_ENABLED", True)
    monkeypatch.setattr(te, "_CC_PITH_PREFETCH_CURRENT_SCALE", 0.5)
    engine.set_prefetch_seed(lambda: {"P1": 40.0})
    assert _merge(engine, {})["P1"] == pytest.approx(0.5)


def test_cap_keeps_strongest(engine, monkeypatch):
    monkeypatch.setattr(te, "_CC_PITH_PREFETCH_WARM_ENABLED", True)
    monkeypatch.setattr(te, "_CC_PITH_PREFETCH_MAX", 2)
    engine.set_prefetch_seed(lambda: {"P1": 0.1, "P2": 0.9, "P3": 0.5})
    assert set(_merge(engine, {})) == {"P2", "P3"}


def test_dedup_keeps_stronger_current(engine, monkeypatch):
    monkeypatch.setattr(te, "_CC_PITH_PREFETCH_WARM_ENABLED", True)
    engine.set_prefetch_seed(lambda: {"A": 1.0, "B": 1.0})   # A:0.9 stays; B:0.4 < 0.5
    out = _merge(engine, BASE)
    assert out["A"] == 0.9 and out["B"] == pytest.approx(0.5)
    assert engine._prefetch_seeded == 0          # dedup never counts as a new seed


def test_missing_node_skipped(engine, monkeypatch):
    monkeypatch.setattr(te, "_CC_PITH_PREFETCH_WARM_ENABLED", True)
    engine.set_prefetch_seed(lambda: {"GHOST": 1.0, "P1": 1.0})
    assert set(_merge(engine, {})) == {"P1"}


def test_raising_seed_is_failsoft_and_logs_once(engine, monkeypatch, caplog):
    monkeypatch.setattr(te, "_CC_PITH_PREFETCH_WARM_ENABLED", True)
    def boom(): raise RuntimeError("seed exploded")
    engine.set_prefetch_seed(boom)
    with caplog.at_level("WARNING"):
        assert _merge(engine, BASE) == BASE
        assert _merge(engine, BASE) == BASE
    assert sum("Prefetch seed merge failing" in r.message for r in caplog.records) == 1


def test_same_seed_set_primed_at_most_repeats_ticks(engine, monkeypatch):
    """One prediction must not be re-primed in write mode 12-60x over its TTL."""
    monkeypatch.setattr(te, "_CC_PITH_PREFETCH_WARM_ENABLED", True)
    monkeypatch.setattr(te, "_CC_PITH_PREFETCH_REPEATS", 2)
    engine.set_prefetch_seed(lambda: {"P1": 1.0})
    assert "P1" in _merge(engine, {})
    assert "P1" in _merge(engine, {})
    assert "P1" not in _merge(engine, {})        # third tick: spent
    engine.set_prefetch_seed(lambda: {"P2": 1.0})  # new prediction -> fresh budget
    assert "P2" in _merge(engine, {})


def test_merge_runs_before_brakes_and_budget(engine, monkeypatch):
    """Structural: the heuristic path must see prefetch seeds in `seen` BEFORE
    _apply_brakes and before the max_activation_nodes slice."""
    import inspect
    src = inspect.getsource(te.TonicEngine._heuristic_inference)
    i_merge, i_brake = src.index("_merge_prefetch_seeds(seen)"), src.index("_apply_brakes(seen)")
    i_slice = src.index("max_activation_nodes]")
    assert i_merge < i_brake < i_slice
    assert "_merge_prefetch_seeds" not in inspect.getsource(te.TonicEngine._generate_latent_token_inner)
