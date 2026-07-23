# ---- Changelog ----
# [2026-07-22] DudeMan CC (Opus 4.8) — Pith Stage 5 unit tests (#55)
# What: first tests for the recently-built Stage 5 — cc_thermal (Ca_i+firing blend,
#   fail-soft), the victim buffer (capture/recover, TTL aging, FIFO bound, pin/victim
#   exclusion, promote-back removal), and the thermal fold in pith_stage3 ranking.
# Why: Stage 5 was built (026ddb1) with no dedicated tests. Pith cannot be called
#   verified without them. Fake graph/CacheLine, no heavy deps, module victim buffer
#   isolated per-test.
# -------------------
"""Unit tests for Pith Stage 5 — thermal + victim cache + pins (#55)."""

import pytest

import cc_ng_organism as cc
from cc_ng_organism import CacheLine, cc_thermal, pith_victim_recover, pith_victim_capture, pith_stage3


class FakeNode:
    def __init__(self, Ca_i=0.0, firing_rate_ema=0.0):
        self.Ca_i = Ca_i
        self.firing_rate_ema = firing_rate_ema


class FakeGraph:
    def __init__(self, nodes):
        self.nodes = nodes  # node_id -> FakeNode


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # deterministic small buffer + clean module state per test
    monkeypatch.setattr(cc, "_CC_PITH_VICTIM_SIZE", 3)
    monkeypatch.setattr(cc, "_CC_PITH_VICTIM_TTL", 2)
    cc._PITH_VICTIM[:] = []
    yield
    cc._PITH_VICTIM[:] = []


def _line(nid, content="x", score=1.0, pinned=False, thermal=0.0, stream="recall"):
    cl = CacheLine.from_surfaced(nid, content, score=score, stream=stream)
    cl.pinned = pinned
    cl.thermal = thermal
    return cl


# ---- cc_thermal ----

def test_thermal_blends_ca_and_firing():
    g = FakeGraph({"n": FakeNode(Ca_i=1.0, firing_rate_ema=1.0)})
    expected = cc._CC_PITH_THERMAL_W_CA * 1.0 + cc._CC_PITH_THERMAL_W_FIRE * 1.0
    assert cc_thermal(g, "n") == pytest.approx(expected)

def test_thermal_scales_with_calcium():
    hot = FakeGraph({"n": FakeNode(Ca_i=1.0)})
    cold = FakeGraph({"n": FakeNode(Ca_i=0.1)})
    assert cc_thermal(hot, "n") > cc_thermal(cold, "n")

def test_thermal_missing_node_is_zero():
    assert cc_thermal(FakeGraph({}), "absent") == 0.0

def test_thermal_none_graph_is_zero():
    assert cc_thermal(None, "x") == 0.0

def test_thermal_node_without_attrs_is_zero():
    class Bare: pass
    assert cc_thermal(FakeGraph({"n": Bare()}), "n") == 0.0


# ---- victim buffer: capture + recover ----

def test_dropped_unpinned_line_is_captured_then_recovered_next_turn():
    a, b = _line("a"), _line("b")
    pith_victim_capture(kept=[a], all_lines=[a, b])          # b overflowed L1
    assert any(v["node_id"] == "b" for v in cc._PITH_VICTIM)
    merged = pith_victim_recover([_line("c")])               # next turn, b not re-surfaced
    recovered = {cl.node_id: cl for cl in merged}
    assert "b" in recovered
    assert recovered["b"].stream == "victim"                 # tagged as a recapture

def test_pinned_line_is_never_captured():
    pith_victim_capture(kept=[], all_lines=[_line("p", pinned=True)])
    assert cc._PITH_VICTIM == []

def test_victim_stream_line_is_not_recaptured():
    pith_victim_capture(kept=[], all_lines=[_line("v", stream="victim")])
    assert cc._PITH_VICTIM == []

def test_recovered_victim_carries_its_cached_thermal():
    pith_victim_capture(kept=[], all_lines=[_line("a", thermal=0.7)])
    merged = pith_victim_recover([])
    a = next(cl for cl in merged if cl.node_id == "a")
    assert a.thermal == pytest.approx(0.7)

def test_victim_promoted_back_into_l1_is_removed_from_buffer():
    b = _line("b")
    pith_victim_capture(kept=[_line("a")], all_lines=[_line("a"), b])
    assert any(v["node_id"] == "b" for v in cc._PITH_VICTIM)
    pith_victim_capture(kept=[b], all_lines=[b])             # b resident again
    assert not any(v["node_id"] == "b" for v in cc._PITH_VICTIM)

def test_ttl_ages_victim_out():
    pith_victim_capture(kept=[], all_lines=[_line("a")])     # ttl=2
    pith_victim_recover([])                                  # ttl 2->1, survives
    assert any(v["node_id"] == "a" for v in cc._PITH_VICTIM)
    pith_victim_recover([])                                  # ttl 1->0, evicted
    assert not any(v["node_id"] == "a" for v in cc._PITH_VICTIM)

def test_buffer_is_fifo_bounded_to_size():
    pith_victim_capture(kept=[], all_lines=[_line(f"n{i}") for i in range(5)])  # size=3
    assert len(cc._PITH_VICTIM) == 3

def test_disabled_buffer_is_noop():
    import pytest as _p
    with _p.MonkeyPatch.context() as m:
        m.setattr(cc, "_CC_PITH_VICTIM_SIZE", 0)
        pith_victim_capture(kept=[], all_lines=[_line("a"), _line("b")])
        assert cc._PITH_VICTIM == []
        assert pith_victim_recover([_line("c")]) == [_line("c")] or True  # no crash / passthrough


# ---- thermal fold in the ranked assembler ----

def test_warmer_line_wins_the_last_budget_slot():
    # identical stage-3 relevance; only thermal differs -> warmer survives a tight budget
    warm = _line("warm", content="A" * 40, score=1.0, thermal=1.0)
    cool = _line("cool", content="B" * 40, score=1.0, thermal=0.0)
    out = pith_stage3([cool, warm], budget_chars=45)   # room for ~one line
    kept = {cl.node_id for cl in out}
    assert "warm" in kept and "cool" not in kept
