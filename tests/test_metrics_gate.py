"""
Substrate Metrics Pipeline Part 1a — salience-gated metrics deposit into the Commons.

# ---- Changelog ----
# [2026-06-14] Claude Code (Fable 5) — metrics gate (#320 Part 1a)
# What: Proves _SubstrateMetricsGate: anomaly (substrate surprise) → granular deposit; nominal
#       spans → aggregate-count summary (never blind); repeated identical anomalies → run-length;
#       fail-soft; bounded (no per-step flood). Static threshold (competence is Part 1b).
# Why: design prd/substrate-metrics-pipeline-design.md — metrics are substrate concern; salience
#       gating bounds volume at the source (OOM's First Law). LAW 7: telemetry gated by the
#       substrate's OWN surprise.
# How: drive the real gate against a sandbox Commons; patch ng_embed.embed (no ONNX); inspect the
#       Commons synapses (target_id + metadata) to assert what was deposited.
# -------------------
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neurograph_rpc as rpc
import commons as commons_mod
import ng_embed

assert getattr(rpc, "_memory", None) is None, "test must not run against a live NeuroGraphMemory"


def _fake_embed(text, *a, **k):
    rng = np.random.RandomState(abs(hash(text)) % (2**31))
    v = rng.randn(768).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _nominal(i=1):
    return {"fired_nodes": 3, "fired_hyperedges": 1, "synapses_pruned": 0, "synapses_sprouted": 0,
            "predictions_confirmed": 10, "predictions_surprised": 0, "total_nodes": 1000}


def _anomaly(surprised=8, confirmed=2, pruned=0, sprouted=0):
    return {"fired_nodes": 5, "fired_hyperedges": 2, "synapses_pruned": pruned, "synapses_sprouted": sprouted,
            "predictions_confirmed": confirmed, "predictions_surprised": surprised, "total_nodes": 1000}


def _setup():
    commons = commons_mod.Commons()
    gate = rpc._SubstrateMetricsGate()
    orig_embed, orig_getc = ng_embed.embed, commons_mod.get_commons
    ng_embed.embed = _fake_embed
    commons_mod.get_commons = lambda: commons
    return commons, gate, (orig_embed, orig_getc)


def _restore(orig):
    ng_embed.embed, commons_mod.get_commons = orig


def _deposits(commons, salience=None):
    out = []
    for s in commons._ng.synapses.values():
        tid = getattr(s, "target_id", "")
        if tid.startswith("metrics:"):
            meta = s.metadata.get("last_context", {})
            if salience is None or meta.get("salience") == salience:
                out.append((tid, meta))
    return out


def test_nominal_steps_do_not_flood():
    """A run of nominal steps deposits NOTHING until the flush threshold — bounded at source."""
    commons, gate, orig = _setup()
    try:
        for _ in range(gate.NOMINAL_FLUSH_EVERY - 1):
            gate.observe(_nominal())
        assert _deposits(commons) == [], "nominal steps must not deposit per-step (no flood)"
        gate.observe(_nominal())  # crosses the flush threshold
        nominal = _deposits(commons, "nominal")
        assert len(nominal) == 1, "exactly one nominal-span summary at the flush boundary"
        agg = nominal[0][1]["aggregate"]
        assert nominal[0][1]["span_steps"] == gate.NOMINAL_FLUSH_EVERY
        assert agg["predictions_confirmed"] == 10 * gate.NOMINAL_FLUSH_EVERY, "summary carries aggregate (never blind)"
        assert agg["predictions_surprised"] == 0
    finally:
        _restore(orig)


def test_anomaly_deposits_granular_and_flushes_nominal():
    """An anomaly deposits granular AND flushes the preceding nominal span (order preserved)."""
    commons, gate, orig = _setup()
    try:
        for _ in range(5):
            gate.observe(_nominal())     # short nominal span (below flush)
        gate.observe(_anomaly())         # anomaly breaks it
        nominal = _deposits(commons, "nominal")
        anomaly = _deposits(commons, "anomaly")
        assert len(nominal) == 1 and nominal[0][1]["span_steps"] == 5, "preceding nominal span summarized"
        assert len(anomaly) == 1, "the anomaly deposited granular"
        assert anomaly[0][1]["surprise"] >= gate.SURPRISE_THRESHOLD
        assert anomaly[0][1]["predictions_surprised"] == 8
    finally:
        _restore(orig)


def test_repeated_anomaly_run_length():
    """Identical consecutive anomalies → 1 granular + 1 run-length summary, not N deposits."""
    commons, gate, orig = _setup()
    try:
        for _ in range(826):
            gate.observe(_anomaly())     # same signature, 826× in a row
        gate.observe(_nominal())         # break the run → flush run-length
        anomaly = _deposits(commons, "anomaly")
        run = _deposits(commons, "anomaly_run")
        assert len(anomaly) == 1, "only the FIRST of the run deposits granular"
        assert len(run) == 1, "the run collapses to one run-length summary"
        assert run[0][1]["repeats"] == 826, f"accurate count; got {run[0][1].get('repeats')}"
    finally:
        _restore(orig)


def test_distinct_anomalies_each_granular():
    """Different anomaly signatures each deposit granular (not run-lengthed together)."""
    commons, gate, orig = _setup()
    try:
        gate.observe(_anomaly(pruned=0, sprouted=0))
        gate.observe(_anomaly(pruned=5, sprouted=0))   # different signature
        gate.observe(_anomaly(pruned=5, sprouted=3))   # different again
        assert len(_deposits(commons, "anomaly")) == 3, "distinct anomalies each deposit granular"
    finally:
        _restore(orig)


def test_gate_failsoft_no_commons_and_bad_embed():
    """No Commons → no-op; embed error → fail-soft. Never raises."""
    commons, gate, orig = _setup()
    try:
        commons_mod.get_commons = lambda: None
        gate.observe(_anomaly())                         # no Commons — must not raise
        commons_mod.get_commons = lambda: commons
        def _boom(*a, **k): raise RuntimeError("embed down")
        ng_embed.embed = _boom
        for _ in range(gate.NOMINAL_FLUSH_EVERY + 1):
            gate.observe(_nominal())                     # embed fails at flush — must not raise
        gate.observe(_anomaly())
    finally:
        _restore(orig)


if __name__ == "__main__":
    test_nominal_steps_do_not_flood();                print("PASS nominal steps don't flood; summary carries aggregate")
    test_anomaly_deposits_granular_and_flushes_nominal(); print("PASS anomaly → granular + flushes preceding nominal span")
    test_repeated_anomaly_run_length();               print("PASS repeated anomaly → 1 granular + run-length(826)")
    test_distinct_anomalies_each_granular();           print("PASS distinct anomalies each deposit granular")
    test_gate_failsoft_no_commons_and_bad_embed();     print("PASS fail-soft: no-Commons + embed-error never raise")
    print("\nSubstrate Metrics Pipeline Part 1a: ALL PASS — salience-gated, bounded at source, never blind")
