"""
ng_salience_gate.py (VENDORED) — standalone tests for the parameterized salience gate.

# ---- Changelog ----
# [2026-06-14] Claude Code (Fable 5) — vendored salience gate tests (#320 Part 2)
# What: Proves SalienceGate independent of any module: salience-gated granular/nominal/run-length,
#       competence graduation (asymmetric, no runaway), parameterized salience_fn + signature_fn,
#       fail-soft. Uses injected commons_provider/embed_fn (no monkeypatch).
# -------------------
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commons as commons_mod
from ng_salience_gate import SalienceGate


def _fake_embed(text, *a, **k):
    rng = np.random.RandomState(abs(hash(text)) % (2**31))
    v = rng.randn(768).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _surprise(m):
    t = m.get("predictions_confirmed", 0) + m.get("predictions_surprised", 0)
    return (m.get("predictions_surprised", 0) / t) if t else 0.0


def _gate(commons, **kw):
    return SalienceGate(
        "neurograph", _surprise,
        agg_fields=("fired_nodes", "predictions_confirmed", "predictions_surprised"),
        signature_fn=lambda m, s: (round(s, 1), m.get("synapses_pruned", 0) > 0,
                                   m.get("synapses_sprouted", 0) > 0),
        commons_provider=lambda: commons, embed_fn=_fake_embed, **kw,
    )


def _nominal():
    return {"fired_nodes": 3, "synapses_pruned": 0, "synapses_sprouted": 0,
            "predictions_confirmed": 10, "predictions_surprised": 0}


def _anomaly(surprised=8, confirmed=2, pruned=0, sprouted=0):
    return {"fired_nodes": 5, "synapses_pruned": pruned, "synapses_sprouted": sprouted,
            "predictions_confirmed": confirmed, "predictions_surprised": surprised}


def _deposits(commons, salience=None):
    out = []
    for s in commons._ng.synapses.values():
        tid = getattr(s, "target_id", "")
        if tid.startswith("metrics:"):
            meta = s.metadata.get("last_context", {})
            if salience is None or meta.get("salience") == salience:
                out.append(meta)
    return out


def test_no_flood_and_aggregate():
    c = commons_mod.Commons(); g = _gate(c)
    for _ in range(g._flush_every - 1):
        g.observe(_nominal())
    assert _deposits(c) == [], "nominal steps must not flood"
    g.observe(_nominal())
    nom = _deposits(c, "nominal")
    assert len(nom) == 1 and nom[0]["span_steps"] == g._flush_every
    assert nom[0]["aggregate"]["predictions_confirmed"] == 10 * g._flush_every, "aggregate, never blind"
    assert "gate_competence" in nom[0], "competence surfaced for observability"


def test_anomaly_granular_flushes_nominal():
    c = commons_mod.Commons(); g = _gate(c)
    for _ in range(5):
        g.observe(_nominal())
    g.observe(_anomaly())
    assert len(_deposits(c, "nominal")) == 1 and _deposits(c, "nominal")[0]["span_steps"] == 5
    an = _deposits(c, "anomaly")
    assert len(an) == 1 and an[0]["signal"] == 0.8 and an[0]["source"] == "neurograph"


def test_run_length():
    c = commons_mod.Commons(); g = _gate(c)
    for _ in range(826):
        g.observe(_anomaly())
    g.observe(_nominal())
    assert len(_deposits(c, "anomaly")) == 1, "only first of the run is granular"
    run = _deposits(c, "anomaly_run")
    assert len(run) == 1 and run[0]["repeats"] == 826


def test_distinct_anomalies_each_granular():
    c = commons_mod.Commons(); g = _gate(c)
    g.observe(_anomaly(pruned=0, sprouted=0))
    g.observe(_anomaly(pruned=5, sprouted=0))
    g.observe(_anomaly(pruned=5, sprouted=3))
    assert len(_deposits(c, "anomaly")) == 3, "distinct signatures each deposit granular"


def test_competence_asymmetric_no_runaway():
    c = commons_mod.Commons(); g = _gate(c)
    assert g._competence == 0.0 and g._effective_threshold() == g._t_min
    for _ in range(10):
        g.observe(_nominal())
    calm = g._competence
    assert 0 < calm <= 10 * g._gain + 1e-9 and g._effective_threshold() > g._t_min
    g.observe(_anomaly())
    assert (calm - g._competence) >= g._loss - 1e-9, "fast loss on spike"
    for _ in range(50):
        g.observe(_anomaly())
    assert g._competence == 0.0 and g._effective_threshold() == g._t_min, "re-sensitized, no runaway"


def test_parameterized_for_a_different_module():
    """A DIFFERENT module's salience signal (QG-style) works with no code change — just params."""
    c = commons_mod.Commons()
    # QG-style: salience from interference intensity, normalized
    g = SalienceGate("quantumgraph",
                     lambda m: min(1.0, m.get("interference_events", 0) / 10.0),
                     agg_fields=("nodes_in_superposition", "interference_events"),
                     commons_provider=lambda: c, embed_fn=_fake_embed)
    g.observe({"interference_events": 0, "nodes_in_superposition": 4})   # calm
    g.observe({"interference_events": 9, "nodes_in_superposition": 7})   # salient
    an = _deposits(c, "anomaly")
    assert len(an) == 1 and an[0]["source"] == "quantumgraph", "vendored gate serves any module by params"


def test_failsoft():
    c = commons_mod.Commons()
    g = SalienceGate("x", _surprise, commons_provider=lambda: None, embed_fn=_fake_embed)
    g.observe(_anomaly())  # no commons — must not raise
    def _boom(*a, **k): raise RuntimeError("embed down")
    g2 = SalienceGate("x", _surprise, commons_provider=lambda: c, embed_fn=_boom)
    g2.observe(_anomaly())  # embed error — must not raise


if __name__ == "__main__":
    test_no_flood_and_aggregate();            print("PASS no flood; nominal summary carries aggregate + competence")
    test_anomaly_granular_flushes_nominal();  print("PASS anomaly granular + flushes preceding nominal span")
    test_run_length();                        print("PASS run-length (826) — one granular + one run summary")
    test_distinct_anomalies_each_granular();  print("PASS distinct anomalies each granular")
    test_competence_asymmetric_no_runaway();  print("PASS competence asymmetric (slow/fast), no runaway")
    test_parameterized_for_a_different_module(); print("PASS parameterized — serves a different module (QG) by params")
    test_failsoft();                          print("PASS fail-soft (no-Commons + embed-error)")
    print("\nng_salience_gate (vendored): ALL PASS — one parameterized gate, every module owns its instance")
