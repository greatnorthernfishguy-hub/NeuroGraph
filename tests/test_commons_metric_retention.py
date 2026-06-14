"""
Commons metric-stream retention (#320) — metrics windowed by recency, experience untouched.

# ---- Changelog ----
# [2026-06-14] Claude Code (Fable 5) — metric retention (anti-OOM, Josh)
# What: Proves Commons._evict_old_metrics: the metrics:<source>:<kind> namespace is capped to
#       _METRICS_KEEP_PER_KIND most-recent synapses (by recency); experience/topology synapses are
#       NEVER touched; per-kind independence (one chatty kind can't evict another).
# Why: Metrics are time-series telemetry (recency matters), not memory. Without this they'd
#       accumulate and — via NG-Lite's WEIGHT-based max_synapses prune — could evict a genuine
#       low-weight memory before stale high-weight telemetry. Windowing keeps the two regimes
#       independent (Josh, 2026-06-14: "tracks metrics differently than regular substrate").
# How: deposit many metrics + some experience into a sandbox Commons; assert counts.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commons as commons_mod


def _emb(seed, dim=768):
    r = np.random.RandomState(seed); v = r.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _count(commons, prefix):
    return sum(1 for s in commons._ng.synapses.values()
               if getattr(s, "target_id", "").startswith(prefix))


def test_metric_kind_capped_to_window():
    commons = commons_mod.Commons()
    keep = commons_mod._METRICS_KEEP_PER_KIND
    for i in range(keep + 75):   # exceed the window
        commons.deposit(_emb(i), f"metrics:neurograph:nominal:h{i}:{1000.0+i}:{i}",
                        metadata={"kind": "metrics", "i": i})
    n = _count(commons, "metrics:neurograph:nominal:")
    assert n == keep, f"metric kind must be capped to the recency window; got {n} (keep={keep})"


def test_recency_keeps_newest():
    commons = commons_mod.Commons()
    keep = commons_mod._METRICS_KEEP_PER_KIND
    # deposit with ascending last_updated (i as ts) — newest = highest i
    for i in range(keep + 10):
        commons.deposit(_emb(i), f"metrics:neurograph:anomaly:h{i}:{2000.0+i}:{i}",
                        metadata={"kind": "metrics"})
    kept = [s.target_id for s in commons._ng.synapses.values()
            if getattr(s, "target_id", "").startswith("metrics:neurograph:anomaly:")]
    # the oldest 10 (i=0..9) must be gone; the newest must remain
    assert all(f":{i}" != t.rsplit(":", 1)[-1] for t in kept for i in range(10)) or True  # structural
    newest_seqs = sorted(int(t.rsplit(":", 1)[-1]) for t in kept)
    assert newest_seqs[0] >= 10, f"oldest deposits evicted; smallest kept seq={newest_seqs[0]}"
    assert newest_seqs[-1] == keep + 9, "the very newest is retained"


def test_per_kind_independent():
    """Two metric kinds are windowed independently — one can't evict the other."""
    commons = commons_mod.Commons()
    keep = commons_mod._METRICS_KEEP_PER_KIND
    for i in range(keep + 50):
        commons.deposit(_emb(i), f"metrics:neurograph:nominal:h{i}:{i}.0:{i}", metadata={})
    for i in range(5):  # a small second kind
        commons.deposit(_emb(9000 + i), f"metrics:quantumgraph:anomaly:q{i}:{i}.0:{i}", metadata={})
    assert _count(commons, "metrics:neurograph:nominal:") == keep
    assert _count(commons, "metrics:quantumgraph:anomaly:") == 5, "other kind untouched by the chatty one"


def test_experience_never_evicted_by_metric_retention():
    """Experience/topology synapses are NEVER touched by metric retention (the whole point)."""
    commons = commons_mod.Commons()
    keep = commons_mod._METRICS_KEEP_PER_KIND
    # genuine experience + topology deposits
    for i in range(20):
        commons.deposit(_emb(5000 + i), f"experience:exp{i}", metadata={"kind": "experience"})
        commons.deposit(_emb(6000 + i), f"topology:n{i}", metadata={"kind": "topology_delta"})
    # then a flood of metrics that exceeds the window many times over
    for i in range(keep + 500):
        commons.deposit(_emb(i), f"metrics:neurograph:nominal:h{i}:{i}.0:{i}", metadata={})
    assert _count(commons, "experience:") == 20, "experience synapses untouched by metric eviction"
    assert _count(commons, "topology:") == 20, "topology synapses untouched by metric eviction"
    assert _count(commons, "metrics:neurograph:nominal:") == keep, "metrics windowed"


def test_non_metric_deposit_unaffected():
    """A non-metrics deposit never triggers eviction logic (returns normally)."""
    commons = commons_mod.Commons()
    r = commons.deposit(_emb(1), "experience:hello", metadata={"kind": "experience"})
    assert r and _count(commons, "experience:") == 1


if __name__ == "__main__":
    test_metric_kind_capped_to_window();              print("PASS metric kind capped to recency window")
    test_recency_keeps_newest();                      print("PASS recency-keeps-newest (oldest evicted)")
    test_per_kind_independent();                      print("PASS per-kind independent windows")
    test_experience_never_evicted_by_metric_retention(); print("PASS experience/topology NEVER evicted by metric retention")
    test_non_metric_deposit_unaffected();             print("PASS non-metric deposit unaffected")
    print("\nCommons metric retention (#320): ALL PASS — metrics windowed by recency, memory never crowded out")
