"""
Commons error-stream retention (#330) — error:* windowed by recency, mirrors metrics:* (#320).

# ---- Changelog ----
# [2026-07-05] Claude Code (Sonnet 5) — #330 operational-logger error:* retention
# What: Proves Commons._evict_old_errors: the error:<module_id>:<ExcType> namespace is capped
#       to _ERRORS_KEEP_PER_KIND most-recent synapses (by recency); experience/topology/metrics
#       synapses are never touched; per-kind independence.
# Why: error:* is the operational-logger's new namespace (signal_error() on ng_commons_eco.py).
#      Without retention, a noisy module's exceptions would accumulate unbounded and could
#      pressure NG-Lite's weight-based max_synapses bound the same way unretained metrics would.
# How: mirrors test_commons_metric_retention.py exactly, substituting the error: namespace.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commons as commons_mod


def _emb(seed, dim=768):
    r = np.random.RandomState(seed)
    v = r.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _count(commons, prefix):
    return sum(1 for s in commons._ng.synapses.values()
               if getattr(s, "target_id", "").startswith(prefix))


def test_error_kind_capped_to_window():
    commons = commons_mod.Commons()
    keep = commons_mod._ERRORS_KEEP_PER_KIND
    for i in range(keep + 75):
        commons.deposit(_emb(i), f"error:immunis:ConnectionError:{i}", metadata={"kind": "error"})
    n = _count(commons, "error:immunis:ConnectionError:")
    assert n == keep, f"error kind must be capped to the recency window; got {n} (keep={keep})"


def test_per_module_type_independent():
    """Two error module:type kinds are windowed independently."""
    commons = commons_mod.Commons()
    keep = commons_mod._ERRORS_KEEP_PER_KIND
    for i in range(keep + 50):
        commons.deposit(_emb(i), f"error:immunis:ConnectionError:{i}", metadata={})
    for i in range(5):
        commons.deposit(_emb(9000 + i), f"error:thc:ValueError:{i}", metadata={})
    assert _count(commons, "error:immunis:ConnectionError:") == keep
    assert _count(commons, "error:thc:ValueError:") == 5, "other kind untouched by the chatty one"


def test_experience_never_evicted_by_error_retention():
    commons = commons_mod.Commons()
    keep = commons_mod._ERRORS_KEEP_PER_KIND
    for i in range(20):
        commons.deposit(_emb(5000 + i), f"experience:exp{i}", metadata={"kind": "experience"})
    for i in range(keep + 500):
        commons.deposit(_emb(i), f"error:elmer:RuntimeError:{i}", metadata={})
    assert _count(commons, "experience:") == 20, "experience synapses untouched by error eviction"
    assert _count(commons, "error:elmer:RuntimeError:") == keep, "errors windowed"


def test_non_error_deposit_unaffected():
    commons = commons_mod.Commons()
    r = commons.deposit(_emb(1), "experience:hello", metadata={"kind": "experience"})
    assert r and _count(commons, "experience:") == 1


if __name__ == "__main__":
    test_error_kind_capped_to_window();                    print("PASS error kind capped to recency window")
    test_per_module_type_independent();                    print("PASS per-module:type independent windows")
    test_experience_never_evicted_by_error_retention();    print("PASS experience never evicted by error retention")
    test_non_error_deposit_unaffected();                   print("PASS non-error deposit unaffected")
    print("\nCommons error retention (#330): ALL PASS")
