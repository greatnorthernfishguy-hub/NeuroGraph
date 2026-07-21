"""
Commons wire-stream retention (#80) — wire:tid.http.{dir}:* windowed by recency, mirrors
metrics:* (#320) / error:* (#330).

# ---- Changelog ----
# [2026-07-21] Claude Code (Sonnet 5) — #80 wire → Commons: _evict_old_wire() Task 1 acceptance
# What: Proves Commons._evict_old_wire: the wire:tid.http.{outbound|inbound} namespace is capped
#       to _WIRE_KEEP_PER_DIR most-recent synapses (by recency); experience/topology/metrics/error
#       synapses are never touched; outbound and inbound are windowed independently; a genuine
#       low-weight memory survives a wire flood (the motivating scenario, design v4 §6 — wire is
#       seeded at weight=0.5, below the metrics/error weight, and would otherwise be first in line
#       for NG-Lite's weight-based max_synapses eviction if left unwindowed).
# Why:  Design `prd/2026-07-21-wire-event-commons-peninsula-design.md` (v4) §6 / implementation
#       plan Task 1. TID's raw HTTP wire deposits land in Syl's SHARED Commons and are high-volume
#       — without windowing they'd accumulate unbounded in the weight-based max_synapses bound.
# How:  mirrors test_commons_metric_retention.py / test_commons_error_retention.py, substituting
#       the wire: namespace's actual shape — wire:tid.http.{dir}:{sha256hash} has only 3 ':'-parts
#       (grouping key = first 2 parts), unlike metrics/error's 4-part shape (grouping key = first 3).
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


def _wire_id(direction, i):
    # Stand-in for sha256(content) — distinct per deposit, no colons (matches real hex digest shape).
    return f"wire:tid.http.{direction}:h{i:08x}"


def test_wire_dir_capped_to_window():
    commons = commons_mod.Commons()
    keep = commons_mod._WIRE_KEEP_PER_DIR
    for i in range(keep + 75):   # exceed the window
        commons.deposit(_emb(i), _wire_id("outbound", i), strength=0.5,
                        metadata={"first_ts": i, "last_ts": i, "count": 1})
    n = _count(commons, "wire:tid.http.outbound:")
    assert n == keep, f"wire dir must be capped to the recency window; got {n} (keep={keep})"


def test_recency_keeps_newest():
    commons = commons_mod.Commons()
    keep = commons_mod._WIRE_KEEP_PER_DIR
    for i in range(keep + 10):
        commons.deposit(_emb(i), _wire_id("inbound", i), strength=0.5, metadata={})
    kept_seqs = sorted(
        int(s.target_id.rsplit(":h", 1)[-1], 16)
        for s in commons._ng.synapses.values()
        if getattr(s, "target_id", "").startswith("wire:tid.http.inbound:")
    )
    assert kept_seqs[0] >= 10, f"oldest deposits evicted; smallest kept seq={kept_seqs[0]}"
    assert kept_seqs[-1] == keep + 9, "the very newest is retained"


def test_outbound_inbound_independent():
    """outbound and inbound are windowed independently — one direction can't evict the other."""
    commons = commons_mod.Commons()
    keep = commons_mod._WIRE_KEEP_PER_DIR
    for i in range(keep + 50):
        commons.deposit(_emb(i), _wire_id("outbound", i), strength=0.5, metadata={})
    for i in range(5):
        commons.deposit(_emb(9000 + i), _wire_id("inbound", i), strength=0.5, metadata={})
    assert _count(commons, "wire:tid.http.outbound:") == keep
    assert _count(commons, "wire:tid.http.inbound:") == 5, "other direction untouched by the chatty one"


def test_experience_never_evicted_by_wire_retention():
    """Experience/topology/metrics/error synapses are NEVER touched by wire retention."""
    commons = commons_mod.Commons()
    keep = commons_mod._WIRE_KEEP_PER_DIR
    for i in range(20):
        commons.deposit(_emb(5000 + i), f"experience:exp{i}", metadata={"kind": "experience"})
        commons.deposit(_emb(6000 + i), f"topology:n{i}", metadata={"kind": "topology_delta"})
    for i in range(20):
        commons.deposit(_emb(7000 + i), f"metrics:neurograph:nominal:h{i}:{i}.0:{i}", metadata={})
        commons.deposit(_emb(8000 + i), f"error:immunis:ConnectionError:{i}", metadata={})
    # a flood of wire that exceeds the window many times over
    for i in range(keep + 500):
        commons.deposit(_emb(i), _wire_id("outbound", i), strength=0.5, metadata={})
    assert _count(commons, "experience:") == 20, "experience synapses untouched by wire eviction"
    assert _count(commons, "topology:") == 20, "topology synapses untouched by wire eviction"
    assert _count(commons, "metrics:neurograph:nominal:") == 20, "metrics untouched by wire eviction"
    assert _count(commons, "error:immunis:ConnectionError:") == 20, "errors untouched by wire eviction"
    assert _count(commons, "wire:tid.http.outbound:") == keep, "wire windowed"


def test_low_weight_memory_survives_wire_flood():
    """The motivating scenario (design v4 §6): wire seeds at weight=0.5, below genuine memory,
    and would be the weight-based max_synapses bound's first casualty if left unwindowed. With
    _evict_old_wire capping the wire: namespace by recency (not weight), a genuine low-weight
    memory synapse must survive a wire flood that vastly exceeds the keep window.

    Wire embeddings are drawn from a small reused pool (NG-Lite node-level max_nodes=1000 is a
    separate, orthogonal bound to the synapse-level windowing under test here — each flood
    deposit still gets a distinct target_id/synapse on a shared node, exactly like repeated wire
    traffic through the same few HTTP endpoints would).
    """
    commons = commons_mod.Commons()
    keep = commons_mod._WIRE_KEEP_PER_DIR
    # A genuine memory, deliberately low-weight/low-strength — the synapse most at risk from
    # weight-based global eviction if wire were allowed to compete unwindowed.
    commons.deposit(_emb(4242), "experience:fragile_low_weight_memory", strength=0.1,
                    metadata={"kind": "experience"})
    for i in range(keep * 5):
        commons.deposit(_emb(i % 20), _wire_id("outbound", i), strength=0.5, metadata={})
    assert _count(commons, "experience:fragile_low_weight_memory") == 1, \
        "low-weight genuine memory must survive a wire flood many times the keep window"
    assert _count(commons, "wire:tid.http.outbound:") == keep, "wire still capped at the keep window"


def test_non_wire_deposit_unaffected():
    """A non-wire deposit never triggers wire eviction logic (returns normally)."""
    commons = commons_mod.Commons()
    r = commons.deposit(_emb(1), "experience:hello", metadata={"kind": "experience"})
    assert r and _count(commons, "experience:") == 1


def test_wire_keep_per_dir_env_configurable():
    """WIRE_KEEP_PER_DIR (LAW 5) is read from the environment at import time."""
    assert commons_mod._WIRE_KEEP_PER_DIR == int(os.environ.get("WIRE_KEEP_PER_DIR", "200"))


if __name__ == "__main__":
    test_wire_dir_capped_to_window();                     print("PASS wire dir capped to recency window")
    test_recency_keeps_newest();                          print("PASS recency-keeps-newest (oldest evicted)")
    test_outbound_inbound_independent();                  print("PASS outbound/inbound independent windows")
    test_experience_never_evicted_by_wire_retention();    print("PASS experience/topology/metrics/error NEVER evicted by wire retention")
    test_low_weight_memory_survives_wire_flood();         print("PASS low-weight genuine memory survives a wire flood")
    test_non_wire_deposit_unaffected();                   print("PASS non-wire deposit unaffected")
    test_wire_keep_per_dir_env_configurable();            print("PASS WIRE_KEEP_PER_DIR env-configurable (LAW 5)")
    print("\nCommons wire retention (#80): ALL PASS — wire windowed by recency, memory never crowded out")
