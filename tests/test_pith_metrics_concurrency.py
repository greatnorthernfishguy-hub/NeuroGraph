# ---- Changelog ----
# [2026-09-10] Claude Code (DudeMan CC, Opus 5) — D5b: metrics thread-safety + resolved config
# What: concurrency tests for the REAL PithMetrics (no stubs) -- coherent snapshots under
#   concurrent updates and resets, no lost updates, and the numerator<=denominator
#   invariant holding at every observation. Plus pith_effective_config() coverage.
# Why: the daemon takes snapshots from a telemetry thread while recalls run on request
#   threads. `+=` on an attribute is load-add-store, so concurrent recalls could lose
#   updates, and an unguarded snapshot could pair a numerator from one instant with a
#   denominator from another -- producing a ratio that is silently wrong rather than
#   obviously broken. These tests are what make "coherent" a checked claim.
# How: real _PITH_METRICS and real pith_stage3 driven from many threads. No graph, no
#   daemon, no substrate, no live NG state.
# -------------------
"""D5b: PithMetrics thread-safety and the resolved-configuration authority."""
import os
import threading

import pytest

import cc_ng_organism as cc
from cc_ng_organism import CacheLine, pith_stage3


def _line(nid, score, stream="pattern", prefetch=False):
    return CacheLine.from_surfaced(node_id=nid, content=f"c-{nid}", score=score,
                                   stream=stream, prefetch_origin=prefetch)


@pytest.fixture(autouse=True)
def _reset_metrics():
    cc._PITH_METRICS.reset()
    yield
    cc._PITH_METRICS.reset()


# ------------------------------------------------------------ recall-turn counter

def test_each_invocation_counts_one_recall_turn():
    """l1_assemblies is the denominator-of-record for the acceptance bar."""
    for _ in range(5):
        pith_stage3([_line("a", 10.0)], budget_chars=5000)
    assert cc._PITH_METRICS.l1_assemblies == 5


def test_empty_recall_still_counts_as_a_turn():
    """A turn that surfaced nothing is still a turn -- otherwise the rate inflates."""
    pith_stage3([], budget_chars=5000)
    assert cc._PITH_METRICS.l1_assemblies == 1
    assert cc._PITH_METRICS.l1_kept_distinct == 0


def test_turn_count_is_exposed_in_snapshot():
    pith_stage3([_line("a", 10.0)], budget_chars=5000)
    assert cc._PITH_METRICS.snapshot()["l1_assemblies"] == 1


# ------------------------------------------------------------ thread safety

def test_no_lost_updates_under_concurrent_recalls():
    """The exact-count property: load-add-store races would show up as a shortfall."""
    threads, per_thread = 8, 40
    lines = [_line("a", 30.0, prefetch=True), _line("b", 20.0), _line("c", 10.0, prefetch=True)]

    def worker():
        for _ in range(per_thread):
            pith_stage3(list(lines), budget_chars=100000)

    ts = [threading.Thread(target=worker) for _ in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in ts), "worker thread hung"

    total = threads * per_thread
    m = cc._PITH_METRICS
    assert m.l1_assemblies == total, f"lost turn updates: {m.l1_assemblies} != {total}"
    assert m.l1_kept_distinct == total * 3, f"lost kept updates: {m.l1_kept_distinct}"
    assert m.l1_prefetch_distinct == total * 2, f"lost prefetch updates: {m.l1_prefetch_distinct}"


def test_snapshots_are_never_torn_under_concurrent_updates():
    """Every observation must satisfy numerator <= denominator.

    A torn read pairs a numerator raised by one recall with a denominator not yet
    raised by it -- which for a ratio means a value that can exceed 1.0 while
    looking perfectly well-formed.
    """
    stop = threading.Event()
    violations = []
    lines = [_line("a", 30.0, prefetch=True), _line("b", 20.0, prefetch=True),
             _line("c", 10.0), _line("d", 5.0)]

    def writer():
        while not stop.is_set():
            pith_stage3(list(lines), budget_chars=100000)

    def reader():
        while not stop.is_set():
            s = cc._PITH_METRICS.snapshot()
            if s["l1_prefetch_distinct"] > s["l1_kept_distinct"]:
                violations.append(("broad", s))
            if s["l1_prefetch_distinct_promotable"] > s["l1_kept_distinct_promotable"]:
                violations.append(("narrow", s))

    ts = [threading.Thread(target=writer) for _ in range(4)]
    ts += [threading.Thread(target=reader) for _ in range(3)]
    for t in ts:
        t.start()
    threading.Event().wait(1.5)
    stop.set()
    for t in ts:
        t.join(timeout=30)
    assert not violations, f"torn snapshot observed: {violations[:3]}"


def test_snapshot_is_coherent_across_a_concurrent_reset():
    """A snapshot must be all-pre-reset or all-post-reset, never a mixture."""
    stop = threading.Event()
    mixtures = []
    lines = [_line("a", 30.0, prefetch=True), _line("b", 20.0, prefetch=True)]

    def writer():
        while not stop.is_set():
            pith_stage3(list(lines), budget_chars=100000)

    def resetter():
        while not stop.is_set():
            cc._PITH_METRICS.reset()

    def reader():
        while not stop.is_set():
            s = cc._PITH_METRICS.snapshot()
            # Post-reset, every sec 13.3 term is zero together. A mixture -- some
            # zeroed, some not, with turns already counted -- means a torn view.
            terms = (s["l1_kept_distinct"], s["l1_prefetch_distinct"],
                     s["l1_kept_distinct_promotable"], s["l1_prefetch_distinct_promotable"])
            if s["l1_prefetch_distinct"] > s["l1_kept_distinct"]:
                mixtures.append(s)
            if any(t < 0 for t in terms):
                mixtures.append(s)

    ts = [threading.Thread(target=writer) for _ in range(3)]
    ts += [threading.Thread(target=resetter), threading.Thread(target=reader),
           threading.Thread(target=reader)]
    for t in ts:
        t.start()
    threading.Event().wait(1.5)
    stop.set()
    for t in ts:
        t.join(timeout=30)
    assert not mixtures, f"incoherent snapshot across reset: {mixtures[:3]}"


def test_reset_zeroes_every_term_together():
    pith_stage3([_line("a", 10.0, prefetch=True)], budget_chars=5000)
    assert cc._PITH_METRICS.l1_kept_distinct > 0
    cc._PITH_METRICS.reset()
    s = cc._PITH_METRICS.snapshot()
    for k in ("l1_kept_distinct", "l1_prefetch_distinct", "l1_kept_distinct_promotable",
              "l1_prefetch_distinct_promotable", "l1_assemblies"):
        assert s[k] == 0, f"{k} survived a reset"


def test_lock_is_not_measured_state():
    """The lock must not appear in the snapshot -- it is machinery."""
    assert "_lock" not in cc._PITH_METRICS.snapshot()


# ------------------------------------------------------------ resolved configuration

def test_effective_config_separates_env_from_resolved():
    c = cc.pith_effective_config()
    assert set(c) == {"env", "resolved", "authority"}
    assert set(c["env"]) == set(cc._PITH_CONFIG_KEYS)
    assert set(c["resolved"]) == set(cc._PITH_CONFIG_KEYS)


def test_unset_env_still_reports_a_resolved_default(monkeypatch):
    """The degraded-daemon trap: env=None must not read as 'setting absent'."""
    monkeypatch.delenv("CC_PITH_L1_BUDGET", raising=False)
    c = cc.pith_effective_config()
    assert c["env"]["CC_PITH_L1_BUDGET"] is None
    assert isinstance(c["resolved"]["CC_PITH_L1_BUDGET"], int)
    assert c["resolved"]["CC_PITH_L1_BUDGET"] > 0


def test_resolved_values_are_typed_not_raw_strings():
    r = cc.pith_effective_config()["resolved"]
    assert isinstance(r["CC_PITH_ENABLED"], bool)
    assert isinstance(r["CC_PITH_L1_BUDGET"], int)
    if r["CC_PITH_PREFETCH_MAX"] is not None:
        assert isinstance(r["CC_PITH_PREFETCH_MAX"], int)


def test_resolved_values_are_clamped_not_echoed():
    """CC_PITH_PREFETCH_MAX is clamped to 0..64 by its owner; 15 is in range."""
    r = cc.pith_effective_config()["resolved"]
    if r["CC_PITH_PREFETCH_MAX"] is not None:
        assert 0 <= r["CC_PITH_PREFETCH_MAX"] <= 64
    if r["CC_PITH_PREFETCH_CURRENT_SCALE"] is not None:
        assert 0.0 <= r["CC_PITH_PREFETCH_CURRENT_SCALE"] <= 1.0


def test_authority_names_the_owning_module():
    a = cc.pith_effective_config()["authority"]
    assert a["CC_PITH_ENABLED"] == "cc_ng_organism"
    assert a["CC_PITH_PREFETCH_MAX"].startswith("tonic_engine")


def test_config_is_an_allowlist_and_leaks_nothing(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN_MUST_NOT_APPEAR", "sentinel-value")
    blob = repr(cc.pith_effective_config())
    assert "sentinel-value" not in blob
    assert "SECRET_TOKEN_MUST_NOT_APPEAR" not in blob


def test_config_never_raises_without_tonic(monkeypatch):
    """Fail-soft: a missing tonic_engine must not sink a snapshot."""
    import sys
    monkeypatch.setitem(sys.modules, "tonic_engine", None)
    c = cc.pith_effective_config()          # must not raise
    assert c["resolved"]["CC_PITH_PREFETCH_MAX"] is None
    assert "unavailable" in c["authority"]["CC_PITH_PREFETCH_MAX"]
