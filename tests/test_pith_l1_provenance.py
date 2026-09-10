# ---- Changelog ----
# [2026-09-10] Claude Code (DudeMan CC, Opus 5) — D5 tests: sec 13.3 same-stage counters
# What: non-interference (emitted list identical with the flag set vs unset), correct
#   counts under duplicates and under budget drops, zero-denominator handling, and
#   numerator-subset-of-denominator.
# Why: the flag exists to be counted, never to be ranked on. If setting it can change
#   what pith_stage3 emits, the instrumentation alters the system it measures and every
#   number it produces is void. That is the test the whole D5 commit rests on.
# How: pith_stage3 called directly with hand-built CacheLines -- no graph, no daemon,
#   no substrate. Counters reset per test via the module-level _PITH_METRICS.
# -------------------
"""D5: spec sec 13.3 same-stage L1 provenance counters."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import cc_ng_organism as cc
from cc_ng_organism import CacheLine, pith_stage3


def _line(nid, score, stream="pattern", pinned=False, prefetch=False, content=None):
    return CacheLine.from_surfaced(
        node_id=nid, content=content if content is not None else f"content for {nid}",
        score=score, pinned=pinned, stream=stream, prefetch_origin=prefetch)


@pytest.fixture(autouse=True)
def _reset_metrics():
    cc._PITH_METRICS.reset()
    yield
    cc._PITH_METRICS.reset()


# ---- the load-bearing one: the flag must not change what is emitted ----

def test_flag_does_not_change_emitted_content_or_order():
    """Identical input, flag set on a subset vs unset -> byte-identical output."""
    def build(flagged):
        return [_line(f"n{i}", score=100.0 - i, prefetch=(flagged and i % 2 == 0))
                for i in range(12)]
    out_unset = pith_stage3(build(False), budget_chars=400)
    out_set = pith_stage3(build(True), budget_chars=400)
    assert [c.node_id for c in out_set] == [c.node_id for c in out_unset], "ORDER changed"
    assert [c.content for c in out_set] == [c.content for c in out_unset], "CONTENT changed"
    assert len(out_set) == len(out_unset)


def test_flag_does_not_change_selection_under_a_tight_budget():
    """A tight budget forces drops; the flag must not influence who survives."""
    def build(flagged):
        return [_line(f"n{i}", score=50.0 - i, prefetch=(flagged and i < 3), content="x" * 90)
                for i in range(10)]
    kept_unset = [c.node_id for c in pith_stage3(build(False), budget_chars=200)]
    kept_set = [c.node_id for c in pith_stage3(build(True), budget_chars=200)]
    assert kept_set == kept_unset
    assert len(kept_unset) < 10, "budget did not actually drop anything -- test is vacuous"


def test_flag_absent_from_every_ranking_expression():
    """Structural: prefetch_origin appears only at construction and the count site."""
    import inspect
    src = inspect.getsource(cc.pith_stage3)
    ranking = [ln for ln in src.splitlines()
               if "prefetch_origin" in ln and "_is_pf" not in ln and "#" not in ln.split("prefetch_origin")[0]]
    assert not ranking, f"prefetch_origin referenced outside the counting block: {ranking}"


# ---- counting correctness ----

def test_counts_distinct_nodes_not_lines():
    """Duplicate node_ids in one invocation count once, in both terms."""
    lines = [_line("dup", 90.0, prefetch=True), _line("dup", 80.0, prefetch=True),
             _line("solo", 70.0)]
    pith_stage3(lines, budget_chars=5000)
    m = cc._PITH_METRICS
    assert m.l1_kept_distinct == 2, m.snapshot()
    assert m.l1_prefetch_distinct == 1, "duplicate inflated the numerator"


def test_dropped_lines_are_not_counted():
    """Budget-dropped lines are absent from both terms -- counting is post-cut."""
    lines = [_line(f"n{i}", score=50.0 - i, prefetch=True, content="y" * 120) for i in range(8)]
    out = pith_stage3(lines, budget_chars=260)
    m = cc._PITH_METRICS
    assert m.l1_kept_distinct == len({c.node_id for c in out})
    assert m.l1_kept_distinct < 8, "nothing was dropped -- test is vacuous"


def test_numerator_is_a_subset_of_denominator():
    lines = [_line("a", 90.0, prefetch=True), _line("b", 80.0), _line("c", 70.0, prefetch=True)]
    pith_stage3(lines, budget_chars=5000)
    m = cc._PITH_METRICS
    assert m.l1_prefetch_distinct <= m.l1_kept_distinct
    assert m.l1_prefetch_distinct_promotable <= m.l1_kept_distinct_promotable


def test_narrow_excludes_monitor_and_victim_from_both_terms():
    lines = [_line("p", 90.0, stream="pattern", prefetch=True),
             _line("m", 80.0, stream="monitor"),
             _line("v", 70.0, stream="victim")]
    pith_stage3(lines, budget_chars=5000)
    m = cc._PITH_METRICS
    assert m.l1_kept_distinct == 3, "broad should count every stream"
    assert m.l1_kept_distinct_promotable == 1, "narrow should exclude monitor+victim"
    assert m.l1_prefetch_distinct_promotable == 1


def test_zero_denominator_leaves_counters_at_zero():
    """Empty L1 -> no counts. A ratio built on this must report null, never 0.0."""
    pith_stage3([], budget_chars=5000)
    m = cc._PITH_METRICS
    assert m.l1_kept_distinct == 0 and m.l1_prefetch_distinct == 0
    assert m.snapshot()["l1_kept_distinct"] == 0


def test_existing_counters_keep_their_meaning():
    """ranked_kept still counts LINES, not distinct nodes -- unchanged by D5."""
    lines = [_line("dup", 90.0), _line("dup", 80.0), _line("solo", 70.0)]
    pith_stage3(lines, budget_chars=5000)
    m = cc._PITH_METRICS
    assert m.ranked_kept == 3, "ranked_kept semantics changed"
    assert m.l1_kept_distinct == 2, "distinct count should differ from ranked_kept here"


def test_snapshot_exposes_all_four_terms():
    snap = cc._PITH_METRICS.snapshot()
    for k in ("l1_kept_distinct", "l1_prefetch_distinct",
              "l1_kept_distinct_promotable", "l1_prefetch_distinct_promotable"):
        assert k in snap, f"{k} missing from snapshot()"
