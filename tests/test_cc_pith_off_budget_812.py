# tests/test_cc_pith_off_budget_812.py
#
# ---- Changelog ----
# [2026-10-04] Claude (lane 812-813-onto-s4) — rebased onto trial s4: tests for the trial's on_surfaced reporter on the
#   #813 budgeted un-Pithed path (whole kept / whole dropped / reference form) and the Pith-ON reference swap, plus the
#   N-2 guard's RuntimeError -> on_degraded('monitor_race') contract. Both adapted behaviours fail without the adaptation.
# [2026-10-01] Claude Sonnet 5.5 (Z12 lane surfacing-whole-812, dispatch #12861) — #812 N-2 fold: tests
# What: N-2 (a monitor-formatter failure is reported through on_monitor_error -- the same
#   hemisphere stat route as a harvest failure, RuntimeError silent as at e4ebf982 -- and the pattern
#   twins come back; the Pith-ON success path never formats), N-3 (PINS the current dedupe-before-
#   budget behaviour: a node whose monitor copy the budget drops can vanish whole), N-4 (a second
#   reference-form shape through the real Pith-OFF path: an over-budget MONITOR item; and the
#   never-fit qualifier pinned), N-5 (a second budget-binding shape: many small items, tight budget).
# Why: le-043 N-2..N-5 / checker-034 N8..N11; Chief-003 fold (row #892).
# How: same in-process fakes; real SurfacingMonitor / cc_pattern_completion_recall / cc_assemble_recall;
#   the hemisphere stat is proven through the REAL cc_ng_host._recall (and the daemon's _recall when
#   ~/docs/scripts/cc-ng-daemon.py exists, loaded as the unification test loads it).
# [2026-10-01] Claude Sonnet 5.5 (Z12 lane surfacing-whole-812, dispatch #12684) — #812 turn 2, part 2 (g)(3) + (h)
# What: PROVES, on the #813-on-#812 branch, that the CC Pith-OFF DEFAULT path is budgeted by the
#   existing #813 mechanism (_cc_render_unpithed: strict-prefix admit against cc_l1_budget /
#   CC_PITH_L1_BUDGET, ONE INFO line, top item kept) with the REAL source chain: a real
#   SurfacingMonitor (whole at the source, #812) + the real cc_pattern_completion_recall +
#   the real cc_assemble_recall, the gate at its code default (OFF). Also pins (h): the interim
#   fork and the whole_content flag are gone, and the monitor block comes from the shared
#   format_context, fail-soft.
# Why:  Exec P468/P474 + Chief-003: the CC Pith-OFF default must have the budget BEFORE merge;
#   "We fix stuff correctly, not monkey patch or work around."
# How:  in-process fakes only (no daemon, graph load, network, embedding model); the
#   GSG re-score and novelty are stubbed (they need the embedder). Run with every CC_PITH_*
#   environment variable scrubbed so the module defaults are what is tested. Exec P379/#770:
#   the modules under test must be this root's copies (hard assertion + the session preamble
#   printed by tests/test_surfacing_whole.py, which the run lists first).
# -------------------

import inspect
import logging
import os
import re
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tests"))

import cc_ng_organism as pith  # noqa: E402
import surface_resolver  # noqa: E402
import surfacing  # noqa: E402
from ces_config import CESConfig  # noqa: E402
from pith_clip_813_scenarios import FakeGraph, FakeVectorDB  # noqa: E402

for _m in (pith, surface_resolver, surfacing):
    assert Path(_m.__file__).resolve().parent == _ROOT, f"{_m.__name__} is not this root's copy: {_m.__file__}"


def _no_cut(s):
    return "…" not in s and "..." not in s


def _wire(monkeypatch):
    # The GSG re-score embeds the query (embedder); novelty is pull-based -- both stubbed.
    monkeypatch.setattr(pith, "cc_gsg_rescore", lambda surfaced, *_a, **_k: surfaced)
    monkeypatch.setattr(pith, "cc_novelty", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(pith, "_CC_PITH_ENABLED", False)          # the gate's code default
    pith._PITH_VICTIM.clear()


def _text(tag, n):
    return f"{tag}-" + ("x" * (n - len(tag) - 1))


def _world(monitor_sizes, pattern_sizes):
    """A graph + a REAL SurfacingMonitor that has really fired the monitor nodes, plus the
    pattern-stream harvest. Returns (ng, monitor_texts, pattern_texts)."""
    g = FakeGraph()
    g.timestep = 1
    monitor_texts, pattern_texts, fired = [], [], []
    for i, n in enumerate(monitor_sizes):
        t = _text(f"MONITOR{i}", n)
        node = g.node(f"mon{i}", t, creation_mode="conversational")
        node.voltage, node.threshold, node.intrinsic_excitability = 2.0, 1.0, 1.0 + 0.1 * i
        monitor_texts.append(t)
        fired.append(f"mon{i}")
    harvest = []
    for i, n in enumerate(pattern_sizes):
        t = _text(f"PATTERN{i}", n)
        g.node(f"pat{i}", t, creation_mode="conversational")
        pattern_texts.append(t)
        harvest.append({"node_id": f"pat{i}", "strength": 100.0 - i})
    cfg = CESConfig()
    cfg.surfacing.min_confidence = 0.1
    monitor = surfacing.SurfacingMonitor(g, FakeVectorDB(), cfg)
    monitor.after_step(types.SimpleNamespace(fired_node_ids=fired))
    ng = types.SimpleNamespace(
        graph=g, vector_db=FakeVectorDB(), _surfacing_monitor=monitor,
        _harvest_associations=lambda q, novelty=0.5, **kw: [dict(h) for h in harvest])
    return ng, monitor_texts, pattern_texts


def _drop_records(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.INFO and "whole items" in r.getMessage()]


def test_the_gate_is_off_by_default_in_the_source():
    """The default path under test: CC_PITH_ENABLED defaults to OFF in the code (the laptop's own
    environment sets it to 1, which is why this run scrubs CC_PITH_*)."""
    src = (_ROOT / "cc_ng_organism.py").read_text()
    assert re.search(r'_CC_PITH_ENABLED\s*=\s*os\.environ\.get\("CC_PITH_ENABLED",\s*"0"\)', src)
    assert not [k for k in os.environ if k.startswith("CC_PITH_")], \
        "run this file with every CC_PITH_* variable scrubbed so module defaults are tested"


def test_pith_off_default_budget_binds_drops_whole_lowest_ranked_one_info_line_keeps_top(
        monkeypatch, caplog):
    """(g)(3) PROOF. 1 monitor item + 3 pattern items, each 1500 chars WHOLE (4 x 1500 > 4000).
    Strict rank prefix (pattern weight 1.0 x norm; monitor 0.6): p0 (1.0), mon (0.6), p1 (0.5),
    p2 (0.0) -> admits p0 + mon (3000), drops p1 + p2 WHOLE. ONE INFO line, count + total size.
    The top item is kept WHOLE. Nothing is cut."""
    _wire(monkeypatch)
    ng, (mon,), (p0, p1, p2) = _world([1500], [1500, 1500, 1500])
    assert ng._surfacing_monitor.get_surfaced()[0]["content"] == mon     # whole at the SOURCE
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out = pith.cc_assemble_recall(ng, "what next", 5, {}, None)
    assert p0 in out and mon in out                                      # top kept whole
    assert p1[:20] not in out and p2[:20] not in out                     # dropped WHOLE, not a piece
    assert _no_cut(out)
    records = _drop_records(caplog)
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    assert "L1 budget 4000 chars" in records[0]
    assert "dropping 2 whole items (3000 chars)" in records[0]
    assert "kept 2 (3000 chars)" in records[0]


def test_pith_off_nothing_dropped_means_no_info_line_and_everything_whole(monkeypatch, caplog):
    _wire(monkeypatch)
    ng, (mon,), (p0, p1) = _world([600], [700, 800])
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out = pith.cc_assemble_recall(ng, "what next", 5, {}, None)
    assert mon in out and p0 in out and p1 in out and _no_cut(out)
    assert _drop_records(caplog) == []
    assert out.startswith("[NeuroGraph Surfaced Knowledge]")             # the shared marker, via CES


def test_pith_off_an_item_alone_over_the_whole_budget_is_still_kept_as_the_reference_form(
        monkeypatch, caplog):
    """The 'ALWAYS keep the top item' corner on the default path: a 30000-char top item cannot
    fit 4000 whole, so it surfaces through its TREES plus the one-line whole-node reference
    (P417, the existing machinery) -- never dropped, never cut mid-text."""
    _wire(monkeypatch)
    ng, _m, _p = _world([], [])
    g = ng.graph
    giant = "GIANT-START " + ("filler words " * 2300) + " GIANT-END"
    g.node("cc:conv::big", giant, creation_mode="conversational")
    trees = []
    for i in range(3):
        tt = f"concept {i}: the checkpoint cadence decision number {i}"
        trees.append(tt)
        g.node(f"tree{i}", tt, _tree_concept=True, _concept=tt)
        g.synapse(f"f{i}", "cc:conv::big", f"tree{i}", 0.2)
        g.synapse(f"b{i}", f"tree{i}", "cc:conv::big", 0.15)
    ng._harvest_associations = lambda q, novelty=0.5, **kw: [{"node_id": "cc:conv::big", "strength": 50.0}]
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out = pith.cc_assemble_recall(ng, "what next", 5, {}, None)
    assert "GIANT-START" not in out and "GIANT-END" not in out           # not emitted over budget
    assert "cc:conv::big" in out and "long node" in out                  # the reference line
    assert all(tt in out for tt in trees)                                # its trees, WHOLE
    assert any("above the reference limit" in r.getMessage() for r in caplog.records)


def test_pith_off_a_failing_monitor_formatter_never_crashes_the_hook(monkeypatch, caplog):
    """The renderer calls the shared monitor.format_context fail-soft: a raising formatter loses
    only the monitor block (one WARNING naming the exception TYPE, never item text)."""
    _wire(monkeypatch)
    ng, (mon,), (p0,) = _world([600], [700])

    def boom(items):
        raise RuntimeError("SECRET-ITEM-TEXT-DO-NOT-LOG " + items[0]["content"][:12])
    ng._surfacing_monitor.format_context = boom
    with caplog.at_level(logging.WARNING, logger=pith.logger.name):
        out = pith.cc_assemble_recall(ng, "what next", 5, {}, None)
    assert p0 in out and mon not in out                                  # the pattern block survives
    warned = [r.getMessage() for r in caplog.records if "monitor block formatting failed" in r.getMessage()]
    assert warned and "RuntimeError" in warned[0] and "SECRET" not in warned[0] and "MONITOR0" not in warned[0]


def test_h_the_interim_fork_and_the_whole_content_flag_are_gone():
    """(h) FAILS on the replay tip (the fork and the flag exist there)."""
    assert not hasattr(pith, "_cc_monitor_items_whole")
    assert not hasattr(pith, "_format_cc_monitor_block")
    assert "whole_content" not in inspect.signature(pith.cc_pattern_completion_recall).parameters
    src = (_ROOT / "cc_ng_organism.py").read_text()
    assert "max_chars=sys.maxsize" not in src and "max_chars=(sys.maxsize" not in src


def test_h_the_recall_call_is_one_plain_resolver_call(monkeypatch):
    """(h) the merged form: resolve_surface_content(node, r, allow_ingested=True), no bound."""
    _wire(monkeypatch)
    ng, _m, _p = _world([], [2000])
    out = pith.cc_pattern_completion_recall(ng, "q", 5)
    assert out and len(out[0]["content"]) == 2000 and _no_cut(out[0]["content"])


# ======================================================================== #812 N-2 fold ==
class _Recorder:
    """on_monitor_error stand-in: records every exception it is handed."""
    def __init__(self):
        self.calls = []

    def __call__(self, exc):
        self.calls.append(exc)


def _build(spec):
    """spec: [(node_id, size, monitor_excitability_or_None, pattern_strength_or_None)].
    A node with BOTH is a node present in both streams (a twin). Returns (ng, texts)."""
    g = FakeGraph()
    g.timestep = 1
    texts, fired, harvest = {}, [], []
    for nid, size, excit, strength in spec:
        t = _text(nid, size)
        node = g.node(nid, t, creation_mode="conversational")
        node.voltage, node.threshold, node.intrinsic_excitability = 2.0, 1.0, (excit or 1.0)
        texts[nid] = t
        if excit is not None:
            fired.append(nid)
        if strength is not None:
            harvest.append({"node_id": nid, "strength": strength})
    cfg = CESConfig()
    cfg.surfacing.min_confidence = 0.1
    monitor = surfacing.SurfacingMonitor(g, FakeVectorDB(), cfg)
    monitor.after_step(types.SimpleNamespace(fired_node_ids=fired))
    ng = types.SimpleNamespace(
        graph=g, vector_db=FakeVectorDB(), _surfacing_monitor=monitor,
        _harvest_associations=lambda q, novelty=0.5, **kw: [dict(h) for h in harvest])
    return ng, texts


_TWIN_SPEC = [("X", 600, 1.0, 100.0),      # in BOTH streams: the monitor item and its pattern twin
              ("P0", 700, None, 99.0)]     # pattern-only


def _fail_formatter(ng, exc_type=ValueError, only_after=0):
    """Make the REAL monitor's format_context raise (after `only_after` good calls)."""
    real = ng._surfacing_monitor.format_context
    state = {"n": 0, "calls": 0}

    def fmt(items):
        state["calls"] += 1
        if state["calls"] > only_after:
            raise exc_type("SECRET-ITEM-TEXT-DO-NOT-LOG " + items[0]["content"][:12])
        return real(items)
    ng._surfacing_monitor.format_context = fmt
    return state


def _pith_failed(monkeypatch):
    monkeypatch.setattr(pith, "_CC_PITH_ENABLED", True)
    monkeypatch.setattr(pith, "pith_stage1",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("pith boom")))


@pytest.mark.parametrize("entry", ["gate_off", "pith_failed"])
def test_n2_a_formatter_failure_is_reported_and_the_pattern_twin_survives(monkeypatch, caplog, entry):
    """N-2 + twin survival. The monitor formatter raises (ValueError): it is reported ONCE through
    on_monitor_error with the exception, the monitor stream is dropped, and the pattern twin of X
    (removed by the display dedup because the monitor copy existed) COMES BACK. Both entries to the
    un-Pithed renderer: the gate OFF, and a Pith failure.
    FAILS on the pre-fold tip 55d56b9a: no on_monitor_error call, and X vanishes from both streams."""
    _wire(monkeypatch)
    if entry == "pith_failed":
        _pith_failed(monkeypatch)
    ng, texts = _build(_TWIN_SPEC)
    _fail_formatter(ng, ValueError)
    rec = _Recorder()
    with caplog.at_level(logging.WARNING, logger=pith.logger.name):
        out = pith.cc_assemble_recall(ng, "q", 5, {}, None, on_monitor_error=rec)
    assert texts["X"] in out and texts["P0"] in out                      # the twin is back
    assert "[NeuroGraph Surfaced Knowledge]" not in out                  # the monitor block is not
    assert len(rec.calls) == 1 and isinstance(rec.calls[0], ValueError)
    warned = [r.getMessage() for r in caplog.records if "monitor block formatting failed" in r.getMessage()]
    assert warned and "ValueError" in warned[0] and "SECRET" not in warned[0]


def test_n2_runtimeerror_stays_silent_as_at_e4ebf982_but_the_twin_still_returns(monkeypatch):
    """The old guard's RuntimeError branch ('dict mutation race') reset the stream WITHOUT reporting.
    Mirrored exactly: no on_monitor_error call, twin restored.
    FAILS on the pre-fold tip: the twin X is missing."""
    _wire(monkeypatch)
    ng, texts = _build(_TWIN_SPEC)
    _fail_formatter(ng, RuntimeError)
    rec = _Recorder()
    out = pith.cc_assemble_recall(ng, "q", 5, {}, None, on_monitor_error=rec)
    assert texts["X"] in out and texts["P0"] in out
    assert rec.calls == []


def test_n2_the_formatter_failure_takes_the_same_route_as_a_harvest_failure(monkeypatch):
    """Both failure points call the SAME on_monitor_error callback once with the exception.
    FAILS on the pre-fold tip (the formatter case calls nothing)."""
    _wire(monkeypatch)
    routes = {}
    for point in ("harvest", "format"):
        ng, _t = _build(_TWIN_SPEC)
        if point == "harvest":
            def boom():
                raise ValueError("harvest exploded")
            ng._surfacing_monitor.get_surfaced = boom
        else:
            _fail_formatter(ng, ValueError)
        rec = _Recorder()
        pith.cc_assemble_recall(ng, "q", 5, {}, None, on_monitor_error=rec)
        routes[point] = rec.calls
    assert len(routes["harvest"]) == 1 and len(routes["format"]) == 1
    assert type(routes["harvest"][0]) is type(routes["format"][0]) is ValueError


def test_n2_pith_on_success_path_never_formats_so_it_is_unchanged(monkeypatch):
    """Why the Pith-ON path is unchanged: its CacheLines are built from the monitor ITEMS and it
    returns before the guard, so the monitor formatter is never called and nothing is reported.
    Passes before AND after the fold (a pin of 'unchanged')."""
    _wire(monkeypatch)
    monkeypatch.setattr(pith, "_CC_PITH_ENABLED", True)
    ng, texts = _build(_TWIN_SPEC)
    state = _fail_formatter(ng, ValueError)                              # would raise if it were called
    rec = _Recorder()
    out = pith.cc_assemble_recall(ng, "q", 5, {}, None, on_monitor_error=rec)
    assert state["calls"] == 0 and rec.calls == []
    assert texts["X"] in out and texts["P0"] in out


def test_n2_the_renderers_own_guard_is_the_last_resort(monkeypatch, caplog):
    """The renderer's guard (a second call on the kept subset) still never crashes the hook: the
    block is lost, one WARNING naming the type, nothing reported (the first call already passed).
    FAILS on the pre-fold tip: its single call is the first, so the 'second call' never fails there."""
    _wire(monkeypatch)
    ng, texts = _build(_TWIN_SPEC)
    _fail_formatter(ng, ValueError, only_after=1)
    rec = _Recorder()
    with caplog.at_level(logging.WARNING, logger=pith.logger.name):
        out = pith.cc_assemble_recall(ng, "q", 5, {}, None, on_monitor_error=rec)
    assert texts["P0"] in out and "[NeuroGraph Surfaced Knowledge]" not in out
    assert rec.calls == []
    assert any("monitor block formatting failed" in r.getMessage() for r in caplog.records)


def _host_world(monkeypatch):
    import cc_ng_host
    _wire(monkeypatch)
    ng, texts = _build(_TWIN_SPEC)
    _fail_formatter(ng, ValueError)
    monkeypatch.setattr(cc_ng_host._STATE, "cc_ng", ng)
    monkeypatch.setattr(cc_ng_host._STATE, "conv_state", {})
    monkeypatch.setattr(cc_ng_host._STATE, "commons", None)
    return cc_ng_host, texts


def test_n2_the_host_hemisphere_error_stat_is_bumped_by_a_formatter_failure(monkeypatch):
    """Through the REAL cc_ng_host._recall + the REAL cc_assemble_recall: a formatter failure bumps
    _STATE.stats['errors'] exactly as the harvest failure pinned by
    test_wrappers_bump_error_stat_on_monitor_harvest_failure does, and the twin still renders.
    FAILS on the pre-fold tip: no bump, twin missing."""
    host, texts = _host_world(monkeypatch)
    before = host._STATE.stats["errors"]
    out = host._recall("q", k=3)
    assert host._STATE.stats["errors"] == before + 1
    assert texts["X"] in out and texts["P0"] in out


def _load_daemon():
    import importlib.util
    path = os.path.expanduser("~/docs/scripts/cc-ng-daemon.py")
    if not os.path.isfile(path):
        pytest.skip("~/docs/scripts/cc-ng-daemon.py not present (daemon hemisphere not under test)")
    spec = importlib.util.spec_from_file_location("cc_ng_daemon_under_test_812", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cc_ng_daemon_under_test_812"] = module
    spec.loader.exec_module(module)
    return module


def test_n2_the_daemon_hemisphere_error_stat_is_bumped_by_a_formatter_failure(monkeypatch):
    """The laptop daemon's _recall wires the same on_monitor_error=_bump_error. Skipped when the
    daemon script is not under ~/docs/scripts (the run points $HOME/docs at a daemon worktree).
    FAILS on the pre-fold tip."""
    daemon = _load_daemon()
    _wire(monkeypatch)
    ng, texts = _build(_TWIN_SPEC)
    _fail_formatter(ng, ValueError)
    monkeypatch.setattr(daemon.STATE, "ng", ng)
    monkeypatch.setattr(daemon.STATE, "conv_state", {})
    monkeypatch.setattr(daemon.STATE, "commons", None)
    before = daemon.STATE.stats["errors"]
    out = daemon._recall("q", 3)
    assert daemon.STATE.stats["errors"] == before + 1
    assert texts["X"] in out and texts["P0"] in out


def test_n3_pins_dedupe_before_budget_a_node_whose_monitor_copy_is_budget_dropped_can_vanish(
        monkeypatch, caplog):
    """N-3 (pre-existing #813; NOT changed by the N-2 fold, so this PINS the CURRENT behaviour).
    X is in both streams. The display dedup removes X's pattern twin BEFORE the budget runs; the
    budget then ranks X's monitor copy last (monitor weight 0.6 x min-max 0.0) and drops it: X
    appears nowhere. (Without the dedup the twin, ranked first, would have survived.) If a later
    change fixes this, this test must flip deliberately. Passes before AND after the fold."""
    _wire(monkeypatch)
    ng, texts = _build([("Y", 1500, 1.5, None),      # monitor-only, higher monitor score
                        ("X", 1500, 1.0, 100.0),     # monitor copy ranks last; its twin would rank first
                        ("P0", 1500, None, 99.0)])   # pattern-only
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out = pith.cc_assemble_recall(ng, "q", 5, {}, None)
    assert texts["P0"] in out and texts["Y"] in out
    assert texts["X"][:20] not in out                                    # X vanished whole: the pin
    records = _drop_records(caplog)
    assert len(records) == 1 and "dropping 1 whole items (1500 chars)" in records[0]


def test_budget_binds_second_shape_many_small_items_under_a_tight_budget(monkeypatch, caplog):
    """A second binding shape (count-driven, not size-driven): budget 1000, eight 300-char pattern
    items. Strict rank prefix keeps the top THREE (900), drops FIVE whole (1500), ONE INFO line,
    output order = rank order. Passes before AND after (coverage)."""
    _wire(monkeypatch)
    monkeypatch.setattr(pith, "_CC_PITH_L1_BUDGET", 1000)
    spec = [(f"Q{i}", 300, None, 100.0 - i) for i in range(8)]
    ng, texts = _build(spec)
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out = pith.cc_assemble_recall(ng, "q", 8, {}, None)
    kept = [i for i in range(8) if texts[f"Q{i}"] in out]
    assert kept == [0, 1, 2]
    assert [out.index(texts[f"Q{i}"]) for i in kept] == sorted(out.index(texts[f"Q{i}"]) for i in kept)
    for i in range(3, 8):
        assert texts[f"Q{i}"][:12] not in out                            # absent whole, never a piece
    assert _no_cut(out)
    records = _drop_records(caplog)
    assert len(records) == 1
    assert "L1 budget 1000 chars" in records[0]
    assert "dropping 5 whole items (1500 chars)" in records[0] and "kept 3 (900 chars)" in records[0]


def _giant_with_trees(graph, nid, n_trees=3):
    giant = "GIANT-START " + ("filler words " * 2300) + " GIANT-END"
    graph.node(nid, giant, creation_mode="conversational")
    trees = []
    for i in range(n_trees):
        tt = f"concept {i}: the checkpoint cadence decision number {i}"
        trees.append(tt)
        graph.node(f"{nid}-t{i}", tt, _tree_concept=True, _concept=tt)
        graph.synapse(f"{nid}-f{i}", nid, f"{nid}-t{i}", 0.2)
        graph.synapse(f"{nid}-b{i}", f"{nid}-t{i}", nid, 0.15)
    return giant, trees


def test_pith_off_an_over_budget_MONITOR_item_surfaces_as_trees_plus_reference(monkeypatch, caplog):
    """Reference form, second shape: the over-budget item arrives through the MONITOR stream (whole at
    the source), not the pattern stream. Real Pith-OFF cc_assemble_recall: it surfaces as its trees,
    whole, plus the one-line whole-node reference; the giant text is never emitted or cut; the small
    pattern item is whole. Passes before AND after (coverage of the existing P417 mechanism)."""
    _wire(monkeypatch)
    ng, texts = _build([("S0", 500, None, 99.0)])
    giant, trees = _giant_with_trees(ng.graph, "cc:conv::bigmon")
    node = ng.graph.nodes["cc:conv::bigmon"]
    node.voltage, node.threshold, node.intrinsic_excitability = 2.0, 1.0, 1.0
    ng._surfacing_monitor.after_step(types.SimpleNamespace(fired_node_ids=["cc:conv::bigmon"]))
    assert ng._surfacing_monitor.get_surfaced()[0]["content"] == giant   # whole at the SOURCE
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out = pith.cc_assemble_recall(ng, "q", 5, {}, None)
    assert "GIANT-START" not in out and "GIANT-END" not in out
    assert "cc:conv::bigmon" in out and "long node" in out
    assert all(tt in out for tt in trees)
    assert texts["S0"] in out
    assert any("above the reference limit" in r.getMessage() for r in caplog.records)


def test_pith_off_an_over_budget_item_with_no_reference_form_is_dropped_loudly_never_cut(
        monkeypatch, caplog):
    """N-4 qualifier, PINNED: 'always keep the top item' holds only while the P417 reference form
    can be built. An over-budget item whose node is unknown to the graph has no reference form
    (_pith_reference_text -> None): it is a never-fit, dropped whole and LOUDLY (the one INFO line
    names it), never cut. (cc_l1_budget clamps to 500-40000, so a tiny budget cannot be used to
    force this.) Passes before AND after (a pin of #813's own record)."""
    _wire(monkeypatch)
    ng, texts = _build([("S0", 500, None, 99.0)])
    ng._surfacing_monitor.get_surfaced = lambda max_items=None: [
        {"node_id": "ghost", "content": "G" * 6000, "score": 1.5}]
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out = pith.cc_assemble_recall(ng, "q", 5, {}, None)
    assert "GGGGGGGG" not in out                                         # not emitted, not cut
    assert texts["S0"] in out                                            # the rest still renders whole
    records = _drop_records(caplog)
    assert len(records) == 1 and "never-fit" in records[0] and "ghost" in records[0]


# ---- [2026-10-04] lane 812-813-onto-s4: the trial's on_surfaced(rendered, dropped) reporter -----------
# On the trial base the un-Pithed path had no budget (dropped was always []); after #813 the ONE budget
# rule drops WHOLE items there and P417 swaps an over-budget item for its trees + reference form.  The
# reporter must name exactly the WHOLE text that was rendered, and only what was really dropped.

class _Surfaced:
    def __init__(self):
        self.calls = []

    def __call__(self, rendered, dropped):
        self.calls.append((rendered, dropped))


def test_on_surfaced_pith_off_reports_the_whole_kept_items_and_the_whole_budget_drops(monkeypatch):
    _wire(monkeypatch)
    ng, (mon,), (p0, p1, p2) = _world([1500], [1500, 1500, 1500])
    rep = _Surfaced()
    out = pith.cc_assemble_recall(ng, "what next", 5, {}, None, on_surfaced=rep)
    assert len(rep.calls) == 1
    rendered, dropped = rep.calls[0]
    assert sorted((r["stream"], r["content"]) for r in rendered) == \
        sorted([("monitor", mon), ("pattern", p0)])
    assert all(r["content"] in out for r in rendered)                    # WHOLE, and really rendered
    assert sorted(r["content"] for r in dropped) == sorted([p1, p2])     # dropped WHOLE
    assert all(r["content"][:20] not in out for r in dropped)


def test_on_surfaced_pith_off_reports_the_reference_form_not_the_giant_text(monkeypatch):
    _wire(monkeypatch)
    ng, texts = _build([("S0", 500, None, 99.0)])
    giant, trees = _giant_with_trees(ng.graph, "cc:conv::bigpat")
    harvest = ng._harvest_associations
    ng._harvest_associations = lambda q, novelty=0.5, **kw: (
        harvest(q, novelty=novelty, **kw) + [{"node_id": "cc:conv::bigpat", "strength": 1.0}])
    rep = _Surfaced()
    out = pith.cc_assemble_recall(ng, "q", 5, {}, None, on_surfaced=rep)
    rendered, dropped = rep.calls[0]
    big = [r for r in rendered if r["node_id"] == "cc:conv::bigpat"]
    assert len(big) == 1 and big[0]["content"] in out                    # the form actually rendered
    assert "GIANT-START" not in big[0]["content"] and all(tt in big[0]["content"] for tt in trees)
    assert all(r["content"] in out for r in rendered)
    assert dropped == []


def test_on_surfaced_pith_off_a_formatter_failure_reports_no_monitor_item_and_the_restored_twin(
        monkeypatch):
    _wire(monkeypatch)
    ng, texts = _build(_TWIN_SPEC)
    _fail_formatter(ng, ValueError)
    rep = _Surfaced()
    out = pith.cc_assemble_recall(ng, "q", 5, {}, None, on_surfaced=rep)
    rendered, _dropped = rep.calls[0]
    assert rendered and all(r["stream"] == "pattern" for r in rendered)
    assert all(r["content"] in out for r in rendered)
    assert texts["X"] in [r["content"] for r in rendered]                # the twin came back


def test_on_surfaced_pith_on_reference_swap_is_rendered_not_dropped(monkeypatch):
    _wire(monkeypatch)
    monkeypatch.setattr(pith, "_CC_PITH_ENABLED", True)
    ng, (mon,), _p = _world([600], [])
    giant, trees = _giant_with_trees(ng.graph, "cc:conv::bigon")
    ng._harvest_associations = lambda q, novelty=0.5, **kw: [{"node_id": "cc:conv::bigon", "strength": 50.0}]
    rep = _Surfaced()
    out = pith.cc_assemble_recall(ng, "what next", 5, {}, None, on_surfaced=rep)
    rendered, dropped = rep.calls[0]
    assert "GIANT-START" not in out
    assert all(r["content"] in out for r in rendered)
    assert [r for r in rendered if r["node_id"] == "cc:conv::bigon"]
    assert not [r for r in dropped if r["node_id"] == "cc:conv::bigon"]  # never both rendered and dropped


def test_n2_runtimeerror_in_the_formatter_is_the_trials_monitor_race_on_degraded(monkeypatch):
    """[lane 812-813-onto-s4] On the trial base a RuntimeError from format_context (then inside the
    harvest try) was reported as on_degraded('monitor_race'); the N-2 guard keeps that contract and
    still does not call on_monitor_error for it."""
    _wire(monkeypatch)
    ng, texts = _build(_TWIN_SPEC)
    _fail_formatter(ng, RuntimeError)
    rec, degraded = _Recorder(), []
    out = pith.cc_assemble_recall(ng, "q", 5, {}, None, on_monitor_error=rec,
                                  on_degraded=lambda code, exc: degraded.append((code, type(exc))))
    assert rec.calls == [] and degraded == [("monitor_race", RuntimeError)]
    assert texts["X"] in out
