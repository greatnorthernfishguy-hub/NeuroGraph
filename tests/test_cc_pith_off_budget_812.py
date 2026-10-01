# tests/test_cc_pith_off_budget_812.py
#
# ---- Changelog ----
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
