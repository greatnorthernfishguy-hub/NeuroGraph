# ---- Changelog ----
# [2026-11-24] DudeMan CC (Claude Fable 5.2, laptop; SQ11, punch #1251 step 2) -- the self-tuning surfacing queue
#   (graph.config['surfacing_selftune']; ng_tract NodeStore.sf_configure / sf_queue / sf_stats).
# What: on REAL graph steps: (1) a node that grows warm enters the queue and a cooler one does not; a node that
#       fires this step is not a candidate; (2) a queued node that goes on to fire lowers the threshold ratio by
#       0.012 x 0.04; one that cools unused raises it by 0.008 x 0.04; (3) the queue holds six; (4) the key off is
#       the existing path and the queue does nothing; (5) a node with nothing showable is passed over at the read;
#       (6) persist_tuning writes the learned ratio into graph.config and a new monitor hands it back to the queue;
#       (7) switching the key off empties the queue and keeps the ratio; (8) a graph without a native node store
#       logs once and stays on the existing path.
# Why:  NeuroGraph tunes its own surfacing threshold (Josh 2026-11-23); port of the original CES Module 3 monitor
#       (docs/archive/surfacing.pdf). The queue's rules themselves are unit-tested in ng-tract-rs surfacing_tune.rs.
# How:  a quiet native graph of unconnected nodes, so every voltage is the test's own.
# -------------------
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import neuro_foundation as NF  # noqa: E402
from ces_config import CESConfig  # noqa: E402
from surfacing import SurfacingMonitor  # noqa: E402
from tests import test_stability_s as S            # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(getattr(NF.ng_tract, "NodeStore", None), "sf_configure"),
                                reason="installed ng_tract has no self-tuning surfacing queue")

KEY = "surfacing_selftune"
ENGAGE = -0.012 * 0.04
EXPIRE = 0.008 * 0.04


class _Vdb:
    def __init__(self, bare=()):
        self.bare = set(bare)

    def get(self, nid):
        return None if nid in self.bare else {"content": "what the node " + nid + " holds, in plain words", "metadata": {}}


def _graph(n=12, kind="native"):
    g = S.graph_of(NF, kind)
    for i in range(n):
        g.create_node("n%02d" % i)
    return g


def _warm(g, nid, frac):
    node = g.nodes[nid]
    node.voltage = frac * node.threshold


def _mon(g, on=True, bare=()):
    if on:
        g.config[KEY] = True
    return SurfacingMonitor(g, _Vdb(bare), CESConfig())


def _ids(mon):
    return [r["node_id"] for r in mon.get_surfaced(100)]


def test_a_warm_node_enters_the_queue_a_cool_one_and_a_firing_one_do_not():
    g = _graph()
    mon = _mon(g)
    _warm(g, "n00", 0.90)       # warm, under its threshold
    _warm(g, "n01", 0.50)       # cool
    _warm(g, "n02", 1.60)       # fires this step
    res = g.step()
    assert "n02" in res.fired_node_ids and "n00" not in res.fired_node_ids
    rows = mon.get_surfaced(100)
    assert [r["node_id"] for r in rows] == ["n00"]
    assert 0.72 <= rows[0]["score"] < 1.0 and "n00" in rows[0]["content"]
    st = mon.tuning_stats()
    assert (st["on"], st["threshold_ratio"], st["total_surfaced"], st["queue_size"]) == (True, 0.72, 1, 1)


def test_a_queued_node_that_fires_lowers_the_ratio_and_one_that_cools_raises_it():
    g = _graph()
    mon = _mon(g)
    _warm(g, "n00", 0.90)
    _warm(g, "n01", 0.90)
    g.step()
    assert sorted(_ids(mon)) == ["n00", "n01"]
    _warm(g, "n00", 1.60)       # goes on to fire: engaged
    _warm(g, "n01", 0.05)       # cools off unused
    res = g.step()
    assert "n00" in res.fired_node_ids
    st = mon.tuning_stats()
    assert (st["total_engaged"], st["total_expired"], st["queue_size"]) == (1, 2, 0)
    assert st["threshold_ratio"] == pytest.approx(0.72 + ENGAGE + EXPIRE, abs=1e-12)
    assert _ids(mon) == []


def test_the_queue_holds_six_warmest_first():
    g = _graph()
    mon = _mon(g)
    for i in range(9):
        _warm(g, "n%02d" % i, 0.80 + 0.01 * i)
    g.step()
    assert _ids(mon) == ["n08", "n07", "n06", "n05", "n04", "n03"]
    assert mon.get_surfaced(2)[1]["node_id"] == "n07" and len(mon.get_surfaced(2)) == 2


def test_key_off_is_the_existing_path_and_the_queue_does_nothing():
    g = _graph()
    mon = _mon(g, on=False)
    _warm(g, "n00", 0.90)
    g.step()
    assert mon.get_surfaced(100) == [] and mon.tuning_stats() is None
    assert g.nodes.sf_stats()[0] is False and g.nodes.sf_queue() == []
    assert mon.persist_tuning() is None and "surfacing_threshold_ratio" not in g.config


def test_a_node_with_nothing_showable_is_passed_over_at_the_read():
    g = _graph()
    mon = _mon(g, bare=("n00",))
    _warm(g, "n00", 0.95)
    _warm(g, "n01", 0.90)
    g.step()
    assert _ids(mon) == ["n01"] and mon.tuning_stats()["queue_size"] == 2


def test_the_learned_ratio_is_saved_to_config_and_handed_back_to_a_new_queue():
    g = _graph()
    mon = _mon(g)
    _warm(g, "n00", 0.90)
    g.step()
    _warm(g, "n00", 0.05)
    g.step()
    ratio = mon.persist_tuning()
    assert ratio == pytest.approx(0.72 + EXPIRE, abs=1e-12) and g.config["surfacing_threshold_ratio"] == ratio

    g2 = _graph()
    g2.config["surfacing_threshold_ratio"] = 0.6
    mon2 = _mon(g2)
    assert mon2.tuning_stats()["threshold_ratio"] == 0.6
    g3 = _graph()
    g3.config["surfacing_threshold_ratio"] = 7.0        # out of bounds: clamped by the queue
    assert _mon(g3).tuning_stats()["threshold_ratio"] == 0.92


def test_switching_the_key_off_empties_the_queue_and_keeps_the_ratio():
    g = _graph()
    mon = _mon(g)
    g.config["surfacing_threshold_ratio"] = 0.5
    mon2 = SurfacingMonitor(g, _Vdb(), CESConfig())     # a second monitor on the same graph: the queue is the store's
    _warm(g, "n00", 0.90)
    g.step()
    assert _ids(mon) == ["n00"]
    g.config[KEY] = False
    assert mon.get_surfaced(100) == [] and g.nodes.sf_queue() == [] and g.nodes.sf_stats()[0] is False
    assert g.nodes.sf_stats()[1] == 0.72 and mon2 is not None


def test_a_graph_without_a_native_node_store_logs_once_and_stays_on_the_existing_path(caplog):
    g = _graph(kind="dict")
    with caplog.at_level(logging.ERROR):
        mon = _mon(g)
        _warm(g, "n00", 0.90)
        g.step()
        assert mon.get_surfaced(100) == [] and mon.get_surfaced(100) == []
    assert caplog.text.count("surfacing_selftune is on but") == 1
