# ---- Changelog ----
# [2026-11-24] DudeMan CC (Claude Fable 5.2, laptop; SQ11, punch #1251 step 2) -- the self-tuning surfacing queue
#   (graph.config['surfacing_selftune']; ng_tract NodeStore.sf_configure / sf_queue / sf_stats).
# What: on REAL graph steps: (1) a node that grows warm enters the queue and a cooler one does not; a node that
#       fires this step is not a candidate; (2) engagement is graded: a surfaced node is watched 30 steps and scored
#       by its fires above its own running rate (+ whether a recall sweep reached it); the ratio moves by that
#       amount when the window closes; what the host showed can be watched too; (3) the queue holds six; (4) the key off is
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


def _steps(g, n):
    res = None
    for _ in range(n):
        res = g.step()
    return res


def test_engagement_is_graded_against_the_nodes_own_rate():
    g = _graph()
    mon = _mon(g)
    g.nodes["n01"].firing_rate_ema = 0.1                # a node that usually fires 3 times in 30 steps
    _warm(g, "n00", 0.90)
    _warm(g, "n01", 0.90)
    g.step()
    assert sorted(_ids(mon)) == ["n00", "n01"]
    _warm(g, "n00", 1.60)                               # n00 (usual rate 0) goes on to fire once
    _warm(g, "n01", 0.05)                               # n01 goes quiet
    res = g.step()
    assert "n00" in res.fired_node_ids
    st = mon.tuning_stats()
    assert (st["total_scored"], st["threshold_ratio"], st["watching"]) == (0, 0.72, 2)   # nothing moves yet
    assert _ids(mon) == []                              # both have left the visible queue
    _steps(g, 29)                                       # their 30-step windows close
    by = {s["node_id"]: s for s in mon.engagement_scores()}
    assert by["n00"]["fires"] == 1 and by["n00"]["excess"] == pytest.approx(1.0) and by["n00"]["engagement"] == pytest.approx(0.5)
    assert by["n01"]["fires"] == 0 and by["n01"]["excess"] == pytest.approx(-3.0) and by["n01"]["engagement"] == -1.0
    assert by["n00"]["source"] == "queue" and by["n00"]["cue"] is False
    st = mon.tuning_stats()
    assert (st["total_scored"], st["total_engaged"], st["total_unused"], st["watching"]) == (2, 1, 1, 0)
    assert st["threshold_ratio"] == pytest.approx(0.72 + 0.5 * ENGAGE + 1.0 * EXPIRE, abs=1e-12)
    assert st["mean_engagement"] == pytest.approx(-0.25)
    assert mon.engagement_scores(drain=True) and mon.engagement_scores() == []


def test_a_node_over_its_own_threshold_but_held_down_is_not_a_candidate():
    g = _graph()
    mon = _mon(g)
    node = g.nodes["n00"]
    node.voltage = 1.5 * node.threshold
    node.refractory_remaining = 3                       # over threshold and cannot fire
    _warm(g, "n01", 0.90)
    res = g.step()
    assert "n00" not in res.fired_node_ids and _ids(mon) == ["n01"]


def test_the_host_can_have_what_it_showed_watched_and_scored():
    g = _graph()
    mon = _mon(g)
    assert mon.note_shown(["n05", "n06", "no-such-node", ""]) == 2
    assert mon.note_shown(["n05"]) == 0                 # already being watched
    _warm(g, "n05", 1.60)
    _steps(g, 30)
    by = {s["node_id"]: s for s in mon.engagement_scores()}
    assert by["n05"]["source"] == "shown" and by["n05"]["engagement"] == pytest.approx(0.5)
    assert by["n06"]["engagement"] == 0.0
    assert mon.tuning_stats()["threshold_ratio"] == 0.72    # what the host showed does not move the queue's ratio


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
    g.nodes["n00"].firing_rate_ema = 0.1
    _warm(g, "n00", 0.90)
    g.step()
    _warm(g, "n00", 0.05)
    _steps(g, 30)                                       # one unused outcome: the ratio has moved
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
    assert mon.get_surfaced(100) == []                  # the read stops at once ...
    assert g.nodes.sf_stats()[0] is True                # ... and does not touch the switch
    assert mon.sync_tuning() is False
    assert g.nodes.sf_queue() == [] and g.nodes.sf_stats()[0] is False
    assert g.nodes.sf_stats()[1] == 0.72 and mon2 is not None


def test_a_graph_without_a_native_node_store_logs_once_and_stays_on_the_existing_path(caplog):
    g = _graph(kind="dict")
    with caplog.at_level(logging.ERROR):
        mon = _mon(g)
        _warm(g, "n00", 0.90)
        g.step()
        assert mon.get_surfaced(100) == [] and mon.get_surfaced(100) == []
    assert caplog.text.count("surfacing_selftune is on but") == 1


def test_reads_never_switch_the_queue_only_sync_does():
    g = _graph()
    mon = _mon(g, on=False)
    g.config[KEY] = True                                # the key turns on after construction
    assert mon.tuning_stats() is None and mon.persist_tuning() is None and mon.get_surfaced(100) == []
    assert g.nodes.sf_stats()[0] is False and "surfacing_threshold_ratio" not in g.config
    assert mon.sync_tuning() is True and g.nodes.sf_stats()[0] is True
    _warm(g, "n00", 0.90)
    g.step()
    assert _ids(mon) == ["n00"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "not a number", None])
def test_a_saved_ratio_that_is_not_a_finite_number_is_ignored(bad):
    g = _graph()
    g.config["surfacing_threshold_ratio"] = bad
    assert _mon(g).tuning_stats()["threshold_ratio"] == 0.72


def test_a_restore_while_on_takes_the_restored_ratio_at_the_next_sync(tmp_path):
    g = _graph()
    mon = _mon(g)
    g.nodes["n00"].firing_rate_ema = 0.1
    _warm(g, "n00", 0.90)
    g.step()
    _warm(g, "n00", 0.05)
    _steps(g, 30)                                       # one unused outcome: the ratio has moved
    learned = mon.persist_tuning()
    assert learned == pytest.approx(0.72 + EXPIRE, abs=1e-12)
    path = str(tmp_path / "c.msgpack")
    g.checkpoint(path)
    g.config["surfacing_threshold_ratio"] = 0.5         # a different value in memory, then restore over it
    g.nodes.sf_configure(True, 0.5)
    g.restore(path)
    assert g.nodes.sf_stats()[0] is False               # the restore put the queue back to off / defaults
    assert mon.persist_tuning() is None                 # nothing is written while it is off
    g.config[KEY] = True
    assert mon.sync_tuning() is True
    assert mon.tuning_stats()["threshold_ratio"] == pytest.approx(learned, abs=1e-12)


def test_a_node_the_cue_reached_scores_as_engaged():
    g = _graph()
    g.config["native_recall_enabled"] = True            # the live path: the sweep is one native call (recall_sweep)
    mon = _mon(g)
    g.nodes["n00"].firing_rate_ema = 0.1                # without the cue this node would score as unused
    _warm(g, "n00", 0.90)
    g.step()
    assert _ids(mon) == ["n00"]
    g.prime_and_propagate(node_ids=["n00"], currents=[5.0], steps=1, write_mode=False)   # the cue reaches it
    _warm(g, "n00", 0.0)
    _steps(g, 30)
    (score,) = mon.engagement_scores()
    assert score["cue"] is True and score["engagement"] > -1.0
    assert score["engagement"] == pytest.approx(max(-1.0, min(1.0, score["excess"] / 2.0 + 1.0)))


def test_a_failing_read_falls_through_to_the_existing_paths(caplog):
    g = _graph()
    mon = _mon(g)
    mon._selftune_surfaced = lambda n: (_ for _ in ()).throw(RuntimeError("boom"))
    with caplog.at_level(logging.ERROR):
        assert mon.get_surfaced(100) == [] and mon.get_surfaced(100) == []
    assert caplog.text.count("could not be read") == 1
