# ---- Changelog ----
# [2026-11-23] DudeMan CC (Claude Fable 5.2, laptop; SQ11, punch #1250) -- the surfacing queue read from the spike stamps
#   (graph.config['surfacing_native_scan']; ng_tract.surfacing_scan).
# What: against the EXISTING pushed queue (SurfacingMonitor.after_step called after every step, as on_message did):
#       (1) every node that fired in the last step has exactly the pushed score; (2) over several steps the native
#       read holds the same nodes with near-equal scores (the declared difference: a node's state now, not at the
#       step it fired); (3) strongest first, max_items kept, a node with nothing showable is passed over; (4) the
#       key off is the old path; (5) the read writes nothing; (6) a graph that is not native logs once and stays on
#       the queue; (7) bad parameters raise.
# Why:  the laptop's queue has been starved since 2026-09-07; this restores it without a per-step Python walk.
# How:  tests.test_hyperedge_store_h2 built() graphs, as tests/test_native_degree_targets.py drives them.
# -------------------
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import neuro_foundation as NF  # noqa: E402
from ces_config import CESConfig  # noqa: E402
from surfacing import SurfacingMonitor  # noqa: E402
from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s18 as S18        # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(NF.ng_tract, "surfacing_scan"),
                                reason="installed ng_tract has no surfacing_scan")

KEY = "surfacing_native_scan"
ALL = 10 ** 6


class _Vdb:
    """Every node has something to show, except the ids in `bare`."""
    def __init__(self, bare=()):
        self.bare = set(bare)

    def get(self, nid):
        return None if nid in self.bare else {"content": "what the node " + nid + " holds, in plain words", "metadata": {}}


def _cfg():
    c = CESConfig()
    c.surfacing.queue_capacity = ALL
    c.surfacing.max_surfaced = ALL
    return c


def _graph(seed, kind="native"):
    g = S.restore_as(NF, kind, H2.built(seed))
    g.config.update(S18.ARM)
    return g


def _run(g, seed, steps, mon):
    side = H2.Side(g, seed)
    last = None
    for i in range(steps):
        def op(gr, i=i):
            rng = random.Random(seed * 100 + i)
            ids = sorted(gr.nodes.keys())
            for nid in rng.sample(ids, min(12, len(ids))):
                gr.nodes[nid].voltage += 1.5
            return gr.step()
        last = side(op)
        mon.after_step(last)            # the push the laptop daemon lost on 2026-09-07
    return last


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nodes_that_fired_in_the_last_step_have_exactly_the_pushed_score(seed):
    g = _graph(seed)
    mon = SurfacingMonitor(g, _Vdb(), _cfg())
    last = _run(g, seed, 9, mon)
    pushed = {r["node_id"]: r["score"] for r in mon.get_surfaced(ALL)}
    fired = [n for n in last.fired_node_ids if n in pushed]
    assert len(fired) >= 3                                          # not vacuous
    g.config[KEY] = True
    native = {r["node_id"]: r["score"] for r in mon.get_surfaced(ALL)}
    for nid in fired:
        assert native[nid] == pushed[nid], nid


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_over_several_steps_the_same_nodes_with_near_equal_scores(seed):
    g = _graph(seed)
    mon = SurfacingMonitor(g, _Vdb(), _cfg())
    _run(g, seed, 14, mon)
    pushed = {r["node_id"]: r["score"] for r in mon.get_surfaced(ALL)}
    g.config[KEY] = True
    rows = mon.get_surfaced(ALL)
    native = {r["node_id"]: r["score"] for r in rows}
    assert len(pushed) >= 10
    both = set(pushed) & set(native)
    assert len(both) / len(set(pushed) | set(native)) >= 0.9          # the declared difference moves few entries
    assert max(abs(native[n] - pushed[n]) for n in both) <= 0.05
    assert [r["score"] for r in rows] == sorted((r["score"] for r in rows), reverse=True)
    assert all(set(r) >= {"node_id", "content", "score", "metadata"} for r in rows)


def test_max_items_is_kept_and_a_node_with_nothing_showable_is_passed_over():
    g = _graph(0)
    full = SurfacingMonitor(g, _Vdb(), _cfg())
    _run(g, 0, 9, full)
    g.config[KEY] = True
    top = full.get_surfaced(5)
    assert len(top) == 5
    bare = {top[0]["node_id"], top[2]["node_id"]}
    holes = SurfacingMonitor(g, _Vdb(bare), _cfg())
    got = holes.get_surfaced(5)
    assert len(got) == 5 and not bare & {r["node_id"] for r in got}
    assert [r["node_id"] for r in got][:3] == [top[1]["node_id"], top[3]["node_id"], top[4]["node_id"]]


def test_key_off_is_the_pushed_queue_and_the_read_writes_nothing():
    g = _graph(1)
    mon = SurfacingMonitor(g, _Vdb(), _cfg())
    _run(g, 1, 9, mon)
    assert mon._native_surfaced(5) is None
    fresh = SurfacingMonitor(g, _Vdb(), _cfg())       # never pushed: the laptop's state since 2026-09-07
    assert fresh.get_surfaced(5) == []
    before = S.state(g)
    g.config[KEY] = True
    assert len(fresh.get_surfaced(5)) == 5              # the same monitor, read from the spike stamps
    g.config.pop(KEY)
    assert S.state(g) == before


def test_a_graph_that_is_not_native_logs_once_and_stays_on_the_queue(caplog):
    g = _graph(0, "dict")
    mon = SurfacingMonitor(g, _Vdb(), _cfg())
    _run(g, 0, 6, mon)
    want = mon.get_surfaced(ALL)
    g.config[KEY] = True
    with caplog.at_level("ERROR"):
        for _ in range(3):
            assert mon.get_surfaced(ALL) == want
    assert sum(KEY in r.message for r in caplog.records) == 1


def test_bad_parameters_raise():
    g = _graph(0)
    for decay in (0.0, 1.5, float("nan")):
        with pytest.raises(ValueError):
            NF.ng_tract.surfacing_scan(g.nodes, g.hyperedges, g.timestep, decay, 0.3, 0.5, 0.3, 0.2, 5)
    with pytest.raises(ValueError):
        NF.ng_tract.surfacing_scan(g.nodes, g.hyperedges, g.timestep, 0.95, float("inf"), 0.5, 0.3, 0.2, 5)
    with pytest.raises(TypeError):
        NF.ng_tract.surfacing_scan(g.nodes, {}, g.timestep, 0.95, 0.3, 0.5, 0.3, 0.2, 5)


def test_a_long_run_of_unshowable_nodes_at_the_top_does_not_leave_the_result_short():
    """Review 2026-11-23 (Grok 4.7): the candidate window widens until max_items are showable."""
    g = _graph(2)
    full = SurfacingMonitor(g, _Vdb(), _cfg())
    _run(g, 2, 14, full)
    g.config[KEY] = True
    order = [r["node_id"] for r in full.get_surfaced(ALL)]
    assert len(order) >= 45
    holes = SurfacingMonitor(g, _Vdb(order[:42]), _cfg())      # more unshowable nodes than the first window (40)
    got = holes.get_surfaced(3)
    assert [r["node_id"] for r in got] == order[42:45]


def test_the_monitors_own_floor_admits_and_the_graphs_floor_drops():
    g = _graph(0)
    cfg = _cfg()
    cfg.surfacing.min_confidence = 10.0                          # nothing is admitted
    mon = SurfacingMonitor(g, _Vdb(), cfg)
    _run(g, 0, 6, mon)
    g.config[KEY] = True
    assert mon.get_surfaced(ALL) == []
