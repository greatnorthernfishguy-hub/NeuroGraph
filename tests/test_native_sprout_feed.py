# ---- Changelog ----
# [2026-11-21] DudeMan CC (Claude Fable 5.2, laptop) -- _sprout_from_tally with the candidates kept in Rust
#   (config native_sprout_feed_enabled; PredictionBook.sprout_tally_feed; SQ10).
# What: (1) on one graph at one moment, the crossings the Python wrapper gets by passing ids out and back, and the
#       crossings of the one native call, are the same list in the same order, and leave the same tally; (2) twin
#       graphs stepped side by side, one with the key: identical node state, tally and checkpoint bytes every step,
#       with sprouts actually made; (3) no recent spike: 0 and no call counted, on both paths.
# Why:  profile LS1121DT: 8.4 % of the step was id tuples crossing the boundary twice.
# How:  tests.test_hyperedge_store_h2 built() start graphs and Side (identical uuid / random streams).
# -------------------
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import neuro_foundation as NF  # noqa: E402
from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s18 as S18        # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(getattr(NF.ng_tract, "PredictionBook", None), "sprout_tally_feed"),
                                reason="installed ng_tract has no sprout_tally_feed")

KEY = "native_sprout_feed_enabled"
TALLY = {"sprout_tally_enabled": True, "sprout_tally_slots": 4, "sprout_tally_theta": 1.2,
         "sprout_tally_horizon_steps": 200}


def _drive(side, seed, i, n=14):
    def op(g):
        rng = random.Random(seed * 100 + i)
        ids = sorted(g.nodes.keys())
        for nid in rng.sample(ids, min(n, len(ids))):
            g.nodes[nid].voltage += 1.5
        return g.step()
    return side(op)


def _twins(seed, extra=None):
    start = H2.built(seed)
    a, b = S.restore_as(NF, "native", start), S.restore_as(NF, "native", start)
    for g in (a, b):
        g.config.update(dict(S18.ARM, **TALLY, **(extra or {})))
    b.config[KEY] = True
    return a, b


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("arm", ["arm", "receiver"])
def test_twins_stay_identical_and_sprout(seed, arm):
    a, b = _twins(seed, {} if arm == "arm" else dict(S18.RPE, **S18.ON))
    sa, sb = H2.Side(a, seed), H2.Side(b, seed)
    for i in range(30):
        _drive(sa, seed, i)
        _drive(sb, seed, i)
        assert S.state(a) == S.state(b), i
        assert a.cofire_tally_state() == b.cofire_tally_state(), i
        assert sorted(a.synapses.keys()) == sorted(b.synapses.keys()), i
    sta, stb = a._sprout_tally_stats_dict(), b._sprout_tally_stats_dict()
    assert sta == stb and stb["calls"] > 0 and stb["crossings"] > 0, stb
    b.config.pop(KEY)
    assert S18.ckpt(a) == S18.ckpt(b)


@pytest.mark.parametrize("seed", [0, 1])
def test_one_call_gives_the_wrappers_crossings_in_order(seed):
    a, b = _twins(seed)
    sa, sb = H2.Side(a, seed), H2.Side(b, seed)
    for i in range(8):
        _drive(sa, seed, i)
        _drive(sb, seed, i)
    rng = random.Random(seed)
    fired = rng.sample(sorted(a.nodes.keys()), 10)
    prm = a._sprout_tally_params()
    window = a.config["co_activation_window"]
    recent = [tuple(x) for x in a._pb.spikes_recent_within(a.timestep, window, list(set(fired)))]
    assert recent
    recent.sort(key=lambda x: x[0])
    py = a._sprout_tally_feed(list(fired), [n for _, n in recent], prm)
    rs = b._pb.sprout_tally_feed(b.synapses, list(fired), b.timestep, window, prm["k"], prm["theta"], prm["lam"],
                                 prm["floor"], prm["gap"])
    assert [tuple(x) for x in rs] == py
    assert a.cofire_tally_state() == b.cofire_tally_state()


def test_no_recent_spike_returns_zero_and_counts_no_call():
    for on in (False, True):
        g = S.restore_as(NF, "native", H2.built(0))
        g.config.update(dict(S18.ARM, **TALLY))
        g.config[KEY] = on
        far = g.timestep + 10_000            # every recorded spike is older than the window
        g.timestep = far
        before = dict(g._sprout_tally_stats_dict())
        assert g._sprout_from_tally(sorted(g.nodes.keys())[:5]) == 0
        assert g._sprout_tally_stats_dict() == before


def test_a_bad_parameter_raises_and_touches_nothing():
    g = S.restore_as(NF, "native", H2.built(0))
    g.config.update(dict(S18.ARM, **TALLY))
    side = H2.Side(g, 0)
    for i in range(4):
        _drive(side, 0, i)
    before = g.cofire_tally_state()
    with pytest.raises(ValueError):
        g._pb.sprout_tally_feed(g.synapses, sorted(g.nodes.keys())[:3], g.timestep, 5, 0, 2.0, 0.5, 0.1, 3)
    assert g.cofire_tally_state() == before


def test_a_bad_parameter_raises_with_no_candidate_too():
    """Review 2026-11-22: the parameter check runs before the empty-window return."""
    g = S.restore_as(NF, "native", H2.built(0))
    with pytest.raises(ValueError):
        g._pb.sprout_tally_feed(g.synapses, [], g.timestep + 10_000, 5, 0, 2.0, 0.5, 0.1, 3)


@pytest.mark.parametrize("seed", [0, 1])
def test_duplicates_and_ids_that_are_not_nodes_match_the_wrapper(seed):
    """Review 2026-11-22: a fired list with a repeated id and an id that is no node gives the wrapper's crossings
    and tally on both paths."""
    a, b = _twins(seed)
    sa, sb = H2.Side(a, seed), H2.Side(b, seed)
    for i in range(8):
        _drive(sa, seed, i)
        _drive(sb, seed, i)
    rng = random.Random(seed + 7)
    base = rng.sample(sorted(a.nodes.keys()), 8)
    fired = base + [base[0], "no-such-node", base[3]]
    prm = a._sprout_tally_params()
    window = a.config["co_activation_window"]
    recent = [tuple(x) for x in a._pb.spikes_recent_within(a.timestep, window, list(set(fired)))]
    recent.sort(key=lambda x: x[0])
    assert len({dt for dt, _ in recent}) < len(recent)        # ties in dt: the stable order is exercised
    py = a._sprout_tally_feed(list(fired), [n for _, n in recent], prm)
    rs = b._pb.sprout_tally_feed(b.synapses, list(fired), b.timestep, window, prm["k"], prm["theta"], prm["lam"],
                                 prm["floor"], prm["gap"])
    assert [tuple(x) for x in rs] == py
    assert a.cofire_tally_state() == b.cofire_tally_state()


def test_the_key_with_the_python_tally_forced_stays_on_the_python_tally(monkeypatch):
    """Review 2026-11-22: with the native tally unavailable the key changes nothing (no state split between the
    Python tally and the store's)."""
    monkeypatch.setattr(NF, "_cofire_tally_native", lambda store: None)
    a, b = _twins(0)
    sa, sb = H2.Side(a, 0), H2.Side(b, 0)
    for i in range(12):
        _drive(sa, 0, i)
        _drive(sb, 0, i)
        assert S.state(a) == S.state(b), i
    assert a.__dict__.get("_cofire_tally") == b.__dict__.get("_cofire_tally")
    assert b.__dict__.get("_cofire_tally") is not None
