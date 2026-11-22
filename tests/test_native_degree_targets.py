# ---- Changelog ----
# [2026-11-21] DudeMan CC (Claude Fable 5.2, laptop) -- the homeostat's degree targets as a NodeStore column
#   (config native_degree_targets_enabled; NodeStore.refresh_degree_targets; SQ10).
# What: (1) on ONE graph at one moment, the Python refresh and then the native refresh: the same target for every
#       node (exact floats), the same diffpc_layer and manifold_type; (2) twin graphs stepped side by side, one with
#       the key, through several scaling intervals: the same node state, targets and checkpoint bytes at every step;
#       (3) the dict moves into the column when the key turns on and back when it turns off; (4) the checkpoint key
#       homeostatic_degree_targets is the column's export and restores into it; (5) a graph that is not native logs
#       once and keeps its dict; (6) the declared difference: a removed node's stale target is not kept; (7) review
#       2026-11-21 (Grok 4.7): a rule shared with a second graph (the sleep-observe shadow) never empties the live
#       column, the shadow has its own copy; sensitivity 0; a NaN prediction error; a NaN fraction raises; a declined
#       homeostat pass raises with the column in use.
# Why:  the Python refresh was 19 % of the step after native propagation (profile LS1121NP); Josh 2026-11-21 "moving
#       all of this stuff into Rust and then removing the Python is the entire point of the exercise".
# How:  tests.test_hyperedge_store_h2 built() start graphs and Side (identical uuid / random streams).
# -------------------
import os
import random
import sys
import tempfile

import msgpack
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import neuro_foundation as NF  # noqa: E402
from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s18 as S18        # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(getattr(NF.ng_tract, "NodeStore", None), "refresh_degree_targets"),
                                reason="installed ng_tract has no refresh_degree_targets")

KEY = "native_degree_targets_enabled"
FAST = {"scaling_interval": 4, "homeostatic_degree_targets_persist": True}


def _hr(g):
    return next(r for r in g._plasticity_rules if isinstance(r, NF.HomeostaticRule))


def _marks(g):
    return {nid: (n.diffpc_layer, n.manifold_type) for nid, n in g.nodes.items()}


def _graph(seed, cfg=None):
    g = S.restore_as(NF, "native", H2.built(seed))
    g.config.update(dict(S18.ARM, **FAST, **(cfg or {})))
    return g


def _drive(side, seed, i):
    def op(g):
        rng = random.Random(seed * 100 + i)
        ids = sorted(g.nodes.keys())
        for nid in rng.sample(ids, min(12, len(ids))):
            g.nodes[nid].voltage += 1.5
        return g.step()
    return side(op)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("frac", [0.2, 0.6])
def test_the_native_refresh_gives_the_python_refresh(seed, frac):
    g = _graph(seed, {"gsg_spherical_fraction": frac})
    side = H2.Side(g, seed)
    for i in range(9):                      # live prediction errors, firing rates and a changed topology
        _drive(side, seed, i)
    rng = random.Random(seed)
    for nid in sorted(g.nodes.keys()):      # distinct prediction errors, so the candidate cut is not all ties
        g.nodes[nid].pred_error_ema = rng.uniform(-1.0, 1.0)
    hr = _hr(g)
    hr._refresh_degree_targets(g)
    py_t, py_m = dict(hr._degree_targets), _marks(g)
    assert py_t and len(set(py_t.values())) > 1
    assert {"spherical", "hyperbolic"} <= {m for _, m in py_m.values()}        # not vacuous
    assert len({layer for layer, _ in py_m.values()}) > 1
    for nid in g.nodes.keys():              # scramble, so the native pass has to write every one
        g.nodes[nid].diffpc_layer = 7
        g.nodes[nid].manifold_type = "x"
    g.config[KEY] = True
    hr._refresh_degree_targets(g)
    assert g.nodes.degree_targets_on() and hr._degree_targets == {}
    assert hr.targets_dict(g) == py_t
    assert list(hr.targets_dict(g)) == list(py_t)       # the same order as the dict the Python built
    assert _marks(g) == py_m


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("arm", ["arm", "receiver"])
def test_twins_stay_identical_through_scaling_intervals(seed, arm):
    extra = {} if arm == "arm" else dict(S18.RPE, **S18.ON)
    start = H2.built(seed)
    a, b = S.restore_as(NF, "native", start), S.restore_as(NF, "native", start)
    for g in (a, b):
        g.config.update(dict(S18.ARM, **FAST, **extra))
    b.config[KEY] = True
    sa, sb = H2.Side(a, seed), H2.Side(b, seed)
    refreshed = 0
    for i in range(22):
        _drive(sa, seed, i)
        _drive(sb, seed, i)
        assert S.state(a) == S.state(b), i
        assert _hr(a).targets_dict(a) == _hr(b).targets_dict(b), i
        assert _marks(a) == _marks(b), i
        refreshed += bool(_hr(b).targets_dict(b))
    assert b.nodes.degree_targets_on() and not a.nodes.degree_targets_on() and refreshed > 10
    b.config.pop(KEY)                       # the checkpoint carries the config; everything else must be equal
    assert S18.ckpt(a) == S18.ckpt(b)


def test_the_dict_moves_into_the_column_and_back():
    g = _graph(0)
    hr = _hr(g)
    hr._refresh_degree_targets(g)
    before = dict(hr._degree_targets)
    assert before and g.nodes.degree_targets_len() == 0
    g.config[KEY] = True
    assert hr.targets_arg(g) is None
    assert hr._degree_targets == {} and g.nodes.degree_targets_len() == len(before)
    assert hr.targets_dict(g) == before
    g.config[KEY] = False
    assert hr.targets_arg(g) == before and hr._degree_targets == before
    assert g.nodes.degree_targets_len() == 0 and not g.nodes.degree_targets_on()


def test_the_checkpoint_key_is_the_column_and_restores_into_it():
    g = _graph(1, {KEY: True})
    side = H2.Side(g, 1)
    for i in range(6):
        _drive(side, 1, i)
    hr = _hr(g)
    want = hr.targets_dict(g)
    assert want and g.nodes.degree_targets_on()
    raw = S18.ckpt(g)
    assert msgpack.unpackb(raw, raw=False, strict_map_key=False)["homeostatic_degree_targets"] == want
    g2 = S.restore_as(NF, "native", raw)
    g2.config.update(g.config)
    h2 = _hr(g2)
    assert h2._degree_targets == want and not g2.nodes.degree_targets_on()      # held as restored until the first use
    assert h2.targets_arg(g2) is None
    assert h2.targets_dict(g2) == want and h2._degree_targets == {}
    # restoring a checkpoint over a graph whose column is in use refills the column from the checkpoint
    g.nodes.degree_targets_clear()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        with open(p, "wb") as f:
            f.write(raw)
        g.restore(p)
    assert _hr(g).targets_arg(g) is None and _hr(g).targets_dict(g) == want


def test_a_graph_that_is_not_native_logs_once_and_keeps_its_dict(caplog):
    g = S.restore_as(NF, "dict", H2.built(0))
    g.config[KEY] = True
    hr = _hr(g)
    with caplog.at_level("ERROR"):
        for _ in range(3):
            hr._refresh_degree_targets(g)
            assert hr.targets_arg(g) is hr._degree_targets
    assert hr._degree_targets and not hr._col_on(g.nodes)
    assert sum(KEY in r.message for r in caplog.records) == 1


def test_declared_difference_a_removed_nodes_stale_target_is_not_kept():
    g = _graph(2, {KEY: True})
    hr = _hr(g)
    hr._refresh_degree_targets(g)
    full = hr.targets_dict(g)
    gone = sorted(g.nodes.keys())[0]
    g.remove_node(gone)
    left = hr.targets_dict(g)
    assert gone in full and gone not in left and len(left) == len(full) - 1


def test_an_unusable_saved_value_raises_before_anything_is_written():
    g = _graph(0, {KEY: True})
    hr = _hr(g)
    hr._refresh_degree_targets(g)
    before = hr.targets_dict(g)
    nid = next(iter(before))
    with pytest.raises(TypeError):
        g.nodes.degree_targets_import({nid: "not a number"})
    assert hr.targets_dict(g) == before


def test_a_rule_shared_with_the_sleep_shadow_never_empties_the_live_column():
    """Review 2026-11-21: the in-use state is the store's. The shadow shares the rule object; whichever of the two
    graphs reaches targets_arg first, with the key on or forced off on the shadow, the live targets stay."""
    for when in ("pending", "in_use"):
        for shadow_key in (True, False):
            g = _graph(0)
            hr = _hr(g)
            hr._refresh_degree_targets(g)                 # the dict holds the targets
            want = dict(hr._degree_targets)
            g.config[KEY] = True
            if when == "in_use":
                assert hr.targets_arg(g) is None          # moved into the live column
            shadow, _hold = g._sleep_observe_shadow_graph({} if shadow_key else {KEY: False})
            assert _hr(shadow) is hr
            got = hr.targets_arg(shadow)                  # the shadow reaches the rule first / in between
            if shadow_key:
                assert got is None and hr.targets_dict(shadow) == want      # its own copy of the live column
            assert hr.targets_arg(g) is None
            assert hr.targets_dict(g) == want, (when, shadow_key)
            assert g.nodes.degree_targets_on()


def test_sensitivity_zero_clears_the_targets_and_leaves_the_marks():
    g = _graph(1, {KEY: True})
    hr = _hr(g)
    hr._refresh_degree_targets(g)
    assert hr.targets_dict(g)
    marks = _marks(g)
    g.config["degree_sensitivity"] = 0.0
    hr._sync_from_config(g)
    assert hr.targets_dict(g) == {} and g.nodes.degree_targets_on() and _marks(g) == marks
    assert hr.targets_arg(g) is None


def test_a_nan_prediction_error_is_never_a_candidate_and_a_nan_fraction_raises():
    g = _graph(2, {KEY: True, "gsg_spherical_fraction": 0.9})
    hr = _hr(g)
    ids = sorted(g.nodes.keys())
    for nid in ids:
        g.nodes[nid].pred_error_ema = 0.0
    hr._refresh_degree_targets(g)
    sph = [n for n in ids if g.nodes[n].manifold_type == "spherical"]
    assert sph
    g.nodes[sph[0]].pred_error_ema = float("nan")
    hr._refresh_degree_targets(g)
    assert g.nodes[sph[0]].manifold_type == "hyperbolic"
    before = (hr.targets_dict(g), _marks(g))
    g.config["gsg_spherical_fraction"] = float("nan")
    with pytest.raises(ValueError):
        hr._refresh_degree_targets(g)
    assert (hr.targets_dict(g), _marks(g)) == before


def test_a_declined_homeostat_pass_raises_with_the_column_in_use():
    g = _graph(0, {KEY: True})
    side = H2.Side(g, 0)
    _drive(side, 0, 0)
    nid = sorted(g.nodes.keys())[0]
    g.nodes[nid].threshold = 3                       # an int: kept as an exact-Python value the pass does not read
    with pytest.raises(RuntimeError, match="native_degree_targets_enabled"):
        _hr(g).apply(g, [], g.timestep)
