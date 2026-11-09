# ---- Changelog ----
# [2026-10-20] Claude (lane stability-s4, #1136 round S4; Josh 2026-10-20 "Good plan, let's do this plan") — proportional,
#   saturating node homeostasis + giving up on hopeless inputs: off == the S3 commit, S3 keys alone == S3, behaviour
# What: (1) every S4 key absent / explicitly False == the S3 commit (b7aa073's module) exactly on the dict, NodeStore-only
#       and native paths (every op of the H2 drive workload + a sleep; final checkpoint BYTES); the S3 candidate's keys
#       alone (+ horizon 500, + orphan S4 parameters) == S3 bitwise; S4 keys without slow, hopeless without evidence by
#       input, and out-of-range parameters log ERROR once and change nothing. (2) prop: a driven-failing node's moves
#       shrink with its headroom (it approaches its rail, never reaches it) and an over-target node far from target moves
#       more than one near it; hopeless: a node whose input always fails and that never fires stops being wound up after
#       the window, while a node a strong input fires now and then keeps being helped; the full S4 set runs the
#       workload without a declined pass and round-trips (older readers too).
# Why:  spec superpowers/specs/2026-10-17-rust-hyperedge-store-design.md §2.3 (S4 rows) / §3.2.
# How:  oracle loaded with `git show <rev>:neuro_foundation.py` (tests.test_hyperedge_store_h2._load); the S / S2 / S3
#       tests' helpers reused; PYTHONHASHSEED fixed by the runner.
# -------------------
"""#1136 round S4: proportional, saturating node homeostasis and giving up on hopeless inputs."""
import logging
import os
import random

import msgpack
import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s3 as S3          # noqa: E402

NF = H2.NF
S3_REV = os.environ.get("NG_S4_S3_REV", "b7aa073")   # the S3 commit (3285f29's neuro_foundation.py + its tests)
S3MOD = H2._load(S3_REV, "nf_s3_commit_s4")
HAVE_S4 = S3.HAVE_S3 and getattr(NF.ng_tract.NodeStore, "stab_round", 0) >= 4
needs_s4 = pytest.mark.skipif(not HAVE_S4, reason="ng_tract without the S4 kernels")

S4_SWITCHES = ("node_homeo_prop_enabled", "node_homeo_hopeless_enabled")
S4_PARAMS = {"node_homeo_prop_unit": 3.0, "node_homeo_prop_max": 2.0, "node_homeo_hopeless_fail": 0.9,
             "node_homeo_hopeless_window": 200.0, "node_homeo_hopeless_leak": 0.05}
S3_FULL = S3.S3_FULL
S4_FULL = dict(S3_FULL, node_homeo_prop_enabled=True, node_homeo_hopeless_enabled=True)
ckpt = S.ckpt
restore_as = S.restore_as


# ---------------------------------------------------------------------------
# (1) identities
# ---------------------------------------------------------------------------

@needs_s4
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_is_the_s3_commit_exactly(kind, seed, how):
    start = H2.built(seed)
    t = S.Pair(restore_as(S3MOD, kind, start), restore_as(NF, kind, start), seed)
    if how == "explicit_false":
        t.b.g.config.update({k: False for k in S4_SWITCHES})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in S4_SWITCHES:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)
    assert not hasattr(t.b.g, "_stab")


@needs_s4
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("keys", ["s3_full", "s3_tau500", "s3_orphan_s4_params"])
def test_s3_keys_alone_equal_the_s3_commit_bitwise(seed, keys):
    cfg = {"s3_full": S3_FULL, "s3_tau500": dict(S3_FULL, syn_scale_rate_tau=500.0), "s3_orphan_s4_params": S3_FULL}[keys]
    start = H2.built(seed)
    t = S.Pair(restore_as(S3MOD, "native", start), restore_as(NF, "native", start), seed)
    t.op("keys", lambda g: g.config.update(cfg))
    if keys == "s3_orphan_s4_params":
        t.b.g.config.update(S4_PARAMS)
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in S4_PARAMS:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


@needs_s4
@pytest.mark.parametrize("case", ["no_slow", "hopeless_no_input_ev", "bad_unit", "bad_max", "bad_window", "bad_leak"])
def test_misconfigured_s4_keys_log_error_once_and_change_nothing(case, caplog):
    if case == "no_slow":
        extra, cfg = dict(node_homeo_prop_enabled=True, node_homeo_hopeless_enabled=True), \
            {k: v for k, v in S3_FULL.items() if k != "node_homeo_slow_enabled"}
    elif case == "hopeless_no_input_ev":
        extra = dict(node_homeo_hopeless_enabled=True)
        cfg = {k: v for k, v in S3_FULL.items() if k != "node_homeo_input_evidence_enabled"}
    else:
        cfg = S3_FULL
        extra = {"bad_unit": dict(node_homeo_prop_enabled=True, node_homeo_prop_unit=1.0),
                 "bad_max": dict(node_homeo_prop_enabled=True, node_homeo_prop_max=0.0),
                 "bad_window": dict(node_homeo_hopeless_enabled=True, node_homeo_hopeless_window=0.0),
                 "bad_leak": dict(node_homeo_hopeless_enabled=True, node_homeo_hopeless_leak=1.5)}[case]
    start = H2.built(0)
    t = S.Pair(restore_as(S3MOD, "native", start), restore_as(NF, "native", start), 0)
    t.op("keys", lambda g: g.config.update(cfg))
    t.b.g.config.update(extra)
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 15, sleep_every=True)
    errs = [r for r in caplog.records if "stability set (#1136)" in r.getMessage() and r.levelno == logging.ERROR]
    assert len(errs) == 1, [r.getMessage() for r in errs]
    for k in extra:
        t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)
    assert t.b.g._stab is None or (t.b.g._stab.prop is None and t.b.g._stab.hope is None)


@needs_s4
def test_s4_keys_not_in_default_config_and_default_config_is_the_tips():
    for k in S4_SWITCHES + tuple(S4_PARAMS):
        assert k not in NF.DEFAULT_CONFIG and k not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(S.TIP.DEFAULT_CONFIG.items())


@needs_s4
@pytest.mark.parametrize("seed", [0, 3])
def test_full_s4_set_runs_the_workload_without_a_declined_pass_and_round_trips(seed):
    start = H2.built(seed)
    g = restore_as(NF, "native", start)
    g.config.update(S4_FULL)
    side = H2.Side(g, seed)

    class _T:
        n = side

        def op(self, name, fn):
            return side(fn)
    H2.drive(_T(), random.Random(seed), 25)
    g.sleep_cycle()
    assert g.hyperedges.pass_declines() == (0, 0, 0, 0)
    assert g._stab.prop == (2.0, 4.0) and g._stab.hope == (0.95, 1000.0, 0.0)
    b = ckpt(g)
    d1, d2 = (msgpack.unpackb(x, raw=False, strict_map_key=False) for x in (b, ckpt(restore_as(NF, "native", b))))
    assert d1["nodes"] == d2["nodes"]                       # no new node key: the S3 rows exactly
    assert all(set(nd) == set(next(iter(d1["nodes"].values()))) for nd in d1["nodes"].values())
    for M, kind in ((S3MOD, "native"), (S3MOD, "dict"), (S.TIP, "dict")):   # older readers step on
        h = restore_as(M, kind, b)
        h.step()


# ---------------------------------------------------------------------------
# (2) behaviour
# ---------------------------------------------------------------------------

SLOW1 = {"node_homeo_slow_enabled": True, "node_homeo_slow_rate_mult": 1.0}


@needs_s4
def test_prop_up_moves_saturate_with_headroom():
    """A driven-failing silent node (threshold 40: its input always fails): S3 raises its excitability by the same
    factor whether it sits at 1.0 or near the 5.0 rail (and piles onto it); with prop the move is max x the step x the headroom left
    toward the rail, so near the rail it barely moves (and never lands on it)."""
    def grow(cfg, x0):
        g, d = S3._driven_run(dict(SLOW1, **cfg), steps=0)
        g.nodes[d].threshold = 40.0
        g.nodes[d].intrinsic_excitability = x0
        for k in range(40):
            S._pace(g, k)
            g.stimulate(d, 0.05)
            g.step()
        assert g.nodes[d].last_spike_time == float("-inf") or g.nodes[d].threshold > 30   # it never fired
        return g.nodes[d].intrinsic_excitability / x0
    s3_lo, s3_hi = grow({}, 1.0), grow({}, 4.8)
    p_lo, p_hi = grow({"node_homeo_prop_enabled": True}, 1.0), grow({"node_homeo_prop_enabled": True}, 4.8)
    assert s3_lo > 1.0 and 4.8 * s3_hi == 5.0                           # S3: the same factor -> lands on the rail
    assert p_lo - 1.0 > 3.0 * (s3_lo - 1.0)                             # silent: the max distance (4 steps), room ~1
    assert 0 < p_hi - 1.0 < 0.1 * (p_lo - 1.0)                          # near the rail: saturated
    assert 4.8 * p_hi < 5.0


def _hope_run(cfg, kick_every=None, steps=300, stim=0.05, thr=4.0, x0=1.0):
    """y1: input arrives every step and always fails (threshold 4.0); with kick_every, a strong input fires it now and
    then (a cue that reaches it)."""
    g, d = S3._driven_run(dict(SLOW1, **cfg), steps=0)
    g.nodes[d].threshold = thr
    g.nodes[d].intrinsic_excitability = x0
    for k in range(steps):
        S._pace(g, k)
        g.stimulate(d, 50.0 if (kick_every and k % kick_every == 0) else stim)
        g.step()
    return g, d


@needs_s4
def test_hopeless_stops_the_wind_up_but_not_for_a_node_that_sometimes_succeeds():
    H = {"node_homeo_hopeless_enabled": True, "node_homeo_hopeless_window": 50.0}
    g3, d = _hope_run({})
    gh, _ = _hope_run(H)
    fall3, fallh = 4.0 - g3.nodes[d].threshold, 4.0 - gh.nodes[d].threshold
    assert fall3 > 0.25                                 # S3: wound up the whole run
    assert 0 <= fallh < 0.25 * fall3                    # hopeless after ~a dozen failed inputs: no further up-drive
    xh = gh.nodes[d].intrinsic_excitability
    assert xh < g3.nodes[d].intrinsic_excitability
    # a node a strong input fires every 40 steps (inside the 50-step window) is never hopeless
    g3k, _ = _hope_run({}, kick_every=40)
    ghk, _ = _hope_run(H, kick_every=40)
    assert ghk.nodes[d].threshold == g3k.nodes[d].threshold
    assert ghk.nodes[d].intrinsic_excitability == g3k.nodes[d].intrinsic_excitability


@needs_s4
def test_hopeless_leak_relaxes_to_baseline():
    """Threshold 3.0 / excitability 3.0, a weak input (0.02: v ~0.67 x excitability) that always fails: once hopeless,
    hold keeps the node where giving up found it; leak relaxes it to baseline (0.85 / 1.0), where it still fails."""
    H = {"node_homeo_hopeless_enabled": True, "node_homeo_hopeless_window": 50.0}
    kw = dict(steps=600, stim=0.02, thr=3.0, x0=3.0)
    gh, d = _hope_run(H, **kw)
    gl, _ = _hope_run(dict(H, node_homeo_hopeless_leak=0.2), **kw)
    assert gh.nodes[d].threshold > 2.5 and gh.nodes[d].intrinsic_excitability > 2.5
    assert gl.nodes[d].threshold == pytest.approx(0.85, abs=0.01)
    assert gl.nodes[d].intrinsic_excitability == pytest.approx(1.0, abs=0.01)
    assert gl.nodes[d].last_spike_time == float("-inf")
