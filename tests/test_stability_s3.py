# ---- Changelog ----
# [2026-10-20] Claude (lane stability-s3, #1136 round S3; Josh 2026-10-20 "I concur with your recommendations. Start the
#   next round") — slow node homeostasis: off == the S2 commit, S2 keys alone == the S2 commit, the slow rule's behaviour
# What: (1) every S3 key absent / explicitly False == the S2 commit (9abe5c3's module) exactly, on the dict, NodeStore-only
#       and native paths (every op of the H2 drive workload + a sleep, strictly; the final checkpoint BYTES); with ONLY
#       the S2 keys set (the S2 full set + t4's afferent horizon, and + recency), the native path == the S2 commit
#       bitwise (bytes included); S3 parameter keys without the switch, and the switch without N, change nothing.
#       (2) Slow: a driven-but-failing node's threshold / excitability move at rate_mult x the rule's rates; the
#       decision reads the LONG rate (a brief burst that puts the fast rate over target does not raise the threshold;
#       a long quiet history keeps it under target); with S off the long rate still advances and is saved; the full S3
#       set runs the workload without a declined pass and round-trips (older readers too).
# Why:  spec superpowers/specs/2026-10-17-rust-hyperedge-store-design.md §2.3 (S3 row) / §3.2.
# How:  oracles loaded with `git show <rev>:neuro_foundation.py` (tests.test_hyperedge_store_h2._load); the S / S2 tests'
#       Pair / drive / restore helpers reused; PYTHONHASHSEED fixed by the runner.
# -------------------
"""#1136 round S3: slow node homeostasis (N judges the long rate, at a fraction of today's rates)."""
import os
import random

import msgpack
import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s2 as S2          # noqa: E402

NF = H2.NF
S2_REV = os.environ.get("NG_S3_S2_REV", "9abe5c3")   # the S2 commit (274a4a6's neuro_foundation.py + its tests)
S2MOD = H2._load(S2_REV, "nf_s2_commit_s3")
HAVE_S3 = S2.HAVE_S2 and getattr(NF.ng_tract.NodeStore, "stab_round", 0) >= 3
needs_s3 = pytest.mark.skipif(not HAVE_S3, reason="ng_tract without the S3 kernels")

S3_KEYS = ("node_homeo_slow_enabled", "node_homeo_slow_rate_mult")
# the S2 settle runs' t4 shape (afferent horizon x10), recency off: the S3 candidate's base
T4L = dict(S2.S2_FULL, node_homeo_input_alpha=0.0002, node_homeo_input_rate_min=0.00005)
S3_FULL = dict(T4L, node_homeo_slow_enabled=True)
ckpt = S.ckpt
restore_as = S.restore_as


# ---------------------------------------------------------------------------
# (1) identities
# ---------------------------------------------------------------------------

@needs_s3
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_is_the_s2_commit_exactly(kind, seed, how):
    start = H2.built(seed)
    t = S.Pair(restore_as(S2MOD, kind, start), restore_as(NF, kind, start), seed)
    if how == "explicit_false":
        t.b.g.config.update({"node_homeo_slow_enabled": False})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    t.b.g.config.pop("node_homeo_slow_enabled", None)
    assert ckpt(t.a.g) == ckpt(t.b.g)
    assert not hasattr(t.b.g, "_stab")


@needs_s3
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("keys", ["s2_full", "t4l", "t4_recent", "t4l_orphan_s3_params"])
def test_s2_keys_alone_equal_the_s2_commit_bitwise(seed, keys):
    """With only S2 keys set, the S3 module makes exactly S2's native calls (bytes included). The orphan case: S3
    PARAMETER keys without the switch change nothing either."""
    cfg = {"s2_full": S2.S2_FULL, "t4l": T4L, "t4_recent": dict(T4L, node_homeo_input_recent_enabled=True),
           "t4l_orphan_s3_params": T4L}[keys]
    start = H2.built(seed)
    t = S.Pair(restore_as(S2MOD, "native", start), restore_as(NF, "native", start), seed)
    t.op("keys", lambda g: g.config.update(cfg))
    if keys == "t4l_orphan_s3_params":
        t.b.g.config.update({"node_homeo_slow_rate_mult": 0.05})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    t.b.g.config.pop("node_homeo_slow_rate_mult", None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


@needs_s3
@pytest.mark.parametrize("seed", [0, 1])
def test_slow_without_n_changes_nothing(seed):
    """Slow is a refinement of N: with N off (here the S2 set minus N) the S2 commit exactly."""
    cfg = {k: v for k, v in T4L.items() if k != "node_homeo_evidence_enabled"}
    start = H2.built(seed)
    t = S.Pair(restore_as(S2MOD, "native", start), restore_as(NF, "native", start), seed)
    t.op("keys", lambda g: g.config.update(cfg))
    t.b.g.config.update({"node_homeo_slow_enabled": True, "node_homeo_slow_rate_mult": 0.1})
    S.drive(t, random.Random(seed), 20, sleep_every=True)
    for k in S3_KEYS:
        t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)


@needs_s3
def test_s3_keys_not_in_default_config_and_default_config_is_the_tips():
    for k in S3_KEYS:
        assert k not in NF.DEFAULT_CONFIG and k not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(S.TIP.DEFAULT_CONFIG.items())


@needs_s3
@pytest.mark.parametrize("seed", [0, 3])
def test_full_s3_set_runs_the_workload_without_a_declined_pass_and_round_trips(seed):
    start = H2.built(seed)
    g = restore_as(NF, "native", start)
    g.config.update(S3_FULL)
    side = H2.Side(g, seed)

    class _T:
        n = side

        def op(self, name, fn):
            return side(fn)
    H2.drive(_T(), random.Random(seed), 25)
    g.sleep_cycle()
    assert g.hyperedges.pass_declines() == (0, 0, 0, 0)
    assert g._stab.slow == (0.2, 1.0 / 2000.0)
    b = ckpt(g)
    # native read -> write: the node rows (incl. syn_rate_ema, the long rate slow reads) are a fixed point. The whole
    # file is not, on ANY branch with this workload (pre-existing, #1140 family, the same with the S2 keys alone):
    # he_active_predictions' predicted_targets (list(set)), recent_spikes and delay_buffer re-serialize in another order.
    d1, d2 = (msgpack.unpackb(x, raw=False, strict_map_key=False) for x in (b, ckpt(restore_as(NF, "native", b))))
    assert d1["nodes"] == d2["nodes"]
    assert all(len(nd) >= 21 and "syn_rate_ema" in nd for nd in d1["nodes"].values())
    for M, kind in ((S2MOD, "native"), (S2MOD, "dict"), (S.TIP, "dict")):   # older readers step on
        h = restore_as(M, kind, b)
        h.step()


# ---------------------------------------------------------------------------
# (2) the slow rule
# ---------------------------------------------------------------------------

def _driven_run(cfg, steps=200):
    """y1 is depolarized but failing (threshold 4.0, a weak input every step) and quiet (under target)."""
    g, y = S._homeo_graph(dict({"node_homeo_evidence_enabled": True, "node_homeo_leak": 0.1,
                                "node_homeo_input_evidence_enabled": True, "co_activation_window": 0}, **cfg))
    d = y[1]
    g.nodes[d].threshold = 4.0
    g.nodes[d].firing_rate_ema = 0.001
    for k in range(steps):
        S._pace(g, k)
        g.stimulate(d, 0.05)                           # v settles ~1.7 x excitability: never reaches the threshold
        g.step()
    return g, d


@needs_s3
@pytest.mark.parametrize("mult", [0.2, 0.1])
def test_slow_moves_a_driven_failing_node_at_rate_mult_of_todays_rates(mult):
    gf, d = _driven_run({})
    gs, _ = _driven_run({"node_homeo_slow_enabled": True, "node_homeo_slow_rate_mult": mult})
    fall_f = 4.0 - gf.nodes[d].threshold
    fall_s = 4.0 - gs.nodes[d].threshold
    up_f = gf.nodes[d].intrinsic_excitability - 1.0
    up_s = gs.nodes[d].intrinsic_excitability - 1.0
    assert fall_f > 0.1 and up_f > 0.0                               # today's rate: both move
    assert fall_s == pytest.approx(mult * fall_f, rel=1e-6)          # slow: the same evidence, mult x the step
    assert 0.0 < up_s < up_f
    assert gs.nodes.stab_columns()[1] is True                        # the long rate switched on (S is off here)


def _burst_run(slow):
    """y2 bursts for 30 steps (its FAST rate goes far over target), then goes quiet; its LONG rate barely moves."""
    cfg = {"node_homeo_evidence_enabled": True, "node_homeo_leak": 0.0, "node_homeo_input_evidence_enabled": True,
           "co_activation_window": 0}
    if slow:
        cfg.update(node_homeo_slow_enabled=True, node_homeo_slow_rate_mult=1.0)   # rate 1.0: isolate "which rate"
    g, y = S._homeo_graph(cfg)
    b = y[2]
    g.nodes[b].firing_rate_ema = 0.001
    g.nodes[b].threshold = 0.5
    for k in range(30):
        S._pace(g, k)
        g.stimulate(b, 2.0)
        g.step()
    return g.nodes[b].threshold


@needs_s3
def test_slow_judges_the_long_rate_a_brief_burst_does_not_raise_the_threshold():
    assert _burst_run(False) > 0.5          # today: the fast rate is over target within a few steps -> threshold up
    assert _burst_run(True) == 0.5          # slow: the long rate is still under target (and the node succeeds: hold)


@needs_s3
def test_slow_long_rate_is_saved_with_s_off_and_restored():
    gs, _ = _driven_run({"node_homeo_slow_enabled": True}, steps=60)
    b = ckpt(gs)
    h = restore_as(NF, "native", b)
    assert h.nodes.stab_columns()[1] is True
    assert h.nodes.stab_columns()[3] == gs.nodes.stab_columns()[3]
    assert ckpt(h) == b
