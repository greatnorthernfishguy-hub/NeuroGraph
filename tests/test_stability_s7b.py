# ---- Changelog ----
# [2026-10-23] Claude (lane stability-s7b, #1136 / #1141; Josh 2026-10-23 ~07:15 "go ahead and start on that node at the
#   limit work") — S7b: gain relaxation for afferented nodes whose input succeeds
# What: (1) the S7b keys absent / explicitly False == the S7 NG branch (b5bbed4's module; S7 changed no NG code) exactly on
#       the dict, NodeStore-only and native paths with the whole S6 set on (every op of the H2 drive workload incl.
#       sweeps + a sleep; final checkpoint BYTES); not in DEFAULT_CONFIG; set keys that cannot run log ERROR once and
#       change nothing. (2) with the key, HomeostaticRule passes relax=(slow excitability step x mult, from, stop,
#       resume, window) and the kernel relaxes only afferented rows whose input succeeds, at / near the rail; the S7
#       census reports "relax"; NodeStore.s7b_rows reads the per-row state. (3) nothing new is saved.
# Why:  hyperedge-store design §2.3 S7b row; S7 dev-log §8 option (a).
# How:  oracle via `git show b5bbed4:neuro_foundation.py` (tests.test_hyperedge_store_h2._load); the S / S6 helpers.
# [2026-10-23] Claude (lane stability-s7b) — law review: + relax alone on a native graph -> ERROR (L7), + a fractional
#   window is refused (L12).
# -------------------
"""#1136 round S7b: gain relaxation of afferented nodes whose input succeeds."""
import logging
import os
import random

import msgpack
import numpy as np
import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s6 as S6          # noqa: E402

NF = H2.NF
S7_REV = os.environ.get("NG_S7B_BASE_REV", "b5bbed4")
CMOD = H2._load(S7_REV, "nf_s7_commit_s7b")
HAVE_S7B = getattr(NF.ng_tract.NodeStore, "stab_round", 0) >= 8 and hasattr(NF.ng_tract.NodeStore, "s7b_rows")
needs_s7b = pytest.mark.skipif(not HAVE_S7B, reason="ng_tract without the S7b kernel")
S7B_SWITCHES = ("node_homeo_relax_enabled",)
S7B_PARAMS = ("node_homeo_relax_mult", "node_homeo_relax_from", "node_homeo_relax_stop", "node_homeo_relax_resume",
              "node_homeo_relax_window")
# the S7 arm's keys (S6 full set + the hopeless leak), as the engine sees them
S7_ARM = dict(S6.S6_FULL, node_homeo_hopeless_leak=0.004)
S7B_ARM = dict(S7_ARM, node_homeo_relax_enabled=True)
ckpt = S.ckpt
restore_as = S.restore_as


# ---------------------------------------------------------------------------
# (1) identities
# ---------------------------------------------------------------------------

@needs_s7b
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_is_the_s7_branch_exactly(kind, seed, how):
    start = H2.built(seed)
    t = S.Pair(restore_as(CMOD, kind, start), restore_as(NF, kind, start), seed)
    if kind == "native":
        t.op("keys", lambda g: g.config.update(S7_ARM))
    if how == "explicit_false":
        t.b.g.config.update({k: False for k in S7B_SWITCHES})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in S7B_SWITCHES:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


@needs_s7b
def test_keys_not_in_default_config():
    for k in S7B_SWITCHES + S7B_PARAMS:
        assert k not in NF.DEFAULT_CONFIG and k not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(CMOD.DEFAULT_CONFIG.items())


@needs_s7b
@pytest.mark.parametrize("case", ["no_slow", "no_input_evidence", "stop_over_fail_min", "resume_over_stop",
                                  "bad_window", "fractional_window", "bad_mult"])
def test_misconfigured_relax_logs_error_once_and_changes_nothing(case, caplog):
    cfg, extra = S7_ARM, dict(node_homeo_relax_enabled=True)
    if case == "no_slow":
        cfg = {k: v for k, v in S7_ARM.items() if k not in ("node_homeo_slow_enabled", "node_homeo_prop_enabled",
                                                            "node_homeo_hopeless_enabled", "node_homeo_agree_enabled")}
    elif case == "no_input_evidence":
        cfg = {k: v for k, v in S7_ARM.items() if k not in ("node_homeo_input_evidence_enabled",
                                                            "node_homeo_hopeless_enabled",
                                                            "syn_scale_quiet_hold_enabled")}
    else:
        extra.update({"stop_over_fail_min": dict(node_homeo_relax_stop=0.6),
                      "resume_over_stop": dict(node_homeo_relax_resume=0.4),
                      "bad_window": dict(node_homeo_relax_window=0.0),
                      "fractional_window": dict(node_homeo_relax_window=1000.5),
                      "bad_mult": dict(node_homeo_relax_mult=1e6)}[case])
    start = H2.built(0)
    t = S.Pair(restore_as(CMOD, "native", start), restore_as(NF, "native", start), 0)
    t.op("keys", lambda g: g.config.update(cfg))
    t.b.g.config.update(extra)
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 12, sleep_every=True)
    errs = [r for r in caplog.records if "stability set (#1136)" in r.getMessage() and r.levelno == logging.ERROR
            and "node_homeo_relax_enabled" in r.getMessage()]
    assert len(errs) == 1, [r.getMessage() for r in errs]
    for k in extra:
        t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)


@needs_s7b
def test_relax_on_a_dict_backed_graph_logs_error_and_stays_off(caplog):
    start = H2.built(0)
    t = S.Pair(restore_as(CMOD, "dict", start), restore_as(NF, "dict", start), 0)
    t.b.g.config.update(dict(S7_ARM, node_homeo_relax_enabled=True))
    t.a.g.config.update(S7_ARM)
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 10)
    assert any("S7b-round" in r.getMessage() for r in caplog.records)
    t.b.g.config.pop("node_homeo_relax_enabled")
    assert ckpt(t.a.g) == ckpt(t.b.g)


@needs_s7b
def test_relax_alone_on_a_native_graph_logs_error_and_stays_off(caplog):
    # [2026-10-23] law review L7: relax set with none of the N keys it refines
    start = H2.built(0)
    t = S.Pair(restore_as(CMOD, "native", start), restore_as(NF, "native", start), 0)
    t.b.g.config.update(node_homeo_relax_enabled=True)
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 10)
    errs = [r for r in caplog.records if "node_homeo_relax_enabled" in r.getMessage() and r.levelno == logging.ERROR]
    assert len(errs) == 1, [r.getMessage() for r in errs]
    t.b.g.config.pop("node_homeo_relax_enabled")
    assert ckpt(t.a.g) == ckpt(t.b.g)


# ---------------------------------------------------------------------------
# (2) the wiring and the rule on a real graph
# ---------------------------------------------------------------------------

def _graph(seed, cfg):
    return S6._graph(seed, cfg)


@needs_s7b
def test_stability_params_carry_the_relax_tuple():
    g = _graph(0, dict(S7B_ARM, node_homeo_relax_mult=2.0, node_homeo_relax_from=4.0, node_homeo_relax_window=500))
    assert g._stability_params().relax == (2.0, 4.0, 0.35, 0.2, 500.0)
    g = _graph(0, S7B_ARM)
    assert g._stability_params().relax == (1.0, 4.5, 0.35, 0.2, 1000.0)
    assert _graph(0, S7_ARM)._stability_params().relax is None


@needs_s7b
def test_relax_moves_only_rail_rows_whose_input_succeeds_on_a_real_graph():
    # one twin with relax, one without; every node put at the rail; some nodes are paced with a strong input that
    # fires them (their input SUCCEEDS), the rest get a weak input that fails (driven and failing)
    cfg = {k: v for k, v in S7_ARM.items() if not k.startswith("interneuron")}
    cfg.update(spontaneous_activity_enabled=False)
    a = _graph(7, cfg)
    b = _graph(7, dict(cfg, node_homeo_relax_enabled=True))
    ids = list(a.nodes)
    for g in (a, b):
        for nid in ids:
            g.nodes[nid].intrinsic_excitability = 5.0
    good = ids[: len(ids) // 3]
    for _ in range(300):
        for g in (a, b):
            for nid in good:
                g.stimulate(nid, 3.0)
            g.step()
    xa, = S6._cols(a, ("intrinsic_excitability",))
    xb, = S6._cols(b, ("intrinsic_excitability",))
    rows = {r[0]: r for r in b.nodes.s7b_rows(0.0)}
    relaxed = [nid for k, nid in enumerate(ids) if xb[k] < xa[k] - 1e-12]
    assert relaxed, "some rail row relaxed"
    for nid in relaxed:
        r = rows[nid]
        # it relaxed because its input succeeded: a recent success and fail_ema under the stop threshold (or it has
        # since stopped by the latch: rx 2)
        assert r[7] >= 0 and r[10] in (1, 2), r
    # nobody went under baseline, and no row WITHOUT a recorded success relaxed
    assert (xb >= 1.0 - 1e-12).all()
    assert all(rows[n][7] >= 0 for n in relaxed)
    st = b.nodes.s7_state()
    assert st["branches"][7] == "relax" and st["branches"][8] == "relax_stopped"
    assert len(st["all"]) == 9
    sa = a.nodes.s7_state()
    assert sa["all"][7] == 0 and sa["all"][8] == 0


# ---------------------------------------------------------------------------
# (3) nothing new is saved
# ---------------------------------------------------------------------------

@needs_s7b
def test_nothing_new_is_saved_and_the_s7_branch_reads_the_file():
    g = _graph(6, S7B_ARM)
    for _ in range(30):
        g.step()
    d = msgpack.unpackb(ckpt(g), raw=False, strict_map_key=False)
    node_keys = set()
    for nd in d["nodes"].values():
        node_keys |= set(nd)
    assert not ({"last_succ", "rx", "relax", "s7_rowbr"} & node_keys)
    h = restore_as(CMOD, "native", ckpt(g))
    assert len(h.nodes) == len(g.nodes)
    h.step()
    assert np.isfinite(S6._cols(h, ("intrinsic_excitability",))[0]).all()
