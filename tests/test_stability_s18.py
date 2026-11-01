# ---- Changelog ----
# [2026-11-01] DudeMan CC (Claude Opus 5.5, laptop; #1202 receiver-side predictions; Josh 2026-11-01 "Yeah, the design
#   looks good") — S18: a neuron is predicted when pending input charges it partway but cannot fire it alone
# What: (1) the switch absent / explicitly False == the synced copy line (1d0a1a0's module) exactly (final checkpoint
#       BYTES); keys not in DEFAULT_CONFIG; bad params log ERROR once and change nothing. (2) the predictive state:
#       partway-not-over, graded, inhibition nets, one state per neuron, capacity by strength, pre-charge never fires the
#       target, each neuron's own rate. (3) credit by share, one-branch gated chains, checkpoints the base loader reads.
# Why:  docs/superpowers/specs/2026-11-01-receiver-side-prediction-design.md; plan 2026-11-01-receiver-side-prediction.md.
# How:  oracle via `git show 1d0a1a0:neuro_foundation.py` (tests.test_hyperedge_store_h2._load); the S / S7b helpers;
#       small hand-built graphs for the mechanism.
# -------------------
"""#1202 round S18: receiver-side, graded predictions."""
import logging
import os
import random

import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s7b as S7B        # noqa: E402

NF = H2.NF
BASE_REV = os.environ.get("NG_S18_BASE_REV", "1d0a1a0")
CMOD = H2._load(BASE_REV, "nf_s17_commit_s18")
S18_SWITCHES = ("prediction_receiver_enabled",)
S18_PARAMS = ("prediction_receiver_rho0", "prediction_receiver_k", "prediction_receiver_eta",
              "prediction_receiver_rho_min", "prediction_receiver_ema_alpha", "prediction_receiver_chain_floor",
              "prediction_receiver_margin")
ARM = S7B.S7B_ARM
RPE = {"three_factor_enabled": True, "reward_rpe_enabled": True}
ON = {"prediction_receiver_enabled": True}
ckpt = S.ckpt
restore_as = S.restore_as


# ---------------------------------------------------------------------------
# (1) the switch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_is_the_base_branch_exactly(kind, seed, how):
    start = H2.built(seed)
    t = S.Pair(restore_as(CMOD, kind, start), restore_as(NF, kind, start), seed)
    t.op("keys", lambda g: g.config.update(dict(ARM, **RPE) if kind == "native" else RPE))
    if how == "explicit_false":
        t.b.g.config.update({k: False for k in S18_SWITCHES})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in S18_SWITCHES:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


def test_keys_not_in_default_config():
    for k in S18_SWITCHES + S18_PARAMS:
        assert k not in NF.DEFAULT_CONFIG and k not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(CMOD.DEFAULT_CONFIG.items())


def test_params_default_when_on():
    g = NF.Graph(dict(ON))
    assert g._stab_receiver() == (0.3, 1.0, 0.002, 0.1, 0.01, 0.05, 0.05)
    assert NF.Graph()._stab_receiver() is None


@pytest.mark.parametrize("bad", [dict(prediction_receiver_rho0=1.5), dict(prediction_receiver_rho_min=0.5),
                                 dict(prediction_receiver_k="many"), dict(prediction_receiver_eta=0)])
def test_bad_params_error_once_and_change_nothing(bad, caplog):
    start = H2.built(0)
    t = S.Pair(restore_as(CMOD, "native", start), restore_as(NF, "native", start), 0)
    t.op("keys", lambda g: g.config.update(dict(ARM, **RPE)))
    t.b.g.config.update(dict(ON, **bad))
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 10, sleep_every=True)
    errs = [r for r in caplog.records if r.levelno == logging.ERROR and "prediction_receiver_enabled" in r.getMessage()]
    assert len(errs) == 1, [r.getMessage() for r in errs]
    for k in list(ON) + list(bad):
        t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)
