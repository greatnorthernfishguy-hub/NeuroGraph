# ---- Changelog ----
# [2026-10-04] Claude (lane prune-lifeline) — NEW tests: normal pruning of FAINT links touching protected nodes, lifelines kept
# What: (G) flag OFF (absent, or explicit False) is byte-identical to the base checkout — step() driver AND engine driver,
#   outputs + checkpoint bytes, separate processes; flag ON changes the run (non-vacuity). (L) flag ON: faint links touching
#   authored wants AND the constitutional node are pruned by all three rules, each protected node's strongest in/out link is
#   kept (either endpoint's lifeline is exempt; tie-break weight desc then synapse_id asc), a raising protection probe = protected
#   (fail closed), the lifeline query is pure and is the strength budget's guarded pick; flag OFF keeps the #92 blanket skip.
#   (E) interaction with compete_protected_links: lifelines never compete when the flag is on (incl. a tie the engine's own
#   rank would break differently), the partner last-link rule and floors still hold, constitutional synapses stay frozen in
#   the engine pass (they are removed by the WAKE prune instead).
# Why:  Josh ruling 2026-10-04 ("yes": protect existence, not unlimited wiring); spec 2026-10-04-synapse-growth-by-competition-design.
# How:  synthetic graphs only. Base checkout for (G): PRUNE_LIFELINE_BASE_CHECKOUT (LAW 5 env override; default a clean
#   worktree of origin/cc-laptop-trial-s4-20261002 @ 39c0422).
# -------------------
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(REPO / "tests"))

import neuro_foundation as nf  # noqa: E402

if Path(nf.__file__).resolve().parent != REPO:
    raise RuntimeError("P379: neuro_foundation resolved to %s, not this worktree (%s); refusing to run" % (nf.__file__, REPO))

BASE_CHECKOUT = Path(os.environ.get("PRUNE_LIFELINE_BASE_CHECKOUT", "/home/josh/worktrees/ng-prune-lifeline-base-20261004"))
FLAG = "prune_protected_faint_links"
CC = "constitutional::rim::choice_clause"
GRACE = nf.DEFAULT_CONFIG["grace_period"]
WT = nf.DEFAULT_CONFIG["weight_threshold"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _syn(g, a, b, w, *, lws=0, inactive=0, peak=None, age=0):
    s = g.create_synapse(a, b, w)
    s.weight = w
    s.low_weight_steps = lws
    s.inactive_steps = inactive
    s.peak_weight = 1.0 if peak is None else peak     # default: strengthened once, so the AGE rule is not what prunes
    s.creation_time = float(g.timestep - age)
    return s.synapse_id


def _hub(flag):
    """A constitutional broadcaster and an authored want, each with one strong link per direction and faint siblings
    already at the weight rule's dwell boundary (one more call below threshold => eligible)."""
    g = nf.Graph({FLAG: True} if flag else {})
    g.timestep = 100
    g.create_node(node_id=CC, metadata={"constitutional": True})
    g.create_node(node_id="want", metadata={"provenance": "cc_authored"})
    for i in range(8):
        g.create_node(node_id="p%d" % i)
    ids = {
        "cc_out_strong": _syn(g, CC, "p0", 2.0),
        "cc_in_faint_best": _syn(g, "p0", CC, 0.007, lws=GRACE),   # CC's strongest IN link is itself faint
        "want_out_strong": _syn(g, "want", "p1", 1.5),
        "want_in_strong": _syn(g, "p1", "want", 0.8),
    }
    for i in range(2, 8):
        ids["cc_out_faint%d" % i] = _syn(g, CC, "p%d" % i, 0.001, lws=GRACE)
        ids["cc_in_faint%d" % i] = _syn(g, "p%d" % i, CC, 0.002, lws=GRACE)
        ids["want_out_faint%d" % i] = _syn(g, "want", "p%d" % i, 0.003, lws=GRACE)
        ids["want_in_faint%d" % i] = _syn(g, "p%d" % i, "want", 0.004, lws=GRACE)
    return g, ids


def _degrees(g, nid):
    return len(g._outgoing.get(nid) or ()), len(g._incoming.get(nid) or ())


# ---------------------------------------------------------------------------
# (G) flag OFF is byte-identical to the base; ON changes the run
# ---------------------------------------------------------------------------
def _run_driver(script, checkout, tmp_path, tag, extra_args):
    ckpt = tmp_path / ("%s.msgpack" % tag)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "NG_EMBED_REMOTE")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    args = [sys.executable, str(REPO / "tests" / script), "--checkout", str(checkout), "--ckpt", str(ckpt)] + extra_args
    p = subprocess.run(args, capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=900)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    assert lines, "driver produced no JSON (rc=%s)\n%s\n%s" % (p.returncode, p.stdout[-2000:], p.stderr[-2000:])
    rec = json.loads(lines[-1])
    assert not rec["void"], rec.get("reason")
    assert p.returncode == 0, p.stderr[-2000:]
    return rec


def _step_drive(checkout, tmp_path, tag, cfg=None):
    extra = ["--steps", "80"] + (["--config-json", json.dumps(cfg)] if cfg is not None else [])
    return _run_driver("strength_budget_golden_driver.py", checkout, tmp_path, tag, extra)


def _engine_drive(checkout, tmp_path, tag, cfg=None):
    extra = ["--config-json", json.dumps(cfg)] if cfg is not None else []
    return _run_driver("prune_lifeline_engine_driver.py", checkout, tmp_path, tag, extra)


@pytest.fixture(scope="module")
def base_checkout():
    assert (BASE_CHECKOUT / "neuro_foundation.py").is_file(), "base checkout missing: %s" % BASE_CHECKOUT
    assert BASE_CHECKOUT.resolve() != REPO
    assert "_protected_lifelines" not in (BASE_CHECKOUT / "neuro_foundation.py").read_text(), "base already has the change"
    return BASE_CHECKOUT


_STEP_KEYS = ("per_step", "synapses_after", "weights_sha", "state_digest", "checkpoint_sha256")
_ENGINE_KEYS = ("engine", "engine_removed", "default_pruned", "synapses_after", "nodes_after", "state_digest",
                "checkpoint_sha256")


@pytest.mark.parametrize("cfg", [None, {FLAG: False}], ids=["absent", "explicit_false"])
def test_G_off_step_driver_is_byte_identical_to_the_base(base_checkout, tmp_path, cfg):
    base = _step_drive(base_checkout, tmp_path, "base", cfg)
    branch = _step_drive(REPO, tmp_path, "branch", cfg)
    assert sum(r[2] for r in base["per_step"]) > 0, "vacuous: the base pruned nothing in 80 steps"
    for k in _STEP_KEYS:
        assert base[k] == branch[k], "flag off differs from base on %s" % k


@pytest.mark.parametrize("cfg", [None, {FLAG: False}], ids=["absent", "explicit_false"])
def test_G_off_engine_driver_is_byte_identical_to_the_base(base_checkout, tmp_path, cfg):
    base = _engine_drive(base_checkout, tmp_path, "base", cfg)
    branch = _engine_drive(REPO, tmp_path, "branch", cfg)
    assert base["engine"]["removed"] > 0 and base["default_pruned"] > 0, "vacuous engine/prune run"
    for k in _ENGINE_KEYS:
        assert base[k] == branch[k], "flag off differs from base on %s" % k


def test_G_control_flag_on_changes_both_runs(tmp_path):
    off = _step_drive(REPO, tmp_path, "s_off")
    on = _step_drive(REPO, tmp_path, "s_on", {FLAG: True})
    assert on["state_digest"] != off["state_digest"]
    assert sum(r[2] for r in on["per_step"]) > sum(r[2] for r in off["per_step"])
    eoff = _engine_drive(REPO, tmp_path, "e_off")
    eon = _engine_drive(REPO, tmp_path, "e_on", {FLAG: True})
    assert eon["default_pruned"] > eoff["default_pruned"]


def test_G_flag_is_not_in_default_config_and_not_saved_unless_set():
    assert FLAG not in nf.DEFAULT_CONFIG
    g = nf.Graph()
    assert FLAG not in g.config


# ---------------------------------------------------------------------------
# (L) the lifeline rule
# ---------------------------------------------------------------------------
def test_L_flag_off_keeps_the_92_blanket_skip():
    g, ids = _hub(flag=False)
    before = set(g.synapses.keys())
    assert g._prune_synapses() == 0
    assert set(g.synapses.keys()) == before
    # the blanket skip does not even advance the dwell counter
    assert g.synapses[ids["cc_out_faint2"]].low_weight_steps == GRACE


def test_L_flag_on_prunes_faint_links_and_keeps_every_lifeline_constitutional_included():
    g, ids = _hub(flag=True)
    life = g._protected_lifelines()
    assert life == {ids["cc_out_strong"], ids["cc_in_faint_best"], ids["want_out_strong"], ids["want_in_strong"]}
    n = g._prune_synapses()
    left = set(g.synapses.keys())
    assert n == 24 and left == life
    # the constitutional node's faint strongest-IN link survives because it is its lifeline, not because it is strong
    assert g.synapses[ids["cc_in_faint_best"]].weight < WT
    for nid in (CC, "want"):
        assert _degrees(g, nid) == (1, 1)
    g._collect_orphan_nodes()
    assert CC in g.nodes and "want" in g.nodes


def test_L_activity_and_age_rules_also_apply_and_a_lifeline_is_spared_by_both():
    g = nf.Graph({FLAG: True})
    g.timestep = 20_000
    g.create_node(node_id="want", metadata={"provenance": "syl_authored"})
    for i in range(4):
        g.create_node(node_id="q%d" % i)
    life = _syn(g, "want", "q0", 0.9, inactive=10 ** 6, peak=0.05, age=10 ** 4)    # strongest, but stale AND unstrengthened
    stale = _syn(g, "want", "q1", 0.5, inactive=10 ** 6)                            # activity rule
    young = _syn(g, "want", "q2", 0.5, peak=0.05, age=GRACE + 1)                    # age rule
    kept = _syn(g, "want", "q3", 0.5)                                               # healthy: no rule fires
    assert g._prune_synapses() == 2
    assert set(g.synapses.keys()) == {life, kept}
    assert stale not in g.synapses and young not in g.synapses


def test_L_a_lifeline_of_either_endpoint_is_exempt():
    g = nf.Graph({FLAG: True})
    g.create_node(node_id="a", metadata={"provenance": "cc_authored"})
    g.create_node(node_id="b", metadata={"constitutional": True})
    g.create_node(node_id="x")
    ab = _syn(g, "a", "b", 0.002, lws=GRACE)       # a's ONLY out link (its lifeline); NOT b's strongest in
    _syn(g, "x", "b", 0.5)                          # b's strongest in
    xa = _syn(g, "x", "a", 0.003, lws=GRACE)       # a's only in link (lifeline)
    bx = _syn(g, "b", "x", 0.004, lws=GRACE)       # b's only out link (lifeline)
    assert g._prune_synapses() == 0
    assert {ab, xa, bx} <= set(g.synapses.keys())


def test_L_tie_break_is_weight_desc_then_synapse_id_asc():
    g = nf.Graph({FLAG: True})
    g.create_node(node_id="w", metadata={"provenance": "cc_authored"})
    for i in range(5):
        g.create_node(node_id="t%d" % i)
    sids = [_syn(g, "w", "t%d" % i, 0.004, lws=GRACE) for i in range(5)]
    keep = min(sids)
    assert g._protected_lifelines() == {keep}
    assert g._prune_synapses() == 4
    assert set(g.synapses.keys()) == {keep}


def test_L_lifelines_are_the_strength_budget_guarded_pick_and_the_query_is_pure():
    g, _ = _hub(flag=True)
    snap = {sid: (s.weight, s.low_weight_steps, s.inactive_steps) for sid, s in g.synapses.items()}
    life = g._protected_lifelines()
    guard = nf._strength_guard_targets(lambda s: g.synapses[s].weight, g._outgoing, g._incoming,
                                       g._strength_protected_ids(), 0.0)
    assert life == set(guard)
    assert {sid: (s.weight, s.low_weight_steps, s.inactive_steps) for sid, s in g.synapses.items()} == snap


def test_L_raising_probe_fails_closed_node_is_treated_as_protected(monkeypatch):
    g = nf.Graph({FLAG: True})
    for n in ("odd", "y", "z"):
        g.create_node(node_id=n)
    only_out = _syn(g, "odd", "y", 0.002, lws=GRACE)     # unprotected node: would be pruned if the probe said False
    other = _syn(g, "y", "z", 0.002, lws=GRACE)
    real = nf.Graph._is_identity_protected

    def probe(self, nid):
        if nid == "odd":
            raise RuntimeError("probe broke")
        return real(self, nid)

    monkeypatch.setattr(nf.Graph, "_is_identity_protected", probe)
    assert only_out in g._protected_lifelines()
    g._prune_synapses()
    assert only_out in g.synapses and other not in g.synapses


def test_L_competing_mode_is_untouched_by_the_flag():
    """The flag does not change what a competing-mode call visits or removes (the caller owns the sets)."""
    res = []
    for flag in (False, True):
        g, ids = _hub(flag=flag)
        comp = [ids["want_out_faint%d" % i] for i in range(2, 8)] + [ids["want_out_strong"]]
        excl = [s for s in g.synapses.keys() if s not in comp]
        key = {s: (0, s) for s in comp}
        rep = {}
        g._prune_synapses(competing_ids=comp, excluded_ids=excl, max_removals=100, order_key=key, report=rep)
        role = {v: k for k, v in ids.items()}
        res.append((rep["eligible"], sorted(role[s] for s in rep["removed_ids"])))
    assert res[0] == res[1]


# ---------------------------------------------------------------------------
# (E) interaction with compete_protected_links
# ---------------------------------------------------------------------------
def _tie_graph(flag):
    """A want whose 3 strongest out-links TIE on weight; the engine's own rank (weight, -peak, inactive, id) picks the
    one with the highest peak, the lifeline rule picks the smallest id. All are activity-eligible."""
    g = nf.Graph({FLAG: True} if flag else {})
    g.create_node(node_id="w", metadata={"provenance": "cc_authored"})
    for i in range(3):
        g.create_node(node_id="t%d" % i)
    for i in range(3):
        _syn(g, "t%d" % i, "t%d" % ((i + 1) % 3), 0.5)       # partners have other links: no last-link holdback
    sids = sorted(_syn(g, "w", "t%d" % i, 0.5, inactive=10 ** 6) for i in range(3))
    g.synapses[sids[0]].peak_weight = 0.6
    g.synapses[sids[1]].peak_weight = 0.9                     # the engine's top-1
    g.synapses[sids[2]].peak_weight = 0.7
    return g, sids


def test_E_engine_never_competes_a_lifeline_when_the_flag_is_on():
    g, sids = _tie_graph(flag=False)
    rec = g.compete_protected_links(1, 10)
    assert sids[0] not in g.synapses and sids[1] in g.synapses and rec["removed"] == 2   # today's engine
    g, sids = _tie_graph(flag=True)
    life = g._protected_lifelines()
    assert sids[0] in life
    rec = g.compete_protected_links(1, 10)
    assert {sids[0], sids[1]} <= set(g.synapses.keys()) and rec["removed"] == 1 and rec["floors_ok"]


def _engine_graph(flag):
    import want_hub_golden_driver as drv
    drv.install_deterministic_uuid()
    g, roles = drv.build_graph(nf, seed=31, extra_config={FLAG: True} if flag else {}, n_leaf=12)
    return g, roles


def test_E_engine_last_link_floors_and_constitutional_freeze_hold_with_the_flag_on():
    g, roles = _engine_graph(flag=True)
    life = g._protected_lifelines()
    rim = roles["rim"]
    F = set(g._outgoing.get(rim, ())) | set(g._incoming.get(rim, ()))
    deg_before = {n: _degrees(g, n) for n in g.nodes}
    before = set(g.synapses.keys())
    rec = g.compete_protected_links(2, 10 ** 6)
    removed = before - set(g.synapses.keys())
    assert rec["removed"] > 0 and rec["floors_ok"]
    assert not (removed & life), "the engine removed a lifeline"
    assert not (removed & F), "the engine touched a constitutional synapse"
    for n, (o, i) in deg_before.items():
        if o + i and not g._is_identity_protected(n):
            assert sum(_degrees(g, n)) >= 1, "partner %s lost its last link in the engine pass" % n
    # the WAKE prune (flag on) is what removes faint constitutional non-lifelines — ruling (a)
    rim_life = {s for s in life if s in F}
    g._prune_synapses()
    left_F = F & set(g.synapses.keys())
    assert rim_life <= left_F
    assert len(left_F) < len(F)


def test_E_after_engine_and_wake_prune_every_protected_node_keeps_its_lifelines():
    g, _ = _engine_graph(flag=True)
    prot = g._strength_protected_ids()
    had = {(n, d) for n in prot for d, idx in (("out", g._outgoing), ("in", g._incoming)) if idx.get(n)}
    for _ in range(3):
        g.compete_protected_links(2, 10 ** 6)
        g._prune_synapses()
        g._collect_orphan_nodes()
    for n, d in had:
        idx = g._outgoing if d == "out" else g._incoming
        assert n in g.nodes and idx.get(n), "%s lost every %s link" % (n, d)
