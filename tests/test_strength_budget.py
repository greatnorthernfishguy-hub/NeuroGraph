# ---- Changelog ----
# [2026-10-04] Claude (lane strength-budget) — NEW tests: per-node strength budget + sleep downscaling
# What: (G) OFF is byte-identical to the base checkout (outputs + checkpoint bytes, separate processes); (R) the rule:
#   off = no-op, normalization math, OUT-then-IN order, interval + counter persistence, invalid setting is loud and harmless;
#   (P) the strongest-link guarantee for constitutional AND authored-want nodes (Josh ruling (a)) in both directions, the
#   fail-closed protection probe, never raised above its own pre-pass weight; (S) sleep_downscale counts / ratios /
#   guarantee / no prune / validation; (E) native SynapseStore.normalize_strength / scale_all == the Python fallback
#   on random graphs (rel tol 1e-12; observed exact).
# Why:  spec ~/docs/superpowers/specs/2026-10-04-synapse-growth-by-competition-design.md (pieces 1 and 2).
# How:  synthetic graphs only — no real checkpoint, vectors, tract, embed or TID. The base checkout for (G) is
#   STRENGTH_BUDGET_BASE_CHECKOUT (LAW 5 env override; default a clean worktree of origin/cc-laptop-trial-s4-20261002).
#   (E) needs an ng_tract wheel with the native methods; with an older wheel it SKIPS and says so (the fallback is then
#   the only path and is covered by every other test here).
# -------------------
import json
import logging
import math
import os
import random
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import neuro_foundation as nf  # noqa: E402

if Path(nf.__file__).resolve().parent != REPO:
    raise RuntimeError("P379: neuro_foundation resolved to %s, not this worktree (%s); refusing to run" % (nf.__file__, REPO))

BASE_CHECKOUT = Path(os.environ.get("STRENGTH_BUDGET_BASE_CHECKOUT", "/home/josh/worktrees/ng-s4-base-20261004"))
FLOOR = 2 * nf.DEFAULT_CONFIG["weight_threshold"]
CC = "constitutional::rim::choice_clause"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _graph(cfg=None):
    return nf.Graph(cfg or {})


def _w(g):
    return {sid: s.weight for sid, s in g.synapses.items()}


def _syn(g, a, b, w):
    return g.create_synapse(a, b, w).synapse_id


def _hub_graph():
    """A Choice-Clause-shaped constitutional broadcaster, an authored want, and plain nodes."""
    g = _graph()
    g.create_node(node_id=CC, metadata={"constitutional": True})
    g.create_node(node_id="want", metadata={"provenance": "cc_authored"})
    for i in range(30):
        g.create_node(node_id="p%02d" % i)
    ids = {}
    for i in range(30):
        ids[("cc", i)] = _syn(g, CC, "p%02d" % i, 2.0 + 0.1 * i)       # out-strength 30*2 + 0.1*435 = 103.5
        ids[("want", i)] = _syn(g, "p%02d" % i, "want", 1.0 + 0.01 * i)  # in-strength ~34.35
    return g, ids


def _random_graph(seed, n_nodes=60, n_syn=900, protected_every=9):
    """Seeded graph with SEEDED uuid4-shaped synapse ids (not ascending in creation order), so two calls with the same
    seed build identical graphs."""
    real_uuid4 = uuid.uuid4
    id_rng = random.Random(seed ^ 0x5EED)
    uuid.uuid4 = lambda: uuid.UUID(int=id_rng.getrandbits(128), version=4)
    try:
        return _random_graph_inner(seed, n_nodes, n_syn, protected_every)
    finally:
        uuid.uuid4 = real_uuid4


def _random_graph_inner(seed, n_nodes, n_syn, protected_every):
    rng = random.Random(seed)
    g = _graph()
    nodes = []
    for i in range(n_nodes):
        nid = "n%03d" % i
        meta = {}
        if i % protected_every == 0:
            meta = {"constitutional": True} if i % 2 == 0 else {"provenance": "syl_authored"}
        g.create_node(node_id=nid, metadata=meta)
        nodes.append(nid)
    used = set()
    menu = [0.0, 0.005, 0.01, 0.02, 0.5, 1.0, 1.0, 2.5, 4.9]
    while len(used) < n_syn:
        a, b = rng.sample(nodes, 2)
        if (a, b) in used:
            continue
        used.add((a, b))
        w = rng.choice(menu) if rng.random() < 0.5 else rng.uniform(0.0, 5.0)
        g.create_synapse(a, b, w)
    return g


def _strongest(g, nid, direction, weights):
    idx = g._outgoing if direction == "out" else g._incoming
    ids = idx.get(nid) or ()
    return min(ids, key=lambda s: (-weights[s], s)) if ids else None


# ---------------------------------------------------------------------------
# (G) OFF is byte-identical to the base
# ---------------------------------------------------------------------------
def _drive(checkout, tmp_path, tag, config_json=""):
    ckpt = tmp_path / ("%s.msgpack" % tag)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "NG_EMBED_REMOTE")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    args = [sys.executable, str(REPO / "tests" / "strength_budget_golden_driver.py"), "--checkout", str(checkout),
            "--ckpt", str(ckpt), "--steps", "80"]
    if config_json:
        args += ["--config-json", config_json]
    p = subprocess.run(args, capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=600)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    assert lines, "driver produced no JSON (rc=%s)\n%s\n%s" % (p.returncode, p.stdout[-2000:], p.stderr[-2000:])
    rec = json.loads(lines[-1])
    assert not rec["void"], rec.get("reason")
    assert p.returncode == 0, p.stderr[-2000:]
    return rec


@pytest.fixture(scope="module")
def base_checkout():
    assert (BASE_CHECKOUT / "neuro_foundation.py").is_file(), "base checkout missing: %s" % BASE_CHECKOUT
    assert BASE_CHECKOUT.resolve() != REPO
    return BASE_CHECKOUT


_G_KEYS = ("per_step", "synapses_after", "weights_sha", "state_digest", "checkpoint_sha256")


def test_G_off_by_default_is_byte_identical_to_the_base(base_checkout, tmp_path):
    base = _drive(base_checkout, tmp_path, "base")
    branch = _drive(REPO, tmp_path, "branch")
    assert "StrengthBudgetRule" not in base["rules"] and "StrengthBudgetRule" in branch["rules"]
    assert sum(len(s[1]) for s in base["per_step"]) > 0, "vacuous: nothing fired, no rule.apply was reached"
    for k in _G_KEYS:
        assert base[k] == branch[k], "OFF differs from base on %s" % k


def test_G_explicit_off_settings_change_nothing_but_the_saved_config(base_checkout, tmp_path):
    off = json.dumps({"strength_budget_enabled": False, "strength_budget_out": 1.0, "strength_budget_in": 1.0,
                      "strength_budget_interval": 1})
    base = _drive(base_checkout, tmp_path, "base_off", off)
    branch = _drive(REPO, tmp_path, "branch_off", off)
    for k in _G_KEYS:
        assert base[k] == branch[k], "explicit OFF differs from base on %s" % k


def test_G_control_enabled_does_change_the_run(tmp_path):
    """Non-vacuity: the same driver with the budget ON produces a different run."""
    off = _drive(REPO, tmp_path, "c_off")
    on = _drive(REPO, tmp_path, "c_on", json.dumps({"strength_budget_enabled": True, "strength_budget_out": 1.0,
                                                    "strength_budget_in": 1.0, "strength_budget_interval": 5}))
    assert on["weights_sha"] != off["weights_sha"]


# ---------------------------------------------------------------------------
# (R) the rule
# ---------------------------------------------------------------------------
def test_R_rule_registered_after_homeostatic_in_init_and_restore(tmp_path):
    g, _ = _hub_graph()
    names = [type(r).__name__ for r in g._plasticity_rules]
    assert names.index("StrengthBudgetRule") == names.index("HomeostaticRule") + 1
    p = tmp_path / "r.msgpack"
    g.checkpoint(str(p))
    h = _graph()
    h.restore(str(p))
    assert [type(r).__name__ for r in h._plasticity_rules] == names


def test_R_off_is_a_no_op_even_with_budgets_set():
    g, _ = _hub_graph()
    g.config.update(strength_budget_out=1.0, strength_budget_in=1.0, strength_budget_interval=1)
    rule = next(r for r in g._plasticity_rules if isinstance(r, nf.StrengthBudgetRule))
    before = _w(g)
    for t in range(10):
        rule.apply(g, [CC], t)
    assert _w(g) == before and rule._steps_since_budget == 0 and rule.last_result is None


def test_R_normalization_math_out_then_in():
    g = _graph()
    for n in ("a", "b", "x"):
        g.create_node(node_id=n)
    s1 = _syn(g, "a", "x", 4.0)
    s2 = _syn(g, "b", "x", 4.0)
    s3 = _syn(g, "a", "b", 4.0)
    # OUT: a has 8.0 > 4 -> s1, s3 = 2.0; b has 4.0 (not > 4) untouched. IN on new weights: x has 2+4=6 > 3 -> *0.5
    res = g.apply_strength_budget(4.0, 3.0)
    w = _w(g)
    assert w[s1] == 2.0 * 3.0 / 6.0 and w[s2] == 4.0 * 3.0 / 6.0 and w[s3] == 2.0
    assert (res["nodes_scaled_out"], res["nodes_scaled_in"], res["synapses_scaled"], res["clamped"]) == (1, 1, 3, 0)


def test_R_sum_equal_to_budget_is_untouched_and_none_skips_a_direction():
    g = _graph()
    for n in ("a", "b", "c"):
        g.create_node(node_id=n)
    s1 = _syn(g, "a", "b", 1.5)
    s2 = _syn(g, "a", "c", 1.5)
    res = g.apply_strength_budget(3.0, None)
    assert res["synapses_scaled"] == 0 and _w(g) == {s1: 1.5, s2: 1.5}
    res = g.apply_strength_budget(None, 1.0)      # b and c each have 1.5 incoming
    assert res["nodes_scaled_in"] == 2 and res["nodes_scaled_out"] == 0 and _w(g) == {s1: 1.0, s2: 1.0}


def test_R_relative_strengths_preserved_within_a_scaled_node():
    g, ids = _hub_graph()
    before = _w(g)
    g.apply_strength_budget(10.0, None)
    after = _w(g)
    out = [ids[("cc", i)] for i in range(30)]
    total = sum(after[s] for s in out)
    assert math.isclose(total, 10.0, rel_tol=1e-12)
    ratios = {after[s] / before[s] for s in out}
    assert max(ratios) - min(ratios) < 1e-15


def test_R_interval_and_counter_persistence(tmp_path):
    g, _ = _hub_graph()
    g.config.update(strength_budget_enabled=True, strength_budget_out=10.0, strength_budget_interval=3)
    rule = next(r for r in g._plasticity_rules if isinstance(r, nf.StrengthBudgetRule))
    before = _w(g)
    rule.apply(g, [CC], 1)
    rule.apply(g, [CC], 2)
    assert _w(g) == before and rule._steps_since_budget == 2
    # counter persists across checkpoint/restore only while non-zero
    p = tmp_path / "c.msgpack"
    g.checkpoint(str(p))
    h = _graph()
    h.restore(str(p))
    hr = next(r for r in h._plasticity_rules if isinstance(r, nf.StrengthBudgetRule))
    assert hr._steps_since_budget == 2
    rule.apply(g, [CC], 3)
    assert _w(g) != before and rule._steps_since_budget == 0 and rule.last_result["nodes_scaled_out"] == 1
    assert "strength_budget_steps_since" not in g._serialize_full()


def test_R_interval_defaults_to_scaling_interval():
    g, _ = _hub_graph()
    g.config.update(strength_budget_enabled=True, strength_budget_out=10.0, scaling_interval=4)
    rule = next(r for r in g._plasticity_rules if isinstance(r, nf.StrengthBudgetRule))
    before = _w(g)
    for t in range(3):
        rule.apply(g, [CC], t)
    assert _w(g) == before
    rule.apply(g, [CC], 3)
    assert _w(g) != before


def test_R_runs_inside_step_when_enabled():
    g, ids = _hub_graph()
    g.config.update(strength_budget_enabled=True, strength_budget_out=10.0, strength_budget_interval=1)
    g.stimulate(CC, 5.0)
    r = g.step()
    assert r.fired_node_ids, "nothing fired: the rule would not be reached"
    out = sum(g.synapses[ids[("cc", i)]].weight for i in range(30))
    assert out <= 10.0 + 1e-9


def test_R_invalid_setting_is_loud_and_touches_nothing(caplog):
    g, _ = _hub_graph()
    g.config.update(strength_budget_enabled=True, strength_budget_out=-1.0, strength_budget_interval=1)
    rule = next(r for r in g._plasticity_rules if isinstance(r, nf.StrengthBudgetRule))
    before = _w(g)
    with caplog.at_level(logging.WARNING):
        rule.apply(g, [CC], 1)
    assert _w(g) == before and "invalid setting" in caplog.text
    caplog.clear()
    for bad_interval in (0, -3, "5", 2.5, True):
        g.config.update(strength_budget_out=1.0, strength_budget_interval=bad_interval)
        with caplog.at_level(logging.WARNING):
            rule.apply(g, [CC], 1)
        assert _w(g) == before and "invalid strength_budget_interval" in caplog.text
    for bad in (0, float("inf"), float("nan"), True, "5"):
        with pytest.raises(ValueError):
            g.apply_strength_budget(bad, None)


def test_R_never_prunes_by_itself():
    g, _ = _hub_graph()
    n = len(g.synapses)
    g.apply_strength_budget(0.001, 0.001)
    assert len(g.synapses) == n


# ---------------------------------------------------------------------------
# (P) strongest-link guarantee — constitutional INCLUDED (ruling (a)) — and fail-closed probe
# ---------------------------------------------------------------------------
def test_P_constitutional_node_is_budgeted_and_keeps_its_strongest_out_link():
    g, ids = _hub_graph()
    before = _w(g)
    strongest = _strongest(g, CC, "out", before)
    assert strongest == ids[("cc", 29)]
    res = g.apply_strength_budget(0.05, None)     # 0.05 over 30 links: every link far below the floor
    after = _w(g)
    assert sum(after[ids[("cc", i)]] for i in range(29)) < 0.05      # the broadcast IS bounded (ruling (a))
    assert after[strongest] == FLOOR and res["clamped"] >= 1
    assert all(after[ids[("cc", i)]] < FLOOR for i in range(29))


def test_P_want_keeps_its_strongest_in_link():
    g, ids = _hub_graph()
    before = _w(g)
    strongest = _strongest(g, "want", "in", before)
    g.apply_strength_budget(None, 0.05)
    after = _w(g)
    assert after[strongest] == FLOOR
    assert sum(1 for i in range(30) if after[ids[("want", i)]] < FLOOR) == 29


def test_P_guarantee_holds_when_the_other_endpoint_scales_the_link():
    """A protected node's strongest IN link is cut by its (unprotected) PRE node's OUT budget — still guarded."""
    g = _graph()
    g.create_node(node_id="pre")
    g.create_node(node_id="w", metadata={"provenance": "syl_authored"})
    for i in range(20):
        g.create_node(node_id="q%d" % i)
    link = _syn(g, "pre", "w", 1.0)
    for i in range(20):
        _syn(g, "pre", "q%d" % i, 5.0)
    g.apply_strength_budget(0.1, None)
    assert g.synapses[link].weight == FLOOR


def test_P_guarantee_never_raises_a_link_above_its_own_pre_pass_weight():
    g = _graph()
    g.create_node(node_id=CC, metadata={"constitutional": True})
    g.create_node(node_id="a")
    s = _syn(g, CC, "a", 0.005)       # already under the floor before the pass
    g.sleep_downscale(0.5)
    assert g.synapses[s].weight == 0.005


def test_P_unprotected_nodes_get_no_guarantee():
    g = _graph()
    for n in ("a", "b", "c"):
        g.create_node(node_id=n)
    s1 = _syn(g, "a", "b", 1.0)
    _syn(g, "a", "c", 1.0)
    g.apply_strength_budget(0.01, None)
    assert g.synapses[s1].weight == 0.005


def test_P_protection_probe_failure_is_treated_as_protected(monkeypatch):
    g = _graph()
    for n in ("odd", "b", "c"):
        g.create_node(node_id=n)
    s1 = _syn(g, "odd", "b", 2.0)
    _syn(g, "odd", "c", 1.0)
    real = g._is_identity_protected

    def probe(nid):
        if nid == "odd":
            raise RuntimeError("probe broke")
        return real(nid)

    monkeypatch.setattr(g, "_is_identity_protected", probe)
    assert "odd" in g._strength_protected_ids()
    g.apply_strength_budget(0.001, None)
    assert g.synapses[s1].weight == FLOOR


def test_P_every_protected_node_keeps_its_strongest_links_on_random_graphs():
    for seed in range(5):
        g = _random_graph(seed)
        w0 = _w(g)
        prot = g._strength_protected_ids()
        assert any((g.nodes[p].metadata or {}).get("constitutional") for p in prot)
        need = {}
        for p in prot:
            for d in ("out", "in"):
                s = _strongest(g, p, d, w0)
                if s:
                    need[(p, d)] = min(w0[s], FLOOR)
        assert need
        # Each pass guards the link that is strongest BEFORE that pass, so across passes the invariant is: every
        # protected node's strongest link, each direction, stays >= min(its original strongest weight, floor).
        for step in (lambda: g.apply_strength_budget(0.5, 0.5), lambda: g.sleep_downscale(0.02),
                     lambda: g.apply_strength_budget(0.01, 0.01), lambda: g.sleep_downscale(0.5)):
            step()
            w = _w(g)
            for (p, d), t in need.items():
                assert w[_strongest(g, p, d, w)] >= t, (seed, p, d)


# ---------------------------------------------------------------------------
# (S) sleep downscaling
# ---------------------------------------------------------------------------
def test_S_scales_every_weight_and_preserves_ratios_without_pruning():
    g = _random_graph(3)
    w0 = _w(g)
    n = len(g.synapses)
    res = g.sleep_downscale(0.98)
    w = _w(g)
    assert len(g.synapses) == n
    assert res["synapses_scaled"] == sum(1 for v in w0.values() if v * 0.98 != v)
    for sid, v in w0.items():
        assert w[sid] == v * 0.98 or w[sid] == min(v, FLOOR)


def test_S_factor_one_is_a_no_op_and_bad_factors_refuse():
    g = _random_graph(4)
    w0 = _w(g)
    assert g.sleep_downscale(1.0)["synapses_scaled"] == 0 and _w(g) == w0
    for bad in (0, 0.0, -0.5, 1.01, float("nan"), float("inf"), True, "0.9"):
        with pytest.raises(ValueError):
            g.sleep_downscale(bad)
    assert _w(g) == w0


def test_S_is_not_called_by_step():
    g, _ = _hub_graph()
    calls = []
    g.sleep_downscale = lambda *a, **k: calls.append(1)
    g.config.update(strength_budget_enabled=True, strength_budget_out=10.0, strength_budget_interval=1)
    for _ in range(5):
        g.stimulate(CC, 5.0)
        g.step()
    assert calls == []


# ---------------------------------------------------------------------------
# (E) native == Python fallback (rel tol 1e-12 stated; observed exact)
# ---------------------------------------------------------------------------
_NATIVE = hasattr(nf.ng_tract.SynapseStore, "normalize_strength") and hasattr(nf.ng_tract.SynapseStore, "scale_all")
REL_TOL = 1e-12


def _close(a, b):
    return a == b or math.isclose(a, b, rel_tol=REL_TOL, abs_tol=1e-300)


@pytest.mark.skipif(not _NATIVE, reason="installed ng_tract (%s) has no normalize_strength/scale_all — fallback only"
                    % getattr(nf.ng_tract, "__file__", "?"))
@pytest.mark.parametrize("seed", range(8))
def test_E_native_equals_fallback_on_random_graphs(seed):
    budgets = [(0.5, 0.5), (3.0, None), (None, 2.0), (7.5, 5.4), (100.0, 100.0)][seed % 5]
    factor = [0.98, 0.5, 0.02, 1.0][seed % 4]
    a, b = _random_graph(100 + seed), _random_graph(100 + seed)
    assert _w(a) == _w(b)
    prot = a._strength_protected_ids()
    floor = 2.0 * a.config["weight_threshold"]
    ra = dict(a.synapses.normalize_strength(budgets[0], budgets[1], prot, floor))
    rb = nf._strength_budget_python(b.synapses, b._outgoing, b._incoming, budgets[0], budgets[1], prot, floor)
    assert ra == rb
    wa, wb = _w(a), _w(b)
    assert wa.keys() == wb.keys() and all(_close(wa[s], wb[s]) for s in wa)
    sa = dict(a.synapses.scale_all(factor, prot, floor))
    sb = nf._scale_all_python(b.synapses, b._outgoing, b._incoming, factor, prot, floor)
    assert sa == sb
    wa, wb = _w(a), _w(b)
    assert all(_close(wa[s], wb[s]) for s in wa)
    # observed: bit-identical (same summation order, same arithmetic) — recorded, the tolerance is the contract
    print("seed=%d max_abs_diff=%r" % (seed, max(abs(wa[s] - wb[s]) for s in wa)))


@pytest.mark.skipif(not _NATIVE, reason="native methods absent")
def test_E_graph_methods_use_native_when_present():
    g, _ = _hub_graph()
    assert g.apply_strength_budget(10.0, None)["native"] is True
    assert g.sleep_downscale(0.98)["native"] is True


@pytest.mark.skipif(not _NATIVE, reason="native methods absent")
def test_E_native_refuses_bad_arguments():
    s = nf.ng_tract.SynapseStore()
    for args in ((0.0, None, [], 0.02), (None, float("inf"), [], 0.02), (1.0, 1.0, [], -1.0)):
        with pytest.raises(ValueError):
            s.normalize_strength(*args)
    for args in ((0.0, [], 0.02), (1.5, [], 0.02), (0.5, [], float("nan"))):
        with pytest.raises(ValueError):
            s.scale_all(*args)


def test_E_fallback_path_is_used_when_native_is_absent(monkeypatch):
    g, _ = _hub_graph()

    class NoNative:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name in ("normalize_strength", "scale_all"):
                raise AttributeError(name)
            return getattr(self._inner, name)

    real = g.synapses
    g.synapses = NoNative(real)
    try:
        assert g.apply_strength_budget(10.0, None)["native"] is False
        assert g.sleep_downscale(0.98)["native"] is False
    finally:
        g.synapses = real
