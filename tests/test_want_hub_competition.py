# ---- Changelog ----
# [2026-09-30] Claude Sonnet 5.5 (Z12 worker, lane want-hub-competition-d, dispatch #11135) — want-hub (d) tests G / A / K / R
# What: NEW test file for the ENGINE change plan-005 sec 2.6 specifies (additive keyword-only parameters on
#   Graph._prune_synapses + ONE orchestrator, Graph.compete_protected_links). Tests G (golden equivalence, two separate
#   checkouts, two processes), A (armed-path counter), K (contract), R (rim / floors / determinism), plus a P379 guard and
#   harness controls. Daemon-slice tests (D) and the real-graph golden (PG-1) are NOT here — separate lanes.
# Why: Exec P399 condition 1 (the golden test is the load-bearing Syl guard), P404 C5, P412, P419; Exec P420/P421 scope:
#   this commit is TESTS ONLY. neuro_foundation.py (PROTECTED) is NOT touched — Josh has not confirmed his backup or said
#   "proceed". The tests that need the new API therefore FAIL by design against base e4ebf982 (they say so in their message,
#   they are collected, none is skipped or xfailed); G's default-path comparison is runnable now but VACUOUS until the
#   protected commit exists (branch code == base code). See handoffs/z12-want-hub-build/returns/build-001.md.
# How: synthetic seeded graphs only (tests/want_hub_golden_driver.py); NO real checkpoint, vectors, Syl file, tract, embed
#   or TID call. Everything is compared against plan-005 as written; where the plan is ambiguous the test says which reading
#   it took (search "READING:") and the reading is listed in build-001.md — nothing was silently chosen.
#   Names bound by these tests are the plan's ILLUSTRATIVE names (plan 4.2 "names illustrative", 4.3 "name suggestion").
# -------------------
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(REPO / "tests"))

import neuro_foundation as nf  # noqa: E402

# --- P379: the module under test MUST be this worktree's copy, or this whole file refuses to load -----------------------
if Path(nf.__file__).resolve().parent != REPO:
    raise RuntimeError(
        "P379: neuro_foundation resolved to %s, not this worktree (%s) — PYTHONPATH/sys.path hazard; refusing to run"
        % (nf.__file__, REPO)
    )

import want_hub_golden_driver as drv  # noqa: E402

BASE_REV = "e4ebf982b1989fd9066d610b94853bc68bf70d37"
# Read-only BASE checkout (never commit there). Env override per LAW 5; the default is the path the brief names.
BASE_CHECKOUT = Path(os.environ.get("WANT_HUB_BASE_CHECKOUT", "/home/josh/NeuroGraph-worktrees/z12-want-hub-base-e4ebf982"))
K = 3          # guaranteed links per direction in the small synthetic graphs
B_SMALL = 12   # budget that binds (fewer than the eligible count)
_NG_MODULES = ("neuro_foundation", "ng_lite", "ng_tract_bridge", "ng_ecosystem", "ng_autonomic", "ng_embed",
               "openclaw_hook", "openclaw_adapter", "stream_parser", "activation_persistence")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def det_ids():
    """Deterministic engine-minted synapse ids (uuid4 is replaced by a counter for the test's duration)."""
    orig = uuid.uuid4
    drv.install_deterministic_uuid()
    yield
    uuid.uuid4 = orig


@pytest.fixture
def new_api():
    """The tests below that need the additive surface FAIL (never skip) when it is absent — build-001 states this."""
    if not drv.new_api_present(nf):
        pytest.fail("NEW API ABSENT at %s: Graph._prune_synapses has no keyword-only %s (protected-file commit not yet made — "
                    "Exec P420/P421: neuro_foundation.py is not touched until Josh confirms his backup and says proceed)"
                    % (nf.__file__, list(drv.NEW_PARAMS)))
    if not hasattr(nf.Graph, "compete_protected_links"):
        pytest.fail("NEW API ABSENT: Graph.compete_protected_links (plan 4.3 name suggestion) does not exist")


def make(seed=21, **kw):
    kw.setdefault("n_leaf", 4)
    kw.setdefault("n_want", 5)
    kw.setdefault("n_plain", 30)
    kw.setdefault("n_syn", 500)
    return drv.build_graph(nf, seed, **kw)


def _rows(g):
    return {sid: drv._synapse_row(sid, s) for sid, s in g.synapses.items()}


def _digest(g):
    return drv.state_digest(g)


def _run_driver(checkout, scenario, out_dir, tag):
    ckpt = Path(out_dir) / ("%s-%s.msgpack" % (scenario, tag))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "NG_EMBED_REMOTE")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    p = subprocess.run([sys.executable, str(REPO / "tests" / "want_hub_golden_driver.py"), "--checkout", str(checkout),
                        "--scenario", scenario, "--ckpt", str(ckpt)],
                       capture_output=True, text=True, env=env, cwd=str(out_dir), timeout=600)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    assert lines, "driver produced no JSON (rc=%s)\nSTDOUT:%s\nSTDERR:%s" % (p.returncode, p.stdout[-2000:], p.stderr[-2000:])
    rec = json.loads(lines[-1])
    assert not rec.get("void"), "VOID run (P379): %s" % rec.get("reason")
    assert p.returncode == 0, "driver rc=%s\n%s" % (p.returncode, p.stderr[-2000:])
    rec["_ckpt"] = str(ckpt)
    return rec


@pytest.fixture(scope="module")
def base_checkout():
    assert BASE_CHECKOUT.is_dir(), "base checkout missing: %s" % BASE_CHECKOUT
    rev = drv.git_rev(str(BASE_CHECKOUT))
    assert rev == BASE_REV, "base checkout is at %s, must be %s" % (rev, BASE_REV)
    return BASE_CHECKOUT


_COMPARE = ("return", "removal_order", "pruned_events", "state_digest", "checkpoint_sha256", "synapses_before", "synapses_after")


# ---------------------------------------------------------------------------
# P379 preamble
# ---------------------------------------------------------------------------
def test_p379_module_under_test_is_this_worktree(capsys):
    assert Path(nf.__file__).resolve().parent == REPO
    for name in _NG_MODULES:
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "__file__", None):
            assert Path(mod.__file__).resolve().parent == REPO, "%s resolved outside the worktree: %s" % (name, mod.__file__)
    with capsys.disabled():
        print("\nP379 neuro_foundation.__file__=%s git_rev=%s base_rev=%s new_api_present=%s"
              % (nf.__file__, drv.git_rev(str(REPO)), BASE_REV, drv.new_api_present(nf)))


# ---------------------------------------------------------------------------
# Harness self-checks — these run on base today and PASS; they are what make the rest trustworthy
# ---------------------------------------------------------------------------
def test_seed_covers_every_prune_boundary(det_ids):
    g, _ = drv.build_graph(nf, 11)
    c = g.config
    rows = [s for _sid, s in g.synapses.items()]
    assert any(s.weight == c["weight_threshold"] for s in rows), "no weight == weight_threshold synapse"
    for lws in (c["grace_period"] - 1, c["grace_period"], c["grace_period"] + 1):
        assert any(s.weight < c["weight_threshold"] and s.low_weight_steps == lws for s in rows), "no low_weight_steps=%d" % lws
    assert any(s.inactive_steps == c["inactivity_threshold"] * s.salience for s in rows), "no inactive == inactivity*salience"
    assert any(s.salience != 1.0 for s in rows)
    assert any(g.timestep - s.creation_time == c["grace_period"] for s in rows), "no age == grace"
    assert any(s.peak_weight == 2.0 * c["initial_sprouting_weight"] for s in rows), "no peak == 2*initial"
    kinds = set()
    for _sid, s in g.synapses.items():
        a_c, b_c = drv.is_constitutional(g, s.pre_node_id), drv.is_constitutional(g, s.post_node_id)
        a_w, b_w = g._is_identity_protected(s.pre_node_id), g._is_identity_protected(s.post_node_id)
        kinds.add(("rim" if (a_c or b_c) else "want-want" if (a_w and b_w) else "want-ordinary" if (a_w or b_w) else "ordinary-ordinary",
                   "rim-want" if ((a_c and b_w) or (b_c and a_w)) else ""))
    assert {"rim", "want-want", "want-ordinary", "ordinary-ordinary"} <= {k for k, _ in kinds}
    assert ("rim", "rim-want") in kinds


def test_reference_eligibility_matches_the_base_default_path(det_ids):
    """My read-only eligibility reference == what the REAL default-path function removes (identity skip applied)."""
    g, _ = drv.build_graph(nf, 11)
    expected = {sid for sid in g.synapses.keys()
                if not (g._is_identity_protected(g.synapses[sid].pre_node_id) or g._is_identity_protected(g.synapses[sid].post_node_id))
                and drv.ref_eligible(g, sid)}
    assert expected, "seed has no eligible synapse — scenario would be vacuous"
    order, _events = drv.instrument(g)
    n = g._prune_synapses()
    assert n == len(expected) and set(order) == expected


def test_reference_sets_are_consistent(det_ids):
    g, roles = make()
    s = drv.ref_sets(g, K)
    assert s.competing and not (s.competing & s.excluded)
    assert s.F and s.F.isdisjoint(s.competing) and s.F.isdisjoint(s.G)
    assert s.last, "no last-link holds anything back — leaf partners missing"
    assert s.arena == (s.competing | s.G | s.last) & s.arena and s.competing | s.G | s.last == s.arena
    for w in drv.protected_wants(g):
        for idx in (g._outgoing, g._incoming):
            nonF = [x for x in idx[w] if x not in s.F]
            # >=, not ==: a want<->want link is guarded if it is in EITHER endpoint's list (plan sec 2.2 union)
            assert len([x for x in nonF if x in s.G]) >= min(K, len(nonF))
    assert len([x for x in s.competing if drv.ref_eligible(g, x)]) > B_SMALL, "B_SMALL would not bind"


# ---------------------------------------------------------------------------
# G — golden equivalence: SAME seeded graph, base e4ebf982 vs branch, two SEPARATE checkouts, two processes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["door_a", "door_b", "direct_defaults"])
def test_G0_control_base_vs_base_is_deterministic(scenario, base_checkout, tmp_path):
    """If two runs of the SAME code disagree, base-vs-branch could never be interpreted — the harness must be exact first."""
    a = _run_driver(base_checkout, scenario, tmp_path, "base1")
    b = _run_driver(base_checkout, scenario, tmp_path, "base2")
    assert a["neuro_foundation_file"] == b["neuro_foundation_file"]
    assert a["git_rev"] == BASE_REV
    for k in _COMPARE:
        assert a[k] == b[k], "harness is non-deterministic on %s: %s" % (scenario, k)
    assert Path(a["_ckpt"]).read_bytes() == Path(b["_ckpt"]).read_bytes()


@pytest.mark.parametrize("scenario", ["door_a", "door_b", "direct_defaults"])
def test_G_default_path_base_vs_branch(scenario, base_checkout, tmp_path):
    """Return value, removal ORDER and count, `pruned` events, full post-state hash (every synapse field, store items()
    order, _dirty_synapses, _synapse_confirmation_history, node set, timestep) AND the serialized checkpoint BYTES.
    Door A = _structural_plasticity (:3495); Door B tail = prime_and_propagate write_mode + tonic_ages_substrate (:2891)."""
    base = _run_driver(base_checkout, scenario, tmp_path, "base")
    branch = _run_driver(REPO, scenario, tmp_path, "branch")
    assert base["neuro_foundation_file"] != branch["neuro_foundation_file"], "the two runs imported the same file (vacuous)"
    assert base["git_rev"] == BASE_REV and branch["git_rev"] != "" and branch["neuro_foundation_file"].startswith(str(REPO))
    if scenario != "door_b":
        assert base["removal_order"], "seed removes nothing on the default path — comparison would be vacuous"
    for k in _COMPARE:
        assert base[k] == branch[k], "default-path divergence in %s (%s)" % (k, scenario)
    assert Path(base["_ckpt"]).read_bytes() == Path(branch["_ckpt"]).read_bytes(), "serialized checkpoint bytes differ"


def test_G_all_new_parameters_at_defaults_equal_the_base_default_path(base_checkout, tmp_path):
    """The all-new-parameters-at-defaults call (P404 C5) on the BRANCH == the plain call on BASE, same seeded graph.
    Needs the new API: FAILS on base/branch-before-protected-commit (recorded in build-001)."""
    base = _run_driver(base_checkout, "direct_defaults", tmp_path, "base")
    branch = _run_driver(REPO, "direct_explicit_none", tmp_path, "branch")
    assert branch["variant"] == "ok", branch["variant"]
    for k in _COMPARE:
        assert base[k] == branch[k], "explicit-None call diverges from base default path in %s" % k
    assert Path(base["_ckpt"]).read_bytes() == Path(branch["_ckpt"]).read_bytes()


# ---------------------------------------------------------------------------
# A — armed-path counter test (one call, competing mode)
# ---------------------------------------------------------------------------
def test_A_only_low_weight_steps_moves_on_competitors_everything_else_untouched(det_ids, new_api):
    g, _ = make()
    s, kw = drv.good_kwargs(g, K, B_SMALL)
    wt = g.config["weight_threshold"]
    before = _rows(g)
    order, events = drv.instrument(g)
    n = g._prune_synapses(**kw)
    after = _rows(g)
    removed = set(order)
    assert n == len(removed) == len(kw["report"]["removed_ids"]) and 0 < n <= B_SMALL
    assert removed <= s.competing, "something outside the competing set was removed"
    assert removed.isdisjoint(s.F | s.G | s.last)
    for sid, row in before.items():
        if sid in removed:
            assert sid not in after
        elif sid in s.competing:
            # weight, peak, inactive_steps, salience, creation_time byte-identical; ONLY low_weight_steps moved, as the function does
            exp = list(row)
            exp[5] = row[5] + 1 if g.synapses[sid].weight < wt else 0
            assert after[sid] == exp, "competitor %s changed more than low_weight_steps" % sid
        else:
            assert after[sid] == row, "F/G/last-link/outside-arena synapse %s was touched (counters included)" % sid
    assert events == [[n, g.timestep]], "exactly one `pruned` event carrying the post-truncation removed count"


def test_A_orchestrator_makes_exactly_one_call_and_passes_the_callers_sets(det_ids, new_api):
    g, _ = make()
    s = drv.ref_sets(g, K)
    calls = []
    orig = g._prune_synapses

    def spy(*a, **k):
        calls.append((a, k))
        return orig(*a, **k)

    g._prune_synapses = spy
    g.compete_protected_links(K, B_SMALL)
    assert len(calls) == 1, "orchestrator must make EXACTLY ONE _prune_synapses call per cycle (plan 4.4)"
    args, k = calls[0]
    assert args == (), "new parameters are keyword-only"
    assert set(k["competing_ids"]) == s.competing
    assert set(k["excluded_ids"]) == s.excluded and s.F <= set(k["excluded_ids"])
    assert k["max_removals"] == B_SMALL
    ok = drv.ref_order_key(g, s.competing)
    assert {sid: tuple(k["order_key"][sid]) for sid in s.competing} == ok


# ---------------------------------------------------------------------------
# K — contract tests
# ---------------------------------------------------------------------------
def test_K_every_refusal_is_a_ValueError_and_mutates_nothing(det_ids, new_api):
    g, _ = make()
    base_digest = _digest(g)
    for label, kw in drv.bad_calls(g, K, B_SMALL):
        with pytest.raises(ValueError):
            g._prune_synapses(**kw)
        assert _digest(g) == base_digest, "refusal (%s) mutated state — validation must precede the loop" % label


def test_K_missing_endpoint_refuses_before_the_loop(det_ids, new_api):
    g, _ = make()
    s, kw = drv.good_kwargs(g, K, B_SMALL)
    victim = sorted(s.competing)[0]
    del g.nodes[g.synapses[victim].pre_node_id]      # leaves the synapse dangling on purpose
    d0 = _digest(g)
    with pytest.raises(ValueError):
        g._prune_synapses(**kw)
    assert _digest(g) == d0


def test_K_refusals_are_explicit_raises_not_asserts_under_python_dash_O(new_api):
    """Same defects, interpreter run with -O (asserts compiled out): they must still be ValueError, state unchanged."""
    script = (
        "import sys, json\n"
        "sys.path[:0] = [%r, %r]\n"
        "import want_hub_golden_driver as d\n"
        "d.install_deterministic_uuid()\n"
        "import neuro_foundation as nf\n"
        "g, _ = d.build_graph(nf, 21, n_leaf=4, n_want=5, n_plain=30, n_syn=500)\n"
        "d0 = d.state_digest(g)\n"
        "out = {'debug': __debug__, 'cases': []}\n"
        "for label, kw in d.bad_calls(g, %d, %d):\n"
        "    try:\n"
        "        g._prune_synapses(**kw)\n"
        "        out['cases'].append([label, 'NO-RAISE', d.state_digest(g) == d0])\n"
        "    except BaseException as e:\n"
        "        out['cases'].append([label, type(e).__name__, d.state_digest(g) == d0])\n"
        "print(json.dumps(out))\n" % (str(REPO), str(REPO / "tests"), K, B_SMALL)
    )
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "NG_EMBED_REMOTE")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    p = subprocess.run([sys.executable, "-O", "-c", script], capture_output=True, text=True, env=env, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads([ln for ln in p.stdout.splitlines() if ln.startswith("{")][-1])
    assert out["debug"] is False, "-O was not in effect"
    assert out["cases"]
    for label, exc, unchanged in out["cases"]:
        assert exc == "ValueError", "%s -> %s under -O (a bare assert would vanish and this would be NO-RAISE)" % (label, exc)
        assert unchanged, "%s mutated state under -O" % label


def test_K_competing_mode_visits_only_competing_ids(det_ids, new_api):
    """Instrumented: the store is wrapped so any whole-table walk (items/iteration/values/keys) or any read of a
    non-competing id is caught. Also every synapse starts at low_weight_steps=7 so any visit is observable."""

    class CountingStore(object):
        def __init__(self, inner):
            self._i, self.walks, self.read = inner, [], set()

        def __getitem__(self, k):
            self.read.add(k)
            return self._i[k]

        def get(self, k, default=None):
            self.read.add(k)
            return self._i.get(k, default)

        def items(self):
            self.walks.append("items")
            return self._i.items()

        def keys(self):
            self.walks.append("keys")
            return self._i.keys()

        def values(self):
            self.walks.append("values")
            return self._i.values()

        def __iter__(self):
            self.walks.append("iter")
            return iter(self._i)

        def __contains__(self, k):
            return k in self._i

        def __len__(self):
            return len(self._i)

        def __getattr__(self, name):
            return getattr(self._i, name)

    g, _ = make()
    for _sid, syn in list(g.synapses.items()):
        syn.low_weight_steps = 7
    s, kw = drv.good_kwargs(g, K, B_SMALL)
    cs = CountingStore(g.synapses)
    g.synapses = cs
    g._prune_synapses(**kw)
    g.synapses = cs._i
    assert cs.walks == [], "competing mode walked the whole table: %s" % cs.walks
    assert cs.read <= s.competing, "read non-competing synapses: %s" % sorted(cs.read - s.competing)[:5]
    for sid, syn in g.synapses.items():
        if sid not in s.competing:
            assert syn.low_weight_steps == 7, "non-competing %s was visited" % sid


def test_K_report_matches_what_is_actually_gone_and_eligible_ge_removed(det_ids, new_api):
    g, _ = make()
    s, kw = drv.good_kwargs(g, K, B_SMALL)
    ids0 = set(g.synapses.keys())
    order, _e = drv.instrument(g)
    n = g._prune_synapses(**kw)
    rep = kw["report"]
    assert set(rep["removed_ids"]) == ids0 - set(g.synapses.keys()) and list(rep["removed_ids"]) == order
    assert rep["eligible"] >= len(rep["removed_ids"]) == n
    # report['eligible'] is len(to_prune) BEFORE truncation (plan 4.2(g)); its VALUE is checked against the read-only
    # reference, computed before the call, in test_K_eligible_count_equals_reference (the call advances the counters).


def test_K_eligible_count_equals_reference(det_ids, new_api):
    g, _ = make()
    s, kw = drv.good_kwargs(g, K, B_SMALL)
    want = len([x for x in s.competing if drv.ref_eligible(g, x)])   # computed BEFORE the call
    g._prune_synapses(**kw)
    assert kw["report"]["eligible"] == want


def test_K_default_path_accepts_max_removals_none_and_report_and_removes_everything_eligible(det_ids, new_api):
    g, _ = make()
    expected = {sid for sid in g.synapses.keys()
                if not (g._is_identity_protected(g.synapses[sid].pre_node_id) or g._is_identity_protected(g.synapses[sid].post_node_id))
                and drv.ref_eligible(g, sid)}
    rep = {}
    order, _e = drv.instrument(g)
    n = g._prune_synapses(max_removals=None, report=rep)
    assert n == len(expected) and set(order) == expected
    assert rep["removed_ids"] == order and rep["eligible"] == n


def test_K_orchestrator_refuses_topk_or_budget_below_one_before_touching_anything(det_ids, new_api):
    g, _ = make()
    d0 = _digest(g)
    for k, b in ((0, B_SMALL), (-1, B_SMALL), (K, 0), (K, -5), (0, 0)):
        with pytest.raises(ValueError):
            g.compete_protected_links(k, b)
        assert _digest(g) == d0


# ---------------------------------------------------------------------------
# R — rim / floors / determinism (the pass is run ALONE: no step(), STDP, homeostasis, inject_reward)
# ---------------------------------------------------------------------------
def _cycles_until_exhausted(g, k, b, limit=500):
    removed_per_cycle = []
    for _ in range(limit):
        n0 = len(g.synapses)
        g.compete_protected_links(k, b)
        removed_per_cycle.append(n0 - len(g.synapses))
        if removed_per_cycle[-1] == 0:
            return removed_per_cycle
    pytest.fail("pass did not exhaust its competitors in %d cycles" % limit)


@pytest.mark.parametrize("budget", [7, 10 ** 6])
def test_R_rim_floors_and_last_link_hold_in_the_worst_state(budget, det_ids, new_api):
    g, roles = make(seed=31)
    F = drv.frozen_rim(g)
    s0 = drv.ref_sets(g, K)
    drv.force_worst(g, F | s0.arena)                  # rim in the worst state AND every competitor eligible
    rim_w = {sid: repr(g.synapses[sid].weight) for sid in F}
    prot = [n for n in g.nodes if g._is_identity_protected(n)]
    floors = {}
    for w in drv.protected_wants(g):
        for name, idx in (("out", g._outgoing), ("in", g._incoming)):
            nonF = [x for x in idx[w] if x not in F]
            floors[(w, name)] = min(K, len(nonF))
    partner_deg = {n: len(g._outgoing.get(n, ())) + len(g._incoming.get(n, ()))
                   for n in g.nodes if not g._is_identity_protected(n)}
    per_cycle = _cycles_until_exhausted(g, K, budget)
    assert all(r <= budget for r in per_cycle), "a call removed more than B"
    assert per_cycle[0] > 0, "nothing was removed — worst-state scenario is vacuous"
    assert F <= set(g.synapses.keys()), "a rim synapse was removed"
    g._collect_orphan_nodes()
    assert roles["rim"] in g.nodes and all(n in g.nodes for n in prot), "a protected node was removed"
    assert {sid: repr(g.synapses[sid].weight) for sid in F} == rim_w, "the pass wrote a rim weight"
    for (w, name), floor in floors.items():
        idx = g._outgoing if name == "out" else g._incoming
        assert len([x for x in idx[w] if x not in F]) >= floor, "%s lost its %s floor" % (w, name)
    for n, d in partner_deg.items():
        if d > 0:
            assert n in g.nodes and (len(g._outgoing.get(n, ())) + len(g._incoming.get(n, ()))) > 0, "%s ended with zero synapses" % n


def test_R_determinism_two_runs_over_identical_state_remove_the_identical_ids(det_ids, new_api):
    runs = []
    for _ in range(2):
        drv.install_deterministic_uuid()   # re-seed the id counter: both runs mint identical ids
        g, _ = make(seed=41)
        order, _e = drv.instrument(g)
        for _c in range(3):
            g.compete_protected_links(K, B_SMALL)
        runs.append((list(order), drv.state_digest(g)))
    assert runs[0][0] and runs[0] == runs[1]


def test_R_removed_set_is_the_top_B_eligible_competitors_by_the_height_key(det_ids, new_api):
    g, _ = make(seed=51)
    s = drv.ref_sets(g, K)
    key = drv.ref_order_key(g, s.competing)
    elig = [x for x in s.competing if drv.ref_eligible(g, x)]
    assert len(elig) > B_SMALL
    expected = sorted(elig, key=lambda x: key[x])[:B_SMALL]
    order, _e = drv.instrument(g)
    g.compete_protected_links(K, B_SMALL)
    assert set(order) == set(expected)
    # READING (plan 4.2(e)/(f)): to_prune is sorted by order_key BEFORE truncation and removed in that order, so the
    # removal ORDER follows the key. The plan does not state the post-truncation order in words; if the build removes in
    # another order this assertion (and only this one) is the place to look.
    assert order == expected


def test_R_when_B_exceeds_the_eligible_count_every_eligible_competitor_goes(det_ids, new_api):
    g, _ = make(seed=52)
    s = drv.ref_sets(g, K)
    elig = {x for x in s.competing if drv.ref_eligible(g, x)}
    order, _e = drv.instrument(g)
    g.compete_protected_links(K, 10 ** 6)
    assert set(order) == elig


def test_R_ordinary_synapses_outside_the_arena_are_never_touched_by_the_pass(det_ids, new_api):
    g, _ = make(seed=53)
    s = drv.ref_sets(g, K)
    before = _rows(g)
    g.compete_protected_links(K, B_SMALL)
    after = _rows(g)
    for sid, row in before.items():
        if sid not in s.competing:
            assert after.get(sid) == row, "%s (not a competitor) changed" % sid
