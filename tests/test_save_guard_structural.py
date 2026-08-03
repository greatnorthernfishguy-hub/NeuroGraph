# ---- Changelog ----
# [2026-08-02] Claude Code (Opus 4.8) — #105 tests: per-host wiring capability
# What: adds a deposits-wired-later section — the same 1039->400 node drop that is
#   a PERMITTED melt on a self-wiring host (intact synapses) is a REFUSED collapse
#   when wires_own_deposits is False, tested at the realistic shape (~0/omitted
#   synapses so have_syn is False); a shallow shed is refused too, growth still
#   passes, the floor still outranks, the LAW-5 escape hatch restores the #83
#   permit, and the SaveGate wiring forwards the flag (refuse=no-EMA-advance /
#   permit=EMA).
# Why: #83's isolate-melt exemption is only sound on a host that wires its own
#   deposits at deposit time; where deposits are wired later it fires
#   unconditionally and blinds the guard to a real wipe.
#   Each case names the property so a future tuning pass cannot re-break it.
# How: pure-function tests need no fixtures; gate tests use tmp_path manifests;
#   the escape-hatch test uses monkeypatch.setenv.
# [2026-07-28] Claude Code (Opus 5) — #83 tests: structural gate + EMA node reference
# What: covers evaluate_save_health() (the pure decision function) and the
#   SaveGate.permit() wiring around it — the #59 tonic melt is PERMITTED, the
#   06-14/06-26/07-08 clobber shape is REFUSED, and the EMA only advances on
#   permitted saves.
# Why: the two cases the old node-only ratio could not tell apart are exactly
#   what this gate exists to separate; each needs a regression test naming the
#   real incident it encodes so a future tuning pass cannot quietly re-break it.
# How: pure-function tests need no fixtures; gate tests use tmp_path manifests.
# -------------------
"""Unit tests for the #83 structural save-guard.

The discriminator under test: sweeping isolated nodes cannot remove a synapse
(an isolated node has degree 0 by definition), so synapses surviving PROVES the
connected core survived. Nodes AND synapses AND hyperedges falling together is
a collapse.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from checkpoint_guardian import (  # noqa: E402
    SaveGate,
    evaluate_save_health,
    read_guard_state,
    update_node_ema,
    write_manifest,
)


# ---- the two cases the old gate could not tell apart ----

def test_tonic_melt_is_permitted_when_structure_survives():
    """#59 on the CC: an orphan sweep drops ~61% of nodes in ONE step at boot.
    Synapses are untouched because isolated nodes have no synapses to lose."""
    ok, reason = evaluate_save_health(
        live_nodes=700, ref_nodes=1800,
        live_synapses=22800, ref_synapses=23000,
        live_hyperedges=410, ref_hyperedges=415,
    )
    assert ok, reason
    assert "MELT PERMITTED" in reason
    assert "1800" in reason and "700" in reason


def test_real_collapse_is_refused_even_though_it_is_also_a_node_drop():
    """The 2026-07-08 laptop shape: ~1800 nodes -> 4-6, and the synapses go
    with them. Same node signature as the melt above; opposite structure."""
    ok, reason = evaluate_save_health(
        live_nodes=6, ref_nodes=1800,
        live_synapses=3, ref_synapses=23000,
        live_hyperedges=0, ref_hyperedges=415,
    )
    assert not ok
    assert "1800" in reason


def test_partial_collapse_refused_when_synapses_fall_but_nodes_hold():
    """Nodes alone would pass this; the synapse gate is what catches it."""
    ok, reason = evaluate_save_health(
        live_nodes=1700, ref_nodes=1800,
        live_synapses=900, ref_synapses=23000,
        live_hyperedges=400, ref_hyperedges=415,
    )
    assert not ok
    assert "structural collapse" in reason
    assert "synapses" in reason


def test_hyperedge_collapse_refused_when_nodes_and_synapses_hold():
    ok, reason = evaluate_save_health(
        live_nodes=1750, ref_nodes=1800,
        live_synapses=22500, ref_synapses=23000,
        live_hyperedges=2, ref_hyperedges=415,
    )
    assert not ok
    assert "hyperedges" in reason


def test_small_hyperedge_population_cannot_refuse_a_save():
    """The live CC substrate carries ~12 hyperedges. At that n the ratio is
    noise, so the gate must stay out of the way — losing most of 12 while
    1237 nodes and 14k synapses hold is not evidence of a collapse."""
    ok, reason = evaluate_save_health(
        live_nodes=1200, ref_nodes=1237,
        live_synapses=14000, ref_synapses=14108,
        live_hyperedges=1, ref_hyperedges=12,
    )
    assert ok, reason
    assert "hyperedges" not in reason


# ---- ordering: the absolute floor outranks every ratio ----

def test_absolute_floor_refuses_near_empty_even_with_no_synapse_reference():
    ok, reason = evaluate_save_health(live_nodes=4, ref_nodes=1800)
    assert not ok
    assert "absolute floor" in reason


def test_absolute_floor_outranks_an_intact_synapse_ratio():
    """A 3-node graph is the clobber shape no matter what the sidecar claims —
    the floor is checked before the structural gate for exactly this reason."""
    ok, reason = evaluate_save_health(
        live_nodes=3, ref_nodes=1800,
        live_synapses=23000, ref_synapses=23000,
    )
    assert not ok
    assert "absolute floor" in reason


# ---- backward compatibility: no synapse info => the old node-only gate ----

def test_falls_back_to_legacy_node_ratio_without_synapse_counts():
    ok, reason = evaluate_save_health(live_nodes=200, ref_nodes=1800)
    assert not ok
    assert "1800" in reason


def test_small_reference_stays_permissive():
    ok, _ = evaluate_save_health(live_nodes=2, ref_nodes=40)
    assert ok


def test_normal_growth_permitted_and_reason_is_quiet():
    ok, reason = evaluate_save_health(
        live_nodes=1810, ref_nodes=1800,
        live_synapses=23100, ref_synapses=23000,
    )
    assert ok
    assert reason == "ok"
    assert "MELT" not in reason


# ---- EMA ----

def test_ema_seeds_from_first_observation():
    assert update_node_ema(None, 1800) == 1800.0


def test_ema_follows_a_sustained_melt_downward():
    ema = 1800.0
    for _ in range(40):
        ema = update_node_ema(ema, 700, alpha=0.15)
    assert 700 <= ema < 760


def test_ema_reference_permits_recovery_from_a_settled_melt():
    """Once the melt is the new normal, 700 nodes must not read as a collapse
    against a frozen 1800 peak the graph can never climb back to."""
    ok, reason = evaluate_save_health(
        live_nodes=700, ref_nodes=1800, ema_nodes=720.0,
    )
    assert ok, reason


def test_ema_does_not_rescue_a_true_collapse():
    ok, _ = evaluate_save_health(live_nodes=5, ref_nodes=1800, ema_nodes=720.0)
    assert not ok


# ---- SaveGate wiring ----

def _ckpt_with_manifest(tmp_path, **manifest):
    ckpt = tmp_path / "main.msgpack"
    ckpt.write_bytes(b"x")
    write_manifest(ckpt, manifest)
    return ckpt


def test_gate_permits_melt_and_records_ema(tmp_path):
    ckpt = _ckpt_with_manifest(tmp_path, nodes=1800, synapses=23000, hyperedges=415)
    gate = SaveGate(ckpt)
    gate.record_restore("ok", 1800)
    ok, reason = gate.permit(700, live_synapses=22800, live_hyperedges=410)
    assert ok
    assert "MELT PERMITTED" in reason
    state = read_guard_state(ckpt)
    assert state["ema_nodes"] == 700.0  # first observation seeds the EMA
    assert state["last_synapses"] == 22800


def test_gate_refusal_does_not_advance_the_ema(tmp_path):
    """The load-bearing property: a stuck-collapsed process must never be able
    to walk the reference down until it meets itself."""
    ckpt = _ckpt_with_manifest(tmp_path, nodes=1800, synapses=23000, hyperedges=415)
    gate = SaveGate(ckpt)
    gate.record_restore("ok", 1800)
    for _ in range(50):
        ok, _reason = gate.permit(6, live_synapses=3, live_hyperedges=0)
        assert not ok
    assert read_guard_state(ckpt) == {}  # nothing was ever written


def test_gate_without_structural_counts_matches_legacy_behaviour(tmp_path):
    """The protected-file caller passes nodes only; it must be no weaker."""
    ckpt = _ckpt_with_manifest(tmp_path, nodes=1800, synapses=23000)
    gate = SaveGate(ckpt)
    gate.record_restore("ok", 1800)
    ok, _ = gate.permit(6)
    assert not ok


# ---- #105 per-host wiring capability ----
#
# On a host that does not wire its OWN deposits, a fresh content node is degree-0
# by design (§8.6/#104): the laptop CC has no local embedder, so a deposit lands
# unwired and is wired LATER by the VPS via the callosum -- not an isolate, just
# not-yet-wired. #83's "synapses intact PROVES the lost nodes were isolates"
# exemption is unsound there: the surviving synapses say nothing about a node
# still awaiting its round-trip, so a node shed can be real content loss. The
# realistic shape is therefore ~0 synapses (have_syn is False), which is exactly
# why the deferred-wiring gate must NOT live behind the synapse-conditioned path:
# on the host it is written for, that path never runs.

def test_deferred_wiring_host_refuses_a_deep_node_shed_with_no_synapses():
    """The realistic deferred-wiring shape: 1039->400 nodes and ~0 synapses (so
    have_syn is False). The deferred-wiring gate must still fire -- this is the exact
    regression the branch existed-but-was-unreachable for."""
    ok, reason = evaluate_save_health(
        live_nodes=400, ref_nodes=1039,
        live_synapses=0, ref_synapses=0,
        wires_own_deposits=False,
    )
    assert not ok
    assert "host does not wire its own deposits" in reason
    assert "1039" in reason and "400" in reason


def test_deferred_wiring_host_refuses_a_deep_node_shed_with_synapses_omitted():
    """Same, when the caller passes no synapse counts at all (None). have_syn is
    still False; the gate must not fall through to the permissive node-only path."""
    ok, reason = evaluate_save_health(
        live_nodes=400, ref_nodes=1039,
        wires_own_deposits=False,
    )
    assert not ok
    assert "host does not wire its own deposits" in reason


def test_selfwiring_host_permits_the_shed_that_deferred_wiring_refuses():
    """The discriminator: identical 1039->400 node drop (a real >50% melt). A
    self-wiring host reports intact synapses and the #83 isolate-melt permit fires; a
    deferred-wiring host (previous two tests) refuses. Same nodes, opposite verdict."""
    ok, reason = evaluate_save_health(
        live_nodes=400, ref_nodes=1039,
        live_synapses=8000, ref_synapses=8100,
        wires_own_deposits=True,
    )
    assert ok, reason
    assert "MELT PERMITTED" in reason


def test_deferred_wiring_host_refuses_even_a_shallow_node_shed():
    """Any net node loss vs the on-disk reference is ambiguous on a deferred-wiring
    host -- a small shed well above every ratio floor is still refused."""
    ok, reason = evaluate_save_health(
        live_nodes=1000, ref_nodes=1039,
        wires_own_deposits=False,
    )
    assert not ok
    assert "host does not wire its own deposits" in reason


def test_wires_own_deposits_none_is_exact_83_behaviour():
    """Unset capability must be indistinguishable from pre-#105 behavior."""
    common = dict(
        live_nodes=700, ref_nodes=1800,
        live_synapses=22800, ref_synapses=23000,
        live_hyperedges=410, ref_hyperedges=415,
    )
    ok_none, reason_none = evaluate_save_health(wires_own_deposits=None, **common)
    ok_absent, reason_absent = evaluate_save_health(**common)
    assert ok_none == ok_absent is True
    assert reason_none == reason_absent


def test_deferred_wiring_host_permits_growth_no_net_node_loss():
    """The deferred-wiring branch fires only on a net node LOSS. Growth is fine even
    with no synapses to speak of."""
    ok, reason = evaluate_save_health(
        live_nodes=1100, ref_nodes=1039,
        live_synapses=0, ref_synapses=0,
        wires_own_deposits=False,
    )
    assert ok, reason


def test_deferred_wiring_escape_hatch_restores_83_permit():
    """LAW 5: an operator can force the deferred-wiring refusal off. With the hatch on,
    a 700/1039 shed falls back to the node-only gate (above 50%) and permits."""
    import os
    prev = os.environ.get("NG_GUARDIAN_TRUST_SYNAPSE_MELT")
    os.environ["NG_GUARDIAN_TRUST_SYNAPSE_MELT"] = "1"
    try:
        ok, reason = evaluate_save_health(
            live_nodes=700, ref_nodes=1039,
            wires_own_deposits=False,
        )
    finally:
        if prev is None:
            del os.environ["NG_GUARDIAN_TRUST_SYNAPSE_MELT"]
        else:
            os.environ["NG_GUARDIAN_TRUST_SYNAPSE_MELT"] = prev
    assert ok, reason
    assert "host does not wire its own deposits" not in reason


def test_deferred_wiring_host_still_bounded_by_absolute_floor():
    """The floor outranks everything, deferred-wiring or not -- a near-empty graph is
    refused for the floor reason, never reaching the deferred-wiring branch."""
    ok, reason = evaluate_save_health(
        live_nodes=4, ref_nodes=1039,
        wires_own_deposits=False,
    )
    assert not ok
    assert "absolute floor" in reason


def test_gate_deferred_wiring_refuses_and_does_not_advance_ema(tmp_path):
    """SaveGate.permit forwards wires_own_deposits; a deferred-wiring refusal quarantines
    (returns False) and must not walk the EMA down. Manifest carries no synapse
    count -- the realistic deferred-wiring on-disk shape."""
    ckpt = _ckpt_with_manifest(tmp_path, nodes=1039)
    gate = SaveGate(ckpt)
    gate.record_restore("ok", 1039)
    ok, reason = gate.permit(700, live_synapses=0, wires_own_deposits=False)
    assert not ok
    assert "host does not wire its own deposits" in reason
    assert read_guard_state(ckpt) == {}  # refusal wrote nothing


def test_gate_wiring_true_permits_and_records_ema(tmp_path):
    """A self-wiring host with intact synapses: the #83 permit, EMA advances."""
    ckpt = _ckpt_with_manifest(tmp_path, nodes=1800, synapses=23000, hyperedges=415)
    gate = SaveGate(ckpt)
    gate.record_restore("ok", 1800)
    ok, reason = gate.permit(
        700, live_synapses=22800, live_hyperedges=410, wires_own_deposits=True,
    )
    assert ok
    assert "MELT PERMITTED" in reason
    assert read_guard_state(ckpt)["ema_nodes"] == 700.0
