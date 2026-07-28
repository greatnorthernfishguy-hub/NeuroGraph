# ---- Changelog ----
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
