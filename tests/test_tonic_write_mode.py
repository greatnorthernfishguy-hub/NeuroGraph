"""
Tests for write-mode prime_and_propagate() — The Tonic verification.

Verifies that write_mode=True on prime_and_propagate():
1. Persists voltage changes (doesn't save/restore)
2. Records last_spike_time on fired nodes
3. Engages STDP — synapse weights change after exploration
4. Read mode (default) still works exactly as before

# ---- Changelog ----
# [2026-03-24] Claude Code (Opus 4.6) — Initial creation
# What: Verification tests for The Tonic's write-mode spreading activation.
# Why: PRD §10.2 — must verify STDP engagement before building The Tonic.
# How: Create minimal graph topologies, run prime_and_propagate in both
#   modes, assert weight changes in write mode, no changes in read mode.
# -------------------
"""

import math
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_foundation import Graph


def _make_test_graph() -> Graph:
    """Create a minimal graph with two connected nodes for STDP testing."""
    config = {
        "decay_rate": 0.97,
        "default_threshold": 0.85,
        "refractory_period": 2,
        "learning_rate": 0.01,
        "prime_strength": 1.0,
        "three_factor_enabled": False,
        "structural_plasticity_enabled": False,
        "he_experience_threshold": 3,
        "prediction_window": 5,
    }
    g = Graph(config=config)

    # Create two nodes: A → B
    nA = g.create_node(node_id="A")
    nB = g.create_node(node_id="B")

    # Create a synapse from A to B with known initial weight
    g.create_synapse(
        pre_node_id="A",
        post_node_id="B",
        weight=0.5,
        delay=1,
    )

    return g


def test_read_mode_preserves_state():
    """Read mode (default) must not alter voltages or weights."""
    g = _make_test_graph()

    # Record initial state
    v_a_before = g.nodes["A"].voltage
    v_b_before = g.nodes["B"].voltage
    syn = list(g.synapses.values())[0]
    w_before = syn.weight

    # Run read-mode propagation with strong current
    result = g.prime_and_propagate(
        node_ids=["A"],
        currents=[2.0],
        steps=5,
    )

    # Voltages must be restored
    assert g.nodes["A"].voltage == v_a_before, \
        f"Read mode changed A's voltage: {v_a_before} -> {g.nodes['A'].voltage}"
    assert g.nodes["B"].voltage == v_b_before, \
        f"Read mode changed B's voltage: {v_b_before} -> {g.nodes['B'].voltage}"

    # Weight must be unchanged
    assert syn.weight == w_before, \
        f"Read mode changed synapse weight: {w_before} -> {syn.weight}"

    # Spike time must NOT be recorded
    assert g.nodes["A"].last_spike_time == -math.inf, \
        f"Read mode recorded spike time on A: {g.nodes['A'].last_spike_time}"

    print(f"PASS: read mode preserved state. {len(result.fired_entries)} nodes fired during propagation.")


def test_write_mode_persists_voltages():
    """Write mode must NOT restore voltages after propagation."""
    g = _make_test_graph()

    v_a_before = g.nodes["A"].voltage
    v_b_before = g.nodes["B"].voltage

    result = g.prime_and_propagate(
        node_ids=["A"],
        currents=[2.0],
        steps=5,
        write_mode=True,
    )

    # After write-mode propagation with 5 steps of decay,
    # voltages should NOT be the same as before (not restored)
    # Node A was injected with 2.0 current, may have fired and reset
    # At minimum, the state should be different from initial
    v_a_after = g.nodes["A"].voltage
    v_b_after = g.nodes["B"].voltage

    # At least one voltage should differ (A was injected, fired, decayed)
    state_changed = (v_a_after != v_a_before) or (v_b_after != v_b_before)
    assert state_changed, \
        f"Write mode restored voltages! A: {v_a_before}->{v_a_after}, B: {v_b_before}->{v_b_after}"

    print(f"PASS: write mode persisted voltage changes. A: {v_a_before:.3f}->{v_a_after:.3f}, B: {v_b_before:.3f}->{v_b_after:.3f}")


def test_write_mode_records_spike_time():
    """Write mode must record last_spike_time on fired nodes."""
    g = _make_test_graph()

    assert g.nodes["A"].last_spike_time == -math.inf, \
        "Node A should start with no spike history"

    result = g.prime_and_propagate(
        node_ids=["A"],
        currents=[2.0],
        steps=5,
        write_mode=True,
    )

    # Node A should have fired (2.0 current >> 0.85 threshold)
    a_fired = any(e.node_id == "A" for e in result.fired_entries)
    assert a_fired, "Node A should have fired with 2.0 current injection"

    assert g.nodes["A"].last_spike_time != -math.inf, \
        f"Write mode did not record spike time on A"

    print(f"PASS: write mode recorded spike time. A.last_spike_time = {g.nodes['A'].last_spike_time}")


def test_write_mode_engages_stdp():
    """Write mode must cause STDP weight changes on synapses."""
    g = _make_test_graph()

    syn = list(g.synapses.values())[0]
    w_before = syn.weight

    # First, make A fire to establish a spike time
    result1 = g.prime_and_propagate(
        node_ids=["A"],
        currents=[2.0],
        steps=3,
        write_mode=True,
    )

    a_fired = any(e.node_id == "A" for e in result1.fired_entries)
    assert a_fired, "Node A must fire in first pass"

    # Now make B fire — B's incoming synapse from A should see
    # A's spike time and apply STDP (LTP if A fired before B)
    result2 = g.prime_and_propagate(
        node_ids=["B"],
        currents=[2.0],
        steps=3,
        write_mode=True,
    )

    b_fired = any(e.node_id == "B" for e in result2.fired_entries)
    assert b_fired, "Node B must fire in second pass"

    w_after = syn.weight

    assert w_after != w_before, \
        f"STDP did not change synapse weight! Before: {w_before}, After: {w_after}"

    # A fired before B → LTP → weight should increase
    assert w_after > w_before, \
        f"Expected LTP (weight increase) but got: {w_before} -> {w_after}"

    print(f"PASS: STDP engaged. Weight changed: {w_before:.6f} -> {w_after:.6f} (delta: {w_after - w_before:.6f})")


def test_write_mode_does_not_break_read_mode():
    """Running write mode then read mode: read mode must still restore."""
    g = _make_test_graph()

    # Write mode pass
    g.prime_and_propagate(
        node_ids=["A"],
        currents=[2.0],
        steps=3,
        write_mode=True,
    )

    # Capture state after write mode
    v_a_after_write = g.nodes["A"].voltage
    v_b_after_write = g.nodes["B"].voltage

    # Read mode pass — should restore to post-write state
    g.prime_and_propagate(
        node_ids=["B"],
        currents=[2.0],
        steps=3,
        write_mode=False,
    )

    # Voltages should be restored to what they were before the read pass
    assert g.nodes["A"].voltage == v_a_after_write, \
        f"Read mode after write mode didn't restore A: {v_a_after_write} -> {g.nodes['A'].voltage}"
    assert g.nodes["B"].voltage == v_b_after_write, \
        f"Read mode after write mode didn't restore B: {v_b_after_write} -> {g.nodes['B'].voltage}"

    print(f"PASS: read mode still restores correctly after write mode")


if __name__ == "__main__":
    tests = [
        test_read_mode_preserves_state,
        test_write_mode_persists_voltages,
        test_write_mode_records_spike_time,
        test_write_mode_engages_stdp,
        test_write_mode_does_not_break_read_mode,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("The Tonic write-mode verification: ALL CLEAR")
    else:
        print("The Tonic write-mode verification: ISSUES FOUND")
