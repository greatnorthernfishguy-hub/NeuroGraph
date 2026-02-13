"""Tests for SNN dynamics: voltage decay, spike propagation, refractory periods.

Covers PRD §2.2.4 simulation loop and §9 acceptance criteria.
"""

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuro_foundation import Graph, Node, SynapseType


class TestVoltageDecay:
    """Voltage should decay toward resting_potential each step."""

    def test_decay_reduces_voltage(self):
        g = Graph()
        n = g.create_node("A")
        g.stimulate("A", 0.5)
        assert n.voltage == pytest.approx(0.5, abs=0.01)
        g.step()
        # After one step: v = 0.5 * 0.95 + 0.05 * 0 = 0.475
        assert n.voltage < 0.5

    def test_decay_toward_resting(self):
        g = Graph()
        n = g.create_node("A")
        n.resting_potential = -0.1
        n.voltage = 0.5
        for _ in range(200):
            g.step()
        # Should approach resting potential
        assert abs(n.voltage - n.resting_potential) < 0.01

    def test_custom_decay_rate(self):
        g = Graph({"decay_rate": 0.5})
        n = g.create_node("A")
        n.voltage = 1.0
        g.step()
        # v = 1.0 * 0.5 + 0.5 * 0.0 = 0.5
        assert n.voltage == pytest.approx(0.5, abs=0.01)


class TestSpikeDetection:
    """Nodes should fire when voltage >= threshold and not refractory."""

    def test_fires_at_threshold(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        n = g.create_node("A")
        g.stimulate("A", 1.0)
        result = g.step()
        assert "A" in result.fired_node_ids

    def test_no_fire_below_threshold(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        g.create_node("A")
        g.stimulate("A", 0.9)
        result = g.step()
        assert "A" not in result.fired_node_ids

    def test_voltage_resets_after_spike(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        n = g.create_node("A")
        g.stimulate("A", 1.5)
        g.step()
        assert n.voltage == pytest.approx(n.resting_potential, abs=0.01)

    def test_spike_history_recorded(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        n = g.create_node("A")
        g.stimulate("A", 1.5)
        g.step()
        assert len(n.spike_history) == 1
        assert n.last_spike_time == 1.0


class TestRefractoryPeriod:
    """Mandatory refractory period prevents rapid re-firing (PRD §3.2.1)."""

    def test_cannot_fire_during_refractory(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0, "refractory_period": 2})
        n = g.create_node("A")

        # Fire once
        g.stimulate("A", 2.0)
        r1 = g.step()
        assert "A" in r1.fired_node_ids

        # Immediately stimulate again above threshold
        g.stimulate("A", 2.0)
        r2 = g.step()
        assert "A" not in r2.fired_node_ids  # Still refractory

        g.stimulate("A", 2.0)
        r3 = g.step()
        assert "A" not in r3.fired_node_ids  # Still refractory (period=2)

        # Now should be able to fire
        g.stimulate("A", 2.0)
        r4 = g.step()
        assert "A" in r4.fired_node_ids

    def test_refractory_decrements(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0, "refractory_period": 3})
        n = g.create_node("A")
        g.stimulate("A", 2.0)
        g.step()  # fires, refractory = 3 (not decremented on firing step)
        assert n.refractory_remaining == 3
        g.step()  # decrement → 2
        assert n.refractory_remaining == 2


class TestSpikePropagation:
    """Spikes should propagate through synapses with correct delays."""

    def test_basic_propagation(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0, "max_weight": 5.0})
        g.create_node("A")
        n_b = g.create_node("B")
        g.create_synapse("A", "B", weight=2.0, delay=1)

        g.stimulate("A", 1.5)
        g.step()  # A fires, spike scheduled at t+1

        # B should receive current at step 2
        r2 = g.step()
        # B should have voltage > 0 from the propagated spike
        # (decay applied first, then delivery)
        assert n_b.voltage > 0 or "B" in r2.fired_node_ids

    def test_delay_respected(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0, "max_weight": 5.0})
        g.create_node("A")
        n_b = g.create_node("B")
        g.create_synapse("A", "B", weight=2.0, delay=3)

        g.stimulate("A", 1.5)
        g.step()  # t=1, A fires

        # B shouldn't get current at t=2 or t=3
        g.step()  # t=2
        v_at_2 = n_b.voltage
        g.step()  # t=3
        v_at_3 = n_b.voltage

        # At t=4, the delayed spike arrives
        g.step()  # t=4
        v_at_4 = n_b.voltage
        assert v_at_4 > v_at_3 or n_b.last_spike_time == 4.0

    def test_inhibitory_propagation(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        g.create_node("A", is_inhibitory=True)
        n_b = g.create_node("B")
        n_b.voltage = 0.8
        g.create_synapse("A", "B", weight=1.0, delay=1)

        g.stimulate("A", 1.5)
        g.step()  # A fires (inhibitory)
        initial_v = n_b.voltage
        g.step()  # Spike arrives at B
        # B's voltage should decrease due to inhibitory signal
        assert n_b.voltage < initial_v

    def test_chain_propagation(self):
        """A → B → C chain should propagate over multiple steps."""
        g = Graph({"default_threshold": 1.0, "decay_rate": 0.99, "max_weight": 5.0})
        g.create_node("A")
        g.create_node("B")
        g.create_node("C")
        g.create_synapse("A", "B", weight=2.0, delay=1)
        g.create_synapse("B", "C", weight=2.0, delay=1)

        g.stimulate("A", 1.5)
        results = g.step_n(5)

        # A should fire at t=1
        assert "A" in results[0].fired_node_ids
        # B should fire at t=2 or t=3 (after delay)
        b_fired = any("B" in r.fired_node_ids for r in results[1:4])
        assert b_fired
        # C should fire after B
        c_fired = any("C" in r.fired_node_ids for r in results[2:5])
        assert c_fired


class TestLargeGraphStability:
    """PRD §9 Acceptance: 1K-node graph runs 10K steps without explosion or silent death."""

    def test_1k_nodes_10k_steps(self):
        g = Graph({
            "decay_rate": 0.95,
            "default_threshold": 1.0,
            "refractory_period": 2,
            "co_activation_window": 5,
            "grace_period": 500,
            "weight_threshold": 0.01,
            "inactivity_threshold": 1000,
        })

        rng = np.random.RandomState(42)

        # Create 1000 nodes (20% inhibitory)
        node_ids = [f"n{i}" for i in range(1000)]
        for nid in node_ids:
            g.create_node(nid, is_inhibitory=(rng.random() < 0.2))

        # Create sparse random connections (~10 per node, stronger weights)
        for i in range(10000):
            pre = node_ids[rng.randint(0, 1000)]
            post = node_ids[rng.randint(0, 1000)]
            if pre != post:
                try:
                    g.create_synapse(pre, post, weight=rng.random() * 1.0 + 0.2)
                except (KeyError, ValueError):
                    pass

        total_spikes = 0
        max_voltage = 0.0
        silent_stretch = 0
        max_silent = 0

        for step_i in range(10000):
            # Moderate random input to ~10% of nodes each step
            for nid in rng.choice(node_ids, size=100, replace=False):
                g.stimulate(nid, rng.random() * 0.5 + 0.1)

            result = g.step()
            n_fired = len(result.fired_node_ids)
            total_spikes += n_fired

            if n_fired == 0:
                silent_stretch += 1
                max_silent = max(max_silent, silent_stretch)
            else:
                silent_stretch = 0

            # Check for explosion (sample every 100 steps to save time)
            if step_i % 100 == 0:
                for node in g.nodes.values():
                    if abs(node.voltage) > max_voltage:
                        max_voltage = abs(node.voltage)

        # Not completely silent
        assert total_spikes > 0, "Network is completely silent (dead)"
        # No explosion: max voltage should be bounded
        assert max_voltage < 1000, f"Voltage explosion detected: {max_voltage}"
        # Not perpetually silent
        assert max_silent < 500, f"Network went silent for {max_silent} consecutive steps"
        # Some firing activity
        avg_rate = total_spikes / (10000 * 1000)
        assert avg_rate > 0.0001, f"Average firing rate too low: {avg_rate}"


# Need numpy for the large test
import numpy as np
