"""Tests for STDP learning: LTP, LTD, weight-dependent scaling, temporal aliasing.

Covers PRD §3.1 and §9 acceptance criteria:
- STDP correctly strengthens causal sequences (A→B strengthens when A fires before B)
- STDP correctly weakens acausal pairs (A→B weakens when A fires after B)
"""

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuro_foundation import Graph, STDPRule, SynapseType


class TestLTP:
    """Long-Term Potentiation: pre fires before post → strengthen."""

    def test_causal_strengthening(self):
        """PRD §9: A→B strengthens when A fires before B."""
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "max_weight": 5.0,
            "learning_rate": 0.01,
            "co_activation_window": 0,  # Disable sprouting for this test
            "grace_period": 100000,
            "inactivity_threshold": 100000,
        })
        g.create_node("A")
        g.create_node("B")
        syn = g.create_synapse("A", "B", weight=1.0, delay=1)

        initial_weight = syn.weight

        # Fire A, then B (causal order)
        g.stimulate("A", 2.0)
        g.step()  # t=1: A fires

        g.stimulate("B", 2.0)
        g.step()  # t=2: B fires, STDP should strengthen A→B

        assert syn.weight > initial_weight, (
            f"Causal pair should strengthen: {syn.weight} vs {initial_weight}"
        )

    def test_repeated_causal_strengthening(self):
        """Repeated causal pairing should progressively strengthen."""
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 0.99,
            "max_weight": 5.0,
            "learning_rate": 0.01,
            "tau_plus": 20.0,
            "tau_minus": 20.0,
            "co_activation_window": 0,
            "grace_period": 100000,
            "inactivity_threshold": 100000,
        })
        g.create_node("A")
        g.create_node("B")
        syn = g.create_synapse("A", "B", weight=0.5, delay=1)

        weights = [syn.weight]
        for _ in range(20):
            g.stimulate("A", 2.0)
            g.step()  # A fires
            g.stimulate("B", 2.0)
            g.step()  # B fires (dt=1, strong LTP)
            # Long cooldown so previous round's B spike decays away before
            # next A fires (prevents LTD from outgoing pass).
            g.step_n(60)
            weights.append(syn.weight)

        assert weights[-1] > weights[0], (
            f"Weight should increase over training: {weights[0]} → {weights[-1]}"
        )


class TestLTD:
    """Long-Term Depression: pre fires after post → weaken."""

    def test_acausal_weakening(self):
        """PRD §9: A→B weakens when A fires after B."""
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "max_weight": 5.0,
            "learning_rate": 0.01,
            "co_activation_window": 0,
            "grace_period": 100000,
            "inactivity_threshold": 100000,
        })
        g.create_node("A")
        g.create_node("B")
        syn = g.create_synapse("A", "B", weight=1.0, delay=1)

        initial_weight = syn.weight

        # Fire B first, then A (acausal order for A→B synapse)
        g.stimulate("B", 2.0)
        g.step()  # t=1: B fires

        g.stimulate("A", 2.0)
        g.step()  # t=2: A fires, STDP should weaken A→B

        assert syn.weight < initial_weight, (
            f"Acausal pair should weaken: {syn.weight} vs {initial_weight}"
        )

    def test_repeated_acausal_weakening(self):
        """Repeated acausal pairing should progressively weaken."""
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 0.99,
            "max_weight": 5.0,
            "learning_rate": 0.01,
            "tau_plus": 20.0,
            "tau_minus": 20.0,
            "co_activation_window": 0,
            "grace_period": 100000,
            "inactivity_threshold": 100000,
        })
        g.create_node("A")
        g.create_node("B")
        syn = g.create_synapse("A", "B", weight=2.0, delay=1)

        weights = [syn.weight]
        for _ in range(20):
            # B fires first (acausal for A→B synapse)
            g.stimulate("B", 2.0)
            g.step()
            # Then A fires 1 step later
            g.stimulate("A", 2.0)
            g.step()
            # Long cooldown to avoid inter-round interference
            g.step_n(60)
            weights.append(syn.weight)

        assert weights[-1] < weights[0], (
            f"Weight should decrease over acausal training: {weights[0]} → {weights[-1]}"
        )


class TestWeightDependentSTDP:
    """Weight-dependent scaling prevents runaway potentiation (PRD §3.1.2)."""

    def test_strong_connections_learn_slower(self):
        """LTP scaled by (max_weight - w) / max_weight (soft saturation)."""
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "max_weight": 5.0,
            "learning_rate": 0.01,
            "co_activation_window": 0,
            "grace_period": 100000,
            "inactivity_threshold": 100000,
        })
        g.create_node("A")
        g.create_node("B")
        g.create_node("C")
        g.create_node("D")

        # Weak connection
        syn_weak = g.create_synapse("A", "B", weight=0.5, delay=1)
        # Strong connection
        syn_strong = g.create_synapse("C", "D", weight=4.5, delay=1)

        # Same causal pairing for both
        g.stimulate("A", 2.0)
        g.stimulate("C", 2.0)
        g.step()  # A,C fire

        g.stimulate("B", 2.0)
        g.stimulate("D", 2.0)
        g.step()  # B,D fire

        dw_weak = syn_weak.weight - 0.5
        dw_strong = syn_strong.weight - 4.5

        assert dw_weak > dw_strong, (
            f"Weak synapse should learn more: dw_weak={dw_weak}, dw_strong={dw_strong}"
        )

    def test_weight_stays_below_max(self):
        """Weight should never exceed max_weight."""
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 0.99,
            "max_weight": 5.0,
            "learning_rate": 0.05,  # Fast learning
            "co_activation_window": 0,
            "grace_period": 100000,
            "inactivity_threshold": 100000,
        })
        g.create_node("A")
        g.create_node("B")
        syn = g.create_synapse("A", "B", weight=4.9, delay=1)

        for _ in range(100):
            g.stimulate("A", 2.0)
            g.step()
            g.step()
            g.step()
            g.stimulate("B", 2.0)
            g.step()
            g.step()
            g.step()

        assert syn.weight <= syn.max_weight, (
            f"Weight {syn.weight} exceeds max {syn.max_weight}"
        )

    def test_weight_stays_non_negative(self):
        """Weight should never go below 0."""
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 0.99,
            "max_weight": 5.0,
            "learning_rate": 0.05,
            "co_activation_window": 0,
            "grace_period": 100000,
            "inactivity_threshold": 100000,
        })
        g.create_node("A")
        g.create_node("B")
        syn = g.create_synapse("A", "B", weight=0.05, delay=1)

        for _ in range(100):
            # Acausal: weaken
            g.stimulate("B", 2.0)
            g.step()
            g.step()
            g.step()
            g.stimulate("A", 2.0)
            g.step()
            g.step()
            g.step()

        assert syn.weight >= 0.0, f"Weight went negative: {syn.weight}"


class TestTemporalAliasing:
    """Δt=0 should be treated as weak LTP (PRD §3.1.2)."""

    def test_simultaneous_spikes_weak_ltp(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "max_weight": 5.0,
            "learning_rate": 0.01,
            "co_activation_window": 0,
            "grace_period": 100000,
            "inactivity_threshold": 100000,
        })
        g.create_node("A")
        g.create_node("B")
        syn = g.create_synapse("A", "B", weight=1.0, delay=1)

        initial_weight = syn.weight

        # Both fire simultaneously
        g.stimulate("A", 2.0)
        g.stimulate("B", 2.0)
        g.step()

        # Should get weak LTP (Δt=0 case)
        assert syn.weight >= initial_weight, (
            f"Simultaneous firing should give weak LTP: {syn.weight} vs {initial_weight}"
        )


class TestAsymmetry:
    """A_minus > A_plus ensures net weakening bias for stability (PRD §3.1.1)."""

    def test_asymmetry_default(self):
        g = Graph()
        stdp = g._plasticity_rules[0]
        assert isinstance(stdp, STDPRule)
        assert stdp.A_minus > stdp.A_plus, (
            f"A_minus ({stdp.A_minus}) must be > A_plus ({stdp.A_plus})"
        )

    def test_ltd_stronger_than_ltp_symmetric_timing(self):
        """With symmetric timing, LTD should outweigh LTP due to A_minus > A_plus.

        We test with a midpoint weight so weight-dependent scaling doesn't
        dominate the comparison.
        """
        # Test LTP: create fresh graph, A before B
        g1 = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "max_weight": 5.0,
            "learning_rate": 0.01,
            "A_plus": 1.0,
            "A_minus": 1.2,
            "co_activation_window": 0,
            "grace_period": 100000,
            "inactivity_threshold": 100000,
        })
        g1.create_node("A")
        g1.create_node("B")
        syn1 = g1.create_synapse("A", "B", weight=2.5, delay=1)

        g1.stimulate("A", 2.0)
        g1.step()  # A fires
        g1.stimulate("B", 2.0)
        g1.step()  # B fires, dt=1 → LTP
        ltp_delta = syn1.weight - 2.5

        # Test LTD: fresh graph, B before A (same dt magnitude)
        g2 = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "max_weight": 5.0,
            "learning_rate": 0.01,
            "A_plus": 1.0,
            "A_minus": 1.2,
            "co_activation_window": 0,
            "grace_period": 100000,
            "inactivity_threshold": 100000,
        })
        g2.create_node("A")
        g2.create_node("B")
        syn2 = g2.create_synapse("A", "B", weight=2.5, delay=1)

        g2.stimulate("B", 2.0)
        g2.step()  # B fires
        g2.stimulate("A", 2.0)
        g2.step()  # A fires, dt=-1 → LTD
        ltd_delta = 2.5 - syn2.weight  # positive = weakened

        # LTD magnitude should exceed LTP (A_minus/A_plus = 1.2)
        # Note: LTP has weight-dependent scaling reducing it, so raw LTD
        # (which has no such scaling) should clearly exceed.
        assert ltd_delta > ltp_delta, (
            f"LTD ({ltd_delta:.6f}) should exceed LTP ({ltp_delta:.6f}) "
            f"due to A_minus > A_plus"
        )
