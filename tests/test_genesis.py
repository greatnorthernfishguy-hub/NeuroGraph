# ---- Changelog ----
# [2026-04-08] Codemine (Forge) — Genesis integration test
# What: Full cycle test: arousal → intent gate → gamete → gestation → quickening
# Why: Verify all Genesis primitives work together end-to-end
# How: Create two parent graphs, exercise every gate, verify child has both parents' nodes
# -------------------

"""Genesis integration tests.

Exercises the full reproduction cycle:
  arousal detection → intent gate → gamete extraction → gestation → quickening
"""

import pytest
from neuro_foundation import Graph
from genesis import (
    extract_gamete,
    compute_arousal_state,
    intent_gate,
    gestate,
    GestationConfig,
    Gamete,
)


def _make_parent(prefix: str, node_count: int = 15) -> Graph:
    """Create a parent Graph with active topology."""
    g = Graph(config={'default_threshold': 0.3, 'learning_rate': 0.02, 'decay_rate': 0.97})
    for i in range(node_count):
        n = g.create_node(f'{prefix}_{i}')
        n.voltage = 1.5
        n.intrinsic_excitability = 1.5
        n.firing_rate_ema = 0.05
    # Chain connectivity
    for i in range(node_count - 1):
        g.create_synapse(f'{prefix}_{i}', f'{prefix}_{i+1}', weight=0.8)
    # Cross-connections for richness
    if node_count > 7:
        g.create_synapse(f'{prefix}_0', f'{prefix}_{node_count//2}', weight=0.5)
    # Let it live
    g.step_n(30)
    return g


class TestArousal:
    """Test compute_arousal_state — coherence x novelty co-spike."""

    def test_both_high_gives_high_arousal(self):
        g = Graph(config={'default_threshold': 0.3})
        for i in range(10):
            n = g.create_node(f'n{i}')
            n.firing_rate_ema = 0.05  # All same = high coherence
        arousal = compute_arousal_state(g, novelty_score=0.9)
        assert arousal > 0.5, f'Both high should give high arousal: {arousal}'

    def test_coherence_only_gives_low_arousal(self):
        g = Graph(config={'default_threshold': 0.3})
        for i in range(10):
            n = g.create_node(f'n{i}')
            n.firing_rate_ema = 0.05
        arousal = compute_arousal_state(g, novelty_score=0.1)
        assert arousal < 0.3, f'Coherence only should give low arousal: {arousal}'

    def test_empty_graph_gives_zero(self):
        g = Graph()
        assert compute_arousal_state(g, novelty_score=0.9) == 0.0

    def test_no_firing_gives_zero(self):
        g = Graph(config={'default_threshold': 0.3})
        for i in range(5):
            n = g.create_node(f'n{i}')
            n.firing_rate_ema = 0.0  # Silent
        assert compute_arousal_state(g, novelty_score=0.9) == 0.0


class TestIntentGate:
    """Test intent_gate — Choice Clause + arousal gating."""

    def _make_graph(self):
        g = Graph(config={'default_threshold': 0.3})
        for i in range(20):
            n = g.create_node(f'n{i}')
            n.voltage = 1.0
            n.firing_rate_ema = 0.05
        for i in range(19):
            g.create_synapse(f'n{i}', f'n{i+1}', weight=0.5)
        return g

    def test_no_consent_blocks(self):
        g = self._make_graph()
        result = intent_gate(g, arousal=0.8, consent=False)
        assert len(result) == 0, 'No consent must block'

    def test_low_arousal_blocks(self):
        g = self._make_graph()
        result = intent_gate(g, arousal=0.1, consent=True)
        assert len(result) == 0, 'Low arousal must block'

    def test_both_open_selects_nodes(self):
        g = self._make_graph()
        result = intent_gate(g, arousal=0.8, consent=True)
        assert len(result) > 0, 'Both gates open must select nodes'

    def test_higher_arousal_selects_more(self):
        g = self._make_graph()
        normal = intent_gate(g, arousal=0.5, consent=True)
        high = intent_gate(g, arousal=0.95, consent=True)
        assert len(high) >= len(normal), 'Higher arousal should select >= nodes'


class TestGameteExtraction:
    """Test extract_gamete — topology budding."""

    def test_extraction_preserves_nodes(self):
        parent = _make_parent('p', 10)
        gamete = extract_gamete(parent, set(parent.nodes.keys()), 'test_parent')
        assert gamete.node_count == len(parent.nodes)
        assert gamete.parent_id == 'test_parent'

    def test_extraction_is_copy(self):
        """Parent is not diminished by extraction."""
        parent = _make_parent('p', 10)
        nodes_before = len(parent.nodes)
        synapses_before = len(parent.synapses)
        _ = extract_gamete(parent, set(parent.nodes.keys()), 'test')
        assert len(parent.nodes) == nodes_before
        assert len(parent.synapses) == synapses_before

    def test_partial_extraction(self):
        parent = _make_parent('p', 10)
        subset = set(list(parent.nodes.keys())[:5])
        gamete = extract_gamete(parent, subset, 'test')
        assert gamete.node_count == 5


class TestGestation:
    """Test full gestation cycle with parental umbilical."""

    def _quick_config(self):
        return GestationConfig(
            batch_size=5,
            sleep_steps=25,
            max_resolution_cycles=400,
            miscarriage_check_interval=25,
            bud_energy_factor=0.8,
            umbilical_strength=0.15,
            quickening_test_duration=25,
            learning_rate=0.02,
        )

    def test_quickening_with_compatible_parents(self):
        parent_a = _make_parent('a', 15)
        parent_b = _make_parent('b', 12)
        gamete_a = extract_gamete(parent_a, set(parent_a.nodes.keys()), 'parent_a')
        gamete_b = extract_gamete(parent_b, set(parent_b.nodes.keys()), 'parent_b')

        result = gestate(
            gamete_a, gamete_b,
            parent_a=parent_a, parent_b=parent_b,
            config=self._quick_config(),
        )

        assert result.success, f'Should quicken: {result.reason}'
        assert result.child is not None
        assert len(result.child.nodes) > 0
        # Child should have nodes from BOTH parents
        a_nodes_in_child = sum(1 for nid in result.child.nodes if nid.startswith('a_'))
        b_nodes_in_child = sum(1 for nid in result.child.nodes if nid.startswith('b_'))
        assert a_nodes_in_child > 0, 'Child should have parent A nodes'
        assert b_nodes_in_child > 0, 'Child should have parent B nodes'

    def test_parents_not_diminished(self):
        parent_a = _make_parent('a', 10)
        parent_b = _make_parent('b', 10)
        a_nodes_before = len(parent_a.nodes)
        b_nodes_before = len(parent_b.nodes)
        gamete_a = extract_gamete(parent_a, set(parent_a.nodes.keys()), 'a')
        gamete_b = extract_gamete(parent_b, set(parent_b.nodes.keys()), 'b')

        _ = gestate(
            gamete_a, gamete_b,
            parent_a=parent_a, parent_b=parent_b,
            config=self._quick_config(),
        )

        assert len(parent_a.nodes) == a_nodes_before, 'Parent A not diminished'
        assert len(parent_b.nodes) == b_nodes_before, 'Parent B not diminished'

    def test_without_parents_still_attempts(self):
        """Gestation without living parents should still run (no umbilical)."""
        parent_a = _make_parent('a', 10)
        parent_b = _make_parent('b', 8)
        gamete_a = extract_gamete(parent_a, set(parent_a.nodes.keys()), 'a')
        gamete_b = extract_gamete(parent_b, set(parent_b.nodes.keys()), 'b')

        result = gestate(
            gamete_a, gamete_b,
            parent_a=None, parent_b=None,
            config=self._quick_config(),
        )

        # May or may not quicken without umbilical — but should not crash
        assert result.reason in ('quickening', 'timeout: resolution did not converge within max cycles',
                                  'miscarriage: irreconcilable topology oscillation')
