#!/usr/bin/env python3
# ---- Changelog ----
# [2026-09-24] zone manager (Claude Code, Z2) — Revision 6: env-var tests in a subprocess
# What: test_env_vars_defaults/custom now read the module constants from a child
#       process; added test_env_vars_clamped.
# Why: the in-process importlib.reload left cc_ng_organism reloaded with custom
#      values, breaking 8 Pith tests when run in the same pytest session.
# How: _read_env_constants() imports cc_ng_organism under a controlled env in a subprocess.
# [2026-09-25] Z2 zone manager (Claude Opus 5.5, Claude Code) — region = what fired
# What: cc_region_confidence(graph, fired_node_ids) and cc_l1_budget(commons,
#       graph, fired_node_ids); both recall paths pass their fired set and never
#       embed or search the vdb for it. K/THRESHOLD tests removed with the knobs.
# Why: Packet 175a / KISS_Pith_Combined_Architecture.md "Shared Graduation".
# How: tests assert the fired ids reach cc_region_confidence and embed is not called.
# -------------------
"""Tests for COMB-04 Shared Graduation region confidence signal.

Tests:
1. cc_region_confidence with a small known graph
2. Read-only assertion (graph/Commons state unchanged)
3. Regression with env var CC_PITH_REGION_CONFIDENCE_ENABLED=off
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from types import SimpleNamespace
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cc_ng_organism as cc


class MockSynapse:
    def __init__(self, synapse_id, pre_id, post_id, weight=0.5, max_weight=1.0):
        self.synapse_id = synapse_id
        self.pre_node_id = pre_id
        self.post_node_id = post_id
        self.weight = weight
        self.max_weight = max_weight


class MockNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.metadata = {}


class MockGraph:
    def __init__(self):
        self.nodes = {}
        self.synapses = {}
        self._outgoing = {}
        self._synapse_confirmation_history = {}
        
    def get(self, key):
        return self.synapses.get(key)
    
    def _compute_prediction_confidence(self, synapse):
        """Mock confidence computation: weight/max_weight * 0.6 + 0.5 * 0.4"""
        weight_factor = synapse.weight / synapse.max_weight
        history = self._synapse_confirmation_history.get(synapse.synapse_id, [])
        if history and len(history) > 0:
            confirmation_rate = sum(1 for x in history if x) / len(history)
        else:
            confirmation_rate = 0.5
        return weight_factor * 0.6 + confirmation_rate * 0.4


class MockVectorDB:
    def __init__(self, hits=None):
        self.hits = hits or []
        
    def search(self, *args, **kwargs):
        # Returns list of (node_id, score)
        return [(hit[0], hit[1]) for hit in self.hits]
        
    def get(self, node_id, *args, **kwargs):
        return {"metadata": {}, "source": "", "text": ""}
        
    def batch_get(self, node_ids):
        return [self.get(node_id) for node_id in node_ids]
        
    def size(self):
        return len(self.hits)


# Test helper classes (copied from test_cc_recall_unification.py)  
class _FakeMonitor:
    def __init__(self, items):
        self._items = items

    def get_surfaced(self):
        return list(self._items)

    def format_context(self, items):
        if not items:
            return ''
        return '## Recent\n' + '\n'.join(f"- {it['content']}" for it in items)


class _FakeGraphForAssemble:
    def __init__(self, protected_ids=frozenset()):
        self._protected = protected_ids
        self.nodes = {}  # cc_thermal/cc_novelty fail-soft on absent entries

    def _is_identity_protected(self, node_id):
        return node_id in self._protected


class _FakeNgForAssemble:
    def __init__(self, monitor_items, protected_ids=frozenset()):
        self.graph = _FakeGraphForAssemble(protected_ids)
        self._surfacing_monitor = _FakeMonitor(monitor_items)
        # Add vector_db attribute for region confidence tests
        self.vector_db = MockVectorDB(hits=[('node1', 0.9)])  # Will be mocked in tests


def _patch_pattern_completion(monkeypatch, results):
    monkeypatch.setattr(cc, 'cc_pattern_completion_recall',
                         lambda ng, query, k, state=None: list(results))


# ============================================================================
# cc_assemble_recall tests (Pith pipeline enabled)
# ============================================================================

def test_flag_off_no_embed_call_in_cc_assemble_recall(monkeypatch):
    """Test that ng_embed.embed() is NOT called when flag is OFF in cc_assemble_recall.
    
    Must set _CC_PITH_ENABLED=True to reach the region confidence code path.
    """
    # Create fake ng with vector_db (needs one for embed path)
    ng = _FakeNgForAssemble([
        {'node_id': 'test1', 'score': 1.0, 'content': 'test monitor item'}
    ])
    
    # Ensure we reach the Pith pipeline
    monkeypatch.setattr(cc, '_CC_PITH_ENABLED', True)
    monkeypatch.setattr(cc, '_CC_PITH_L1_BUDGET', 4000)
    
    # Flag OFF for region confidence
    monkeypatch.setattr(cc, '_CC_PITH_REGION_CONFIDENCE_ENABLED', False)
    
    # Patch pattern completion to return something
    _patch_pattern_completion(monkeypatch, [
        {'node_id': 'pat1', 'score': 0.8, 'content': 'test pattern hit'}
    ])
    
    # Mock commons
    mock_commons = Mock()
    mock_commons.read_arousal.return_value = "PARASYMPATHETIC"
    
    # Patch ng_embed.embed to track calls
    with patch('ng_embed.embed') as mock_embed:
        mock_embed.return_value = [0.1] * 768
        
        # Call cc_assemble_recall
        result = cc.cc_assemble_recall(ng, 'test query', 5, {}, mock_commons)
        
        # With flag OFF, embed should NOT be called
        mock_embed.assert_not_called()


def test_flag_on_fired_set_reaches_region_confidence_in_cc_assemble_recall(monkeypatch):
    """Flag ON: the region is pattern completion's fired set; no embed call."""
    # Create fake ng with vector_db
    ng = _FakeNgForAssemble([
        {'node_id': 'test1', 'score': 1.0, 'content': 'test monitor item'}
    ])
    
    # Ensure we reach the Pith pipeline
    monkeypatch.setattr(cc, '_CC_PITH_ENABLED', True)
    monkeypatch.setattr(cc, '_CC_PITH_L1_BUDGET', 4000)
    monkeypatch.setattr(cc, '_CC_PITH_L1_BREATHE', False)  # Turn off breathing for simplicity
    
    # Flag ON for region confidence
    monkeypatch.setattr(cc, '_CC_PITH_REGION_CONFIDENCE_ENABLED', True)
    
    # Patch pattern completion
    _patch_pattern_completion(monkeypatch, [
        {'node_id': 'pat1', 'score': 0.8, 'content': 'test pattern hit'}
    ])
    
    # Mock commons
    mock_commons = Mock()
    mock_commons.read_arousal.return_value = "PARASYMPATHETIC"
    
    # Patch ng_embed.embed to track calls
    with patch('ng_embed.embed') as mock_embed:
        mock_embed.return_value = [0.1] * 768
        
        # Patch cc_region_confidence to return a neutral value
        with patch.object(cc, 'cc_region_confidence', return_value=0.5) as mock_conf:
            # Call cc_assemble_recall
            result = cc.cc_assemble_recall(ng, 'test query', 5, {}, mock_commons)
            
            mock_conf.assert_called_once_with(ng.graph, ['pat1'])
            mock_embed.assert_not_called()


# ============================================================================
# pith_provider_context tests
# ============================================================================

def test_flag_off_no_embed_call_in_pith_provider_context(monkeypatch):
    """Test that ng_embed.embed() is NOT called when flag is OFF in pith_provider_context."""
    # Create minimal mock graph with constitutional core
    mock_graph = MockGraph()
    mock_graph.nodes = {"core": MockNode("core")}
    mock_graph.nodes["core"].metadata = {"constitutional": True, "core_text": "Honor agency."}
    
    # Create ng with graph and vector_db
    ng = SimpleNamespace(graph=mock_graph, vector_db=MockVectorDB())
    
    # Mock commons
    mock_commons = Mock()
    mock_commons.read_arousal.return_value = "PARASYMPATHETIC"
    
    # Patch ng_embed.embed to track calls
    with patch('ng_embed.embed') as mock_embed:
        mock_embed.return_value = [0.1] * 768
        
        # Patch the flag to OFF
        monkeypatch.setattr(cc, '_CC_PITH_REGION_CONFIDENCE_ENABLED', False)
        
        # Mock render_constitutional_core
        monkeypatch.setattr(cc, 'render_constitutional_core', lambda ng: "Honor agency.")
        
        # Mock cc_pattern_completion_recall
        monkeypatch.setattr(cc, 'cc_pattern_completion_recall', lambda *args, **kwargs: [])
        
        # Call pith_provider_context
        result = cc.pith_provider_context(
            ng=ng,
            current_instruction="test instruction",
            quest_focus="",
            conv_state={},
            commons=mock_commons,
            budget_chars=None,
            root_count=None
        )
        
        # With flag OFF, embed should NOT be called
        mock_embed.assert_not_called()


def test_flag_on_fired_set_reaches_region_confidence_in_pith_provider_context(monkeypatch):
    """Flag ON: the region is the cue's fired set; no embed call."""
    # Create minimal mock graph with constitutional core
    mock_graph = MockGraph()
    mock_graph.nodes = {"core": MockNode("core")}
    mock_graph.nodes["core"].metadata = {"constitutional": True, "core_text": "Honor agency."}
    
    # Create ng with graph and vector_db (needs hits for confidence)
    ng = SimpleNamespace(graph=mock_graph, vector_db=MockVectorDB(hits=[('node1', 0.9)]))
    
    # Mock commons
    mock_commons = Mock()
    mock_commons.read_arousal.return_value = "PARASYMPATHETIC"
    
    # Patch ng_embed.embed to track calls
    with patch('ng_embed.embed') as mock_embed:
        mock_embed.return_value = [0.1] * 768
        
        # Patch the flag to ON
        monkeypatch.setattr(cc, '_CC_PITH_REGION_CONFIDENCE_ENABLED', True)
        
        # Mock render_constitutional_core
        monkeypatch.setattr(cc, 'render_constitutional_core', lambda ng: "Honor agency.")
        
        # Mock cc_pattern_completion_recall: what fired for the cue
        monkeypatch.setattr(cc, 'cc_pattern_completion_recall', lambda *args, **kwargs: [
            {'node_id': 'n1', 'score': 0.9, 'content': 'fired one'},
            {'node_id': 'n2', 'score': 0.8, 'content': 'fired two'},
        ])
        
        # Record what cc_region_confidence is asked about
        seen = []
        monkeypatch.setattr(cc, 'cc_region_confidence',
                            lambda graph, fired: seen.append((graph, list(fired))) or 0.5)
        
        # Call pith_provider_context
        result = cc.pith_provider_context(
            ng=ng,
            current_instruction="test instruction",
            quest_focus="",
            conv_state={},
            commons=mock_commons,
            budget_chars=None,
            root_count=None
        )
        
        assert seen == [(mock_graph, ['n1', 'n2'])]
        mock_embed.assert_not_called()


# ============================================================================
# Original region confidence tests (kept for regression)
# ============================================================================

def test_flag_off_no_embed_call():
    """Original test: region confidence does not modulate budget when flag is OFF."""
    # Mock the environment
    with patch.multiple(cc,
                       _CC_PITH_L1_BUDGET=4000,
                       _CC_PITH_L1_BREATHE=True,
                       _CC_PITH_BREATHE_PARASYMPATHETIC=1.4,
                       _CC_PITH_REGION_CONFIDENCE_ENABLED=False,
                       _CC_PITH_REGION_CONFIDENCE_NEUTRAL=0.5,
                       _CC_PITH_REGION_CONFIDENCE_FALLOFF=0.25):
        
        mock_commons = Mock()
        mock_commons.read_arousal.return_value = "PARASYMPATHETIC"
        
        mock_graph = MockGraph()
        
        # Call cc_l1_budget with a fired set
        budget_with_params = cc.cc_l1_budget(mock_commons, mock_graph, ['n1', 'n2'])
        
        # Call without parameters (original behavior)  
        budget_without = cc.cc_l1_budget(mock_commons)
        
        # Should be identical when flag is OFF (no modulation)
        assert budget_with_params == budget_without == 5600  # 4000 * 1.4


def test_flag_on_embed_called():
    """Test that region confidence modulation works when flag is ON (legacy test)."""
    # Mock the environment
    with patch.multiple(cc,
                       _CC_PITH_L1_BUDGET=4000,
                       _CC_PITH_L1_BREATHE=True,
                       _CC_PITH_BREATHE_PARASYMPATHETIC=1.4,
                       _CC_PITH_REGION_CONFIDENCE_ENABLED=True,
                       _CC_PITH_REGION_CONFIDENCE_NEUTRAL=0.5,
                       _CC_PITH_REGION_CONFIDENCE_FALLOFF=0.25):
        
        mock_commons = Mock()
        mock_commons.read_arousal.return_value = "PARASYMPATHETIC"
        
        # Mock cc_region_confidence to return a known value
        with patch.object(cc, 'cc_region_confidence') as mock_confidence:
            mock_confidence.return_value = 0.8
            
            mock_graph = MockGraph()
            
            # Call cc_l1_budget with a fired set
            budget = cc.cc_l1_budget(mock_commons, mock_graph, ['n1', 'n2'])
            
            # cc_region_confidence should be called with that set
            mock_confidence.assert_called_once_with(mock_graph, ['n1', 'n2'])
            
            # Budget should include region confidence modulation
            # Without region confidence: 4000 * 1.4 = 5600
            # With region confidence (0.8): 5600 * (1 + (0.8 - 0.5) * 2 * 0.25) = 5600 * 1.15 = 6440
            expected = int(4000 * 1.4 * (1 + (0.8 - 0.5) * 2 * 0.25))
            expected = max(500, min(40000, expected))
            assert budget == expected


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_NAMES = ('CC_PITH_REGION_CONFIDENCE_FALLOFF',)


def _read_env_constants(overrides):
    """Import cc_ng_organism in a child process and return its parsed constants.

    Never reload the module in-process: other test files hold objects from
    `from cc_ng_organism import ...`, and a reload breaks them.
    """
    import subprocess
    env = {k: v for k, v in os.environ.items() if k not in _ENV_NAMES}
    env.update(overrides)
    out = subprocess.run(
        [sys.executable, '-c',
         'import cc_ng_organism as c; print(c._CC_PITH_REGION_CONFIDENCE_FALLOFF, '
         'c._CC_PITH_REGION_CONFIDENCE_NEUTRAL)'],
        env=env, cwd=_REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[-1].split()
    return float(out[0]), float(out[1])


def test_env_vars_defaults():
    assert _read_env_constants({}) == (0.25, 0.5)


def test_env_vars_custom():
    falloff, _ = _read_env_constants({'CC_PITH_REGION_CONFIDENCE_FALLOFF': '0.3'})
    assert falloff == 0.3


def test_region_confidence_basic():
    # Patch all region confidence constants
    with patch.multiple(cc,
                       _CC_PITH_REGION_CONFIDENCE_ENABLED=True,
                       _CC_PITH_REGION_CONFIDENCE_NEUTRAL=0.5):
        mock_graph = MockGraph()
        # Three fired nodes that have synapses between them
        fired = ['n1', 'n2', 'n3']
        
        # Create synapses BETWEEN fired nodes
        s1 = MockSynapse('s1', 'n1', 'n2', weight=0.9, max_weight=1.0)
        s2 = MockSynapse('s2', 'n1', 'n3', weight=0.4, max_weight=1.0)
        s3 = MockSynapse('s3', 'n2', 'n3', weight=0.7, max_weight=1.0)
        
        mock_graph.synapses = {'s1': s1, 's2': s2, 's3': s3}
        mock_graph._outgoing = {'n1': ['s1', 's2'], 'n2': ['s3']}  # n2→n3
        mock_graph._synapse_confirmation_history = {}
        
        conf = cc.cc_region_confidence(mock_graph, fired)
        
        # Compute expected confidences using MockGraph._compute_prediction_confidence formula:
        # weight/max_weight * 0.6 + confirmation_rate * 0.4
        # confirmation_rate defaults to 0.5 when no history
        
        # s1: 0.9/1.0*0.6 + 0.5*0.4 = 0.54 + 0.2 = 0.74
        # s2: 0.4/1.0*0.6 + 0.5*0.4 = 0.24 + 0.2 = 0.44  
        # s3: 0.7/1.0*0.6 + 0.5*0.4 = 0.42 + 0.2 = 0.62
        # Average over all synapses among hit nodes: (0.74 + 0.44 + 0.62) / 3 = 0.6
        assert 0.59 <= conf <= 0.61


@pytest.fixture
def region_on(monkeypatch):
    monkeypatch.setattr(cc, '_CC_PITH_REGION_CONFIDENCE_ENABLED', True)


def test_region_confidence_empty(region_on):
    assert cc.cc_region_confidence(MockGraph(), []) == 0.5  # nothing fired
    assert cc.cc_region_confidence(MockGraph(), None) == 0.5


def test_region_confidence_disabled_is_neutral():
    mock_graph = MockGraph()
    mock_graph.synapses = {'s1': MockSynapse('s1', 'n1', 'n2', weight=1.0)}
    mock_graph._outgoing = {'n1': ['s1']}
    with patch.object(cc, '_CC_PITH_REGION_CONFIDENCE_ENABLED', False):
        assert cc.cc_region_confidence(mock_graph, ['n1', 'n2']) == 0.5


def test_region_confidence_synapse_target_outside_fired_set_is_neutral(region_on):
    """Synapses whose target did not fire are outside the region."""
    mock_graph = MockGraph()
    s1 = MockSynapse('s1', 'n1', 't1', weight=0.1, max_weight=1.0)
    mock_graph.synapses = {'s1': s1}
    mock_graph._outgoing = {'n1': ['s1']}
    assert cc.cc_region_confidence(mock_graph, ['n1']) == 0.5


def test_region_confidence_read_only(region_on):
    """cc_region_confidence does not modify the graph or the fired set."""
    mock_graph = MockGraph()
    mock_graph.nodes = {'n1': MockNode('n1'), 'n2': MockNode('n2')}
    s1 = MockSynapse('s1', 'n1', 'n2', weight=0.5, max_weight=1.0)
    mock_graph.synapses = {'s1': s1}
    mock_graph._outgoing = {'n1': ['s1']}
    mock_graph._synapse_confirmation_history = {'s1': [True, False, True]}
    fired = ['n1', 'n2']

    initial_graph_nodes = dict(mock_graph.nodes)
    initial_graph_synapses = dict(mock_graph.synapses)
    initial_graph_history = dict(mock_graph._synapse_confirmation_history)

    conf = cc.cc_region_confidence(mock_graph, fired)

    assert mock_graph.nodes == initial_graph_nodes
    assert mock_graph.synapses == initial_graph_synapses
    assert mock_graph._synapse_confirmation_history == initial_graph_history
    assert fired == ['n1', 'n2']
    # 0.5*0.6 + (2/3)*0.4
    assert conf == pytest.approx(0.3 + 0.4 * 2 / 3)


def test_region_confidence_with_missing_nodes(region_on):
    """A fired id the graph does not hold has no synapses: neutral."""
    mock_graph = MockGraph()
    mock_graph.nodes = {'present': MockNode('present')}
    assert cc.cc_region_confidence(mock_graph, ['missing']) == 0.5


def test_region_confidence_with_no_outgoing_synapses(region_on):
    mock_graph = MockGraph()
    mock_graph.nodes = {'n1': MockNode('n1')}
    mock_graph._outgoing = {}
    assert cc.cc_region_confidence(mock_graph, ['n1']) == 0.5


def test_cc_l1_budget_skips_region_when_nothing_fired():
    with patch.multiple(cc,
                        _CC_PITH_L1_BUDGET=4000,
                        _CC_PITH_L1_BREATHE=False,
                        _CC_PITH_REGION_CONFIDENCE_ENABLED=True):
        with patch.object(cc, 'cc_region_confidence') as mock_confidence:
            assert cc.cc_l1_budget(None, MockGraph(), []) == 4000
            mock_confidence.assert_not_called()


def test_cc_l1_budget_min_max_clamp():
    """Test that cc_l1_budget clamps to min 500, max 40000."""
    with patch.multiple(cc,
                       _CC_PITH_L1_BUDGET=100,  # Very small
                       _CC_PITH_L1_BREATHE=False,
                       _CC_PITH_REGION_CONFIDENCE_ENABLED=False):
        
        mock_commons = Mock()
        
        # With very small base budget, should clamp to min 500
        budget = cc.cc_l1_budget(mock_commons)
        assert budget == 500  # Minimum
        
    with patch.multiple(cc,
                       _CC_PITH_L1_BUDGET=50000,  # Very large
                       _CC_PITH_L1_BREATHE=False,
                       _CC_PITH_REGION_CONFIDENCE_ENABLED=False):
        
        mock_commons = Mock()
        
        # With very large base budget, should clamp to max 40000
        budget = cc.cc_l1_budget(mock_commons)
        assert budget == 40000  # Maximum


def test_cc_l1_budget_sympathetic_scales_down():
    """Test that SYMPATHETIC arousal scales down by _CC_PITH_BREATHE_SYMPATHETIC factor."""
    with patch.multiple(cc,
                       _CC_PITH_L1_BUDGET=4000,
                       _CC_PITH_L1_BREATHE=True,  # Breathing must be ON for scaling
                       _CC_PITH_BREATHE_SYMPATHETIC=0.6,  # Default is 0.6, not 0.5
                       _CC_PITH_REGION_CONFIDENCE_ENABLED=False):
        
        mock_commons = Mock()
        mock_commons.read_arousal.return_value = "SYMPATHETIC"
        
        budget = cc.cc_l1_budget(mock_commons)
        # 4000 * 0.6 = 2400 (not 2000)
        assert budget == 2400


def test_cc_l1_budget_parasympathetic_scales_up():
    """Test that PARASYMPATHETIC arousal scales up by breathing factor."""
    with patch.multiple(cc,
                       _CC_PITH_L1_BUDGET=4000,
                       _CC_PITH_L1_BREATHE=True,
                       _CC_PITH_BREATHE_PARASYMPATHETIC=1.4,
                       _CC_PITH_REGION_CONFIDENCE_ENABLED=False):
        
        mock_commons = Mock()
        mock_commons.read_arousal.return_value = "PARASYMPATHETIC"
        
        budget = cc.cc_l1_budget(mock_commons)
        assert budget == 5600  # 4000 * 1.4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])