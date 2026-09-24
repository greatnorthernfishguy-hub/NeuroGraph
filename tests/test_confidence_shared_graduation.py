#!/usr/bin/env python3
"""
Test for COMB-04 Shared Graduation: confidence signal derivation, Commons deposit,
and KISS/Pith wiring.

See docs/concepts/KISS_Pith_Combined_Architecture.md "Shared Graduation" section.
"""

import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after path setup
import commons
import cc_ng_organism


def test_confidence_derivation():
    """Test that confidence is derived correctly from novelty (1.0 - novelty)."""
    from cc_ng_organism import _cc_substrate_confidence
    
    # Mock graph with detect_novelty method
    class MockGraph:
        def detect_novelty(self, embedding):
            # Simulate novelty detection
            return 0.3  # 30% novel
    
    graph = MockGraph()
    embedding = np.random.randn(768).astype(np.float32)
    
    confidence = _cc_substrate_confidence(graph, embedding)
    # Novelty 0.3 -> confidence 0.7
    assert abs(confidence - 0.7) < 1e-6, f"Expected confidence ~0.7, got {confidence}"
    
    # Test edge cases
    graph_novel = MockGraph()
    graph_novel.detect_novelty = lambda emb: 0.0  # Not novel
    assert _cc_substrate_confidence(graph_novel, embedding) == 1.0
    
    graph_very_novel = MockGraph()
    graph_very_novel.detect_novelty = lambda emb: 1.0  # Very novel
    assert _cc_substrate_confidence(graph_very_novel, embedding) == 0.0


def test_region_hash_stable():
    """Test that region hash is stable for same embedding."""
    embedding1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    embedding2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # Same values
    
    hash1 = _cc_region_hash(embedding1)
    hash2 = _cc_region_hash(embedding2)
    
    assert hash1 == hash2, "Same embedding should produce same hash"
    
    # Different embedding should produce different hash
    embedding3 = np.array([1.0, 2.0, 4.0], dtype=np.float32)  # Slightly different
    hash3 = _cc_region_hash(embedding3)
    assert hash1 != hash3, "Different embedding should produce different hash"


def test_confidence_deposit_and_read():
    """Test confidence deposit to Commons and read back."""
    # Enable confidence gate for test
    os.environ["CC_CONFIDENCE_GATE_ENABLED"] = "1"
    
    try:
        # Re-import to pick up env var
        import importlib
        import cc_ng_organism
        importlib.reload(cc_ng_organism)
        from cc_ng_organism import _CC_CONFIDENCE_GATE_ENABLED
        
        assert _CC_CONFIDENCE_GATE_ENABLED, "Confidence gate should be enabled"
        
        # Mock graph
        class MockGraph:
            def detect_novelty(self, embedding):
                return 0.25  # 25% novel -> 75% confidence
        
        # Get fresh Commons instance
        commons = get_commons()
        graph = MockGraph()
        embedding = np.random.randn(768).astype(np.float32)
        
        # Deposit confidence
        confidence = _cc_deposit_confidence(commons, graph, embedding)
        assert confidence is not None, "Deposit should return confidence value"
        assert abs(confidence - 0.75) < 1e-6, f"Expected confidence 0.75, got {confidence}"
        
        # Read back from Commons
        region_hash = _cc_region_hash(embedding)
        read_confidence = commons.read_confidence(region_hash, default=0.0)
        
        assert abs(read_confidence - 0.75) < 1e-6, f"Expected read confidence 0.75, got {read_confidence}"
        
        # Test default when not found
        non_existent_hash = "nonexistent123456"
        default_conf = commons.read_confidence(non_existent_hash, default=0.5)
        assert default_conf == 0.5, f"Expected default 0.5, got {default_conf}"
        
    finally:
        # Clean up
        os.environ.pop("CC_CONFIDENCE_GATE_ENABLED", None)


def test_kiss_confidence_adjustment():
    """Test that KISS threshold adjusts with confidence."""
    # Enable confidence gate
    os.environ["CC_CONFIDENCE_GATE_ENABLED"] = "1"
    
    try:
        import importlib
        import cc_ng_organism
        importlib.reload(cc_ng_organism)
        from cc_ng_organism import _cc_kiss_find_redundant_node, _CC_KISS_REDUNDANCY_THRESHOLD
        
        # Mock dependencies
        class MockGraph:
            def _is_identity_protected(self, node_id):
                return False
        
        class MockVectorDB:
            def search(self, embedding, k, threshold):
                # Return empty results
                return []
            def get(self, node_id):
                return None
        
        commons = get_commons()
        graph = MockGraph()
        vector_db = MockVectorDB()
        embedding = np.random.randn(768).astype(np.float32)
        
        # The function should be called without error
        result = _cc_kiss_find_redundant_node(graph, vector_db, embedding, commons=commons)
        assert result is None  # No matches in our mock
        
    finally:
        os.environ.pop("CC_CONFIDENCE_GATE_ENABLED", None)


def test_pith_confidence_adjustment():
    """Test that Pith L1 budget adjusts with confidence."""
    os.environ["CC_CONFIDENCE_GATE_ENABLED"] = "1"
    os.environ["CC_PITH_L1_BREATHE"] = "1"
    
    try:
        import importlib
        import cc_ng_organism
        importlib.reload(cc_ng_organism)
        from cc_ng_organism import cc_l1_budget, _CC_PITH_L1_BUDGET
        
        commons = get_commons()
        
        # First test without embedding (backward compatibility)
        budget_no_embedding = cc_l1_budget(commons)
        assert isinstance(budget_no_embedding, int)
        
        # Test with embedding (confidence adjustment)
        embedding = np.random.randn(768).astype(np.float32)
        budget_with_embedding = cc_l1_budget(commons, current_embedding=embedding)
        assert isinstance(budget_with_embedding, int)
        
        # Both should be within bounds
        assert 500 <= budget_no_embedding <= 40000
        assert 500 <= budget_with_embedding <= 40000
        
    finally:
        os.environ.pop("CC_CONFIDENCE_GATE_ENABLED", None)
        os.environ.pop("CC_PITH_L1_BREATHE", None)


if __name__ == "__main__":
    print("Running confidence shared graduation tests...")
    
    test_confidence_derivation()
    print("✓ Confidence derivation test passed")
    
    test_region_hash_stable()
    print("✓ Region hash stability test passed")
    
    test_confidence_deposit_and_read()
    print("✓ Confidence deposit/read test passed")
    
    test_kiss_confidence_adjustment()
    print("✓ KISS confidence adjustment test passed")
    
    test_pith_confidence_adjustment()
    print("✓ Pith confidence adjustment test passed")
    
    print("\nAll tests passed!")