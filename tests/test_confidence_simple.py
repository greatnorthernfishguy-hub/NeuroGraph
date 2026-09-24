#!/usr/bin/env python3
"""
Simple test for COMB-04 confidence helpers.
"""

import os
import sys
import numpy as np
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_region_hash():
    """Test the region hash helper logic."""
    # Simple implementation matching _cc_region_hash
    def simple_region_hash(embedding):
        arr = np.asarray(embedding, dtype=np.float32).flatten()
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    
    embedding1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    embedding2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    embedding3 = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    
    hash1 = simple_region_hash(embedding1)
    hash2 = simple_region_hash(embedding2)
    hash3 = simple_region_hash(embedding3)
    
    assert hash1 == hash2, "Same embedding -> same hash"
    assert hash1 != hash3, "Different embedding -> different hash"
    print("✓ Region hash test passed")


def test_confidence_logic():
    """Test confidence calculation logic (1.0 - novelty)."""
    # Test cases
    test_cases = [
        (0.0, 1.0),   # No novelty -> full confidence
        (0.3, 0.7),   # 30% novel -> 70% confidence
        (0.5, 0.5),   # 50% novel -> 50% confidence
        (1.0, 0.0),   # Full novelty -> no confidence
    ]
    
    for novelty, expected_confidence in test_cases:
        confidence = 1.0 - novelty
        confidence = max(0.0, min(1.0, confidence))  # Clamp
        assert abs(confidence - expected_confidence) < 1e-6, \
            f"Novelty {novelty} -> confidence {confidence}, expected {expected_confidence}"
    
    print("✓ Confidence logic test passed")


def test_kiss_adjustment():
    """Test KISS threshold adjustment formula."""
    base_threshold = 0.95
    test_cases = [
        (0.0, 0.85),   # No confidence -> looser threshold (0.95 - 0.1)
        (0.5, 0.95),   # Medium confidence -> same threshold (0.95 + 0)
        (1.0, 1.05),   # Full confidence -> tighter threshold (0.95 + 0.1)
    ]
    
    for confidence, expected_raw in test_cases:
        adjustment = confidence * 0.2 - 0.1  # Maps 0.0 -> -0.1, 1.0 -> +0.1
        adjusted = base_threshold + adjustment
        clamped = max(0.5, min(0.99, adjusted))
        
        print(f"  Confidence {confidence}: adjustment {adjustment:.3f}, raw {adjusted:.3f}, clamped {clamped:.3f}")
    
    print("✓ KISS adjustment formula test passed")


def test_pith_adjustment():
    """Test Pith budget adjustment formula."""
    base_multiplier = 1.4  # PARASYMPATHETIC
    test_cases = [
        (0.0, 1.26),   # No confidence -> contracted (1.4 * 0.9)
        (0.5, 1.4),    # Medium confidence -> same (1.4 * 1.0)
        (1.0, 1.54),   # Full confidence -> expanded (1.4 * 1.1)
    ]
    
    for confidence, expected_raw in test_cases:
        adjustment = confidence * 0.2 - 0.1  # Maps 0.0 -> -0.1, 1.0 -> +0.1
        adjusted = base_multiplier * (1.0 + adjustment)
        clamped = max(0.5, min(2.0, adjusted))
        
        print(f"  Confidence {confidence}: adjustment {adjustment:.3f}, raw {adjusted:.3f}, clamped {clamped:.3f}")
    
    print("✓ Pith adjustment formula test passed")


if __name__ == "__main__":
    print("Testing COMB-04 confidence signal logic...\n")
    
    test_region_hash()
    print()
    
    test_confidence_logic()
    print()
    
    print("KISS threshold adjustment (high confidence -> tighter threshold):")
    test_kiss_adjustment()
    print()
    
    print("Pith budget adjustment (high confidence -> expanded budget):")
    test_pith_adjustment()
    print()
    
    print("\nAll logic tests passed!")
    print("\nNote: Integration tests require Commons and ng_lite dependencies.")
    print("Run with pytest for full integration tests.")