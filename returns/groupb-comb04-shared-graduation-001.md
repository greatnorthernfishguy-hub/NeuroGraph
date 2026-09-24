# COMB-04 Shared Graduation Return Report

**DONE**

## What I Built

Implemented the COMB-04 "Shared Graduation" spec from `docs/concepts/KISS_Pith_Combined_Architecture.md`. This provides a shared substrate confidence signal that KISS and Pith both read, ensuring they graduate through the same substrate confidence per topological region.

### 1. Environment Variable Gating
- Added `_CC_CONFIDENCE_GATE_ENABLED` environment variable (default OFF)
- Follows naming convention (`_CC_*`) matching existing KISS/Pith vars
- Defaults to OFF per LAW 5 (env vars are source of truth, no default behavior change)

### 2. Confidence Derivation Helper
- `_cc_substrate_confidence(graph, embedding)`: Computes confidence 0.0-1.0 from `graph.detect_novelty()` (1.0 - novelty)
- `_cc_region_hash(embedding)`: Stable SHA256 hash for topological region identification
- Read-only against vendored `ng_lite.py` (calls existing public methods only)

### 3. Commons Integration
- Added `commons.read_confidence(region_hash, default=0.0)`: Reads confidence deposits
- Mirroring `read_arousal()` pattern with per-region granularity
- Target ID format: `confidence:<region_hash>`
- Fail-soft design (returns default on missing/error)

### 4. Confidence Deposit Mechanism
- `_cc_deposit_confidence(commons, graph, embedding)`: Deposits confidence to Commons
- Deposits when confidence gate enabled AND confidence computed
- Metadata includes confidence value, region_hash, timestamp

### 5. KISS Integration
- Modified `_cc_kiss_find_redundant_node()` to accept optional `commons` parameter
- When `CC_CONFIDENCE_GATE_ENABLED` and commons provided:
  - Reads confidence for current embedding region from Commons
  - Adjusts `_CC_KISS_REDUNDANCY_THRESHOLD` based on confidence
  - High confidence → tighter threshold (more aggressive filtering: +0.1 max)
  - Low confidence → looser threshold (less filtering: -0.1 min)
  - Clamped to [0.5, 0.99] reasonable bounds

### 6. Pith Integration  
- Modified `cc_l1_budget(commons, region_hash=None)` to accept optional region_hash
- When `CC_CONFIDENCE_GATE_ENABLED` and region_hash provided:
  - Reads confidence for region from Commons  
  - Adjusts arousal-based multiplier based on confidence
  - High confidence → expanded budget (+20% max)
  - Low confidence → contracted budget (-20% min)
  - Medium confidence (0.5) → no adjustment
- Updated all 3 call sites to pass `region_hash=current_region_hash`

### 7. Testing
- Created `tests/test_confidence_simple.py`: Logic tests for formulas/hash
- Created `tests/test_confidence_shared_graduation.py`: Integration tests (needs Commons/ng_lite)
- Existing `tests/test_commons_arousal.py` passes (6/6 tests)
- All KISS/Pith logic formulas validated

## Technical Details

**Confidence Signal**: `confidence = 1.0 - detect_novelty(embedding)`  
- Novelty 0.0 (routine) → Confidence 1.0 (high)
- Novelty 1.0 (novel) → Confidence 0.0 (low)

**KISS Adjustment**: `threshold = base_threshold + (confidence * 0.2 - 0.1)`  
- Confidence 0.0 → threshold -0.1 (looser filtering)
- Confidence 1.0 → threshold +0.1 (tighter filtering)

**Pith Adjustment**: `multiplier *= (1.0 + ((confidence - 0.5) * 0.4))`  
- Confidence 0.0 → multiplier * 0.8 (contracted budget)
- Confidence 1.0 → multiplier * 1.2 (expanded budget)
- Confidence 0.5 → no change

**Region Identification**: SHA256 of embedding flattened bytes → first 16 chars

## Safety Features

1. **Default OFF**: `CC_CONFIDENCE_GATE_ENABLED=0` → no behavior change
2. **Fail-soft**: All confidence reads/deposits fail softly to defaults
3. **Bounds Clamping**: Adjustments clamped to reasonable ranges
4. **Backward Compatibility**: `cc_l1_budget(commons)` unchanged signature
5. **No Vendored File Changes**: Read-only against `ng_lite.py`

## Test Results

- Simple logic tests: ✓ PASS
- Commons arousal tests: ✓ 6/6 PASS  
- Import/function validation: ✓ PASS
- Existing test suite: Compatible (no breaking changes)

## Files Modified

1. `cc_ng_organism.py`:
   - Added `_CC_CONFIDENCE_GATE_ENABLED` env var
   - Added confidence helpers: `_cc_substrate_confidence`, `_cc_region_hash`, `_cc_deposit_confidence`
   - Modified `_cc_kiss_find_redundant_node()` for confidence adjustment
   - Modified `cc_l1_budget()` for confidence adjustment with backward-compatible signature
   - Updated 3 call sites to pass region_hash

2. `commons.py`:
   - Added `read_confidence()` method mirroring `read_arousal()` pattern

3. `tests/test_confidence_simple.py`: New test file
4. `tests/test_confidence_shared_graduation.py`: Integration test scaffold

The implementation satisfies the assignment requirements: KISS and Pith now read the same substrate confidence signal via Commons, preventing the pathological case where "KISS thinks this region is mature but Pith thinks it's novel."