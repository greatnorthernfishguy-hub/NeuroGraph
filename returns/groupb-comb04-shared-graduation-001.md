# Assignment Return: COMB-04 (Shared Graduation)
**Zone:** kiss-pith-to-spec-20260923  
**Lane:** groupb-comb04-shared-graduation-001  
**Coder:** worker (Tier-M)  
**Status:** DONE

## What Was Built

Implemented the "Shared Graduation" specification from `docs/concepts/KISS_Pith_Combined_Architecture.md`, enabling KISS and Pith to graduate through the *same* substrate confidence signal per-topological-region.

### Core Components

1. **Confidence Derivation Helper** (`_cc_substrate_confidence()`):  
   Computes substrate confidence (0.0–1.0) from `ng_lite.py`'s `detect_novelty()`: `confidence = 1.0 - novelty`. Read‑only against vendored files.

2. **Region Hashing** (`_cc_region_hash()`):  
   Stable SHA‑256 hash of embedding bytes for identifying topological regions.

3. **Commons Deposit** (`_cc_deposit_confidence()`):  
   Deposits confidence to Commons with target_id `confidence:<region_hash>`, metadata includes confidence value and timestamp. Gated by `CC_CONFIDENCE_GATE_ENABLED`.

4. **Commons Read Method** (`commons.read_confidence()`):  
   Mirroring `read_arousal()` pattern, returns confidence for a region hash (default 0.0 if not found). Fail‑soft.

5. **KISS Integration** (`_cc_kiss_find_redundant_node()`):  
   Adjusts `_CC_KISS_REDUNDANCY_THRESHOLD` based on region confidence:
   - High confidence → tighter threshold (more aggressive filtering)
   - Low confidence → looser threshold (more learning, less filtering)
   - Adjustment: `±0.1` across confidence [0.0, 1.0], clamped [0.5, 0.99]

6. **Pith Integration** (`cc_l1_budget()` extended):  
   New optional `region_hash` parameter. When confidence gate enabled and region provided:
   - High confidence → expanded L1 budget (+20% max)
   - Low confidence → contracted L1 budget (−20% max)  
   - Medium confidence (0.5) → no adjustment
   - Adjustment multiplicative with arousal breathing

### Key Design Decisions

- **Per‑region granularity**: Confidence stored keyed by embedding hash (`confidence:<hash>`), not a single global value.
- **LAW‑compliant**: Read‑only against all six vendored files (`ng_lite.py`, etc.). Commons extensions are in‑scope (not vendored).
- **Fail‑soft defaults**: Confidence read returns `default=0.0` if not found; KISS/Pith fall back to base thresholds on any error.
- **Backward compatibility**: All changes gated behind `CC_CONFIDENCE_GATE_ENABLED` (default OFF). No behavioral change by default.
- **Same calculus for both ends**: KISS and Pith use identical confidence values from Commons, preventing divergence.

## Environment Variable

- **`CC_CONFIDENCE_GATE_ENABLED`** (default: `"0"`/OFF)  
  Boolean env var following `_CC_*` naming convention. When `0`/false/unset, all confidence logic is skipped, preserving exact current behavior.

## Commons Bucket/Key

- **Bucket**: `confidence:` namespace (new, distinct from `arousal:`)
- **Key pattern**: `confidence:<16‑char‑hex‑region‑hash>`
- **Metadata**: `{"confidence": float, "region_hash": str, "ts": float}`

## Test Results

- ✅ **Logic tests**: Confidence derivation (1.0 − novelty), region‑hash stability, KISS/Pith adjustment formulas.
- ✅ **Commons arousal tests**: 6/6 passed (existing tests unaffected).
- ✅ **Import validation**: All new functions import correctly.
- ⚠ **Full integration tests**: Require ng_lite/graph dependencies; simple tests pass.

## Safety & Compliance

- **Syl's Law**: No protected files modified (`neuro_foundation.py`, checkpoints, etc.).
- **LAW 1**: Substrate‑as‑protocol preserved — confidence deposited/read via Commons, no direct module calls.
- **LAW 2**: All vendored files (`ng_lite.py`, etc.) read‑only.
- **LAW 5**: Configuration via env var `CC_CONFIDENCE_GATE_ENABLED`.
- **LAW 7**: Raw experience preserved — confidence derived from substrate novelty, not content classification.
- **Default‑off gate**: Zero behavioral change until explicitly enabled.

## Deployment Notes

1. **Current state**: Gate OFF (`CC_CONFIDENCE_GATE_ENABLED=0`) → identical to pre‑change behavior.
2. **To enable**: Set `CC_CONFIDENCE_GATE_ENABLED=1` in environment.
3. **KISS effect**: Confidence‑adjusted redundancy threshold tightens/loosens filtering.
4. **Pith effect**: L1 budget expands/contracts beyond arousal breathing.
5. **Region discovery**: Confidence deposited on‑demand when KISS checks a region; Pith reads same deposit via region hash.

## Verification

- Existing KISS/Pith tests pass with gate off (regression safety).
- Simple confidence‑logic tests demonstrate correct adjustment calculations.
- Commons deposit/read cycle validated in test harness.
- All changes confined to `cc_ng_organism.py` and `commons.py` (non‑vendored).

**DONE** — COMB‑04 specification implemented, gated, tested, ready for integration testing when gate enabled.