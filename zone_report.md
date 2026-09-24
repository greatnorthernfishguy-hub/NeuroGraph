Zone manager T3 thread `0192bec5-4d8d-43f3-aea8-b4589ab2ebfc` (Quest session `kiss-pith-to-spec-20260923-zone`):

**COMB-04 Shared Graduation implementation complete.**

**Chief thread `376988eb-fbb2-4f3e-95a1-8ae77e63da31` notified.**

## Execution Summary

- **Task**: Implement shared substrate confidence signal for KISS/Pith per COMB-04 spec
- **Status**: **DONE** - Assignment fully executed per requirements
- **Worktree**: `groupb-comb04-shared-graduation-001`
- **Commit**: `COMB-04 Shared Graduation: confidence signal for KISS/Pith shared substrate graduation`

## Key Deliverables

1. **New env var**: `CC_CONFIDENCE_GATE_ENABLED` (default OFF, LAW 5 compliance)
2. **Confidence derivation**: `_cc_substrate_confidence()` from `ng_lite.detect_novelty()`
3. **Commons integration**: `commons.read_confidence()` + `confidence:<hash>` deposits
4. **KISS wiring**: Threshold adjusts ±0.1 based on region confidence
5. **Pith wiring**: L1 budget adjusts ±20% based on region confidence
6. **Tests**: Logic tests + integration scaffold
7. **Return report**: `returns/groupb-comb04-shared-graduation-001.md`

## Compliance Verified

- ✅ No vendored files modified (read-only against `ng_lite.py`)
- ✅ Default behavior unchanged (env var default OFF)
- ✅ LAW 5 followed (env vars source of truth)
- ✅ Commons pattern mirrored (new bucket/key)
- ✅ Both ends wired with same signal
- ✅ Tests written
- ✅ Changelog headers added
- ✅ No merge to `main`, no push, no live process touched

The pathological divergence case ("KISS thinks mature but Pith thinks novel") is now prevented by shared substrate confidence signal.

**Work committed locally. Ready for zone review.**