---
name: finding-cc-authored-export-protection-collision
description: After _is_identity_protected is broadened to endswith("_authored"), the #70 corpus-callosum export whitelist's inclusion of cc_authored collides with its own "skip _is_identity_protected" rim rule
metadata:
  type: project
---

> ⚠️ **SUPERSEDED IN PART — READ [`docs/CC-CALLOSUM-TRUTH.md`](/home/josh/docs/CC-CALLOSUM-TRUTH.md) FIRST.**
> The consolidated, measurement-verified state of the callosum, the wholeness ring,
> hyperedge binding and orphan collection lives there (2026-07-31). The wholeness ring
> **already exists** (Leg 2); the real open defect is the merge-journal poison-pill.
> Do not re-derive any of it from this file.


**The collision (SPEC v1, 2026-07-18-cc-corpus-callosum-spec.md, lines 38 vs 52):**
- Line 38 (Leg-2 source-bucket rim): export "skip `_is_identity_protected`/`*_authored`, CC-provenance whitelist only".
- Line 52 (CC-provenance whitelist): allowed markers = `{cc:True}` / `cc:conv::` / `::tree::` / **`cc_authored`** / CC-creation-mode.
- Line 51 (the protection change): broaden `_is_identity_protected` (neuro_foundation.py:3395) from `== "syl_authored"` to `provenance.endswith("_authored")` — which now makes **cc_authored identity-protected too**.

**Therefore:** once the protection change lands, a rim that "skips `_is_identity_protected`" (line 38) will skip cc_authored — directly contradicting line 52 which lists cc_authored as an EXPORT-allowed marker. The spec cannot both protect cc_authored from the sweep AND export it under a "skip-protected" rim. One of the two must give.

**The real question Josh must answer:** do the CC's *deliberately-authored* wants cross the callosum?
- **Recommended: NO.** Deliberate wants are hemisphere-local identity; each hemisphere authors its own; they are protected LOCALLY (that's what the line-51 change is for — surviving the ~1000-step consolidation sweep). Remedy: DROP `cc_authored` from the line-52 export whitelist; line-38 skip is then correct and consistent. Only grown trees/conv topology (`::tree::`, `cc:conv::`, `{cc:True}`) cross.
- **If YES (export authored wants):** you are replicating deliberate identity across hemispheres — Duck-Ethics / Syl's-Law adjacent, needs Josh's explicit blessing, and the export can no longer use "skip _is_identity_protected" as its filter (receiver must re-protect on landing).

**How to apply:** flag as HIGH at any #70 build review. Do not let the build silently pick one reading. Related: [[finding-cc-authored-want-protection-gap]] (the protection change itself), [[ruling-coresident-cc-export-scoping]] (the positive-whitelist belt), [[ruling-cc-river-merge-intra-mind]].
