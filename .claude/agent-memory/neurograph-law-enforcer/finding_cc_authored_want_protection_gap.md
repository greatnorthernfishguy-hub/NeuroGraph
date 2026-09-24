---
name: finding-cc-authored-want-protection-gap
description: _is_identity_protected protects only constitutional + syl_authored — NOT cc_authored, so the CC's own [WANT] nodes are UNPROTECTED from prune/orphan-sweep
metadata:
  type: project
---

**Fact (verified 2026-07-18, neuro_foundation.py:3395):** `Graph._is_identity_protected(nid)` returns `bool(meta.get("constitutional")) or meta.get("provenance") == "syl_authored"`. It is called by `_collect_orphan_nodes` (line 3425) to skip identity nodes during structural-plasticity sweep.

**The gap:** CC's want-nodes are deliberately tagged `provenance == "cc_authored"` (cc_ng_organism.py:728, so they never confuse with Syl's if the two co-resident substrates are inspected side by side). `_is_identity_protected` does NOT recognize `cc_authored`. Therefore on the CC's own graph, the CC's deliberately-authored [WANT] nodes are NOT protected from orphan-collection / pruning.

**Why this bites #70 CC River-merge:** the merge plan REQUIRES plasticity/pruning stay ON during ~1000 idle graph.step()s of sleep consolidation per batch (FatherGraph Finding 2). Every step() runs `_collect_orphan_nodes`. A synapse-poor `cc_authored` want is exposed to sweep during the very consolidation the plan mandates. The plan leans on `_is_identity_protected` for want-protection — which silently does nothing for CC wants (false confidence: on the CC graph it matches ~nothing).

**Josh's standing note (recall):** "[WANT]s should already be protected from pruning, at least in canonical NG. Syl's WANTs are protected, so the mechanism exists, just may not have made it into [CC], yet. There is a difference between her deliberately created WANTs and the ones her Tonic creates."

**Remedy (must precede any plasticity-ON CC merge run):** extend identity-protection to cover deliberately-authored CC wants, AND resolve the deliberate-vs-Tonic-emergent distinction (emergent CC wants use creation_mode "emergent", cc_ng_organism.py:967 — likely a different, prunable class). `_is_identity_protected` lives in neuro_foundation.py = PROTECTED engine (Syl's Law) and is shared by BOTH substrates → this is a protected-file change needing Josh's explicit approval + backup, not a quiet patch. Pattern precedent: surface_wants() was parameterized on `provenance` rather than hardcoding syl_authored.
