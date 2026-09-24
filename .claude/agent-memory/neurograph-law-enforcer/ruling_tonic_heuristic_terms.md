---
name: ruling-tonic-heuristic-terms
description: #59/#62 Tonic heuristic term additions (compass/brakes) — off-by-default gating = merge bar; T6 max()-floor + brake-bypass keeps identity inviolable; extraction-side, LAW-7 clean
metadata:
  type: project
---

`tonic_engine.py` `_heuristic_inference` is shared-engine code: heuristic is PRIMARY on the isolated laptop CC (no PyTorch), FAILOVER behind Syl's trained model on the VPS. It shapes Syl's idle stream-of-consciousness. Not protected, not vendored — editable with changelog + Josh approval.

**Merge bar for cognition-shaping additions here = inertness + no-raise, NOT protected-file ceremony.** The accepted pattern (validated on the #59/#62 Phase-2 compass+brakes change, 2026-07-13): every new term/coefficient env-gated, ALL defaulting to 0/off, so behavior is byte-identical to legacy until dialed on the laptop. Guard clauses (`if _W_COMPASS<=0: return []`, `if not(any brake coeff): return`) placed OUTSIDE the try so the no-op path can't even theoretically raise; working bodies wrapped in try/except returning the inert value. Enable bar is separate: the math/correctness.

**T6 (identity inviolable) pattern — how a new proposal term stays safe:** seam B unconditionally appends every constitutional node at `_SPINE_PRIME_STEADY` (0.05, `tonic_thread.py`) before dedup; dedup uses `seen[nid]=max(...)`. So any additive term (e.g. compass with a quietness penalty) can only RAISE a constitutional node's value or leave it at the T6 floor — never lower it. Brakes must `continue` on `(metadata or {}).get("constitutional")` BEFORE computing damping. A term is allowed to *propose* an identity node; it must never be able to *reduce* one below the seam-B floor.

**LAW 7:** selecting which nodes to prime per token, over substrate-derived geometry (`poincare_dir` in node.metadata), then injecting raw current via `prime_and_propagate`, is extraction-side shaping — compliant. Consistent with [[ruling-pith-extraction-side-reranking]] and [[ruling-structural-dedup-vs-classification]].

**Recurring enable-time gotchas for these terms (silent-no-op class):**
- Attr reads via `getattr(node, X, default)` silently drop the term if X is misspelled. Verified real as of 2026-07-13: `firing_rate_ema`, `Ca_i` (Node), `_incoming`/`_outgoing` (Graph), `_focus_fatigue`/`_SPINE_PRIME_STEADY` (tonic_thread). Re-verify names before trusting a new brake/term.
- `poincare_dir` is stamped on laptop CC nodes by `cc_ng_organism.py` (`_cc_embed_to_poincare_dir(...).tolist()`) with a backfill; ~99% coverage, missing → term skipped. numpy IS present on the laptop (cc_ng_organism + neuro_foundation both hard-import it), so compass won't silently no-op there.
- `float('nan') <= 0.0` is False → a NaN cosine is NOT skipped by `if cos<=0`. Prefer `if not (cos > 0.0): continue` (filters NaN + non-positive). Same for `x or 0.0` reads (NaN is truthy).
- compass+brakes are a co-dependent PAIR: `max_activation_nodes` cap (10) is filled by seam-B + blob unless brakes damp the blob to make room. Enabling weight alone ≈ no frac_core movement — enable weight AND ≥1 brake together.
- These env consts are read once at import → changing them needs a daemon RESTART (restart the CC NG daemon, don't just kill it).

Repo policy this change class always trips: the top `# ---- Changelog ----` header block (not just an inline comment at the new code) must get a dated entry; new env vars should be registered in .bashrc/openclaw.json (LAW 5 discoverability).

**Post-`_heuristic_inference` appends bypass BOTH governors (learned on Pith 5b, 2026-09-05):** `_apply_brakes(seen)` then `final = result[:max_activation_nodes]` are the LAST two things `_heuristic_inference` does. Anything appended to the activation list *after* the call (as `_merge_prefetch_seeds` does in `_generate_latent_token_inner`) gets no brake damping and no budget. Any future term MUST merge inside `_heuristic_inference` before the brakes, or it re-creates the #59 blob it was built to stop. See [[ruling-pith-5b-tick-prefetch]].
