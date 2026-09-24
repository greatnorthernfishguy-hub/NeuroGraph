---
name: ruling-cc-recall-unification
description: CC Recall Unification (2026-07-22) — cc_assemble_recall extraction is LAW-3 clean; one behavior divergence (dropped STATE error-stat) + gate-ON path untested
metadata:
  type: project
---

CC Recall Unification (spec: docs/superpowers/plans/2026-07-22-cc-recall-unification-spec.md). Extracted the laptop `_recall` body verbatim into `cc_ng_organism.cc_assemble_recall(ng, query, k, conv_state, commons, allow_pattern_completion)`; both `_recall`s (cc_ng_host.py VPS, docs/scripts/cc-ng-daemon.py laptop) became thin param-driven wrappers.

**Verdict: COMPLIANT.** LAW-3 win (kills the two-copy drift at the recall path). LAW-1/Syl's-Law IMPROVED over reference (param-driven, no module-global STATE reach). LAW-7 clean (pure extraction-side, per [[ruling-pith-extraction-side-reranking]]). No vendored file touched. Daemon duplicate helpers (`_recall_debug_log`, `_RECALL_DEBUG_PATH`, `_CC_RECALL_DEBUG`, `_PITH_WARN_INTERVAL_S`, `_last_pith_warn_ts`) fully REMOVED from cc-ng-daemon.py — no LAW-3 shrapnel. Gate-OFF is byte-identical to reference. All unguarded module symbols (`_CC_PITH_ENABLED` @1936, `_PITH_METRICS` @2085) confirmed to exist. 13/13 tests pass.

**The ONE behavior divergence from verbatim (MEDIUM):** reference monitor-harvest generic-`except Exception` did `STATE.stats['errors'] += 1`; extracted version dropped it (can't touch STATE under Syl's-Law). Structurally UNRECOVERABLE by the wrapper — cc_assemble_recall swallows the monitor exception and never re-raises, so the wrapper's outer except never fires for it. Recall *output* stays byte-identical (DoD #2 holds) but the error counter now under-counts monitor-harvest failures. Not a recall regression; an observability delta vs "same behavior" (Josh 2026-07-22). Substrate-native remedy if wanted: pass a metrics sink into the shared fn, never reach back into per-half STATE.

**Test-coverage gap vs the spec's own DoD (MEDIUM):** DoD #3 ("gate-on runs the full Pith pipeline, unit-proven") and DoD #4 ("shared fn: ...Pith on/off, fail-soft") are NOT met. NO test sets `_CC_PITH_ENABLED=True` against cc_assemble_recall. test_cc_recall_unification.py monkeypatches cc_assemble_recall entirely (wrapper-delegation only); test_cc_recall_dedup.py forces the gate OFF. So the gate-ON integration path (CacheLine build, _pinned, thermal loop, victim recover/capture) AND the Pith fail-soft counter/warn path are unit-untested. **Load-bearing for the #74 review: #74 flips CC_PITH_ENABLED on — the path it activates has no direct test today.**

Handler/dispatch layer (`_handle_*`/`handle_*`) was scoped OUT and remains copy-pasted+drifting (host has vector-DB fallback in recall handler, daemon doesn't; export differs; PostToolUse sync vs threaded) — pre-existing, unaddressed by this refactor.
