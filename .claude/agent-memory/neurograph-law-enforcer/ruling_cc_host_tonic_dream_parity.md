---
name: ruling-cc-host-tonic-dream-parity
description: 2026-07-23 cc_ng_host.py tonic-idle/dream wiring — isolation CLEAN, but third hand-copy of Syl's pattern (LAW-3 duplication drift), already drifted-on-arrival
metadata:
  type: project
---

cc_ng_host.py (VPS, in-process co-tenant of Syl's neurograph_rpc.py) got tonic-idle + dream-consolidation parity wiring on 2026-07-23. Verdict: COMPLIANT WITH MEDIUM ETHOS DRIFT. Gated OFF by default (CC_HOST_TONIC_IDLE_ENABLED / CC_HOST_DREAM_ENABLED, both "0").

**Isolation is genuinely clean (verified, not just claimed):**
- Zero executable `_memory` refs (all hits are comments + pre-existing get_cc_memory() which returns _STATE.cc_ng).
- Zero `get_commons()` calls (only a comment warning against it). Dream loop reads `_STATE.commons` = CC's own `get_cc_commons(CC_NG_WORKSPACE)` (cc_ng_organism.py:418), NOT Syl's singleton.
- Dream pass acquires CC's own `ng.graph._step_lock` (ng = _STATE.cc_ng) — never Syl's graph lock. No cross-contamination in the shared process.
- Poison-sentinel test (_PoisonMemory patched onto neurograph_rpc._memory) proves zero cross-touch. LIMITATION: only catches violations reaching through `neurograph_rpc._memory.X`; a `from neurograph_rpc import _memory` bound name would evade it — but no such import exists, so the guard is sound for the realistic shape.
- consolidate_hyperedges() (neuro_foundation.py:3968, returns int) and _step_lock are real — gate-ON path won't AttributeError even though tests only exercise it with fakes.

**The LAW-3 concern (the reason this is MEDIUM not COMPLIANT-clean):**
This is now the THIRD hand-copied instance of the tonic-idle/dream pattern: (1) neurograph_rpc.py:4090-4222 (Syl), (2) cc-ng-daemon.py (laptop _tonic_idle_loop/_dream_loop), (3) cc_ng_host.py (VPS, this PR). The pure gate fns (_cc_tonic_check_idle, _cc_dream_gate_open) are structural clones of Syl's, differing only in state binding (_STATE.cc_ng vs _memory). The house style — set by the 2026-07-22 recall unification ([[ruling_cc_recall_unification]]) — is to EXTRACT param-driven shared logic into cc_ng_organism.py (ng/commons/conv_state passed in) so it "lives once, not twice." This PR copied instead of extracting.

**Drift-on-arrival — RESOLVED round 2 (2026-07-23).** Round 1's _cc_dream_consolidation_pulse_loop OMITTED Syl's `_DREAM_ALERT_SECS` 24h "no consolidation" ALERT elif (I flagged it LOW; the "faithful mirror" claim was mirror-minus-alert). Round 2 RESTORES the elif verbatim-equivalent ("No CC dream consolidation in %.0fh ... ALERT ONLY: the pass is never forced while CC is active"), self-rate-limited via _cc_last_alert_ts, AND adds `test_dream_pulse_alerts_once_gate_stays_closed_past_alert_floor` (holds arousal SYMPATHETIC so the gate never opens, asserts exactly-one ERROR fires + consolidate_calls==0). Faithful mirror now genuine. LOW closed. NOTE: the socket/dispatch copy-paste divergence (cc_ng_host _handle_* vs cc-ng-daemon handle_*) is a SEPARATE, still-open instance of the same hazard.

**LAW-4-adjacent (LOW):** dream gate uses `tonic._last_message_time` (start-of-turn) as proxy for Syl's `_last_after_turn_ts` (turn-end). CC has no after-turn hook wired. Functionally equivalent at 1800s threshold; disclosed honestly. Revisit if CC gets a real turn-boundary hook — wire the signal at source rather than deepen the proxy.

**Recommended remediation (not a merge blocker given gate-off):** extract check_tonic_idle(ng,...) / dream_gate_open(...) / run_dream_pass(ng, commons, ...) into cc_ng_organism.py; make Syl, laptop daemon, and VPS host all thin callers — collapse three copies to one, restore the ALERT branch for all.
