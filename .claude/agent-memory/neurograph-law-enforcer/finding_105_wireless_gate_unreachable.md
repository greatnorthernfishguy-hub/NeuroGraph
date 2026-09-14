---
name: finding-105-wireless-gate-unreachable
description: #105 wire-less save-gate branch is nested under `if have_syn:` so it's dead code on a genuinely wire-less host (ref_synapses < min_ref_syn=100); tests pass only on unrealistic 23k-synapse inputs
metadata:
  type: project
---

Branch `cc-105-guardian-wireless-gate` (reviewed 2026-08-02, uncommitted). **RESOLVED same day** — wire-less branch hoisted to section "1b" (after abs-floor, before synapse gate), now fires independent of `have_syn`; old nested copy removed (that block reverts to exact #83); deep-shed refuse tests re-pointed to realistic shape (synapses=0/0 and synapses=None, both have_syn False) + a discriminator test (same 1039→400 drop: wired permits, wire-less refuses). Verified by inspection at checkpoint_guardian.py:229-251.

**Finding (CRITICAL correctness gap, not a Law violation):** In `checkpoint_guardian.py::evaluate_save_health`, the #105 wire-less refuse branch (`wiring_capable is False and live_nodes < ref_nodes ...`) is placed **inside** the `if have_syn:` block. `have_syn` requires `ref_synapses >= NG_GUARDIAN_MIN_REF_SYNAPSES` (default 100). A genuinely wire-less host (the change's own target: laptop CC, "0/1039 cc_gateway ever wired") has ~0 synapses, so `have_syn` is False and the new branch never executes. Control falls to the legacy node-only path (step 4) which permits sheds down to 50% node retention — so a real 1039→700 content shed is PERMITTED, defeating the fix's stated invariant ("ANY net node loss refused").

**Why:** The 9 new tests in test_save_guard_structural.py all pass `ref_synapses=22900–23000` — a state no wire-less host ever has. They validate an unreachable code path and give false green.

**How to apply:** Remedy is to hoist the wire-less check above/outside `if have_syn:` so it fires on the near-zero-synapse state a wire-less host actually presents, and re-point at least one test to the realistic (ref_synapses ~0) shape. LAW 3 (restore intended behavior), not widen a threshold. Also: openclaw_hook.py #105 edit (protected file, not one of the seven vendored) lacks the Josh-approval/backup marker the sibling #83 entry carried — confirm authorization.

Related: [[ruling_93_106_probation_journal]], [[finding_save_guard_cc_gateway_exemption]], [[ruling_half_brain_wholeness_ring]]. General pattern to watch: **protection nested under a guard whose precondition the target state cannot satisfy = dead protection.**
