---
name: pith-gate-tests-need-monkeypatch
description: CC_PITH_ENABLED=1 is set process-wide in this env's ~/.bashrc; gate-dependent tests MUST monkeypatch cc_ng_organism._CC_PITH_ENABLED, not rely on env default
metadata:
  type: feedback
---

Any test that exercises the gate-OFF (pre-Pith) recall path in `cc_ng_organism.cc_assemble_recall` must `monkeypatch.setattr(cc_ng_organism, "_CC_PITH_ENABLED", <bool>)` — do NOT assume the env default.

**Why:** This environment's `~/.bashrc` exports `CC_PITH_ENABLED=1` process-wide (LAW 5 source of truth). `_CC_PITH_ENABLED` is a module-level constant read once at import. So a "gate-off" test that relies on the unset default actually runs through the Pith pipeline instead — the exact nondeterminism that made `test_cc_recall_dedup.py` a false pass (law-enforcer Finding 2, 2026-07-22). Confirmed: with `CC_PITH_ENABLED=1` ambient, the full unification+dedup suite (19 tests) still passes only because each gate-dependent test forces the constant.

**How to apply:** When reviewing/writing CC recall or Pith tests, verify the gate constant is monkeypatched (pattern also used in `tests/test_pith_stage5.py`). Gate-ON tests set it True and must patch the *real* pith_stage1/stage3 (not stubs) to actually prove the pipeline runs — the prior DoD gap. Relates to [[ruling_cc_recall_unification]].
