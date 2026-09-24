---
name: finding-lenia-watermark-flushable-buffer
description: Lenia #137/#140 — advancing resume watermark every pair is only correct for a buffer an out-of-band save() can flush (LIL); COO-delta path desyncs → silent gap. RESOLVED via Option A (2026-08-11), audited COMPLIANT.
metadata:
  type: project
---

**RESOLVED 2026-08-11 (#140), audited COMPLIANT.** Option A landed in lenia/kernel.py:
LIL advances watermark per-pair (:673-674); COO advances ONLY at fold, in lockstep with
`_fold_delta_into_csr()` (:711-713), placed at the checkpoint call-site (NOT inside the
helper) so loop-end completion can set watermark=None (:733) instead of (i,j). Both
directions closed (silent gap + double-count). LAW 3 (repair-in-place) + LAW 4
(fix-at-write-source) exemplary. Regression test
`test_double_interrupt_through_resume_path_no_double_count` drives the COO fold-advance
(crash→resume→crash-after-checkpoint→finish), asserts wm2!=wm1 + full np.allclose.
LOW advisory (pre-existing, non-blocking): completion sets watermark=None BEFORE the
final fold (:733 before :735) — a crash in the final fold persists watermark=None →
next boot full-rebuilds LIL over non-empty CSR (#137 balloon). Self-healing; fold-first
would be safer. Punchlist candidate, not a regression from this fix.

--- original finding (below) ---

Lenia DistanceCache.populate() (lenia/kernel.py) has TWO write-buffer modes and the
resume-watermark advance policy must match what `save()` can actually persist.

- Full rebuild (`not _incremental`): writes go to `self._components_lil` (instance
  state). An out-of-band `save()` calls `_ensure_csr_current()` which flushes the
  WHOLE LIL to CSR. So the persisted CSR contains every written pair → advancing
  `self._watermark=(i,j)` on EVERY pair is correct (watermark == persisted content).

- Incremental / resume (`_incremental = start_index>0 or resume_watermark is not None`):
  writes accumulate in LOCAL `_delta_rows/_cols/_vals` COO buffers, folded into CSR
  only at checkpoints and at end (`_fold_delta_into_csr()`). `save()` CANNOT reach
  these locals — so an out-of-band save persists CSR only up to the last fold, while
  the watermark (advanced every pair) points PAST it.

**Why:** #137 (2026-08-11) moved `self._watermark=(i,j)` out of the checkpoint block
to every pair to fix a real double-count on the full-rebuild→catch-all-save→resume
path (Syl's live symptom). Correct for LIL. But it applied the same policy to the COO
path, where it introduces a symmetric SILENT GAP: neurograph_rpc.py always calls a
catch-all `lenia_cache.save()` after a *caught* populate() exception (a concurrent
graph mutation raising in distance_vector — explicitly acknowledged as unprotected).
On the resume/growth (incremental) branch that save records watermark=crash_pair over
a CSR only folded to the last checkpoint; next boot's resume skips `<= watermark` and
never recomputes the (last_fold, crash] pairs → permanent 0/"not connected" holes in
distances feeding Lenia. Hard kills are safe (catch-all doesn't run; last checkpoint
is fold-then-save consistent); only the soft caught-exception path bites.

**How to apply:** The invariant is "watermark must name the last pair a save() would
persist." Source fix (LAW 4), two options: (A) advance `self._watermark` every pair
for LIL, but for the COO path advance it only inside `_fold_delta_into_csr()`; or
(B) promote the COO delta to instance state and fold it in `_ensure_csr_current()`
so an out-of-band save genuinely flushes the whole buffer (makes the changelog's
stated premise true). The equivalence test only interrupts the FIRST (full-rebuild)
populate — add a double-interrupt test that crashes the RESUME run and asserts
full-rebuild value-equivalence. When reviewing further Lenia resume/#13x work, check
this desync first.
